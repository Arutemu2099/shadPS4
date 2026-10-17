// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <cstdlib>
#include <filesystem>
#include "common/arch.h"
#include "common/assert.h"
#include "common/decoder.h"
#include "common/signal_context.h"
#include "common/path_util.h"
#include "common/singleton.h"
#include "core/cpu_patches.h" // Windows static guest red-zone protection
#include "core/libraries/kernel/threads/exception.h"
#include "core/linker.h"
#include "core/module.h"
#include "core/signals.h"
#include "emulator.h"

#ifdef _WIN32
#include <windows.h>
static constexpr DWORD MS_VC_EXCEPTION = 0x406D1388;
#else
#include <csignal>
#include <pthread.h>
#endif

#ifdef ARCH_X86_64
#include <Zydis/Formatter.h>
#endif

#ifndef _WIN32
namespace Libraries::Kernel {
void SigactionHandler(int native_signum, siginfo_t* inf, ucontext_t* raw_context);
extern std::array<OrbisKernelExceptionHandler, 32> Handlers;
} // namespace Libraries::Kernel
#endif

namespace Core {

// Resolve a guest address to "module+0xoffset". Guest binaries are stripped SELFs so we cannot
// symbolize them, but knowing which module a return address lands in is usually enough to
// identify the call path - and frames landing in HLE libraries name themselves.
//
// Only executable segments count. Matching a module's whole reserved range instead reports data
// pointers as code, which turns every stale data word on the stack into a plausible-looking
// return address. The offset stays relative to the module base so it remains comparable with
// addresses read out of a disassembler.
static std::string ResolveGuestAddress(VAddr addr) {
    auto* linker = Common::Singleton<Core::Linker>::Instance();
    if (linker == nullptr) {
        return {};
    }
    for (s32 i = 0;; i++) {
        const auto* module = linker->GetModule(i);
        if (module == nullptr) {
            break;
        }
        const VAddr base = module->GetBaseAddress();
        if (base == 0 || module->aligned_base_size == 0) {
            continue;
        }
        const auto& info = module->info;
        const u32 num_segments = std::min<u32>(info.num_segments, SCE_DBG_MAX_SEGMENTS);
        bool has_exec_segment = false;
        for (u32 s = 0; s < num_segments; s++) {
            const auto& segment = info.segments[s];
            if ((segment.prot & PF_EXEC) == 0 || segment.size == 0) {
                continue;
            }
            has_exec_segment = true;
            if (addr >= segment.address && addr < segment.address + segment.size) {
                const std::string& name = module->name;
                return fmt::format("{}+{:#x}", name.empty() ? "<unnamed>" : name, addr - base);
            }
        }
        // A module with no executable segment recorded means the segment table did not populate;
        // fall back to the whole range there rather than silently dropping real frames.
        if (!has_exec_segment && addr >= base && addr < base + module->aligned_base_size) {
            const std::string& name = module->name;
            return fmt::format("{}+{:#x}?", name.empty() ? "<unnamed>" : name, addr - base);
        }
    }
    return {};
}

static std::string DisassembleInstruction(void* code_address) {
    char buffer[256] = "<unable to decode>";

#ifdef ARCH_X86_64
    ZydisDecodedInstruction instruction;
    ZydisDecodedOperand operands[ZYDIS_MAX_OPERAND_COUNT];
    const auto status =
        Common::Decoder::Instance()->decodeInstruction(instruction, operands, code_address);
    if (ZYAN_SUCCESS(status)) {
        ZydisFormatter formatter;
        ZydisFormatterInit(&formatter, ZYDIS_FORMATTER_STYLE_INTEL);
        ZydisFormatterFormatInstruction(&formatter, &instruction, operands,
                                        instruction.operand_count_visible, buffer, sizeof(buffer),
                                        reinterpret_cast<u64>(code_address), ZYAN_NULL);
    }
#endif

    return buffer;
}

#if defined(_WIN32)

// Guest code is built without dependable frame pointers, so walking an RBP chain does not work.
// Scan the stack instead and report every value that lands inside a loaded module. This is a
// heuristic: stale slots that happen to look like code addresses will show up as false frames,
// so treat the output as candidates, not a verified call stack.
static std::vector<VAddr> LogGuestBacktrace(const CONTEXT* context) noexcept {
    std::vector<VAddr> candidates;
    if (context == nullptr || context->Rsp == 0) {
        return candidates;
    }
    const auto rsp = static_cast<VAddr>(context->Rsp);

    // Confirm the stack region is committed before dereferencing it. Faulting inside the crash
    // handler would lose the entire report.
    MEMORY_BASIC_INFORMATION mbi{};
    if (VirtualQuery(reinterpret_cast<LPCVOID>(rsp), &mbi, sizeof(mbi)) == 0 ||
        mbi.State != MEM_COMMIT) {
        return candidates;
    }
    const auto region_end = reinterpret_cast<VAddr>(mbi.BaseAddress) + mbi.RegionSize;

    LOG_CRITICAL(Debug, "Guest stack scan from rsp {} (candidates, may include stale slots):",
                 fmt::ptr(reinterpret_cast<void*>(rsp)));

    constexpr size_t MaxSlots = 4096;
    constexpr size_t MaxFrames = 32;
    size_t frames = 0;
    size_t scanned = 0;
    for (size_t i = 0; i < MaxSlots && frames < MaxFrames; i++) {
        const VAddr slot = rsp + i * sizeof(VAddr);
        if (slot + sizeof(VAddr) > region_end) {
            break;
        }
        scanned++;
        const VAddr value = *reinterpret_cast<const VAddr*>(slot);
        const std::string where = ResolveGuestAddress(value);
        if (!where.empty()) {
            LOG_CRITICAL(Debug, "  #{:02} {} {}", frames,
                         fmt::ptr(reinterpret_cast<void*>(value)), where);
            candidates.push_back(value);
            frames++;
        }
    }
    LOG_CRITICAL(Debug, "  ({} frames from {} stack slots; slots not landing in an executable "
                        "segment were rejected)",
                 frames, scanned);
    return candidates;
}

constexpr size_t DisasmBytesBefore = 0x600;
constexpr size_t DisasmBytesAfter = 0x100;

#ifdef ARCH_X86_64

// True if the instruction writes any part of the register whose 64-bit form is `target`, e.g.
// `mov esi, eax` counts as a write to RSI. Hidden operands are included, so implicit writes
// (`call` touching RSP, `div` touching RAX/RDX) are caught too.
static bool WritesRegister(const ZydisDecodedInstruction& inst,
                           const ZydisDecodedOperand* operands, ZydisRegister target) {
    for (u32 i = 0; i < inst.operand_count; i++) {
        const auto& op = operands[i];
        if (op.type != ZYDIS_OPERAND_TYPE_REGISTER ||
            (op.actions & ZYDIS_OPERAND_ACTION_MASK_WRITE) == 0) {
            continue;
        }
        if (ZydisRegisterGetLargestEnclosing(ZYDIS_MACHINE_MODE_LONG_64, op.reg.value) == target) {
            return true;
        }
    }
    return false;
}

// Disassemble the instructions leading up to `center` and log them.
//
// x86 instructions are variable-length and the guest is stripped, so there is no way to know
// where the instruction preceding an address begins. Brute-force it: decode forward from every
// possible start offset in the window and keep only the chains that decode cleanly all the way
// to a boundary landing exactly on `center`. Any two decode streams that ever share a boundary
// are identical from that point on, so the real instruction stream collects a vote from every
// chain that has self-synchronised onto it. The vote count is therefore also the confidence
// measure: a line with few votes is near the start of the window and may be misaligned.
static void LogDisassemblyWindow(VAddr center, size_t insts_before, size_t insts_after,
                                 const char* label) noexcept {
    if (center == 0) {
        return;
    }

    // Query the region containing `center` and clamp the window into it, so the window can never
    // run off the end of committed memory and fault inside the crash handler.
    MEMORY_BASIC_INFORMATION mbi{};
    if (VirtualQuery(reinterpret_cast<LPCVOID>(center), &mbi, sizeof(mbi)) == 0 ||
        mbi.State != MEM_COMMIT) {
        LOG_CRITICAL(Debug, "{} {}: not committed memory, cannot disassemble", label,
                     fmt::ptr(reinterpret_cast<void*>(center)));
        return;
    }
    const auto region_base = reinterpret_cast<VAddr>(mbi.BaseAddress);
    const VAddr region_end = region_base + mbi.RegionSize;
    const VAddr window_start =
        std::max(region_base, center > DisasmBytesBefore ? center - DisasmBytesBefore : region_base);
    const VAddr window_end = std::min(region_end, center + DisasmBytesAfter);
    if (window_end <= center || center < window_start) {
        return;
    }

    // Decode from a copy. Nothing else in this function then touches guest memory.
    const size_t window_size = window_end - window_start;
    const size_t center_off = center - window_start;
    std::vector<u8> code(window_size);
    std::memcpy(code.data(), reinterpret_cast<const void*>(window_start), window_size);

    auto* decoder = Common::Decoder::Instance();
    ZydisDecodedInstruction inst;
    ZydisDecodedOperand operands[ZYDIS_MAX_OPERAND_COUNT];

    std::vector<u32> votes(center_off + 1, 0);
    std::vector<size_t> chain;
    for (size_t start = 0; start < center_off; start++) {
        chain.clear();
        size_t off = start;
        while (off < center_off) {
            if (!ZYAN_SUCCESS(decoder->decodeInstruction(inst, operands, code.data() + off,
                                                         window_size - off)) ||
                inst.length == 0) {
                break;
            }
            chain.push_back(off);
            off += inst.length;
        }
        if (off != center_off) {
            continue; // ran off the rails, or stepped over `center` instead of onto it
        }
        for (const size_t boundary : chain) {
            votes[boundary]++;
        }
    }

    // Boundaries near `center` always collect the most votes - every chain that reaches center
    // passes through them - so picking the maximum just returns the last instruction. Take the
    // *earliest* boundary that several independent start offsets agree on instead: that is the
    // furthest back we can read while still being sure of the alignment.
    constexpr u32 MinVotes = 8;
    u32 best_votes = 0;
    size_t best_start = center_off;
    for (size_t off = 0; off < center_off; off++) {
        if (votes[off] >= MinVotes) {
            best_votes = votes[off];
            best_start = off;
            break;
        }
    }

    LOG_CRITICAL(Debug, "{} {} ({}), alignment recovered from {:#x} bytes back, {} votes:", label,
                 fmt::ptr(reinterpret_cast<void*>(center)), ResolveGuestAddress(center),
                 center_off - best_start, best_votes);
    if (best_votes == 0) {
        LOG_CRITICAL(Debug, "  no decode chain reaches this address; window is data or garbage");
        best_start = center_off; // still print forward from the fault itself
    }

    struct Entry {
        VAddr addr;
        u32 vote;
        std::string bytes;
        std::string text;
        std::string note;
    };
    std::vector<Entry> entries;
    size_t center_index = 0;
    for (size_t off = best_start; off < window_size;) {
        if (!ZYAN_SUCCESS(decoder->decodeInstruction(inst, operands, code.data() + off,
                                                     window_size - off)) ||
            inst.length == 0) {
            break;
        }
        Entry entry{};
        entry.addr = window_start + off;
        entry.vote = off < center_off ? votes[off] : 0;
        for (u32 i = 0; i < inst.length && i < 12; i++) {
            entry.bytes += fmt::format("{:02x} ", code[off + i]);
        }
        entry.text = decoder->disassembleInst(inst, operands, entry.addr);
        if (WritesRegister(inst, operands, ZYDIS_REGISTER_RSI)) {
            entry.note += "  <== writes rsi";
        }
        if (WritesRegister(inst, operands, ZYDIS_REGISTER_RAX)) {
            entry.note += "  <== writes rax";
        }
        if (inst.meta.category == ZYDIS_CATEGORY_CALL) {
            entry.note += "  [call]";
        }
        if (off == center_off) {
            center_index = entries.size();
        }
        entries.push_back(std::move(entry));
        off += inst.length;
        if (off > center_off && entries.size() > center_index + insts_after) {
            break;
        }
    }

    const size_t first = center_index > insts_before ? center_index - insts_before : 0;
    for (size_t i = first; i < entries.size(); i++) {
        const auto& entry = entries[i];
        LOG_CRITICAL(Debug, "  {} {} v{:<5} {:<36} {}{}", i == center_index ? "==>" : "   ",
                     fmt::ptr(reinterpret_cast<void*>(entry.addr)), entry.vote, entry.bytes,
                     entry.text, entry.note);
    }
}

#else

static void LogDisassemblyWindow(VAddr, size_t, size_t, const char*) noexcept {}

#endif

// Guest binaries are stripped SELFs that cannot be disassembled offline, so dump the code around
// the faulting instruction to a file as well. The file is the fallback if the in-emulator
// disassembly below picks the wrong alignment.
static void DumpGuestCodeAroundFault(void* rip) noexcept {
    if (rip == nullptr) {
        return;
    }
    const auto fault = reinterpret_cast<VAddr>(rip);
    const VAddr start = fault - DisasmBytesBefore;

    MEMORY_BASIC_INFORMATION mbi{};
    if (VirtualQuery(reinterpret_cast<LPCVOID>(start), &mbi, sizeof(mbi)) == 0 ||
        mbi.State != MEM_COMMIT) {
        return;
    }
    const auto region_end = reinterpret_cast<VAddr>(mbi.BaseAddress) + mbi.RegionSize;
    const size_t length =
        std::min<size_t>(DisasmBytesBefore + DisasmBytesAfter, region_end - start);

    try {
        const auto path = Common::FS::GetUserPath(Common::FS::PathType::LogDir) / "crash_code.bin";
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        if (!out) {
            return;
        }
        out.write(reinterpret_cast<const char*>(start), static_cast<std::streamsize>(length));
        LOG_CRITICAL(Debug, "Dumped {:#x} bytes of guest code at base {} to {}", length,
                     fmt::ptr(reinterpret_cast<void*>(start)), Common::FS::PathToUTF8String(path));
    } catch (const std::exception&) {
        // Never let diagnostics take down the crash handler.
    }
}

static void LogGuestRegisters(const CONTEXT* context) noexcept {
    if (context == nullptr) {
        return;
    }
    LOG_CRITICAL(Debug, "  rax={:#018x} rbx={:#018x} rcx={:#018x} rdx={:#018x}", context->Rax,
                 context->Rbx, context->Rcx, context->Rdx);
    LOG_CRITICAL(Debug, "  rsi={:#018x} rdi={:#018x} rbp={:#018x} rsp={:#018x}", context->Rsi,
                 context->Rdi, context->Rbp, context->Rsp);
    LOG_CRITICAL(Debug, "  r8 ={:#018x} r9 ={:#018x} r10={:#018x} r11={:#018x}", context->R8,
                 context->R9, context->R10, context->R11);
    LOG_CRITICAL(Debug, "  r12={:#018x} r13={:#018x} r14={:#018x} r15={:#018x}", context->R12,
                 context->R13, context->R14, context->R15);
}

// Every exception the emulator *handles* and resumes from still went through Windows exception
// dispatch, which builds its machine frame and CONTEXT below the interrupted rsp - i.e. straight
// through the guest's SysV red zone. A guest leaf function keeping live values in [rsp-0x38] would
// see them come back as garbage. Record the handled faults per thread so the crash report can say
// whether one actually landed near the frame that later faulted, instead of assuming it did.
namespace {
struct HandledFault {
    u64 rsp;
    u64 rip;
    u64 fault_address;
    u32 code;
};
constexpr size_t HandledFaultHistory = 12;
thread_local std::array<HandledFault, HandledFaultHistory> handled_faults{};
thread_local u64 handled_fault_count = 0;
} // namespace

static void LogHandledFaultHistory(u64 crash_rsp) noexcept {
    LOG_CRITICAL(Debug, "Handled (resumed) exceptions on this thread: {}", handled_fault_count);
    const u64 shown = std::min<u64>(handled_fault_count, HandledFaultHistory);
    for (u64 i = 0; i < shown; i++) {
        // Walk newest-first.
        const auto& f = handled_faults[(handled_fault_count - 1 - i) % HandledFaultHistory];
        // Windows dispatch writes below the interrupted rsp. If that rsp sat at or below the
        // crashing frame's rsp, this fault's dispatch overlapped the crashing red zone.
        const s64 delta = static_cast<s64>(f.rsp) - static_cast<s64>(crash_rsp);
        LOG_CRITICAL(Debug,
                     "  [-{}] code={:#x} rip={:#018x} ({}) rsp={:#018x} (crash_rsp{:+#x}) data={:#018x}{}",
                     i + 1, f.code, f.rip, ResolveGuestAddress(f.rip), f.rsp, delta,
                     f.fault_address,
                     (delta >= 0 && delta <= 0x200) ? "  <== OVERLAPS CRASHING RED ZONE" : "");
    }
}

static LONG WINAPI SignalHandler(EXCEPTION_POINTERS* pExp) noexcept {
    const auto* signals = Signals::Instance();
    // Windows static guest red-zone protection
    const bool use_static_windows_guest_red_zone_protection =
        WindowsGuestRedZoneProtection::IsStaticPatchingEnabled();
    DWORD code = 0;
    PVOID address = nullptr;

    if (pExp != nullptr && pExp->ExceptionRecord != nullptr) {
        code = pExp->ExceptionRecord->ExceptionCode;
        address = pExp->ExceptionRecord->ExceptionAddress;
    }

    bool handled = false;
    bool static_protection_exception = false; // Windows static guest red-zone protection
    switch (code) {
    case EXCEPTION_ACCESS_VIOLATION:
        static_protection_exception = true; // Windows static guest red-zone protection
        handled = signals->DispatchAccessViolation(
            pExp, reinterpret_cast<void*>(pExp->ExceptionRecord->ExceptionInformation[1]));
        break;
    case EXCEPTION_ILLEGAL_INSTRUCTION:
        static_protection_exception = true; // Windows static guest red-zone protection
        handled = signals->DispatchIllegalInstruction(pExp);
        break;
    case EXCEPTION_PRIV_INSTRUCTION: // Windows static guest red-zone protection
        if (use_static_windows_guest_red_zone_protection) {
            static_protection_exception = true;
            handled = signals->DispatchIllegalInstruction(pExp);
        }
        break;
    case DBG_PRINTEXCEPTION_C:
    case DBG_PRINTEXCEPTION_WIDE_C:
        // Used by OutputDebugString functions.
        return EXCEPTION_CONTINUE_EXECUTION;
    case MS_VC_EXCEPTION:
        LOG_DEBUG(Debug, "Pass MS_VC_EXCEPTION at {} to handler", address);
        return EXCEPTION_EXECUTE_HANDLER;
    default:
        break;
    }

    if (handled) {
        if (pExp != nullptr && pExp->ContextRecord != nullptr) {
            auto& rec = handled_faults[handled_fault_count % HandledFaultHistory];
            rec.rsp = pExp->ContextRecord->Rsp;
            rec.rip = pExp->ContextRecord->Rip;
            rec.code = code;
            rec.fault_address = (pExp->ExceptionRecord != nullptr &&
                                 pExp->ExceptionRecord->NumberParameters >= 2)
                                    ? pExp->ExceptionRecord->ExceptionInformation[1]
                                    : 0;
        }
        handled_fault_count++;
        return EXCEPTION_CONTINUE_EXECUTION;
    }

    // Windows static guest red-zone protection
    // SHADPS4_DLSS (2026-10-17): with DLSS on, a fault raised by code in another module (not
    // shadPS4.exe, not guest memory - e.g. NVIDIA's driver while it sets up VK_NVX_binary_import)
    // goes on to that module's own handlers instead of shutting the emulator down.
    static const bool pass_foreign = [] {
        const char* v = std::getenv("SHADPS4_DLSS");
        return v && v[0] == '1';
    }();
    if (pass_foreign && address != nullptr) {
        HMODULE module = nullptr;
        if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                   GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                               static_cast<LPCWSTR>(address), &module) &&
            module != GetModuleHandleW(nullptr)) {
            static std::atomic<u32> passed{0};
            if (passed.fetch_add(1, std::memory_order_relaxed) < 8) {
                wchar_t name[MAX_PATH] = {};
                GetModuleFileNameW(module, name, MAX_PATH);
                LOG_WARNING(Debug, "Exception {:#x} at {} in {} passed to its own handlers", code,
                            fmt::ptr(address), Common::FS::PathToUTF8String(std::filesystem::path{name}));
            }
            return EXCEPTION_CONTINUE_SEARCH;
        }
    }

    const bool report_unhandled = use_static_windows_guest_red_zone_protection
                                      ? static_protection_exception
                                      : code != EXCEPTION_BREAKPOINT;
    if (report_unhandled) { // Windows static guest red-zone protection
        // Windows hands us the access type and the faulting data address in ExceptionInformation;
        // the POSIX path already logs both, so report them here too rather than just the RIP.
        // Only decode the instruction for read/write faults: on an execute violation the RIP is
        // itself the bad address, and decoding it would fault again inside this handler.
        const bool have_av_info = code == EXCEPTION_ACCESS_VIOLATION && pExp != nullptr &&
                                  pExp->ExceptionRecord != nullptr &&
                                  pExp->ExceptionRecord->NumberParameters >= 2;
        if (have_av_info) {
            const ULONG_PTR operation = pExp->ExceptionRecord->ExceptionInformation[0];
            const auto* fault_address =
                reinterpret_cast<void*>(pExp->ExceptionRecord->ExceptionInformation[1]);
            const char* access = operation == 0   ? "Read from"
                                 : operation == 1 ? "Write to"
                                                  : "Execute at";
            const std::string disasm =
                operation == 8 ? "<execute violation>" : DisassembleInstruction(address);
            const std::string where = ResolveGuestAddress(reinterpret_cast<VAddr>(address));
            LOG_CRITICAL(Debug, "Unhandled access violation at code address {} {}: {} address {}: {}",
                         fmt::ptr(address), where.empty() ? "<unknown module>" : where, access,
                         fmt::ptr(fault_address), disasm);
        } else {
            LOG_CRITICAL(Debug, "Unhandled Exception code {:#x} at {}", code, address);
        }
        LogGuestRegisters(pExp != nullptr ? pExp->ContextRecord : nullptr);
        LogHandledFaultHistory(pExp != nullptr && pExp->ContextRecord != nullptr
                                   ? pExp->ContextRecord->Rsp
                                   : 0);
        const auto frames = LogGuestBacktrace(pExp != nullptr ? pExp->ContextRecord : nullptr);
        DumpGuestCodeAroundFault(address);
        LogDisassemblyWindow(reinterpret_cast<VAddr>(address), 24, 4, "Guest code at fault");
        // A bad pointer arriving in an argument register was most likely set up by the caller, so
        // the window before each return address matters as much as the one before the fault.
        for (size_t i = 0; i < frames.size() && i < 3; i++) {
            LogDisassemblyWindow(frames[i], 12, 2, "Guest code at stack candidate");
        }
        Common::Singleton<Core::Emulator>::Instance()->Shutdown();
    }

    return EXCEPTION_CONTINUE_SEARCH;
}

#else

void SignalHandler(int sig, siginfo_t* info, void* raw_context) {
    const auto* signals = Signals::Instance();

    auto* code_address = Common::GetRip(raw_context);

    switch (sig) {
    case SIGSEGV:
    case SIGBUS: {
        const bool is_write = Common::IsWriteError(raw_context);
        if (!signals->DispatchAccessViolation(raw_context, info->si_addr)) {
            // If the guest has installed a custom signal handler, and the access violation didn't
            // come from HLE memory tracking, pass the signal on
            if (Libraries::Kernel::Handlers[Libraries::Kernel::NativeToOrbisSignal(sig)]) {
                Libraries::Kernel::SigactionHandler(sig, info,
                                                    reinterpret_cast<ucontext_t*>(raw_context));
                return;
            }
            UNREACHABLE_MSG("Unhandled access violation at code address {}: {} address {}",
                            fmt::ptr(code_address), is_write ? "Write to" : "Read from",
                            fmt::ptr(info->si_addr));
        }
        break;
    }
    case SIGILL:
        if (!signals->DispatchIllegalInstruction(raw_context)) {
            if (Libraries::Kernel::Handlers[Libraries::Kernel::NativeToOrbisSignal(sig)]) {
                Libraries::Kernel::SigactionHandler(sig, info,
                                                    reinterpret_cast<ucontext_t*>(raw_context));
                return;
            }
            UNREACHABLE_MSG("Unhandled illegal instruction at code address {}: {}",
                            fmt::ptr(code_address), DisassembleInstruction(code_address));
        }
        break;
    default:
        if (sig == SIGSLEEP) {
            // Sleep thread until signal is received again
            sigset_t sigset;
            sigemptyset(&sigset);
            sigaddset(&sigset, SIGSLEEP);
            sigwait(&sigset, &sig);
        }
        break;
    }
}

#endif

SignalDispatch::SignalDispatch() {
#if defined(_WIN32)
    ASSERT_MSG(handle = AddVectoredExceptionHandler(0, SignalHandler),
               "Failed to register exception handler.");
#else
    struct sigaction action{};
    action.sa_sigaction = SignalHandler;
    action.sa_flags = SA_SIGINFO | SA_ONSTACK;
    sigemptyset(&action.sa_mask);

    ASSERT_MSG(sigaction(SIGSEGV, &action, nullptr) == 0 &&
                   sigaction(SIGBUS, &action, nullptr) == 0,
               "Failed to register access violation signal handler.");
    ASSERT_MSG(sigaction(SIGILL, &action, nullptr) == 0,
               "Failed to register illegal instruction signal handler.");
    ASSERT_MSG(sigaction(SIGSLEEP, &action, nullptr) == 0,
               "Failed to register sleep signal handler.");
#endif
}

SignalDispatch::~SignalDispatch() {
#if defined(_WIN32)
    ASSERT_MSG(RemoveVectoredExceptionHandler(handle), "Failed to remove exception handler.");
#else
    struct sigaction action{};
    action.sa_handler = SIG_DFL;
    action.sa_flags = 0;
    sigemptyset(&action.sa_mask);

    ASSERT_MSG(sigaction(SIGSEGV, &action, nullptr) == 0 &&
                   sigaction(SIGBUS, &action, nullptr) == 0,
               "Failed to remove access violation signal handler.");
    ASSERT_MSG(sigaction(SIGILL, &action, nullptr) == 0,
               "Failed to remove illegal instruction signal handler.");
#endif
}

bool SignalDispatch::DispatchAccessViolation(void* context, void* fault_address) const {
    for (const auto& [handler, _] : access_violation_handlers) {
        if (handler(context, fault_address)) {
            return true;
        }
    }
    return false;
}

bool SignalDispatch::DispatchIllegalInstruction(void* context) const {
    for (const auto& [handler, _] : illegal_instruction_handlers) {
        if (handler(context)) {
            return true;
        }
    }
    return false;
}

} // namespace Core
