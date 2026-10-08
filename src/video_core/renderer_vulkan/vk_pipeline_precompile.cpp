// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// SHADPS4_PRECOMPILE (default off, 2026-10-07): translate compute shaders from the game's own
// shader list (SHADPS4_PRECOMPILE_DIR = a folder of <hash:016x>.bin files plus manifest.csv, made
// from the user's game dump) instead of waiting for the game's first dispatch of each one. Runs
// inside WarmUp on the main thread, before the game starts, so nothing else touches the caches.
// Bits of the value:
//   2 VERIFY: re-translate every cached compute permutation from the list's code with the context
//     it was cached with (its RuntimeInfo and flattened user data) and compare the SPIR-V word for
//     word with the cached binary. Changes nothing.
//   1 COMPILE: every listed compute program the cache does not hold gets up to
//     SHADPS4_PRECOMPILE_GUESSES (default 4) permutations, one per context learned from cached
//     programs of the same code shape (resource list), most frequent first. Each is registered,
//     saved to the cache and its pipeline created like a preloaded one. Every guessed input is
//     part of the permutation key (RuntimeInfo now keys num_user_data and the LDS size), so a
//     wrong guess is a miss at the game's dispatch, never a wrong shader.
//   4 HOLDOUT: every cached program is treated as unseen: its guesses come from the other
//     programs' contexts only, and each guess's spec is compared with the program's real cached
//     permutations (and, on a spec match, the SPIR-V with the cached SPIR-V). Changes nothing;
//     predicts the hit rate and catches unkeyed inputs that change the shader.
// Programs that use images are skipped (their T# cannot be guessed from buffers), as are shapes
// no cached program has.
// SHADPS4_PRECOMPILE_STRIDE (2026-10-08, default 0): every live miss of the first version
// differed from its guesses in one buffer's stride only (the particle record size), which the
// learned contexts take from other programs. Per learned context, candidates whose indexed (idxen)
// guest buffers get a predicted stride: 'rounded' = highest inst_offset + access size of this
// program's own code rounded up to 16; 'table' = the most frequent real stride of cached programs
// with the same highest accessed byte (rounded if none). 0 learned only; 1 rounded; 2 table;
// 3 rounded then learned; 4 table then learned; 5 table, rounded, learned. HOLDOUT reports each
// mode as COMPILE would walk it.
// Sidecars in the cache folder: precompile-inprogress.txt holds the hash being translated by
// COMPILE; a launch that dies there (a translator assert) leaves it, and the next launch moves it
// to precompile-failed.txt and never translates it again (Fable finding 6). Self-test:
// SHADPS4_PRECOMPILE_FAIL_AT=<hex hash> or =first stops the emulator at that program.
// Record: notes/precompile-build-2026-10-07.md.

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/alignment.h"
#include "common/assert.h"
#include "common/hash.h"
#include "common/logging/log.h"
#include "common/serdes.h"
#include "core/emulator_settings.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/frontend/decode.h"
#include "shader_recompiler/info.h"
#include "shader_recompiler/ir/program.h"
#include "shader_recompiler/ir/passes/srt.h"
#include "shader_recompiler/recompiler.h"
#include "shader_recompiler/runtime_info.h"
#include "video_core/amdgpu/pixel_format.h"
#include "video_core/cache_storage.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_pipeline_serialization.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"

namespace Vulkan {

namespace {

using Clock = std::chrono::steady_clock;
using UserData = std::array<u32, Shader::ShaderParams::NumShaderUserData>;

double MsSince(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

/// Four copies read as a valid V# at any dword offset: dst_sel xyzw, data format 14 (32_32_32_32),
/// number format 7 (float), type 0, no swizzle / add_tid, nonzero records, stride 119.
constexpr u32 FillerDword = 0x00077FACu;
constexpr u32 FlatFillerDwords = 1024;

struct ExtractEntry {
    std::filesystem::path bin;
    u32 code_length{};
};

int PrecompileMode() {
    const char* value = std::getenv("SHADPS4_PRECOMPILE");
    return value ? std::atoi(value) : 0;
}

u32 PrecompileGuesses() {
    const char* value = std::getenv("SHADPS4_PRECOMPILE_GUESSES");
    const int n = value ? std::atoi(value) : 4;
    return static_cast<u32>(std::clamp(n, 1, 7)); // Program::MaxPermutations is 8
}

/// Compute rows of manifest.csv: shader_hash (hex, as BinaryInfo::shader_hash), stage, code_length.
std::unordered_map<u64, ExtractEntry> ReadManifest(const std::filesystem::path& dir) {
    std::unordered_map<u64, ExtractEntry> entries;
    std::ifstream file{dir / "manifest.csv"};
    std::string line;
    if (!std::getline(file, line)) {
        return entries;
    }
    auto split = [](const std::string& text) {
        std::vector<std::string> fields;
        size_t start = 0;
        while (true) {
            const size_t comma = text.find(',', start);
            fields.push_back(text.substr(start, comma - start));
            if (comma == std::string::npos) {
                return fields;
            }
            start = comma + 1;
        }
    };
    const auto header = split(line);
    auto column = [&](std::string_view name) {
        for (size_t i = 0; i < header.size(); ++i) {
            if (header[i] == name) {
                return static_cast<int>(i);
            }
        }
        return -1;
    };
    const int c_hash = column("shader_hash");
    const int c_stage = column("stage");
    const int c_length = column("code_length");
    if (c_hash < 0 || c_stage < 0 || c_length < 0) {
        LOG_ERROR(Render_Vulkan, "Precompile: manifest.csv lacks shader_hash / stage / code_length");
        return entries;
    }
    const int needed = std::max({c_hash, c_stage, c_length});
    while (std::getline(file, line)) {
        const auto fields = split(line);
        if (static_cast<int>(fields.size()) <= needed || fields[c_stage] != "cs") {
            continue;
        }
        const u64 hash = std::strtoull(fields[c_hash].c_str(), nullptr, 16);
        const u32 length = static_cast<u32>(std::strtoul(fields[c_length].c_str(), nullptr, 10));
        if (length == 0 || (length & 3) != 0) {
            continue;
        }
        entries[hash] = {dir / (fields[c_hash] + ".bin"), length};
    }
    return entries;
}

std::vector<u32> ReadCode(const ExtractEntry& entry) {
    std::vector<u32> code(entry.code_length / sizeof(u32));
    std::ifstream file{entry.bin, std::ios::binary};
    file.read(reinterpret_cast<char*>(code.data()), entry.code_length);
    if (file.gcount() != static_cast<std::streamsize>(entry.code_length)) {
        code.clear();
    }
    return code;
}

/// What decoding the code (no translation) says about precompiling it.
struct CodeTraits {
    bool images{};  ///< an image instruction: not precompiled (images are not guessed)
    bool movrel{};  ///< V_MOVREL*: its translation reads num_allocated_vgprs, which a guess cannot know
};

CodeTraits ScanCode(const std::vector<u32>& code) {
    using Shader::Gcn::Opcode;
    CodeTraits traits{};
    Shader::Gcn::GcnCodeSlice slice(code.data(), code.data() + code.size());
    Shader::Gcn::GcnDecodeContext decoder;
    while (!slice.atEnd()) {
        const auto inst = decoder.decodeInstruction(slice);
        traits.images |= inst.encoding == Shader::Gcn::InstEncoding::MIMG;
        traits.movrel |= inst.opcode == Opcode::V_MOVRELD_B32 ||
                         inst.opcode == Opcode::V_MOVRELS_B32 ||
                         inst.opcode == Opcode::V_MOVRELSD_B32;
    }
    return traits;
}

bool IsGuessableBuffer(const Shader::BufferResource& buffer) {
    return buffer.buffer_type == Shader::BufferType::Guest && !buffer.inline_cbuf;
}

/// The code-derived resource list. 0 = not handled (images, samplers or fmasks).
u64 ShapeHash(const Shader::Info& info) {
    if (!info.images.empty() || !info.samplers.empty() || !info.fmasks.empty()) {
        return 0;
    }
    u64 hash = HashCombine(0x5052454350ull, info.buffers.size());
    for (const auto& buffer : info.buffers) {
        hash = HashCombine(hash, static_cast<u64>(buffer.buffer_type));
        hash = HashCombine(hash, static_cast<u64>(buffer.used_types));
        hash = HashCombine(hash, (u64(buffer.is_written) << 1) | u64(buffer.is_formatted));
        hash = HashCombine(hash, IsGuessableBuffer(buffer) ? u64(buffer.sharp_idx) : ~0ull);
    }
    return hash;
}

constexpr std::string_view PrecompileInProgressFile = "precompile-inprogress.txt";
constexpr std::string_view PrecompileFailedFile = "precompile-failed.txt";
constexpr std::string_view PrecompileSuspectFile = "precompile-suspect.txt";

/// SHADPS4_PRECOMPILE_STRIDE (see the top and StrideVariants): 0 learned strides only.
int StrideMode() {
    const char* value = std::getenv("SHADPS4_PRECOMPILE_STRIDE");
    return value ? std::atoi(value) : 0;
}

/// How a translated program addresses each of its buffers (index = Info::buffers index).
struct BufferAccess {
    bool idxen{};   ///< some access indexes it: the V# stride is in the address
    u32 max_end{};  ///< highest inst_offset + access size over the indexed accesses (soffset,
                    ///< a register, is not counted)
    bool sized{true}; ///< false if an indexed access's size is unknown (untyped format access)
};

/// Bytes one buffer instruction touches past its address; 0 if unknown (a formatted access
/// whose format comes from the V#), -1 if not a buffer access.
int AccessBytes(const Shader::IR::Inst& inst) {
    using Shader::IR::Opcode;
    switch (inst.GetOpcode()) {
    case Opcode::LoadBufferU8:
    case Opcode::StoreBufferU8:
        return 1;
    case Opcode::LoadBufferU16:
    case Opcode::StoreBufferU16:
        return 2;
    case Opcode::LoadBufferU32:
    case Opcode::LoadBufferF32:
    case Opcode::StoreBufferU32:
    case Opcode::StoreBufferF32:
        return 4;
    case Opcode::LoadBufferU32x2:
    case Opcode::LoadBufferF32x2:
    case Opcode::LoadBufferU64:
    case Opcode::StoreBufferU32x2:
    case Opcode::StoreBufferF32x2:
    case Opcode::StoreBufferU64:
        return 8;
    case Opcode::LoadBufferU32x3:
    case Opcode::LoadBufferF32x3:
    case Opcode::StoreBufferU32x3:
    case Opcode::StoreBufferF32x3:
        return 12;
    case Opcode::LoadBufferU32x4:
    case Opcode::LoadBufferF32x4:
    case Opcode::StoreBufferU32x4:
    case Opcode::StoreBufferF32x4:
        return 16;
    case Opcode::LoadBufferFormatF32:
    case Opcode::StoreBufferFormatF32: {
        const auto info = inst.Flags<Shader::IR::BufferInstInfo>();
        if (!info.typed) {
            return 0;
        }
        return static_cast<int>(AmdGpu::NumBitsPerElement(info.inst_data_fmt.Value()) / 8);
    }
    case Opcode::BufferAtomicIAdd64:
    case Opcode::BufferAtomicSMin64:
    case Opcode::BufferAtomicUMin64:
    case Opcode::BufferAtomicSMax64:
    case Opcode::BufferAtomicUMax64:
        return 8;
    default:
        if (inst.GetOpcode() >= Opcode::BufferAtomicIAdd32 &&
            inst.GetOpcode() <= Opcode::BufferAtomicFCmpSwap32) {
            return 4;
        }
        return -1;
    }
}

std::vector<BufferAccess> AnalyzeBufferAccess(const Shader::IR::Program& program,
                                              const Shader::Info& info) {
    const size_t num_buffers = info.buffers.size();
    std::vector<BufferAccess> access(num_buffers);
    for (const Shader::IR::Block* block : program.blocks) {
        for (const Shader::IR::Inst& inst : block->Instructions()) {
            const int bytes = AccessBytes(inst);
            if (bytes < 0 || !inst.Arg(0).IsImmediate()) {
                continue;
            }
            const u32 handle = inst.Arg(0).U32();
            if (handle >= num_buffers) {
                continue;
            }
            const auto flags = inst.Flags<Shader::IR::BufferInstInfo>();
            if (!flags.index_enable) {
                continue;
            }
            auto& a = access[handle];
            a.idxen = true;
            a.sized &= bytes > 0;
            // Opus review finding 4: TranslateProgram lowers format accesses to raw loads sized by
            // the probe's filler V# (16 bytes); an untyped access to a formatted buffer has no size
            // of its own here.
            if (flags.inst_data_fmt.Value() == AmdGpu::DataFormat::FormatInvalid &&
                info.buffers[handle].is_formatted) {
                a.sized = false;
            }
            a.max_end = std::max(a.max_end, static_cast<u32>(flags.inst_offset.Value()) +
                                                static_cast<u32>(bytes));
        }
    }
    return access;
}

/// The predicted record size of an indexed buffer: its highest accessed byte, rounded up to 16.
u32 PredictStride(const BufferAccess& a) {
    return a.idxen && a.sized && a.max_end > 0 ? Common::AlignUp(a.max_end, 16) : 0;
}

constexpr u32 StrideShift = 16; // V# dword 1 bits [29:16]
constexpr u32 StrideMask = 0x3FFFu << StrideShift;

struct Context {
    Shader::RuntimeInfo runtime_info{};
    std::vector<std::pair<u32, std::array<u32, 4>>> sharps; // sharp_idx, V#
    u32 count{};
};

/// Real strides of indexed guest buffers by their highest accessed byte, learned from cached
/// programs (census s195: the most frequent one is right for 96.5% of indexed buffers, leave-one-
/// out, against 83.3% for the rounded end alone).
struct StrideTable {
    std::unordered_map<u32, std::unordered_map<u32, s32>> counts; // max_end -> stride -> count
    size_t pairs{};

    void Add(u32 end, u32 stride, s32 delta) {
        counts[end][stride] += delta;
        pairs += delta;
    }
    /// The most frequent stride for `end` (ties: the smaller), 0 if none.
    u32 Predict(u32 end) const {
        const auto it = counts.find(end);
        if (it == counts.end()) {
            return 0;
        }
        u32 best = 0;
        s32 best_count = 0;
        for (const auto& [stride, count] : it->second) {
            if (count > best_count || (count == best_count && count > 0 && stride < best)) {
                best = stride;
                best_count = count;
            }
        }
        return best_count > 0 ? best : 0;
    }
};

/// Candidate kinds per learned context: its own strides, the rounded end (P16), the table.
enum class StrideVariant : u32 { Learned = 0, Rounded = 1, Table = 2 };

/// SHADPS4_PRECOMPILE_STRIDE: 0 learned; 1 rounded; 2 table; 3 rounded then learned; 4 table
/// then learned; 5 table, rounded, learned (per learned context, identical ones once).
std::vector<StrideVariant> StrideVariants(int mode) {
    using V = StrideVariant;
    switch (mode) {
    case 1:
        return {V::Rounded};
    case 2:
        return {V::Table};
    case 3:
        return {V::Rounded, V::Learned};
    case 4:
        return {V::Table, V::Learned};
    case 5:
        return {V::Table, V::Rounded, V::Learned};
    default:
        return {V::Learned};
    }
}

u32 PredictFor(StrideVariant variant, const BufferAccess& a, const StrideTable* table) {
    if (variant == StrideVariant::Table && table && a.idxen && a.sized) {
        if (const u32 stride = table->Predict(a.max_end)) {
            return stride;
        }
    }
    return variant == StrideVariant::Learned ? 0 : PredictStride(a);
}

/// `ctx` with the predicted stride in the V# of every indexed guest buffer that has one.
/// `changed` is set if any stride differs from the learned one.
Context WithPredictedStrides(const Context& ctx, const std::vector<Shader::BufferResource>& buffers,
                             const std::vector<BufferAccess>& access, bool* changed = nullptr,
                             StrideVariant variant = StrideVariant::Rounded,
                             const StrideTable* table = nullptr) {
    Context out = ctx;
    if (variant == StrideVariant::Learned) {
        return out;
    }
    for (auto& [sharp_idx, sharp] : out.sharps) {
        u32 predicted = 0;
        for (size_t i = 0; i < buffers.size() && i < access.size(); ++i) {
            const auto& buffer = buffers[i];
            if (IsGuessableBuffer(buffer) && buffer.sharp_idx == sharp_idx) {
                predicted = std::max(predicted, PredictFor(variant, access[i], table));
            }
        }
        if (predicted == 0 || predicted > 0x3FFFu) {
            continue;
        }
        const u32 dword1 = (sharp[1] & ~StrideMask) | (predicted << StrideShift);
        if (changed && dword1 != sharp[1]) {
            *changed = true;
        }
        sharp[1] = dword1;
    }
    return out;
}

struct Candidate {
    u32 rank{};
    StrideVariant variant{};
    Context ctx;
};

/// COMPILE's candidate contexts for `mode`, at most `limit`: per learned context (most frequent
/// first) its variants in order, a variant identical to an earlier one of the same context once.
std::vector<Candidate> MakeCandidates(const std::vector<const Context*>& ranked, int mode,
                                      size_t limit,
                                      const std::vector<Shader::BufferResource>& buffers,
                                      const std::vector<BufferAccess>& access,
                                      const StrideTable* table) {
    std::vector<Candidate> out;
    const auto variants = StrideVariants(mode);
    for (u32 rank = 0; rank < ranked.size() && out.size() < limit; ++rank) {
        const size_t first_of_rank = out.size();
        for (const StrideVariant variant : variants) {
            if (out.size() >= limit) {
                break;
            }
            Context ctx = WithPredictedStrides(*ranked[rank], buffers, access, nullptr, variant, table);
            bool duplicate = false;
            for (size_t i = first_of_rank; i < out.size() && !duplicate; ++i) {
                duplicate = out[i].ctx.sharps == ctx.sharps;
            }
            if (!duplicate) {
                out.push_back({rank, variant, std::move(ctx)});
            }
        }
    }
    return out;
}

struct ShapeContexts {
    std::unordered_map<u64, Context> by_key;
    std::vector<const Context*> ranked;
};

/// The context of a cached permutation whose guest buffers are all valid, or nullopt.
std::optional<std::pair<u64, Context>> LearnContext(const Shader::Info& info,
                                                    const Shader::StageSpecialization& spec,
                                                    u32& add_tid_seen) {
    Context ctx{};
    ctx.runtime_info = spec.runtime_info;
    const auto& rt = ctx.runtime_info;
    u64 key = HashCombine(u64(rt.num_user_data), u64(rt.cs_info.shared_memory_size));
    for (const u32 v : rt.cs_info.workgroup_size) {
        key = HashCombine(key, u64(v));
    }
    for (const bool v : rt.cs_info.tgid_enable) {
        key = HashCombine(key, u64(v));
    }
    for (const auto& buffer : info.buffers) {
        if (!IsGuessableBuffer(buffer)) {
            continue;
        }
        if (buffer.sharp_idx + 4 > info.flattened_ud_buf.size()) {
            return std::nullopt;
        }
        std::array<u32, 4> sharp{};
        std::copy_n(info.flattened_ud_buf.begin() + buffer.sharp_idx, 4, sharp.begin());
        AmdGpu::Buffer vsharp{};
        std::memcpy(&vsharp, sharp.data(), sizeof(vsharp));
        if (!vsharp.Valid() || !vsharp) {
            return std::nullopt; // an empty buffer at that dispatch: learn from a full one
        }
        add_tid_seen += vsharp.add_tid_enable;
        // Base address bits (dword 0, dword 1 [11:0]) are not part of any key.
        key = HashCombine(key, u64(buffer.sharp_idx));
        key = HashCombine(key, u64(sharp[1] & ~0xfffu));
        key = HashCombine(key, u64(sharp[3]));
        ctx.sharps.emplace_back(buffer.sharp_idx, sharp);
    }
    return std::make_pair(key, std::move(ctx));
}

struct RealPerm {
    std::unique_ptr<Shader::Info> info;
    Shader::StageSpecialization spec;
    size_t perm_idx{};
    u64 shape{};
    u64 ctx_key{}; // 0 = not learned
    bool full{};   // every guest buffer non-empty at the dispatch it was made for
};

} // Anonymous namespace

std::unordered_set<u64> ReadPrecompileHashes(std::string_view file_name) {
    std::unordered_set<u64> hashes;
    const auto path = Storage::DataBase::Instance().SidecarPath(file_name);
    if (path.empty()) {
        return hashes;
    }
    std::ifstream file{path};
    std::string line;
    while (std::getline(file, line)) {
        if (line.size() >= 16) {
            hashes.insert(std::strtoull(line.c_str(), nullptr, 16));
        }
    }
    return hashes;
}

void AppendPrecompileHash(std::string_view file_name, u64 hash) {
    const auto path = Storage::DataBase::Instance().SidecarPath(file_name);
    if (path.empty()) {
        return;
    }
    std::ofstream file{path, std::ios::app};
    file << fmt::format("{:016x}\n", hash);
    file.flush();
}

void NotePrecompiledUse(const Program& program, const Shader::StageSpecialization& live, bool hit) {
    static u32 hits = 0, misses = 0;
    hit ? ++hits : ++misses;
    const auto& rt = live.runtime_info;
    u32 first_stride = 0;
    if (!live.buffers.empty()) {
        first_stride = live.buffers[0].stride;
    }
    u32 guess_stride = 0;
    if (!program.modules.empty() && !program.modules[0].spec.buffers.empty()) {
        guess_stride = program.modules[0].spec.buffers[0].stride;
    }
    LOG_INFO(Render_Vulkan,
             "PrecompiledUse: cs_{:#x} {} (hits={} misses={}) live nud={} lds={} wg={}x{}x{} "
             "stride0={} | guess0 nud={} lds={} stride0={} perms={}",
             program.info.pgm_hash, hit ? "hit" : "MISS", hits, misses, rt.num_user_data,
             rt.cs_info.shared_memory_size, rt.cs_info.workgroup_size[0],
             rt.cs_info.workgroup_size[1], rt.cs_info.workgroup_size[2], first_stride,
             program.modules.empty() ? 0u : program.modules[0].spec.runtime_info.num_user_data,
             program.modules.empty()
                 ? 0u
                 : program.modules[0].spec.runtime_info.cs_info.shared_memory_size,
             guess_stride, program.modules.size());
    if (hit || misses > 64) {
        return;
    }
    for (size_t m = 0; m < program.modules.size(); ++m) {
        const auto& guess = program.modules[m].spec;
        std::string why;
        if (!(guess.runtime_info == live.runtime_info)) {
            why = "runtime_info";
        } else if (guess.buffers.size() != live.buffers.size()) {
            why = fmt::format("buffer count {} vs {}", guess.buffers.size(), live.buffers.size());
        } else {
            // Every differing buffer (2026-10-08; the first version stopped at the first one).
            for (size_t i = 0; i < live.buffers.size(); ++i) {
                if (!live.bitset[i] || guess.buffers[i] == live.buffers[i]) {
                    continue;
                }
                const auto& g = guess.buffers[i];
                const auto& l = live.buffers[i];
                auto with_stride = g;
                with_stride.stride = l.stride;
                if (with_stride == l) {
                    why += fmt::format("{}buffer {}: stride {} vs {}", why.empty() ? "" : "; ", i,
                                       u32(g.stride), u32(l.stride));
                    continue;
                }
                why += fmt::format("{}buffer {}: stride {} vs {}, formatted {} vs {}, fmt {}/{} vs "
                                   "{}/{}, swizzle {} vs {}, add_tid {} vs {}",
                                   why.empty() ? "" : "; ", i, u32(g.stride), u32(l.stride),
                                   u32(g.is_formatted), u32(l.is_formatted), u32(g.data_format),
                                   u32(g.num_format), u32(l.data_format), u32(l.num_format),
                                   u32(g.swizzle_enable), u32(l.swizzle_enable),
                                   u32(g.add_tid_enable), u32(l.add_tid_enable));
            }
            if (why.empty()) {
                why = "other (images / samplers / fmasks / start bindings)";
            }
        }
        LOG_INFO(Render_Vulkan, "PrecompiledUse:   cs_{:#x} guess {} differs: {}",
                 program.info.pgm_hash, m, why);
    }
}

void PipelineCache::PrecompileFromExtract() {
    const int mode = PrecompileMode();
    if (mode == 0) {
        return;
    }
    const char* dir_env = std::getenv("SHADPS4_PRECOMPILE_DIR");
    if (!dir_env || !*dir_env) {
        LOG_WARNING(Render_Vulkan, "Precompile: SHADPS4_PRECOMPILE_DIR is not set");
        return;
    }
    const bool verify = (mode & 2) != 0;
    bool compile = (mode & 1) != 0;
    if (compile && EmulatorSettings.IsPipelineCacheArchived()) {
        // COMPILE saves entries before FinishPreload, which an archived (zip) cache refuses on the
        // IO thread (Fable finding 4): only the folder cache can take them.
        LOG_WARNING(Render_Vulkan, "Precompile: COMPILE needs the folder cache; archived cache, skipped");
        compile = false;
    }
    const bool holdout = (mode & 4) != 0;
    const bool learn = compile || holdout;
    const auto t_start = Clock::now();
    const auto manifest = ReadManifest(dir_env);
    LOG_INFO(Render_Vulkan, "Precompile: mode {} dir {} compute shaders in the list {} ({:.0f} ms)",
             mode, dir_env, manifest.size(), MsSince(t_start));
    if (manifest.empty()) {
        return;
    }

    // One pass over the cached metas: verify, and learn the contexts per shape.
    std::unordered_set<u64> cached_programs;
    std::unordered_map<u64, ShapeContexts> shapes;
    std::unordered_map<u64, std::vector<RealPerm>> reals; // HOLDOUT only
    // COMPILE's stride table (=2/4/5): one full cached permutation's buffer strides per program.
    struct TableSource {
        u64 shape{};
        std::vector<u32> strides;
    };
    std::unordered_map<u64, TableSource> table_sources;
    u32 metas = 0, unloadable = 0, compute = 0, no_code = 0, no_spv = 0, matched = 0, differed = 0;
    u32 learned = 0, unlearnable = 0, add_tid_seen = 0, precompiled_metas = 0;
    double verify_ms = 0.0;
    Storage::DataBase::Instance().ForEachBlob(
        Storage::BlobType::ShaderMeta, [&](std::vector<u8>&& data) {
            ++metas;
            Serialization::Archive ar{std::move(data)};
            Shader::Info cached{};
            Shader::StageSpecialization spec{};
            spec.info = &cached;
            std::optional<Shader::Gcn::FetchShaderData> fetch;
            size_t perm_idx{};
            bool marked = false;
            if (!LoadShaderMeta(ar, cached, fetch, spec, perm_idx, &marked)) {
                ++unloadable;
                return;
            }
            if (cached.stage != Shader::Stage::Compute) {
                return;
            }
            ++compute;
            cached_programs.insert(cached.pgm_hash);
            if (marked) {
                // A precompiled (guessed) context is not evidence of what the game dispatches:
                // never learn from it, never verify or hold it out (Fable finding 5).
                ++precompiled_metas;
                return;
            }
            u64 shape = 0, ctx_key = 0;
            if (learn) {
                shape = ShapeHash(cached);
                auto ctx = shape ? LearnContext(cached, spec, add_tid_seen) : std::nullopt;
                if (ctx) {
                    ctx_key = ctx->first;
                    auto& entry = shapes[shape].by_key[ctx->first];
                    if (entry.count == 0) {
                        entry = std::move(ctx->second);
                    }
                    ++entry.count;
                    ++learned;
                    if (compile && !table_sources.contains(cached.pgm_hash)) {
                        auto& source = table_sources[cached.pgm_hash];
                        source.shape = shape;
                        for (const auto& buffer : spec.buffers) {
                            source.strides.push_back(buffer.stride);
                        }
                    }
                } else {
                    ++unlearnable;
                }
            }
            if (holdout && shape != 0) {
                auto& real = reals[cached.pgm_hash].emplace_back();
                real.info = std::make_unique<Shader::Info>(cached);
                real.spec = spec;
                real.spec.info = real.info.get();
                real.perm_idx = perm_idx;
                real.shape = shape;
                real.ctx_key = ctx_key;
                real.full = ctx_key != 0; // LearnContext accepts only all-non-empty buffers
            }
            if (!verify) {
                return;
            }
            const auto it = manifest.find(cached.pgm_hash);
            if (it == manifest.end()) {
                ++no_code;
                return;
            }
            std::vector<u32> cached_spv;
            Storage::DataBase::Instance().Load(
                Storage::BlobType::ShaderBinary,
                fmt::format("{:#018x}_{}", cached.pgm_hash, perm_idx), cached_spv);
            const auto code = ReadCode(it->second);
            if (cached_spv.empty() || code.empty()) {
                ++no_spv;
                return;
            }

            const auto t_translate = Clock::now();
            UserData user_data{};
            std::copy_n(cached.flattened_ud_buf.begin(),
                        std::min(user_data.size(), cached.flattened_ud_buf.size()),
                        user_data.begin());
            const Shader::ShaderParams params{
                .user_data = user_data,
                .code = code,
                .hash = cached.pgm_hash,
            };
            Shader::Info info{Shader::Stage::Compute, Shader::LogicalStage::Compute, params};
            // As GetProgram: a permutation's DMA use is pinned to its program's.
            info.pinned_uses_dma = cached.uses_dma;
            auto runtime_info = spec.runtime_info;
            Shader::Backend::Bindings binding = spec.start;
            Shader::g_synthetic_flat = &cached.flattened_ud_buf;
            const auto program = Shader::TranslateProgram(code, pools, info, runtime_info, profile);
            const auto spv =
                Shader::Backend::SPIRV::EmitSPIRV(profile, runtime_info, program, binding);
            Shader::g_synthetic_flat = nullptr;
            verify_ms += MsSince(t_translate);

            if (spv == cached_spv) {
                ++matched;
                return;
            }
            ++differed;
            if (differed <= 12) {
                size_t first = 0;
                while (first < spv.size() && first < cached_spv.size() &&
                       spv[first] == cached_spv[first]) {
                    ++first;
                }
                LOG_WARNING(Render_Vulkan,
                            "Precompile verify: cs_{:#x} perm {} differs: {} vs cached {} words, "
                            "first difference at word {}",
                            cached.pgm_hash, perm_idx, spv.size(), cached_spv.size(), first);
            }
        });
    if (verify) {
        LOG_INFO(Render_Vulkan,
                 "Precompile verify: metas={} unloadable={} compute={} no_code={} no_spv={} "
                 "matched={} differed={} translate_ms={:.0f} total_ms={:.0f}",
                 metas, unloadable, compute, no_code, no_spv, matched, differed, verify_ms,
                 MsSince(t_start));
    }
    if (!learn) {
        return;
    }
    for (auto& [shape, entry] : shapes) {
        for (const auto& [key, ctx] : entry.by_key) {
            entry.ranked.push_back(&ctx);
        }
        std::ranges::stable_sort(entry.ranked, [](const Context* a, const Context* b) {
            return a->count > b->count;
        });
    }
    LOG_INFO(Render_Vulkan,
             "Precompile learn: compute metas={} learned={} unlearnable={} shapes={} "
             "add_tid_sharps={} programs_cached={} precompiled_metas_skipped={}",
             compute, learned, unlearnable, shapes.size(), add_tid_seen, cached_programs.size(),
             precompiled_metas);

    const u32 guesses = PrecompileGuesses();
    const int stride_mode = StrideMode();
    const std::vector<u32> filler(FlatFillerDwords, FillerDword);
    // Pass 1, the shape: every user SGPR from user data, filler sharps everywhere. Also how the
    // code addresses each buffer (the stride predictor's input), read from the same IR.
    struct Probe {
        u64 shape{};
        size_t flat_size{};
        std::vector<Shader::BufferResource> buffers;
        std::vector<BufferAccess> access;
    };
    auto probe_shape = [&](const std::vector<u32>& code, u64 hash) {
        Probe probe{};
        UserData user_data{};
        user_data.fill(FillerDword);
        const Shader::ShaderParams params{.user_data = user_data, .code = code, .hash = hash};
        Shader::Info info{Shader::Stage::Compute, Shader::LogicalStage::Compute, params};
        Shader::RuntimeInfo probe_rt{};
        probe_rt.Initialize(Shader::Stage::Compute);
        probe_rt.num_user_data = Shader::ShaderParams::NumShaderUserData;
        probe_rt.cs_info.workgroup_size = {64, 1, 1};
        probe_rt.cs_info.tgid_enable = {true, false, false};
        probe_rt.cs_info.shared_memory_size = static_cast<u32>(32_KB);
        Shader::g_synthetic_flat = &filler;
        const auto ir = Shader::TranslateProgram(code, pools, info, probe_rt, profile);
        Shader::g_synthetic_flat = nullptr;
        probe.access = AnalyzeBufferAccess(ir, info);
        probe.buffers.assign(info.buffers.begin(), info.buffers.end());
        probe.flat_size = info.flattened_ud_buf.size();
        probe.shape = ShapeHash(info);
        return probe;
    };
    auto make_flat = [&](const Context& ctx, size_t flat_size) {
        std::vector<u32> flat(std::max<size_t>(FlatFillerDwords, flat_size), FillerDword);
        for (const auto& [idx, sharp] : ctx.sharps) {
            if (idx + 4 <= flat.size()) {
                std::copy(sharp.begin(), sharp.end(), flat.begin() + idx);
            }
        }
        return flat;
    };

    if (holdout) {
        u32 tested = 0, image_users = 0, movrel_users = 0, probe_mismatch = 0, no_context = 0,
            missed = 0;
        u32 spv_equal = 0, spv_diff = 0, spv_equal_partial = 0, spv_diff_partial = 0;
        std::array<u32, 8> hit_at{};
        // 2026-10-08: the stride census and the predicted-stride modes.
        u32 idxen_buffers = 0, idxen_unsized = 0, pred16_right = 0, pred4_right = 0,
            multi_stride = 0, table_right = 0;
        u32 miss_stride_only = 0, miss_stride_idxen_only = 0;
        u32 pred_spv_equal = 0, pred_spv_diff = 0, unused_equal_partial = 0, unused_diff_partial = 0;
        constexpr int NumModes = 6;
        std::array<u32, NumModes> mode_hit1{}, mode_hits{}, mode_missed{}, mode_perms{};
        const auto t_holdout = Clock::now();
        // Specs kept across translations point here (operator== only checks it is set).
        const Shader::Info spec_anchor{};

        // Phase 1: which programs are tested, their probes, and the table of real strides.
        struct Item {
            u64 hash{};
            std::vector<RealPerm>* perms{};
            std::vector<u32> code;
            Probe probe;
            std::vector<const Context*> ranked;
            std::vector<std::pair<u32, u32>> pairs; // (max_end, real stride) of its idxen buffers
        };
        std::vector<Item> items;
        StrideTable table;
        for (auto& [hash, perms] : reals) {
            std::ranges::stable_partition(perms, [](const RealPerm& r) { return r.full; });
            const auto it_code = manifest.find(hash);
            if (it_code == manifest.end()) {
                continue;
            }
            auto code = ReadCode(it_code->second);
            if (code.empty()) {
                continue;
            }
            const auto traits = ScanCode(code);
            if (traits.images || traits.movrel) {
                ++(traits.images ? image_users : movrel_users);
                continue;
            }
            auto probe = probe_shape(code, hash);
            if (probe.shape != perms.front().shape) {
                ++probe_mismatch;
                continue;
            }
            const auto it_shape = shapes.find(probe.shape);
            if (it_shape == shapes.end()) {
                ++no_context;
                continue;
            }
            // The program's own contributions leave the counts: rank what the others taught.
            std::unordered_map<u64, u32> own;
            for (const auto& real : perms) {
                if (real.ctx_key) {
                    ++own[real.ctx_key];
                }
            }
            std::vector<std::pair<u32, const Context*>> ranked;
            for (const auto& [key, ctx] : it_shape->second.by_key) {
                const auto it_own = own.find(key);
                const u32 own_count = it_own == own.end() ? 0u : it_own->second;
                if (ctx.count > own_count) {
                    ranked.emplace_back(ctx.count - own_count, &ctx);
                }
            }
            if (ranked.empty()) {
                ++no_context;
                continue;
            }
            std::ranges::stable_sort(ranked, [](const auto& a, const auto& b) {
                return a.first > b.first;
            });
            ++tested;
            auto& item = items.emplace_back();
            item.hash = hash;
            item.perms = &perms;
            item.code = std::move(code);
            item.probe = std::move(probe);
            for (const auto& [count, ctx] : ranked) {
                item.ranked.push_back(ctx);
            }
            // Census: each indexed guest buffer's predicted stride against the real one (the
            // first full real permutation's; multi_stride counts buffers whose full real
            // permutations disagree, which no single prediction can match).
            const RealPerm* real_full = perms.front().full ? &perms.front() : nullptr;
            const auto& pb = item.probe;
            for (size_t i = 0; real_full && i < pb.buffers.size() && i < pb.access.size() &&
                               i < real_full->spec.buffers.size();
                 ++i) {
                if (!IsGuessableBuffer(pb.buffers[i]) || !pb.access[i].idxen) {
                    continue;
                }
                ++idxen_buffers;
                if (!pb.access[i].sized) {
                    ++idxen_unsized;
                    continue;
                }
                const u32 real_stride = real_full->spec.buffers[i].stride;
                for (const auto& r : perms) {
                    if (r.full && i < r.spec.buffers.size() &&
                        r.spec.buffers[i].stride != real_stride) {
                        ++multi_stride;
                        break;
                    }
                }
                pred16_right += PredictStride(pb.access[i]) == real_stride ? 1 : 0;
                pred4_right += Common::AlignUp(pb.access[i].max_end, 4) == real_stride ? 1 : 0;
                item.pairs.emplace_back(pb.access[i].max_end, real_stride);
                table.Add(pb.access[i].max_end, real_stride, 1);
            }
        }

        // Phase 2: the guesses, each program against a table without its own pairs.
        struct Eval {
            u32 nud{};
            Shader::StageSpecialization spec;
            const RealPerm* match{};
        };
        for (auto& item : items) {
            const u64 hash = item.hash;
            auto& perms = *item.perms;
            const auto& code = item.code;
            const auto& probe = item.probe;
            for (const auto& [end, stride] : item.pairs) {
                table.Add(end, stride, -1);
            }
            for (const auto& [end, stride] : item.pairs) {
                table_right += table.Predict(end) == stride ? 1 : 0;
            }

            // One candidate context: translated as COMPILE would; the first real permutation
            // whose spec it matches. With spv_check, a match's SPIR-V is compared with the cached
            // one (full real permutations into eq / diff, empty-buffer ones into the partial
            // pair). With `cls` and no match: is the difference strides only, and only in
            // indexed buffers?
            struct MissClass {
                bool stride_only{};
                bool idxen_only{};
            };
            auto eval = [&](const Context& ctx, bool spv_check, u32& eq, u32& diff, u32& eq_part,
                            u32& diff_part, u32 rank, const char* label, MissClass* cls) {
                Eval result{};
                result.nud = ctx.runtime_info.num_user_data;
                const auto flat = make_flat(ctx, probe.flat_size);
                UserData ud{};
                std::copy_n(flat.begin(), ud.size(), ud.begin());
                const Shader::ShaderParams params{.user_data = ud, .code = code, .hash = hash};
                Shader::Info guess{Shader::Stage::Compute, Shader::LogicalStage::Compute, params};
                auto runtime_info = ctx.runtime_info;
                Shader::Backend::Bindings binding{};
                Shader::g_synthetic_flat = &flat;
                const auto ir = Shader::TranslateProgram(code, pools, guess, runtime_info, profile);
                const auto spv =
                    Shader::Backend::SPIRV::EmitSPIRV(profile, runtime_info, ir, binding);
                Shader::g_synthetic_flat = nullptr;
                result.spec = Shader::StageSpecialization{guess, runtime_info, profile, {}};
                result.spec.info = &spec_anchor;
                for (const auto& real : perms) {
                    if (!(result.spec == real.spec)) {
                        continue;
                    }
                    result.match = &real;
                    if (spv_check) {
                        std::vector<u32> cached_spv;
                        Storage::DataBase::Instance().Load(
                            Storage::BlobType::ShaderBinary,
                            fmt::format("{:#018x}_{}", hash, real.perm_idx), cached_spv);
                        if (!real.full) {
                            ++(spv == cached_spv ? eq_part : diff_part);
                        } else if (spv == cached_spv) {
                            ++eq;
                        } else {
                            ++diff;
                            if (diff <= 12) {
                                size_t first = 0;
                                while (first < spv.size() && first < cached_spv.size() &&
                                       spv[first] == cached_spv[first]) {
                                    ++first;
                                }
                                LOG_WARNING(Render_Vulkan,
                                            "Precompile holdout{}: cs_{:#x} rank {} matches perm {} "
                                            "by spec but its SPIR-V differs ({} vs {} words, first "
                                            "at {})",
                                            label, hash, rank, real.perm_idx, spv.size(),
                                            cached_spv.size(), first);
                            }
                        }
                    }
                    return result;
                }
                if (cls) {
                    for (const auto& real : perms) {
                        if (!real.full || result.spec.buffers.size() != real.spec.buffers.size()) {
                            continue;
                        }
                        auto same_strides = result.spec;
                        bool idxen_only = true;
                        for (size_t i = 0; i < same_strides.buffers.size(); ++i) {
                            if (same_strides.buffers[i].stride != real.spec.buffers[i].stride) {
                                same_strides.buffers[i].stride = real.spec.buffers[i].stride;
                                idxen_only &= i < probe.access.size() && probe.access[i].idxen;
                            }
                        }
                        if (same_strides == real.spec) {
                            cls->stride_only = true;
                            cls->idxen_only |= idxen_only;
                        }
                    }
                }
                return result;
            };

            // Translations shared by the legacy line and every mode: (rank, variant).
            std::map<std::pair<u32, u32>, Eval> memo;
            // Legacy (as s180-s195): the first `guesses` learned contexts, no dedup, SPIR-V
            // compared at the first hit only.
            const u32 n = std::min<u32>(guesses, static_cast<u32>(item.ranked.size()));
            bool hit = false;
            MissClass cls{};
            for (u32 rank = 0; rank < n; ++rank) {
                auto result = eval(*item.ranked[rank], !hit, spv_equal, spv_diff,
                                   spv_equal_partial, spv_diff_partial, rank, "",
                                   rank == 0 ? &cls : nullptr);
                const bool matched = result.match != nullptr;
                memo.emplace(std::make_pair(rank, u32(StrideVariant::Learned)), std::move(result));
                if (matched && !hit) {
                    hit = true;
                    ++hit_at[rank];
                }
            }
            missed += !hit;
            if (!hit) {
                miss_stride_only += cls.stride_only ? 1 : 0;
                miss_stride_idxen_only += cls.idxen_only ? 1 : 0;
            }
            // Every mode as COMPILE walks it: candidates in order, other user-SGPR counts
            // skipped, identical specs once, at most `guesses` permutations from at most
            // 2 x `guesses` candidates. Predicted variants are SPIR-V-checked on every match.
            const size_t max_attempts = size_t{guesses} * 2;
            for (int mode = 0; mode < NumModes; ++mode) {
                const auto candidates = MakeCandidates(item.ranked, mode, max_attempts,
                                                       probe.buffers, probe.access, &table);
                std::vector<const Shader::StageSpecialization*> accepted;
                u32 first_nud = 0;
                int hit_pos = -1;
                for (const auto& cand : candidates) {
                    if (accepted.size() >= guesses) {
                        break;
                    }
                    const auto key = std::make_pair(cand.rank, u32(cand.variant));
                    auto it = memo.find(key);
                    if (it == memo.end()) {
                        it = memo.emplace(key, eval(cand.ctx, cand.variant != StrideVariant::Learned,
                                                    pred_spv_equal, pred_spv_diff,
                                                    unused_equal_partial, unused_diff_partial,
                                                    cand.rank, " (predicted)", nullptr))
                                 .first;
                    }
                    const Eval& result = it->second;
                    if (!accepted.empty() && result.nud != first_nud) {
                        continue;
                    }
                    if (accepted.empty()) {
                        first_nud = result.nud;
                    }
                    bool duplicate = false;
                    for (const auto* spec : accepted) {
                        duplicate |= *spec == result.spec;
                    }
                    if (duplicate) {
                        continue;
                    }
                    accepted.push_back(&result.spec);
                    if (result.match && hit_pos < 0) {
                        hit_pos = static_cast<int>(accepted.size()) - 1;
                    }
                }
                mode_perms[mode] += static_cast<u32>(accepted.size());
                mode_hit1[mode] += hit_pos == 0 ? 1 : 0;
                mode_hits[mode] += hit_pos >= 0 ? 1 : 0;
                mode_missed[mode] += hit_pos < 0 ? 1 : 0;
            }
            for (const auto& [end, stride] : item.pairs) {
                table.Add(end, stride, 1);
            }
        }
        LOG_INFO(Render_Vulkan,
                 "Precompile holdout: programs={} tested={} image_users={} movrel_users={} probe_mismatch={} "
                 "no_context={} hit@1={} @2={} @3={} @4={} @5+={} missed={} full: spv_equal={} "
                 "spv_diff={} | empty-buffer perms: spv_equal={} spv_diff={} guesses={} ms={:.0f}",
                 reals.size(), tested, image_users, movrel_users, probe_mismatch, no_context, hit_at[0],
                 hit_at[1], hit_at[2], hit_at[3], hit_at[4] + hit_at[5] + hit_at[6] + hit_at[7],
                 missed, spv_equal, spv_diff, spv_equal_partial, spv_diff_partial, guesses,
                 MsSince(t_holdout));
        LOG_INFO(Render_Vulkan,
                 "Precompile stride census: idxen_buffers={} unsized={} multi_stride={} "
                 "pred16_right={} pred4_right={} table_loo_right={} | misses (learned strides): "
                 "{} of which stride_only={} stride_only_idxen={}",
                 idxen_buffers, idxen_unsized, multi_stride, pred16_right, pred4_right,
                 table_right, missed, miss_stride_only, miss_stride_idxen_only);
        for (int mode = 0; mode < NumModes; ++mode) {
            LOG_INFO(Render_Vulkan,
                     "Precompile holdout as COMPILE, SHADPS4_PRECOMPILE_STRIDE={}: tested={} "
                     "hit@1={} within={} missed={} permutations={} (table leave-one-out)",
                     mode, tested, mode_hit1[mode], mode_hits[mode], mode_missed[mode],
                     mode_perms[mode]);
        }
        LOG_INFO(Render_Vulkan,
                 "Precompile holdout predicted strides: spv_equal={} spv_diff={} (every predicted "
                 "candidate that matched a full real permutation)",
                 pred_spv_equal, pred_spv_diff);
    }
    if (!compile) {
        return;
    }

    // Fable finding 6: a translation that stops the emulator (an assert in the decoder or the
    // translator) must not stop every later launch too. The hash being read, decoded or translated
    // is in the in-progress sidecar (flushed) until the pass ends. A launch that finds one left over
    // records it as a suspect; a hash left over twice (the user may also have closed a slow boot
    // once) is never precompiled again (Opus review 2026-10-07, finding 2). Only COMPILE (and its
    // stride table) is guarded: VERIFY and HOLDOUT are diagnostics.
    const auto inprogress_path =
        Storage::DataBase::Instance().SidecarPath(PrecompileInProgressFile);
    if (!inprogress_path.empty()) {
        std::ifstream last{inprogress_path};
        std::string line;
        if (std::getline(last, line) && line.size() >= 16) {
            const u64 died_at = std::strtoull(line.c_str(), nullptr, 16);
            const bool again = ReadPrecompileHashes(PrecompileSuspectFile).contains(died_at);
            AppendPrecompileHash(again ? PrecompileFailedFile : PrecompileSuspectFile, died_at);
            LOG_WARNING(Render_Vulkan,
                        "Precompile: the last launch stopped while on cs_{:#x}; {} ({})", died_at,
                        again ? "the second time, it is never precompiled again"
                              : "suspect (twice -> never precompiled again)",
                        again ? PrecompileFailedFile : PrecompileSuspectFile);
        }
    }
    const auto failed = ReadPrecompileHashes(PrecompileFailedFile);
    const char* fail_at_env = std::getenv("SHADPS4_PRECOMPILE_FAIL_AT");
    // "first": the first program COMPILE translates, before any entry of this pass is saved.
    const bool fail_first = fail_at_env && std::string_view{fail_at_env} == "first";
    const u64 fail_at = fail_at_env && !fail_first ? std::strtoull(fail_at_env, nullptr, 16) : 0;
    std::ofstream progress;
    if (!inprogress_path.empty()) {
        progress.open(inprogress_path, std::ios::binary | std::ios::trunc);
    }
    auto mark_progress = [&](u64 hash) {
        if (progress.is_open()) {
            progress.seekp(0);
            progress << fmt::format("{:016x}\n", hash);
            progress.flush();
        }
        if (fail_first || (fail_at != 0 && hash == fail_at)) {
            UNREACHABLE_MSG("Precompile: SHADPS4_PRECOMPILE_FAIL_AT cs_{:#x} (self-test of the "
                            "in-progress sidecar)",
                            hash);
        }
    };

    // SHADPS4_PRECOMPILE_STRIDE=2/4/5: the table of real strides by highest accessed byte, from
    // one full cached permutation per learned program (each probed once, ~1 ms). Built only when
    // there is something to compile.
    StrideTable compile_table;
    const StrideTable* table_ptr = nullptr;
    if (stride_mode == 2 || stride_mode == 4 || stride_mode == 5) {
        bool any_new = false;
        for (const auto& [hash, entry] : manifest) {
            if (!cached_programs.contains(hash) && !program_cache.contains(hash) &&
                !precompile_retired.contains(hash) && !failed.contains(hash)) {
                any_new = true;
                break;
            }
        }
        if (any_new) {
            const auto t_table = Clock::now();
            const size_t arena_before = Shader::SrtWalkerArenaUsed();
            u32 probed = 0, skipped = 0, arena_stop = 0;
            for (const auto& [hash, source] : table_sources) {
                // Opus delta review 2026-10-07: every translation appends SRT walker code to a fixed
                // 32 MB arena that throws when full (Fable finding 7); the table is an optimization,
                // so it stops well before (13-14 MB used in the gates of the night).
                if (Shader::SrtWalkerArenaUsed() > 24_MB) {
                    ++arena_stop;
                    break;
                }
                const auto it = manifest.find(hash);
                if (it == manifest.end() || failed.contains(hash)) {
                    ++skipped;
                    continue;
                }
                mark_progress(hash); // before the decoder (ScanCode asserts on bad code)
                const auto code = ReadCode(it->second);
                if (code.empty()) {
                    ++skipped;
                    continue;
                }
                const auto traits = ScanCode(code);
                if (traits.images || traits.movrel) {
                    ++skipped;
                    continue;
                }
                const auto probe = probe_shape(code, hash);
                if (probe.shape != source.shape) {
                    ++skipped;
                    continue;
                }
                ++probed;
                for (size_t i = 0; i < probe.buffers.size() && i < probe.access.size() &&
                                   i < source.strides.size();
                     ++i) {
                    if (IsGuessableBuffer(probe.buffers[i]) && probe.access[i].idxen &&
                        probe.access[i].sized) {
                        compile_table.Add(probe.access[i].max_end, source.strides[i], 1);
                    }
                }
            }
            table_ptr = &compile_table;
            LOG_INFO(Render_Vulkan,
                     "Precompile stride table: sources={} probed={} skipped={} pairs={} ends={} "
                     "ms={:.0f} | SRT arena {} -> {} KiB{}",
                     table_sources.size(), probed, skipped, compile_table.pairs,
                     compile_table.counts.size(), MsSince(t_table), arena_before >> 10,
                     Shader::SrtWalkerArenaUsed() >> 10,
                     arena_stop ? " (stopped at 24 MiB)" : "");
        }
    }

    // COMPILE: every listed program the cache does not hold.
    u32 listed_new = 0, image_users = 0, movrel_users = 0, no_shape = 0, drifted = 0,
        programs_built = 0, other_nud_skipped = 0, skipped_failed = 0, retired_skipped = 0;
    u32 permutations = 0, duplicate_specs = 0, layout_mismatch = 0, predicted_contexts = 0;
    double translate_ms = 0.0;
    const auto t_compile = Clock::now();
    for (const auto& [hash, entry] : manifest) {
        if (cached_programs.contains(hash) || program_cache.contains(hash)) {
            continue;
        }
        ++listed_new;
        if (precompile_retired.contains(hash)) {
            // Fix A: guessed once and missed live; the live path owns it now.
            ++retired_skipped;
            continue;
        }
        if (failed.contains(hash)) {
            ++skipped_failed;
            continue;
        }
        mark_progress(hash); // before the decoder (ScanCode asserts on bad code)
        const auto code = ReadCode(entry);
        if (code.empty()) {
            continue;
        }
        const auto traits = ScanCode(code);
        if (traits.images || traits.movrel) {
            ++(traits.images ? image_users : movrel_users);
            continue;
        }
        const auto t_translate = Clock::now();

        const auto probe = probe_shape(code, hash);
        const u64 shape = probe.shape;
        const size_t flat_size = probe.flat_size;
        const auto it_shape = shape ? shapes.find(shape) : shapes.end();
        if (it_shape == shapes.end()) {
            ++no_shape;
            translate_ms += MsSince(t_translate);
            continue;
        }

        // The candidate contexts, most frequent first; with SHADPS4_PRECOMPILE_STRIDE the indexed
        // buffers' strides predicted from this program's code (StrideVariants).
        const size_t max_attempts = size_t{guesses} * 2;
        const auto candidates = MakeCandidates(it_shape->second.ranked, stride_mode, max_attempts,
                                               probe.buffers, probe.access, table_ptr);
        for (const auto& candidate : candidates) {
            predicted_contexts += candidate.variant != StrideVariant::Learned ? 1 : 0;
        }

        // Pass 2, one permutation per candidate context.
        std::unique_ptr<Program> program;
        u32 first_num_user_data = 0;
        std::vector<std::vector<u32>> flats;
        std::vector<UserData> uds;
        flats.reserve(max_attempts);
        uds.reserve(max_attempts);
        for (const Candidate& candidate : candidates) {
            const Context* ctx = &candidate.ctx;
            if ((program && program->modules.size() >= guesses) || uds.size() >= max_attempts) {
                break;
            }
            // Fable finding 1: every permutation is pushed through the program's shared Info, made
            // with the first guess; a guess with another user-SGPR count would push a different
            // layout. One num_user_data per program (the most frequent context's).
            if (program && ctx->runtime_info.num_user_data != first_num_user_data) {
                ++other_nud_skipped;
                continue;
            }
            if (!program) {
                first_num_user_data = ctx->runtime_info.num_user_data;
            }
            auto& flat = flats.emplace_back(make_flat(*ctx, flat_size));
            auto& ud = uds.emplace_back();
            std::copy_n(flat.begin(), ud.size(), ud.begin());
            const Shader::ShaderParams params{.user_data = ud, .code = code, .hash = hash};
            auto runtime_info = ctx->runtime_info;
            Shader::g_synthetic_flat = &flat;

            vk::ShaderModule module{};
            Shader::StageSpecialization spec{};
            if (!program) {
                program = std::make_unique<Program>(Shader::Stage::Compute,
                                                    Shader::LogicalStage::Compute, params);
                Shader::Backend::Bindings binding{};
                const auto ir =
                    Shader::TranslateProgram(code, pools, program->info, runtime_info, profile);
                auto spv = Shader::Backend::SPIRV::EmitSPIRV(profile, runtime_info, ir, binding);
                if (ShapeHash(program->info) != shape) {
                    ++drifted;
                    program.reset();
                    break;
                }
                module = CompileSPV(spv, instance.GetDevice());
                RegisterShaderBinary(std::move(spv), hash, 0);
                spec = Shader::StageSpecialization(program->info, runtime_info, profile, {});
            } else {
                // As GetProgram for a new permutation: the spec from the program's Info refreshed
                // with this context, the module from a fresh Info pinned to the program's DMA use.
                program->info.user_data = ud;
                program->info.RefreshFlatBuf();
                spec = Shader::StageSpecialization(program->info, runtime_info, profile, {});
                if (std::ranges::find(program->modules, spec, &Program::Module::spec) !=
                    program->modules.end()) {
                    ++duplicate_specs;
                    continue;
                }
                Shader::Info new_info{Shader::Stage::Compute, Shader::LogicalStage::Compute,
                                      params};
                new_info.pinned_uses_dma = program->info.uses_dma;
                Shader::Backend::Bindings binding{};
                const auto ir =
                    Shader::TranslateProgram(code, pools, new_info, runtime_info, profile);
                auto spv = Shader::Backend::SPIRV::EmitSPIRV(profile, runtime_info, ir, binding);
                if (!SameBindingLayout(new_info, program->info)) {
                    ++layout_mismatch;
                    continue;
                }
                module = CompileSPV(spv, instance.GetDevice());
                RegisterShaderBinary(std::move(spv), hash, program->modules.size());
            }
            const size_t perm_idx = program->modules.size();
            const u64 perm_hash = HashCombine(hash, perm_idx);
            RegisterShaderMeta(program->info, std::nullopt, spec, perm_hash, perm_idx, true);
            program->AddPermut(module, std::move(spec));

            const ComputePipelineKey key{perm_hash};
            ComputePipeline::SerializationSupport sdata{};
            const auto [it_pl, is_new] = compute_pipelines.try_emplace(key);
            if (is_new) {
                it_pl.value() = std::make_unique<ComputePipeline>(
                    instance, scheduler, desc_heap, profile, *pipeline_cache, key, program->info,
                    module, sdata, true);
                RegisterPipelineData(key, sdata);
            }
            ++permutations;
        }
        Shader::g_synthetic_flat = nullptr;
        translate_ms += MsSince(t_translate);
        if (!program || program->modules.empty()) {
            continue;
        }
        // Guest user data and the code address are refreshed by GetProgram before any use.
        program->info.user_data = {};
        program->precompiled = true;
        program_cache.try_emplace(hash, std::move(program));
        ++programs_built;
    }
    // The pass ended: nothing is in translation.
    if (progress.is_open()) {
        // Same stream, fixed position: a kill can no longer leave the last program blamed.
        // getline reads "done" even with an old hash's tail after it.
        progress.seekp(0);
        progress << "done\n";
        progress.flush();
    }
    LOG_INFO(Render_Vulkan,
             "Precompile compile: listed_new={} image_users={} movrel_users={} no_shape={} drifted={} other_nud_skipped={} "
             "programs={} permutations={} duplicate_specs={} layout_mismatch={} guesses={} "
             "stride_mode={} predicted_contexts={} retired_skipped={} failed_skipped={} "
             "translate_ms={:.0f} compile_phase_ms={:.0f} total_ms={:.0f}",
             listed_new, image_users, movrel_users, no_shape, drifted, other_nud_skipped, programs_built, permutations,
             duplicate_specs, layout_mismatch, guesses, stride_mode, predicted_contexts,
             retired_skipped, skipped_failed, translate_ms, MsSince(t_compile),
             MsSince(t_start));
    // Fable finding 7: every translation appends an SRT walker to a fixed 32 MB arena.
    LOG_INFO(Render_Vulkan, "Precompile: SRT walker arena {} KiB used of 32768",
             Shader::SrtWalkerArenaUsed() >> 10);
}

} // namespace Vulkan
