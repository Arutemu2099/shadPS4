// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <codecvt>
#include <fstream>
#include <sstream>
#include <string>
#include <nlohmann/json.hpp>
#include <pugixml.hpp>
#include "common/elf_info.h"
#include "common/logging/log.h"
#include "common/path_util.h"
#include "core/emulator_state.h"
#include "core/file_format/psf.h"
#include "memory_patcher.h"

namespace VideoCore {
void PS4_SYSV_ABI ParticleArenaDiscardHook(); // buffer_cache.cpp
}

namespace MemoryPatcher {

EXPORT uintptr_t g_eboot_address;
uint64_t g_eboot_image_size;
std::string g_game_serial;
std::string patch_file;
bool patches_applied = false;
std::vector<patchInfo> pending_patches;
static bool g_particle_late_read_applied = false; // V1 written (ApplyParticleLightOneSubmit needs it)

std::string toHex(u64 value, size_t byteSize) {
    std::stringstream ss;
    ss << std::hex << std::setfill('0') << std::setw(byteSize * 2) << value;
    return ss.str();
}

static bool isHexSym(const std::string& s) {
    return (s.size() >= 2 && (s[0] == '$' || s[0] == '#'));
}

static bool isHex0x(const std::string& s) {
    return (s.size() >= 3 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X'));
}

// possible prefixes == (s == `0x` || s == `#` || s == `$`) will be interpet as hex value, else is
// decimal
static int convertNumBase(const std::string& s) {
    return (isHex0x(s) || isHexSym(s)) ? 16 : 10;
}

std::string convertValueToHex(const std::string type, const std::string valueStr) {
    std::string result;

    if (type == "byte") {
        const u32 value = std::stoul(valueStr, nullptr, convertNumBase(valueStr));
        result = toHex(value, 1);
    } else if (type == "bytes16") {
        const u32 value = std::stoul(valueStr, nullptr, convertNumBase(valueStr));
        result = toHex(value, 2);
    } else if (type == "bytes32") {
        const u32 value = std::stoul(valueStr, nullptr, convertNumBase(valueStr));
        result = toHex(value, 4);
    } else if (type == "bytes64") {
        const u64 value = std::stoull(valueStr, nullptr, convertNumBase(valueStr));
        result = toHex(value, 8);
    } else if (type == "float32") {
        union {
            float f;
            uint32_t i;
        } floatUnion;
        floatUnion.f = std::stof(valueStr);
        result = toHex(std::byteswap(floatUnion.i), sizeof(floatUnion.i));
    } else if (type == "float64") {
        union {
            double d;
            uint64_t i;
        } doubleUnion;
        doubleUnion.d = std::stod(valueStr);
        result = toHex(std::byteswap(doubleUnion.i), sizeof(doubleUnion.i));
    } else if (type == "utf8") {
        std::vector<unsigned char> byteArray =
            std::vector<unsigned char>(valueStr.begin(), valueStr.end());
        byteArray.push_back('\0');
        std::stringstream ss;
        for (unsigned char c : byteArray) {
            ss << std::hex << std::setfill('0') << std::setw(2) << static_cast<int>(c);
        }
        result = ss.str();
    } else if (type == "utf16") {
        std::wstring wide_str(valueStr.size(), L'\0');
        std::mbstowcs(&wide_str[0], valueStr.c_str(), valueStr.size());
        wide_str.resize(std::wcslen(wide_str.c_str()));

        std::u16string valueStringU16;

        for (wchar_t wc : wide_str) {
            if (wc <= 0xFFFF) {
                valueStringU16.push_back(static_cast<char16_t>(wc));
            } else {
                wc -= 0x10000;
                valueStringU16.push_back(static_cast<char16_t>(0xD800 | (wc >> 10)));
                valueStringU16.push_back(static_cast<char16_t>(0xDC00 | (wc & 0x3FF)));
            }
        }

        std::vector<unsigned char> byteArray;
        // convert to little endian
        for (char16_t ch : valueStringU16) {
            unsigned char low_byte = static_cast<unsigned char>(ch & 0x00FF);
            unsigned char high_byte = static_cast<unsigned char>((ch >> 8) & 0x00FF);

            byteArray.push_back(low_byte);
            byteArray.push_back(high_byte);
        }
        byteArray.push_back('\0');
        byteArray.push_back('\0');
        std::stringstream ss;

        for (unsigned char ch : byteArray) {
            ss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(ch);
        }
        result = ss.str();
    } else if (type == "bytes") {
        result = valueStr;
    } else if (type == "mask" || type == "mask_jump32") {
        result = valueStr;
    } else {
        LOG_INFO(Loader, "Error applying Patch, unknown type: {}", type);
    }
    return result;
}

void ApplyPendingPatches();

void ApplyPatchesFromXML(std::filesystem::path path) {
    pugi::xml_document doc;
    pugi::xml_parse_result result = doc.load_file(path.c_str());

    auto* param_sfo = Common::Singleton<PSF>::Instance();
    auto app_version = param_sfo->GetString("APP_VER").value_or("Unknown version");

    if (result) {
        auto patchXML = doc.child("Patch");
        for (pugi::xml_node_iterator it = patchXML.children().begin();
             it != patchXML.children().end(); ++it) {

            if (std::string(it->name()) == "Metadata") {
                if (std::string(it->attribute("isEnabled").value()) == "true") {
                    std::string currentPatchName = it->attribute("Name").value();
                    std::string metadataAppVer = it->attribute("AppVer").value();
                    bool versionMatches = metadataAppVer == app_version;

                    auto patchList = it->first_child();
                    for (pugi::xml_node_iterator patchLineIt = patchList.children().begin();
                         patchLineIt != patchList.children().end(); ++patchLineIt) {

                        std::string type = patchLineIt->attribute("Type").value();
                        if (!versionMatches && type != "mask" && type != "mask_jump32")
                            continue;

                        std::string address = patchLineIt->attribute("Address").value();
                        std::string patchValue = patchLineIt->attribute("Value").value();
                        std::string maskOffsetStr = patchLineIt->attribute("Offset").value();
                        std::string targetStr = "";
                        std::string sizeStr = "";
                        if (type == "mask_jump32") {
                            targetStr = patchLineIt->attribute("Target").value();
                            sizeStr = patchLineIt->attribute("Size").value();
                        } else {
                            try {
                                patchValue = convertValueToHex(type, patchValue);
                            } catch (std::exception& e) {
                                ASSERT_MSG(false,
                                           "Failed to parse patch value \"{}\" for \"{}\" in "
                                           "patch \"{}\", error: \"{}\"\n"
                                           "If the patch was working on earlier versions, then it "
                                           "was using a format that shadPS4 handled incorrectly, "
                                           "and the patch should instead be fixed.",
                                           patchValue, address, currentPatchName, e.what());
                            }
                        }

                        bool littleEndian = false;
                        if (type == "bytes16" || type == "bytes32" || type == "bytes64") {
                            littleEndian = true;
                        }

                        MemoryPatcher::PatchMask patchMask = MemoryPatcher::PatchMask::None;
                        int maskOffsetValue = 0;

                        if (type == "mask")
                            patchMask = MemoryPatcher::PatchMask::Mask;

                        if (type == "mask_jump32")
                            patchMask = MemoryPatcher::PatchMask::Mask_Jump32;

                        if ((type == "mask" || type == "mask_jump32") && !maskOffsetStr.empty()) {
                            maskOffsetValue = std::stoi(maskOffsetStr, 0, 10);
                        }

                        const patchInfo patch = {
                            .gameSerial = "*",
                            .modNameStr = currentPatchName,
                            .offsetStr = address,
                            .valueStr = patchValue,
                            .targetStr = targetStr,
                            .sizeStr = sizeStr,
                            .littleEndian = littleEndian,
                            .patchMask = patchMask,
                            .maskOffset = maskOffsetValue,
                        };
                        MemoryPatcher::PatchMemory(patch);
                    }
                }
            }
        }
    } else {
        LOG_ERROR(Loader, "Could not parse patch XML: {}", result.description());
    }
}

// SHADPS4_PARTICLE_LATE_READ=1 (2026-10-12, notes/one-frame-late-particles-2026-10-12.md sec. 3.1, V1):
// inFAMOUS Second Son (CUSA00004 / 01.07) reads its particle control blocks (survivor counts, light
// counts, CPU collision) at the start of the next frame, just before 0x3ee7d0 consumes them, instead
// of right after kicking this frame's particle chain. Game:Main then no longer waits for that chain's
// GPU work mid-frame. Three writes (eboot offsets): a cave in the unreferenced 0x3ee160 that calls
// 0x3edef0 (readback) and 0x3ee470 (collision join + label wait), then sets up 0x3ee7d0's arguments
// and jumps back; the main loop jumps to it at 0x29cc89; the old 0x3edef0 call at 0x29cd1e becomes a
// 5-byte NOP. The end-of-frame 0x3ee470 call at 0x29cd56 stays. Refused unless every byte matches.
static void ApplyParticleLateRead() {
    const char* env = std::getenv("SHADPS4_PARTICLE_LATE_READ");
    if (!env || env[0] != '1') {
        return;
    }
    // No serial / APP_VER check (2026-10-19): a DLC param.sfo in a player's game folder read as
    // "Unknown version" and switched this off; the byte compare below is the gate.
    struct Bytes {
        u64 offset;
        std::vector<u8> bytes;
    };
    // Original bytes: the patched ranges plus the instructions around them the cave relies on.
    const std::vector<Bytes> original = {
        // FUN_8003ee160 (no references): 43 code bytes + 5 pad
        {0x3ee160, {0x55, 0x48, 0x89, 0xe5, 0x53, 0x50, 0x48, 0x8b, 0x5f, 0x58, 0xeb, 0x13,
                    0x0f, 0x1f, 0x40, 0x00, 0x48, 0x89, 0xdf, 0xe8, 0x58, 0x5d, 0x00, 0x00,
                    0x48, 0x8b, 0x9b, 0xb0, 0x00, 0x00, 0x00, 0x48, 0x85, 0xdb, 0x75, 0xec,
                    0x48, 0x83, 0xc4, 0x08, 0x5b, 0x5d, 0xc3, 0x90, 0x90, 0x90, 0x90, 0x90}},
        // call 0x41b8d0; lea rax,[0x800f1bcd0]; lea rdi,[0x807d3eb10]; mov rsi,[rax+0x1dca0];
        // call 0x3ee7d0
        {0x29cc84, {0xe8, 0x47, 0xec, 0x17, 0x00, 0x48, 0x8d, 0x05, 0x40, 0xf0, 0xc7,
                    0x00, 0x48, 0x8d, 0x3d, 0x79, 0x1e, 0xaa, 0x07, 0x48, 0x8b, 0xb0,
                    0xa0, 0xdc, 0x01, 0x00, 0xe8, 0x2d, 0x1b, 0x15, 0x00}},
        // lea rbx,[0x807d3eb10]; mov rdi,rbx; call 0x3edef0
        {0x29cd14, {0x48, 0x8d, 0x1d, 0xf5, 0x1d, 0xaa, 0x07, 0x48, 0x89, 0xdf, 0xe8, 0xcd, 0x11,
                    0x15, 0x00}},
        // mov rdi,rbx; call 0x3ee470 (kept)
        {0x29cd53, {0x48, 0x89, 0xdf, 0xe8, 0x15, 0x17, 0x15, 0x00}},
    };
    const std::vector<Bytes> patched = {
        // lea rbx,[0x807d3eb10]; mov rdi,rbx; call 0x3edef0; mov rdi,rbx; call 0x3ee470;
        // lea rax,[0x800f1bcd0]; mov rdi,rbx; mov rsi,[rax+0x1dca0]; jmp 0x29cc9e
        {0x3ee160, {0x48, 0x8d, 0x1d, 0xa9, 0x09, 0x95, 0x07, 0x48, 0x89, 0xdf, 0xe8, 0x81,
                    0xfd, 0xff, 0xff, 0x48, 0x89, 0xdf, 0xe8, 0xf9, 0x02, 0x00, 0x00, 0x48,
                    0x8d, 0x05, 0x52, 0xdb, 0xb2, 0x00, 0x48, 0x89, 0xdf, 0x48, 0x8b, 0xb0,
                    0xa0, 0xdc, 0x01, 0x00, 0xe9, 0x11, 0xeb, 0xea, 0xff}},
        // jmp 0x3ee160 + 16 NOPs up to the call 0x3ee7d0 at 0x29cc9e
        {0x29cc89, {0xe9, 0xd2, 0x14, 0x15, 0x00, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90,
                    0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90}},
        // nopl 0x0(rax,rax,1) in place of call 0x3edef0
        {0x29cd1e, {0x0f, 0x1f, 0x44, 0x00, 0x00}},
    };
    for (const auto& o : original) {
        const auto* live = reinterpret_cast<const u8*>(g_eboot_address + o.offset);
        if (std::memcmp(live, o.bytes.data(), o.bytes.size()) != 0) {
            std::string found;
            for (size_t i = 0; i < o.bytes.size(); ++i) {
                found += fmt::format("{:02x}", live[i]);
            }
            LOG_ERROR(Loader, "SHADPS4_PARTICLE_LATE_READ: refused, eboot+{:#x} holds {}", o.offset,
                      found);
            return;
        }
    }
    // SHADPS4_PARTICLE_ARENA_DISCARD=1 (2026-10-13): the main loop first calls a host hook that
    // drops the GPU-modified state of the arena half 0x3ee7d0 is about to reuse (dead data), then
    // jumps to the cave: movabs rax, hook; call rax; jmp 0x3ee160; 4 NOPs (21 bytes at 0x29cc89).
    std::vector<Bytes> writes = patched;
    const char* discard = std::getenv("SHADPS4_PARTICLE_ARENA_DISCARD");
    if (discard && discard[0] == '1') {
        const u64 hook = reinterpret_cast<u64>(&VideoCore::ParticleArenaDiscardHook);
        std::vector<u8> b = {0x48, 0xb8};
        for (int i = 0; i < 8; ++i) {
            b.push_back(static_cast<u8>(hook >> (8 * i)));
        }
        // Opus review (q43): the hook runs AFTER the cave (the cave may still read a stale
        // control block in the half being discarded): 0x29cc89 jmp cave (as V1); the cave's
        // last jmp returns to 0x29cc8e = push rdi; push rsi; movabs rax, hook; call rax;
        // pop rsi; pop rdi (16 bytes, ends at 0x29cc9e = call 0x3ee7d0).
        std::vector<u8> tail = {0x57, 0x56};
        tail.insert(tail.end(), b.begin(), b.end());
        tail.insert(tail.end(), {0xff, 0xd0, 0x5e, 0x5f});
        std::vector<u8> jump = {0xe9, 0xd2, 0x14, 0x15, 0x00};
        jump.insert(jump.end(), tail.begin(), tail.end());
        writes[1] = {0x29cc89, jump};
        writes[0].bytes[41] = 0x01; // cave: jmp 0x29cc8e (E9 01 EB EA FF) instead of 0x29cc9e
        if (writes[1].bytes.size() != 21 || writes[0].bytes[40] != 0xe9) {
            LOG_ERROR(Loader, "SHADPS4_PARTICLE_ARENA_DISCARD: bad patch layout, not applied");
            writes = patched;
        } else {
            LOG_INFO(Loader, "SHADPS4_PARTICLE_ARENA_DISCARD: frame-start hook {:#x} after the cave",
                     hook);
        }
    }
    for (const auto& p : writes) {
        std::memcpy(reinterpret_cast<void*>(g_eboot_address + p.offset), p.bytes.data(),
                    p.bytes.size());
    }
    g_particle_late_read_applied = true;
    LOG_INFO(Loader,
             "SHADPS4_PARTICLE_LATE_READ: applied V1 (particle readback one frame later): cave "
             "{:#x}, jump {:#x}, nop {:#x}",
             g_eboot_address + 0x3ee160, g_eboot_address + 0x29cc89, g_eboot_address + 0x29cd1e);
}

// SHADPS4_PARTICLE_LIGHT_ONE_SUBMIT=1 (2026-10-16, notes/study-fight-dips-2026-10-15.md sec. 5, G1):
// 0x3f3ed0 records two DMA_DATAs per light-emitting particle effect and then submits the ring
// (call 0x1691e0 at eboot 0x3f4036) - one ring submit per light, ~25 per typical fight frame, ~74 in
// the worst. With V1 (PARTICLE_LATE_READ) the copies stay in the command buffer 0x807d3cf20 and go out
// with 0x3ee7d0's first submit of the same frame, so the call becomes a 5-byte NOP. Only with V1
// applied: in the vanilla order the copies would wait for the next frame (lights one frame late).
static void ApplyParticleLightOneSubmit() {
    const char* env = std::getenv("SHADPS4_PARTICLE_LIGHT_ONE_SUBMIT");
    if (!env || env[0] != '1') {
        return;
    }
    if (!g_particle_late_read_applied) {
        LOG_ERROR(Loader, "SHADPS4_PARTICLE_LIGHT_ONE_SUBMIT: refused, needs SHADPS4_PARTICLE_LATE_READ "
                          "applied (V1)");
        return;
    }
    constexpr u64 offset = 0x3f4036;
    static constexpr u8 original[] = {0xe8, 0xa5, 0x51, 0xd7, 0xff}; // call 0x1691e0
    static constexpr u8 nop5[] = {0x0f, 0x1f, 0x44, 0x00, 0x00};
    auto* live = reinterpret_cast<u8*>(g_eboot_address + offset);
    if (std::memcmp(live, original, sizeof(original)) != 0) {
        LOG_ERROR(Loader, "SHADPS4_PARTICLE_LIGHT_ONE_SUBMIT: refused, eboot+{:#x} holds "
                          "{:02x} {:02x} {:02x} {:02x} {:02x}",
                  offset, live[0], live[1], live[2], live[3], live[4]);
        return;
    }
    std::memcpy(live, nop5, sizeof(nop5));
    LOG_INFO(Loader, "SHADPS4_PARTICLE_LIGHT_ONE_SUBMIT: per-light ring submit at {:#x} is a NOP",
             g_eboot_address + offset);
}

// SHADPS4_HOUSE_SPLIT=1 (2026-10-14, notes/study-house-draws-2026-10-14.md sec. 3): the game's
// command-buffer splitter FUN_80043cef0 cuts a pass's item list into jobs of <= 1000 items only where
// key & mask changes; the mask comes from FUN_800197130 (default 0xffff000000000000 = class runs).
// The rooftop's ~3,300 distant houses are one class run, so ONE job records them all. Writing
// 0xffffffffffff0000 (everything above bit 16: a superset of today's cut points) lets the splitter
// cut that run into ~5 jobs recorded in parallel. SHADPS4_HOUSE_CHUNK=N also sets the 1000-item chunk
// (FUN_800197050's imm32). Refused unless every original byte matches. The game's context pool has
// no bounds check (at most 254 per frame): the census in gnmdriver logs the count per frame.
static int g_screen_w = 0, g_screen_h = 0;

void SetScreenSize(int width, int height) {
    g_screen_w = width;
    g_screen_h = height;
}

// SHADPS4_ASPECT (1.1, user 2026-10-20; EXPERIMENTS "v1.1 C"): the game fills the screen's shape.
// The renderer's size setup FUN_8001eff00 stores the render size (+0xc / +0x10), the output size
// ([0] / [4]) and the 2D screen space (globals 0x801d0ccc0 / ccc4: HUD layout, and the camera's
// aspect = ccc0 / ccc4 at camera creation, 0x80058030c); the mode toggle FUN_8001f0be0 re-applies
// render / output from esi / edi. All eight 1920 / 1080 immediates get the target size: the
// buffers, the video-out buffers, the HUD (anchored to the edges) and the camera follow (Hor+: the
// vertical view stays, wider screens see more at the sides). Measured on 21:9 (s1426): round
// shapes, more picture left and right, HUD at the new edges, 0 Critical.
//   auto (default): only screens wider than 16:9 (narrow ones keep 16:9 with bars until the camera
//        can add picture above / below instead of cutting the sides); crop: narrow screens too
//        (Hor+: the sides are cut); W:H (e.g. 21:9, 32:9, 4:3): that shape; off.
// Decided once at boot (the size setup runs once): a new window shape needs a restart. DLSS steps
// aside at a non-1080p frame (vk_rasterizer DlssResolve).
static void ApplyAspectRatio() {
    const char* env = std::getenv("SHADPS4_ASPECT");
    const std::string mode = env ? env : "auto";
    if (mode == "off" || mode == "0") {
        return;
    }
    double aspect = 0.0;
    if (const auto colon = mode.find(':'); colon != std::string::npos) {
        const double w = std::atof(mode.substr(0, colon).c_str());
        const double h = std::atof(mode.substr(colon + 1).c_str());
        aspect = h > 0.0 ? w / h : 0.0;
    } else if (g_screen_w > 0 && g_screen_h > 0) {
        aspect = double(g_screen_w) / double(g_screen_h);
    }
    constexpr double base = 16.0 / 9.0;
    if (aspect <= 0.0 || std::abs(aspect - base) < 0.01) {
        return; // 16:9 or unknown: nothing to do
    }
    const bool narrow = aspect < base;
    if (narrow && mode == "auto") {
        LOG_INFO(Loader, "SHADPS4_ASPECT: the screen ({}x{}) is narrower than 16:9; the game keeps "
                         "16:9 with bars (SHADPS4_ASPECT=crop fills it by cutting the sides)",
                 g_screen_w, g_screen_h);
        return;
    }
    // Multiples of 8; wide: height 1080, narrow: width 1920 (at most 3840 x 1920).
    const auto round8 = [](double v) { return u32(std::lround(v / 8.0) * 8); };
    const u32 w = narrow ? 1920u : std::min(3840u, round8(1080.0 * aspect));
    const u32 h = narrow ? std::min(1920u, round8(1920.0 / aspect)) : 1080u;
    struct Imm {
        u64 offset;
        u32 original;
        bool width;
    };
    // Eboot 01.07 offsets (byte-compared before any write; FUN_8001eff00, FUN_8001f0be0).
    static constexpr Imm imms[] = {
        {0x1eff4a, 0x780, true},  {0x1eff51, 0x438, false}, // render size
        {0x1eff57, 0x780, true},  {0x1eff5e, 0x438, false}, // output size
        {0x1f0403, 0x780, true},  {0x1f03fb, 0x438, false}, // 2D screen space = camera aspect
        {0x1f0d3e, 0x780, true},  {0x1f0d4d, 0x438, false}, // mode toggle (normal mode)
    };
    for (const auto& imm : imms) {
        u32 live;
        std::memcpy(&live, reinterpret_cast<const void*>(g_eboot_address + imm.offset), 4);
        if (live != imm.original) {
            LOG_ERROR(Loader, "SHADPS4_ASPECT: refused, eboot+{:#x} holds {:#x} (not {:#x})",
                      imm.offset, live, imm.original);
            return;
        }
    }
    for (const auto& imm : imms) {
        const u32 value = imm.width ? w : h;
        std::memcpy(reinterpret_cast<void*>(g_eboot_address + imm.offset), &value, 4);
    }
    LOG_INFO(Loader, "SHADPS4_ASPECT: the game renders {}x{} (aspect {:.3f}, screen {}x{}, {})", w,
             h, aspect, g_screen_w, g_screen_h, mode);
}

static void ApplyHouseSplit() {
    const char* env = std::getenv("SHADPS4_HOUSE_SPLIT");
    const char* chunk_env = std::getenv("SHADPS4_HOUSE_CHUNK");
    // =2 (diagnostic, study 11:35): FUN_800197130 -> mov rax, -1; ret - every run > 1000 items is
    // cut at its chunk (200 in passes 0x13 / 0x17); tells apart "houses in another pass / equal keys"
    // (contexts up, span down) from "the pass is walked serially" (contexts up, span flat).
    const bool all_ones = env && env[0] == '2';
    const bool mask_on = env && (env[0] == '1' || all_ones);
    const u32 chunk = chunk_env ? static_cast<u32>(std::strtoul(chunk_env, nullptr, 10)) : 0;
    if (!mask_on && chunk == 0) {
        return;
    }
    // No serial / APP_VER check (2026-10-19, as PARTICLE_LATE_READ): the byte compare is the gate.
    // FUN_800197130 (split mask by pass) and FUN_800197050 (chunk by pass), whole functions.
    static constexpr u8 mask_fn[58] = {
        0xba, 0x01, 0x00, 0x00, 0x00, 0x40, 0x88, 0xf9, 0xb8, 0x00, 0x00, 0xff, 0x7f, 0x48, 0xbe,
        0x00, 0x00, 0x00, 0x00, 0x1f, 0x00, 0x00, 0x00, 0x48, 0xd3, 0xe2, 0x48, 0x0f, 0xa3, 0xfe,
        0x72, 0x19, 0xf7, 0xc2, 0x00, 0x44, 0x88, 0x00, 0xb9, 0x00, 0x00, 0xff, 0xff, 0x48, 0xb8,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0x48, 0x0f, 0x45, 0xc1, 0xc3};
    static constexpr u8 chunk_fn[19] = {0x83, 0xcf, 0x04, 0xb8, 0xc8, 0x00, 0x00, 0x00, 0x83, 0xff,
                                        0x17, 0x74, 0x05, 0xb8, 0xe8, 0x03, 0x00, 0x00, 0xc3};
    const auto matches = [](u64 offset, const u8* bytes, size_t n) {
        const auto* live = reinterpret_cast<const u8*>(g_eboot_address + offset);
        if (std::memcmp(live, bytes, n) == 0) {
            return true;
        }
        std::string found;
        for (size_t i = 0; i < n; ++i) {
            found += fmt::format("{:02x}", live[i]);
        }
        LOG_ERROR(Loader, "SHADPS4_HOUSE_SPLIT: refused, eboot+{:#x} holds {}", offset, found);
        return false;
    };
    if (!matches(0x197130, mask_fn, sizeof(mask_fn)) ||
        !matches(0x197050, chunk_fn, sizeof(chunk_fn))) {
        return;
    }
    if (all_ones) {
        // mov rax, -1; ret at the function start (48 c7 c0 ff ff ff ff c3)
        static constexpr u8 ret_all[8] = {0x48, 0xc7, 0xc0, 0xff, 0xff, 0xff, 0xff, 0xc3};
        std::memcpy(reinterpret_cast<void*>(g_eboot_address + 0x197130), ret_all, sizeof(ret_all));
    } else if (mask_on) {
        // imm64 of mov rax, 0xffff000000000000 at 0x19715d -> 0xffffffffffff0000
        static constexpr u8 mask[8] = {0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
        std::memcpy(reinterpret_cast<void*>(g_eboot_address + 0x19715d), mask, sizeof(mask));
    }
    if (chunk != 0) {
        // imm32 of mov eax, 1000 at 0x19705e
        std::memcpy(reinterpret_cast<void*>(g_eboot_address + 0x19705e), &chunk, sizeof(chunk));
    }
    LOG_INFO(Loader, "SHADPS4_HOUSE_SPLIT: split mask {} | chunk {}",
             all_ones  ? "all ones in every pass (diagnostic)"
             : mask_on ? "default 0xffffffffffff0000"
                       : "unchanged",
             chunk != 0 ? chunk : 1000);
}

void OnGameLoaded() {
    ApplyParticleLateRead();
    ApplyParticleLightOneSubmit();
    ApplyHouseSplit();
    ApplyAspectRatio();
    std::filesystem::path patch_dir = Common::FS::GetUserPath(Common::FS::PathType::PatchesDir);
    if (!patch_file.empty()) {

        auto file_path = (patch_dir / patch_file).native();
        if (std::filesystem::exists(patch_file)) {
            ApplyPatchesFromXML(patch_file);
        } else {
            ApplyPatchesFromXML(file_path);
        }
    } else if (EmulatorState::GetInstance()->IsAutoPatchesLoadEnabled()) {
        for (auto const& repo : std::filesystem::directory_iterator(patch_dir)) {
            if (!repo.is_directory()) {
                continue;
            }
            std::ifstream json_file{repo.path() / "files.json"};
            nlohmann::json available_patches = nlohmann::json::parse(json_file);
            std::filesystem::path game_patch_file;
            for (auto const& [filename, serials] : available_patches.items()) {
                if (std::find(serials.begin(), serials.end(), g_game_serial) != serials.end()) {
                    game_patch_file = repo.path() / filename;
                    break;
                }
            }
            if (std::filesystem::exists(game_patch_file)) {
                ApplyPatchesFromXML(game_patch_file);
            }
        }
    }
    ApplyPendingPatches();
}

void AddPatchToQueue(const patchInfo& patchToAdd) {
    if (patches_applied) {
        PatchMemory(patchToAdd);
        return;
    }
    pending_patches.push_back(patchToAdd);
}

void ApplyPendingPatches() {
    patches_applied = true;
    for (size_t i = 0; i < pending_patches.size(); ++i) {
        const patchInfo& currentPatch = pending_patches[i];

        if (currentPatch.gameSerial != "*" && currentPatch.gameSerial != g_game_serial)
            continue;

        PatchMemory(currentPatch);
    }

    pending_patches.clear();
}

void PatchMemory(const patchInfo& patch) {
    // Send a request to modify the process memory.
    void* cheatAddress = nullptr;

    if (patch.patchMask == PatchMask::None) {
        if (patch.isOffset) {
            cheatAddress =
                reinterpret_cast<void*>(g_eboot_address + std::stoi(patch.offsetStr, 0, 16));
        } else {
            cheatAddress = reinterpret_cast<void*>(g_eboot_address +
                                                   (std::stoi(patch.offsetStr, 0, 16) - 0x400000));
        }
    }

    if (patch.patchMask == PatchMask::Mask) {
        cheatAddress = reinterpret_cast<void*>(PatternScan(patch.offsetStr) + patch.maskOffset);
    }

    if (patch.patchMask == PatchMask::Mask_Jump32) {
        int jumpSize = std::stoi(patch.sizeStr);

        constexpr int MAX_PATTERN_LENGTH = 256;
        if (jumpSize < 5) {
            LOG_ERROR(Loader, "Jump size must be at least 5 bytes");
            return;
        }
        if (jumpSize > MAX_PATTERN_LENGTH) {
            LOG_ERROR(Loader, "Jump size must be no more than {} bytes.", MAX_PATTERN_LENGTH);
            return;
        }

        // Find the base address using "Address"
        uintptr_t baseAddress = PatternScan(patch.offsetStr);
        if (baseAddress == 0) {
            LOG_ERROR(Loader, "PatternScan failed for mask_jump32 with pattern: {}",
                      patch.offsetStr);
            return;
        }
        uintptr_t patchAddress = baseAddress + patch.maskOffset;

        // Fills the original region (jumpSize bytes) with NOPs
        std::vector<u8> nopBytes(jumpSize, 0x90);
        std::memcpy(reinterpret_cast<void*>(patchAddress), nopBytes.data(), nopBytes.size());

        // Use "Target" to locate the start of the code cave
        uintptr_t jump_target = PatternScan(patch.targetStr);
        if (jump_target == 0) {
            LOG_ERROR(Loader, "PatternScan failed to Target with pattern: {}", patch.targetStr);
            return;
        }

        // Converts the Value attribute to a byte array (payload)
        std::vector<u8> payload;
        for (size_t i = 0; i < patch.valueStr.length(); i += 2) {
            std::string tempStr = patch.valueStr.substr(i, 2);
            const char* byteStr = tempStr.c_str();
            char* endPtr;
            unsigned int byteVal = std::strtoul(byteStr, &endPtr, 16);

            if (endPtr != byteStr + 2) {
                LOG_ERROR(Loader, "Invalid byte in Value: {}", patch.valueStr.substr(i, 2));
                return;
            }
            payload.push_back(static_cast<u8>(byteVal));
        }

        // Calculates the end of the code cave (where the return jump will be inserted)
        uintptr_t code_cave_end = jump_target + payload.size();

        // Write the payload to the code cave, from jump_target
        std::memcpy(reinterpret_cast<void*>(jump_target), payload.data(), payload.size());

        // Inserts the initial jump in the original region to divert to the code cave
        u8 jumpInstruction[5];
        jumpInstruction[0] = 0xE9;
        s32 relJump = static_cast<s32>(jump_target - patchAddress - 5);
        std::memcpy(&jumpInstruction[1], &relJump, sizeof(relJump));
        std::memcpy(reinterpret_cast<void*>(patchAddress), jumpInstruction,
                    sizeof(jumpInstruction));

        // Inserts jump back at the end of the code cave to resume execution after patching
        u8 jumpBack[5];
        jumpBack[0] = 0xE9;
        // Calculates the relative offset to return to the instruction immediately following the
        // overwritten region
        s32 target_return = static_cast<s32>((patchAddress + jumpSize) - (code_cave_end + 5));
        std::memcpy(&jumpBack[1], &target_return, sizeof(target_return));
        std::memcpy(reinterpret_cast<void*>(code_cave_end), jumpBack, sizeof(jumpBack));

        LOG_INFO(Loader,
                 "Applied Patch mask_jump32: {}, PatchAddress: {:#x}, JumpTarget: {:#x}, "
                 "CodeCaveEnd: {:#x}, JumpSize: {}",
                 patch.modNameStr, patchAddress, jump_target, code_cave_end, jumpSize);
        return;
    }

    if (cheatAddress == nullptr) {
        LOG_ERROR(Loader, "Failed to get address for patch {}", patch.modNameStr);
        return;
    }

    std::vector<unsigned char> bytePatch;

    for (size_t i = 0; i < patch.valueStr.length(); i += 2) {
        unsigned char byte = static_cast<unsigned char>(
            std::strtol(patch.valueStr.substr(i, 2).c_str(), nullptr, 16));

        bytePatch.push_back(byte);
    }

    if (patch.littleEndian) {
        std::reverse(bytePatch.begin(), bytePatch.end());
    }

    std::memcpy(cheatAddress, bytePatch.data(), bytePatch.size());

    LOG_INFO(Loader, "Applied patch: {}, Offset: {:#x}, Value: {}", patch.modNameStr,
             (uintptr_t)cheatAddress, patch.valueStr);
}

static std::vector<int32_t> PatternToByte(const std::string& pattern) {
    std::vector<int32_t> bytes;
    const char* start = pattern.data();
    const char* end = start + pattern.size();

    for (const char* current = start; current < end; ++current) {
        if (*current == '?') {
            ++current;
            if (*current == '?')
                ++current;
            bytes.push_back(-1);
        } else {
            bytes.push_back(strtoul(current, const_cast<char**>(&current), 16));
        }
    }

    return bytes;
}

uintptr_t PatternScan(const std::string& signature) {
    std::vector<int32_t> patternBytes = PatternToByte(signature);
    const auto scanBytes = static_cast<uint8_t*>((void*)g_eboot_address);

    const int32_t* sigPtr = patternBytes.data();
    const size_t sigSize = patternBytes.size();

    uint32_t foundResults = 0;
    for (uint32_t i = 0; i < g_eboot_image_size - sigSize; ++i) {
        bool found = true;
        for (uint32_t j = 0; j < sigSize; ++j) {
            if (scanBytes[i + j] != sigPtr[j] && sigPtr[j] != -1) {
                found = false;
                break;
            }
        }

        if (found) {
            foundResults++;
            return reinterpret_cast<uintptr_t>(&scanBytes[i]);
        }
    }

    return 0;
}

} // namespace MemoryPatcher
