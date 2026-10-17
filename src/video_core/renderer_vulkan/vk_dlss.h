// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <cstdlib>
#include <cstring>
#include "common/types.h"
#include "shader_recompiler/info.h"
#include "video_core/amdgpu/regs.h"

// SHADPS4_DLSS=1 (2026-10-17, LATER-PLAN 2; plan: notes/study-dlss-plan-2026-10-14.md, build D1).
// Second Son's SMAA T2x temporal resolve (fs_a6d23116) reads exactly DLSS's inputs and writes the
// anti-aliased frame the HUD is drawn onto; that one draw per frame becomes a back-side action
// (Rasterizer::DrawClass::DlssResolve), so the worker paths carry it like a resolve or a depth copy
// and nothing is added per draw beyond one address compare in ClassifyDraw.
// Step 1 (plumbing, this file's first version): the action copies the un-anti-aliased tonemap
// output into the resolve's target - the game without temporal AA - to prove the class reaches
// the back in order. The DLSS evaluate replaces that copy in step 2.
namespace Vulkan::Dlss {

constexpr u64 ResolvePsHash = 0xa6d23116; ///< T2x temporal resolve (writes 0x26bd70000)
// Measured in every capture since 09-24 (study sec. 1.1-1.2): fixed guest addresses.
constexpr VAddr ColorInAddr = 0x2f9e10000; ///< tonemap output, RGBA8 1920x1080, not anti-aliased
constexpr VAddr MotionAddr = 0x3059b0000;  ///< RG16F motion vectors (UV, current minus previous)
constexpr VAddr DepthAddr = 0x26b200000;   ///< D32S8, reversed Z

inline bool Enabled() {
    static const bool on = [] {
        const char* v = std::getenv("SHADPS4_DLSS");
        return v && v[0] == '1';
    }();
    return on;
}

/// Guest code address of fs_a6d23116 once a pipeline lookup has met it (0 = not yet / off).
inline std::atomic<u64> g_resolve_ps{0};

/// PipelineCache::GetProgram: remembers where the resolve shader lives.
inline void NoteProgram(bool fragment, u64 hash, VAddr base) {
    if (fragment && hash == ResolvePsHash && Enabled()) [[unlikely]] {
        g_resolve_ps.store(base, std::memory_order_relaxed);
    }
}

/// ClassifyDraw: is this draw's pixel shader the resolve?
inline bool IsResolve(VAddr ps_address) {
    const u64 a = g_resolve_ps.load(std::memory_order_relaxed);
    return a != 0 && a == ps_address;
}

// The frame's jitter phase (study sec. 1.4): the game alternates two quarter-pixel offsets and
// writes the matching T2x subsample index (1 or 2; 0 = T2x off) into the SMAA blend-weights
// draw's 80-byte CB, float 12. That draw runs two draws before the resolve in the same command
// buffer, so the thread parsing it keeps the value and the resolve's record carries it to the back
// (Rasterizer::SplitDraw::dlss_phase): exact per frame, no guess from a frame counter that a
// loading screen could flip.
constexpr u64 WeightsPsHash = 0x961c93ba; ///< SMAA blend weights
inline std::atomic<u64> g_weights_ps{0};
inline std::atomic<u32> g_weights_ud{~0u}; ///< user-data dword of its CB's V# (direct only)
/// The parsing thread's last weights draw: 1 / 2 = subsample index, 3 = T2x off, 0 = unknown.
inline thread_local u8 t_phase = 0;

/// Where the weights program's Info is known: remember its address and its CB's user-data slot.
inline void NoteWeightsInfo(const Shader::Info& info, VAddr base) {
    if (info.pgm_hash != WeightsPsHash || info.stage != Shader::Stage::Fragment || !Enabled())
        [[likely]] {
        return;
    }
    for (const auto& b : info.buffers) {
        if (!b.IsSpecial() && !b.inline_cbuf && b.sharp_idx + 4 <= 16) {
            g_weights_ud.store(b.sharp_idx, std::memory_order_relaxed);
            g_weights_ps.store(base, std::memory_order_relaxed);
            return;
        }
    }
}

/// Next to every ClassifyDraw call that parses the stream in order (front, lean workers, inline).
inline void NoteDraw(const AmdGpu::Regs& regs) {
    const u64 w = g_weights_ps.load(std::memory_order_relaxed);
    if (w == 0 || regs.ps_program.Address<VAddr>() != w) [[likely]] {
        return;
    }
    const u32 ud = g_weights_ud.load(std::memory_order_relaxed);
    if (ud + 4 > 16) {
        return;
    }
    AmdGpu::Buffer vsharp;
    std::memcpy(&vsharp, &regs.ps_program.user_data[ud], sizeof(vsharp));
    if (vsharp.base_address == 0 || vsharp.GetSize() < 52) {
        t_phase = 0;
        return;
    }
    float f;
    std::memcpy(&f, reinterpret_cast<const void*>(vsharp.base_address + 48), sizeof(f));
    t_phase = (f > 0.5f && f < 1.5f) ? 1 : (f > 1.5f && f < 2.5f) ? 2 : (f > -0.5f && f < 0.5f) ? 3 : 0;
}

} // namespace Vulkan::Dlss
