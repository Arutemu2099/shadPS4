// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

// The front's own helpers the architecture-A worker parser (lean_parse.*) must call so it behaves
// exactly as ProcessGraphics (notes/architecture-a-2026-10-10.md sec. 16). Defined in liverpool.cpp
// as thin wrappers over the file-local functions.

#include <array>
#include <atomic>
#include <span>

#include "video_core/amdgpu/pm4_cmds.h"

namespace AmdGpu {
/// SetContextReg bumps the context epoch on a real value change only when this is true.
bool LeanViewportMemoEnabled();
/// A NOP whose body is back work (flip IRQ, guest markers): ProcessGraphics runs it through RunGfxPacket.
bool LeanNopHasBackEffect(const PM4Header* header, bool guest_markers_enabled);
/// After every packet ProcessGraphics calls BumpEpoch(opcode) unless this is true.
bool LeanKeepsSubmitEpoch(PM4ItOpcode opcode);

/// SHADPS4_LEAN: set on the lean worker threads (Rasterizer::ReadMemory / InvalidateMemory count
/// their faults instead of treating them as the front's).
inline thread_local bool t_lean_worker = false;
/// SHADPS4_LEAN: 0 off, 1 L1 shadow compare, 2 L2 (workers parse the head spans for real).
inline int LeanMode() {
    static const int mode = [] {
        const char* v = std::getenv("SHADPS4_LEAN");
        return v ? std::atoi(v) : 0;
    }();
    return mode;
}
/// Set by a fault on this lean worker; the parser stops its span before emitting (FaultFallback).
inline thread_local bool t_lean_fault = false;
struct LeanFaultStats {
    std::atomic<u64> fallback{};            ///< faults on lean workers (must read 0)
    std::array<std::atomic<u64>, 4> first{}; ///< their first addresses
    std::atomic<u32> n_first{};
    /// Pre-check stops: 0 SRT page read-protected, 1 SRT address unmapped, 2 V# page
    /// read-protected, 3 V# address unmapped, 4 walker code not understood, 5 flat buffer overflow.
    std::array<std::atomic<u64>, 8> precheck{};
    /// Shared-lock takes of the pipeline cache by lean workers, and the draws they were for
    /// (watch item: contention with the front's exclusive lock).
    std::atomic<u64> lock_takes{}, lock_draws{};
};
inline LeanFaultStats g_lean_fault_stats;
inline void LeanNoteWorkerFault(u64 addr) {
    auto& s = g_lean_fault_stats;
    s.fallback.fetch_add(1, std::memory_order_relaxed);
    const u32 i = s.n_first.fetch_add(1, std::memory_order_relaxed);
    if (i < s.first.size()) {
        s.first[i].store(addr, std::memory_order_relaxed);
    }
    t_lean_fault = true;
}
} // namespace AmdGpu
