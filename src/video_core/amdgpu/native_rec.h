// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <cstdlib>
#include "common/types.h"

// SHADPS4_NATIVE_REC (2026-10-09 evening, notes/native-path-2026-10-09.md sec. 4): the game's job
// threads write one record per DrawIndex at the HLE call, and group consecutive draws into runs
// (same shader hashes, only whitelisted packets between: user-data SET_SH_REG, plain NOPs,
// NUM_INSTANCES, the same INDEX_TYPE / primitive type, the no-op L2-prefetch DMA). The record
// index rides in the DrawIndex trailing NOP. =1: the front verifies every record against its own
// state (no behaviour change).
namespace AmdGpu::NativeRec {

constexpr u32 Magic = 0x4e415431u; // 'NAT1' in the trailing NOP's data_block[0]
constexpr u32 RingBits = 14;
constexpr u32 RingSize = 1u << RingBits;
constexpr u32 MaxSlots = 16;
constexpr u32 MaxRun = 4096;

struct Record {
    std::atomic<u32> stamp; ///< idx + 1 once written (seqlock: 0 while being written)
    u32 run_k;              ///< position in the run, 0 = the run's head
    u32 tag;                ///< (slot << 28) | shader-hash run id
    u32 head_idx;           ///< ring index of the run's head
    const u32* cmd;         ///< this draw's DRAW_INDEX_2 packet
    u64 index_addr;
    u32 index_count;
    u32 instances;          ///< NUM_INSTANCES (valid when inst_known)
    u32 inst_known;
    u32 vs_known;           ///< bits of ud_vs the job thread's mirror knows
    u32 ps_known;
    u32 gap_dwords;         ///< dwords walked before this draw (0 for a head after a break)
    std::array<u32, 16> ud_vs;
    std::array<u32, 16> ud_ps;
    // Head only, updated by every draw the run takes (final once the command buffer is
    // submitted: the game submits after its jobs finished).
    u32 count;
    const u32* end; ///< dword after the run's last trailing NOP
};

inline std::array<std::atomic<Record*>, MaxSlots> g_rings{};

inline const int g_mode = [] {
    const char* v = std::getenv("SHADPS4_NATIVE_REC");
    return v ? std::atoi(v) : 0;
}();

} // namespace AmdGpu::NativeRec
