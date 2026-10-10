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
    // Opus review 7: written by later draws after the head's stamp was published.
    std::atomic<u32> count;
    std::atomic<const u32*> end; ///< dword after the run's last trailing NOP
    // Stage a2: what the job thread derived itself through the program table (0: not derived).
    u32 derived;
    u32 flat_vs_dw;
    u32 flat_ps_dw;
    u32 vi_bytes;
    u64 payload_pos;      ///< byte position in the thread's payload ring: flat VS, flat PS, vi
    const void* pipeline; ///< the Vulkan::GraphicsPipeline the derivation used
    u64 fetch_addr;
    u32 vs_pgm; ///< the bound VS / PS program address >> 8 (the game binds per-object copies)
    u32 ps_pgm;
    // SHADPS4_NATIVE_GPU_DIRECT (sec. 10): 1 = the GPU record of this draw is written (g_gpu_recs
    // at the same ring index); the run's ranges live in g_unions[union_idx & UnionMask]; the
    // values of the fetch shader's vertex / instance offset SGPRs (0 when none); per fetch
    // attribute, the stream's guest address (0: null stream).
    u32 gpu_ok;
    u32 union_idx;
    u32 vo_ud;
    u32 io_ud;
    u32 vcount;
    std::array<u64, 16> vaddr;
};

// SHADPS4_NATIVE_GPU_DIRECT (notes/native-path-2026-10-09.md sec. 10): the job thread writes,
// per derived draw, the record the native shaders read (Shader::NativeRecordDwords dwords: user
// data in push layout, each Guest binding's guest address (low 32 bits) and size, each pulled
// vertex location's guest address (low 32 bits)); the GPU subtracts a per-batch base. And per
// run, the union of every binding's range over the run's draws so far (count published last).
inline const int g_gpu_mode = [] {
    const char* v = std::getenv("SHADPS4_NATIVE_GPU_DIRECT");
    return v ? std::atoi(v) : 0;
}();
constexpr u32 GpuRecDwords = 128;
inline std::array<std::atomic<u32*>, MaxSlots> g_gpu_recs{};
constexpr u32 UnionBits = 12;
constexpr u32 UnionMask = (1u << UnionBits) - 1;
constexpr u32 UnionSlots = 40; // Shader::NUM_BUFFERS
struct RunUnion {
    std::atomic<u32> stamp; ///< head ring index + 1 (0 while being started)
    std::atomic<u32> n;     ///< draws of the run the ranges cover (published after them)
    u64 guest_mask;         ///< Guest binding slots
    u64 eq_mask;            ///< slots whose (address, size) was the same for every draw
    std::array<u64, UnionSlots> lo, hi, first_addr;
    std::array<u32, UnionSlots> first_size;
    u32 vcount;
    std::array<u64, 16> vlo, vhi;
    u64 ilo, ihi;
};
inline std::array<std::atomic<RunUnion*>, MaxSlots> g_unions{};

constexpr u32 PayloadBits = 23; // 8 MB per thread
constexpr u64 PayloadSize = u64{1} << PayloadBits;
inline std::array<std::atomic<u8*>, MaxSlots> g_payloads{};
/// Opus review 2: per slot, the payload position the job thread has claimed up to (published
/// before it writes there). A reader's bytes at [pos, ...) are intact iff claim - pos <= size.
inline std::array<std::atomic<u64>, MaxSlots> g_payload_claim{};
/// Slot 15 is shared by every thread past the 14th: no records there (review 6).
constexpr u32 MaxRecordSlot = 14;

/// Program table: (VS hash, PS hash, fetch shader address) -> the pipeline the front looked up
/// for such a draw. Written by the front only, read lock-free by the job threads.
struct ProgramEntry {
    std::atomic<u64> key{};
    u64 vs_hash{}, ps_hash{}, fetch_addr{};
    std::atomic<const void*> pipeline{};
};
constexpr u32 TableSize = 8192;
inline std::array<ProgramEntry, TableSize> g_table{};
/// VS hash -> fetch_shader_sgpr_base + 1 (0: unknown), 0xffffffff: no fetch shader.
struct VsEntry {
    std::atomic<u64> vs_hash{};
    std::atomic<u32> fetch_base{};
};
inline std::array<VsEntry, TableSize> g_vs_table{};

inline u64 TableKey(u64 vs, u64 ps, u64 fetch) {
    const u64 k = vs * 0x9E3779B97F4A7C15ull ^ (ps + 0x632BE59BD9B4E019ull) * 0xC2B2AE3D27D4EB4Full ^
                  (fetch + 1) * 0x165667B19E3779F9ull;
    return k ? k : 1;
}

inline const void* TableFind(u64 vs, u64 ps, u64 fetch) {
    const u64 k = TableKey(vs, ps, fetch);
    for (u32 i = 0; i < 16; ++i) {
        const auto& e = g_table[(k + i) & (TableSize - 1)];
        const u64 ek = e.key.load(std::memory_order_acquire);
        if (ek == 0) {
            return nullptr;
        }
        if (ek == k && e.vs_hash == vs && e.ps_hash == ps && e.fetch_addr == fetch) {
            return e.pipeline.load(std::memory_order_acquire);
        }
    }
    return nullptr;
}

/// Front thread only.
inline void TablePublish(u64 vs, u64 ps, u64 fetch, const void* pipeline) {
    const u64 k = TableKey(vs, ps, fetch);
    for (u32 i = 0; i < 16; ++i) {
        auto& e = g_table[(k + i) & (TableSize - 1)];
        const u64 ek = e.key.load(std::memory_order_relaxed);
        if (ek == k && e.vs_hash == vs && e.ps_hash == ps && e.fetch_addr == fetch) {
            if (e.pipeline.load(std::memory_order_relaxed) != pipeline) {
                e.pipeline.store(pipeline, std::memory_order_release);
            }
            return;
        }
        if (ek == 0) {
            e.vs_hash = vs;
            e.ps_hash = ps;
            e.fetch_addr = fetch;
            e.pipeline.store(pipeline, std::memory_order_relaxed);
            e.key.store(k, std::memory_order_release);
            return;
        }
    }
}

inline u32 VsFetchBase(u64 vs) {
    const u64 k = vs ? vs : 1;
    for (u32 i = 0; i < 16; ++i) {
        const auto& e = g_vs_table[(k * 0x9E3779B97F4A7C15ull + i) & (TableSize - 1)];
        const u64 h = e.vs_hash.load(std::memory_order_acquire);
        if (h == 0) {
            return 0;
        }
        if (h == k) {
            return e.fetch_base.load(std::memory_order_relaxed);
        }
    }
    return 0;
}

/// Front thread only. fetch_base: fetch_shader_sgpr_base + 1, or 0xffffffff for none.
inline void VsPublish(u64 vs, u32 fetch_base) {
    const u64 k = vs ? vs : 1;
    for (u32 i = 0; i < 16; ++i) {
        auto& e = g_vs_table[(k * 0x9E3779B97F4A7C15ull + i) & (TableSize - 1)];
        const u64 h = e.vs_hash.load(std::memory_order_relaxed);
        if (h == k) {
            return;
        }
        if (h == 0) {
            e.fetch_base.store(fetch_base, std::memory_order_relaxed);
            e.vs_hash.store(k, std::memory_order_release);
            return;
        }
    }
}

inline std::array<std::atomic<Record*>, MaxSlots> g_rings{};

inline const int g_mode = [] {
    const char* v = std::getenv("SHADPS4_NATIVE_REC");
    return v ? std::atoi(v) : 0;
}();

} // namespace AmdGpu::NativeRec
