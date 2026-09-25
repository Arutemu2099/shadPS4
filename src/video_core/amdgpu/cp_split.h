// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <cstring>
#include <memory>
#include <stop_token>

#include "common/assert.h"
#include "common/types.h"
#include "video_core/amdgpu/cb_db_extent.h"
#include "video_core/amdgpu/regs.h"

/**
 * Command-processor front/back split (notes/cp-split-plan-2026-09-25.md,
 * notes/cp-split-inventory-2026-09-25.md). DRAFT, default off.
 *
 * SHADPS4_CP_SPLIT unset or 0: none of this is constructed and the command processor runs as
 * before, on one thread.
 *
 * SHADPS4_CP_SPLIT=1: two threads.
 *  - The BACK is the existing CP thread (Liverpool::process_thread). It keeps every identity the
 *    rest of the emulator relies on: the name "shadPS4:GpuCommandProcessor" (the stack sampler
 *    targets it), Liverpool::gpu_id (IsGpuThread, SendCommand runs inline and the guest command
 *    queue runs here), FaultMonitor::t_is_cp_thread, the CP core placement, the scheduler's single
 *    producer. It owns the buffer/texture caches, the scheduler, IrqC signals and every CP-side
 *    guest-memory write, and a mirror of the register file (BackMirror) that it keeps equal to the
 *    front's by applying the front's register journal.
 *  - The FRONT is a new thread "shadPS4:GpuCommandFront". It runs the PM4 coroutines (DCB, CCB,
 *    ASC), owns Liverpool::regs, FilterDraw's classifier and the pipeline cache lookup, and
 *    appends records to an ordered SPSC ring. It never touches a cache, the scheduler or guest
 *    memory writes, and never sets gpu_id or t_is_cp_thread.
 *
 * Depth 0 (this draft): the front publishes and waits for the back after every record that is
 * not pure register state (Commit(true)). Register records ride along with the next synchronous
 * record. So every guest-visible effect happens in the same order and at the same PM4 position
 * as today; only the thread differs.
 *
 * SPLIT-TODO(depth>0): values >= 2 are reserved for "run ahead up to n-1 draws". What must change
 * then: Commit(true) for draws/dispatches stops waiting (bounded by depth), the drain points in
 * Liverpool (WaitRegMem, CondExec, MemSemaphore wait, VO label wait, task done) already call
 * Drain(); Call records (depth-0-only: they point at the front's stack) must be replaced by owned
 * records (ASC dispatch/DMA/fences, DumpConstRam with a heap-slice copy); front read faults must
 * drain before ReadMemory; SplitHazard/g_cp_wait diagnostics must be noted at parse time.
 */
namespace AmdGpu::CpSplit {

/// SHADPS4_CP_SPLIT: 0 = off (default), 1 = split at depth 0. Read once.
int Mode();
inline bool Enabled() {
    return Mode() > 0;
}
/// SHADPS4_CP_SPLIT_VERIFY=1: at every draw/dispatch record (the front is parked at depth 0) the
/// back compares its whole register mirror, CB/DB extents, context epoch and cs snapshot with the
/// front's, and each copied stage (user data, flat buffer, pgm_base) with the Info the front's
/// lookup just refreshed. The first mismatch is logged with the word offset and asserts.
bool Verify();

/// The back's copy of the front state it reads (Liverpool::BackRegs() and friends).
struct BackMirror {
    Regs regs{};
    std::array<CbDbExtent, NUM_COLOR_BUFFERS> cb_extent{};
    CbDbExtent db_extent{};
    u64 ctx_epoch{};
};

enum class RecordType : u16 {
    Wrap,       ///< Filler to the end of the ring; the next record starts at offset 0.
    RegWrite,   ///< RegWritePayload + count dwords, copied into the back's mirror.
    ClearState, ///< mirror.regs.SetDefaults().
    Extent,     ///< ExtentPayload.
    EpochBump,  ///< EpochPayload: VideoCore::BumpSubmitEpoch in stream order.
    Draw,       ///< Liverpool's draw record + per-stage data (defined in liverpool.cpp).
    Dispatch,   ///< Liverpool's dispatch record + one stage (defined in liverpool.cpp).
    Packet,     ///< A verbatim copy of one gfx PM4 type-3 packet, run by ExecuteGfxPacket.
    Call,       ///< CallPayload: a function on the front's stack. Depth 0 only.
};

/// Flags for Packet records.
enum RecordFlags : u16 {
    GuestMarkers = 1 << 0,
};

struct RecordHeader {
    RecordType type;
    u16 flags;
    u32 size; ///< Header included, multiple of RecordAlign.
    u64 seq;  ///< 1-based record number (Wrap: 0).
};
static_assert(sizeof(RecordHeader) == 16);

constexpr u32 RecordAlign = 16;

struct RegWritePayload {
    u32 word_offset;
    u32 count;
    // u32 data[count] follows
};
struct ExtentPayload {
    u32 slot; ///< 0-7 = CB, 8 = DB.
    u32 raw;
};
struct EpochPayload {
    u32 opcode;
    u32 pad;
};
struct CallPayload {
    void (*invoke)(void*);
    void* context;
};

/// Per-stage user data of one draw/dispatch, copied on the front right after the lookup
/// (inventory section 4.4). Followed by ud_dw user-data dwords, then flat_dw flat-buffer dwords;
/// the whole block is padded to 8 bytes.
struct StagePayload {
    const void* info; ///< const Shader::Info*: only its immutable part is read by the back.
    VAddr pgm_base;
    u32 logical_stage;
    u32 ud_dw;
    u32 flat_dw;
    u32 pad;
};
static_assert(sizeof(StagePayload) % 8 == 0);

/// Draw arguments the back needs besides the registers (the register-side ones - num_indices,
/// num_instances, index base/type - travel in the journal).
struct DrawArgs {
    bool is_indexed{};
    bool is_indirect{};
    u32 index_offset{};
    VAddr arg_address{};
    u32 offset{};
    u32 stride{};
    u32 max_count{};
    VAddr count_address{};
};

/**
 * Single-producer (front) / single-consumer (back) byte ring plus the two wait points:
 * the back waiting for work and the front waiting for the back (Commit(true), Drain).
 * Waits spin briefly with _mm_pause, then park on std::atomic::wait (WaitOnAddress).
 */
class Channel {
public:
    static constexpr u32 DefaultRingBytes = 4u << 20;

    explicit Channel(u32 ring_bytes = DefaultRingBytes);
    ~Channel();

    Channel(const Channel&) = delete;
    Channel& operator=(const Channel&) = delete;

    // ---- Front (the only producer) ----

    /// Reserves one record with `payload_bytes` of payload and returns the payload (16-aligned,
    /// uninitialised). Invisible to the back until published. Waits while the ring is full.
    void* Reserve(RecordType type, u32 payload_bytes, u16 flags = 0);
    template <typename T>
    T* Reserve(RecordType type, u32 payload_bytes = sizeof(T), u16 flags = 0) {
        static_assert(alignof(T) <= RecordAlign);
        return static_cast<T*>(Reserve(type, payload_bytes, flags));
    }
    /// Ends the record reserved last. sync (depth 0 for anything but register state): publish
    /// everything and wait until the back has executed it. Otherwise it rides along.
    void Commit(bool sync);
    /// Publishes everything and waits until the back has executed all of it.
    void Drain();

    void EmitRegWrite(u32 word_offset, const u32* data, u32 count);
    void EmitClearState();
    void EmitExtent(u32 slot, u32 raw);
    void EmitEpochBump(u32 opcode);

    /// Runs `fn` on the back while the front waits. `fn` lives on the front's stack, so this is
    /// only sound at depth 0. SPLIT-TODO(depth>0): replace every use with an owned record.
    template <typename F>
    void CallSync(F& fn) {
        auto* call = Reserve<CallPayload>(RecordType::Call);
        call->invoke = [](void* context) { (*static_cast<F*>(context))(); };
        call->context = static_cast<void*>(&fn);
        Commit(true);
    }

    /// The front thread is about to return; lets the back leave its loop.
    void FrontExited();

    // ---- Back (the only consumer) ----

    /// The next published record, or null. Stays valid until Done(it).
    const RecordHeader* Next();
    /// Releases the record returned by Next() and reports it executed.
    void Done(const RecordHeader* record);
    /// Returns when a record is published, `num_commands` is non-zero, `stoken` is stopped or the
    /// front has exited (or spuriously).
    void WaitForWork(const std::atomic<u32>& num_commands, const std::stop_token& stoken);
    bool FrontHasExited() const {
        return front_exited.load(std::memory_order_acquire);
    }

    // ---- Any thread ----

    /// Wakes the back if it is parked (SendCommand, stop requests).
    void WakeBackIfParked();
    void WakeBack();

    struct Counters {
        u64 records{}, syncs{}, bytes{}, front_parks{}, full_waits{};
    };
    /// Front only. Logs and resets every ~10 s of records (SPLIT-TODO: per-type counts).
    void MaybeLog();

private:
    void Publish();
    void WaitForSpace(u64 end);
    void WaitDone(u64 seq);
    bool BackHasWork(const std::atomic<u32>& num_commands) const;

    std::unique_ptr<u8[]> storage_owner;
    u8* storage{};
    u32 capacity{};
    u64 mask{};

    // Producer side (front). Written only by the front.
    alignas(64) std::atomic<u64> write_pos{0}; ///< Published end, seq_cst store.
    u64 reserve_pos{0};
    u64 cached_read{0};
    u64 emitted{0};
    Counters counters{};
    u64 log_window_start_ns{0};

    // Consumer side (back).
    alignas(64) std::atomic<u64> read_pos{0};
    u64 read{0};
    u64 cached_write{0};

    alignas(64) std::atomic<u64> done_seq{0};
    alignas(64) std::atomic<bool> front_parked{false};
    alignas(64) std::atomic<bool> back_parked{false};
    std::atomic<u32> doorbell{0};
    std::atomic<bool> front_exited{false};
};

} // namespace AmdGpu::CpSplit
