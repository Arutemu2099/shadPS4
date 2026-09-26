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
 * SHADPS4_CP_SPLIT=1 (depth 0) or 2 (run-ahead): two threads.
 *  - The BACK is the existing CP thread (Liverpool::process_thread). It keeps every identity the
 *    rest of the emulator relies on: the name "shadPS4:GpuCommandProcessor" (the stack sampler
 *    targets it), Liverpool::gpu_id (IsGpuThread, SendCommand runs inline and the guest command
 *    queue runs here), FaultMonitor::t_is_cp_thread, the CP core placement, the scheduler's single
 *    producer. It owns the buffer/texture caches, the scheduler, IrqC signals and every CP-side
 *    guest-memory write, and a mirror of the register file (BackMirror) that it keeps equal to the
 *    front's by applying the front's register journal.
 *  - The FRONT is a new thread "shadPS4:GpuCommandFront" (t_is_front). It runs the PM4 coroutines
 *    (DCB, CCB, ASC), owns Liverpool::regs, FilterDraw's classifier and the pipeline cache lookup,
 *    and appends records to an ordered SPSC ring. It never touches a cache, the scheduler or guest
 *    memory writes, and never sets gpu_id or t_is_cp_thread.
 *
 * Depth 0 (=1): the front publishes and waits for the back after every record that is not pure
 * register state (Sync). Register records ride along with the next synchronous record. So every
 * guest-visible effect happens in the same order and at the same PM4 position as today; only the
 * thread differs.
 *
 * Run-ahead (=2): graphics-queue draw records are posted (PostRunAhead) instead of synced, and the
 * front keeps parsing while the back executes them, bounded by SHADPS4_CP_SPLIT_DEPTH outstanding
 * posted records (default 64). Everything else still syncs, as at depth 0 (a Sync waits for the
 * back to finish EVERYTHING published, so it is a full drain). With SHADPS4_CP_SPLIT_POST_DMA=1,
 * gfx DMA_DATA packets are posted too (see Liverpool::PostGfxPacket for the guards).
 * Drain points (Sync with a DrainReason) are where the front reads what the back produces:
 * WAIT_REG_MEM, COND_EXEC, MEM_SEMAPHORE waits, the VO label wait, task done (num_submits is the
 * guest's "GPU idle"), the idle tail, every Call record, and read faults taken on the front
 * (Liverpool::DrainForFrontFault). Draw records own everything the back reads besides the mirror:
 * args, class, pipeline pointer (stable: only ReplaceShader erases), ctx epoch, per-stage user
 * data / flat buffer / pgm_base copies; the Info pointer is used for its immutable part only.
 */
namespace AmdGpu::CpSplit {

/// SHADPS4_CP_SPLIT: 0 = off (default), 1 = split at depth 0, 2 = run-ahead. Read once.
int Mode();
inline bool Enabled() {
    return Mode() > 0;
}
inline bool RunAhead() {
    return Mode() >= 2;
}
/// Largest SHADPS4_CP_SPLIT_DEPTH accepted (the front keeps the seq of this many posted records).
constexpr u32 MaxDepth = 1024;
/// SHADPS4_CP_SPLIT_DEPTH: most posted records the back may owe the front (run-ahead only).
/// Default 64, clamped to [1, MaxDepth].
u32 Depth();
/// SHADPS4_CP_SPLIT_POST_DMA=1: run-ahead also posts gfx DMA_DATA (default: they sync).
bool PostDma();
/// SHADPS4_CP_SPLIT_VERIFY=1. Depth 0: at every draw/dispatch record (the front is parked) the
/// back compares its whole register mirror, CB/DB extents, context epoch and cs snapshot with the
/// front's, and each copied stage (user data, flat buffer, pgm_base) with the Info the front's
/// lookup just refreshed. Run-ahead: each posted draw carries a copy of the register windows the
/// back reads (gfx SH, context, uconfig up to the VGT draw state) plus the extents, and the back
/// memcmps its mirror against it; every VerifyEvery()-th draw is synced and gets the full depth-0
/// check. The first mismatch is logged with the word offset and asserts.
bool Verify();
/// SHADPS4_CP_SPLIT_VERIFY_EVERY: run-ahead VERIFY syncs one draw in this many (default 256).
u32 VerifyEvery();
/// SHADPS4_CP_SPLIT_DONE_BATCH (default 1, max 1024): the back publishes done_seq every N records,
/// and always when it has consumed everything it has seen published.
u32 DoneBatch();

/// True on the front thread (set by Liverpool::FrontProcess). The fault handlers use it to drain
/// before a read fault downloads GPU data (inventory 4.6).
inline thread_local bool t_is_front = false;

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
    Call,       ///< CallPayload: a function on the front's stack. Always synchronous.
};

/// Flags for Packet records.
enum RecordFlags : u16 {
    GuestMarkers = 1 << 0,
};

/// Why the front waited for the back to finish everything published (counters only).
enum class DrainReason : u8 {
    DepthZero,       ///< Depth 0: every draw is synchronous.
    VerifyFull,      ///< Run-ahead VERIFY: the periodic synchronous draw.
    Dispatch,        ///< Gfx-queue dispatch record.
    PacketDma,       ///< Gfx DMA_DATA (run-ahead without POST_DMA).
    PacketDmaGuard,  ///< POST_DMA: this DMA writes into the rest of the command buffer.
    PacketFence,     ///< EVENT_WRITE_EOP / EOS.
    PacketWriteData, ///< Gfx WRITE_DATA.
    PacketOther,     ///< Flip / guest-marker NOP, ZPASS_DONE, PFP_SYNC_ME.
    Call,            ///< ASC work, DumpConstRam (Call records).
    IdleTail,        ///< The submission loop's idle tail (Call).
    WaitRegMem,      ///< Gfx/ASC WAIT_REG_MEM (and the VO label wait).
    CondExec,
    MemSemaphore,
    TaskDone,        ///< Before --num_submits.
    IbStart,         ///< POST_DMA: an INDIRECT_BUFFER while posted packets are outstanding.
    FrontFault,      ///< A fault taken on the front (DrainForFrontFault).
    FrontExit,
    Count,
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
 * the back waiting for work and the front waiting for the back (Sync, the depth bound).
 * Waits spin briefly with _mm_pause, then park on std::atomic::wait (WaitOnAddress).
 *
 * A record is Reserve()d, filled, then Commit()ted; only committed records are ever published,
 * so a Sync from a fault handler that interrupted the front between Reserve and Commit (the
 * verbatim DCB copy in RunGfxPacket is the one guest-memory read in that window) publishes and
 * waits for everything but the open record.
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
    /// uninitialised). Invisible to the back until committed and published. Waits while the ring
    /// is full.
    void* Reserve(RecordType type, u32 payload_bytes, u16 flags = 0);
    template <typename T>
    T* Reserve(RecordType type, u32 payload_bytes = sizeof(T), u16 flags = 0) {
        static_assert(alignof(T) <= RecordAlign);
        return static_cast<T*>(Reserve(type, payload_bytes, flags));
    }
    /// Ends the record reserved last. Does not publish: it rides along with the next Post/Sync.
    void Commit();
    /// Publishes everything committed and waits until the back has executed all of it.
    void Sync(DrainReason reason);
    /// Run-ahead: publishes everything committed without waiting, then waits only while more
    /// than Depth() posted records are outstanding. `is_draw` selects the counter.
    void PostRunAhead(bool is_draw);
    /// Sync(reason) if a posted non-draw record (POST_DMA) may not have executed yet.
    void SyncIfPostedPackets(DrainReason reason);

    void EmitRegWrite(u32 word_offset, const u32* data, u32 count);
    void EmitClearState();
    void EmitExtent(u32 slot, u32 raw);
    void EmitEpochBump(u32 opcode);

    /// Runs `fn` on the back while the front waits. `fn` lives on the front's stack, which is
    /// sound because the Sync waits for it at any depth: the front is parked until it returns.
    template <typename F>
    void CallSync(F& fn, DrainReason reason) {
        auto* call = Reserve<CallPayload>(RecordType::Call);
        call->invoke = [](void* context) { (*static_cast<F*>(context))(); };
        call->context = static_cast<void*>(&fn);
        Commit();
        Sync(reason);
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

    static constexpr u32 NumDrainReasons = static_cast<u32>(DrainReason::Count);
    /// Front-owned counters, logged and reset every ~10 s by MaybeLog. Times are in TSC ticks.
    struct Counters {
        u64 records{}, bytes{}, full_waits{}, front_parks{};
        u64 posted_draws{}, posted_packets{}, depth_waits{}, depth_wait_ticks{};
        u64 max_lead{};       ///< Most posted records outstanding at a post (after the bound).
        u64 max_lead_seq{};   ///< Most records (any type) published but not done, at a post.
        std::array<u64, NumDrainReasons> drains{};
        std::array<u64, NumDrainReasons> drain_waits{}; ///< Drains that found the back busy.
        std::array<u64, NumDrainReasons> drain_ticks{};
    };
    /// Front only. Checks the clock every 1024 calls; logs and resets every ~10 s.
    void MaybeLog();

private:
    void Publish();
    void WaitForSpace(u64 end);
    /// Waits until done_seq >= seq. Returns the ticks spent waiting (0 if it was already done).
    u64 WaitDone(u64 seq);
    bool BackHasWork(const std::atomic<u32>& num_commands) const;
    void PublishDone(u64 seq);

    std::unique_ptr<u8[]> storage_owner;
    u8* storage{};
    u32 capacity{};
    u64 mask{};

    // Producer side (front). Written only by the front.
    alignas(64) std::atomic<u64> write_pos{0}; ///< Published end, seq_cst store.
    u64 reserve_pos{0};
    u64 committed_pos{0};
    u64 committed_seq{0};
    u64 cached_read{0};
    u64 emitted{0};
    bool record_open{false};
    // Run-ahead bound: seqs of the last MaxDepth + 1 posted records, oldest not known done first.
    u32 depth{1};
    std::unique_ptr<u64[]> posted_seqs;
    u64 posts{0};        ///< Records posted so far.
    u64 oldest_post{0};  ///< Index of the oldest post not known to be done.
    u64 last_posted_packet_seq{0};
    Counters counters{};
    u64 log_window_start_ns{0};
    u64 log_window_start_ticks{0};
    u32 log_calls{0};

    // Consumer side (back).
    alignas(64) std::atomic<u64> read_pos{0};
    u64 read{0};
    u64 cached_write{0};
    u64 pending_done{0}; ///< Newest executed seq not yet published in done_seq.
    u32 done_batch{1};
    u32 done_count{0};
    std::atomic<u64> back_idle_ticks{0}; ///< Written by the back, read and reset by MaybeLog.

    alignas(64) std::atomic<u64> done_seq{0};
    alignas(64) std::atomic<bool> front_parked{false};
    alignas(64) std::atomic<bool> back_parked{false};
    std::atomic<u32> doorbell{0};
    std::atomic<bool> front_exited{false};
};

} // namespace AmdGpu::CpSplit
