// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <chrono>
#include "common/types.h"

/**
 * Periodic reporting of handled-fault rates against presented frames.
 *
 * Windows dispatches every emulator-handled access violation through the kernel, which is
 * expensive. Two independent sources exist, and they are counted separately because they mean
 * different things:
 *
 *   - read faults  - only ever produced when GPU readbacks are set to Precise, which is the only
 *                    mode that read-protects guest pages holding GPU-modified data. A guest read
 *                    of such a page traps so the emulator can download the data first.
 *   - write faults - guest writes to GPU-tracked pages. Present in every readback mode.
 *   - jit faults   - guest `fs:`-segment accesses patched on demand; unrelated to readbacks.
 *
 * Reporting frames alongside them makes "the fault rate climbs while the framerate falls" a
 * single readable line rather than a correlation across two instruments.
 */
namespace Common::FaultMonitor {

// Incremented on the faulting thread; relaxed ordering, never read there.
extern std::atomic<u64> guest_read_faults;
extern std::atomic<u64> guest_read_unhandled;
extern std::atomic<u64> guest_read_ns;
extern std::atomic<u64> guest_write_faults;
extern std::atomic<u64> guest_write_unhandled;
extern std::atomic<u64> guest_write_ns;
extern std::atomic<u64> jit_patch_faults;
extern std::atomic<u64> jit_patch_ns;
extern std::atomic<u64> presented_frames;

/**
 * Fragmentation of `BufferCache::gpu_modified_ranges`.
 *
 * That interval set is Added to on every written buffer binding (GPU thread, per draw) and
 * Subtracted from on every readback download. A subtract of a small range inside a larger
 * interval splits it in two, so the fragment count can climb without bound while the Adds that
 * would coalesce it get slower. Unlike the fault handlers, this cost is paid on the GPU
 * command-processor thread, outside any handler - which is where the measured per-frame growth
 * has to be.
 */
extern std::atomic<u64> range_fragments;
extern std::atomic<u64> range_adds;
extern std::atomic<u64> range_subtracts;

/**
 * GPU submission and completion-wait accounting.
 *
 * Every quantity that can be counted per frame - faults, range operations, handler time, GPU
 * thread CPU - is flat while wall time per frame doubles, and the guest thread was measured
 * spinning on a GPU fence. That means the growth is latency waiting for the GPU to complete
 * work, so the thing to measure is how long `Scheduler::Wait` actually blocks and how many
 * submits it takes per frame. `BufferCache::ReadMemory` forces a full submit-and-wait per read
 * fault, so submits per frame should be large; if wait time per frame grows while the submit
 * count stays flat, the GPU side is getting slower rather than being asked to do more.
 */
extern std::atomic<u64> gpu_submits;
extern std::atomic<u64> gpu_waits;
extern std::atomic<u64> gpu_wait_ns;

/**
 * Readback download volume, so cutting the submit count can be judged instead of assumed.
 *
 * `BufferCache::ReadMemory` serviced one 8-byte read fault per full submit-and-fence-wait, while
 * the page protection it is unwinding is 4K granular - ~167 read faults/frame produced ~208 GPU
 * submits/frame on this title. Widening each download to an aligned window trades bytes copied
 * for submits issued, and that trade can go the wrong way if the read set is sparse inside the
 * window. These two columns are how to tell which way it went: `download_per_frame` should fall
 * with the submit count, `download_kb_per_frame` should stay near where it was.
 */
extern std::atomic<u64> download_calls;
extern std::atomic<u64> download_bytes;

/**
 * The two places the GPU command-processor thread can park inside the guest command stream.
 *
 * A symbolized stack profile at the framerate floor put that thread 93.7% blocked in a system DLL
 * with the readback path accounting for only ~9-13%, and `Liverpool::Process`'s inner loop does no
 * blocking of its own - so the wait is inside a resumed task. `WaitRegMem` spins on a guest memory
 * value (`while (!Test()) YIELD;`) and the video-out label wait blocks on a condition variable.
 * `ReleaseMem`, which signals the per-frame fence the guest busy-waits on, sits immediately after
 * the ASC `WaitRegMem`. If wait time per frame grows here while the count stays flat, this is the
 * decay; if both are flat, it is not.
 */
extern std::atomic<u64> waitregmem_count;
extern std::atomic<u64> waitregmem_ns;
extern std::atomic<u64> volabel_count;
extern std::atomic<u64> volabel_ns;

/**
 * Time inside `Presenter::Present`, and the flip backlog at the time of each flip.
 *
 * Measured: frame time minus the VO-label wait is a constant ~40 ms across the whole decay, and
 * the wait itself grows 0 -> 106 ms then collapses to 14 ms on a settings toggle. The guest is
 * blocked waiting for a display buffer, which `VideoOutDriver::Flip` releases after
 * `presenter->Present`. This splits the two remaining explanations: presentation genuinely
 * getting slower (present_ms_per_frame grows) versus the flip queue / buffer-label bookkeeping
 * being in a bad state (present stays fast, pending flips climb).
 */
extern std::atomic<u64> present_count;
extern std::atomic<u64> present_ns;
extern std::atomic<u64> flip_pending_total;

/**
 * Breakdown of `Presenter::Present`, which measurement shows IS the decay: frame time minus
 * Present is a constant ~8.5 ms from 19 fps down to 7 fps, while Present itself grows 0.8 -> 139 ms.
 * Only four things in it can block, so time all four:
 *
 *   acquire   - `swapchain.AcquireNextImage()`, i.e. waiting on the display/compositor
 *   flush     - `scheduler.Flush(info)` -> SubmitExecution (which itself takes submit_mutex)
 *   lock      - the bare `std::scoped_lock{Scheduler::submit_mutex}` before presenting. That
 *               mutex is STATIC and every one of the ~208 submits/frame from the readback path
 *               takes it, so the presenter may simply be starved of it.
 *   present   - `swapchain.Present()` -> vkQueuePresentKHR
 */
extern std::atomic<u64> present_acquire_ns;
extern std::atomic<u64> present_flush_ns;
extern std::atomic<u64> present_lock_ns;
extern std::atomic<u64> present_khr_ns;

/**
 * The other end of the same pipeline. `Presenter::Present` runs on the flip thread; the frame is
 * produced on the GPU thread, which first calls `GetRenderFrame()` and blocks there twice:
 *
 *   rf_queue - `free_cv.wait` for a Frame to come back from the flip thread
 *   rf_fence - `waitForFences(frame->present_done)`, i.e. the GPU finishing that frame's work
 *
 * Measured early in a run the frame time is ~38 ms with acquire at ~0, and late in the same run
 * it is ~78 ms with acquire at ~70 - so the block may simply be migrating between the two ends
 * rather than appearing from nowhere. Timing both ends is the only way to tell a growing total
 * stall from a stall that moved. If rf_fence carries the early cost and stays flat while acquire
 * grows, the GPU is not the thing getting slower.
 */
extern std::atomic<u64> renderframe_queue_ns;
extern std::atomic<u64> renderframe_fence_ns;
extern std::atomic<u64> renderframe_count;

/**
 * Time spent inside the handler is measured as well as the number of entries, because the two
 * answer different questions. If the framerate decays while faults-per-frame stays flat, then
 * either each fault got more expensive (handler nanoseconds per fault rises) or the slowdown is
 * not in this path at all (nanoseconds per fault stays flat). Counting alone cannot separate
 * those; this is a two-clock-reads-per-fault cost to be able to.
 */
using Clock = std::chrono::steady_clock;

inline Clock::time_point Now() {
    return Clock::now();
}

inline u64 SinceNs(Clock::time_point start) {
    return static_cast<u64>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count());
}

/**
 * Records which guest page a read fault hit.
 *
 * The guest thread was measured spinning on a GPU fence label in guest memory. Under Precise that
 * page is read-protected, so every poll of the label should trap and force a full download. If
 * that is what is happening, read faults will be concentrated on a very small number of pages
 * rather than spread over the address space - and a targeted fix becomes possible. Cheap enough
 * to leave on: read faults run ~1100/s, and this is a hash insert under a mutex.
 */
void RecordReadFaultPage(u64 addr);

inline void CountGuestReadFault(bool handled, u64 ns, u64 addr) {
    guest_read_faults.fetch_add(1, std::memory_order_relaxed);
    guest_read_ns.fetch_add(ns, std::memory_order_relaxed);
    if (!handled) {
        guest_read_unhandled.fetch_add(1, std::memory_order_relaxed);
    }
    RecordReadFaultPage(addr);
}

inline void CountGuestWriteFault(bool handled, u64 ns) {
    guest_write_faults.fetch_add(1, std::memory_order_relaxed);
    guest_write_ns.fetch_add(ns, std::memory_order_relaxed);
    if (!handled) {
        guest_write_unhandled.fetch_add(1, std::memory_order_relaxed);
    }
}

inline void CountJitPatchFault(u64 ns) {
    jit_patch_faults.fetch_add(1, std::memory_order_relaxed);
    jit_patch_ns.fetch_add(ns, std::memory_order_relaxed);
}

inline void CountPresentedFrame() {
    presented_frames.fetch_add(1, std::memory_order_relaxed);
}

/// `fragments` is the interval count after the operation. Cheap: interval_set::iterative_size()
/// is the underlying node count, O(1).
inline void CountRangeAdd(u64 fragments) {
    range_adds.fetch_add(1, std::memory_order_relaxed);
    range_fragments.store(fragments, std::memory_order_relaxed);
}

inline void CountGpuSubmit() {
    gpu_submits.fetch_add(1, std::memory_order_relaxed);
}

inline void CountWaitRegMem(u64 ns) {
    waitregmem_count.fetch_add(1, std::memory_order_relaxed);
    waitregmem_ns.fetch_add(ns, std::memory_order_relaxed);
}

inline void CountPresentPhases(u64 acquire, u64 flush, u64 lock, u64 khr) {
    present_acquire_ns.fetch_add(acquire, std::memory_order_relaxed);
    present_flush_ns.fetch_add(flush, std::memory_order_relaxed);
    present_lock_ns.fetch_add(lock, std::memory_order_relaxed);
    present_khr_ns.fetch_add(khr, std::memory_order_relaxed);
}

inline void CountRenderFrame(u64 queue_ns, u64 fence_ns) {
    renderframe_count.fetch_add(1, std::memory_order_relaxed);
    renderframe_queue_ns.fetch_add(queue_ns, std::memory_order_relaxed);
    renderframe_fence_ns.fetch_add(fence_ns, std::memory_order_relaxed);
}

inline void CountPresent(u64 ns, u64 pending) {
    present_count.fetch_add(1, std::memory_order_relaxed);
    present_ns.fetch_add(ns, std::memory_order_relaxed);
    flip_pending_total.fetch_add(pending, std::memory_order_relaxed);
}

inline void CountVoLabelWait(u64 ns) {
    volabel_count.fetch_add(1, std::memory_order_relaxed);
    volabel_ns.fetch_add(ns, std::memory_order_relaxed);
}

/// `bytes` is the size actually staged for copy, i.e. after clipping to what is GPU-modified.
inline void CountReadbackDownload(u64 bytes) {
    download_calls.fetch_add(1, std::memory_order_relaxed);
    download_bytes.fetch_add(bytes, std::memory_order_relaxed);
}

inline void CountGpuWait(u64 ns) {
    gpu_waits.fetch_add(1, std::memory_order_relaxed);
    gpu_wait_ns.fetch_add(ns, std::memory_order_relaxed);
}

inline void CountRangeSubtract(u64 fragments) {
    range_subtracts.fetch_add(1, std::memory_order_relaxed);
    range_fragments.store(fragments, std::memory_order_relaxed);
}

/// Starts the reporter thread. Safe to call more than once; only the first call has an effect.
/// Writes one line every interval to the log and to a CSV in the log directory.
void Start();

} // namespace Common::FaultMonitor
