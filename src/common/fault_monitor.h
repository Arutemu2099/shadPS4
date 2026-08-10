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
 * Per-draw CPU cost, measured directly instead of inferred from a stack scan.
 *
 * `thread-stack-profile.ps1` reports INCLUSIVE frequencies from scanning the stack for return
 * addresses, and it has known false positives. In the prologue forest it put
 * `GetGraphicsPipeline` at 36% and `GraphicsPipelineKey::operator==` at 32% - but that key is
 * hashed with XXH3 over a few hundred bytes, which at 7,686 draws/frame is ~0.4 ms of an 88 ms
 * frame. The percentages cannot be cost shares, and an optimisation was nearly built on them.
 *
 * These are real scoped timers on the four things `Rasterizer::Draw` does per draw, plus the
 * texture-cache call inside `BeginRendering` that a stack scan blamed and a fix failed to move.
 * Overhead is 8 clock reads per draw, ~1.5 ms/frame at 7,700 draws - about 2%, which inflates the
 * measured total slightly and is worth remembering when the columns are summed.
 */
extern std::atomic<u64> draw_count;
extern std::atomic<u64> draw_pipeline_ns;   ///< PipelineCache::GetGraphicsPipeline (key rebuild + lookup)
extern std::atomic<u64> draw_bindres_ns;    ///< Rasterizer::BindResources
extern std::atomic<u64> draw_beginrender_ns;///< Rasterizer::BeginRendering (includes find_depth)
extern std::atomic<u64> draw_finddepth_ns;  ///< TextureCache::FindDepthTarget, nested in the above

/// Split of BindResources, which measured 6.8 us of the 8.0 us spent per draw. Accumulated per
/// shader stage, so these sum to more than one call's worth - divide by draws for us/draw.
extern std::atomic<u64> draw_bindbuf_ns;
extern std::atomic<u64> draw_bindtex_ns;

/// Split of BindBuffers, which is 5.00 us/draw - 62% of all per-draw cost. Two passes over the
/// stage's buffer descriptors: the first resolves each sharp to a cached buffer (FindBuffer, a
/// page-table lookup), the second synchronizes it (ObtainBuffer -> SynchronizeBuffer ->
/// ForEachUploadRange, which takes a per-region spinlock and scans a page bitset). A stack scan
/// pointed at the second; that is exactly the kind of claim that has been wrong twice, so measure.
extern std::atomic<u64> bindbuf_find_ns;
extern std::atomic<u64> bindbuf_obtain_ns;

/**
 * Split of the ObtainBuffer pass itself, which measured 4.93 us/draw.
 *
 * `BufferCache::ObtainBuffer` has two halves that share nothing, and only one of them is the page
 * tracker. Small read-only bindings that are not GPU-modified take an early exit that runs
 * `IsRegionGpuModified` and then MEMCPIES the guest data into the stream buffer; everything else
 * falls through to `SynchronizeBuffer` -> `ForEachUploadRange`. A stack scan attributed the whole
 * 4.93 us to the second half. Stack scans on this thread have been wrong by ~65x once already, so
 * the split is measured before anything is cut.
 *
 *   query  - `IsRegionGpuModified` in the early-exit test (locks a region, copies a 1024-bit mask)
 *   stream - `stream_buffer.Copy`, i.e. the guest-memory memcpy on the early-exit path
 *   sync   - `SynchronizeBuffer` on the fall-through path
 */
extern std::atomic<u64> obtain_query_ns;
extern std::atomic<u64> obtain_stream_ns;
extern std::atomic<u64> obtain_stream_calls;
extern std::atomic<u64> obtain_stream_bytes;
extern std::atomic<u64> obtain_sync_ns;
extern std::atomic<u64> obtain_sync_calls;

/**
 * Inside the sync half: the three phases of `MemoryTracker::ForEachUploadRange`, plus the page
 * protection calls it triggers.
 *
 * Measured 2026-08-09: the sync half is 2.86 us per CALL, which is ~9,000 cycles. Nothing in a
 * 128-byte bitset can cost that, so the specified `AnyInRange` early-out would be aimed at the
 * wrong line if shipped on that number alone. The two candidates that CAN cost microseconds are
 * both in here and neither is a bitset op:
 *
 *   scan  - phase 1, the CPU-modified walk (the part the early-out would skip)
 *   copy  - `on_upload()` -> UploadCopies, and in SynchronizeBuffer a non-empty copy list also
 *           forces `scheduler.EndRendering()`, i.e. a render-pass break mid-draw-stream
 *   mark  - phase 2, `ChangeRegionState<GPU, true>` on written bindings, which under Precise
 *           read-protects the pages through `UpdatePageWatchersForRegion` -> `Protect` -> a
 *           VirtualProtect SYSCALL
 *
 * `protect_*` counts how often protection changes actually reach the syscall and what they cost,
 * separated by direction because write-protection happens in every readbacks mode and
 * read-protection only under Precise.
 */
extern std::atomic<u64> upload_scan_ns;
extern std::atomic<u64> upload_copy_ns;
extern std::atomic<u64> upload_mark_ns;
extern std::atomic<u64> upload_written_calls;
extern std::atomic<u64> protect_write_calls;
extern std::atomic<u64> protect_write_ns;
extern std::atomic<u64> protect_read_calls;
extern std::atomic<u64> protect_read_ns;

/// Split of `StreamBuffer::Copy`, the ObtainBuffer early-exit path. Four phases: reserving space
/// in the stream buffer, the `IsValidMapping` guest-address check, the copy itself (which for a
/// guest address goes through `CopySparseMemory` - shared lock plus a VMA tree walk), and Commit.
/**
 * Split of `Rasterizer::BindTextures`, 2.39 us/draw and never examined.
 *
 * MEASURED 2026-08-09, forest, 7,040 draws/frame - and it corrected a wrong guess immediately:
 *
 *   findimage  0.49 us/draw   21%   FindImage + GetAssociatedDepth
 *   view       1.65 us/draw   72%   FindTexture + Transit + layout bookkeeping  <- the cost
 *   sampler    0.07 us/draw    3%   the sampler loop
 *
 * A last-value memo over `GetSampler` was written against an estimate of 0.3-0.6 us/draw for that
 * loop, on the reasoning that it hashes the S# with XXH3 and takes a global mutex once per binding
 * per draw. The real figure is **0.07 us/draw**, over by 5-8x, and the memo measured a 75.6% miss
 * rate. It was reverted unbuilt-on. Shipping this split in the same build is what caught it in one
 * run instead of a playtest cycle.
 *
 * `Image::GetBarriers` inside `Transit` was also the top SELF-time leaf in an old stack profile -
 * the one entry in it that was self time rather than inclusive frequency. Two instruments agree.
 */
extern std::atomic<u64> bindtex_findimage_ns;
extern std::atomic<u64> bindtex_view_ns;
extern std::atomic<u64> bindtex_sampler_ns;

/**
 * The rest of `Rasterizer::Draw` - the third of the frame no timer ever covered.
 *
 * Measured 2026-08-09 in the forest: frame time is 10.68 us/draw while the timed spans
 * (GetGraphicsPipeline 1.03 + BindResources 5.98 + BeginRendering 0.22) account for 7.23. The
 * missing **3.45 us/draw is ~33% of a 76.9 ms frame** and had never been looked at, because the
 * original instrumentation timed "the three things Draw does" and Draw in fact does eleven.
 *
 *   pre       - PopPendingOperations + FilterDraw + PrepareRenderState
 *   vtxbuf    - BindVertexBuffers + BindIndexBuffer. A SECOND buffer-cache pass per draw;
 *               bindbuf_us_each only ever covered the shader-stage buffers.
 *   descwrite - pipeline->BindResources: the descriptor writes (~10 bindings/draw) and the
 *               collected buffer barriers actually reaching Vulkan
 *   dynstate  - UpdateDynamicState
 *   schedbegin- scheduler.BeginRendering (distinct from Rasterizer::BeginRendering, which is
 *               already timed as beginrender_us_each)
 *   emit      - bindPipeline + drawIndexed/draw + ResetBindings
 *
 * Adds ~5 clock reads/draw over the previous instrumentation, ~0.8% at this frame time.
 */
/**
 * The command-processor thread's own frame budget, which decides whether parallel recording is
 * even the right lever.
 *
 * Measured 2026-08-09: `Rasterizer::Draw` accounts for 51.5 ms of a 70.4 ms forest frame. The other
 * 18.9 ms has never been split, and "µs per draw" hides it because that figure is frame time
 * divided by draws - work that is per-FRAME gets silently amortised into a per-draw column.
 *
 *   cp_wait   - the outer condvar wait in `Liverpool::Process`, i.e. the CP thread IDLE waiting for
 *               the guest to submit. This is NOT emulator cost, and it is the number that matters
 *               most: if a large share of the frame is spent here, the guest thread is a co-
 *               bottleneck and parallelising command recording cannot buy the full amount.
 *   cp_active - the inner submit loop: ProcessCommands plus every resumed task
 *   dispatch  - compute dispatches (526 per forest frame) - inside cp_active, alongside Draw
 *
 * cp_active minus draw minus dispatch is PM4 parsing, register writes, WaitRegMem/ReleaseMem and
 * render-pass transitions. Timed per submit batch rather than per draw, so the overhead is
 * negligible.
 */
extern std::atomic<u64> cp_wait_ns;
extern std::atomic<u64> cp_active_ns;
extern std::atomic<u64> dispatch_count;
extern std::atomic<u64> dispatch_ns;

extern std::atomic<u64> draw_pre_ns;
extern std::atomic<u64> draw_vtxbuf_ns;
extern std::atomic<u64> draw_descwrite_ns;
extern std::atomic<u64> draw_dynstate_ns;
extern std::atomic<u64> draw_schedbegin_ns;
extern std::atomic<u64> draw_emit_ns;

extern std::atomic<u64> stream_map_ns;
extern std::atomic<u64> stream_valid_ns;
extern std::atomic<u64> stream_copy_ns;
extern std::atomic<u64> stream_commit_ns;

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

/// Elapsed between two already-taken timestamps, so a split measurement costs one clock read at
/// the boundary instead of two.
inline u64 DiffNs(Clock::time_point start, Clock::time_point end) {
    return static_cast<u64>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
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

inline void CountDraw(u64 pipeline_ns, u64 bindres_ns, u64 beginrender_ns) {
    draw_count.fetch_add(1, std::memory_order_relaxed);
    draw_pipeline_ns.fetch_add(pipeline_ns, std::memory_order_relaxed);
    draw_bindres_ns.fetch_add(bindres_ns, std::memory_order_relaxed);
    draw_beginrender_ns.fetch_add(beginrender_ns, std::memory_order_relaxed);
}

inline void CountFindDepthTarget(u64 ns) {
    draw_finddepth_ns.fetch_add(ns, std::memory_order_relaxed);
}

inline void CountBindSplit(u64 buffers_ns, u64 textures_ns) {
    draw_bindbuf_ns.fetch_add(buffers_ns, std::memory_order_relaxed);
    draw_bindtex_ns.fetch_add(textures_ns, std::memory_order_relaxed);
}

inline void CountBindBufferPass(u64 find_ns, u64 obtain_ns) {
    bindbuf_find_ns.fetch_add(find_ns, std::memory_order_relaxed);
    bindbuf_obtain_ns.fetch_add(obtain_ns, std::memory_order_relaxed);
}

/// Early-exit path of ObtainBuffer: the GPU-modified query, then the guest-memory copy.
inline void CountObtainStream(u64 query_ns, u64 copy_ns, u64 bytes) {
    obtain_query_ns.fetch_add(query_ns, std::memory_order_relaxed);
    obtain_stream_ns.fetch_add(copy_ns, std::memory_order_relaxed);
    obtain_stream_calls.fetch_add(1, std::memory_order_relaxed);
    obtain_stream_bytes.fetch_add(bytes, std::memory_order_relaxed);
}

/// Fall-through path of ObtainBuffer. `query_ns` is 0 when the early-exit test was skipped
/// outright (written or oversized binding), so query time stays comparable across both paths.
inline void CountCpWait(u64 ns) {
    cp_wait_ns.fetch_add(ns, std::memory_order_relaxed);
}

inline void CountCpActive(u64 ns) {
    cp_active_ns.fetch_add(ns, std::memory_order_relaxed);
}

inline void CountDispatch(u64 ns) {
    dispatch_count.fetch_add(1, std::memory_order_relaxed);
    dispatch_ns.fetch_add(ns, std::memory_order_relaxed);
}

inline void CountDrawTail(u64 pre_ns, u64 vtxbuf_ns, u64 descwrite_ns, u64 dynstate_ns,
                          u64 schedbegin_ns, u64 emit_ns) {
    draw_pre_ns.fetch_add(pre_ns, std::memory_order_relaxed);
    draw_vtxbuf_ns.fetch_add(vtxbuf_ns, std::memory_order_relaxed);
    draw_descwrite_ns.fetch_add(descwrite_ns, std::memory_order_relaxed);
    draw_dynstate_ns.fetch_add(dynstate_ns, std::memory_order_relaxed);
    draw_schedbegin_ns.fetch_add(schedbegin_ns, std::memory_order_relaxed);
    draw_emit_ns.fetch_add(emit_ns, std::memory_order_relaxed);
}

inline void CountBindTexSplit(u64 findimage_ns, u64 view_ns, u64 sampler_ns) {
    bindtex_findimage_ns.fetch_add(findimage_ns, std::memory_order_relaxed);
    bindtex_view_ns.fetch_add(view_ns, std::memory_order_relaxed);
    bindtex_sampler_ns.fetch_add(sampler_ns, std::memory_order_relaxed);
}

inline void CountStreamCopy(u64 map_ns, u64 valid_ns, u64 copy_ns, u64 commit_ns) {
    stream_map_ns.fetch_add(map_ns, std::memory_order_relaxed);
    stream_valid_ns.fetch_add(valid_ns, std::memory_order_relaxed);
    stream_copy_ns.fetch_add(copy_ns, std::memory_order_relaxed);
    stream_commit_ns.fetch_add(commit_ns, std::memory_order_relaxed);
}

inline void CountUploadPhases(u64 scan_ns, u64 copy_ns, u64 mark_ns, bool is_written) {
    upload_scan_ns.fetch_add(scan_ns, std::memory_order_relaxed);
    upload_copy_ns.fetch_add(copy_ns, std::memory_order_relaxed);
    upload_mark_ns.fetch_add(mark_ns, std::memory_order_relaxed);
    if (is_written) {
        upload_written_calls.fetch_add(1, std::memory_order_relaxed);
    }
}

template <bool is_read>
inline void CountProtect(u64 ns) {
    if constexpr (is_read) {
        protect_read_calls.fetch_add(1, std::memory_order_relaxed);
        protect_read_ns.fetch_add(ns, std::memory_order_relaxed);
    } else {
        protect_write_calls.fetch_add(1, std::memory_order_relaxed);
        protect_write_ns.fetch_add(ns, std::memory_order_relaxed);
    }
}

inline void CountObtainSync(u64 query_ns, u64 sync_ns) {
    obtain_query_ns.fetch_add(query_ns, std::memory_order_relaxed);
    obtain_sync_ns.fetch_add(sync_ns, std::memory_order_relaxed);
    obtain_sync_calls.fetch_add(1, std::memory_order_relaxed);
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
