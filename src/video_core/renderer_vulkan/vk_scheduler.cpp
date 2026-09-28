// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstdlib>
#include <cstring>

#include "common/assert.h"
#include "common/debug.h"
#include "common/fault_monitor.h"
#include "common/thread.h"
#include "imgui/renderer/texture_manager.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

#include <thread>
#include "video_core/buffer_cache/buffer.h"
#include "video_core/renderer_vulkan/vk_gpu_batch_timer.h"

namespace Vulkan {

namespace {
// Per-draw flags, read once at startup instead of through function-local statics (each of
// those adds a guard check per call).
const int g_skip_empty_dynstate = [] {
    const char* value = std::getenv("SHADPS4_SKIP_EMPTY_DYNSTATE");
    return value ? value[0] - '0' : 0;
}();
const bool g_pop_fastpath = [] {
    const char* value = std::getenv("SHADPS4_POP_FASTPATH");
    return value && value[0] == '1';
}();
} // namespace

std::mutex Scheduler::submit_mutex;
std::atomic<VkSemaphore> Scheduler::async_timeline{VK_NULL_HANDLE};
std::atomic<u64> Scheduler::async_submitted_tick{0};

Scheduler::Scheduler(const Instance& instance, bool allow_recording_worker)
    : instance{instance}, master_semaphore{instance}, command_pool{instance, &master_semaphore} {
#if TRACY_GPU_ENABLED
    profiler_scope = reinterpret_cast<tracy::VkCtxScope*>(std::malloc(sizeof(tracy::VkCtxScope)));
#endif
    const char* enabled = std::getenv("SHADPS4_ASYNC_RECORDING");
    if (allow_recording_worker && enabled && std::string_view{enabled} == "1") {
        const char* timing = std::getenv("SHADPS4_ASYNC_RECORDING_TIMING");
        recording_measure_timing = timing && std::string_view{timing} == "1";
        recording_worker =
            std::make_unique<Common::CommandRecordingWorker<>>(recording_measure_timing);
        recording_worker->Record([] { Common::SetCurrentThreadName("shadPS4:VkRecorder"); });
        LOG_INFO(Render_Vulkan, "AsyncRecording: enabled, ordered primary command worker, {} chunks",
                 recording_worker->ActiveChunks());
        const char* async = std::getenv("SHADPS4_ASYNC_SUBMIT");
        async_submit = async && std::string_view{async} == "1";
        if (async_submit) {
            async_timeline.store(static_cast<VkSemaphore>(master_semaphore.Handle()));
            LOG_INFO(Render_Vulkan, "AsyncSubmit: the recording worker ends and submits batches");
            if (GpuBatchTimer::Enabled()) {
                batch_timer = std::make_unique<GpuBatchTimer>(instance, master_semaphore);
            }
        }
    }
    if (async_submit && instance.HasAscQueue()) {
        asc_enabled = true;
        asc_semaphore = std::make_unique<MasterSemaphore>(instance);
        // Recycled by the ASC timeline: an ASC command buffer must not be reset while the compute
        // queue still runs it (keyed by the main semaphore, this only held through I1; the
        // SHADPS4_ASC_NO_I1 probe froze, 2026-09-28).
        asc_pool = std::make_unique<CommandPool>(instance, asc_semaphore.get(),
                                                 instance.GetReadbackQueueFamilyIndex());
        LOG_INFO(Render_Vulkan, "AscStream: enabled");
    }
    AllocateWorkerCommandBuffers();
    priority_pending_ops_thread =
        std::jthread(std::bind_front(&Scheduler::PriorityPendingOpsThread, this));
}

Scheduler::~Scheduler() {
    recording_worker.reset();
#if TRACY_GPU_ENABLED
    std::free(profiler_scope);
#endif
}

void Scheduler::DrainRecording() {
    if (recording_worker) {
        recording_worker->Drain();
    }
}

void Scheduler::CommitDynamicState() {
    if (!IsAsyncRecording()) {
        dynamic_state.Commit(instance, current_cmdbuf);
        return;
    }
    // SHADPS4_SKIP_EMPTY_DYNSTATE=1: do not record a ~0.8 KB DynamicState snapshot when Commit
    // would emit nothing. Commit clears exactly the bits it emits, so it emits nothing iff the
    // postponed set computed below equals the current dirty set (padding bits only make the
    // comparison conservative). =2 also logs the skip rate every 2^20 calls.
    const int skip_empty = g_skip_empty_dynstate;
    auto& state = dynamic_state;
    const auto old_dirty = state.dirty_state;
    // Keep the conditional dirtiness which Commit deliberately postpones until
    // its corresponding test is enabled. All values in the packet are owned.
    decltype(state.dirty_state) kept{};
    kept.depth_compare_op = old_dirty.depth_compare_op && !state.depth_test_enabled;
    kept.depth_bounds = old_dirty.depth_bounds && !state.depth_bounds_test_enabled;
    kept.depth_bias = old_dirty.depth_bias && !state.depth_bias_enabled;
    if (!state.stencil_test_enabled) {
        kept.stencil_front_ops = old_dirty.stencil_front_ops;
        kept.stencil_back_ops = old_dirty.stencil_back_ops;
        kept.stencil_front_reference = old_dirty.stencil_front_reference;
        kept.stencil_back_reference = old_dirty.stencil_back_reference;
        kept.stencil_front_write_mask = old_dirty.stencil_front_write_mask;
        kept.stencil_back_write_mask = old_dirty.stencil_back_write_mask;
        kept.stencil_front_compare_mask = old_dirty.stencil_front_compare_mask;
        kept.stencil_back_compare_mask = old_dirty.stencil_back_compare_mask;
    }
    kept.feedback_loop_enabled =
        old_dirty.feedback_loop_enabled && !instance.IsAttachmentFeedbackLoopLayoutSupported();
    if (skip_empty > 0) {
        const bool empty = std::memcmp(&kept, &old_dirty, sizeof(kept)) == 0;
        if (skip_empty == 2) {
            static u64 calls = 0, skipped = 0;
            skipped += empty;
            if ((++calls & ((1u << 20) - 1)) == 0) {
                LOG_INFO(Render_Vulkan, "SKIP_EMPTY_DYNSTATE: skipped {} of {} commits ({:.1f}%)",
                         skipped, calls, 100.0 * skipped / calls);
            }
        }
        if (empty) {
            return;
        }
    }
    Record([snapshot = state, instance = &instance](vk::CommandBuffer cmdbuf) mutable {
        snapshot.Commit(*instance, cmdbuf);
    });
    state.dirty_state = kept;
}

void Scheduler::PushDescriptors(vk::PipelineBindPoint bind_point, vk::PipelineLayout layout,
                                std::span<const vk::WriteDescriptorSet> writes) {
    native_heaps_bound = false;
    if (!IsAsyncRecording()) {
        current_cmdbuf.pushDescriptorSetKHR(bind_point, layout, 0, writes);
        return;
    }
    size_t bytes = AlignedPayloadSize(writes.size_bytes());
    for (const auto& write : writes) {
        // Extension chains are not produced by the guest pipeline path. Keep a
        // scoped synchronous fallback for future callers with other descriptors.
        if (write.pNext) {
            auto access = AcquireCommandBuffer();
            access.Handle().pushDescriptorSetKHR(bind_point, layout, 0, writes);
            return;
        }
        if (write.pBufferInfo) {
            bytes += AlignedPayloadSize(sizeof(vk::DescriptorBufferInfo) * write.descriptorCount);
        }
        if (write.pImageInfo) {
            bytes += AlignedPayloadSize(sizeof(vk::DescriptorImageInfo) * write.descriptorCount);
        }
        if (write.pTexelBufferView) {
            bytes += AlignedPayloadSize(sizeof(vk::BufferView) * write.descriptorCount);
        }
    }
    if (bytes > 32768) {
        auto access = AcquireCommandBuffer();
        access.Handle().pushDescriptorSetKHR(bind_point, layout, 0, writes);
        return;
    }
    recording_worker->RecordWithPayload(bytes, [&](std::span<std::byte> storage) {
        auto* cursor = storage.data();
        auto* copies = reinterpret_cast<vk::WriteDescriptorSet*>(cursor);
        std::memcpy(copies, writes.data(), writes.size_bytes());
        cursor += AlignedPayloadSize(writes.size_bytes());
        for (size_t i = 0; i < writes.size(); ++i) {
            auto& write = copies[i];
            if (write.pBufferInfo) {
                write.pBufferInfo =
                    CopyPayload(cursor, std::span{write.pBufferInfo, write.descriptorCount}).data();
            }
            if (write.pImageInfo) {
                write.pImageInfo =
                    CopyPayload(cursor, std::span{write.pImageInfo, write.descriptorCount}).data();
            }
            if (write.pTexelBufferView) {
                write.pTexelBufferView =
                    CopyPayload(cursor, std::span{write.pTexelBufferView, write.descriptorCount})
                        .data();
            }
        }
        return [this, bind_point, layout, copies, count = writes.size()] {
            current_cmdbuf.pushDescriptorSetKHR(
                bind_point, layout, 0, std::span<const vk::WriteDescriptorSet>{copies, count});
        };
    });
}

void Scheduler::BeginRendering(const RenderState& new_state) {
    if (is_rendering && render_state == new_state) {
        return;
    }
    if (batch_timer && !is_rendering && last_end_site != ~0u && render_state == new_state) {
        GpuBreakSites().Resumed(last_end_site);
    }
    EndRendering();
    is_rendering = true;
    if (batch_timer) {
        GpuBreaks().render_begins.fetch_add(1, std::memory_order_relaxed);
        last_end_site = ~0u;
    }
    render_state = new_state;

    Record([render_state = new_state](vk::CommandBuffer cmdbuf) {
        std::array<vk::RenderingAttachmentInfo, 8> color_attachments;
        for (u32 i = 0; i < render_state.num_color_attachments; ++i) {
            const auto& cb = render_state.color_attachments[i];
            color_attachments[i] = vk::RenderingAttachmentInfo{
                .imageView = cb.image_view,
                .imageLayout = cb.image_layout,
                .loadOp = cb.is_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad,
                .storeOp = vk::AttachmentStoreOp::eStore,
                .clearValue =
                    vk::ClearValue{.color = vk::ClearColorValue{.uint32 = cb.clear_value}},
            };
        }

        const auto& db = render_state.depth_stencil_attachment;
        const vk::RenderingAttachmentInfo depth_attachment = {
            .imageView = db.image_view,
            .imageLayout = db.image_layout,
            .loadOp = db.depth_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .clearValue = vk::ClearValue{.depthStencil =
                                             vk::ClearDepthStencilValue{
                                                 .depth = std::bit_cast<float>(db.clear_value[0])}},
        };
        const vk::RenderingAttachmentInfo stencil_attachment = {
            .imageView = db.image_view,
            .imageLayout = db.image_layout,
            .loadOp = db.stencil_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .clearValue =
                vk::ClearValue{.depthStencil =
                                   vk::ClearDepthStencilValue{.stencil = db.clear_value[1]}},
        };

        const vk::RenderingInfo rendering_info = {
            .renderArea =
                {
                    .offset = {0, 0},
                    .extent = {render_state.width, render_state.height},
                },
            .layerCount = render_state.num_layers,
            .colorAttachmentCount = render_state.num_color_attachments,
            .pColorAttachments = color_attachments.data(),
            .pDepthAttachment = db.has_depth ? &depth_attachment : nullptr,
            .pStencilAttachment = db.has_stencil ? &stencil_attachment : nullptr,
        };

        cmdbuf.beginRendering(rendering_info);
    });
}

void Scheduler::EndRendering(std::source_location loc) {
    if (!is_rendering) {
        return;
    }
    is_rendering = false;
    if (batch_timer) {
        last_end_site = GpuBreakSites().Site(loc);
        GpuBreakSites().End(last_end_site);
    }
    Record([](vk::CommandBuffer cmdbuf) { cmdbuf.endRendering(); });
}

void Scheduler::Flush(SubmitInfo& info) {
    // When flushing, we only send data to the driver; no waiting is necessary.
    SubmitExecution(info);
}

void Scheduler::Flush() {
    SubmitInfo info{};
    Flush(info);
}

void Scheduler::Finish() {
    // When finishing, we need to wait for the submission to have executed on the device.
    const u64 presubmit_tick = CurrentTick();
    SubmitInfo info{};
    SubmitExecution(info);
    Wait(presubmit_tick);
}

void Scheduler::Wait(u64 tick) {
    WaitMain(tick);
    // R0: the ASC segments submitted before that main flush (a no-op while I1 holds).
    if (const u64 value = AscAt(tick); value != 0) {
        asc_semaphore->Wait(value);
    }
}

u64 Scheduler::AscMainWaitValue(bool no_i1) {
    // SHADPS4_ASC_I1 (notes/asc-i1-relax-design-2026-09-28.md): unset/1 -> I1, the last ASC value
    // submitted; 2 -> I1 plus counters of the precise value; 0 -> the precise value only: the
    // newest unretired ASC segment an access recorded in this batch depends on (RAW/WAR/WAW),
    // always <= the value just submitted by SubmitAscSegment, so never an unsubmitted value.
    static const char mode = [] {
        const char* value = std::getenv("SHADPS4_ASC_I1");
        return value ? value[0] : '1';
    }();
    const u64 last = asc_last_submitted.load(std::memory_order_relaxed);
    const u64 precise = std::min(VideoCore::g_main_wait_asc, last);
    VideoCore::g_main_wait_asc = 0;
    asc_semaphore->Refresh();
    VideoCore::g_asc_known = asc_semaphore->KnownGpuTick();
    // The batch being flushed took the tick NextTick() is about to return; accesses from now on
    // belong to the next one (+1). Only the falsifier reads it.
    VideoCore::g_asc_open_tick = master_semaphore.CurrentTick() + 1;
    if (mode == '0' || mode == '2') {
        static u64 flushes = 0, i1_waits = 0, precise_waits = 0;
        ++flushes;
        i1_waits += last > asc_waited_by_main && last > VideoCore::g_asc_known;
        precise_waits += precise > asc_waited_by_main && precise > VideoCore::g_asc_known;
        if ((flushes & 0xffff) == 0) {
            LOG_INFO(Render_Vulkan,
                     "AscDeps: mode={} main_flushes={} i1_unfinished_waits={} precise_waits={} "
                     "slow_checks={} hook_miss={} whole_buffers={}",
                     mode, flushes, i1_waits, precise_waits, VideoCore::g_asc_dep_slow,
                     VideoCore::g_asc_hook_miss, VideoCore::g_asc_whole_buffers);
        }
    }
    if (no_i1) {
        return 0;
    }
    return mode == '0' ? precise : last;
}

void Scheduler::WaitMain(u64 tick) {
    if (tick >= master_semaphore.CurrentTick()) {
        // Make sure we are not waiting for the current tick without signalling
        SubmitInfo info{};
        Flush(info);
    }
    const auto start = Common::FaultMonitor::Now();
    master_semaphore.Wait(tick);
    Common::FaultMonitor::CountGpuWait(Common::FaultMonitor::SinceNs(start));
}

void Scheduler::PopPendingOperations() {
    // SHADPS4_POP_FASTPATH=1: called on every draw and dispatch; skip the lock and the driver
    // counter query when nothing is queued, and poll the counter only every 16th call while the
    // oldest operation is not yet known to be free. Measured 2026-09-24 on 386F5167: ~1.6% of the
    // CP's wall time. pending_ops is only touched on this thread.
    const bool fast = g_pop_fastpath;
    if (fast) {
        if (pending_ops.empty()) {
            return;
        }
        static thread_local u32 calls = 0;
        if (!master_semaphore.IsFree(pending_ops.front().gpu_tick) && (++calls & 15) != 0) {
            return;
        }
    }
    std::unique_lock lk(priority_pending_ops_mutex);
    master_semaphore.Refresh();
    // R0: an operation retires when its main tick and the ASC segments before it are done.
    while (!pending_ops.empty() && master_semaphore.IsFree(pending_ops.front().gpu_tick) &&
           AscRetired(pending_ops.front().gpu_tick)) {
        pending_ops.front().callback();
        pending_ops.pop();
    }
}

void Scheduler::AllocateWorkerCommandBuffers() {
    DrainRecording();
    const vk::CommandBufferBeginInfo begin_info = {
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
    };

    main_cmdbuf = command_pool.Commit();
    Check(main_cmdbuf.begin(begin_info));
    current_cmdbuf = worker_stream == 1 ? asc_cmdbuf : main_cmdbuf;
    GraphicsBindTracker::Invalidate();

    // Invalidate dynamic state so it gets applied to the new command buffer.
    dynamic_state.Invalidate();
    native_heaps_bound = false;

#if TRACY_GPU_ENABLED
    auto* profiler_ctx = instance.GetProfilerContext();
    if (profiler_ctx) {
        static const auto scope_loc =
            GPU_SCOPE_LOCATION("Guest Frame", MarkersPalette::GpuMarkerColor);
        new (profiler_scope) tracy::VkCtxScope{profiler_ctx, &scope_loc, current_cmdbuf, true};
    }
#endif
}

void Scheduler::SubmitExecutionAsync(SubmitInfo& info) {
    // SHADPS4_ASYNC_SUBMIT=1: the command processor only picks the tick; the recording worker
    // ends the command buffer, submits it and begins the next one after every command recorded
    // before this point. Measured 2026-09-24: the CP spent ~4.2% of its wall time draining the
    // worker for submits/borrows and ~1.6% inside vkQueueSubmit.
    Common::FaultMonitor::CountGpuSubmit();
    EndRendering();
    if (asc_enabled) {
        // I1: the ASC work recorded before this batch is submitted first and waited on.
        SubmitAscSegment();
        // SHADPS4_ASC_NO_I1=1: DIAGNOSTIC ONLY, UNSAFE (data races, resource reuse): skip the
        // wait, to measure the upper bound of relaxing I1 (notes/asc-i1-relax-design-2026-09-28.md).
        static const bool no_i1 = [] {
            const char* value = std::getenv("SHADPS4_ASC_NO_I1");
            return value && value[0] == '1';
        }();
        const u64 need = AscMainWaitValue(no_i1);
        if (need > asc_waited_by_main) {
            // Wait masks are per index (index 1 = colour output): only index 0 or 2 is AllCommands.
            ASSERT(info.num_wait_semas != 1);
            info.AddWait(asc_semaphore->Handle(), need);
            asc_waited_by_main = need;
        }
    }
    const u64 signal_value = master_semaphore.NextTick();
    NoteAscAtFlush(signal_value);
    info.AddSignal(master_semaphore.Handle(), signal_value);
    // SHADPS4_SUBMIT_LOCK_SKIP=1: take the global submit lock for the overlay's texture uploads
    // only when one is queued. The recorder holds this lock across vkQueueSubmit, the readback
    // prebatch queue and the presenter take it too, and a30 stacks showed the command processor
    // blocked here 3.4% (fire) / 6.9% (forest) of its wall time with nothing to upload
    // (runs threads-stack-*-a30-20260928-a).
    static const bool submit_lock_skip = [] {
        const char* value = std::getenv("SHADPS4_SUBMIT_LOCK_SKIP");
        return value && value[0] == '1';
    }();
    if (!submit_lock_skip || ImGui::Core::TextureManager::HasPendingUploads()) {
        std::scoped_lock lk{submit_mutex};
        ImGui::Core::TextureManager::Submit();
    }
    const u32 draws = batch_draws;
    const u32 dispatches = batch_dispatches;
    batch_draws = 0;
    batch_dispatches = 0;
    batch_ops = 0;
    recording_worker->Record([this, info, signal_value, draws, dispatches]() mutable {
        if (batch_timer) {
            bool asc_wait = false;
            for (u32 i = 0; i < info.num_wait_semas; ++i) {
                asc_wait |= asc_enabled && info.wait_semas[i] == asc_semaphore->Handle();
            }
            batch_timer->End(main_cmdbuf, signal_value, draws, dispatches, asc_wait);
        }
        Check(main_cmdbuf.end());
        static constexpr std::array<vk::PipelineStageFlags, 3> wait_stage_masks = {
            vk::PipelineStageFlagBits::eAllCommands,
            vk::PipelineStageFlagBits::eColorAttachmentOutput,
            vk::PipelineStageFlagBits::eAllCommands,
        };
        const vk::TimelineSemaphoreSubmitInfo timeline_si = {
            .waitSemaphoreValueCount = info.num_wait_semas,
            .pWaitSemaphoreValues = info.wait_ticks.data(),
            .signalSemaphoreValueCount = info.num_signal_semas,
            .pSignalSemaphoreValues = info.signal_ticks.data(),
        };
        const vk::SubmitInfo submit_info = {
            .pNext = &timeline_si,
            .waitSemaphoreCount = info.num_wait_semas,
            .pWaitSemaphores = info.wait_semas.data(),
            .pWaitDstStageMask = wait_stage_masks.data(),
            .commandBufferCount = 1U,
            .pCommandBuffers = &main_cmdbuf,
            .signalSemaphoreCount = info.num_signal_semas,
            .pSignalSemaphores = info.signal_semas.data(),
        };
        // SHADPS4_ASC_I1_STATS=1 (diagnostic, notes/asc-i1-relax-design-2026-09-28.md step 0b): at
        // submit time, is the ASC value this batch waits on (I1) still unfinished, and is the main
        // queue then already idle (previous batch done) so the wait certainly stalls it?
        static const bool i1_stats = [] {
            const char* value = std::getenv("SHADPS4_ASC_I1_STATS");
            return value && value[0] == '1';
        }();
        if (i1_stats && asc_enabled) [[unlikely]] {
            static u64 submits = 0, asc_waits = 0, pending = 0, pending_main_idle = 0,
                       backlog_sum = 0;
            ++submits;
            for (u32 i = 0; i < info.num_wait_semas; ++i) {
                if (info.wait_semas[i] != asc_semaphore->Handle()) {
                    continue;
                }
                ++asc_waits;
                asc_semaphore->Refresh();
                const u64 known = asc_semaphore->KnownGpuTick();
                if (known < info.wait_ticks[i]) {
                    ++pending;
                    backlog_sum += info.wait_ticks[i] - known;
                    master_semaphore.Refresh();
                    if (master_semaphore.IsFree(signal_value - 1)) {
                        ++pending_main_idle;
                    }
                }
            }
            if (submits % 10000 == 0) {
                LOG_INFO(Render_Vulkan,
                         "AscI1Stats: main_submits={} with_asc_wait={} asc_unfinished_at_submit={} "
                         "({:.1f}%) of_which_main_idle={} mean_asc_backlog={:.2f}",
                         submits, asc_waits, pending, 100.0 * double(pending) / double(submits),
                         pending_main_idle,
                         pending ? double(backlog_sum) / double(pending) : 0.0);
            }
        }
        {
            std::scoped_lock lk{submit_mutex};
            const auto result = instance.GetGraphicsQueue().submit(submit_info, info.fence);
            ASSERT_MSG(result != vk::Result::eErrorDeviceLost, "Device lost during submit");
            async_submitted_tick.store(signal_value, std::memory_order_release);
        }
        main_cmdbuf = command_pool.Commit();
        Check(main_cmdbuf.begin(vk::CommandBufferBeginInfo{
            .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
        }));
        if (worker_stream == 0) {
            current_cmdbuf = main_cmdbuf;
        }
        GraphicsBindTracker::Invalidate();
        if (batch_timer) {
            batch_timer->Begin(main_cmdbuf);
        }
    });
    // A partial chunk is otherwise only handed over when full or drained; a waiter on this tick
    // would then wait for commands nobody records.
    recording_worker->Publish();
    // What AllocateWorkerCommandBuffers resets on this thread for a new command buffer.
    dynamic_state.Invalidate();
    native_heaps_bound = false;
    master_semaphore.Refresh();
    PopPendingOperations();
}

void Scheduler::SubmitExecution(SubmitInfo& info) {
    if (async_submit && synchronous_scopes == 0) {
        SubmitExecutionAsync(info);
        return;
    }
    if (async_submit) {
        // Inside a borrow scope. The worker may still hold queued submissions, which take
        // submit_mutex: drain it before taking the lock below, or both wait forever.
        DrainRecording();
    }
    // A batch on this queue must not wait for a draw-timeline value the async worker has not
    // submitted yet: it would block the queue ahead of its own signal.
    if (const VkSemaphore async = async_timeline.load(); async != VK_NULL_HANDLE) {
        for (u32 i = 0; i < info.num_wait_semas; ++i) {
            if (static_cast<VkSemaphore>(info.wait_semas[i]) == async) {
                while (async_submitted_tick.load(std::memory_order_acquire) < info.wait_ticks[i]) {
                    std::this_thread::yield();
                }
            }
        }
    }
    std::scoped_lock lk{submit_mutex};
    Common::FaultMonitor::CountGpuSubmit();
    EndRendering();
    DrainRecording();
    if (asc_enabled) {
        SubmitAscSegment(); // Drained: runs on this thread.
        const u64 need = AscMainWaitValue(false);
        if (need > asc_waited_by_main) {
            info.AddWait(asc_semaphore->Handle(), need);
            asc_waited_by_main = need;
        }
    }
    if (recording_worker && ++recording_submits % 1000 == 0) {
        const auto counts = recording_worker->GetCounters();
        LOG_INFO(
            Render_Vulkan, "AsyncRecording: commands={} chunks={} bytes={} borrows={} submits={}",
            counts.commands, counts.chunks, counts.bytes, synchronous_borrows, recording_submits);
        if (recording_measure_timing) {
            LOG_INFO(Render_Vulkan, "AsyncRecordingTime: worker_ns={} drain_ns={} space_wait_ns={}",
                     counts.worker_ns, counts.drain_ns, counts.space_wait_ns);
            for (const auto& [caller, calls] : recording_borrow_counts) {
                LOG_INFO(Render_Vulkan, "AsyncBorrow: calls={} caller={}", calls, caller);
            }
        }
    }
    const u64 signal_value = master_semaphore.NextTick();
    NoteAscAtFlush(signal_value);
    last_batch_draws = batch_draws;
    batch_draws = 0;
    batch_dispatches = 0;
    batch_ops = 0;

#if TRACY_GPU_ENABLED
    auto* profiler_ctx = instance.GetProfilerContext();
    if (profiler_ctx) {
        profiler_scope->~VkCtxScope();
        TracyVkCollect(profiler_ctx, current_cmdbuf);
    }
#endif

    EndRendering();
    Check(main_cmdbuf.end());

    const vk::Semaphore timeline = master_semaphore.Handle();
    info.AddSignal(timeline, signal_value);

    static constexpr std::array<vk::PipelineStageFlags, 3> wait_stage_masks = {
        vk::PipelineStageFlagBits::eAllCommands,
        vk::PipelineStageFlagBits::eColorAttachmentOutput,
        vk::PipelineStageFlagBits::eAllCommands,
    };

    const vk::TimelineSemaphoreSubmitInfo timeline_si = {
        .waitSemaphoreValueCount = info.num_wait_semas,
        .pWaitSemaphoreValues = info.wait_ticks.data(),
        .signalSemaphoreValueCount = info.num_signal_semas,
        .pSignalSemaphoreValues = info.signal_ticks.data(),
    };

    const vk::SubmitInfo submit_info = {
        .pNext = &timeline_si,
        .waitSemaphoreCount = info.num_wait_semas,
        .pWaitSemaphores = info.wait_semas.data(),
        .pWaitDstStageMask = wait_stage_masks.data(),
        .commandBufferCount = 1U,
        .pCommandBuffers = &main_cmdbuf,
        .signalSemaphoreCount = info.num_signal_semas,
        .pSignalSemaphores = info.signal_semas.data(),
    };

    ImGui::Core::TextureManager::Submit();
    auto submit_result = instance.GetGraphicsQueue().submit(submit_info, info.fence);
    ASSERT_MSG(submit_result != vk::Result::eErrorDeviceLost, "Device lost during submit");
    if (async_submit) {
        // A synchronous submission of the async timeline (inside a borrow scope) counts too.
        async_submitted_tick.store(signal_value, std::memory_order_release);
    }

    master_semaphore.Refresh();
    AllocateWorkerCommandBuffers();

    // Apply pending operations
    PopPendingOperations();
}

void Scheduler::BeginAscCmdbufOnWorker() {
    asc_cmdbuf = asc_pool->Commit();
    Check(asc_cmdbuf.begin(vk::CommandBufferBeginInfo{
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
    }));
}

void Scheduler::SwitchStream(u32 stream) {
    if (!asc_enabled || stream == active_stream) {
        return;
    }
    if (active_stream == 0) {
        // A render pass lives in the main command buffer only.
        EndRendering();
    }
    if (stream == 1 && !asc_dirty) {
        open_asc_value = asc_semaphore->NextTick();
        asc_dirty = true;
        asc_wait_main = 0;
    }
    active_stream = stream;
    native_heaps_bound = false;
    VideoCore::g_record_stream = stream;
    auto apply = [this, stream] {
        if (stream == 1 && !asc_cmdbuf) {
            BeginAscCmdbufOnWorker();
        }
        worker_stream = stream;
        current_cmdbuf = stream == 1 ? asc_cmdbuf : main_cmdbuf;
    };
    if (IsAsyncRecording()) {
        recording_worker->Record(std::move(apply));
    } else {
        apply();
    }
}

u64 Scheduler::SubmitAscSegment() {
    if (!asc_enabled || !asc_dirty) {
        return 0;
    }
    const u64 value = open_asc_value;
    const u64 wait_main = asc_wait_main;
    asc_dirty = false;
    asc_last_submitted = value;
    auto submit = [this, value, wait_main] { AscSubmitOnWorker(value, wait_main); };
    if (IsAsyncRecording()) {
        recording_worker->Record(std::move(submit));
        recording_worker->Publish();
    } else {
        submit();
    }
    if (active_stream == 1) {
        // Still recording ASC work: continue in a new segment. It keeps the main wait: the rest
        // of the dispatch being recorded depends on the same main batches (Opus review
        // 2026-09-28: a mid-pass-2 flush otherwise left it unordered after them).
        open_asc_value = asc_semaphore->NextTick();
        asc_dirty = true;
        asc_wait_main = wait_main;
    }
    return value;
}

void Scheduler::AscSubmitOnWorker(u64 value, u64 wait_main) {
    if (!asc_cmdbuf) {
        BeginAscCmdbufOnWorker(); // An empty segment still signals its value.
    }
    Check(asc_cmdbuf.end());
    const vk::Semaphore wait_sema = master_semaphore.Handle();
    const vk::Semaphore signal_sema = asc_semaphore->Handle();
    const vk::PipelineStageFlags wait_stage = vk::PipelineStageFlagBits::eAllCommands;
    const u32 wait_count = wait_main != 0 ? 1u : 0u;
    const vk::TimelineSemaphoreSubmitInfo timeline_si = {
        .waitSemaphoreValueCount = wait_count,
        .pWaitSemaphoreValues = &wait_main,
        .signalSemaphoreValueCount = 1,
        .pSignalSemaphoreValues = &value,
    };
    const vk::SubmitInfo submit_info = {
        .pNext = &timeline_si,
        .waitSemaphoreCount = wait_count,
        .pWaitSemaphores = &wait_sema,
        .pWaitDstStageMask = &wait_stage,
        .commandBufferCount = 1U,
        .pCommandBuffers = &asc_cmdbuf,
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &signal_sema,
    };
    const auto result = instance.GetAscQueue().submit(submit_info);
    ASSERT_MSG(result != vk::Result::eErrorDeviceLost, "Device lost during ASC submit");
    asc_cmdbuf = nullptr;
    if (worker_stream == 1) {
        BeginAscCmdbufOnWorker();
        current_cmdbuf = asc_cmdbuf;
    }
}

void Scheduler::PriorityPendingOpsThread(std::stop_token stoken) {
    Common::SetCurrentThreadName("shadPS4:GpuSchedPriorityPendingOpsRunner");

    while (!stoken.stop_requested()) {
        PendingOp op;
        {
            std::unique_lock lk(priority_pending_ops_mutex);
            priority_pending_ops_cv.wait(lk, stoken,
                                         [this] { return !priority_pending_ops.empty(); });
            if (stoken.stop_requested()) {
                break;
            }

            op = std::move(priority_pending_ops.front());
            priority_pending_ops.pop();
        }

        master_semaphore.Wait(op.gpu_tick);
        if (const u64 value = AscAt(op.gpu_tick); value != 0) {
            asc_semaphore->Wait(value); // R0
        }
        if (stoken.stop_requested()) {
            break;
        }

        op.callback();
    }
}

void DynamicState::Commit(const Instance& instance, const vk::CommandBuffer& cmdbuf) {
    if (dirty_state.viewports) {
        dirty_state.viewports = false;
        cmdbuf.setViewportWithCount(viewports);
    }
    if (dirty_state.scissors) {
        dirty_state.scissors = false;
        cmdbuf.setScissorWithCount(scissors);
    }
    if (dirty_state.depth_test_enabled) {
        dirty_state.depth_test_enabled = false;
        cmdbuf.setDepthTestEnable(depth_test_enabled);
    }
    if (dirty_state.depth_write_enabled) {
        dirty_state.depth_write_enabled = false;
        // Note that this must be set in a command buffer even if depth test is disabled.
        cmdbuf.setDepthWriteEnable(depth_write_enabled);
    }
    if (depth_test_enabled && dirty_state.depth_compare_op) {
        dirty_state.depth_compare_op = false;
        cmdbuf.setDepthCompareOp(depth_compare_op);
    }
    if (dirty_state.depth_bounds_test_enabled) {
        dirty_state.depth_bounds_test_enabled = false;
        if (instance.IsDepthBoundsSupported()) {
            cmdbuf.setDepthBoundsTestEnable(depth_bounds_test_enabled);
        }
    }
    if (depth_bounds_test_enabled && dirty_state.depth_bounds) {
        dirty_state.depth_bounds = false;
        if (instance.IsDepthBoundsSupported()) {
            cmdbuf.setDepthBounds(depth_bounds_min, depth_bounds_max);
        }
    }
    if (dirty_state.depth_bias_enabled) {
        dirty_state.depth_bias_enabled = false;
        cmdbuf.setDepthBiasEnable(depth_bias_enabled);
    }
    if (depth_bias_enabled && dirty_state.depth_bias) {
        dirty_state.depth_bias = false;
        cmdbuf.setDepthBias(depth_bias_constant, depth_bias_clamp, depth_bias_slope);
    }
    if (dirty_state.stencil_test_enabled) {
        dirty_state.stencil_test_enabled = false;
        cmdbuf.setStencilTestEnable(stencil_test_enabled);
    }
    if (stencil_test_enabled) {
        if (dirty_state.stencil_front_ops && dirty_state.stencil_back_ops &&
            stencil_front_ops == stencil_back_ops) {
            dirty_state.stencil_front_ops = false;
            dirty_state.stencil_back_ops = false;
            cmdbuf.setStencilOp(vk::StencilFaceFlagBits::eFrontAndBack, stencil_front_ops.fail_op,
                                stencil_front_ops.pass_op, stencil_front_ops.depth_fail_op,
                                stencil_front_ops.compare_op);
        } else {
            if (dirty_state.stencil_front_ops) {
                dirty_state.stencil_front_ops = false;
                cmdbuf.setStencilOp(vk::StencilFaceFlagBits::eFront, stencil_front_ops.fail_op,
                                    stencil_front_ops.pass_op, stencil_front_ops.depth_fail_op,
                                    stencil_front_ops.compare_op);
            }
            if (dirty_state.stencil_back_ops) {
                dirty_state.stencil_back_ops = false;
                cmdbuf.setStencilOp(vk::StencilFaceFlagBits::eBack, stencil_back_ops.fail_op,
                                    stencil_back_ops.pass_op, stencil_back_ops.depth_fail_op,
                                    stencil_back_ops.compare_op);
            }
        }
        if (dirty_state.stencil_front_reference && dirty_state.stencil_back_reference &&
            stencil_front_reference == stencil_back_reference) {
            dirty_state.stencil_front_reference = false;
            dirty_state.stencil_back_reference = false;
            cmdbuf.setStencilReference(vk::StencilFaceFlagBits::eFrontAndBack,
                                       stencil_front_reference);
        } else {
            if (dirty_state.stencil_front_reference) {
                dirty_state.stencil_front_reference = false;
                cmdbuf.setStencilReference(vk::StencilFaceFlagBits::eFront,
                                           stencil_front_reference);
            }
            if (dirty_state.stencil_back_reference) {
                dirty_state.stencil_back_reference = false;
                cmdbuf.setStencilReference(vk::StencilFaceFlagBits::eBack, stencil_back_reference);
            }
        }
        if (dirty_state.stencil_front_write_mask && dirty_state.stencil_back_write_mask &&
            stencil_front_write_mask == stencil_back_write_mask) {
            dirty_state.stencil_front_write_mask = false;
            dirty_state.stencil_back_write_mask = false;
            cmdbuf.setStencilWriteMask(vk::StencilFaceFlagBits::eFrontAndBack,
                                       stencil_front_write_mask);
        } else {
            if (dirty_state.stencil_front_write_mask) {
                dirty_state.stencil_front_write_mask = false;
                cmdbuf.setStencilWriteMask(vk::StencilFaceFlagBits::eFront,
                                           stencil_front_write_mask);
            }
            if (dirty_state.stencil_back_write_mask) {
                dirty_state.stencil_back_write_mask = false;
                cmdbuf.setStencilWriteMask(vk::StencilFaceFlagBits::eBack, stencil_back_write_mask);
            }
        }
        if (dirty_state.stencil_front_compare_mask && dirty_state.stencil_back_compare_mask &&
            stencil_front_compare_mask == stencil_back_compare_mask) {
            dirty_state.stencil_front_compare_mask = false;
            dirty_state.stencil_back_compare_mask = false;
            cmdbuf.setStencilCompareMask(vk::StencilFaceFlagBits::eFrontAndBack,
                                         stencil_front_compare_mask);
        } else {
            if (dirty_state.stencil_front_compare_mask) {
                dirty_state.stencil_front_compare_mask = false;
                cmdbuf.setStencilCompareMask(vk::StencilFaceFlagBits::eFront,
                                             stencil_front_compare_mask);
            }
            if (dirty_state.stencil_back_compare_mask) {
                dirty_state.stencil_back_compare_mask = false;
                cmdbuf.setStencilCompareMask(vk::StencilFaceFlagBits::eBack,
                                             stencil_back_compare_mask);
            }
        }
    }
    if (dirty_state.primitive_restart_enable) {
        dirty_state.primitive_restart_enable = false;
        cmdbuf.setPrimitiveRestartEnable(primitive_restart_enable);
    }
    if (dirty_state.rasterizer_discard_enable) {
        dirty_state.rasterizer_discard_enable = false;
        cmdbuf.setRasterizerDiscardEnable(rasterizer_discard_enable);
    }
    if (dirty_state.cull_mode) {
        dirty_state.cull_mode = false;
        cmdbuf.setCullMode(cull_mode);
    }
    if (dirty_state.front_face) {
        dirty_state.front_face = false;
        cmdbuf.setFrontFace(front_face);
    }
    if (dirty_state.blend_constants) {
        dirty_state.blend_constants = false;
        cmdbuf.setBlendConstants(blend_constants.data());
    }
    if (dirty_state.color_write_masks) {
        dirty_state.color_write_masks = false;
        if (instance.IsDynamicColorWriteMaskSupported()) {
            cmdbuf.setColorWriteMaskEXT(0, color_write_masks);
        }
    }
    if (dirty_state.line_width) {
        dirty_state.line_width = false;
        cmdbuf.setLineWidth(line_width);
    }
    if (dirty_state.feedback_loop_enabled && instance.IsAttachmentFeedbackLoopLayoutSupported()) {
        dirty_state.feedback_loop_enabled = false;
        cmdbuf.setAttachmentFeedbackLoopEnableEXT(feedback_loop_enabled
                                                      ? vk::ImageAspectFlagBits::eColor
                                                      : vk::ImageAspectFlagBits::eNone);
        // Not dynamic in the graphics pipelines: only the next bind overrides it, so the next
        // draw must bind even if its pipeline is already bound (SHADPS4_SKIP_REBIND).
        GraphicsBindTracker::Invalidate();
    }
}

} // namespace Vulkan
