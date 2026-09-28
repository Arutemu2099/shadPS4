// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <condition_variable>
#include <mutex>
#include <source_location>
#include <span>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <queue>

#include "common/command_recording_worker.h"
#include "common/logging/log.h"

#include "common/unique_function.h"
#include "video_core/amdgpu/regs_color.h"
#include "video_core/amdgpu/regs_primitive.h"
#include "video_core/renderer_vulkan/vk_master_semaphore.h"
#include "video_core/renderer_vulkan/vk_resource_pool.h"

namespace tracy {
class VkCtxScope;
}

namespace Vulkan {

class Instance;

struct RenderAttachment {
    vk::ImageView image_view;
    vk::ImageLayout image_layout;
    std::array<u32, 4> clear_value;
    union {
        u32 is_clear;
        struct {
            bool has_depth;
            bool depth_clear;
            bool has_stencil;
            bool stencil_clear;
        };
    };
};
static_assert(std::has_unique_object_representations_v<RenderAttachment>);

struct RenderState {
    std::array<RenderAttachment, 8> color_attachments;
    RenderAttachment depth_stencil_attachment;
    u16 width;
    u16 height;
    u16 num_layers;
    u16 num_color_attachments;

    bool operator==(const RenderState& other) const noexcept {
        return std::memcmp(this, &other, sizeof(RenderState)) == 0;
    }
};
static_assert(std::has_unique_object_representations_v<RenderState>);

struct SubmitInfo {
    std::array<vk::Semaphore, 3> wait_semas;
    std::array<u64, 3> wait_ticks;
    std::array<vk::Semaphore, 3> signal_semas;
    std::array<u64, 3> signal_ticks;
    vk::Fence fence;
    u32 num_wait_semas;
    u32 num_signal_semas;

    void AddWait(vk::Semaphore semaphore, u64 tick = 1) {
        wait_semas[num_wait_semas] = semaphore;
        wait_ticks[num_wait_semas++] = tick;
    }

    void AddSignal(vk::Semaphore semaphore, u64 tick = 1) {
        signal_semas[num_signal_semas] = semaphore;
        signal_ticks[num_signal_semas++] = tick;
    }

    void AddSignal(vk::Fence fence) {
        this->fence = fence;
    }
};

using Viewports = boost::container::static_vector<vk::Viewport, AmdGpu::NUM_VIEWPORTS>;
using Scissors = boost::container::static_vector<vk::Rect2D, AmdGpu::NUM_VIEWPORTS>;
using ColorWriteMasks = std::array<vk::ColorComponentFlags, AmdGpu::NUM_COLOR_BUFFERS>;
struct StencilOps {
    vk::StencilOp fail_op{};
    vk::StencilOp pass_op{};
    vk::StencilOp depth_fail_op{};
    vk::CompareOp compare_op{};

    bool operator==(const StencilOps& other) const {
        return fail_op == other.fail_op && pass_op == other.pass_op &&
               depth_fail_op == other.depth_fail_op && compare_op == other.compare_op;
    }
};
struct DynamicState {
    struct {
        bool viewports : 1;
        bool scissors : 1;

        bool depth_test_enabled : 1;
        bool depth_write_enabled : 1;
        bool depth_compare_op : 1;

        bool depth_bounds_test_enabled : 1;
        bool depth_bounds : 1;

        bool depth_bias_enabled : 1;
        bool depth_bias : 1;

        bool stencil_test_enabled : 1;
        bool stencil_front_ops : 1;
        bool stencil_front_reference : 1;
        bool stencil_front_write_mask : 1;
        bool stencil_front_compare_mask : 1;
        bool stencil_back_ops : 1;
        bool stencil_back_reference : 1;
        bool stencil_back_write_mask : 1;
        bool stencil_back_compare_mask : 1;

        bool primitive_restart_enable : 1;
        bool rasterizer_discard_enable : 1;
        bool cull_mode : 1;
        bool front_face : 1;

        bool blend_constants : 1;
        bool color_write_masks : 1;
        bool line_width : 1;
        bool feedback_loop_enabled : 1;
    } dirty_state{};

    Viewports viewports{};
    Scissors scissors{};

    bool depth_test_enabled{};
    bool depth_write_enabled{};
    vk::CompareOp depth_compare_op{};

    bool depth_bounds_test_enabled{};
    float depth_bounds_min{};
    float depth_bounds_max{};

    bool depth_bias_enabled{};
    float depth_bias_constant{};
    float depth_bias_clamp{};
    float depth_bias_slope{};

    bool stencil_test_enabled{};
    StencilOps stencil_front_ops{};
    u32 stencil_front_reference{};
    u32 stencil_front_write_mask{};
    u32 stencil_front_compare_mask{};
    StencilOps stencil_back_ops{};
    u32 stencil_back_reference{};
    u32 stencil_back_write_mask{};
    u32 stencil_back_compare_mask{};

    bool primitive_restart_enable{};
    bool rasterizer_discard_enable{};
    vk::CullModeFlags cull_mode{};
    vk::FrontFace front_face{};

    std::array<float, 4> blend_constants{};
    ColorWriteMasks color_write_masks{};
    float line_width{};
    bool feedback_loop_enabled{};

    /// Commits the dynamic state to the provided command buffer.
    void Commit(const Instance& instance, const vk::CommandBuffer& cmdbuf);

    /// Invalidates all dynamic state to be flushed into the next command buffer.
    void Invalidate() {
        std::memset(&dirty_state, 0xFF, sizeof(dirty_state));
    }

    void SetViewports(const Viewports& viewports_) {
        if (!std::ranges::equal(viewports, viewports_)) {
            viewports = viewports_;
            dirty_state.viewports = true;
        }
    }

    void SetScissors(const Scissors& scissors_) {
        if (!std::ranges::equal(scissors, scissors_)) {
            scissors = scissors_;
            dirty_state.scissors = true;
        }
    }

    void SetDepthTestEnabled(const bool enabled) {
        if (depth_test_enabled != enabled) {
            depth_test_enabled = enabled;
            dirty_state.depth_test_enabled = true;
        }
    }

    void SetDepthWriteEnabled(const bool enabled) {
        if (depth_write_enabled != enabled) {
            depth_write_enabled = enabled;
            dirty_state.depth_write_enabled = true;
        }
    }

    void SetDepthCompareOp(const vk::CompareOp compare_op) {
        if (depth_compare_op != compare_op) {
            depth_compare_op = compare_op;
            dirty_state.depth_compare_op = true;
        }
    }

    void SetDepthBoundsTestEnabled(const bool enabled) {
        if (depth_bounds_test_enabled != enabled) {
            depth_bounds_test_enabled = enabled;
            dirty_state.depth_bounds_test_enabled = true;
        }
    }

    void SetDepthBounds(const float min, const float max) {
        if (depth_bounds_min != min || depth_bounds_max != max) {
            depth_bounds_min = min;
            depth_bounds_max = max;
            dirty_state.depth_bounds = true;
        }
    }

    void SetDepthBiasEnabled(const bool enabled) {
        if (depth_bias_enabled != enabled) {
            depth_bias_enabled = enabled;
            dirty_state.depth_bias_enabled = true;
        }
    }

    void SetDepthBias(const float constant, const float clamp, const float slope) {
        if (depth_bias_constant != constant || depth_bias_clamp != clamp ||
            depth_bias_slope != slope) {
            depth_bias_constant = constant;
            depth_bias_clamp = clamp;
            depth_bias_slope = slope;
            dirty_state.depth_bias = true;
        }
    }

    void SetStencilTestEnabled(const bool enabled) {
        if (stencil_test_enabled != enabled) {
            stencil_test_enabled = enabled;
            dirty_state.stencil_test_enabled = true;
        }
    }

    void SetStencilOps(const StencilOps& front_ops, const StencilOps& back_ops) {
        if (stencil_front_ops != front_ops) {
            stencil_front_ops = front_ops;
            dirty_state.stencil_front_ops = true;
        }
        if (stencil_back_ops != back_ops) {
            stencil_back_ops = back_ops;
            dirty_state.stencil_back_ops = true;
        }
    }

    void SetStencilReferences(const u32 front_reference, const u32 back_reference) {
        if (stencil_front_reference != front_reference) {
            stencil_front_reference = front_reference;
            dirty_state.stencil_front_reference = true;
        }
        if (stencil_back_reference != back_reference) {
            stencil_back_reference = back_reference;
            dirty_state.stencil_back_reference = true;
        }
    }

    void SetStencilWriteMasks(const u32 front_write_mask, const u32 back_write_mask) {
        if (stencil_front_write_mask != front_write_mask) {
            stencil_front_write_mask = front_write_mask;
            dirty_state.stencil_front_write_mask = true;
        }
        if (stencil_back_write_mask != back_write_mask) {
            stencil_back_write_mask = back_write_mask;
            dirty_state.stencil_back_write_mask = true;
        }
    }

    void SetStencilCompareMasks(const u32 front_compare_mask, const u32 back_compare_mask) {
        if (stencil_front_compare_mask != front_compare_mask) {
            stencil_front_compare_mask = front_compare_mask;
            dirty_state.stencil_front_compare_mask = true;
        }
        if (stencil_back_compare_mask != back_compare_mask) {
            stencil_back_compare_mask = back_compare_mask;
            dirty_state.stencil_back_compare_mask = true;
        }
    }

    void SetPrimitiveRestartEnabled(const bool enabled) {
        if (primitive_restart_enable != enabled) {
            primitive_restart_enable = enabled;
            dirty_state.primitive_restart_enable = true;
        }
    }

    void SetCullMode(const vk::CullModeFlags cull_mode_) {
        if (cull_mode != cull_mode_) {
            cull_mode = cull_mode_;
            dirty_state.cull_mode = true;
        }
    }

    void SetFrontFace(const vk::FrontFace front_face_) {
        if (front_face != front_face_) {
            front_face = front_face_;
            dirty_state.front_face = true;
        }
    }

    void SetBlendConstants(const std::array<float, 4> blend_constants_) {
        if (blend_constants != blend_constants_) {
            blend_constants = blend_constants_;
            dirty_state.blend_constants = true;
        }
    }

    void SetRasterizerDiscardEnabled(const bool enabled) {
        if (rasterizer_discard_enable != enabled) {
            rasterizer_discard_enable = enabled;
            dirty_state.rasterizer_discard_enable = true;
        }
    }

    void SetColorWriteMasks(const ColorWriteMasks& color_write_masks_) {
        if (!std::ranges::equal(color_write_masks, color_write_masks_)) {
            color_write_masks = color_write_masks_;
            dirty_state.color_write_masks = true;
        }
    }

    void SetLineWidth(const float width) {
        if (line_width != width) {
            line_width = width;
            dirty_state.line_width = true;
        }
    }

    void SetAttachmentFeedbackLoopEnabled(const bool enabled) {
        if (feedback_loop_enabled != enabled) {
            feedback_loop_enabled = enabled;
            dirty_state.feedback_loop_enabled = true;
        }
    }
};

/// SHADPS4_SKIP_REBIND=1: the draw paths skip vkCmdBindPipeline(graphics) when the same pipeline
/// is already bound in the same command buffer. Updated in recording order (the recording worker,
/// or the back after a drain). Every command-buffer begin and every other graphics bind into a
/// scheduler command buffer (blit helper, post-process pass) invalidates it, so a missed case
/// can only cost an extra bind.
struct GraphicsBindTracker {
    static bool Enabled() {
        static const bool enabled = [] {
            const char* value = std::getenv("SHADPS4_SKIP_REBIND");
            return value != nullptr && value[0] == '1';
        }();
        return enabled;
    }
    static void Invalidate() {
        last_pipeline.store(VkPipeline{}, std::memory_order_relaxed);
    }
    /// True if the draw must bind `pipeline` into `cmdbuf`; records it as bound.
    static bool NeedBind(vk::CommandBuffer cmdbuf, vk::Pipeline pipeline) {
        const VkCommandBuffer cb = cmdbuf;
        const VkPipeline pl = pipeline;
        if ((++binds & 0xfffff) == 0) {
            LOG_INFO(Render_Vulkan, "SkipRebind: {} graphics binds, {} skipped ({:.1f}%)", binds,
                     skipped, 100.0 * double(skipped) / double(binds));
        }
        if (last_cmdbuf.load(std::memory_order_relaxed) == cb &&
            last_pipeline.load(std::memory_order_relaxed) == pl) {
            ++skipped;
            return false;
        }
        last_cmdbuf.store(cb, std::memory_order_relaxed);
        last_pipeline.store(pl, std::memory_order_relaxed);
        return true;
    }
    static inline std::atomic<VkCommandBuffer> last_cmdbuf{};
    static inline std::atomic<VkPipeline> last_pipeline{};
    static inline u64 binds{};   ///< Recording order only (no concurrent writers).
    static inline u64 skipped{};
};

class Scheduler {
public:
    explicit Scheduler(const Instance& instance, bool allow_recording_worker = false);
    ~Scheduler();

    /// Sends the current execution context to the GPU
    /// and increments the scheduler timeline semaphore.
    void Flush(SubmitInfo& info);

    /// Sends the current execution context to the GPU
    /// and increments the scheduler timeline semaphore.
    void Flush();

    /// Sends the current execution context to the GPU and waits for it to complete.
    void Finish();

    /// Waits for the given tick to trigger on the GPU.
    void Wait(u64 tick);
    /// Main timeline only (see IsFreeMain).
    void WaitMain(u64 tick);
    /// The ASC value the batch being flushed must wait (I1 or precise; SHADPS4_ASC_I1).
    u64 AscMainWaitValue(bool no_i1);

    /// Attempts to execute operations whose tick the GPU has caught up with.
    void PopPendingOperations();

    /// Starts a new rendering scope with provided state.
    void BeginRendering(const RenderState& new_state);

    /// Ends current rendering scope. `loc` attributes the break under SHADPS4_GPU_BATCH_TIMING.
    void EndRendering(std::source_location loc = std::source_location::current());

    /// Returns the current render state.
    const RenderState& GetRenderState() const {
        return render_state;
    }

    /// Returns the current pipeline dynamic state tracking.
    DynamicState& GetDynamicState() {
        return dynamic_state;
    }

    // A raw handle may only be used while this scope is alive. Nested recordings run
    // inline, so helpers retaining a handle cannot race the recording worker.
    class CommandBufferAccess {
    public:
        explicit CommandBufferAccess(Scheduler& scheduler, std::source_location location)
            : scheduler{scheduler} {
            if (scheduler.synchronous_scopes == 0) {
                scheduler.DrainRecording();
            }
            // A borrower may bind legacy descriptor state behind the scheduler's back.
            scheduler.native_heaps_bound = false;
            ++scheduler.synchronous_scopes;
            ++scheduler.synchronous_borrows;
            ++scheduler.stream_seq[scheduler.active_stream];
            if (scheduler.recording_measure_timing) {
                ++scheduler.recording_borrow_counts[location.function_name()];
            }
        }
        ~CommandBufferAccess() {
            scheduler.native_heaps_bound = false;
            --scheduler.synchronous_scopes;
        }
        CommandBufferAccess(const CommandBufferAccess&) = delete;
        CommandBufferAccess& operator=(const CommandBufferAccess&) = delete;
        vk::CommandBuffer Handle() const {
            return scheduler.current_cmdbuf;
        }

    private:
        Scheduler& scheduler;
    };

    [[nodiscard]] CommandBufferAccess AcquireCommandBuffer(
        std::source_location location = std::source_location::current()) {
        return CommandBufferAccess{*this, location};
    }

    void DrainRecording();
    /// Hands the recording worker everything recorded so far without waiting.
    void PublishRecording() {
        if (recording_worker) {
            recording_worker->Publish();
        }
    }
    void CommitDynamicState();
    void PushDescriptors(vk::PipelineBindPoint bind_point, vk::PipelineLayout layout,
                         std::span<const vk::WriteDescriptorSet> writes);

    bool IsAsyncRecording() const {
        return recording_worker && synchronous_scopes == 0;
    }
    /// Command-processor thread: moves whenever anything is recorded (worker commands) or the
    /// command buffer is borrowed directly. Equal values = nothing recorded in between.
    u64 RecordSeq() const {
        return (recording_worker ? recording_worker->AcceptedCommandsProducer() : 0) +
               synchronous_borrows;
    }

    /// Descriptor heap binds are invalidated by legacy descriptor state and do not survive a
    /// command buffer. Tracked in recording order on the command-processor thread.
    bool NativeHeapsBound() const {
        return native_heaps_bound && synchronous_scopes == 0;
    }
    void MarkNativeHeapsBound() {
        native_heaps_bound = true;
    }
    void InvalidateNativeHeaps() {
        native_heaps_bound = false;
    }

    template <typename F>
    void Record(F&& command) {
        ++stream_seq[active_stream];
        if (!IsAsyncRecording()) {
            command(current_cmdbuf);
            return;
        }
        // The handle is read when the worker runs the command: with SHADPS4_ASYNC_SUBMIT the
        // worker itself ends, submits and replaces the command buffer, in recording order.
        recording_worker->Record([this, command = std::forward<F>(command)]() mutable {
            command(current_cmdbuf);
        });
    }

    // Clone transient arrays into the same unpublished chunk as their callback.
    // Factories never execute guest-memory reads or reenter the scheduler.
    template <typename F, typename... T>
    void RecordArrays(F&& command, std::span<const T>... arrays) {
        static_assert((std::is_trivially_copyable_v<T> && ...));
        ++stream_seq[active_stream];
        if (!IsAsyncRecording()) {
            command(current_cmdbuf, arrays...);
            return;
        }
        const size_t bytes = (size_t{} + ... + AlignedPayloadSize(arrays.size_bytes()));
        if (bytes > 32768) {
            auto access = AcquireCommandBuffer();
            command(access.Handle(), arrays...);
            return;
        }
        recording_worker->RecordWithPayload(bytes, [&](std::span<std::byte> storage) {
            auto* cursor = storage.data();
            auto copies = std::tuple{CopyPayload(cursor, arrays)...};
            return [this, command = std::forward<F>(command), copies]() mutable {
                std::apply([&](auto... spans) { command(current_cmdbuf, spans...); }, copies);
            };
        });
    }

    /// Returns the current command buffer tick.
    [[nodiscard]] u64 CurrentTick() const noexcept {
        return master_semaphore.CurrentTick();
    }

    /// Returns true when a tick has been triggered by the GPU. With the ASC stream this is the
    /// dual-timeline retirement (R0, notes/asc-i1-relax-design-2026-09-28.md): main tick `tick`
    /// complete AND every ASC segment submitted before that main flush complete. While I1 holds,
    /// main already waited those segments, so the extra check only refreshes the ASC counter.
    [[nodiscard]] bool IsFree(u64 tick) noexcept {
        if (!IsFreeMain(tick)) {
            return false;
        }
        return AscRetired(tick);
    }

    /// Main timeline only (fault jobs whose copy ran on main: they must not inherit ASC waits).
    [[nodiscard]] bool IsFreeMain(u64 tick) noexcept {
        if (master_semaphore.IsFree(tick)) {
            return true;
        }
        master_semaphore.Refresh();
        return master_semaphore.IsFree(tick);
    }

    /// The last ASC value submitted before main tick `tick` was flushed (0: none / ASC off).
    /// Evicted entries answer the newest submitted value (conservative).
    [[nodiscard]] u64 AscAt(u64 tick) const noexcept {
        if (!asc_enabled || tick == 0) {
            return 0;
        }
        const u64 i = tick % AscAtSlots;
        const u64 tag = asc_at_tag[i].load(std::memory_order_acquire);
        if (tag == tick) {
            const u64 value = asc_at_val[i].load(std::memory_order_relaxed);
            std::atomic_thread_fence(std::memory_order_acquire);
            if (asc_at_tag[i].load(std::memory_order_relaxed) == tick) {
                return value;
            }
        }
        if (tag > tick) {
            // Evicted (>= 4096 flushes ago): the newest evicted value is >= the true value and
            // already old; the newest submitted value would stall on current ASC work (Opus review:
            // StreamBuffer ring wraps waited on it, 17% of the CP back).
            ++asc_at_evicted_hits;
            return asc_at_evicted.load(std::memory_order_acquire);
        }
        ++asc_at_fallback_hits; // not flushed yet (callers check main first) or being written
        return asc_last_submitted.load(std::memory_order_acquire);
    }

    [[nodiscard]] bool AscRetired(u64 tick) noexcept {
        const u64 value = AscAt(tick);
        if (value == 0 || asc_semaphore->IsFree(value)) {
            return true;
        }
        asc_semaphore->Refresh();
        ++asc_retired_refreshes;
        const bool free = asc_semaphore->IsFree(value);
        asc_retired_blocked += !free; // R0 falsifier: ~0 while I1 holds
        return free;
    }

    /// Returns the master timeline semaphore.
    [[nodiscard]] MasterSemaphore* GetMasterSemaphore() noexcept {
        return &master_semaphore;
    }

    /// Draws recorded since the last submission, and in the last submitted batch (diagnostic).
    void NoteDraw() noexcept {
        ++batch_draws;
        ++batch_ops;
    }
    /// Draws plus dispatches recorded since the last submission of any kind (SHADPS4_FLUSH_EVERY_OPS);
    /// NoteDraw counts direct draws, NoteOp dispatches and indirect draws.
    void NoteOp() noexcept {
        ++batch_ops;
    }
    /// A compute dispatch: NoteOp plus the per-batch dispatch count (SHADPS4_GPU_BATCH_TIMING).
    void NoteDispatch() noexcept {
        if (active_stream != 0) {
            return; // ASC stream: its segments are cut at guest ASC RELEASE_MEMs.
        }
        ++batch_ops;
        ++batch_dispatches;
    }

    /**
     * SHADPS4_ASC_STREAM=1 (with SHADPS4_ASYNC_SUBMIT=1, SHADPS4_READBACK_QUEUE=3): eligible guest
     * async-compute dispatches are recorded into a second command stream submitted on a
     * compute-only queue (design notes/asc-stream-design-2026-09-28.md). Stream 0 = main,
     * 1 = ASC. Streams switch in recording order (the worker applies a marker), so everything
     * recorded in between lands in that stream's command buffer.
     * I1: every main submission waits on the ASC value last submitted before it, so a main tick
     * completing still implies all work recorded before it completed (tick lifetimes hold).
     * I2: an ASC segment waits on the newest main batch its bindings depend on (AddAscWaitMain);
     * only values already flushed on the CP are ever waited on (no cycle).
     */
    bool HasAscStream() const noexcept {
        return asc_enabled;
    }
    u32 ActiveStream() const noexcept {
        return active_stream;
    }
    void SwitchStream(u32 stream);
    void AddAscWaitMain(u64 main_tick) noexcept {
        asc_wait_main = std::max(asc_wait_main, main_tick);
    }
    /// Submits the open ASC segment (if any) and returns its timeline value, else 0.
    u64 SubmitAscSegment();
    /// Value the open ASC segment will signal (0 if none is open).
    u64 OpenAscValue() const noexcept {
        return asc_dirty ? open_asc_value : 0;
    }
    MasterSemaphore* GetAscSemaphore() noexcept {
        return asc_semaphore.get();
    }
    /// Commands recorded into one stream (switch markers excluded).
    u64 RecordSeqOf(u32 stream) const noexcept {
        return stream_seq[stream];
    }
    /// SHADPS4_GPU_BATCH_TIMING timer, or null (diagnostic).
    [[nodiscard]] class GpuBatchTimer* BatchTimer() const noexcept {
        return batch_timer.get();
    }
    [[nodiscard]] u32 BatchOps() const noexcept {
        return batch_ops;
    }
    [[nodiscard]] u32 LastBatchDraws() const noexcept {
        return last_batch_draws;
    }

    /// Defers an operation until the gpu has reached the current cpu tick.
    /// Will be run when submitting or calling PopPendingOperations.
    void DeferOperation(Common::UniqueFunction<void>&& func) {
        pending_ops.emplace(std::move(func), CurrentTick());
    }

    /// Defers an operation until the gpu has reached the current cpu tick.
    /// Runs as soon as possible in another thread.
    void DeferPriorityOperation(Common::UniqueFunction<void>&& func) {
        {
            std::unique_lock lk(priority_pending_ops_mutex);
            priority_pending_ops.emplace(std::move(func), CurrentTick());
        }
        priority_pending_ops_cv.notify_one();
    }

    static std::mutex submit_mutex;

    /// SHADPS4_ASYNC_SUBMIT: timeline of the scheduler whose submissions the recording worker
    /// makes, and the highest value the worker has actually handed to vkQueueSubmit. A
    /// synchronous submission waiting on that timeline first waits for the value to be
    /// submitted, so no batch on the queue waits for a signal queued behind it.
    static std::atomic<VkSemaphore> async_timeline;
    static std::atomic<u64> async_submitted_tick;

private:
    static constexpr size_t AlignedPayloadSize(size_t bytes) {
        constexpr size_t alignment = alignof(std::max_align_t);
        return (bytes + alignment - 1) & ~(alignment - 1);
    }
    template <typename T>
    static std::span<const T> CopyPayload(std::byte*& cursor, std::span<const T> source) {
        static_assert(alignof(T) <= alignof(std::max_align_t));
        const auto* result = reinterpret_cast<const T*>(cursor);
        if (!source.empty()) {
            std::memcpy(cursor, source.data(), source.size_bytes());
            cursor += AlignedPayloadSize(source.size_bytes());
        }
        return {result, source.size()};
    }

    void AllocateWorkerCommandBuffers();

    void SubmitExecution(SubmitInfo& info);

    void PriorityPendingOpsThread(std::stop_token stoken);

private:
    const Instance& instance;
    std::unique_ptr<Common::CommandRecordingWorker<>> recording_worker;
    bool recording_measure_timing = false;
    std::unordered_map<std::string_view, u64> recording_borrow_counts;
    u32 synchronous_scopes = 0;
    bool native_heaps_bound = false;
    u64 synchronous_borrows = 0;
    u64 recording_submits = 0;
    bool async_submit = false;
    std::unique_ptr<class GpuBatchTimer> batch_timer;
    void SubmitExecutionAsync(SubmitInfo& info);
    u32 batch_draws = 0;
    u32 batch_dispatches = 0;
    // SHADPS4_ASC_STREAM state. Worker side (or the CP while drained): main_cmdbuf, asc_cmdbuf,
    // worker_stream. CP side: the rest.
    bool asc_enabled = false;
    std::unique_ptr<MasterSemaphore> asc_semaphore;
    std::unique_ptr<CommandPool> asc_pool;
    static constexpr u64 AscAtSlots = 4096;
    std::array<std::atomic<u64>, AscAtSlots> asc_at_tag{};
    std::array<std::atomic<u64>, AscAtSlots> asc_at_val{};
    std::atomic<u64> asc_at_evicted{0};
    mutable std::atomic<u64> asc_at_evicted_hits{0};
    mutable std::atomic<u64> asc_at_fallback_hits{0};
    std::atomic<u64> asc_retired_refreshes{0};
    std::atomic<u64> asc_retired_blocked{0};
    void NoteAscAtFlush(u64 main_tick) noexcept {
        if (!asc_enabled) {
            return;
        }
        const u64 i = main_tick % AscAtSlots;
        if (asc_at_tag[i].load(std::memory_order_relaxed) != 0) {
            // Slots are overwritten in tick order, so the evicted values are monotonic.
            asc_at_evicted.store(asc_at_val[i].load(std::memory_order_relaxed),
                                 std::memory_order_release);
        }
        asc_at_tag[i].store(0, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_release);
        asc_at_val[i].store(asc_last_submitted.load(std::memory_order_relaxed),
                            std::memory_order_relaxed);
        asc_at_tag[i].store(main_tick, std::memory_order_release);
        if ((main_tick & 0xffff) == 0) {
            LOG_INFO(Render_Vulkan,
                     "AscR0: flushes={} evicted_hits={} fallback_hits={} retired_refreshes={} "
                     "retired_blocked={}",
                     main_tick, asc_at_evicted_hits.load(), asc_at_fallback_hits.load(),
                     asc_retired_refreshes.load(), asc_retired_blocked.load());
        }
    }
    vk::CommandBuffer main_cmdbuf;
    vk::CommandBuffer asc_cmdbuf;
    u32 worker_stream = 0;
    u32 active_stream = 0;
    bool asc_dirty = false;
    u64 open_asc_value = 0;
    u64 asc_wait_main = 0;
    std::atomic<u64> asc_last_submitted{0};
    std::array<u64, 4> asc_wait_dist{};
    u64 asc_waited_by_main = 0;
    std::array<u64, 2> stream_seq{};
    void AscSubmitOnWorker(u64 value, u64 wait_main);
    void BeginAscCmdbufOnWorker();

    u32 last_batch_draws = 0;
    u32 batch_ops = 0;
    MasterSemaphore master_semaphore;
    CommandPool command_pool;
    DynamicState dynamic_state;
    vk::CommandBuffer current_cmdbuf;
    std::condition_variable_any event_cv;
    struct PendingOp {
        Common::UniqueFunction<void> callback;
        u64 gpu_tick;
    };
    std::queue<PendingOp> pending_ops;
    std::queue<PendingOp> priority_pending_ops;
    std::mutex priority_pending_ops_mutex;
    std::condition_variable_any priority_pending_ops_cv;
    std::jthread priority_pending_ops_thread;
    RenderState render_state;
    bool is_rendering = false;
    u32 last_end_site = ~0u;
    tracy::VkCtxScope* profiler_scope{};
};

} // namespace Vulkan
