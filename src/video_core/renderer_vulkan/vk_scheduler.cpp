// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstdlib>
#include <cstring>

#include "common/assert.h"
#include "common/debug.h"
#include "common/fault_monitor.h"
#include "common/thread.h"
#include "imgui/renderer/texture_manager.h"
#include "video_core/renderer_vulkan/vk_draw_packet.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

#include <thread>
#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#endif
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
        const char* async = std::getenv("SHADPS4_ASYNC_SUBMIT");
        async_submit = async && std::string_view{async} == "1";
        // SHADPS4_PAR_REC lanes need the worker to submit (a submission ends a batch); the ASC
        // stream (two command buffers switched on the worker) and the batch timer (one query
        // pool) assume one recorder.
        par_rec = ParRecRequested();
        if (par_rec != 0) {
            const char* why = !async_submit                  ? "SHADPS4_ASYNC_SUBMIT is off"
                              : instance.HasAscQueue()       ? "the ASC stream is on"
                              : GpuBatchTimer::Enabled()     ? "SHADPS4_GPU_BATCH_TIMING is on"
                                                             : nullptr;
            if (why) {
                LOG_WARNING(Render_Vulkan, "ParRec: {} lanes requested, off because {}", par_rec,
                            why);
                par_rec = 0;
            }
        }
        g_par_rec_lanes = par_rec;
        for (u32 i = 0; i < par_rec; ++i) {
            rec_lanes[i].pool = std::make_unique<CommandPool>(instance, &master_semaphore);
        }
        recording_worker = std::make_unique<Common::CommandRecordingWorker<>>(
            recording_measure_timing, par_rec,
            [this](std::size_t lane) {
                t_rec_lane = &rec_lanes[lane];
                Common::SetCurrentThreadName(
                    lane == 0 ? "shadPS4:VkRecorder"
                              : (lane == 1 ? "shadPS4:VkRecorder1"
                                           : (lane == 2 ? "shadPS4:VkRecorder2"
                                                        : "shadPS4:VkRecorder3")));
                if (lane == 0) {
                    Common::ApplyHotThreadPlacement(Common::HotThread::Recorder);
                }
            },
            [this] {
                rec_submitted.fetch_or(RecAbortBit, std::memory_order_acq_rel);
                rec_submitted.notify_all();
            });
        if (par_rec == 0) {
            recording_worker->Record([] {
                Common::SetCurrentThreadName("shadPS4:VkRecorder");
                Common::ApplyHotThreadPlacement(Common::HotThread::Recorder);
            });
        }
        Common::CommandRecordingWorker<>::SetChunkHook(
            [] { Common::SampleThreadPlacement("recorder", 2); });
        LOG_INFO(Render_Vulkan, "AsyncRecording: enabled, ordered primary command worker, {} chunks",
                 recording_worker->ActiveChunks());
        if (par_rec != 0) {
            LOG_INFO(Render_Vulkan,
                     "ParRec: {} recording lanes (batch k on lane k % {}, submitted in batch order)",
                     par_rec, par_rec);
        }
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
        if (const char* t = std::getenv("SHADPS4_ASC_TIMING"); t && t[0] == '1') {
            asc_timer = std::make_unique<AscTimer>();
            asc_timer->pool = instance.GetDevice()
                                  .createQueryPoolUnique(vk::QueryPoolCreateInfo{
                                      .queryType = vk::QueryType::eTimestamp,
                                      .queryCount = AscTimer::Slots * 2,
                                  })
                                  .value;
            asc_timer->period_ns = instance.GetPhysicalDevice().getProperties().limits.timestampPeriod;
            asc_timer->window = std::chrono::steady_clock::now();
        }
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

bool Scheduler::TransferCoalesceEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("SHADPS4_TRANSFER_COALESCE");
        return value && value[0] == '1';
    }();
    return enabled;
}

namespace {
struct TransferCoalesceStats {
    u64 ops = 0, groups = 0, tt_barriers = 0, overflow = 0;
};
// Per recorder thread (SHADPS4_PAR_REC lanes record at once); log-only counters.
thread_local TransferCoalesceStats g_tcs;

bool RangesOverlap(const Scheduler::TransferRange& a, const Scheduler::TransferRange& b) {
    return a.buffer == b.buffer && a.lo < b.hi && b.lo < a.hi;
}
} // namespace

void Scheduler::BeginTransferOnWorker(vk::CommandBuffer cmdbuf, const TransferRange& read,
                                      const TransferRange& write) {
    TransferGroup& group = RecorderTransferGroup();
    ++g_tcs.ops;
    if (!group.open) {
        // Everything recorded before the group -> the group's transfers (was: one such barrier
        // per op, scoped to its buffer; global is a superset).
        const vk::MemoryBarrier2 pre = {
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .dstAccessMask =
                vk::AccessFlagBits2::eTransferRead | vk::AccessFlagBits2::eTransferWrite,
        };
        cmdbuf.pipelineBarrier2(vk::DependencyInfo{
            .memoryBarrierCount = 1,
            .pMemoryBarriers = &pre,
        });
        group.open = true;
        group.num_reads = 0;
        group.num_writes = 0;
        ++g_tcs.groups;
    } else {
        bool hazard = group.num_reads == group.reads.size() ||
                      group.num_writes == group.writes.size();
        g_tcs.overflow += hazard;
        for (u32 i = 0; i < group.num_writes && !hazard; ++i) {
            hazard = (write.buffer && RangesOverlap(write, group.writes[i])) ||
                     (read.buffer && RangesOverlap(read, group.writes[i]));
        }
        for (u32 i = 0; i < group.num_reads && !hazard; ++i) {
            hazard = write.buffer && RangesOverlap(write, group.reads[i]);
        }
        if (hazard) {
            // Transfers of one group run unordered: order this op after the earlier ones.
            const vk::MemoryBarrier2 tt = {
                .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
                .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
                .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
                .dstAccessMask =
                    vk::AccessFlagBits2::eTransferRead | vk::AccessFlagBits2::eTransferWrite,
            };
            cmdbuf.pipelineBarrier2(vk::DependencyInfo{
                .memoryBarrierCount = 1,
                .pMemoryBarriers = &tt,
            });
            group.num_reads = 0;
            group.num_writes = 0;
            ++g_tcs.tt_barriers;
        }
    }
    if (read.buffer) {
        group.reads[group.num_reads++] = read;
    }
    if (write.buffer) {
        group.writes[group.num_writes++] = write;
    }
    if ((g_tcs.ops & 0x3ffff) == 0) {
        LOG_INFO(Render_Vulkan, "TransferCoalesce: ops={} groups={} tt_barriers={} overflow={}",
                 g_tcs.ops, g_tcs.groups, g_tcs.tt_barriers, g_tcs.overflow);
    }
}

void Scheduler::CloseTransferGroupSlow(TransferGroup& group, vk::CommandBuffer cmdbuf) {
    group.open = false;
    // The group's transfers -> everything recorded after it (was: one such barrier per op).
    const vk::MemoryBarrier2 post = {
        .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &post,
    });
}

void Scheduler::DrainRecording() {
    if (recording_worker) {
        recording_worker->Drain();
    }
}

void Scheduler::LogRecordingWindow() {
    if (!recording_worker) {
        return;
    }
    static Common::CommandRecordingWorker<>::Counters prev{};
    const auto now = recording_worker->GetCounters();
    LOG_INFO(Render_Vulkan,
             "AsyncRecordingWindow: timing={} commands={} chunks={} bytes={} worker_ms={:.1f} "
             "producer_drain_wait_ms={:.1f} producer_space_wait_ms={:.1f} active_chunks={}",
             recording_measure_timing, now.commands - prev.commands, now.chunks - prev.chunks,
             now.bytes - prev.bytes, double(now.worker_ns - prev.worker_ns) / 1e6,
             double(now.drain_ns - prev.drain_ns) / 1e6,
             double(now.space_wait_ns - prev.space_wait_ns) / 1e6, recording_worker->ActiveChunks());
    prev = now;
}

void Scheduler::CommitDynamicState() {
    if (!IsAsyncRecording()) {
        Rc().dynamic_state.Commit(instance, current_cmdbuf);
        return;
    }
    // SHADPS4_SKIP_EMPTY_DYNSTATE=1: do not record a ~0.8 KB DynamicState snapshot when Commit
    // would emit nothing. Commit clears exactly the bits it emits, so it emits nothing iff the
    // postponed set computed below equals the current dirty set (padding bits only make the
    // comparison conservative). =2 also logs the skip rate every 2^20 calls.
    const int skip_empty = g_skip_empty_dynstate;
    auto& state = Rc().dynamic_state;
    const auto old_dirty = state.dirty_state;
    // Keep the conditional dirtiness which Commit deliberately postpones until
    // its corresponding test is enabled. All values in the packet are owned.
    const DynamicDirtyBits kept = PostponedDynamicDirty();
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

Scheduler::DynamicDirtyBits Scheduler::PostponedDynamicDirty() const {
    const auto& state = Rc().dynamic_state;
    const auto old_dirty = state.dirty_state;
    DynamicDirtyBits kept{};
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
    return kept;
}

bool Scheduler::CommitDynamicStateWouldRecord() const {
    if (!IsAsyncRecording() || g_skip_empty_dynstate <= 0) {
        return true;
    }
    // CommitDynamicState's own skip test, on the same inputs.
    const DynamicDirtyBits kept = PostponedDynamicDirty();
    return std::memcmp(&kept, &Rc().dynamic_state.dirty_state, sizeof(kept)) != 0;
}

void Scheduler::RecordDrawPacket(const DrawPacketDesc& desc) {
    ASSERT_MSG(IsAsyncRecording(), "Draw packets need the recording worker");
    ++Rc().stream_seq[active_stream];
    if (desc.header.num_writes != 0) {
        Rc().native_heaps_bound = false; // as PushDescriptors
    }
    std::span<const PacketBinding> table = desc.table;
    std::span<const vk::DescriptorBufferInfo> buffer_infos = desc.buffer_infos;
    std::span<const vk::DescriptorImageInfo> image_infos = desc.image_infos;
    bool keep_descriptors = false;
    // SHADPS4_DRAW_PACKET_DELTA: see scripts/patch/draw-packet-delta-2026-10-02.py. Graphics
    // packets only; this thread recorded everything since the previous packet, and every other
    // graphics descriptor push sets packet_state_unknown.
    static const bool packet_delta = [] {
        const char* value = std::getenv("SHADPS4_DRAW_PACKET_DELTA");
        return value && value[0] == '1';
    }();
    if (packet_delta && desc.header.bind_point == vk::PipelineBindPoint::eGraphics &&
        !table.empty()) {
        auto& ps = Rc().packet_push;
        ++ps.packets;
        const VkPipelineLayout layout = desc.header.layout;
        if (Rc().dynamic_state.packet_state_unknown) {
            ++ps.unknown;
        } else if (ps.layout != layout) {
            ++ps.layout_changed;
        } else {
            static_assert(sizeof(PacketBinding) == sizeof(u64));
            bool same = ps.table.size() == table.size() &&
                        ps.buffer_infos.size() == buffer_infos.size() &&
                        ps.image_infos.size() == image_infos.size() &&
                        std::memcmp(ps.table.data(), table.data(), table.size_bytes()) == 0;
            for (size_t i = 0; same && i < buffer_infos.size(); ++i) {
                const auto& a = ps.buffer_infos[i];
                const auto& b = buffer_infos[i];
                same = a.buffer == b.buffer && a.offset == b.offset && a.range == b.range;
            }
            for (size_t i = 0; same && i < image_infos.size(); ++i) {
                const auto& a = ps.image_infos[i];
                const auto& b = image_infos[i];
                same = a.sampler == b.sampler && a.imageView == b.imageView &&
                       a.imageLayout == b.imageLayout;
            }
            keep_descriptors = same;
            ps.differs += !same;
        }
        if (keep_descriptors) {
            ++ps.kept;
            table = {};
            buffer_infos = {};
            image_infos = {};
        } else {
            ps.layout = layout;
            ps.table.resize(table.size());
            std::memcpy(ps.table.data(), table.data(), table.size_bytes());
            ps.buffer_infos.assign(buffer_infos.begin(), buffer_infos.end());
            ps.image_infos.assign(image_infos.begin(), image_infos.end());
            Rc().dynamic_state.packet_state_unknown = false;
        }
        if ((ps.packets & 0x1fffff) == 0) {
            LOG_INFO(Render_Vulkan,
                     "DrawPacketDelta: packets={} kept={} ({:.1f}%) unknown={} "
                     "layout_changed={} differs={}",
                     ps.packets, ps.kept, 100.0 * ps.kept / ps.packets, ps.unknown,
                     ps.layout_changed, ps.differs);
            ps.packets = ps.kept = ps.unknown = ps.layout_changed = ps.differs = 0;
        }
    }
    const size_t bytes = DrawPacketPayloadBytes(desc.barriers.size(), table.size(),
                                                buffer_infos.size(), image_infos.size());
    ASSERT_MSG(bytes <= DrawPacketMaxBytes, "Draw packet of {} bytes", bytes);
    WorkerRecordWithPayload(bytes, [&](std::span<std::byte> storage) {
        std::byte* cursor = storage.data();
        auto* header = reinterpret_cast<DrawPacketHeader*>(cursor);
        std::memcpy(header, &desc.header, sizeof(DrawPacketHeader));
        if (keep_descriptors) {
            header->flags |= DrawPacketKeepDescriptors;
            header->num_writes = 0;
            header->num_buffer_infos = 0;
            header->num_image_infos = 0;
        }
        cursor += DrawPacketAlign(sizeof(DrawPacketHeader));
        // The same layout ViewDrawPacket reads back (an empty span advances nothing).
        CopyPayload(cursor, desc.barriers);
        CopyPayload(cursor, table);
        CopyPayload(cursor, buffer_infos);
        CopyPayload(cursor, image_infos);
        return [this, header] { ExecuteDrawPacket(*header); };
    });
}

/// SHADPS4_DRAW_PACKET_STATS=1 (recorder thread only): how much of a packet repeats the packet
/// before it in the same command buffer. A census: it copies the packet's tables.
static void DrawPacketRecorderStats(vk::CommandBuffer cmdbuf, const DrawPacketHeader& header,
                                    const DrawPacketView& view) {
    struct State {
        VkCommandBuffer cmdbuf{};
        VkPipelineLayout layout{};
        VkPipeline pipeline{};
        std::vector<PacketBinding> table;
        std::vector<vk::DescriptorBufferInfo> buffers;
        std::vector<vk::DescriptorImageInfo> images;
        u64 packets{}, same_layout{}, same_pipeline{}, writes{}, writes_in_same{}, writes_equal{};
        u64 all_equal{}, push_data_equal{};
        std::array<u8, sizeof(header.push_data)> push{};
    };
    static State st;
    ++st.packets;
    st.writes += header.num_writes;
    const bool table_same =
        st.cmdbuf == static_cast<VkCommandBuffer>(cmdbuf) &&
        st.layout == static_cast<VkPipelineLayout>(header.layout) &&
        st.table.size() == view.table.size() &&
        (view.table.empty() || std::memcmp(st.table.data(), view.table.data(),
                                           view.table.size_bytes()) == 0) &&
        st.buffers.size() == view.buffer_infos.size() &&
        st.images.size() == view.image_infos.size();
    st.same_pipeline += st.cmdbuf == static_cast<VkCommandBuffer>(cmdbuf) &&
                        st.pipeline == static_cast<VkPipeline>(header.pipeline);
    if (table_same) {
        ++st.same_layout;
        st.writes_in_same += header.num_writes;
        size_t bi = 0, ii = 0;
        u32 equal = 0;
        for (const PacketBinding& binding : view.table) {
            const auto type = static_cast<vk::DescriptorType>(binding.type);
            bool same = true;
            if (IsPacketBufferType(type)) {
                for (u32 k = 0; k < binding.count; ++k, ++bi) {
                    const auto& a = st.buffers[bi];
                    const auto& b = view.buffer_infos[bi];
                    same &= a.buffer == b.buffer && a.offset == b.offset && a.range == b.range;
                }
            } else {
                for (u32 k = 0; k < binding.count; ++k, ++ii) {
                    const auto& a = st.images[ii];
                    const auto& b = view.image_infos[ii];
                    same &= a.sampler == b.sampler && a.imageView == b.imageView &&
                            a.imageLayout == b.imageLayout;
                }
            }
            equal += same;
        }
        st.writes_equal += equal;
        st.all_equal += equal == header.num_writes;
        st.push_data_equal +=
            std::memcmp(st.push.data(), &header.push_data, sizeof(header.push_data)) == 0;
    }
    st.cmdbuf = static_cast<VkCommandBuffer>(cmdbuf);
    st.layout = static_cast<VkPipelineLayout>(header.layout);
    st.pipeline = static_cast<VkPipeline>(header.pipeline);
    st.table.assign(view.table.begin(), view.table.end());
    st.buffers.assign(view.buffer_infos.begin(), view.buffer_infos.end());
    st.images.assign(view.image_infos.begin(), view.image_infos.end());
    std::memcpy(st.push.data(), &header.push_data, sizeof(header.push_data));
    if ((st.packets & 0x1fffff) == 0) {
        LOG_INFO(Render_Vulkan,
                 "DrawPacketRec: packets={} writes/packet={:.2f} same_layout={:.1f}% "
                 "same_pipeline={:.1f}% | in same-layout packets: writes equal={:.1f}% "
                 "all writes equal={:.1f}% push constants equal={:.1f}%",
                 st.packets, static_cast<double>(st.writes) / st.packets,
                 100.0 * st.same_layout / st.packets, 100.0 * st.same_pipeline / st.packets,
                 st.writes_in_same ? 100.0 * st.writes_equal / st.writes_in_same : 0.0,
                 st.same_layout ? 100.0 * st.all_equal / st.same_layout : 0.0,
                 st.same_layout ? 100.0 * st.push_data_equal / st.same_layout : 0.0);
        st.packets = st.same_layout = st.same_pipeline = st.writes = st.writes_in_same =
            st.writes_equal = st.all_equal = st.push_data_equal = 0;
    }
}

void Scheduler::ExecuteDrawPacket(const DrawPacketHeader& header) {
    // Recording-worker thread. Order = the legacy records: barrier, push constants, push
    // descriptors, then (when folded) bind + draw / dispatch.
    const vk::CommandBuffer cmdbuf = RecorderCmdbuf();
    CloseTransferGroup(cmdbuf);
    const DrawPacketView view = ViewDrawPacket(header);
    if (header.num_barriers != 0) {
        cmdbuf.pipelineBarrier2(vk::DependencyInfo{
            .dependencyFlags = vk::DependencyFlagBits::eByRegion,
            .bufferMemoryBarrierCount = header.num_barriers,
            .pBufferMemoryBarriers = view.barriers.data(),
        });
    }
    cmdbuf.pushConstants(header.layout, header.stage_flags, 0u, sizeof(header.push_data),
                         &header.push_data);
    static const bool packet_stats = [] {
        const char* value = std::getenv("SHADPS4_DRAW_PACKET_STATS");
        return value && value[0] == '1';
    }();
    if (packet_stats) [[unlikely]] {
        DrawPacketRecorderStats(cmdbuf, header, view);
    }
    if (header.num_writes != 0) {
        // Not value-initialized: BuildDrawPacketWrites assigns every write it returns (the
        // constructors of 64 writes per packet were 5.6% of the recorder, 2026-10-02).
        std::aligned_storage_t<sizeof(vk::WriteDescriptorSet) * DrawPacketMaxWrites,
                               alignof(vk::WriteDescriptorSet)>
            write_storage;
        const std::span<vk::WriteDescriptorSet> writes{
            reinterpret_cast<vk::WriteDescriptorSet*>(&write_storage), DrawPacketMaxWrites};
        const u32 count =
            BuildDrawPacketWrites(view.table, view.buffer_infos, view.image_infos, writes);
        ASSERT_MSG(count == header.num_writes, "Draw packet rebuilt {} of {} writes", count,
                   header.num_writes);
        cmdbuf.pushDescriptorSetKHR(header.bind_point, header.layout, 0,
                                    std::span<const vk::WriteDescriptorSet>{writes.data(), count});
    }
    if (header.flags & DrawPacketDraw) {
        if (!GraphicsBindTracker::Enabled() ||
            GraphicsBindTracker::NeedBind(cmdbuf, header.pipeline)) {
            cmdbuf.bindPipeline(vk::PipelineBindPoint::eGraphics, header.pipeline);
        }
        if (header.flags & DrawPacketIndexed) {
            cmdbuf.drawIndexed(header.count, header.instances, 0, s32(header.vertex_offset),
                               header.instance_offset);
        } else {
            cmdbuf.draw(header.count, header.instances, header.vertex_offset,
                        header.instance_offset);
        }
    } else if (header.flags & DrawPacketDispatch) {
        cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, header.pipeline);
        cmdbuf.dispatch(header.dim_x, header.dim_y, header.dim_z);
    }
}

void Scheduler::PushDescriptors(vk::PipelineBindPoint bind_point, vk::PipelineLayout layout,
                                std::span<const vk::WriteDescriptorSet> writes) {
    Rc().native_heaps_bound = false;
    Rc().dynamic_state.packet_state_unknown = true; // a push outside the draw packets
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
    WorkerRecordWithPayload(bytes, [&](std::span<std::byte> storage) {
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
            RecorderCmdbuf().pushDescriptorSetKHR(
                bind_point, layout, 0, std::span<const vk::WriteDescriptorSet>{copies, count});
        };
    });
}

void Scheduler::BeginRendering(const RenderState& new_state) {
    if (Rc().is_rendering && Rc().render_state == new_state) {
        return;
    }
    if (batch_timer && !Rc().is_rendering && Rc().last_end_site != ~0u &&
        Rc().render_state == new_state) {
        GpuBreakSites().Resumed(Rc().last_end_site);
    }
    EndRendering();
    Rc().is_rendering = true;
    if (batch_timer) {
        GpuBreaks().render_begins.fetch_add(1, std::memory_order_relaxed);
        Rc().last_end_site = ~0u;
    }
    Rc().render_state = new_state;

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
    if (!Rc().is_rendering) {
        return;
    }
    Rc().is_rendering = false;
    if (batch_timer) {
        Rc().last_end_site = GpuBreakSites().Site(loc);
        GpuBreakSites().End(Rc().last_end_site);
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
                     "slow_checks={} hook_miss={} whole_buffers={} raises whole/entry={}/{} "
                     "segment_wait_distance[none/1/2-3/4+]={}/{}/{}/{} top:{}",
                     mode, flushes, i1_waits, precise_waits, VideoCore::g_asc_dep_slow,
                     VideoCore::g_asc_hook_miss, VideoCore::g_asc_whole_buffers,
                     VideoCore::g_asc_raise_whole, VideoCore::g_asc_raise_entry, asc_wait_dist[0],
                     asc_wait_dist[1], asc_wait_dist[2], asc_wait_dist[3], [] {
                         std::string top;
                         if (!VideoCore::g_asc_raise_by_base) {
                             return top;
                         }
                         std::vector<std::pair<u64, u64>> v(VideoCore::g_asc_raise_by_base->begin(),
                                                            VideoCore::g_asc_raise_by_base->end());
                         std::sort(v.begin(), v.end(),
                                   [](const auto& a, const auto& b) { return a.second > b.second; });
                         for (size_t k = 0; k < std::min<size_t>(v.size(), 5); ++k) {
                             top += fmt::format(" {:#x}={}", v[k].first, v[k].second);
                         }
                         return top;
                     }());
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
    if (par_rec != 0) {
        // Every lane always holds one begun command buffer (startup; after a synchronous
        // submission, the lane that submitted). The open batch's is borrowed while drained.
        for (u32 i = 0; i < par_rec; ++i) {
            RecLane& lane = rec_lanes[i];
            if (!lane.cmdbuf) {
                lane.cmdbuf = lane.pool->CommitInFlight(lane.cmdbuf_slot);
                Check(lane.cmdbuf.begin(begin_info));
            }
        }
        current_cmdbuf = OpenLane().cmdbuf;
        GraphicsBindTracker::Invalidate();
        Rc().dynamic_state.Invalidate();
        Rc().native_heaps_bound = false;
        return;
    }

    main_cmdbuf = command_pool.Commit();
    Check(main_cmdbuf.begin(begin_info));
    current_cmdbuf = worker_stream == 1 ? asc_cmdbuf : main_cmdbuf;
    GraphicsBindTracker::Invalidate();

    // Invalidate dynamic state so it gets applied to the new command buffer.
    Rc().dynamic_state.Invalidate();
    Rc().native_heaps_bound = false;

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
    const u32 draws = Rc().batch_draws;
    const u32 dispatches = Rc().batch_dispatches;
    Rc().batch_draws = 0;
    Rc().batch_dispatches = 0;
    Rc().batch_ops = 0;
    if (par_rec != 0) {
        // SHADPS4_PAR_REC: this batch's lane ends, submits (in batch order) and replaces its own
        // command buffer; the batch end routes everything recorded after it to the next lane.
        const u64 batch = recording_worker->ProducerBatch();
        recording_worker->Record([this, info, signal_value, batch] {
            SubmitLaneOnWorker(info, signal_value, batch);
        });
        recording_worker->Publish(true);
        Rc().dynamic_state.Invalidate();
        Rc().native_heaps_bound = false;
        master_semaphore.Refresh();
        PopPendingOperations();
        if (++rec_log_batches >= 16384) {
            LogParRec();
        }
        return;
    }
    recording_worker->Record([this, info, signal_value, draws, dispatches]() mutable {
        if (transfer_group[0].open) {
            CloseTransferGroupSlow(transfer_group[0], main_cmdbuf);
        }
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
            // SHADPS4_DEBUG_SUBMIT_STALL_MS (diagnostic): one artificial recorder stall, see
            // scripts/patch/debug-submit-stall-2026-10-02.py.
            static const u32 stall_ms = [] {
                const char* value = std::getenv("SHADPS4_DEBUG_SUBMIT_STALL_MS");
                return value ? static_cast<u32>(std::strtoul(value, nullptr, 10)) : 0u;
            }();
            if (stall_ms != 0) [[unlikely]] {
                static const u32 at_s = [] {
                    const char* value = std::getenv("SHADPS4_DEBUG_SUBMIT_STALL_AT_S");
                    return value ? static_cast<u32>(std::strtoul(value, nullptr, 10)) : 10u;
                }();
                static const auto first_submit = std::chrono::steady_clock::now();
                static bool stalled = false;
                if (!stalled && std::chrono::steady_clock::now() - first_submit >=
                                    std::chrono::seconds(at_s)) {
                    stalled = true;
                    LOG_WARNING(Render_Vulkan,
                                "DebugSubmitStall: the recorder sleeps {} ms before submitting "
                                "tick {}",
                                stall_ms, signal_value);
                    std::this_thread::sleep_for(std::chrono::milliseconds(stall_ms));
                    LOG_WARNING(Render_Vulkan, "DebugSubmitStall: awake, submitting");
                }
            }
            std::scoped_lock lk{submit_mutex};
            const auto result = instance.GetGraphicsQueue().submit(submit_info, info.fence);
            if (result == vk::Result::eErrorDeviceLost) {
                instance.LogDeviceFault();
            }
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
    Rc().dynamic_state.Invalidate();
    Rc().native_heaps_bound = false;
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
    if (par_rec != 0) {
        SubmitExecutionLaneSync(info);
        return;
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
    last_batch_draws = Rc().batch_draws;
    Rc().batch_draws = 0;
    Rc().batch_dispatches = 0;
    Rc().batch_ops = 0;

#if TRACY_GPU_ENABLED
    auto* profiler_ctx = instance.GetProfilerContext();
    if (profiler_ctx) {
        profiler_scope->~VkCtxScope();
        TracyVkCollect(profiler_ctx, current_cmdbuf);
    }
#endif

    EndRendering();
    if (transfer_group[0].open) {
        CloseTransferGroupSlow(transfer_group[0], main_cmdbuf);
    }
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
    if (submit_result == vk::Result::eErrorDeviceLost) {
        instance.LogDeviceFault();
    }
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

bool Scheduler::WaitRecTurn(u64 batch, bool submit) {
    u64 value = rec_submitted.load(std::memory_order_acquire);
    if (value == batch) {
        return true;
    }
    using Clock = std::chrono::steady_clock;
    const auto start = recording_measure_timing ? Clock::now() : Clock::time_point{};
    for (u32 spins = 0; value != batch; ++spins) {
        if (value & RecAbortBit) {
            return false;
        }
        ASSERT_MSG(value < batch, "ParRec: batch {} waits, {} batches already submitted", batch,
                   value);
        if (spins < 64) {
#if defined(_M_X64) || defined(__x86_64__)
            _mm_pause();
#endif
        } else {
            rec_submitted.wait(value, std::memory_order_acquire);
        }
        value = rec_submitted.load(std::memory_order_acquire);
    }
    if (RecLane* const lane = t_rec_lane) {
        (submit ? lane->submit_waits : lane->ordered_waits).fetch_add(1, std::memory_order_relaxed);
        if (recording_measure_timing) {
            lane->wait_ns.fetch_add(
                static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                     Clock::now() - start)
                                     .count()),
                std::memory_order_relaxed);
        }
    }
    return true;
}

void Scheduler::SubmitLaneCmdbufLocked(RecLane& lane, const SubmitInfo& info, u64 signal_value) {
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
        .pCommandBuffers = &lane.cmdbuf,
        .signalSemaphoreCount = info.num_signal_semas,
        .pSignalSemaphores = info.signal_semas.data(),
    };
    const auto result = instance.GetGraphicsQueue().submit(submit_info, info.fence);
    if (result == vk::Result::eErrorDeviceLost) {
        instance.LogDeviceFault();
    }
    ASSERT_MSG(result != vk::Result::eErrorDeviceLost, "Device lost during submit");
    async_submitted_tick.store(signal_value, std::memory_order_release);
    // Reusable once the GPU signals this submission (it was tagged in flight until now).
    lane.pool->Retire(lane.cmdbuf_slot, signal_value);
    lane.submits.fetch_add(1, std::memory_order_relaxed);
}

void Scheduler::SubmitLaneOnWorker(const SubmitInfo& info, u64 signal_value, u64 batch) {
    RecLane* const lane_ptr = t_rec_lane;
    ASSERT_MSG(lane_ptr != nullptr, "ParRec: a lane submission ran off a lane thread");
    RecLane& lane = *lane_ptr;
    if (lane.transfer_group.open) {
        CloseTransferGroupSlow(lane.transfer_group, lane.cmdbuf);
    }
    Check(lane.cmdbuf.end());
    // Queue order == timeline order == batch order: wait until every earlier batch is submitted.
    if (!WaitRecTurn(batch, true)) {
        return;
    }
    {
        std::scoped_lock lk{submit_mutex};
        SubmitLaneCmdbufLocked(lane, info, signal_value);
    }
    rec_submitted.fetch_add(1, std::memory_order_acq_rel);
    rec_submitted.notify_all();
    // The lane's next batch is batch + lanes; its signal value is unknown until then.
    lane.cmdbuf = lane.pool->CommitInFlight(lane.cmdbuf_slot);
    Check(lane.cmdbuf.begin(vk::CommandBufferBeginInfo{
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
    }));
    GraphicsBindTracker::Invalidate();
}

void Scheduler::SubmitExecutionLaneSync(SubmitInfo& info) {
    // Inside a borrow scope: SubmitExecution drained every lane, nothing new was published.
    std::scoped_lock lk{submit_mutex};
    Common::FaultMonitor::CountGpuSubmit();
    EndRendering();
    DrainRecording();
    const u64 signal_value = master_semaphore.NextTick();
    NoteAscAtFlush(signal_value);
    last_batch_draws = Rc().batch_draws;
    Rc().batch_draws = 0;
    Rc().batch_dispatches = 0;
    Rc().batch_ops = 0;
    const u64 batch = recording_worker->ProducerBatch();
    ASSERT_MSG(rec_submitted.load(std::memory_order_acquire) == batch,
               "ParRec: synchronous submission of batch {} with {} submitted", batch,
               rec_submitted.load());
    RecLane& lane = OpenLane();
    if (lane.transfer_group.open) {
        CloseTransferGroupSlow(lane.transfer_group, lane.cmdbuf);
    }
    Check(lane.cmdbuf.end());
    info.AddSignal(master_semaphore.Handle(), signal_value);
    ImGui::Core::TextureManager::Submit();
    SubmitLaneCmdbufLocked(lane, info, signal_value);
    rec_submitted.fetch_add(1, std::memory_order_acq_rel);
    rec_submitted.notify_all();
    lane.cmdbuf = vk::CommandBuffer{};
    recording_worker->Publish(true); // what is recorded from now on belongs to the next batch
    master_semaphore.Refresh();
    AllocateWorkerCommandBuffers(); // begins this lane's next buffer, borrows the next lane's
    PopPendingOperations();
}

void Scheduler::LogParRec() {
    rec_log_batches = 0;
    using Clock = std::chrono::steady_clock;
    struct Prev {
        u64 submits, submit_waits, ordered_waits, wait_ns, chunk_ns, chunks;
    };
    static std::array<Prev, MaxRecLanes> prev{};
    static u64 prev_borrows = 0;
    static Clock::time_point prev_time = Clock::now();
    const auto now = Clock::now();
    std::string lanes;
    for (u32 i = 0; i < par_rec; ++i) {
        const RecLane& lane = rec_lanes[i];
        const Prev cur{lane.submits.load(std::memory_order_relaxed),
                       lane.submit_waits.load(std::memory_order_relaxed),
                       lane.ordered_waits.load(std::memory_order_relaxed),
                       lane.wait_ns.load(std::memory_order_relaxed),
                       recording_worker->LaneWorkerNs(i), recording_worker->LaneChunks(i)};
        const Prev& p = prev[i];
        const double chunk_ms = double(cur.chunk_ns - p.chunk_ns) / 1e6;
        const double wait_ms = double(cur.wait_ns - p.wait_ns) / 1e6;
        lanes += fmt::format(" | lane{} submits={} chunks={} submit_waits={} ordered_waits={} "
                             "chunk_ms={:.1f} wait_ms={:.1f} run_ms={:.1f}",
                             i, cur.submits - p.submits, cur.chunks - p.chunks,
                             cur.submit_waits - p.submit_waits, cur.ordered_waits - p.ordered_waits,
                             chunk_ms, wait_ms, chunk_ms - wait_ms);
        prev[i] = cur;
    }
    LOG_INFO(Render_Vulkan, "ParRec: lanes={} window_ms={:.1f} borrows={} timing={}{}", par_rec,
             std::chrono::duration<double, std::milli>(now - prev_time).count(),
             synchronous_borrows - prev_borrows, recording_measure_timing, lanes);
    prev_borrows = synchronous_borrows;
    prev_time = now;
}

void Scheduler::BeginAscCmdbufOnWorker() {
    asc_cmdbuf = asc_pool->Commit();
    Check(asc_cmdbuf.begin(vk::CommandBufferBeginInfo{
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
    }));
    if (asc_timer) {
        AscTimerBegin();
    }
}

void Scheduler::AscTimerBegin() {
    auto& t = *asc_timer;
    AscTimerCollect();
    if (t.write_seq - t.read_seq >= AscTimer::Slots) {
        t.began = false; // ring full: skip this segment
        return;
    }
    t.open_slot = static_cast<u32>(t.write_seq % AscTimer::Slots);
    asc_cmdbuf.resetQueryPool(*t.pool, t.open_slot * 2, 2);
    asc_cmdbuf.writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, *t.pool, t.open_slot * 2);
    t.began = true;
}

void Scheduler::AscTimerEnd(u64 value) {
    auto& t = *asc_timer;
    if (!t.began) {
        return;
    }
    asc_cmdbuf.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, *t.pool,
                              t.open_slot * 2 + 1);
    t.value[t.open_slot] = value;
    ++t.write_seq;
    t.began = false;
}

void Scheduler::AscTimerCollect() {
    auto& t = *asc_timer;
    while (t.read_seq < t.write_seq) {
        const u32 slot = static_cast<u32>(t.read_seq % AscTimer::Slots);
        if (asc_semaphore->KnownGpuTick() < t.value[slot]) {
            asc_semaphore->Refresh();
            if (asc_semaphore->KnownGpuTick() < t.value[slot]) {
                break;
            }
        }
        std::array<u64, 2> ts{};
        const auto result = instance.GetDevice().getQueryPoolResults(
            *t.pool, slot * 2, 2, sizeof(ts), ts.data(), sizeof(u64),
            vk::QueryResultFlagBits::e64);
        ++t.read_seq;
        if (result == vk::Result::eSuccess && ts[1] >= ts[0]) {
            const u64 ns = u64(double(ts[1] - ts[0]) * t.period_ns);
            t.busy_ns += ns;
            t.max_ns = std::max(t.max_ns, ns);
            ++t.segments;
        }
    }
    const auto now = std::chrono::steady_clock::now();
    const double wall_ns =
        double(std::chrono::duration_cast<std::chrono::nanoseconds>(now - t.window).count());
    if (wall_ns >= 10e9) {
        LOG_INFO(Render_Vulkan,
                 "AscTiming: segments={} busy_ms={:.1f} busy_pct={:.1f} mean_us={:.1f} max_us={:.1f} "
                 "(per {:.1f} s window; ASC queue, segment start to end)",
                 t.segments, double(t.busy_ns) / 1e6, 100.0 * double(t.busy_ns) / wall_ns,
                 t.segments ? double(t.busy_ns) / double(t.segments) / 1000.0 : 0.0,
                 double(t.max_ns) / 1000.0, wall_ns / 1e9);
        t.busy_ns = t.segments = t.max_ns = 0;
        t.window = now;
    }
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
    Rc().native_heaps_bound = false;
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
    {
        // I2 coupling: how far back is the main batch this segment waits (0 = none).
        const u64 now = master_semaphore.CurrentTick();
        const u64 d = wait_main == 0 ? 0 : now - std::min(now, wait_main);
        ++asc_wait_dist[wait_main == 0 ? 0 : d <= 1 ? 1 : d <= 3 ? 2 : 3];
    }
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
    if (transfer_group[1].open) {
        CloseTransferGroupSlow(transfer_group[1], asc_cmdbuf);
    }
    if (asc_timer) {
        AscTimerEnd(value);
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
