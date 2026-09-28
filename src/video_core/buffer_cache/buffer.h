// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstddef>
#include <cstdlib>
#include <optional>
#include <utility>
#include <vector>
#include "common/fault_monitor.h"
#include "common/types.h"
#include "core/memory.h"
#include "video_core/amdgpu/resource.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace VideoCore {
/// SHADPS4_ASC_STREAM: the command stream being recorded (0 main, 1 ASC); CP thread only.
/// Buffer barrier state is kept per stream (Buffer::SelectStreamState).
inline u32 g_record_stream = 0;
} // namespace VideoCore

namespace Vulkan {
class Instance;
class Scheduler;
} // namespace Vulkan

VK_DEFINE_HANDLE(VmaAllocation)
VK_DEFINE_HANDLE(VmaAllocator)

struct VmaAllocationInfo;

namespace VideoCore {

/// Hints and requirements for the backing memory type of a commit
enum class MemoryUsage {
    DeviceLocal, ///< Requests device local buffer.
    Upload,      ///< Requires a host visible memory type optimized for CPU to GPU uploads
    Download,    ///< Requires a host visible memory type optimized for GPU to CPU readbacks
    Stream,      ///< Requests device local host visible buffer, falling back host memory.
};

constexpr vk::BufferUsageFlags ReadFlags =
    vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eUniformBuffer |
    vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eVertexBuffer |
    vk::BufferUsageFlagBits::eIndirectBuffer;

constexpr vk::BufferUsageFlags AllFlags =
    ReadFlags | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer;

struct UniqueBuffer {
    explicit UniqueBuffer(vk::Device device, VmaAllocator allocator);
    ~UniqueBuffer();

    UniqueBuffer(const UniqueBuffer&) = delete;
    UniqueBuffer& operator=(const UniqueBuffer&) = delete;

    UniqueBuffer(UniqueBuffer&& other) noexcept
        : device{std::exchange(other.device, VK_NULL_HANDLE)},
          allocator{std::exchange(other.allocator, VK_NULL_HANDLE)},
          allocation{std::exchange(other.allocation, VK_NULL_HANDLE)},
          buffer{std::exchange(other.buffer, VK_NULL_HANDLE)},
          bda_addr{std::exchange(other.bda_addr, 0)} {}
    UniqueBuffer& operator=(UniqueBuffer&& other) noexcept;

    void Create(const vk::BufferCreateInfo& image_ci, MemoryUsage usage,
                VmaAllocationInfo* out_alloc_info);

    operator vk::Buffer() const {
        return buffer;
    }

    vk::Device device;
    VmaAllocator allocator;
    VmaAllocation allocation;
    vk::Buffer buffer{};
    vk::DeviceAddress bda_addr = 0;
};

class Buffer {
public:
    explicit Buffer(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler,
                    MemoryUsage usage, VAddr cpu_addr_, vk::BufferUsageFlags flags,
                    u64 size_bytes_);

    /// SHADPS4_HOST_MIRROR: wraps a VkBuffer bound to imported host memory
    /// (VK_EXT_external_memory_host). The memory is not VMA's and lives until process exit.
    explicit Buffer(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler,
                    vk::Buffer imported, VAddr cpu_addr_, u64 size_bytes_);

    Buffer& operator=(const Buffer&) = delete;
    Buffer(const Buffer&) = delete;

    Buffer& operator=(Buffer&&) = default;
    Buffer(Buffer&&) = default;

    void IncreaseStreamScore(int score) noexcept {
        stream_score += score;
    }

    [[nodiscard]] int StreamScore() const noexcept {
        return stream_score;
    }

    [[nodiscard]] bool IsInBounds(VAddr addr, u64 size) const noexcept {
        return addr >= cpu_addr && addr + size <= cpu_addr + SizeBytes();
    }

    [[nodiscard]] VAddr CpuAddr() const noexcept {
        return cpu_addr;
    }

    [[nodiscard]] u64 Offset(VAddr other_cpu_addr) const noexcept {
        return other_cpu_addr - cpu_addr;
    }

    size_t SizeBytes() const {
        return size_bytes;
    }

    void SetLRUId(u64 id) noexcept {
        lru_id = id;
    }

    u64 LRUId() const noexcept {
        return lru_id;
    }

    vk::Buffer Handle() const noexcept {
        return buffer;
    }

    vk::DeviceAddress BufferDeviceAddress() const noexcept {
        ASSERT_MSG(buffer.bda_addr != 0, "Can't get BDA from a non BDA buffer");
        return buffer.bda_addr;
    }

    /// SHADPS4_SMART_BUFFER_BARRIERS=1: emit a barrier only for a real hazard. A read after a
    /// read needs none; a read after a write needs one only until that write has been made
    /// visible to the reading stage and access; a write waits for every earlier read and write.
    /// Measured 2026-09-24: ~490 buffer barrier calls per frame (each also ends the render pass)
    /// under the default "any change of access or stage" rule.
    /// Read once at startup (a function-local static would add a guard check per binding).
    static inline const bool smart_barriers_enabled = [] {
        const char* value = std::getenv("SHADPS4_SMART_BUFFER_BARRIERS");
        return value != nullptr && value[0] == '1';
    }();
    static bool SmartBarriers() {
        return smart_barriers_enabled;
    }

    /// `run` (SHADPS4_GUEST_DISPATCH_ORDER, smart barriers only): nonzero for a dispatch binding;
    /// equal values mean consecutive dispatches of one guest queue with no guest sync packet and
    /// nothing else recorded in between. GCN overlaps such dispatches (the game orders dependent
    /// ones with RELEASE_MEM / WAIT_REG_MEM / ACQUIRE_MEM / events), so no barrier among them.
    std::optional<vk::BufferMemoryBarrier2> GetBarrier(vk::AccessFlags2 dst_acess_mask,
                                                       vk::PipelineStageFlagBits2 dst_stage,
                                                       u32 offset = 0, u64 run = 0) {
        SelectStreamState();
        if (SmartBarriers()) {
            return GetHazardBarrier(dst_acess_mask, vk::PipelineStageFlags2{dst_stage}, offset,
                                    run);
        }
        if (dst_acess_mask == access_mask && stage == dst_stage) {
            return {};
        }

        DEBUG_ASSERT(offset < size_bytes);

        const auto barrier = vk::BufferMemoryBarrier2{
            .srcStageMask = stage,
            .srcAccessMask = access_mask,
            .dstStageMask = dst_stage,
            .dstAccessMask = dst_acess_mask,
            .buffer = buffer.buffer,
            .offset = offset,
            .size = size_bytes - offset,
        };
        access_mask = dst_acess_mask;
        stage = dst_stage;
        return barrier;
    }

    void Fill(u64 offset, u32 num_bytes, u32 value);

private:
    static constexpr vk::AccessFlags2 WriteAccessBits =
        vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eShaderStorageWrite |
        vk::AccessFlagBits2::eTransferWrite | vk::AccessFlagBits2::eMemoryWrite |
        vk::AccessFlagBits2::eHostWrite | vk::AccessFlagBits2::eColorAttachmentWrite |
        vk::AccessFlagBits2::eDepthStencilAttachmentWrite |
        vk::AccessFlagBits2::eTransformFeedbackWriteEXT |
        vk::AccessFlagBits2::eTransformFeedbackCounterWriteEXT;

    std::optional<vk::BufferMemoryBarrier2> GetHazardBarrier(vk::AccessFlags2 dst_access,
                                                             vk::PipelineStageFlags2 dst_stages,
                                                             u32 offset, u64 run) {
        if (run != 0) {
            // Guest dispatch run: the run's accesses are never hazards against each other (the
            // game ordered nothing between them). Each barrier emitted in a run covers both
            // shader access kinds, so the run's later accesses of this buffer are visible too.
            const bool write = bool(dst_access & WriteAccessBits);
            if (run == hz_run) {
                if (!write) {
                    hz_read_stages |= dst_stages;
                    return {};
                }
                if (hz_run_wrote) {
                    hz_write_stages |= dst_stages;
                    hz_write_access |= dst_access & WriteAccessBits;
                    return {};
                }
                // First write of the run: order it once against reads (and unseen writes) that
                // may precede the run; reads made by the run itself are included conservatively.
            }
            hz_run = run;
            hz_run_wrote = write;
            dst_access |= vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite;
            if (!write) {
                // A read opening the run: the normal read rule, widened access.
                const bool covered = (hz_visible_stages & dst_stages) == dst_stages &&
                                     (hz_visible_access & dst_access) == dst_access;
                hz_read_stages |= dst_stages;
                if (!hz_write_access || covered) {
                    return {};
                }
                const auto barrier = vk::BufferMemoryBarrier2{
                    .srcStageMask = hz_write_stages,
                    .srcAccessMask = hz_write_access,
                    .dstStageMask = dst_stages,
                    .dstAccessMask = dst_access,
                    .buffer = buffer.buffer,
                    .offset = offset,
                    .size = size_bytes - offset,
                };
                hz_visible_stages |= dst_stages;
                hz_visible_access |= dst_access;
                return barrier;
            }
            // A write (opening the run or its first write): always ordered against any earlier
            // read or write (no write-repeat shortcut: a guest sync may precede this run).
            std::optional<vk::BufferMemoryBarrier2> barrier{};
            if (hz_write_access || hz_read_stages) {
                vk::PipelineStageFlags2 src_stages = hz_write_stages | hz_read_stages;
                if (!src_stages) {
                    src_stages = vk::PipelineStageFlagBits2::eAllCommands;
                }
                barrier = vk::BufferMemoryBarrier2{
                    .srcStageMask = src_stages,
                    .srcAccessMask = hz_write_access,
                    .dstStageMask = dst_stages,
                    .dstAccessMask = dst_access,
                    .buffer = buffer.buffer,
                    .offset = offset,
                    .size = size_bytes - offset,
                };
            }
            hz_run_wrote = true;
            hz_write_stages = dst_stages;
            hz_write_access = dst_access & WriteAccessBits;
            hz_read_stages = {};
            hz_visible_stages = {};
            hz_visible_access = {};
            return barrier;
        }
        hz_run = 0;
        const bool is_write = bool(dst_access & WriteAccessBits);
        if (!is_write) {
            const bool covered = (hz_visible_stages & dst_stages) == dst_stages &&
                                 (hz_visible_access & dst_access) == dst_access;
            hz_read_stages |= dst_stages;
            if (!hz_write_access || covered) {
                return {};
            }
            const auto barrier = vk::BufferMemoryBarrier2{
                .srcStageMask = hz_write_stages,
                .srcAccessMask = hz_write_access,
                .dstStageMask = dst_stages,
                .dstAccessMask = dst_access,
                .buffer = buffer.buffer,
                .offset = offset,
                .size = size_bytes - offset,
            };
            hz_visible_stages |= dst_stages;
            hz_visible_access |= dst_access;
            return barrier;
        }
        // Same as the default rule: a repeat of the same write with no read in between gets no
        // barrier (consecutive dispatches writing one buffer, as before this option).
        if (!hz_read_stages && hz_write_stages == dst_stages &&
            hz_write_access == (dst_access & WriteAccessBits)) {
            return {};
        }
        vk::PipelineStageFlags2 src_stages = hz_write_stages | hz_read_stages;
        if (!src_stages) {
            src_stages = vk::PipelineStageFlagBits2::eAllCommands;
        }
        const auto barrier = vk::BufferMemoryBarrier2{
            .srcStageMask = src_stages,
            .srcAccessMask = hz_write_access,
            .dstStageMask = dst_stages,
            .dstAccessMask = dst_access,
            .buffer = buffer.buffer,
            .offset = offset,
            .size = size_bytes - offset,
        };
        hz_write_stages = dst_stages;
        hz_write_access = dst_access & WriteAccessBits;
        hz_read_stages = {};
        hz_visible_stages = {};
        hz_visible_access = {};
        return barrier;
    }

    /// SHADPS4_ASC_STREAM: barriers order work inside one command stream only; the other stream's
    /// accesses are ordered by semaphores (I1/I2). The fields below hold the state of
    /// `state_stream`; the other stream's state is parked here and swapped in on demand.
    struct ParkedBarrierState {
        vk::PipelineStageFlags2 hz_write_stages{vk::PipelineStageFlagBits2::eAllCommands};
        vk::AccessFlags2 hz_write_access{vk::AccessFlagBits2::eMemoryWrite};
        vk::PipelineStageFlags2 hz_read_stages{};
        vk::PipelineStageFlags2 hz_visible_stages{};
        vk::AccessFlags2 hz_visible_access{};
        u64 hz_run{};
        bool hz_run_wrote{};
        vk::Flags<vk::AccessFlagBits2> access_mask{
            vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite |
            vk::AccessFlagBits2::eTransferRead | vk::AccessFlagBits2::eTransferWrite};
        vk::PipelineStageFlagBits2 stage{vk::PipelineStageFlagBits2::eAllCommands};
    };
    ParkedBarrierState parked_state{};
    u32 state_stream = 0;
    void SelectStreamState() {
        if (state_stream == g_record_stream) [[likely]] {
            return;
        }
        std::swap(hz_write_stages, parked_state.hz_write_stages);
        std::swap(hz_write_access, parked_state.hz_write_access);
        std::swap(hz_read_stages, parked_state.hz_read_stages);
        std::swap(hz_visible_stages, parked_state.hz_visible_stages);
        std::swap(hz_visible_access, parked_state.hz_visible_access);
        std::swap(hz_run, parked_state.hz_run);
        std::swap(hz_run_wrote, parked_state.hz_run_wrote);
        std::swap(access_mask, parked_state.access_mask);
        std::swap(stage, parked_state.stage);
        state_stream = g_record_stream;
    }

    // Hazard state for SmartBarriers(); starts as "unknown write by anything".
    vk::PipelineStageFlags2 hz_write_stages{vk::PipelineStageFlagBits2::eAllCommands};
    vk::AccessFlags2 hz_write_access{vk::AccessFlagBits2::eMemoryWrite};
    vk::PipelineStageFlags2 hz_read_stages{};
    vk::PipelineStageFlags2 hz_visible_stages{};
    vk::AccessFlags2 hz_visible_access{};
    u64 hz_run{}; ///< guest dispatch run of the last access (0 = not a dispatch run)
    bool hz_run_wrote{}; ///< that run has written this buffer

public:

public:
    VAddr cpu_addr = 0;
    bool is_picked{};
    bool is_coherent{};
    bool is_deleted{};
    /// SHADPS4_ASC_SPAN_STATS: newest main-timeline tick that accessed / wrote this buffer from
    /// a graphics-queue binding (diagnostic).
    u64 diag_main_access_tick{};
    u64 diag_main_write_tick{};
    /// SHADPS4_ASC_STREAM: newest main-stream tick that accessed / wrote this buffer (bindings,
    /// uploads, copies, fault copies). An ASC segment binding it waits for that main batch; if it
    /// is the batch still being recorded the dispatch stays on the main stream.
    mutable u64 main_access_tick{};
    mutable u64 main_write_tick{};
    void NoteMainAccess(u64 tick, bool write) const {
        if (g_record_stream != 0) {
            return;
        }
        main_access_tick = tick;
        if (write) {
            main_write_tick = tick;
        }
    }
    int stream_score = 0;
    size_t size_bytes = 0;
    u64 lru_id = 0;
    std::span<u8> mapped_data;
    const Vulkan::Instance* instance;
    Vulkan::Scheduler* scheduler;
    MemoryUsage usage;
    UniqueBuffer buffer;
    vk::Flags<vk::AccessFlagBits2> access_mask{
        vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite |
        vk::AccessFlagBits2::eTransferRead | vk::AccessFlagBits2::eTransferWrite};
    vk::PipelineStageFlagBits2 stage{vk::PipelineStageFlagBits2::eAllCommands};
};

class StreamBuffer : public Buffer {
public:
    explicit StreamBuffer(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler,
                          MemoryUsage usage, u64 size_bytes_);

    /// Reserves a region of memory from the stream buffer.
    std::pair<u8*, u64> Map(u64 size, u64 alignment = 0, bool allow_wait = true);

    /// Ensures that reserved bytes of memory are available to the GPU.
    void Commit();

    /// Maps and commits a memory region with user provided data
    u64 Copy(auto src, size_t size, size_t alignment = 0) {
        // Measured 2026-08-09: 6.6 calls per draw on the ObtainBuffer early-exit path, 2.29
        // us/draw, of which 1.98 us is CopySparseMemory - i.e. ~43 MB per frame actually copied
        // out of guest memory into the stream buffer, mostly cold-miss and write-combine traffic
        // rather than lookup overhead. Cutting it needs the copy to not happen (a memo keyed on
        // address+size, invalidated by the CPU-modified tracking), not a faster copy. Timers were
        // removed after the split was recorded; they cost 0.57 us/draw on their own.
        const auto [data, offset] = Map(size, alignment);
        auto* memory = Core::Memory::Instance();
        const VAddr src_vaddr = reinterpret_cast<const VAddr>(src);
        if (memory->IsValidMapping(src_vaddr)) {
            memory->CopySparseMemory(src_vaddr, data, size);
        } else {
            std::memcpy(data, reinterpret_cast<const void*>(src), size);
        }
        Commit();
        return offset;
    }

private:
    struct Watch {
        u64 tick{};
        u64 upper_bound{};
    };

    /// Increases the amount of watches available.
    void ReserveWatches(std::vector<Watch>& watches, std::size_t grow_size);

    /// Waits pending watches until requested upper bound.
    bool WaitPendingOperations(u64 requested_upper_bound, bool allow_wait);

private:
    u64 offset{};
    u64 mapped_size{};
    std::vector<Watch> current_watches;
    std::size_t current_watch_cursor{};
    std::optional<size_t> invalidation_mark;
    std::vector<Watch> previous_watches;
    std::size_t wait_cursor{};
    u64 wait_bound{};
};

} // namespace VideoCore
