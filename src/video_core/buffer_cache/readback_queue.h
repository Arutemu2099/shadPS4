// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <mutex>
#include "common/assert.h"
#include "video_core/renderer_vulkan/vk_common.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_master_semaphore.h"

namespace VideoCore {

/**
 * Second queue of the graphics family for guest readback copies (SHADPS4_READBACK_QUEUE=1).
 *
 * On the main queue a readback copy runs after everything submitted before it, although
 * measured 2026-09-23 69.5% of readbacks read data whose producing batch had already
 * completed. Here the copy waits only for its producer's main-timeline value, then signals
 * this queue's own timeline; the timeline wait carries the cross-queue memory dependency.
 * Mode 1 (same family) needs no ownership transfer. Mode 2 (transfer-only family) reads
 * EXCLUSIVE buffers without a queue-family ownership transfer: undefined by the spec for
 * contents, relied on here only as an experiment on NVIDIA, where buffers carry no
 * family-specific layout or compression. Command processor only.
 */
class ReadbackQueue {
public:
    explicit ReadbackQueue(const Vulkan::Instance& instance_)
        : ReadbackQueue(instance_, instance_.GetReadbackQueue(),
                        instance_.GetReadbackQueueFamilyIndex(), nullptr) {}

    /// SHADPS4_READBACK_PREBATCH=1: `queue_` is the main graphics queue and `submit_lock_` the
    /// scheduler's submit mutex. A copy submitted here runs after everything already submitted
    /// and before the batch the command processor is still recording.
    ReadbackQueue(const Vulkan::Instance& instance_, vk::Queue queue_, u32 family,
                  std::mutex* submit_lock_)
        : instance{instance_}, queue{queue_}, submit_lock{submit_lock_}, semaphore{instance_} {
        const auto device = instance.GetDevice();
        auto [pool_result, new_pool] = device.createCommandPoolUnique(vk::CommandPoolCreateInfo{
            .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer |
                     vk::CommandPoolCreateFlagBits::eTransient,
            .queueFamilyIndex = family,
        });
        ASSERT_MSG(pool_result == vk::Result::eSuccess, "ReadbackQueue: command pool failed");
        pool = std::move(new_pool);
        auto [alloc_result, buffers] = device.allocateCommandBuffers(vk::CommandBufferAllocateInfo{
            .commandPool = *pool,
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = static_cast<u32>(NumSlots),
        });
        ASSERT_MSG(alloc_result == vk::Result::eSuccess, "ReadbackQueue: allocation failed");
        for (size_t i = 0; i < NumSlots; ++i) {
            cmdbufs[i] = buffers[i];
        }
    }

    ~ReadbackQueue() {
        // Nothing may still execute from this pool when it is destroyed.
        semaphore.Wait(semaphore.CurrentTick() - 1);
    }

    /// Records with `record(cmdbuf)`, submits after `wait_value` on `wait_semaphore` (none if
    /// 0) and returns the value this queue's timeline reaches when the copy is done.
    template <typename Record>
    u64 Submit(vk::Semaphore wait_semaphore, u64 wait_value, Record&& record) {
        const size_t slot = next_slot++ % NumSlots;
        semaphore.Wait(slot_values[slot]);
        const vk::CommandBuffer cmdbuf = cmdbufs[slot];
        ASSERT(cmdbuf.reset() == vk::Result::eSuccess);
        ASSERT(cmdbuf.begin(vk::CommandBufferBeginInfo{
                   .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit}) ==
               vk::Result::eSuccess);
        record(cmdbuf);
        ASSERT(cmdbuf.end() == vk::Result::eSuccess);

        const u64 signal_value = semaphore.NextTick();
        const vk::Semaphore signal_semaphore = semaphore.Handle();
        const vk::PipelineStageFlags wait_stage = vk::PipelineStageFlagBits::eTransfer;
        const u32 wait_count = wait_value != 0 ? 1u : 0u;
        const vk::TimelineSemaphoreSubmitInfo timeline_info = {
            .waitSemaphoreValueCount = wait_count,
            .pWaitSemaphoreValues = &wait_value,
            .signalSemaphoreValueCount = 1,
            .pSignalSemaphoreValues = &signal_value,
        };
        const vk::SubmitInfo submit_info = {
            .pNext = &timeline_info,
            .waitSemaphoreCount = wait_count,
            .pWaitSemaphores = &wait_semaphore,
            .pWaitDstStageMask = &wait_stage,
            .commandBufferCount = 1,
            .pCommandBuffers = &cmdbuf,
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = &signal_semaphore,
        };
        vk::Result result;
        if (submit_lock) {
            std::scoped_lock lk{*submit_lock};
            result = queue.submit(submit_info);
        } else {
            result = queue.submit(submit_info);
        }
        ASSERT_MSG(result == vk::Result::eSuccess, "ReadbackQueue: submit failed: {}",
                   vk::to_string(result));
        slot_values[slot] = signal_value;
        return signal_value;
    }

    Vulkan::MasterSemaphore& Semaphore() noexcept {
        return semaphore;
    }

    /// SHADPS4_READBACK_PREBATCH with SHADPS4_ASYNC_SUBMIT: the command processor reserves the
    /// value here, in order, and the recording worker submits it with SubmitReserved when it
    /// reaches that point of the command stream - after every batch recorded before it and
    /// before the batch still being recorded.
    u64 ReserveTick() noexcept {
        return semaphore.NextTick();
    }

    template <typename Record>
    void SubmitReserved(u64 signal_value, Record&& record) {
        const size_t slot = next_slot++ % NumSlots;
        semaphore.Wait(slot_values[slot]);
        const vk::CommandBuffer cmdbuf = cmdbufs[slot];
        ASSERT(cmdbuf.reset() == vk::Result::eSuccess);
        ASSERT(cmdbuf.begin(vk::CommandBufferBeginInfo{
                   .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit}) ==
               vk::Result::eSuccess);
        record(cmdbuf);
        ASSERT(cmdbuf.end() == vk::Result::eSuccess);
        const vk::Semaphore signal_semaphore = semaphore.Handle();
        const vk::TimelineSemaphoreSubmitInfo timeline_info = {
            .signalSemaphoreValueCount = 1,
            .pSignalSemaphoreValues = &signal_value,
        };
        const vk::SubmitInfo submit_info = {
            .pNext = &timeline_info,
            .commandBufferCount = 1,
            .pCommandBuffers = &cmdbuf,
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = &signal_semaphore,
        };
        vk::Result result;
        if (submit_lock) {
            std::scoped_lock lk{*submit_lock};
            result = queue.submit(submit_info);
        } else {
            result = queue.submit(submit_info);
        }
        ASSERT_MSG(result == vk::Result::eSuccess, "ReadbackQueue: submit failed: {}",
                   vk::to_string(result));
        slot_values[slot] = signal_value;
    }

    /// True when this is the main graphics queue (copies are ordered by submission).
    bool InQueueOrder() const noexcept {
        return submit_lock != nullptr;
    }

private:
    static constexpr size_t NumSlots = 32;
    const Vulkan::Instance& instance;
    vk::Queue queue;
    std::mutex* submit_lock{};
    Vulkan::MasterSemaphore semaphore;
    vk::UniqueCommandPool pool;
    std::array<vk::CommandBuffer, NumSlots> cmdbufs{};
    std::array<u64, NumSlots> slot_values{};
    size_t next_slot = 0;
};

} // namespace VideoCore
