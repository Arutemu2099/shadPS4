// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <thread>

#include "common/logging/log.h"
#include "common/thread.h"
#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_master_semaphore.h"

namespace Vulkan {

/**
 * SHADPS4_GPU_BATCH_TIMING=1 (with SHADPS4_ASYNC_SUBMIT=1): GPU timestamps at the start and end
 * of every main-queue batch, written by the recording worker. A reader thread waits for each
 * batch to complete and logs, every 10 s: GPU busy share of wall time, mean batch duration,
 * idle gaps between consecutive batches (other submissions on the queue fall into the gaps) and
 * GPU time per draw. Diagnostic only.
 */
class GpuBatchTimer {
public:
    static bool Enabled() {
        static const bool enabled = [] {
            const char* value = std::getenv("SHADPS4_GPU_BATCH_TIMING");
            return value != nullptr && value[0] == '1';
        }();
        return enabled;
    }

    GpuBatchTimer(const Instance& instance_, MasterSemaphore& semaphore_)
        : instance{instance_}, semaphore{semaphore_} {
        const auto device = instance.GetDevice();
        auto [result, pool] = device.createQueryPool(vk::QueryPoolCreateInfo{
            .queryType = vk::QueryType::eTimestamp,
            .queryCount = Slots * 2,
        });
        ASSERT(result == vk::Result::eSuccess);
        query_pool = pool;
        device.resetQueryPool(query_pool, 0, Slots * 2);
        period_ns = instance.GetPhysicalDevice().getProperties().limits.timestampPeriod;
        reader = std::jthread([this](std::stop_token stop) { ReadLoop(stop); });
    }

    ~GpuBatchTimer() {
        reader.request_stop();
        if (reader.joinable()) {
            reader.join();
        }
        instance.GetDevice().destroyQueryPool(query_pool);
    }

    /// Recording worker, at the start of a batch's command buffer.
    void Begin(vk::CommandBuffer cmdbuf) {
        const u64 slot = write_seq % Slots;
        cmdbuf.resetQueryPool(query_pool, u32(slot * 2), 2);
        cmdbuf.writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, query_pool, u32(slot * 2));
        began = true;
    }

    /// Recording worker, just before ending a batch submitted with `signal_value`.
    void End(vk::CommandBuffer cmdbuf, u64 signal_value, u32 draws) {
        if (!began) {
            return;
        }
        const u64 slot = write_seq % Slots;
        cmdbuf.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, query_pool,
                              u32(slot * 2 + 1));
        signal_values[slot] = signal_value;
        batch_draws[slot] = draws;
        written.store(++write_seq, std::memory_order_release);
        began = false;
    }

private:
    static constexpr u64 Slots = 1024;

    void ReadLoop(std::stop_token stop) {
        Common::SetCurrentThreadName("shadPS4:GpuBatchTimer");
        u64 read_seq = 0;
        u64 prev_end = 0;
        u64 busy = 0, gaps = 0, gap_count = 0, batches = 0, draws = 0, max_batch = 0;
        auto window_start = std::chrono::steady_clock::now();
        while (!stop.stop_requested()) {
            if (read_seq >= written.load(std::memory_order_acquire)) {
                std::this_thread::sleep_for(std::chrono::microseconds(500));
                continue;
            }
            const u64 slot = read_seq % Slots;
            const u64 value = signal_values[slot];
            while (!stop.stop_requested() && semaphore.QueryCounter() < value) {
                std::this_thread::sleep_for(std::chrono::microseconds(200));
            }
            std::array<u64, 2> ts{};
            const auto result = instance.GetDevice().getQueryPoolResults(
                query_pool, u32(slot * 2), 2, sizeof(ts), ts.data(), sizeof(u64),
                vk::QueryResultFlagBits::e64);
            ++read_seq;
            if (result != vk::Result::eSuccess || ts[1] < ts[0]) {
                continue;
            }
            const u64 start = u64(double(ts[0]) * period_ns);
            const u64 end = u64(double(ts[1]) * period_ns);
            const u64 duration = end - start;
            busy += duration;
            max_batch = std::max(max_batch, duration);
            if (prev_end != 0 && start > prev_end && start - prev_end < 50'000'000) {
                gaps += start - prev_end;
                ++gap_count;
            }
            prev_end = end;
            ++batches;
            draws += batch_draws[slot];
            const auto now = std::chrono::steady_clock::now();
            const double wall_ns =
                double(std::chrono::duration_cast<std::chrono::nanoseconds>(now - window_start)
                           .count());
            if (wall_ns >= 10e9) {
                LOG_INFO(Render_Vulkan,
                         "GpuBatchTiming: batches={} busy_pct={:.1f} batch_us={:.1f} "
                         "max_batch_us={:.1f} gap_us={:.1f} draws_per_batch={:.1f} "
                         "gpu_us_per_draw={:.3f}",
                         batches, 100.0 * double(busy) / wall_ns,
                         double(busy) / double(batches) / 1000.0, double(max_batch) / 1000.0,
                         gap_count ? double(gaps) / double(gap_count) / 1000.0 : 0.0,
                         double(draws) / double(batches),
                         draws ? double(busy) / double(draws) / 1000.0 : 0.0);
                busy = gaps = gap_count = batches = draws = max_batch = 0;
                window_start = now;
            }
        }
    }

    const Instance& instance;
    MasterSemaphore& semaphore;
    vk::QueryPool query_pool;
    double period_ns = 1.0;
    std::array<u64, Slots> signal_values{};
    std::array<u32, Slots> batch_draws{};
    u64 write_seq = 0;
    bool began = false;
    std::atomic<u64> written{0};
    std::jthread reader;
};

} // namespace Vulkan
