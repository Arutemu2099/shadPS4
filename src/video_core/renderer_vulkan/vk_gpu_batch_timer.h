// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <source_location>
#include <string>
#include <vector>
#include <thread>

#include "common/logging/log.h"
#include "common/thread.h"
#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_master_semaphore.h"

namespace Vulkan {

/// Per-window counters of GPU pipeline breaks, bumped on the CP (diagnostic, same flag).
struct GpuBreakCounters {
    std::atomic<u64> render_begins{0};
    std::atomic<u64> buffer_barrier_calls{0};
    std::atomic<u64> buffer_barriers{0};
    std::atomic<u64> image_barrier_calls{0};
    std::atomic<u64> image_barriers{0};
};
inline GpuBreakCounters& GpuBreaks() {
    static GpuBreakCounters counters;
    return counters;
}

/// Per source location (same flag, diagnostic): render-pass ends, how many of those were followed
/// by a BeginRendering with the identical state (a pure restart), and image barriers by kind.
class BreakSites {
public:
    /// Hazard kinds; the same four again (+4) when the barrier also changes layout.
    enum Kind : u32 { WriteToRead, WriteToWrite, ReadToWrite, ReadToRead, NumKinds = 8 };

    u32 Site(const std::source_location& loc) {
        std::scoped_lock lk{mutex};
        return SiteLocked(loc);
    }
    void End(u32 site) {
        std::scoped_lock lk{mutex};
        ++entries[site].ends;
    }
    void Resumed(u32 site) {
        std::scoped_lock lk{mutex};
        ++entries[site].resumed;
    }
    void Image(const std::source_location& loc, vk::ImageLayout old_layout,
               vk::ImageLayout new_layout, vk::AccessFlags2 src, vk::AccessFlags2 dst) {
        constexpr auto writes =
            vk::AccessFlagBits2::eTransferWrite | vk::AccessFlagBits2::eShaderWrite |
            vk::AccessFlagBits2::eShaderStorageWrite | vk::AccessFlagBits2::eMemoryWrite |
            vk::AccessFlagBits2::eColorAttachmentWrite |
            vk::AccessFlagBits2::eDepthStencilAttachmentWrite;
        const bool src_w = static_cast<bool>(src & writes);
        const bool dst_w = static_cast<bool>(dst & writes);
        const u32 kind = (src_w ? (dst_w ? WriteToWrite : WriteToRead)
                          : dst_w ? ReadToWrite
                                  : ReadToRead) +
                         (old_layout != new_layout ? 4u : 0u);
        std::scoped_lock lk{mutex};
        ++entries[SiteLocked(loc)].img[kind];
    }
    /// Top sites by activity since the last call, then clears the counts.
    std::string Report() {
        std::scoped_lock lk{mutex};
        std::vector<const Entry*> order;
        for (const auto& e : entries) {
            order.push_back(&e);
        }
        const auto weight = [](const Entry* e) {
            u64 w = e->ends;
            for (const u64 v : e->img) {
                w += v;
            }
            return w;
        };
        std::ranges::sort(order,
                          [&](const Entry* a, const Entry* b) { return weight(a) > weight(b); });
        std::string out;
        for (size_t i = 0; i < order.size() && i < 14; ++i) {
            const Entry* e = order[i];
            if (weight(e) == 0) {
                break;
            }
            const char* base = e->file;
            for (const char* p = e->file; *p; ++p) {
                if (*p == '/' || *p == '\\') {
                    base = p + 1;
                }
            }
            out += fmt::format(" | {}:{} ends={} resumed={} same[w2r/w2w/r2w/r2r]={}/{}/{}/{} "
                               "relayout[w2r/w2w/r2w/r2r]={}/{}/{}/{}",
                               base, e->line, e->ends, e->resumed, e->img[0], e->img[1],
                               e->img[2], e->img[3], e->img[4], e->img[5], e->img[6], e->img[7]);
        }
        for (auto& e : entries) {
            e.ends = e.resumed = 0;
            e.img = {};
        }
        return out;
    }

private:
    struct Entry {
        const char* file;
        u32 line;
        u64 ends = 0;
        u64 resumed = 0;
        std::array<u64, NumKinds> img{};
    };
    u32 SiteLocked(const std::source_location& loc) {
        for (u32 i = 0; i < entries.size(); ++i) {
            if (entries[i].line == loc.line() &&
                std::strcmp(entries[i].file, loc.file_name()) == 0) {
                return i;
            }
        }
        entries.push_back(Entry{loc.file_name(), loc.line()});
        return static_cast<u32>(entries.size() - 1);
    }
    std::mutex mutex;
    std::vector<Entry> entries;
};
inline BreakSites& GpuBreakSites() {
    static BreakSites sites;
    return sites;
}

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
                         "gpu_us_per_draw={:.3f} render_begins={} buf_barrier_calls={} buf_barriers={} "
                         "img_barrier_calls={} img_barriers={}",
                         batches, 100.0 * double(busy) / wall_ns,
                         double(busy) / double(batches) / 1000.0, double(max_batch) / 1000.0,
                         gap_count ? double(gaps) / double(gap_count) / 1000.0 : 0.0,
                         double(draws) / double(batches),
                         draws ? double(busy) / double(draws) / 1000.0 : 0.0,
                         GpuBreaks().render_begins.exchange(0),
                         GpuBreaks().buffer_barrier_calls.exchange(0),
                         GpuBreaks().buffer_barriers.exchange(0),
                         GpuBreaks().image_barrier_calls.exchange(0),
                         GpuBreaks().image_barriers.exchange(0));
                LOG_INFO(Render_Vulkan, "GpuBreakSites:{}", GpuBreakSites().Report());
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
