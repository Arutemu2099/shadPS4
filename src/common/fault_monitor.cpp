// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <ctime>
#include <fstream>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include <fmt/format.h>

#include "common/fault_monitor.h"
#include "common/logging/log.h"
#include "common/path_util.h"
#include "common/thread.h"

namespace Common::FaultMonitor {

std::atomic<u64> guest_read_faults{0};
std::atomic<u64> guest_read_unhandled{0};
std::atomic<u64> guest_read_ns{0};
std::atomic<u64> guest_write_faults{0};
std::atomic<u64> guest_write_unhandled{0};
std::atomic<u64> guest_write_ns{0};
std::atomic<u64> jit_patch_faults{0};
std::atomic<u64> jit_patch_ns{0};
std::atomic<u64> presented_frames{0};
std::atomic<u64> range_fragments{0};
std::atomic<u64> range_adds{0};
std::atomic<u64> range_subtracts{0};
std::atomic<u64> gpu_submits{0};
std::atomic<u64> gpu_waits{0};
std::atomic<u64> gpu_wait_ns{0};

namespace {

// Short enough to see when the decay starts, long enough that a multi-hour run stays small:
// 240 lines per hour.
constexpr auto REPORT_INTERVAL = std::chrono::seconds(15);

std::mutex page_mutex;
std::unordered_map<u64, u64> read_fault_pages;

} // namespace

void RecordReadFaultPage(u64 addr) {
    std::scoped_lock lk(page_mutex);
    ++read_fault_pages[addr >> 12];
}

namespace {

/// Top read-faulting pages, plus how concentrated the distribution is. If a handful of pages
/// account for most faults, a targeted fix is possible; if it is spread thin, it is not.
std::string TopReadFaultPages(size_t top_n) {
    std::vector<std::pair<u64, u64>> pages;
    u64 total = 0;
    {
        std::scoped_lock lk(page_mutex);
        pages.assign(read_fault_pages.begin(), read_fault_pages.end());
    }
    for (const auto& [page, count] : pages) {
        total += count;
    }
    if (pages.empty() || total == 0) {
        return "no read faults";
    }
    const size_t n = std::min(top_n, pages.size());
    std::partial_sort(pages.begin(), pages.begin() + n, pages.end(),
                      [](const auto& a, const auto& b) { return a.second > b.second; });

    std::string out = fmt::format("{} distinct pages, ", pages.size());
    u64 top_sum = 0;
    for (size_t i = 0; i < n; i++) {
        top_sum += pages[i].second;
    }
    out += fmt::format("top {} = {:.1f}% |", n, 100.0 * top_sum / total);
    for (size_t i = 0; i < n; i++) {
        out += fmt::format(" {:#x}={:.1f}%", pages[i].first << 12,
                           100.0 * pages[i].second / static_cast<double>(total));
    }
    return out;
}

std::mutex stop_mutex;
std::condition_variable stop_cv;
bool stop_requested = false;

struct Sample {
    u64 read;
    u64 read_unhandled;
    u64 read_ns;
    u64 write;
    u64 write_unhandled;
    u64 write_ns;
    u64 jit;
    u64 jit_ns;
    u64 frames;
    u64 fragments;
    u64 adds;
    u64 subtracts;
    u64 submits;
    u64 waits;
    u64 wait_ns;
};

Sample Read() {
    return Sample{
        guest_read_faults.load(std::memory_order_relaxed),
        guest_read_unhandled.load(std::memory_order_relaxed),
        guest_read_ns.load(std::memory_order_relaxed),
        guest_write_faults.load(std::memory_order_relaxed),
        guest_write_unhandled.load(std::memory_order_relaxed),
        guest_write_ns.load(std::memory_order_relaxed),
        jit_patch_faults.load(std::memory_order_relaxed),
        jit_patch_ns.load(std::memory_order_relaxed),
        presented_frames.load(std::memory_order_relaxed),
        range_fragments.load(std::memory_order_relaxed),
        range_adds.load(std::memory_order_relaxed),
        range_subtracts.load(std::memory_order_relaxed),
        gpu_submits.load(std::memory_order_relaxed),
        gpu_waits.load(std::memory_order_relaxed),
        gpu_wait_ns.load(std::memory_order_relaxed),
    };
}

std::ofstream OpenCsv() {
    const auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &now);
#else
    localtime_r(&now, &tm);
#endif
    char name[64];
    std::strftime(name, sizeof(name), "fault_rate_%Y%m%d_%H%M%S.csv", &tm);

    const auto path = FS::GetUserPath(FS::PathType::LogDir) / name;
    std::ofstream csv(path);
    if (csv) {
        csv << "elapsed_s,fps,read_per_s,write_per_s,jit_per_s,"
               "faults_per_frame,read_us_each,write_us_each,handler_pct_of_wall,"
               "read_total,read_unhandled,write_total,write_unhandled,jit_total,frames_total,"
               "range_fragments,range_adds_per_s,range_subs_per_s,"
               "submits_per_frame,waits_per_frame,gpu_wait_ms_per_frame\n";
        csv.flush();
        LOG_INFO(Render, "FaultMonitor: rate CSV at {}", FS::PathToUTF8String(path));
    } else {
        LOG_WARNING(Render, "FaultMonitor: could not open {}", FS::PathToUTF8String(path));
    }
    return csv;
}

void ReportLoop() {
    Common::SetCurrentThreadName("shadPS4:FaultMonitor");

    std::ofstream csv = OpenCsv();
    const auto started = std::chrono::steady_clock::now();
    auto prev_time = started;
    Sample prev = Read();

    while (true) {
        std::unique_lock lk(stop_mutex);
        if (stop_cv.wait_for(lk, REPORT_INTERVAL, [] { return stop_requested; })) {
            break;
        }
        lk.unlock();

        const auto now = std::chrono::steady_clock::now();
        const Sample cur = Read();

        const double dt = std::chrono::duration<double>(now - prev_time).count();
        const double elapsed = std::chrono::duration<double>(now - started).count();
        if (dt <= 0.0) {
            continue;
        }

        const u64 d_read = cur.read - prev.read;
        const u64 d_write = cur.write - prev.write;
        const u64 d_jit = cur.jit - prev.jit;
        const u64 d_frames = cur.frames - prev.frames;
        const u64 d_read_ns = cur.read_ns - prev.read_ns;
        const u64 d_write_ns = cur.write_ns - prev.write_ns;
        const u64 d_jit_ns = cur.jit_ns - prev.jit_ns;
        const u64 d_adds = cur.adds - prev.adds;
        const u64 d_subs = cur.subtracts - prev.subtracts;
        const u64 d_submits = cur.submits - prev.submits;
        const u64 d_waits = cur.waits - prev.waits;
        const u64 d_wait_ns = cur.wait_ns - prev.wait_ns;

        // The decisive split: if submits/frame stays flat while gpu_wait_ms/frame grows, the GPU
        // side is getting slower rather than being asked to do more.
        const double submits_per_frame =
            d_frames ? static_cast<double>(d_submits) / static_cast<double>(d_frames) : 0.0;
        const double waits_per_frame =
            d_frames ? static_cast<double>(d_waits) / static_cast<double>(d_frames) : 0.0;
        const double wait_ms_per_frame =
            d_frames ? static_cast<double>(d_wait_ns) / static_cast<double>(d_frames) / 1e6 : 0.0;

        const double read_rate = static_cast<double>(d_read) / dt;
        const double write_rate = static_cast<double>(d_write) / dt;
        const double jit_rate = static_cast<double>(d_jit) / dt;
        const double fps = static_cast<double>(d_frames) / dt;

        // Does the framerate fall because there are MORE faults per frame, or because each
        // fault costs MORE? These two columns separate them.
        const double per_frame =
            d_frames ? static_cast<double>(d_read + d_write) / static_cast<double>(d_frames) : 0.0;
        const double read_us = d_read ? static_cast<double>(d_read_ns) / d_read / 1000.0 : 0.0;
        const double write_us = d_write ? static_cast<double>(d_write_ns) / d_write / 1000.0 : 0.0;
        // Handler time as a share of one thread's wall time over the interval.
        const double handler_pct =
            static_cast<double>(d_read_ns + d_write_ns + d_jit_ns) / (dt * 1e9) * 100.0;

        LOG_INFO(Render,
                 "FaultRate: t={:.0f}s fps={:.1f} read={:.0f}/s write={:.0f}/s jit={:.0f}/s "
                 "| per_frame={:.0f} read={:.2f}us write={:.2f}us handler={:.1f}% of wall "
                 "| ranges frags={} add={:.0f}/s sub={:.0f}/s "
                 "| gpu submits/frame={:.1f} waits/frame={:.1f} wait={:.1f}ms/frame "
                 "| totals read={} write={} jit={} frames={} unhandled(r/w)={}/{}",
                 elapsed, fps, read_rate, write_rate, jit_rate, per_frame, read_us, write_us,
                 handler_pct, cur.fragments, static_cast<double>(d_adds) / dt,
                 static_cast<double>(d_subs) / dt, submits_per_frame, waits_per_frame,
                 wait_ms_per_frame, cur.read, cur.write, cur.jit, cur.frames, cur.read_unhandled,
                 cur.write_unhandled);

        if (d_read > 0) {
            LOG_INFO(Render, "FaultPages: {}", TopReadFaultPages(6));
        }

        if (csv) {
            csv << fmt::format(
                "{:.1f},{:.2f},{:.1f},{:.1f},{:.1f},{:.1f},{:.4f},{:.4f},{:.2f},{},{},{},{},{},{},"
                "{},{:.1f},{:.1f},{:.2f},{:.2f},{:.3f}\n",
                elapsed, fps, read_rate, write_rate, jit_rate, per_frame, read_us, write_us,
                handler_pct, cur.read, cur.read_unhandled, cur.write, cur.write_unhandled, cur.jit,
                cur.frames, cur.fragments, static_cast<double>(d_adds) / dt,
                static_cast<double>(d_subs) / dt, submits_per_frame, waits_per_frame,
                wait_ms_per_frame);
            csv.flush(); // a killed process must not lose the curve
        }

        prev = cur;
        prev_time = now;
    }
}

// Joined at static destruction, which requests the stop and wakes the thread immediately.
struct Reporter {
    Reporter() : thread(ReportLoop) {}
    ~Reporter() {
        {
            std::scoped_lock lk(stop_mutex);
            stop_requested = true;
        }
        stop_cv.notify_all();
        if (thread.joinable()) {
            thread.join();
        }
    }
    std::thread thread;
};

} // namespace

void Start() {
    static Reporter reporter;
    (void)reporter;
}

} // namespace Common::FaultMonitor
