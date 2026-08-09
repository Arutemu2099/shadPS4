// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <chrono>
#include <condition_variable>
#include <ctime>
#include <fstream>
#include <mutex>
#include <thread>

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

namespace {

// Short enough to see when the decay starts, long enough that a multi-hour run stays small:
// 240 lines per hour.
constexpr auto REPORT_INTERVAL = std::chrono::seconds(15);

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
               "read_total,read_unhandled,write_total,write_unhandled,jit_total,frames_total\n";
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
                 "| totals read={} write={} jit={} frames={} unhandled(r/w)={}/{}",
                 elapsed, fps, read_rate, write_rate, jit_rate, per_frame, read_us, write_us,
                 handler_pct, cur.read, cur.write, cur.jit, cur.frames, cur.read_unhandled,
                 cur.write_unhandled);

        if (csv) {
            csv << fmt::format(
                "{:.1f},{:.2f},{:.1f},{:.1f},{:.1f},{:.1f},{:.4f},{:.4f},{:.2f},{},{},{},{},{},{}\n",
                elapsed, fps, read_rate, write_rate, jit_rate, per_frame, read_us, write_us,
                handler_pct, cur.read, cur.read_unhandled, cur.write, cur.write_unhandled, cur.jit,
                cur.frames);
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
