// SPDX-FileCopyrightText: 2013 Dolphin Emulator Project
// SPDX-FileCopyrightText: 2014 Citra Emulator Project
// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <chrono>
#include "common/types.h"

namespace Common {

enum class ThreadPriority : u32 {
    Low = 0,
    Normal = 1,
    High = 2,
    VeryHigh = 3,
    Critical = 4,
};

void SetCurrentThreadRealtime(std::chrono::nanoseconds period_ns);

void SetCurrentThreadPriority(ThreadPriority new_priority);

void SetCurrentThreadName(const char* name);

void SetThreadName(void* thread, const char* name);

bool AccurateSleep(std::chrono::nanoseconds duration, std::chrono::nanoseconds* remaining,
                   bool interruptible);

class AccurateTimer {
    std::chrono::nanoseconds target_interval{};
    std::chrono::nanoseconds total_wait{};

    std::chrono::high_resolution_clock::time_point start_time;

public:
    explicit AccurateTimer(std::chrono::nanoseconds target_interval);

    void Start();

    void End();

    std::chrono::nanoseconds GetTotalWait() const {
        return total_wait;
    }
};

std::string GetCurrentThreadName();

/// SHADPS4_CP_QOS=1: turn off process power throttling (EcoQoS) and raise this thread to
/// THREAD_PRIORITY_HIGHEST; =2 additionally restricts it to the performance cores (highest CPU
/// set efficiency class). Default off. Windows only; call on the thread itself.
void ApplyCommandProcessorPlacement();
/// SHADPS4_HOT_CORES=<mask> (default 0): bit 0 gives the Vulkan recording worker, bit 1 the CP
/// front thread a whole performance core each (the two before the one SHADPS4_CP_QOS=3 gives the
/// CP back); the rest of the process is kept off those cores. Measured 2026-10-01 without it: the
/// recorder (80-87% busy) changes logical processor at ~20% of its chunk boundaries and spends
/// 2-15% of them on efficiency cores. Windows only; call on the thread itself.
enum class HotThread { Recorder, Front };
void ApplyHotThreadPlacement(HotThread which);
/// SHADPS4_CP_CORE_STATS=1: sample the logical CPU this thread runs on (call often, it is cheap)
/// and periodically log the share on performance vs efficiency cores. Diagnostic only.
void SampleCommandProcessorCore();
/// SHADPS4_THREAD_STATS=1 (diagnostic, default off): call at a natural point of a hot thread's
/// loop. Every 2^stride_log2-th call notes the logical processor the thread is on; four times a
/// second it times a fixed dependent-operation loop with rdtsc (loop iterations per TSC tick
/// follow the core's real clock, which the invariant TSC does not); every ~10 s the thread logs
/// its processor histogram, migrations, CPU time and the clock probe (median and minimum).
/// Explains run-to-run speed modes: placement and frequency versus extra work.
/// Cost, measured 2026-10-01 on a41 in fire: -3.3% FPS with the flag on (117.75 vs 121.3 / 122.0;
/// the per-record call on the CP back, not the sampled work). Attribution runs only, never in
/// an FPS comparison. Off: one inline test of a global per call site.
extern const bool g_thread_stats;
void SampleThreadPlacementSlow(const char* role, unsigned stride_log2);
inline void SampleThreadPlacement(const char* role, unsigned stride_log2 = 6) {
    if (g_thread_stats) [[unlikely]] {
        SampleThreadPlacementSlow(role, stride_log2);
    }
}

} // namespace Common
