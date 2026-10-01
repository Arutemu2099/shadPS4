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
/// SHADPS4_CP_CORE_STATS=1: sample the logical CPU this thread runs on (call often, it is cheap)
/// and periodically log the share on performance vs efficiency cores. Diagnostic only.
void SampleCommandProcessorCore();
/// SHADPS4_THREAD_STATS=1 (diagnostic, default off): call at a natural point of a hot thread's
/// loop. Every 2^stride_log2-th call notes the logical processor the thread is on; four times a
/// second it times a fixed dependent-operation loop with rdtsc (loop iterations per TSC tick
/// follow the core's real clock, which the invariant TSC does not); every ~10 s the thread logs
/// its processor histogram, migrations, CPU time and the clock probe (median and minimum).
/// Explains run-to-run speed modes: placement and frequency versus extra work.
void SampleThreadPlacement(const char* role, unsigned stride_log2 = 6);

} // namespace Common
