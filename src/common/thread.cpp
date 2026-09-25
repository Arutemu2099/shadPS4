// SPDX-FileCopyrightText: 2013 Dolphin Emulator Project
// SPDX-FileCopyrightText: 2014 Citra Emulator Project
// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <string>
#include <thread>

#include "core/libraries/kernel/threads/pthread.h"

#include "common/error.h"
#include "common/logging/log.h"
#include "common/thread.h"
#include "ntapi.h"
#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/mach_time.h>
#include <pthread.h>
#elif defined(_WIN32)
#include <windows.h>
#include "common/string_util.h"
#else
#if defined(__Bitrig__) || defined(__DragonFly__) || defined(__FreeBSD__) || defined(__OpenBSD__)
#include <pthread_np.h>
#else
#include <pthread.h>
#endif
#include <sched.h>
#endif
#ifndef _WIN32
#include <unistd.h>
#endif

#ifdef __FreeBSD__
#define cpu_set_t cpuset_t
#endif

namespace Common {

#ifdef __APPLE__

void SetCurrentThreadRealtime(const std::chrono::nanoseconds period_ns) {
    // CPU time to grant.
    const std::chrono::nanoseconds computation_ns = period_ns / 2;

    // Determine the timebase for converting time to ticks.
    struct mach_timebase_info timebase{};
    mach_timebase_info(&timebase);
    const auto ticks_per_ns =
        static_cast<double>(timebase.denom) / static_cast<double>(timebase.numer);

    const auto period_ticks =
        static_cast<u32>(static_cast<double>(period_ns.count()) * ticks_per_ns);
    const auto computation_ticks =
        static_cast<u32>(static_cast<double>(computation_ns.count()) * ticks_per_ns);

    thread_time_constraint_policy policy = {
        .period = period_ticks,
        .computation = computation_ticks,
        // Should not matter since preemptible is false, but needs to be >= computation regardless.
        .constraint = computation_ticks,
        .preemptible = false,
    };

    int ret = thread_policy_set(
        pthread_mach_thread_np(pthread_self()), THREAD_TIME_CONSTRAINT_POLICY,
        reinterpret_cast<thread_policy_t>(&policy), THREAD_TIME_CONSTRAINT_POLICY_COUNT);
    if (ret != KERN_SUCCESS) {
        LOG_ERROR(Common, "Could not set thread to real-time with period {} ns: {}",
                  period_ns.count(), ret);
    }
}

#else

void SetCurrentThreadRealtime(const std::chrono::nanoseconds period_ns) {
    // Not implemented
}

#endif

#ifdef _WIN32

void SetCurrentThreadPriority(ThreadPriority new_priority) {
    auto handle = GetCurrentThread();
    int windows_priority = 0;
    switch (new_priority) {
    case ThreadPriority::Low:
        windows_priority = THREAD_PRIORITY_BELOW_NORMAL;
        break;
    case ThreadPriority::Normal:
        windows_priority = THREAD_PRIORITY_NORMAL;
        break;
    case ThreadPriority::High:
        windows_priority = THREAD_PRIORITY_ABOVE_NORMAL;
        break;
    case ThreadPriority::VeryHigh:
        windows_priority = THREAD_PRIORITY_HIGHEST;
        break;
    case ThreadPriority::Critical:
        windows_priority = THREAD_PRIORITY_TIME_CRITICAL;
        break;
    default:
        windows_priority = THREAD_PRIORITY_NORMAL;
        break;
    }
    SetThreadPriority(handle, windows_priority);
}

bool AccurateSleep(const std::chrono::nanoseconds duration, std::chrono::nanoseconds* remaining,
                   const bool interruptible) {
    const auto begin_sleep = std::chrono::high_resolution_clock::now();

    LARGE_INTEGER interval{
        .QuadPart = -1 * (duration.count() / 100u),
    };
    HANDLE timer = ::CreateWaitableTimer(NULL, TRUE, NULL);
    SetWaitableTimer(timer, &interval, 0, NULL, NULL, 0);
    const auto ret = WaitForSingleObjectEx(timer, INFINITE, interruptible);
    ::CloseHandle(timer);

    if (remaining) {
        const auto end_sleep = std::chrono::high_resolution_clock::now();
        const auto sleep_time = end_sleep - begin_sleep;
        *remaining = duration > sleep_time ? duration - sleep_time : std::chrono::nanoseconds(0);
    }
    return ret == WAIT_OBJECT_0;
}

#else

void SetCurrentThreadPriority(ThreadPriority new_priority) {
    pthread_t this_thread = pthread_self();

    const auto scheduling_type = SCHED_OTHER;
    s32 max_prio = sched_get_priority_max(scheduling_type);
    s32 min_prio = sched_get_priority_min(scheduling_type);
    u32 level = std::max(static_cast<u32>(new_priority) + 1, 4U);

    struct sched_param params;
    if (max_prio > min_prio) {
        params.sched_priority = min_prio + ((max_prio - min_prio) * level) / 4;
    } else {
        params.sched_priority = min_prio - ((min_prio - max_prio) * level) / 4;
    }

    pthread_setschedparam(this_thread, scheduling_type, &params);
}

bool AccurateSleep(const std::chrono::nanoseconds duration, std::chrono::nanoseconds* remaining,
                   const bool interruptible) {
    timespec request = {
        .tv_sec = duration.count() / 1'000'000'000,
        .tv_nsec = duration.count() % 1'000'000'000,
    };
    timespec remain;
    int ret;
    while ((ret = nanosleep(&request, &remain)) < 0 && errno == EINTR) {
        if (interruptible) {
            break;
        }
        request = remain;
    }
    if (remaining) {
        *remaining = std::chrono::nanoseconds(remain.tv_sec * 1'000'000'000 + remain.tv_nsec);
    }
    return ret == 0 || errno != EINTR;
}

#endif

#ifdef _WIN32

// Sets the debugger-visible name of the current thread.
void SetCurrentThreadName(const char* name) {
    if (Libraries::Kernel::g_curthread) {
        Libraries::Kernel::g_curthread->name = name;
    }
    SetThreadDescription(GetCurrentThread(), UTF8ToUTF16W(name).data());
}

void SetThreadName(void* thread, const char* name) {
    SetThreadDescription(thread, UTF8ToUTF16W(name).data());
}

#else // !_WIN32, so must be POSIX threads

// MinGW with the POSIX threading model does not support pthread_setname_np
#if !defined(_WIN32) || defined(_MSC_VER)
void SetCurrentThreadName(const char* name) {
    if (Libraries::Kernel::g_curthread) {
        Libraries::Kernel::g_curthread->name = name;
    }
#ifdef __APPLE__
    pthread_setname_np(name);
#elif defined(__Bitrig__) || defined(__DragonFly__) || defined(__FreeBSD__) || defined(__OpenBSD__)
    pthread_set_name_np(pthread_self(), name);
#elif defined(__NetBSD__)
    pthread_setname_np(pthread_self(), "%s", (void*)name);
#elif defined(__linux__)
    // Linux limits thread names to 15 characters and will outright reject any
    // attempt to set a longer name with ERANGE.
    std::string truncated(name, std::min(strlen(name), static_cast<std::size_t>(15)));
    if (int e = pthread_setname_np(pthread_self(), truncated.c_str())) {
        errno = e;
        LOG_ERROR(Common, "Failed to set thread name to '{}': {}", truncated, GetLastErrorMsg());
    }
#else
    pthread_setname_np(pthread_self(), name);
#endif
}

void SetThreadName(void* thread, const char* name) {
    // TODO
}
#endif

#if defined(_WIN32)
void SetCurrentThreadName(const char* name) {
    if (Libraries::Kernel::g_curthread) {
        Libraries::Kernel::g_curthread->name = name;
    }
    // Do Nothing on MinGW
}

void SetThreadName(void* thread, const char* name) {
    // Do Nothing on MinGW
}
#endif

#endif

AccurateTimer::AccurateTimer(std::chrono::nanoseconds target_interval)
    : target_interval(target_interval) {}

void AccurateTimer::Start() {
    const auto begin_sleep = std::chrono::high_resolution_clock::now();
    if (total_wait.count() > 0) {
        AccurateSleep(total_wait, nullptr, false);
    }
    start_time = std::chrono::high_resolution_clock::now();
    total_wait -= std::chrono::duration_cast<std::chrono::nanoseconds>(start_time - begin_sleep);
}

void AccurateTimer::End() {
    auto now = std::chrono::high_resolution_clock::now();
    total_wait +=
        target_interval - std::chrono::duration_cast<std::chrono::nanoseconds>(now - start_time);
}

#ifdef _WIN32
namespace {
// Efficiency class per logical processor index from the CPU set list (P-cores report the
// highest class on hybrid parts). Index 0..63 only; enough for this machine (20 threads).
struct CpuClasses {
    std::array<u8, 64> cls{};
    std::array<ULONG, 64> ids{};
    u8 max_class{};
    size_t count{};
};
const CpuClasses& GetCpuClasses() {
    static const CpuClasses classes = [] {
        CpuClasses c{};
        ULONG len = 0;
        GetSystemCpuSetInformation(nullptr, 0, &len, GetCurrentProcess(), 0);
        std::vector<u8> buf(len);
        auto* info = reinterpret_cast<PSYSTEM_CPU_SET_INFORMATION>(buf.data());
        if (!GetSystemCpuSetInformation(info, len, &len, GetCurrentProcess(), 0)) {
            return c;
        }
        for (ULONG off = 0; off < len;) {
            auto* e = reinterpret_cast<PSYSTEM_CPU_SET_INFORMATION>(buf.data() + off);
            if (e->Type == CpuSetInformation && e->CpuSet.LogicalProcessorIndex < 64) {
                const auto idx = e->CpuSet.LogicalProcessorIndex;
                c.cls[idx] = e->CpuSet.EfficiencyClass;
                c.ids[idx] = e->CpuSet.Id;
                c.max_class = std::max<u8>(c.max_class, e->CpuSet.EfficiencyClass);
                c.count = std::max<size_t>(c.count, idx + 1);
            }
            off += e->Size;
        }
        return c;
    }();
    return classes;
}
} // namespace

void ApplyCommandProcessorPlacement() {
    const char* value = std::getenv("SHADPS4_CP_QOS");
    const int mode = value ? value[0] - '0' : 0;
    if (mode <= 0) {
        return;
    }
    PROCESS_POWER_THROTTLING_STATE throttling{};
    throttling.Version = PROCESS_POWER_THROTTLING_CURRENT_VERSION;
    throttling.ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED;
    throttling.StateMask = 0; // always high QoS
    const bool proc_ok = SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling,
                                               &throttling, sizeof(throttling));
    THREAD_POWER_THROTTLING_STATE thread_throttling{};
    thread_throttling.Version = THREAD_POWER_THROTTLING_CURRENT_VERSION;
    thread_throttling.ControlMask = THREAD_POWER_THROTTLING_EXECUTION_SPEED;
    thread_throttling.StateMask = 0;
    const bool thread_ok = SetThreadInformation(GetCurrentThread(), ThreadPowerThrottling,
                                                &thread_throttling, sizeof(thread_throttling));
    const bool prio_ok = SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
    bool sets_ok = true;
    size_t p_cores = 0;
    if (mode >= 2) {
        const auto& c = GetCpuClasses();
        std::vector<ULONG> ids;
        for (size_t i = 0; i < c.count; ++i) {
            if (c.cls[i] == c.max_class) {
                ids.push_back(c.ids[i]);
            }
        }
        p_cores = ids.size();
        sets_ok = !ids.empty() &&
                  SetThreadSelectedCpuSets(GetCurrentThread(), ids.data(), ULONG(ids.size()));
    }
    LOG_INFO(Common, "SHADPS4_CP_QOS={}: process throttling off {}, thread {}, priority {}, "
                     "P-core set {} ({} logical)",
             mode, proc_ok, thread_ok, prio_ok, sets_ok, p_cores);
}

void SampleCommandProcessorCore() {
    static const bool enabled = [] {
        const char* value = std::getenv("SHADPS4_CP_CORE_STATS");
        return value && value[0] == '1';
    }();
    if (!enabled) {
        return;
    }
    static std::array<u64, 64> hits{};
    static u64 samples = 0;
    const DWORD cpu = GetCurrentProcessorNumber();
    ++hits[cpu & 63];
    if (++samples % 200000 != 0) {
        return;
    }
    const auto& c = GetCpuClasses();
    u64 p = 0;
    std::string top;
    for (size_t i = 0; i < 64; ++i) {
        if (hits[i] && c.cls[i] == c.max_class) {
            p += hits[i];
        }
        if (hits[i] * 20 >= samples) {
            top += fmt::format(" cpu{}(c{}):{:.0f}%", i, c.cls[i], 100.0 * hits[i] / samples);
        }
    }
    LOG_INFO(Common, "CP_CORE_STATS: samples {} on P-cores {:.1f}%{}", samples,
             100.0 * p / samples, top);
    hits.fill(0);
    samples = 0;
}
#else
void ApplyCommandProcessorPlacement() {}
void SampleCommandProcessorCore() {}
#endif

std::string GetCurrentThreadName() {
    using namespace Libraries::Kernel;
    if (g_curthread && !g_curthread->name.empty()) {
        return g_curthread->name;
    }
#ifdef _WIN32
    PWSTR name = nullptr;
    if (FAILED(GetThreadDescription(GetCurrentThread(), &name)) || !name) {
        return "<unknown name>";
    }
    std::string result = Common::UTF16ToUTF8(name);
    LocalFree(name);
    return result;
#else
    char name[256];
    if (pthread_getname_np(pthread_self(), name, sizeof(name)) != 0) {
        return "<unknown name>";
    }
    return std::string{name};
#endif
}

} // namespace Common
