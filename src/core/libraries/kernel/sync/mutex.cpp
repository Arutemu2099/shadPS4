// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "mutex.h"

#include <cstdlib>
#include <string_view>
#include <thread>

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/switch_mutex.h"

#ifdef _WIN64
#pragma comment(lib, "synchronization.lib")
#endif

namespace Libraries::Kernel {

#ifdef _WIN64
bool TimedMutex::UserMode() {
    static const bool enabled = [] {
        const char* value = std::getenv("SHADPS4_USER_MUTEX");
        return value != nullptr && std::string_view{value} == "1";
    }();
    return enabled;
}

bool TimedMutex::LockSlow(u32 timeout_ms) {
    // Brief spin first: guest critical sections are usually short.
    for (int i = 0; i < 64; ++i) {
        u32 expected = 0;
        if (state.load(std::memory_order_relaxed) == 0 &&
            state.compare_exchange_weak(expected, 1, std::memory_order_acquire,
                                        std::memory_order_relaxed)) {
            return true;
        }
        YieldProcessor();
    }
    // Mark contended; whoever takes it from here also leaves it marked, so unlock wakes.
    if (state.exchange(2, std::memory_order_acquire) == 0) {
        return true;
    }
    u32 contended = 2;
    WaitOnAddress(&state, &contended, sizeof(contended), timeout_ms);
    return state.exchange(2, std::memory_order_acquire) == 0;
}
#endif

TimedMutex::TimedMutex() {
#ifdef _WIN64
    if (!UserMode()) {
        mtx = CreateMutex(nullptr, false, nullptr);
        ASSERT(mtx);
    }
#endif
}

TimedMutex::~TimedMutex() {
#ifdef _WIN64
    if (mtx) {
        CloseHandle(mtx);
    }
#endif
}

void TimedMutex::lock() {
#ifdef _WIN64
    if (UserMode()) {
        if (try_lock()) {
            return;
        }
        while (!LockSlow(INFINITE)) {
        }
        return;
    }
    for (;;) {
        u64 res = WaitForSingleObjectEx(mtx, INFINITE, true);
        if (res == WAIT_OBJECT_0) {
            return;
        }
    }
#else
    mtx.lock();
#endif
}

bool TimedMutex::try_lock() {
#ifdef _WIN64
    if (UserMode()) {
        u32 expected = 0;
        return state.compare_exchange_strong(expected, 1, std::memory_order_acquire,
                                             std::memory_order_relaxed);
    }
    return WaitForSingleObjectEx(mtx, 0, true) == WAIT_OBJECT_0;
#else
    return mtx.try_lock();
#endif
}

void TimedMutex::unlock() {
#ifdef _WIN64
    if (UserMode()) {
        if (state.exchange(0, std::memory_order_release) == 2) {
            WakeByAddressSingle(&state);
        }
        return;
    }
    ReleaseMutex(mtx);
#else
    mtx.unlock();
#endif
}

} // namespace Libraries::Kernel

namespace Common {

void FutexWait(std::atomic<u32>* address, u32 value) {
#ifdef _WIN64
    WaitOnAddress(address, &value, sizeof(value), INFINITE);
#endif
}

void FutexWakeOne(std::atomic<u32>* address) {
#ifdef _WIN64
    WakeByAddressSingle(address);
#endif
}

bool FlushProcessWriteBuffersAll() {
#ifdef _WIN64
    FlushProcessWriteBuffers();
    return true;
#else
    return false;
#endif
}

void ReportMutexReentry(const char* name, void* outer_caller, void* inner_caller) {
    static std::atomic<u64> events{0};
    const u64 n = events.fetch_add(1, std::memory_order_relaxed) + 1;
    if (n > 20 && (n & (n - 1)) != 0) {
        return;
    }
#ifdef _WIN64
    const auto base = reinterpret_cast<uintptr_t>(GetModuleHandleW(nullptr));
#else
    const uintptr_t base = 0;
#endif
    // Offsets from the EXE base: symbolize with the archived PDB (llvm-symbolizer --obj).
    LOG_ERROR(Kernel, "SwitchMutex re-entry #{} ({}) tid={} outer=exe+{:#x} inner=exe+{:#x}", n, name,
              GetCurrentThreadId(), reinterpret_cast<uintptr_t>(outer_caller) - base,
              reinterpret_cast<uintptr_t>(inner_caller) - base);
}

#ifdef _WIN64
static_assert(sizeof(SRWLOCK) == sizeof(void*));
void SrwLockExclusive(void** lock) {
    AcquireSRWLockExclusive(reinterpret_cast<PSRWLOCK>(lock));
}
bool SrwTryLockExclusive(void** lock) {
    return TryAcquireSRWLockExclusive(reinterpret_cast<PSRWLOCK>(lock)) != 0;
}
void SrwUnlockExclusive(void** lock) {
    ReleaseSRWLockExclusive(reinterpret_cast<PSRWLOCK>(lock));
}
#else
void SrwLockExclusive(void**) {}
bool SrwTryLockExclusive(void**) {
    return false;
}
void SrwUnlockExclusive(void**) {}
#endif

} // namespace Common
