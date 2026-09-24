// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <cstdlib>
#include <mutex>

#include <immintrin.h>

#include "common/types.h"

namespace Common {

/// Defined in core/libraries/kernel/sync/mutex.cpp (keeps <windows.h> out of this header).
void FutexWait(std::atomic<u32>* address, u32 value);
void FutexWakeOne(std::atomic<u32>* address);

/**
 * A mutex that is either std::mutex or an inline futex lock (0 free, 1 held, 2 held with
 * waiters; waiters sleep on WaitOnAddress), chosen once per process by an environment flag.
 * The futex path costs one atomic operation for an uncontended lock or unlock instead of a call
 * through the runtime into the SRW lock. BasicLockable, so scoped_lock/unique_lock work; not for
 * use with condition variables.
 */
template <const char* FlagName>
class SwitchMutex {
public:
    static bool Fast() {
        static const bool fast = [] {
#ifdef _WIN32
            const char* value = std::getenv(FlagName);
            return value != nullptr && value[0] == '1';
#else
            return false;
#endif
        }();
        return fast;
    }

    void lock() {
        if (!Fast()) {
            slow.lock();
            return;
        }
        u32 expected = 0;
        if (state.compare_exchange_strong(expected, 1, std::memory_order_acquire,
                                          std::memory_order_relaxed)) {
            return;
        }
        LockContended();
    }

    bool try_lock() {
        if (!Fast()) {
            return slow.try_lock();
        }
        u32 expected = 0;
        return state.compare_exchange_strong(expected, 1, std::memory_order_acquire,
                                             std::memory_order_relaxed);
    }

    void unlock() {
        if (!Fast()) {
            slow.unlock();
            return;
        }
        if (state.exchange(0, std::memory_order_release) == 2) {
            FutexWakeOne(&state);
        }
    }

private:
    void LockContended() {
        for (int i = 0; i < 128; ++i) {
            _mm_pause();
            u32 expected = 0;
            if (state.load(std::memory_order_relaxed) == 0 &&
                state.compare_exchange_weak(expected, 1, std::memory_order_acquire,
                                            std::memory_order_relaxed)) {
                return;
            }
        }
        while (state.exchange(2, std::memory_order_acquire) != 0) {
            FutexWait(&state, 2);
        }
    }

    std::atomic<u32> state{0};
    std::mutex slow;
};

inline constexpr char FastTexMutexFlag[] = "SHADPS4_FAST_TEX_MUTEX";

} // namespace Common
