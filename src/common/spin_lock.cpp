// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/spin_lock.h"

#if _MSC_VER
#include <intrin.h>
#if _M_AMD64
#define __x86_64__ 1
#endif
#if _M_ARM64
#define __aarch64__ 1
#endif
#else
#if __x86_64__
#include <xmmintrin.h>
#endif
#endif

namespace {

void ThreadPause() {
#if __x86_64__
    _mm_pause();
#elif __aarch64__ && _MSC_VER
    __yield();
#elif __aarch64__
    asm("yield");
#endif
}

} // Anonymous namespace

#include <cstdlib>

namespace {

/// SHADPS4_SPINLOCK_PARK=N (default off; "1" = 512): a waiter spins N pauses, then parks on the
/// lock word (std::atomic::wait = WaitOnAddress) until the holder's unlock wakes it. Measured
/// 2026-10-15 (4c/8t, s1166): guest job threads spent 25% of their wall time in lock() on region
/// locks; a preempted holder left the waiters spinning whole quanta on the cores it needed.
/// Read once, on first use (a function-local static: no static-init-order window in which a
/// lock could be taken in one mode and released in the other).
unsigned ParkSpins() {
    static const unsigned spins = [] {
        const char* value = std::getenv("SHADPS4_SPINLOCK_PARK");
        if (value == nullptr || value[0] == '0' || value[0] == '\0') {
            return 0u;
        }
        const unsigned long n = std::strtoul(value, nullptr, 10);
        return n <= 1 ? 512u : static_cast<unsigned>(n > 1u << 20 ? 1u << 20 : n);
    }();
    return spins;
}

} // Anonymous namespace

namespace Common {

void SpinLock::lock() {
    unsigned expected = 0;
    if (state.compare_exchange_strong(expected, 1, std::memory_order_acquire,
                                      std::memory_order_relaxed)) [[likely]] {
        return;
    }
    if (const unsigned spins = ParkSpins(); spins != 0) {
        LockPark(spins);
        return;
    }
    while (state.exchange(1, std::memory_order_acquire) != 0) {
        ThreadPause();
    }
}

void SpinLock::LockPark(unsigned spins) {
    for (unsigned i = 0; i < spins; ++i) {
        ThreadPause();
        unsigned expected = 0;
        if (state.load(std::memory_order_relaxed) == 0 &&
            state.compare_exchange_weak(expected, 1, std::memory_order_acquire,
                                        std::memory_order_relaxed)) {
            return;
        }
    }
    // Drepper, "Futexes Are Tricky", mutex 3: whoever takes the lock from here marks it 2, so the
    // unlock that follows wakes one parked waiter (a spurious wake only re-checks).
    while (state.exchange(2, std::memory_order_acquire) != 0) {
        state.wait(2, std::memory_order_relaxed);
    }
}

void SpinLock::unlock() {
    if (ParkSpins() != 0) {
        if (state.exchange(0, std::memory_order_release) == 2) {
            state.notify_one();
        }
        return;
    }
    state.store(0, std::memory_order_release);
}

bool SpinLock::try_lock() {
    unsigned expected = 0;
    return state.compare_exchange_strong(expected, 1, std::memory_order_acquire,
                                         std::memory_order_relaxed);
}

} // namespace Common
