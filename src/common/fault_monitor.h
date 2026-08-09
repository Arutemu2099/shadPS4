// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <chrono>
#include "common/types.h"

/**
 * Periodic reporting of handled-fault rates against presented frames.
 *
 * Windows dispatches every emulator-handled access violation through the kernel, which is
 * expensive. Two independent sources exist, and they are counted separately because they mean
 * different things:
 *
 *   - read faults  - only ever produced when GPU readbacks are set to Precise, which is the only
 *                    mode that read-protects guest pages holding GPU-modified data. A guest read
 *                    of such a page traps so the emulator can download the data first.
 *   - write faults - guest writes to GPU-tracked pages. Present in every readback mode.
 *   - jit faults   - guest `fs:`-segment accesses patched on demand; unrelated to readbacks.
 *
 * Reporting frames alongside them makes "the fault rate climbs while the framerate falls" a
 * single readable line rather than a correlation across two instruments.
 */
namespace Common::FaultMonitor {

// Incremented on the faulting thread; relaxed ordering, never read there.
extern std::atomic<u64> guest_read_faults;
extern std::atomic<u64> guest_read_unhandled;
extern std::atomic<u64> guest_read_ns;
extern std::atomic<u64> guest_write_faults;
extern std::atomic<u64> guest_write_unhandled;
extern std::atomic<u64> guest_write_ns;
extern std::atomic<u64> jit_patch_faults;
extern std::atomic<u64> jit_patch_ns;
extern std::atomic<u64> presented_frames;

/**
 * Time spent inside the handler is measured as well as the number of entries, because the two
 * answer different questions. If the framerate decays while faults-per-frame stays flat, then
 * either each fault got more expensive (handler nanoseconds per fault rises) or the slowdown is
 * not in this path at all (nanoseconds per fault stays flat). Counting alone cannot separate
 * those; this is a two-clock-reads-per-fault cost to be able to.
 */
using Clock = std::chrono::steady_clock;

inline Clock::time_point Now() {
    return Clock::now();
}

inline u64 SinceNs(Clock::time_point start) {
    return static_cast<u64>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count());
}

inline void CountGuestReadFault(bool handled, u64 ns) {
    guest_read_faults.fetch_add(1, std::memory_order_relaxed);
    guest_read_ns.fetch_add(ns, std::memory_order_relaxed);
    if (!handled) {
        guest_read_unhandled.fetch_add(1, std::memory_order_relaxed);
    }
}

inline void CountGuestWriteFault(bool handled, u64 ns) {
    guest_write_faults.fetch_add(1, std::memory_order_relaxed);
    guest_write_ns.fetch_add(ns, std::memory_order_relaxed);
    if (!handled) {
        guest_write_unhandled.fetch_add(1, std::memory_order_relaxed);
    }
}

inline void CountJitPatchFault(u64 ns) {
    jit_patch_faults.fetch_add(1, std::memory_order_relaxed);
    jit_patch_ns.fetch_add(ns, std::memory_order_relaxed);
}

inline void CountPresentedFrame() {
    presented_frames.fetch_add(1, std::memory_order_relaxed);
}

/// Starts the reporter thread. Safe to call more than once; only the first call has an effect.
/// Writes one line every interval to the log and to a CSV in the log directory.
void Start();

} // namespace Common::FaultMonitor
