// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <chrono>
#include <immintrin.h>
#include <condition_variable>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <vector>

#include "common/thread.h"
#include "video_core/buffer_cache/region_manager.h"

namespace VideoCore {

/**
 * SHADPS4_DEFER_READ_PROTECT=1: the read protection the command processor applies when a draw or
 * dispatch binds a buffer for GPU writes (precise readbacks) is handed to a dedicated thread.
 * Measured 2026-09-24 on 386F5167: those VirtualProtect calls were 4.2% of the CP's wall time.
 *
 * The tracker bits change on the CP at once; only the OS protection lags. Guest code may only
 * read GPU results after a signal the CP writes (EOS/EOP/RELEASE_MEM labels, WRITE_DATA,
 * interrupts, GPU idle), and the CP calls Fence() before each of those, which waits until every
 * queued protection has been applied. Watcher counts commute (see DeferredWatch), so applying
 * these after later updates of the same pages leaves the same final state.
 */
class ProtectQueue {
public:
    static bool Enabled() {
        static const bool enabled = [] {
            const char* env = std::getenv("SHADPS4_DEFER_READ_PROTECT");
            return env != nullptr && env[0] == '1';
        }();
        return enabled;
    }

    /**
     * SHADPS4_PROTECT_WAKE_STATS=1: counters for the hand-off, logged every 512 presented frames.
     * Measured 2026-09-25 (a19): Push was 0.63% of the CP and the yield loop in Fence 1.17%. The
     * question is whether a waited fence mostly waits for the Protector to wake (then the CP
     * should apply an unclaimed batch itself) or for VirtualProtect (then that would only move the
     * time onto the CP): compare wait per waited fence with apply time per batch, and see how
     * many waits began before the Protector had claimed the batch the fence needs.
     */
    static bool WakeStatsEnabled() {
        const char* env = std::getenv("SHADPS4_PROTECT_WAKE_STATS");
        return env != nullptr && env[0] == '1';
    }

    static ProtectQueue& Instance() {
        static ProtectQueue queue;
        return queue;
    }

    /// Queues the watcher updates and empties `items`. Command processor only.
    void Push(std::vector<DeferredWatch>& items) {
        if (items.empty()) {
            return;
        }
        {
            std::scoped_lock lk{mutex};
            if (!thread.joinable()) {
                thread = std::jthread([this](std::stop_token stop) { Loop(stop); });
            }
            if (wake_stats) [[unlikely]] {
                CountPush(items.size());
            }
            pending.insert(pending.end(), items.begin(), items.end());
            enqueued.store(enqueued.load(std::memory_order_relaxed) + 1,
                           std::memory_order_release);
        }
        cv.notify_one();
        items.clear();
        if (wake_stats) [[unlikely]] {
            MaybeLogStats();
        }
    }

    /// Waits until every protection queued so far is applied. Cheap when the queue is idle.
    void Fence() const {
        if (wake_stats) [[unlikely]] {
            return FenceCounted();
        }
        const u64 target = enqueued.load(std::memory_order_acquire);
        if (applied.load(std::memory_order_acquire) >= target) {
            return;
        }
        WaitApplied(target);
    }

private:
    using Clock = std::chrono::steady_clock;

    void WaitApplied(u64 target) const {
        u32 spins = 0;
        while (applied.load(std::memory_order_acquire) < target) {
            if (++spins < 256) {
                _mm_pause();
            } else {
                std::this_thread::yield();
            }
        }
    }

    void Loop(std::stop_token stop) {
        Common::SetCurrentThreadName("shadPS4:Protector");
        std::vector<DeferredWatch> work;
        while (!stop.stop_requested()) {
            u64 seq;
            Clock::time_point first_push{};
            {
                std::unique_lock lk{mutex};
                if (wake_stats) [[unlikely]] {
                    sleeping = pending.empty();
                }
                cv.wait(lk, stop, [this] { return !pending.empty(); });
                if (stop.stop_requested()) {
                    return;
                }
                work.swap(pending);
                seq = enqueued.load(std::memory_order_relaxed);
                if (wake_stats) [[unlikely]] {
                    sleeping = false;
                    first_push = batch_first_push;
                    stats.claimed.store(seq, std::memory_order_release);
                }
            }
            if (wake_stats) [[unlikely]] {
                const auto start = Clock::now();
                const u64 items = work.size();
                ApplyDeferredWatches(work);
                stats.batches.fetch_add(1, std::memory_order_relaxed);
                stats.batch_items.fetch_add(items, std::memory_order_relaxed);
                stats.pickup_ns.fetch_add(Ns(first_push, start), std::memory_order_relaxed);
                stats.apply_ns.fetch_add(Ns(start, Clock::now()), std::memory_order_relaxed);
            } else {
                ApplyDeferredWatches(work);
            }
            applied.store(seq, std::memory_order_release);
        }
    }

    static u64 Ns(Clock::time_point start, Clock::time_point end) {
        return static_cast<u64>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
    }

    /// Under `mutex`. `sleeping` says whether this notify will wake a blocked Protector (cut (a)
    /// would skip the others); the first push into an empty queue stamps the batch.
    void CountPush(u64 items) {
        stats.push_calls.fetch_add(1, std::memory_order_relaxed);
        stats.push_items.fetch_add(items, std::memory_order_relaxed);
        stats.push_wakes.fetch_add(sleeping ? 1 : 0, std::memory_order_relaxed);
        sleeping = false; // later pushes before the Protector runs wake nobody
        if (pending.empty()) {
            batch_first_push = Clock::now();
        }
    }

    void FenceCounted() const {
        const u64 target = enqueued.load(std::memory_order_acquire);
        stats.fence_calls.fetch_add(1, std::memory_order_relaxed);
        if (applied.load(std::memory_order_acquire) < target) {
            // Unclaimed: the batch this fence needs is still in `pending`, so the wait includes
            // the Protector's wake-up, not just its VirtualProtect calls.
            const bool unclaimed = stats.claimed.load(std::memory_order_acquire) < target;
            const auto start = Clock::now();
            WaitApplied(target);
            stats.fence_waits.fetch_add(1, std::memory_order_relaxed);
            stats.unclaimed_waits.fetch_add(unclaimed ? 1 : 0, std::memory_order_relaxed);
            stats.wait_ns.fetch_add(Ns(start, Clock::now()), std::memory_order_relaxed);
        }
        MaybeLogStats();
    }

    /// Command processor only (Push and Fence callers).
    void MaybeLogStats() const {
        if ((++stats.log_checks & 0xff) != 0) {
            return;
        }
        const u64 frames = Common::FaultMonitor::presented_frames.load(std::memory_order_relaxed);
        if (stats.log_frames == 0 || frames < stats.log_frames + 512) {
            if (stats.log_frames == 0 && frames != 0) {
                // Start the first window at the first presented frame, not at boot.
                stats.log_frames = frames;
                stats.Take();
            }
            return;
        }
        const auto s = stats.Take();
        const u64 window = frames - stats.log_frames;
        const double n = static_cast<double>(window);
        stats.log_frames = frames;
        const auto per = [](u64 a, u64 b) { return b != 0 ? static_cast<double>(a) / b : 0.0; };
        const double wait_us = per(s.wait_ns, s.fence_waits) / 1000.0;
        const double apply_us = per(s.apply_ns, s.batches) / 1000.0;
        LOG_INFO(Render,
                 "ProtectWake: frames={} fence_calls/frame={:.1f} fence_waits/frame={:.2f} "
                 "wait_us/waited_fence={:.2f} unclaimed_waits={:.1f}% wait_us/frame={:.1f} | "
                 "batches/frame={:.2f} apply_us/batch={:.2f} items/batch={:.1f} "
                 "pickup_us/batch={:.2f} | push_calls/frame={:.1f} push_notifies/frame={:.1f} "
                 "notifies_waking={:.1f}% | wait_over_apply={:.2f}",
                 window, s.fence_calls / n, s.fence_waits / n, wait_us,
                 100.0 * per(s.unclaimed_waits, s.fence_waits), s.wait_ns / n / 1000.0,
                 s.batches / n, apply_us, per(s.batch_items, s.batches),
                 per(s.pickup_ns, s.batches) / 1000.0, s.push_calls / n, s.push_calls / n,
                 100.0 * per(s.push_wakes, s.push_calls), apply_us > 0 ? wait_us / apply_us : 0.0);
    }

    struct WakeStats {
        struct Snapshot {
            u64 fence_calls, fence_waits, unclaimed_waits, wait_ns;
            u64 push_calls, push_items, push_wakes;
            u64 batches, batch_items, apply_ns, pickup_ns;
        };
        // Written by the CP (fence_*, wait_ns, push_*) and the Protector (batch/apply/pickup).
        std::atomic<u64> fence_calls{0}, fence_waits{0}, unclaimed_waits{0}, wait_ns{0};
        std::atomic<u64> push_calls{0}, push_items{0}, push_wakes{0};
        std::atomic<u64> batches{0}, batch_items{0}, apply_ns{0}, pickup_ns{0};
        /// Highest `enqueued` value the Protector has swapped out of `pending`.
        std::atomic<u64> claimed{0};
        u64 log_checks = 0; // CP only
        u64 log_frames = 0; // CP only

        Snapshot Take() {
            const auto take = [](std::atomic<u64>& c) {
                return c.exchange(0, std::memory_order_relaxed);
            };
            return {take(fence_calls), take(fence_waits), take(unclaimed_waits), take(wait_ns),
                    take(push_calls),  take(push_items),  take(push_wakes),      take(batches),
                    take(batch_items), take(apply_ns),    take(pickup_ns)};
        }
    };

    std::mutex mutex;
    std::condition_variable_any cv;
    std::vector<DeferredWatch> pending;
    std::atomic<u64> enqueued{0};
    std::atomic<u64> applied{0};
    const bool wake_stats = WakeStatsEnabled();
    bool sleeping = false;                // under `mutex`, stats only
    Clock::time_point batch_first_push{}; // under `mutex`, stats only
    mutable WakeStats stats;
    std::jthread thread;
};

/// Called by the command processor before it makes any guest-visible completion signal.
inline void ProtectFence() {
    if (ProtectQueue::Enabled()) {
        ProtectQueue::Instance().Fence();
    }
}

} // namespace VideoCore
