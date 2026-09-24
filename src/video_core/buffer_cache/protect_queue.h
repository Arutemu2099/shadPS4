// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
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
            pending.insert(pending.end(), items.begin(), items.end());
            enqueued.store(enqueued.load(std::memory_order_relaxed) + 1,
                           std::memory_order_release);
        }
        cv.notify_one();
        items.clear();
    }

    /// Waits until every protection queued so far is applied. Cheap when the queue is idle.
    void Fence() const {
        const u64 target = enqueued.load(std::memory_order_acquire);
        if (applied.load(std::memory_order_acquire) >= target) {
            return;
        }
        u32 spins = 0;
        while (applied.load(std::memory_order_acquire) < target) {
            if (++spins < 256) {
                _mm_pause();
            } else {
                std::this_thread::yield();
            }
        }
    }

private:
    void Loop(std::stop_token stop) {
        Common::SetCurrentThreadName("shadPS4:Protector");
        std::vector<DeferredWatch> work;
        while (!stop.stop_requested()) {
            u64 seq;
            {
                std::unique_lock lk{mutex};
                cv.wait(lk, stop, [this] { return !pending.empty(); });
                if (stop.stop_requested()) {
                    return;
                }
                work.swap(pending);
                seq = enqueued.load(std::memory_order_relaxed);
            }
            ApplyDeferredWatches(work);
            applied.store(seq, std::memory_order_release);
        }
    }

    std::mutex mutex;
    std::condition_variable_any cv;
    std::vector<DeferredWatch> pending;
    std::atomic<u64> enqueued{0};
    std::atomic<u64> applied{0};
    std::jthread thread;
};

/// Called by the command processor before it makes any guest-visible completion signal.
inline void ProtectFence() {
    if (ProtectQueue::Enabled()) {
        ProtectQueue::Instance().Fence();
    }
}

} // namespace VideoCore
