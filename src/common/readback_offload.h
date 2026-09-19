// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <string_view>

#include "common/logging/log.h"
#include "common/types.h"

/**
 * Flags and diagnostics for the experimental readback wait offload in BufferCache::ReadMemory.
 *
 *   SHADPS4_READBACK_OFFLOAD=1          guest-origin faults wait for the GPU on their own thread
 *   SHADPS4_READBACK_OFFLOAD_VERIFY=1   second GPU readback compared before publication
 *   SHADPS4_READBACK_OFFLOAD_PROFILE=1  per-origin counters, wait ns and a rare log line
 *
 * All default off. With PROFILE=0 no clock is read and no counter is touched. Counters are
 * relaxed atomics because faults arrive on guest threads and on the command processor.
 */
namespace Common::ReadbackOffload {

inline bool EnvFlag(const char* name) {
    const char* env = std::getenv(name);
    return env != nullptr && std::string_view{env} == "1";
}

inline bool Enabled() {
    static const bool value = EnvFlag("SHADPS4_READBACK_OFFLOAD");
    return value;
}

inline bool VerifyEnabled() {
    static const bool value = EnvFlag("SHADPS4_READBACK_OFFLOAD_VERIFY");
    return value;
}

inline bool ProfileEnabled() {
    static const bool value = EnvFlag("SHADPS4_READBACK_OFFLOAD_PROFILE");
    return value;
}

/// Which thread took the fault. Anything that is not the command processor counts as Guest.
enum class Origin : u32 {
    CpSelf = 0,
    Guest = 1,
};

/// One synchronous download's submit-and-wait, split. total_ns is what sync_wait_ns has always
/// reported; submit_ns is Scheduler::Flush and gpu_wait_ns is Scheduler::Wait on that tick.
struct SyncTiming {
    u64 total_ns{};
    u64 submit_ns{};
    u64 gpu_wait_ns{};
};

struct Counters {
    std::array<std::atomic<u64>, 2> faults{};
    /// Every synchronous download of that origin that reached the GPU, fallbacks included.
    std::array<std::atomic<u64>, 2> sync_downloads{};
    std::array<std::atomic<u64>, 2> sync_wait_ns{};
    std::array<std::atomic<u64>, 2> sync_submit_ns{};
    std::array<std::atomic<u64>, 2> sync_gpu_wait_ns{};
    std::atomic<u64> jobs{};
    std::atomic<u64> successes{};
    std::atomic<u64> seq_vetoes{};
    std::atomic<u64> list_vetoes{};
    /// Vetoed jobs and capacity misses. Both are also counted in sync_downloads[Guest], and
    /// capacity_wait_ns is likewise a subset of sync_wait_ns[Guest].
    std::atomic<u64> sync_fallbacks{};
    std::atomic<u64> capacity_fallbacks{};
    std::atomic<u64> mapping_fallbacks{};
    std::atomic<u64> capacity_wait_ns{};
    std::atomic<u64> prepare_empty{};
    std::atomic<u64> cp_settles{};
    /// GPU wait the offload moved off the command processor, against cp_settle_wait_ns, the
    /// wait it took back there because something overlapped a job before its owner returned.
    std::atomic<u64> guest_wait_ns{};
    std::atomic<u64> cp_settle_wait_ns{};
    /// Command processor hops an invalidation made to settle jobs before touching the tracker.
    std::atomic<u64> invalidate_settle_hops{};
    /// Invalidations whose GPU data a job had already published, CPU-marked without a download.
    std::atomic<u64> write_marks{};
    /// Snapshots dropped because an image was tiled into the buffer without a tracker marking.
    std::atomic<u64> alias_bumps{};
    std::atomic<u64> verify_checks{};
    /// Jobs vetoed because no independent second readback could be made. Never published.
    std::atomic<u64> verify_skips{};
    std::atomic<u64> verify_mismatches{};
    std::atomic<u64> next_log_ns{};
};

inline Counters& State() {
    static Counters counters;
    return counters;
}

inline u64 NowNs() {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count());
}

/// Timestamp for a profiled interval; 0 and no clock read when profiling is off.
inline u64 ProfileNowNs() {
    return ProfileEnabled() ? NowNs() : 0;
}

inline void Add(std::atomic<u64>& counter, u64 value = 1) {
    if (ProfileEnabled()) {
        counter.fetch_add(value, std::memory_order_relaxed);
    }
}

inline void CountFault(Origin origin) {
    Add(State().faults[static_cast<u32>(origin)]);
}

/// A synchronous resolve that reached the GPU, and the ns its submit and its wait took.
/// An empty download returns before either, leaves the timing at 0 and is not counted here.
inline void CountSyncResolve(Origin origin, const SyncTiming& timing) {
    if (timing.total_ns != 0) {
        const u32 index = static_cast<u32>(origin);
        Add(State().sync_downloads[index]);
        Add(State().sync_wait_ns[index], timing.total_ns);
        Add(State().sync_submit_ns[index], timing.submit_ns);
        Add(State().sync_gpu_wait_ns[index], timing.gpu_wait_ns);
    }
}

/// Adds the ns since start_ns, a ProfileNowNs value, to counter.
inline void AddSince(std::atomic<u64>& counter, u64 start_ns) {
    if (ProfileEnabled()) {
        counter.fetch_add(NowNs() - start_ns, std::memory_order_relaxed);
    }
}

/// At most one line per ten seconds, from whichever thread wins the exchange.
inline void MaybeLog() {
    if (!ProfileEnabled()) {
        return;
    }
    constexpr u64 period_ns = 10'000'000'000ULL;
    auto& s = State();
    const u64 now = NowNs();
    u64 next = s.next_log_ns.load(std::memory_order_relaxed);
    if (now < next) {
        return;
    }
    if (!s.next_log_ns.compare_exchange_strong(next, now + period_ns,
                                               std::memory_order_relaxed)) {
        return;
    }
    if (next == 0) {
        // First fault only arms the timer.
        return;
    }
    const auto get = [](const std::atomic<u64>& counter) {
        return counter.load(std::memory_order_relaxed);
    };
    // Existing fields keep their names and order; new ones are appended. Field names stay
    // [a-z_] so a key=value scan of the line needs no change.
    LOG_INFO(Render,
             "ReadbackOffload: on={} cp_faults={} cp_sync={} cp_sync_wait_ns={} guest_faults={} "
             "guest_sync={} guest_sync_wait_ns={} jobs={} ok={} seq_veto={} list_veto={} "
             "sync_fallback={} cap_fallback={} cap_wait_ns={} empty={} cp_settles={} "
             "guest_wait_ns={} verify={} verify_skip={} verify_bad={} cp_sync_submit_ns={} "
             "cp_sync_gpu_wait_ns={} guest_sync_submit_ns={} guest_sync_gpu_wait_ns={} "
             "cp_settle_wait_ns={} inval_hops={} write_marks={} alias_bumps={} map_fallback={}",
             Enabled(), get(s.faults[0]), get(s.sync_downloads[0]), get(s.sync_wait_ns[0]),
             get(s.faults[1]), get(s.sync_downloads[1]), get(s.sync_wait_ns[1]), get(s.jobs),
             get(s.successes), get(s.seq_vetoes), get(s.list_vetoes), get(s.sync_fallbacks),
             get(s.capacity_fallbacks), get(s.capacity_wait_ns), get(s.prepare_empty),
             get(s.cp_settles), get(s.guest_wait_ns), get(s.verify_checks), get(s.verify_skips),
             get(s.verify_mismatches), get(s.sync_submit_ns[0]), get(s.sync_gpu_wait_ns[0]),
             get(s.sync_submit_ns[1]), get(s.sync_gpu_wait_ns[1]), get(s.cp_settle_wait_ns),
             get(s.invalidate_settle_hops), get(s.write_marks), get(s.alias_bumps),
             get(s.mapping_fallbacks));
}

} // namespace Common::ReadbackOffload
