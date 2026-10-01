// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstdlib>
#include <deque>
#include <memory>
#include <utility>
#include <vector>
#include <immintrin.h>

#include "common/types.h"
#include "video_core/buffer_cache/region_definitions.h"

namespace VideoCore {

/// SHADPS4_BACK_CLEAN=1 (=2 verify), default off.
///
/// The command processor asks the memory tracker "is any page of this range CPU-dirty?" for every
/// read binding of every draw (~49,000 times a frame in Second Son) and 99.7% of the answers are
/// "no". Each question reads tracker state that guest write-fault handlers modify on other cores
/// (region lock, dirty bits, summary flags: ~2,850 faults a frame), so the reader keeps paying
/// for cache lines it does not own, and takes the region lock whenever the region has any dirty
/// page. Measured 2026-09-30: MemoryTracker::ForEachUploadRange is 8.6% of the CP back's wall
/// time in the fire scene, 3.3 of it in the region spin lock.
///
/// The tracker stays as it is and stays authoritative. On top of it the back keeps a private
/// bit per page, "verified clean": set after the locked walk of that page found it clean or took
/// its upload, cleared when the page is marked dirty again. Whoever marks pages dirty appends the
/// page range to a log (DirtyLog) while holding the region lock and before setting the tracker
/// bits; the back drains the log before every question. A question whose pages are all verified
/// clean touches no shared tracker state except the log's tail counter; otherwise only the
/// unverified page runs take the locked walk.
///
/// Why this is the same answer the tracker gives (notes/render-state-memo-2026-10-01.md):
/// - A page turns dirty in exactly one place, RegionManager::ChangeRegionState<CPU, true>, always
///   under the region lock; the append sits in front of it. So "tracker bit set" implies "entry
///   appended", and a drain that starts later sees the entry.
/// - A guest write that precedes the submission being processed had its fault handler return
///   before the write, so its entry precedes the back's drain (same happens-before chain the
///   lock-based read relies on).
/// - The walk of an unverified run always takes the region lock (no lock-free "region is clean"
///   shortcut): a handler that has appended but not yet set its bits holds that lock, so the walk
///   waits for the bits. A handler that starts after the walk appends after the walk's drain, and
///   that entry clears the bit again at the next drain.
/// - Overflow (the back did not drain for more than half the ring, e.g. while idle on a loading
///   screen) drops every verified bit. Entries skipped that way belong to handlers that took
///   their region lock before the back's tail read, so later walks see their bits.
inline const int g_back_clean = [] {
    const char* env = std::getenv("SHADPS4_BACK_CLEAN");
    const int mode = env ? env[0] - '0' : 0;
    return mode == 1 || mode == 2 ? mode : 0;
}();

class DirtyLog {
public:
    static constexpr u64 Capacity = 1ULL << 16;
    static constexpr u64 Mask = Capacity - 1;
    static constexpr u64 TagMask = (1ULL << 24) - 1;

    /// Any thread, under the region lock of the pages, before their tracker bits are set.
    /// `count` is at most NUM_PAGES_PER_REGION (one region per entry).
    static void Append(u64 first_page, u64 count) noexcept {
        if (g_back_clean == 0) {
            return;
        }
        const u64 index = Tail().fetch_add(1, std::memory_order_seq_cst);
        slots[index & Mask].store(((index & TagMask) << 40) | (first_page << 12) | (count - 1),
                                  std::memory_order_release);
    }

    /// The single consumer (CP back). Calls f(first_page, count) for every entry appended before
    /// the call. Returns false if entries were lost: the caller must forget everything it knows.
    template <typename F>
    static bool Drain(u64& head, F&& f) {
        const u64 end = Tail().load(std::memory_order_seq_cst);
        if (end == head) [[likely]] {
            return true;
        }
        if (end - head > Capacity / 2) {
            head = end;
            return false;
        }
        while (head != end) {
            const u64 want = head & TagMask;
            u64 value = slots[head & Mask].load(std::memory_order_acquire);
            for (u32 spins = 0; (value >> 40) != want; ++spins) {
                // The appender is between its fetch_add and its store (it holds a region lock),
                // or the ring lapped us. Do not wait long: the appender may be suspended
                // (debugger pause, stack sampler, shutdown), and unlike the region lock this
                // wait would stall every read sync. Giving up is the overflow path: all
                // verified bits are dropped, and the skipped appender took its region lock
                // before this tail read, so the next locked walk of its pages waits for it.
                if (spins == 2048 || Tail().load(std::memory_order_relaxed) - head >= Capacity) {
                    head = Tail().load(std::memory_order_seq_cst);
                    return false;
                }
                _mm_pause();
                value = slots[head & Mask].load(std::memory_order_acquire);
            }
            f((value >> 12) & 0xfffffffULL, (value & 0x3ff) + 1);
            ++head;
        }
        return true;
    }

    static u64 Start() noexcept {
        return Tail().load(std::memory_order_seq_cst);
    }

private:
    static_assert(NUM_PAGES_PER_REGION <= 1024, "count - 1 must fit 10 bits");
    static_assert(40 - TRACKER_PAGE_BITS <= 28, "page number must fit 28 bits");
    struct alignas(64) PaddedCounter {
        std::atomic<u64> value;
    };
    static_assert(sizeof(PaddedCounter) == 64);
    // Starts one lap in, so no index of the first lap carries tag 0 (the slots' initial value).
    static inline PaddedCounter tail_storage{Capacity};
    static std::atomic<u64>& Tail() noexcept {
        return tail_storage.value;
    }
    alignas(64) static inline std::array<std::atomic<u64>, Capacity> slots{};
};

/// The back's private "verified clean" bits. One thread only.
class BackCleanCache {
public:
    using Run = std::pair<u64, u64>; ///< first page, page count

    struct Stats {
        u64 queries{}, clean{}, runs{}, run_pages{}, drained{}, resets{}, raced{}, mismatches{};
    };
    Stats stats;

    BackCleanCache() {
        if (g_back_clean != 0) {
            top = std::make_unique<Region*[]>(NumRegions);
            full = std::make_unique<u64[]>(NumRegions / 64);
            head = DirtyLog::Start();
        }
    }

    /// Applies every dirty marking made since the last call.
    void Sync() {
        const bool ok = DirtyLog::Drain(head, [this](u64 first_page, u64 count) {
            ++stats.drained;
            Clear(first_page, count);
        });
        if (!ok) [[unlikely]] {
            ++generation;
            ++stats.resets;
            std::fill_n(full.get(), NumRegions / 64, u64{0});
        }
    }

    /// Appends the page runs of [addr, addr + size) that are not verified clean, merged.
    void CollectUnverified(VAddr addr, u64 size, std::vector<Run>& runs) const {
        u64 page = addr >> TRACKER_PAGE_BITS;
        const u64 end = (addr + size + TRACKER_BYTES_PER_PAGE - 1) >> TRACKER_PAGE_BITS;
        while (page < end) {
            const u64 region = page / NUM_PAGES_PER_REGION;
            // Fully verified regions are skipped through the summary, 64 regions per word:
            // the average query spans ~11 regions (4 MiB each) and a per-region block lookup
            // cost as much as the tracker walk it replaced (profile 2026-10-01).
            const u32 skip = static_cast<u32>(std::countr_one(full[region / 64] >> (region % 64)));
            if (skip != 0) {
                page = (region + skip) * NUM_PAGES_PER_REGION;
                continue;
            }
            const u64 base = region * NUM_PAGES_PER_REGION;
            const u64 lo = page - base;
            const u64 hi = std::min<u64>(end - base, NUM_PAGES_PER_REGION);
            const Region* r = top[region];
            if (!r || r->generation != generation) {
                AddRun(runs, page, hi - lo);
            } else if (r->count != NUM_PAGES_PER_REGION) {
                for (u64 word = lo / 64; word <= (hi - 1) / 64; ++word) {
                    u64 missing = ~r->bits[word] & WordMask(word, lo, hi);
                    while (missing != 0) {
                        const u64 start = std::countr_zero(missing);
                        const u64 length = std::countr_one(missing >> start);
                        AddRun(runs, base + word * 64 + start, length);
                        missing &= length == 64 ? 0 : ~(((1ULL << length) - 1) << start);
                    }
                }
            }
            page = base + hi;
        }
    }

    /// The locked walk took these pages: each is clean in the tracker, or will be logged again.
    void MarkVerified(u64 first_page, u64 count) {
        u64 page = first_page;
        const u64 end = first_page + count;
        while (page < end) {
            const u64 region = page / NUM_PAGES_PER_REGION;
            const u64 base = region * NUM_PAGES_PER_REGION;
            const u64 lo = page - base;
            const u64 hi = std::min<u64>(end - base, NUM_PAGES_PER_REGION);
            Region*& r = top[region];
            if (!r) {
                r = &pool.emplace_back();
            }
            if (r->generation != generation) {
                r->generation = generation;
                r->count = 0;
                r->bits.fill(0);
            }
            for (u64 word = lo / 64; word <= (hi - 1) / 64; ++word) {
                const u64 mask = WordMask(word, lo, hi);
                r->count += static_cast<u32>(std::popcount(mask & ~r->bits[word]));
                r->bits[word] |= mask;
            }
            if (r->count == NUM_PAGES_PER_REGION) {
                full[region / 64] |= 1ULL << (region % 64);
            }
            page = base + hi;
        }
    }

    static constexpr u64 NumRegions = 1ULL << (40 - TRACKER_HIGHER_PAGE_BITS);

private:
    struct alignas(64) Region {
        u64 generation{};
        u32 count{}; ///< set bits
        std::array<u64, NUM_PAGES_PER_REGION / 64> bits{};
    };

    /// Bits of `word` that lie in the region-relative page range [lo, hi).
    static u64 WordMask(u64 word, u64 lo, u64 hi) {
        const u64 first = std::max<u64>(lo, word * 64) - word * 64;
        const u64 last = std::min<u64>(hi, word * 64 + 64) - word * 64; // exclusive, 1..64
        const u64 upto = last == 64 ? ~0ULL : (1ULL << last) - 1;
        return upto & ~((1ULL << first) - 1);
    }

    static void AddRun(std::vector<Run>& runs, u64 first_page, u64 count) {
        if (!runs.empty() && runs.back().first + runs.back().second == first_page) {
            runs.back().second += count;
        } else {
            runs.emplace_back(first_page, count);
        }
    }

    void Clear(u64 first_page, u64 count) {
        const u64 region = first_page / NUM_PAGES_PER_REGION;
        if (region >= NumRegions) {
            return;
        }
        Region* r = top[region];
        if (!r || r->generation != generation) {
            return;
        }
        full[region / 64] &= ~(1ULL << (region % 64));
        const u64 lo = first_page - region * NUM_PAGES_PER_REGION;
        const u64 hi = std::min<u64>(lo + count, NUM_PAGES_PER_REGION);
        for (u64 word = lo / 64; word <= (hi - 1) / 64; ++word) {
            const u64 mask = WordMask(word, lo, hi);
            r->count -= static_cast<u32>(std::popcount(mask & r->bits[word]));
            r->bits[word] &= ~mask;
        }
    }

    std::unique_ptr<Region*[]> top;
    /// One bit per region: every page of it is verified (current generation). Private.
    std::unique_ptr<u64[]> full;
    std::deque<Region> pool;
    u64 generation{1};
    u64 head{};
};

} // namespace VideoCore
