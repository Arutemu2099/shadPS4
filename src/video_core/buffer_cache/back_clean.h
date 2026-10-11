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

#include "common/fault_monitor.h"
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

/// SHADPS4_RMEMO_CENSUS=1 (diagnostic, default off; notes/buffer-sync-validity-2026-10-07.md):
/// classifies read-binding memo misses by page-granular validity (the BackClean answer) and by a
/// per-Buffer dirty counter (a second DirtyLog consumer on the back), and times the miss path.
/// Needs SHADPS4_BACK_CLEAN=1 (DirtyLog appends nothing without it). Excludes SHADPS4_PB_CENSUS
/// (both use ReadMemoEntry::pb_written).
inline const bool g_rmemo_census = [] {
    const char* env = std::getenv("SHADPS4_RMEMO_CENSUS");
    return env && env[0] == '1';
}();

/// SHADPS4_READ_SYNC_FAST=1 (=2 verify), default off (notes/buffer-sync-validity-2026-10-07.md
/// sec. 6): a read sync that BackClean answers "no unverified page" returns from ObtainBuffer
/// before SynchronizeBuffer. Needs SHADPS4_BACK_CLEAN=1.
inline const int g_read_sync_fast = [] {
    const char* env = std::getenv("SHADPS4_READ_SYNC_FAST");
    const int mode = env ? env[0] - '0' : 0;
    return mode == 1 || mode == 2 ? mode : 0;
}();

/// SHADPS4_NATIVE_COHERENT=1 (=2 verify), default off (notes/native-path-2026-10-09.md sec. 17,
/// milestone 1 of the native renderer): CPU-written guest memory is made coherent with the cached
/// GPU buffers once per guest submission instead of per binding. At every submission start (and
/// at every other point the CP's legal view of guest memory moves: a satisfied memory wait, a
/// CP-side guest write) the back drains this log and uploads every dirty page that lies in a
/// cached buffer (BufferCache::CoherenceSync); new buffers are synced whole when created, the
/// CP's own DMA writes when they happen. A read binding then needs no validity check at all: it
/// is the buffer lookup and the access barrier. A page the guest writes after its submission
/// started is not legally visible to that submission (submit_epoch.h) and is uploaded at the next
/// point. Needs SHADPS4_BACK_CLEAN=1 (the log) and the CP split (the points arrive as records).
/// SHADPS4_ASC_EARLY (l4a3): set while the back runs a record ahead of the graphics backlog -
/// its read bindings take the per-binding validity sync instead of the coherence skip.
inline thread_local bool t_force_binding_sync = false;
inline thread_local u64 t_forced_binding_checks = 0;
/// SHADPS4_ASC_EARLY_INV=1 (l4a4 census, back thread only): ring position of the record the back runs
/// (in order: t_inorder_pos; early: t_early_pos) and the ranges early records wrote.
inline const bool g_early_inv = [] {
    const char* env = std::getenv("SHADPS4_ASC_EARLY_INV");
    return env && env[0] == '1';
}();
inline thread_local u64 t_inorder_pos = 0;
inline thread_local u64 t_early_pos = 0;
struct EarlyWrite {
    u64 lo, hi, pos;
};
inline std::array<EarlyWrite, 512> g_early_writes{};
inline u64 g_early_w_head = 0, g_early_w_tail = 0; // back only
inline std::atomic<u64> g_early_inv_reads{0}, g_early_inv_checks{0}, g_early_inv_writes{0};
/// SHADPS4_LANE_CENSUS=1 (n1, sec. 21 step 0, back thread only): see n1_lane_census.py.
inline const bool g_lane_census = [] {
    const char* env = std::getenv("SHADPS4_LANE_CENSUS");
    return env && env[0] == '1';
}();
struct LaneNote {
    u64 lo, hi, tick, seq;
    u32 kind; // 0 main write, 1 main upload, 2 early write, 3 early upload
};
struct LaneTable {
    std::array<std::array<LaneNote, 8>, 4096> b{};
    std::array<u8, 4096> pos{};
};
inline LaneTable g_lc_main{}, g_lc_early{};
inline u64 g_lc_seq = 0;
inline u64 g_lc_done = 0;     // main ticks complete (known semaphore value)
inline u32 t_lc_rec_flags = 0; // bit 0: open-batch hit, bit 1: in-flight hit (current early record)
/// 0 early reads checked, 1/2 newest producer main write open / in flight, 3/4 main upload open / in
/// flight, 5 early writes checked, 6/7 over main write-or-upload open / in flight, 8 in-order reads
/// checked, 9/10/11 in-order reads over unfinished early writes via coherent skip / full / lean,
/// 12 notes truncated to 64 windows, 13 notes off the back thread, 14 notes, 15 early reads whose
/// newest producer is an early record (lane-internal), 16 early records with an open-batch hit,
/// 17 early records with an in-flight hit only, 18 early records.
inline std::array<std::atomic<u64>, 20> g_lc{};
inline u32 LaneBucket(u64 w) noexcept {
    return u32((w * 0x9E3779B97F4A7C15ull) >> 52) & 4095;
}
inline void LaneNoteRange(u64 lo, u64 hi, u64 tick, u32 kind) noexcept {
    if (hi <= lo) {
        return;
    }
    LaneTable& t = kind >= 2 ? g_lc_early : g_lc_main;
    const u64 seq = ++g_lc_seq;
    const u64 w0 = lo >> 16;
    u64 w1 = (hi - 1) >> 16;
    if (w1 - w0 > 63) {
        w1 = w0 + 63;
        g_lc[12].fetch_add(1, std::memory_order_relaxed);
    }
    g_lc[14].fetch_add(1, std::memory_order_relaxed);
    for (u64 w = w0; w <= w1; ++w) {
        const u32 k = LaneBucket(w);
        auto& last = t.b[k][(t.pos[k] - 1) & 7];
        if (last.seq != 0 && last.lo == lo && last.hi == hi && last.kind == kind) {
            last.tick = tick;
            last.seq = seq;
            continue;
        }
        t.b[k][t.pos[k]++ & 7] = {lo, hi, tick, seq, kind};
    }
}
/// The newest note overlapping [lo, hi) whose main tick is not complete.
inline const LaneNote* LaneFind(const LaneTable& t, u64 lo, u64 hi) noexcept {
    const LaneNote* best = nullptr;
    if (hi <= lo) {
        return best;
    }
    const u64 w0 = lo >> 16;
    const u64 w1 = std::min((hi - 1) >> 16, w0 + 63);
    for (u64 w = w0; w <= w1; ++w) {
        for (const LaneNote& e : t.b[LaneBucket(w)]) {
            if (e.seq != 0 && e.tick > g_lc_done && e.lo < hi && lo < e.hi &&
                (!best || e.seq > best->seq)) {
                best = &e;
            }
        }
    }
    return best;
}
inline void EarlyInvPrune(u64 read_pos) noexcept {
    while (g_early_w_head != g_early_w_tail &&
           g_early_writes[g_early_w_head % g_early_writes.size()].pos < read_pos) {
        ++g_early_w_head;
    }
}

inline const int g_native_coherent = [] {
    const char* env = std::getenv("SHADPS4_NATIVE_COHERENT");
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
        // Bit 10: appended by the CP thread (its own guest writes; NATIVE_COHERENT=2 classifies).
        const u64 cp = Common::FaultMonitor::t_is_cp_thread ? (1ULL << 10) : 0;
        slots[index & Mask].store(((index & TagMask) << 40) | (first_page << 12) | cp | (count - 1),
                                  std::memory_order_release);
    }

    /// As Drain, but stops at `end` (clamped to the tail) and passes f(first, count, by_cp).
    /// A second consumer with its own head (NATIVE_COHERENT). Returns false if entries were lost
    /// (head is then the tail).
    template <typename F>
    static bool DrainTo(u64& head, u64 end, F&& f) {
        const u64 tail = Tail().load(std::memory_order_seq_cst);
        if (end > tail) {
            end = tail;
        }
        if (end <= head) [[likely]] {
            return true;
        }
        if (tail - head > Capacity / 2) {
            head = tail;
            return false;
        }
        while (head != end) {
            const u64 want = head & TagMask;
            u64 value = slots[head & Mask].load(std::memory_order_acquire);
            for (u32 spins = 0; (value >> 40) != want; ++spins) {
                if (spins == 2048 || Tail().load(std::memory_order_relaxed) - head >= Capacity) {
                    head = Tail().load(std::memory_order_seq_cst);
                    return false;
                }
                _mm_pause();
                value = slots[head & Mask].load(std::memory_order_acquire);
            }
            f((value >> 12) & 0xfffffffULL, (value & 0x3ff) + 1, (value & (1ULL << 10)) != 0);
            ++head;
        }
        return true;
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

    /// The dirty-log index the last Sync() drained up to (entries before it are applied).
    u64 DrainedTo() const noexcept {
        return head;
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

    /// True when CollectUnverified(addr, size) would append nothing: every page of the range is
    /// verified clean. Same walk, stops at the first unverified page and builds no run list.
    bool AllVerified(VAddr addr, u64 size) const {
        u64 page = addr >> TRACKER_PAGE_BITS;
        const u64 end = (addr + size + TRACKER_BYTES_PER_PAGE - 1) >> TRACKER_PAGE_BITS;
        while (page < end) {
            const u64 region = page / NUM_PAGES_PER_REGION;
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
                return false;
            }
            if (r->count != NUM_PAGES_PER_REGION) {
                for (u64 word = lo / 64; word <= (hi - 1) / 64; ++word) {
                    if ((~r->bits[word] & WordMask(word, lo, hi)) != 0) {
                        return false;
                    }
                }
            }
            page = base + hi;
        }
        return true;
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
