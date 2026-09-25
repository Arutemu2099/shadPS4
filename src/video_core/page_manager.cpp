// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <bit>
#include <chrono>
#include <cstdlib>
#include <boost/container/small_vector.hpp>
#include "common/assert.h"
#include "common/debug.h"
#include "common/div_ceil.h"
#include "common/fault_monitor.h"
#include "common/lean.h"
#include "common/range_lock.h"
#include "common/signal_context.h"
#include "core/memory.h"
#include "core/signals.h"
#include "video_core/page_manager.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"

#ifndef _WIN64
#include <sys/mman.h>
#include "common/adaptive_mutex.h"
#ifdef ENABLE_USERFAULTFD
#include <thread>
#include <fcntl.h>
#include <linux/userfaultfd.h>
#include <poll.h>
#include <sys/ioctl.h>
#include "common/error.h"
#endif
#else
#include <windows.h>
#include "common/spin_lock.h"
#endif

#ifdef __linux__
#include "common/adaptive_mutex.h"
#else
#include "common/spin_lock.h"
#endif

namespace VideoCore {

constexpr size_t PM_PAGE_SIZE = 4_KB;
constexpr size_t PM_PAGE_BITS = 12;

/// See PageManager::SetDeferredProtectSink. Only write-protect (track, !is_read) requests go
/// here; unprotects and read protection always reach the OS immediately.
static thread_local std::vector<PageManager::ProtectRange>* t_protect_sink = nullptr;

struct PageManager::Impl {
    struct PageState {
        u8 num_write_watchers : 7;
        // At the moment only buffer cache can request read watchers.
        // And buffers cannot overlap, thus only 1 can exist per page.
        u8 num_read_watchers : 1;

        Core::MemoryPermission WritePerm() const noexcept {
            return num_write_watchers == 0 ? Core::MemoryPermission::Write
                                           : Core::MemoryPermission::None;
        }

        Core::MemoryPermission ReadPerm() const noexcept {
            return num_read_watchers == 0 ? Core::MemoryPermission::Read
                                          : Core::MemoryPermission::None;
        }

        Core::MemoryPermission Perms() const noexcept {
            return ReadPerm() | WritePerm();
        }

        template <s32 delta, bool is_read>
        u8 AddDelta() {
            if constexpr (is_read) {
                if constexpr (delta == 1) {
                    return ++num_read_watchers;
                } else if (delta == -1) {
                    ASSERT_MSG(num_read_watchers > 0, "Not enough watchers");
                    return --num_read_watchers;
                } else {
                    return num_read_watchers;
                }
            } else {
                if constexpr (delta == 1) {
                    return ++num_write_watchers;
                } else if (delta == -1) {
                    ASSERT_MSG(num_write_watchers > 0, "Not enough watchers");
                    return --num_write_watchers;
                } else {
                    return num_write_watchers;
                }
            }
        }
    };

    static constexpr size_t ADDRESS_BITS = 40;
    static constexpr size_t NUM_ADDRESS_PAGES = 1ULL << (40 - PM_PAGE_BITS);
    static constexpr size_t NUM_ADDRESS_LOCKS = NUM_ADDRESS_PAGES / PAGES_PER_LOCK;
    inline static Vulkan::Rasterizer* rasterizer;
#ifdef ENABLE_USERFAULTFD
    Impl(Vulkan::Rasterizer* rasterizer_) {
        rasterizer = rasterizer_;
        uffd = syscall(__NR_userfaultfd, O_CLOEXEC | O_NONBLOCK | UFFD_USER_MODE_ONLY);
        ASSERT_MSG(uffd != -1, "{}", Common::GetLastErrorMsg());

        // Request uffdio features from kernel.
        uffdio_api api;
        api.api = UFFD_API;
        api.features = UFFD_FEATURE_THREAD_ID;
        const int ret = ioctl(uffd, UFFDIO_API, &api);
        ASSERT(ret == 0 && api.api == UFFD_API);

        // Create uffd handler thread
        ufd_thread = std::jthread([&](std::stop_token token) { UffdHandler(token); });
    }

    void OnMap(VAddr address, size_t size) {
        uffdio_register reg;
        reg.range.start = address;
        reg.range.len = size;
        reg.mode = UFFDIO_REGISTER_MODE_WP;
        const int ret = ioctl(uffd, UFFDIO_REGISTER, &reg);
        ASSERT_MSG(ret != -1, "Uffdio register failed");
    }

    void OnUnmap(VAddr address, size_t size) {
        uffdio_range range;
        range.start = address;
        range.len = size;
        const int ret = ioctl(uffd, UFFDIO_UNREGISTER, &range);
        ASSERT_MSG(ret != -1, "Uffdio unregister failed");
    }

    void Protect(VAddr address, size_t size, Core::MemoryPermission perms) {
        bool allow_write = True(perms & Core::MemoryPermission::Write);
        uffdio_writeprotect wp;
        wp.range.start = address;
        wp.range.len = size;
        wp.mode = allow_write ? 0 : UFFDIO_WRITEPROTECT_MODE_WP;
        const int ret = ioctl(uffd, UFFDIO_WRITEPROTECT, &wp);
        ASSERT_MSG(ret != -1, "Uffdio writeprotect failed with error: {}",
                   Common::GetLastErrorMsg());
    }

    void UffdHandler(std::stop_token token) {
        while (!token.stop_requested()) {
            pollfd pollfd;
            pollfd.fd = uffd;
            pollfd.events = POLLIN;

            // Block until the descriptor is ready for data reads.
            const int pollres = poll(&pollfd, 1, -1);
            switch (pollres) {
            case -1:
                perror("Poll userfaultfd");
                continue;
                break;
            case 0:
                continue;
            case 1:
                break;
            default:
                UNREACHABLE_MSG("Unexpected number of descriptors {} out of poll", pollres);
            }

            // We don't want an error condition to have occured.
            ASSERT_MSG(!(pollfd.revents & POLLERR), "POLLERR on userfaultfd");

            // We waited until there is data to read, we don't care about anything else.
            if (!(pollfd.revents & POLLIN)) {
                continue;
            }

            // Read message from kernel.
            uffd_msg msg;
            const int readret = read(uffd, &msg, sizeof(msg));
            ASSERT_MSG(readret != -1 || errno == EAGAIN, "Unexpected result of uffd read");
            if (errno == EAGAIN) {
                continue;
            }
            ASSERT_MSG(readret == sizeof(msg), "Unexpected short read, exiting");
            ASSERT(msg.arg.pagefault.flags & UFFD_PAGEFAULT_FLAG_WP);

            // Notify rasterizer about the fault.
            const VAddr addr = msg.arg.pagefault.address;
            rasterizer->InvalidateMemory(addr, 1);
        }
    }

    std::jthread ufd_thread;
    int uffd;
#else
    Impl(Vulkan::Rasterizer* rasterizer_) {
        rasterizer = rasterizer_;

        // Should be called first.
        constexpr auto priority = std::numeric_limits<u32>::min();
        Core::Signals::Instance()->RegisterAccessViolationHandler(GuestFaultSignalHandler,
                                                                  priority);
    }

    void OnMap(VAddr address, size_t size) {
        // No-op
    }

    void OnUnmap(VAddr address, size_t size) {
        // No-op
    }

    void Protect(VAddr address, size_t size, Core::MemoryPermission perms) {
        RENDERER_TRACE;
        auto* memory = Core::Memory::Instance();
        auto& impl = memory->GetAddressSpace();
        ASSERT_MSG(perms != Core::MemoryPermission::Write,
                   "Attempted to protect region as write-only which is not a valid permission");
        impl.Protect(address, size, perms);
    }

    static bool GuestFaultSignalHandler(void* context, void* fault_address) {
        const auto addr = reinterpret_cast<VAddr>(fault_address);
        const auto start = Common::FaultMonitor::Now();
        if (Common::IsWriteError(context)) {
            // A fault taken by the CP itself is part of the command stream it is executing, so
            // only guest threads' faults may be deferred (submit_epoch.h).
            const bool handled = rasterizer->InvalidateMemory(
                addr, 8,
                Common::FaultMonitor::t_is_cp_thread ? VideoCore::InvalidateOrigin::CpWriteFault
                                                     : VideoCore::InvalidateOrigin::GuestWriteFault);
            Common::FaultMonitor::CountGuestWriteFault(handled,
                                                       Common::FaultMonitor::SinceNs(start));
            if (handled && Common::FaultMonitor::WriteFaultPagesEnabled()) {
                Common::FaultMonitor::RecordWriteFaultPage(addr);
            }
            return handled;
        } else {
            // Read faults only exist under Precise readbacks - it is the only mode that
            // read-protects guest pages holding GPU-modified data.
            const bool handled = rasterizer->ReadMemory(addr, 8);
            Common::FaultMonitor::CountGuestReadFault(handled, Common::FaultMonitor::SinceNs(start),
                                                      addr);
            return handled;
        }
    }
#endif

    template <bool track, bool is_read>
    void UpdatePageWatchers(VAddr addr, u64 size) {
        RENDERER_TRACE;

        size_t page = addr >> PM_PAGE_BITS;
        const u64 page_end = Common::DivCeil(addr + size, PM_PAGE_SIZE);

        // Acquire locks for the range of pages
        auto lk = LockPages(page, page_end);

        auto perms = cached_pages[page].Perms();
        u64 range_begin = 0;
        u64 range_bytes = 0;
        u64 potential_range_bytes = 0;

        const auto release_pending = [&] {
            if (range_bytes > 0) {
                RENDERER_TRACE;
                if constexpr (track && !is_read) {
                    if (t_protect_sink) {
                        t_protect_sink->push_back({range_begin << PM_PAGE_BITS, range_bytes});
                        range_bytes = 0;
                        potential_range_bytes = 0;
                        return;
                    }
                }
                Protect(range_begin << PM_PAGE_BITS, range_bytes, perms);
                range_bytes = 0;
                potential_range_bytes = 0;
            }
        };

        // Iterate requested pages
        const u64 aligned_addr = page << PM_PAGE_BITS;
        const u64 aligned_end = page_end << PM_PAGE_BITS;
        // SHADPS4_WATCH_MAPCHECK=0: skip this log-only check (a locked hash lookup per call; the
        // warning was never seen in the forest logs). SHADPS4_LEAN=1 implies it.
        static const bool map_check = [] {
            const char* value = std::getenv("SHADPS4_WATCH_MAPCHECK");
            return !::Common::Lean() && (!value || value[0] != '0');
        }();
        if (map_check && !rasterizer->IsMapped(aligned_addr, aligned_end - aligned_addr)) {
            LOG_WARNING(Render,
                        "Tracking memory region {:#x} - {:#x} which is not fully GPU mapped.",
                        aligned_addr, aligned_end);
        }

        for (; page != page_end; ++page) {
            PageState& state = cached_pages[page];

            // Apply the change to the page state
            const u8 new_count = state.AddDelta<track ? 1 : -1, is_read>();

            if (auto new_perms = state.Perms(); new_perms != perms) [[unlikely]] {
                // If the protection changed add pending (un)protect action
                release_pending();
                perms = new_perms;
            } else if (range_bytes != 0) {
                // If the protection did not change, extend the potential range
                potential_range_bytes += PM_PAGE_SIZE;
            }

            // Only start a new range if the page must be (un)protected
            if ((new_count == 0 && !track) || (new_count == 1 && track)) {
                if (range_bytes == 0) {
                    // Start a new potential range
                    range_begin = page;
                    potential_range_bytes = PM_PAGE_SIZE;
                }
                // Extend current range up to potential range
                range_bytes = potential_range_bytes;
            }
        }

        // Add pending (un)protect action
        release_pending();
    }

    template <bool track, bool is_read>
    void UpdatePageWatchersForRegion(VAddr base_addr, RegionBits& mask) {
        RENDERER_TRACE;
        auto start_range = mask.FirstRange();
        auto end_range = mask.LastRange();

        if (start_range.second == end_range.second) {
            // if all pages are contiguous, use the regular UpdatePageWatchers
            const VAddr start_addr = base_addr + (start_range.first << PM_PAGE_BITS);
            const u64 size = (start_range.second - start_range.first) << PM_PAGE_BITS;
            return UpdatePageWatchers<track, is_read>(start_addr, size);
        }

        size_t base_page = (base_addr >> PM_PAGE_BITS);
        ASSERT(base_page % PAGES_PER_LOCK == 0);
        // The stripes the mask's page span touches; without SHADPS4_PAGE_LOCK_PAGES that is the
        // region's one lock, as before.
        auto lk = LockPages(base_page + start_range.first, base_page + end_range.second);
        auto perms = cached_pages[base_page + start_range.first].Perms();
        u64 range_begin = 0;
        u64 range_bytes = 0;
        u64 potential_range_bytes = 0;

        const auto release_pending = [&] {
            if (range_bytes > 0) {
                RENDERER_TRACE;
                if constexpr (track && !is_read) {
                    if (t_protect_sink) {
                        t_protect_sink->push_back({range_begin << PM_PAGE_BITS, range_bytes});
                        range_bytes = 0;
                        potential_range_bytes = 0;
                        return;
                    }
                }
                Protect((range_begin << PM_PAGE_BITS), range_bytes, perms);
                range_bytes = 0;
                potential_range_bytes = 0;
            }
        };

        // Iterate pages
        for (size_t page = start_range.first; page < end_range.second; ++page) {
            PageState& state = cached_pages[base_page + page];
            const bool update = mask.Get(page);

            // Apply the change to the page state
            const u8 new_count =
                update ? state.AddDelta<track ? 1 : -1, is_read>() : state.AddDelta<0, is_read>();

            if (auto new_perms = state.Perms(); new_perms != perms) [[unlikely]] {
                // If the protection changed add pending (un)protect action
                release_pending();
                perms = new_perms;
            } else if (range_bytes != 0) {
                // If the protection did not change, extend the potential range
                potential_range_bytes += PM_PAGE_SIZE;
            }

            // If the page is not being updated, skip it
            if (!update) {
                continue;
            }

            // If the page must be (un)protected
            if ((new_count == 0 && !track) || (new_count == 1 && track)) {
                if (range_bytes == 0) {
                    // Start a new potential range
                    range_begin = base_page + page;
                    potential_range_bytes = PM_PAGE_SIZE;
                }
                // Extend current rango up to potential range
                range_bytes = potential_range_bytes;
            }
        }

        // Add pending (un)protect action
        release_pending();
    }

    /// Applies the current watcher state of every page in the range to the OS protection.
    void ReconcileProtection(VAddr addr, u64 size) {
        RENDERER_TRACE;
        const size_t first = addr >> PM_PAGE_BITS;
        const size_t end = Common::DivCeil(addr + size, PM_PAGE_SIZE);
        if (end <= first) {
            return;
        }
        auto lk = LockPages(first, end);
        const auto perms_of = [&](size_t page) {
            const auto perms = cached_pages[page].Perms();
            // Write-only is not expressible on the CPU; the stricter state is always safe.
            return perms == Core::MemoryPermission::Write ? Core::MemoryPermission::None : perms;
        };
        size_t run = first;
        auto perms = perms_of(first);
        for (size_t page = first + 1; page < end; ++page) {
            const auto next = perms_of(page);
            if (next != perms) {
                Protect(run << PM_PAGE_BITS, (page - run) << PM_PAGE_BITS, perms);
                run = page;
                perms = next;
            }
        }
        Protect(run << PM_PAGE_BITS, (end - run) << PM_PAGE_BITS, perms);
    }

    std::array<PageState, NUM_ADDRESS_PAGES> cached_pages{};
#ifdef PTHREAD_ADAPTIVE_MUTEX_INITIALIZER_NP
    using LockType = Common::AdaptiveMutex;
#else
    using LockType = Common::SpinLock;
#endif
    std::array<LockType, NUM_ADDRESS_LOCKS> locks{};

    /**
     * SHADPS4_PAGE_LOCK_PAGES=<n> (power of two, 16..NUM_PAGES_PER_REGION): lock stripes of n pages
     * instead of `locks`' one per 4 MB region. Measured 2026-09-25 (a19): the CP spun 0.86% of its
     * time in LockRange while re-protecting CPU-dirty pages, behind fault handlers, the Protector
     * and ReconcileProtection holding the same region. Every page's counts and protection still
     * change only under the stripe that covers it. Costs NUM_ADDRESS_PAGES / n extra locks
     * (n=64: 2^22 x 4 B = 16 MB on Windows), allocated only when the flag is set.
     *
     * Fine stripes are taken in ascending order, iteratively: every holder takes its whole span in
     * one LockPages call and takes no other stripe while holding it, so the order alone rules out
     * deadlock. (LockRange/TryLockRange recurse once per lock, and an image span can cover
     * hundreds of fine stripes - on a guest thread's stack when a fault handler untracks it.)
     *
     * SHADPS4_PAGE_LOCK_STATS=1: on the CP thread, try the whole span first and count the
     * acquisitions where that failed (the ones that would spin), with the time spent getting them.
     */
    static u32 FineLockBits() {
        const char* value = std::getenv("SHADPS4_PAGE_LOCK_PAGES");
        if (!value) {
            return 0;
        }
        const u64 pages = std::strtoull(value, nullptr, 10);
        if (pages < 16 || pages > PAGES_PER_LOCK || !std::has_single_bit(pages)) {
            LOG_WARNING(Render, "SHADPS4_PAGE_LOCK_PAGES={} ignored (power of two, 16-{})", value,
                        PAGES_PER_LOCK);
            return 0;
        }
        if (pages == PAGES_PER_LOCK) {
            return 0;
        }
        LOG_INFO(Render, "PageLock: {} pages per lock stripe ({} locks)", pages,
                 NUM_ADDRESS_PAGES / pages);
        return static_cast<u32>(std::countr_zero(pages));
    }

    static bool LockStatsEnabled() {
        const char* value = std::getenv("SHADPS4_PAGE_LOCK_STATS");
        return value && value[0] == '1';
    }

    using LockGuard = Common::RangeLockGuard<LockType*>;

    /// Locks every stripe covering pages [first, end).
    LockGuard LockPages(size_t first, size_t end) {
        if (lock_slow_path) [[unlikely]] {
            return LockPagesSlow(first, end);
        }
        return LockGuard(locks.data() + first / PAGES_PER_LOCK,
                         locks.data() + Common::DivCeil(end, PAGES_PER_LOCK));
    }

    static bool TryLockOrdered(LockType* begin, LockType* end) {
        for (LockType* it = begin; it != end; ++it) {
            if (!it->try_lock()) {
                while (it != begin) {
                    (--it)->unlock();
                }
                return false;
            }
        }
        return true;
    }

    static void LockOrdered(LockType* begin, LockType* end) {
        for (; begin != end; ++begin) {
            begin->lock();
        }
    }

    LockGuard LockPagesSlow(size_t first, size_t end) {
        const bool fine = fine_locks != nullptr;
        LockType* begin = locks.data() + first / PAGES_PER_LOCK;
        LockType* last = locks.data() + Common::DivCeil(end, PAGES_PER_LOCK);
        if (fine) {
            begin = fine_locks.get() + (first >> fine_lock_bits);
            last = fine_locks.get() + ((end + (1ULL << fine_lock_bits) - 1) >> fine_lock_bits);
        }
        const bool count = lock_stats && Common::FaultMonitor::t_is_cp_thread;
        if (count) {
            // Both tries either take the whole span or release what they took.
            if (fine ? TryLockOrdered(begin, last) : Common::TryLockRange(begin, last) == last) {
                CountCpLock(false, 0);
                return LockGuard(begin, last, std::adopt_lock);
            }
        }
        const auto start =
            count ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        if (fine) {
            LockOrdered(begin, last);
        } else {
            Common::LockRange(begin, last);
        }
        if (count) {
            CountCpLock(true, static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                   std::chrono::steady_clock::now() - start)
                                                   .count()));
        }
        return LockGuard(begin, last, std::adopt_lock);
    }

    /// CP thread only (t_is_cp_thread), so plain counters. Logged every 512 presented frames.
    void CountCpLock(bool spun, u64 spin_ns) {
        ++cp_locks;
        cp_spins += spun;
        cp_spin_ns += spin_ns;
        if ((cp_locks & 0xfff) != 0) {
            return;
        }
        const u64 frames = Common::FaultMonitor::presented_frames.load(std::memory_order_relaxed);
        if (cp_stats_frames == 0 || frames < cp_stats_frames + 512) {
            if (cp_stats_frames == 0 && frames != 0) {
                // Start the first window at the first presented frame, not at boot.
                cp_stats_frames = frames;
                cp_locks = cp_spins = cp_spin_ns = 0;
            }
            return;
        }
        const double n = static_cast<double>(frames - cp_stats_frames);
        LOG_INFO(Render,
                 "PageLockSpin: stripe_pages={} frames={} cp_locks/frame={:.1f} "
                 "cp_spins/frame={:.2f} ({:.2f}% of locks) cp_spin_us/frame={:.1f}",
                 fine_locks ? (1ULL << fine_lock_bits) : PAGES_PER_LOCK, frames - cp_stats_frames,
                 cp_locks / n, cp_spins / n, 100.0 * cp_spins / cp_locks, cp_spin_ns / n / 1000.0);
        cp_stats_frames = frames;
        cp_locks = cp_spins = cp_spin_ns = 0;
    }

    const u32 fine_lock_bits = FineLockBits();
    const std::unique_ptr<LockType[]> fine_locks =
        fine_lock_bits != 0 ? std::make_unique<LockType[]>(NUM_ADDRESS_PAGES >> fine_lock_bits)
                            : nullptr;
    const bool lock_stats = LockStatsEnabled();
    // The one flag the default path reads: either option routes locking through LockPagesSlow.
    const bool lock_slow_path = fine_lock_bits != 0 || lock_stats;
    u64 cp_locks = 0;
    u64 cp_spins = 0;
    u64 cp_spin_ns = 0;
    u64 cp_stats_frames = 0;
};

PageManager::PageManager(Vulkan::Rasterizer* rasterizer_)
    : impl{std::make_unique<Impl>(rasterizer_)} {}

PageManager::~PageManager() = default;

void PageManager::SetDeferredProtectSink(std::vector<ProtectRange>* sink) {
    t_protect_sink = sink;
}

void PageManager::ReconcileProtection(std::span<const ProtectRange> ranges) const {
    for (const ProtectRange& range : ranges) {
        impl->ReconcileProtection(range.addr, range.size);
    }
}

void PageManager::OnGpuMap(VAddr address, size_t size) {
    impl->OnMap(address, size);
}

void PageManager::OnGpuUnmap(VAddr address, size_t size) {
    impl->OnUnmap(address, size);
}

template <bool track>
void PageManager::UpdatePageWatchers(VAddr addr, u64 size) const {
    impl->UpdatePageWatchers<track, false>(addr, size);
}

template <bool track, bool is_read>
void PageManager::UpdatePageWatchersForRegion(VAddr base_addr, RegionBits& mask) const {
    impl->UpdatePageWatchersForRegion<track, is_read>(base_addr, mask);
}

template void PageManager::UpdatePageWatchers<true>(VAddr addr, u64 size) const;
template void PageManager::UpdatePageWatchers<false>(VAddr addr, u64 size) const;
template void PageManager::UpdatePageWatchersForRegion<true, true>(VAddr base_addr,
                                                                   RegionBits& mask) const;
template void PageManager::UpdatePageWatchersForRegion<true, false>(VAddr base_addr,
                                                                    RegionBits& mask) const;
template void PageManager::UpdatePageWatchersForRegion<false, true>(VAddr base_addr,
                                                                    RegionBits& mask) const;
template void PageManager::UpdatePageWatchersForRegion<false, false>(VAddr base_addr,
                                                                     RegionBits& mask) const;

} // namespace VideoCore
