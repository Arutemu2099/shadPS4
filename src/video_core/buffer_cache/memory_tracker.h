// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <type_traits>
#include <vector>

#include "common/debug.h"
#include "common/fault_monitor.h"
#include "common/types.h"
#include "core/emulator_settings.h"
#include "video_core/buffer_cache/region_manager.h"

namespace VideoCore {

class MemoryTracker {
public:
    static constexpr size_t MAX_CPU_PAGE_BITS = 40;
    static constexpr size_t NUM_HIGH_PAGES = 1ULL << (MAX_CPU_PAGE_BITS - TRACKER_HIGHER_PAGE_BITS);
    static constexpr size_t MANAGER_POOL_SIZE = 32;

public:
    explicit MemoryTracker(PageManager& tracker_) : tracker{&tracker_} {}
    ~MemoryTracker() = default;

    /// Returns true if a region has been modified from the CPU
    bool IsRegionCpuModified(VAddr query_cpu_addr, u64 query_size) noexcept {
        return IteratePages<true>(
            query_cpu_addr, query_size, [](RegionManager* manager, u64 offset, size_t size) {
                std::scoped_lock lk{manager->lock};
                return manager->template IsRegionModified<Type::CPU>(offset, size);
            });
    }

    /// Returns true if a region has been modified from the GPU
    bool IsRegionGpuModified(VAddr query_cpu_addr, u64 query_size) noexcept {
        return IteratePages<false>(
            query_cpu_addr, query_size, [](RegionManager* manager, u64 offset, size_t size) {
                std::scoped_lock lk{manager->lock};
                return manager->template IsRegionModified<Type::GPU>(offset, size);
            });
    }

    /// Mark region as CPU modified, notifying the device_tracker about this change
    void MarkRegionAsCpuModified(VAddr dirty_cpu_addr, u64 query_size) {
        IteratePages<false>(dirty_cpu_addr, query_size,
                            [](RegionManager* manager, u64 offset, size_t size) {
                                std::scoped_lock lk{manager->lock};
                                manager->template ChangeRegionState<Type::CPU, true>(
                                    manager->GetCpuAddr() + offset, size);
                            });
    }

    /// Unmark region as modified from the host GPU
    void UnmarkRegionAsGpuModified(VAddr dirty_cpu_addr, u64 query_size) noexcept {
        IteratePages<false>(dirty_cpu_addr, query_size,
                            [](RegionManager* manager, u64 offset, size_t size) {
                                std::scoped_lock lk{manager->lock};
                                manager->template ChangeRegionState<Type::GPU, false>(
                                    manager->GetCpuAddr() + offset, size);
                            });
    }

    /// Removes all protection from a page and ensures GPU data has been flushed if requested
    void InvalidateRegion(VAddr cpu_addr, u64 size, auto&& on_flush) noexcept {
        IteratePages<false>(
            cpu_addr, size, [&on_flush](RegionManager* manager, u64 offset, size_t size) {
                const bool should_flush = [&] {
                    // Perform both the GPU modification check and CPU state change with the lock
                    // in case we are racing with GPU thread trying to mark the page as GPU
                    // modified. If we need to flush the flush function is going to perform CPU
                    // state change.
                    std::scoped_lock lk{manager->lock};
                    if (EmulatorSettings.GetReadbacksMode() != GpuReadbacksMode::Disabled &&
                        manager->template IsRegionModified<Type::GPU>(offset, size)) {
                        return true;
                    }
                    manager->template ChangeRegionState<Type::CPU, true>(
                        manager->GetCpuAddr() + offset, size);
                    return false;
                }();
                if (should_flush) {
                    on_flush();
                }
            });
    }

    /// Call 'func' for each CPU modified range and unmark those pages as CPU modified
    /// SHADPS4_UPLOAD_CLEAN_SKIP=0 disables the pre-lock skip, so the fix can be A/B'd against
    /// itself in one binary. Default on.
    static bool CleanSkipEnabled() {
        static const bool enabled = [] {
            const char* env = std::getenv("SHADPS4_UPLOAD_CLEAN_SKIP");
            return env == nullptr || std::atoi(env) != 0;
        }();
        return enabled;
    }

    void ForEachUploadRange(VAddr query_cpu_range, u64 query_size, bool is_written, auto&& func,
                            auto&& on_upload) {
        // Three phases with very different costs - scan is bitset work, mark can reach a
        // VirtualProtect syscall. Timed separately so a fix is aimed at the one that is expensive.
        const auto t_scan = Common::FaultMonitor::Now();
        // Shape counters: plain locals in the hot loop, one atomic add at the end. See
        // CountUploadShape in fault_monitor.h for why this is counted and not timed.
        u64 shape_managers = 0;
        u64 shape_ranges = 0;
        u64 shape_skipped = 0;
        const auto counting_func = [&func, &shape_ranges](u64 addr, u64 size) {
            ++shape_ranges;
            func(addr, size);
        };
        // Measured 2026-08-19: 99.63% of these calls find zero modified ranges while visiting
        // 11.24 region managers each - ~555,000 lock-and-scan visits per frame to find ~400
        // ranges, 9.5 ms of a 52 ms frame. The AnyInRange early-out from ed9979cb sits INSIDE
        // ForEachModifiedRange, after the lock is already taken, so it cannot remove the lock.
        // The summary flag can, and is read without it - see RegionManager::MayHaveCpuModified.
        //
        // Only on the !is_written path: when is_written the lock is taken here and released in
        // the second IteratePages below, so skipping would unbalance it. That costs almost
        // nothing - is_written is 4.5% of calls.
        const bool clean_skip = CleanSkipEnabled() && !is_written;
        // Coarse fast path: a 42 MB query that is clean everywhere costs one or two word reads
        // instead of 11 pointer chases. Measured 2026-08-19: queries average 41 MB and 99.6% of
        // them find nothing, so this is the common case by a wide margin. !is_written only, for
        // the same lock-pairing reason as below.
        if (clean_skip && AllRegionsSkippable(query_cpu_range, query_size)) {
            on_upload();
            Common::FaultMonitor::CountUploadShape(0, 0, 0, query_size);
            Common::FaultMonitor::CountUploadFastPath();
            return;
        }
        IteratePages<true>(query_cpu_range, query_size,
                           [&counting_func, &shape_managers, &shape_skipped, clean_skip,
                            is_written](RegionManager* manager, u64 offset, size_t size) {
                               ++shape_managers;
                               if (clean_skip && !manager->MayHaveCpuModified()) {
                                   ++shape_skipped;
                                   return;
                               }
                               manager->lock.lock();
                               manager->template ForEachModifiedRange<Type::CPU, true>(
                                   manager->GetCpuAddr() + offset, size, counting_func);
                               if (!is_written) {
                                   manager->lock.unlock();
                               }
                           });
        Common::FaultMonitor::CountUploadShape(shape_managers, shape_ranges, shape_skipped,
                                               query_size);
        const auto t_copy = Common::FaultMonitor::Now();
        on_upload();
        const auto t_mark = Common::FaultMonitor::Now();
        if (!is_written) {
            Common::FaultMonitor::CountUploadPhases(Common::FaultMonitor::DiffNs(t_scan, t_copy),
                                                    Common::FaultMonitor::DiffNs(t_copy, t_mark), 0,
                                                    false);
            return;
        }
        IteratePages<false>(query_cpu_range, query_size,
                            [&func, is_written](RegionManager* manager, u64 offset, size_t size) {
                                manager->template ChangeRegionState<Type::GPU, true>(
                                    manager->GetCpuAddr() + offset, size);
                                manager->lock.unlock();
                            });
        Common::FaultMonitor::CountUploadPhases(Common::FaultMonitor::DiffNs(t_scan, t_copy),
                                                Common::FaultMonitor::DiffNs(t_copy, t_mark),
                                                Common::FaultMonitor::SinceNs(t_mark), true);
    }

    /// Call 'func' for each GPU modified range and unmark those pages as GPU modified
    template <bool clear>
    void ForEachDownloadRange(VAddr query_cpu_range, u64 query_size, auto&& func) {
        IteratePages<false>(query_cpu_range, query_size,
                            [&func](RegionManager* manager, u64 offset, size_t size) {
                                std::scoped_lock lk{manager->lock};
                                manager->template ForEachModifiedRange<Type::GPU, clear>(
                                    manager->GetCpuAddr() + offset, size, func);
                            });
    }

private:
    /**
     * @brief IteratePages Iterates L2 word manager page table.
     * @param cpu_address Start byte cpu address
     * @param size Size in bytes of the region of iterate.
     * @param func Callback for each word manager.
     * @return
     */
    template <bool create_region_on_fail, typename Func>
    bool IteratePages(VAddr cpu_address, size_t size, Func&& func) {
        RENDERER_TRACE;
        using FuncReturn = typename std::invoke_result<Func, RegionManager*, u64, size_t>::type;
        static constexpr bool BOOL_BREAK = std::is_same_v<FuncReturn, bool>;
        std::size_t remaining_size{size};
        std::size_t page_index{cpu_address >> TRACKER_HIGHER_PAGE_BITS};
        u64 page_offset{cpu_address & TRACKER_HIGHER_PAGE_MASK};
        while (remaining_size > 0) {
            const std::size_t copy_amount{
                std::min<std::size_t>(TRACKER_HIGHER_PAGE_SIZE - page_offset, remaining_size)};
            auto* manager{top_tier[page_index]};
            if (manager) {
                if constexpr (BOOL_BREAK) {
                    if (func(manager, page_offset, copy_amount)) {
                        return true;
                    }
                } else {
                    func(manager, page_offset, copy_amount);
                }
            } else if constexpr (create_region_on_fail) {
                CreateRegion(page_index);
                manager = top_tier[page_index];
                if constexpr (BOOL_BREAK) {
                    if (func(manager, page_offset, copy_amount)) {
                        return true;
                    }
                } else {
                    func(manager, page_offset, copy_amount);
                }
            }
            page_index++;
            page_offset = 0;
            remaining_size -= copy_amount;
        }
        return false;
    }

    void CreateRegion(std::size_t page_index) {
        const VAddr base_cpu_addr = page_index << TRACKER_HIGHER_PAGE_BITS;
        if (free_managers.empty()) {
            manager_pool.emplace_back();
            auto& last_pool = manager_pool.back();
            for (size_t i = 0; i < MANAGER_POOL_SIZE; i++) {
                std::construct_at(&last_pool[i], tracker, 0);
                free_managers.push_back(&last_pool[i]);
            }
        }
        // Each manager tracks a 4_MB virtual address space.
        auto* new_manager = free_managers.back();
        new_manager->SetCpuAddress(base_cpu_addr);
        free_managers.pop_back();
        new_manager->SetSummarySlot(&skippable[page_index / 64], 1ULL << (page_index % 64));
        top_tier[page_index] = new_manager;
    }

    /// True if every 4 MB region the query touches is known to exist and be entirely clean, so
    /// the whole scan can be skipped without walking top_tier at all. A cleared bit (no manager,
    /// or a manager that may be dirty) fails the check, which is the safe direction.
    bool AllRegionsSkippable(VAddr cpu_address, size_t size) const {
        const size_t first = cpu_address >> TRACKER_HIGHER_PAGE_BITS;
        const size_t last = (cpu_address + size - 1) >> TRACKER_HIGHER_PAGE_BITS;
        if (last >= NUM_HIGH_PAGES) {
            return false;
        }
        for (size_t word = first / 64; word <= last / 64; ++word) {
            const size_t lo = std::max(first, word * 64) - word * 64;
            const size_t hi = std::min(last, word * 64 + 63) - word * 64;
            const u64 mask =
                (hi - lo == 63) ? ~0ULL : (((1ULL << (hi - lo + 1)) - 1) << lo);
            if ((skippable[word].load(std::memory_order_acquire) & mask) != mask) {
                return false;
            }
        }
        return true;
    }

    PageManager* tracker;
    std::deque<std::array<RegionManager, MANAGER_POOL_SIZE>> manager_pool;
    std::vector<RegionManager*> free_managers;
    std::array<RegionManager*, NUM_HIGH_PAGES> top_tier{};
    /// One bit per 4 MB region: set means "exists and entirely clean", i.e. skippable. 32 KB for
    /// the whole 40-bit space. Maintained by RegionManager under its lock, read without it.
    std::array<std::atomic<u64>, NUM_HIGH_PAGES / 64> skippable{};
};

} // namespace VideoCore
