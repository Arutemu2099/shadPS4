// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>

#include "common/div_ceil.h"
#include "common/logging/log.h"
#include "core/emulator_settings.h"

#ifdef __unix__
#include "common/adaptive_mutex.h"
#else
#include "common/spin_lock.h"
#endif
#include "common/debug.h"
#include "common/fault_monitor.h"
#include "common/types.h"
#include "video_core/buffer_cache/region_definitions.h"
#include "video_core/page_manager.h"

namespace VideoCore {

#ifdef PTHREAD_ADAPTIVE_MUTEX_INITIALIZER_NP
using LockType = Common::AdaptiveMutex;
#else
using LockType = Common::SpinLock;
#endif

/**
 * Allows tracking CPU and GPU modification of pages in a contigious 16MB virtual address region.
 * Information is stored in bitsets for spacial locality and fast update of single pages.
 */
class RegionManager {
public:
    explicit RegionManager(PageManager* tracker_, VAddr cpu_addr_)
        : tracker{tracker_}, cpu_addr{cpu_addr_} {
        cpu.Fill();
        gpu.Clear();
        writeable.Fill();
        readable.Fill();
    }
    explicit RegionManager() = default;

    /**
     * True if this region may have any CPU-modified page. Read WITHOUT the region lock, so it is
     * deliberately conservative: a stale `true` only costs the caller the work it would have done
     * anyway, while a stale `false` would skip a real upload and render stale geometry.
     *
     * False therefore has to imply "the cpu bitset was empty". That holds because the bitset has
     * exactly two mutators, `ChangeRegionState<Type::CPU>` and `ForEachModifiedRange`, both under
     * the region lock, and both maintain this flag: it is raised BEFORE bits are set (so a reader
     * seeing false cannot be racing a set that already happened), and lowered only after observing
     * `None()` on the whole region.
     *
     * Starts true because the constructor does `cpu.Fill()` - a fresh region is entirely dirty.
     */
    bool MayHaveCpuModified() const noexcept {
        return any_cpu.load(std::memory_order_acquire);
    }

    void SetCpuAddress(VAddr new_cpu_addr) {
        cpu_addr = new_cpu_addr;
    }

    VAddr GetCpuAddr() const {
        return cpu_addr;
    }

    static constexpr size_t SanitizeAddress(size_t address) {
        return static_cast<size_t>(std::max<s64>(static_cast<s64>(address), 0LL));
    }

    template <Type type>
    RegionBits& GetRegionBits() noexcept {
        if constexpr (type == Type::CPU) {
            return cpu;
        } else if constexpr (type == Type::GPU) {
            return gpu;
        }
    }

    template <Type type>
    const RegionBits& GetRegionBits() const noexcept {
        if constexpr (type == Type::CPU) {
            return cpu;
        } else if constexpr (type == Type::GPU) {
            return gpu;
        }
    }

    /**
     * Change the state of a range of pages
     *
     * @param dirty_addr    Base address to mark or unmark as modified
     * @param size          Size in bytes to mark or unmark as modified
     */
    template <Type type, bool enable>
    void ChangeRegionState(u64 dirty_addr, u64 size) noexcept(type == Type::GPU) {
        RENDERER_TRACE;
        const size_t offset = dirty_addr - cpu_addr;
        const size_t start_page = SanitizeAddress(offset) / TRACKER_BYTES_PER_PAGE;
        const size_t end_page =
            Common::DivCeil(SanitizeAddress(offset + size), TRACKER_BYTES_PER_PAGE);
        if (start_page >= NUM_PAGES_PER_REGION || end_page <= start_page) {
            return;
        }

        RegionBits& bits = GetRegionBits<type>();
        if constexpr (enable) {
            if constexpr (type == Type::CPU) {
                // Raise the summary BEFORE the bits, so a lock-free reader can never observe
                // false while a set that precedes it in program order has already landed.
                any_cpu.store(true, std::memory_order_release);
            }
            bits.SetRange(start_page, end_page);
        } else {
            bits.UnsetRange(start_page, end_page);
            if constexpr (type == Type::CPU) {
                if (bits.None()) {
                    any_cpu.store(false, std::memory_order_release);
                }
            }
        }
        if constexpr (type == Type::CPU) {
            UpdateProtection<!enable, false>();
        } else if (EmulatorSettings.GetReadbacksMode() == GpuReadbacksMode::Precise) {
            UpdateProtection<enable, true>();
        }
    }

    /**
     * Loop over each page in the given range, turn off those bits and notify the tracker if
     * needed. Call the given function on each turned off range.
     *
     * @param query_cpu_range Base CPU address to loop over
     * @param size            Size in bytes of the CPU range to loop over
     * @param func            Function to call for each turned off region
     */
    template <Type type, bool clear>
    void ForEachModifiedRange(VAddr query_cpu_range, s64 size, auto&& func) {
        RENDERER_TRACE;
        const size_t offset = query_cpu_range - cpu_addr;
        const size_t start_page = SanitizeAddress(offset) / TRACKER_BYTES_PER_PAGE;
        const size_t end_page =
            Common::DivCeil(SanitizeAddress(offset + size), TRACKER_BYTES_PER_PAGE);
        if (start_page >= NUM_PAGES_PER_REGION || end_page <= start_page) {
            return;
        }

        RegionBits& bits = GetRegionBits<type>();

        // Nothing modified in the queried range means there is no work in this call at all, and
        // the common case by a wide margin: measured on the per-draw upload path, ~76% of calls
        // find no CPU-modified page. Without this the call still materialises a 1024-bit masked
        // copy, runs UnsetRange over it, and diffs the whole 4 MB region's protection state.
        //
        // Only Type::CPU. The skip is sound because `writeable` is re-synced to `cpu` on every
        // path that mutates `cpu` (ChangeRegionState<CPU> and this function, both under the
        // region lock), so with no bit set in range the UnsetRange is a no-op and
        // UpdateProtection's `cpu ^ writeable` diff is necessarily empty. The GPU side has no
        // such invariant: ChangeRegionState<GPU> only touches `readable` under Precise, while
        // this function updates it whenever readbacks are not Disabled, so under Relaxed `gpu`
        // and `readable` legitimately diverge and the diff must still be taken.
        if constexpr (type == Type::CPU) {
            if (!bits.AnyInRange(start_page, end_page)) {
                return;
            }
        }

        RegionBits mask(bits, start_page, end_page);

        if constexpr (clear) {
            bits.UnsetRange(start_page, end_page);
            if constexpr (type == Type::CPU) {
                if (bits.None()) {
                    any_cpu.store(false, std::memory_order_release);
                }
                UpdateProtection<true, false>();
            } else if (EmulatorSettings.GetReadbacksMode() != GpuReadbacksMode::Disabled) {
                UpdateProtection<false, true>();
            }
        }

        for (const auto& [start, end] : mask) {
            func(cpu_addr + start * TRACKER_BYTES_PER_PAGE, (end - start) * TRACKER_BYTES_PER_PAGE);
        }
    }

    /**
     * Returns true when a region has been modified
     *
     * @param offset Offset in bytes from the start of the buffer
     * @param size   Size in bytes of the region to query for modifications
     */
    template <Type type>
    [[nodiscard]] bool IsRegionModified(u64 offset, u64 size) noexcept {
        RENDERER_TRACE;
        const size_t start_page = SanitizeAddress(offset) / TRACKER_BYTES_PER_PAGE;
        const size_t end_page =
            Common::DivCeil(SanitizeAddress(offset + size), TRACKER_BYTES_PER_PAGE);
        if (start_page >= NUM_PAGES_PER_REGION || end_page <= start_page) {
            return false;
        }

        const RegionBits& bits = GetRegionBits<type>();
        RegionBits test(bits, start_page, end_page);
        return test.Any();
    }

    LockType lock;

private:
    /**
     * Notify tracker about changes in the CPU tracking state of a word in the buffer
     *
     * @param word_index   Index to the word to notify to the tracker
     * @param current_bits Current state of the word
     * @param new_bits     New state of the word
     *
     * @tparam track True when the tracker should start tracking the new pages
     */
    template <bool track, bool is_read>
    void UpdateProtection() {
        RENDERER_TRACE;
        RegionBits mask = is_read ? (~gpu ^ readable) : (cpu ^ writeable);
        if (mask.None()) {
            return;
        }
        if constexpr (is_read) {
            readable = ~gpu;
        } else {
            writeable = cpu;
        }
        // This is the only path out of the tracker that reaches a VirtualProtect syscall. Count
        // and time it: it is the one candidate in the per-draw sync path that can plausibly cost
        // microseconds, and bitset arithmetic cannot.
        const auto t_protect = Common::FaultMonitor::Now();
        tracker->UpdatePageWatchersForRegion<track, is_read>(cpu_addr, mask);
        Common::FaultMonitor::CountProtect<is_read>(Common::FaultMonitor::SinceNs(t_protect));
    }

    PageManager* tracker;
    VAddr cpu_addr = 0;
    // Conservative summary of `cpu`, readable without the lock. See MayHaveCpuModified().
    std::atomic<bool> any_cpu{true};
    RegionBits cpu;
    RegionBits gpu;
    RegionBits writeable;
    RegionBits readable;
};

} // namespace VideoCore
