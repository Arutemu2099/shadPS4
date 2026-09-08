// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <cstdint>

namespace VideoCore {

// Global conservative invalidation for clean-range memos. Only publish a new
// epoch AFTER the dirty summaries and bits have been updated under their lock.
inline std::atomic<std::uint64_t> g_dirty_epoch{1};

template <typename Mutation>
void PublishCpuDirty(Mutation&& mutation) noexcept {
    mutation();
    g_dirty_epoch.fetch_add(1, std::memory_order_release);
}

} // namespace VideoCore
