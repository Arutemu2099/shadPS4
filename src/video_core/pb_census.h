// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <cstdlib>
#include "common/types.h"

namespace VideoCore {

/// SHADPS4_PB_CENSUS (diagnostic, default off; 1 = stamps / classes / timing, 3 = also per-binding
/// keys): the parallel-back census, see Rasterizer::PbCensusEnd.
inline const int g_pb_census = [] {
    const char* value = std::getenv("SHADPS4_PB_CENSUS");
    return value ? std::atoi(value) : 0;
}();

/// The back's census draw index (bumped at every DrawTail entry while the census is on).
inline std::atomic<u64> g_pb_draw{0};

/// Per draw: the newest census draw index that wrote a memo entry this draw read on a hit.
inline thread_local u64 t_pb_memo_max = 0;

inline u64 PbDraw() {
    return g_pb_draw.load(std::memory_order_relaxed);
}

inline void PbFoldMemo(u64 written) {
    if (written > t_pb_memo_max) {
        t_pb_memo_max = written;
    }
}

} // namespace VideoCore
