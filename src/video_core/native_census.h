// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include "common/types.h"

namespace VideoCore {

// SHADPS4_NATIVE_CENSUS=1 (diagnostic, default off; native-path study 2026-10-09): the host-side
// vertex / index bindings of the current draw, filled by the buffer cache on the translating
// thread, read by the rasterizer's batchability census at the end of the draw.
struct NativeCensusVi {
    u32 n{};
    std::array<u64, 16> buf{};
    std::array<u64, 16> off{};
    std::array<u32, 16> stride{};
    u64 ib_buf{};
    u64 ib_off{};
    u32 ib_type{};
};
extern const bool g_native_census;
/// SHADPS4_NATIVE_PASS: 0 off, 1 native shadow batches, 2 verify (generic path renders).
extern const int g_native_pass;

/// SHADPS4_NATIVE_PASS join: one draw's vertex / index bindings as the batch would use them.
struct NativeJoinVi {
    u32 n{};
    std::array<u64, 16> buf{};
    std::array<u64, 16> off{};
    std::array<u32, 16> stride{};
    u64 ib_buf{};
    u64 ib_off{};
};
inline thread_local NativeCensusVi t_native_vi{};

} // namespace VideoCore
