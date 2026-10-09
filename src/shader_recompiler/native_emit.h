// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"
#include "shader_recompiler/resource.h"

namespace Shader {

// SHADPS4_NATIVE_PASS (2026-10-09, notes/native-renderer-2026-10-09.md sec. 3): a native batch
// variant of a graphics pipeline draws a run of guest draws with one indirect multi-draw. Its
// vertex shader reads each draw's user-data registers and full buffer byte offsets from a
// records storage buffer indexed by the draw index, instead of the push constants. Set by the
// pipeline cache around the compile of a native variant (thread-local: the compile runs on the
// calling thread).
struct NativeEmit {
    bool active{};
    u32 records_binding{};
};
inline thread_local NativeEmit t_native_emit{};

constexpr u32 NativeRecordUdOffset = 0;
constexpr u32 NativeRecordOffOffset = NUM_USER_DATA_REGS;
/// Each Guest binding's V# size in bytes: native loads past it return 0, as the generic path's
/// descriptor range makes robustBufferAccess do (Opus code review 2026-10-09 #1).
constexpr u32 NativeRecordSizeOffset = NUM_USER_DATA_REGS + NUM_BUFFERS;
constexpr u32 NativeRecordDwords = NUM_USER_DATA_REGS + 2 * NUM_BUFFERS; // 96 dwords, 384 bytes

} // namespace Shader
