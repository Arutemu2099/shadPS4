// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include "common/types.h"
#include "shader_recompiler/resource.h"

namespace Shader {

/// SHADPS4_NATIVE_COLOUR: how the native vertex shader decodes one fetch attribute it pulls
/// itself from the vertex storage buffer (what the Vulkan vertex input format would do).
enum class NativeVtxKind : u8 {
    None,   // not pullable
    Float,  // 32-bit float components
    Uint,   // 32-bit integer components, as uint
    Sint,   // 32-bit integer components, as int
    Half,   // 16-bit float
    Unorm,  // 8 / 16-bit normalized
    Snorm,
    Uint16, // 8 / 16-bit integer, zero-extended
    Sint16, // 8 / 16-bit integer, sign-extended
    Uscaled,
    Sscaled,
};
struct NativeVtxAttr {
    NativeVtxKind kind{};
    u8 bits{};       // component width: 8, 16 or 32; 0 = packed 10:10:10:2 in one dword
    u8 count{};      // components: 1-4
    u8 valid{};
    u32 stride{};    // V# stride (baked: part of the variant key)
};

// SHADPS4_NATIVE_PASS (2026-10-09, notes/native-renderer-2026-10-09.md sec. 3): a native batch
// variant of a graphics pipeline draws a run of guest draws with one indirect multi-draw. Its
// vertex shader reads each draw's user-data registers and full buffer byte offsets from a
// records storage buffer indexed by the draw index, instead of the push constants. Set by the
// pipeline cache around the compile of a native variant (thread-local: the compile runs on the
// calling thread).
// SHADPS4_NATIVE_COLOUR (notes/colour-pass-2026-10-09.md sec. 3): `colour` also makes the
// fragment shader native (its record base comes from a flat draw-index varying the vertex shader
// writes at NativeDrawIndexLocation), and `pull` makes the vertex shader fetch every vertex
// attribute itself from one whole vertex storage buffer at `vertex_binding`, at the record's
// per-location byte offset, so draws whose vertex streams sit at unrelated offsets can share
// one multi-draw.
struct NativeEmit {
    bool active{};
    u32 records_binding{};
    bool colour{};
    bool pull{};
    u32 vertex_binding{};
    std::array<NativeVtxAttr, 32> attrs{}; // by fetch semantic (= vertex input location)
    /// Diagnostic (SHADPS4_NATIVE_COLOUR_DIAG 128): locations (bit mask) that keep a real
    /// vertex input instead of being pulled (valid only for unjoined batches).
    u32 keep_inputs{};
    /// Diagnostic (SHADPS4_NATIVE_COLOUR_DIAG 4096): every attribute keeps its real vertex input
    /// AND is pulled; mismatches go to a debug SSBO at `debug_binding` (layout: dword 0 count,
    /// entries of 16 dwords from dword 16: location, vertex index, draw, stream offset, pulled
    /// x y z w bits, real x y z w bits, vs hash, kind, bits, count; slot 64 = discard).
    bool compare{};
    u32 debug_binding{};
    u32 vs_hash{};
};
inline thread_local NativeEmit t_native_emit{};

constexpr u32 NativeRecordUdOffset = 0;
constexpr u32 NativeRecordOffOffset = NUM_USER_DATA_REGS;
/// Each Guest binding's V# size in bytes: native loads past it return 0, as the generic path's
/// descriptor range makes robustBufferAccess do (Opus code review 2026-10-09 #1).
constexpr u32 NativeRecordSizeOffset = NUM_USER_DATA_REGS + NUM_BUFFERS;
/// SHADPS4_NATIVE_COLOUR: per vertex input location (< 16), the byte offset of the draw's
/// stream in the vertex storage buffer and the bytes available from there (the bound size).
constexpr u32 NativeMaxPulledLocations = 16;
constexpr u32 NativeRecordVoffOffset = NUM_USER_DATA_REGS + 2 * NUM_BUFFERS;
constexpr u32 NativeRecordVsizeOffset = NativeRecordVoffOffset + NativeMaxPulledLocations;
constexpr u32 NativeRecordDwords = NativeRecordVsizeOffset + NativeMaxPulledLocations; // 128
static_assert(NativeRecordDwords == 128);
/// The flat varying carrying the draw index from the native VS to the native PS.
constexpr u32 NativeDrawIndexLocation = 31;

} // namespace Shader
