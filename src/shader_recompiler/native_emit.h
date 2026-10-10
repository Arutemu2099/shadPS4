// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <cstdlib>
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
    /// The flat draw-index varying's location: just above every VS output / PS input (the
    /// fixed 31 overflowed maxVertexOutputComponents with the built-ins, s324 validation).
    u32 draw_location{};
    u32 debug_binding{};
    u32 vs_hash{};
    /// SHADPS4_NATIVE_GPU_DIRECT=1 (notes/native-path-2026-10-09.md sec. 10): the record of draw
    /// i is entry (table[first] + i) & table[mask] of the records buffer (a game job thread's
    /// record ring, or the batch's own records with first 0); each Guest binding's record word and
    /// each pulled location's are guest addresses (low 32 bits) minus the table's base.
    bool ring{};
    u32 table_binding{};
    /// SHADPS4_NATIVE_BINDLESS (sec. 12): the PS samples image i / sampler j from the global
    /// arrays (set 1, binding 0 / 1) at the record's slot words NativeRecordImgOffset + i /
    /// NativeRecordSmpOffset + j. Set-0 numbering is unchanged.
    bool bindless{};
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
/// SHADPS4_NATIVE_BINDLESS: PS image / sampler slots in the global descriptor arrays.
constexpr u32 NativeBindlessMax = 16;
constexpr u32 NativeRecordImgOffset = NativeRecordVsizeOffset + NativeMaxPulledLocations; // 128
constexpr u32 NativeRecordSmpOffset = NativeRecordImgOffset + NativeBindlessMax;           // 144
constexpr u32 NativeRecordDwords = NativeRecordSmpOffset + NativeBindlessMax;              // 160
static_assert(NativeRecordDwords == 160);
/// SHADPS4_NATIVE_GPU_DIRECT: the per-batch table (ring mode).
constexpr u32 NativeTableFirst = 0;
constexpr u32 NativeTableMask = 1;
constexpr u32 NativeTableBase = 2;
constexpr u32 NativeTableVbase = 2 + NUM_BUFFERS;
/// Coverage (sec. 11): push-layout user-data slots taken from the table (mask) and their values;
/// Guest binding slots whose size comes from the table (mask over two dwords) and the sizes
/// (their offset is folded into the base: record word - base = the opener's offset).
constexpr u32 NativeTablePushInh = NativeTableVbase + NativeMaxPulledLocations; // 58
constexpr u32 NativeTableInhSlots = NativeTablePushInh + 1;                      // 59, 60
constexpr u32 NativeTableUd = 64;
constexpr u32 NativeTableSize = NativeTableUd + NUM_USER_DATA_REGS; // 80
constexpr u32 NativeTableDwords = 128;
static_assert(NativeTableInhSlots + 2 <= NativeTableUd);
static_assert(NativeTableSize + NUM_BUFFERS <= NativeTableDwords);
/// SHADPS4_NATIVE_GPU_DIRECT=1: native variants are compiled in ring mode.
inline const bool g_native_ring = [] {
    const char* v = std::getenv("SHADPS4_NATIVE_GPU_DIRECT");
    return v && (v[0] == '1' || v[0] == '3') && v[1] == 0;
}();
/// The highest location the flat draw-index varying may use (with 4 built-in components the
/// 128-component output budget allows locations up to 30; one spare).
constexpr u32 NativeDrawIndexMaxLocation = 29;

} // namespace Shader
