// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/types.h"

namespace VideoCore {

/// Where one shader's code sits inside the game's own files (2026-10-16,
/// notes/study-shader-stutter-2026-10-16.md sec. 4 step 1). Offsets only: no code is copied.
struct ShaderLocation {
    u32 file{};        ///< index into ShaderIndex::files
    u64 offset{};      ///< file offset of the first code dword (the 0xBEEB03FF token)
    u32 code_length{}; ///< bytes (BinaryInfo::length)
    u32 type{};        ///< BinaryInfo::type: 0 pixel, 1 vertex, 4 compute
    u32 crc32{};       ///< BinaryInfo::crc32 (a seed stamp checks it, with code_length)
};

struct ShaderIndex {
    std::vector<std::filesystem::path> files;
    std::unordered_map<u64, ShaderLocation> shaders; ///< by BinaryInfo::shader_hash
};

/// Finds every shader binary in the game folder: an "OrbShdr" BinaryInfo record whose code start
/// is the nearest 0xBEEB03FF token before it with (code[1] + 1) * 8 == record - code (the rule
/// of SearchBinaryInfo). Reads every file once on several threads; the result is kept in
/// `index_file` with a fingerprint of the folder (file count, bytes, newest write time) and read
/// back on later launches until the folder changes.
ShaderIndex BuildShaderIndex(const std::filesystem::path& game_folder,
                             const std::filesystem::path& index_file);

/// xxh3 of the running emulator EXE file (0 if unreadable): with the scm description, the seed
/// stamp's build id (uncommitted builds share one description).
u64 ExecutableHash();

/// The shader's code read from its game file; empty on any read error.
std::vector<u32> ReadShaderCode(const ShaderIndex& index, const ShaderLocation& location);

} // namespace VideoCore
