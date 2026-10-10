// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

// The front's own helpers the architecture-A worker parser (lean_parse.*) must call so it behaves
// exactly as ProcessGraphics (notes/architecture-a-2026-10-10.md sec. 16). Defined in liverpool.cpp
// as thin wrappers over the file-local functions.

#include <span>

#include "video_core/amdgpu/pm4_cmds.h"

namespace AmdGpu {
/// SetContextReg bumps the context epoch on a real value change only when this is true.
bool LeanViewportMemoEnabled();
/// A NOP whose body is back work (flip IRQ, guest markers): ProcessGraphics runs it through RunGfxPacket.
bool LeanNopHasBackEffect(const PM4Header* header, bool guest_markers_enabled);
/// After every packet ProcessGraphics calls BumpEpoch(opcode) unless this is true.
bool LeanKeepsSubmitEpoch(PM4ItOpcode opcode);
} // namespace AmdGpu
