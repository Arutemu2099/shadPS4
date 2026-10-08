// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <cstdlib>
#include "common/fault_monitor.h"
#include "common/types.h"

namespace AmdGpu::MtbCensus {

/// SHADPS4_MTB_CENSUS=1 (diagnostic, default off; notes/multithreaded-back-design-2026-10-08.md P0):
/// cuts the back's record stream into virtual segments (at sequence-point records and every
/// SHADPS4_MTB_SEGMENT draws/dispatches, default 128) and reports how much of the back's time is
/// in draws a lookup-only worker could translate (before the segment's first serial-only event).
inline const bool g_enabled = [] {
    const char* env = std::getenv("SHADPS4_MTB_CENSUS");
    return env && env[0] == '1';
}();

enum Reason : u32 {
    None = 0,
    BufferCreate,
    BufferDelete,
    BufferUpload,
    ImageRegister,
    ImageUnregister,
    ImageUpload,
    StreamAlloc,
    OcclusionSlot,
    Count,
};

/// The serial-only event seen during the current draw / dispatch (first one wins). CP thread.
inline u32 g_draw_event = None;

/// Hooks: called where the serial path does something a lookup-only worker may not.
inline void Note(Reason reason) {
    if (g_enabled && g_draw_event == None && ::Common::FaultMonitor::t_is_cp_thread) {
        g_draw_event = reason;
    }
}

} // namespace AmdGpu::MtbCensus
