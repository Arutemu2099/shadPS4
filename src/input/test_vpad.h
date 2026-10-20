// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

namespace Input::TestVpad {

/// SHADPS4_TEST_VPAD=<file> (test switch): attach an SDL virtual Xbox-style pad without a gyro or
/// touchpad, driven by <file> (test_vpad.cpp). WindowSDL::InitTimers, once; no-op without the flag.
void Start();

} // namespace Input::TestVpad
