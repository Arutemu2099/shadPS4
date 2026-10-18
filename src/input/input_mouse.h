// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "SDL3/SDL.h"
#include "common/types.h"

namespace Input {

enum MouseMode {
    Off = 0,
    Joystick,
    Gyro,
    Touchpad,
};

bool ToggleMouseModeTo(MouseMode m);
MouseMode GetMouseMode();
void SetMouseToJoystick(int joystick);
void SetMouseParams(float mouse_deadzone_offset, float mouse_speed, float mouse_speed_offset);
void SetMouseGyroRollMode(bool mode);

void EmulateJoystick(GameController* controller, u32 interval);
void EmulateGyro(GameController* controller, u32 interval);

void ApplyMouseInputBlockers();
/// SHADPS4_TOUCHPAD_BLOCK_FIX=1: while the mouse drives the pad (touchpad / gyro / stick), its
/// presses and motion belong to the game, not to the ImGui overlay windows.
bool MouseEventsBelongToPad();
/// SHADPS4_MOUSE_AIM: R2 / L2 were added on the left / right mouse button (input_handler.cpp).
extern bool g_mouse_aim_clicks;
/// F7 (mouse to joystick) is on; with SHADPS4_MOUSE_AIM=1 the mouse turns the camera directly.
bool IsMouseJoystickMode();
/// SHADPS4_MOUSE_AIM: the hook is in, F7 is on and the mouse is not bound to the left stick.
bool MouseAimActive();
/// SHADPS4_SYNTH_MOTION helpers.
bool IsMouseGyroMode();
bool IsMouseGyroRollMode();

// Polls the mouse for changes
Uint32 MousePolling(void* param, Uint32 id, Uint32 interval);

} // namespace Input
