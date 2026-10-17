// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstdlib>
#include <cmath>

#include "common/assert.h"
#include "common/types.h"
#include "input/controller.h"
#include "input/input_handler.h"
#include "input_mouse.h"

#include <common/singleton.h>
#include <emulator.h>
#include "SDL3/SDL.h"

extern Frontend::WindowSDL* g_window;

namespace Input {

extern std::list<std::pair<InputEvent, bool>> pressed_keys;

int mouse_joystick_binding = 0;
float mouse_deadzone_offset = 0.5, mouse_speed = 1, mouse_speed_offset = 0.1250;
bool mouse_gyro_roll_mode = false;
Uint32 mouse_polling_id = 0;
MouseMode mouse_mode = MouseMode::Off;

// Switches mouse to a set mode or turns mouse emulation off if it was already in that mode.
// Returns whether the mode is turned on.
bool ToggleMouseModeTo(MouseMode m) {
    if (mouse_mode == m) {
        mouse_mode = MouseMode::Off;
    } else {
        mouse_mode = m;
    }
    return mouse_mode == m;
}

void SetMouseToJoystick(int joystick) {
    mouse_joystick_binding = joystick;
}

void SetMouseParams(float mdo, float ms, float mso) {
    mouse_deadzone_offset = mdo;
    mouse_speed = ms;
    mouse_speed_offset = mso;
}

void SetMouseGyroRollMode(bool mode) {
    mouse_gyro_roll_mode = mode;
}

bool IsMouseGyroMode() {
    return mouse_mode == MouseMode::Gyro;
}

bool IsMouseGyroRollMode() {
    return mouse_gyro_roll_mode;
}

void EmulateJoystick(GameController* controller, u32 interval) {

    Axis axis_x, axis_y;
    switch (mouse_joystick_binding) {
    case 1:
        axis_x = Axis::LeftX;
        axis_y = Axis::LeftY;
        break;
    case 2:
        axis_x = Axis::RightX;
        axis_y = Axis::RightY;
        break;
    default:
        return; // no update needed
    }

    float d_x = 0, d_y = 0;
    SDL_GetRelativeMouseState(&d_x, &d_y);

    float output_speed =
        SDL_clamp(sqrt(d_x * d_x + d_y * d_y) * mouse_speed + mouse_speed_offset * 128,
                  mouse_deadzone_offset * 128, 128.0);

    float angle = atan2(d_y, d_x);
    float a_x = cos(angle) * output_speed, a_y = sin(angle) * output_speed;

    if (d_x != 0 || d_y != 0) {
        controller->Axis(axis_x, GetAxis(-0x80, 0x7f, a_x), false);
        controller->Axis(axis_y, GetAxis(-0x80, 0x7f, a_y), false);
    } else {
        controller->Axis(axis_x, GetAxis(-0x80, 0x7f, 0), false);
        controller->Axis(axis_y, GetAxis(-0x80, 0x7f, 0), false);
    }
}

constexpr float constant_down_accel[3] = {0.0f, 9.81f, 0.0f};
void EmulateGyro(GameController* controller, u32 interval) {
    if (g_synth_motion) {
        return; // SHADPS4_SYNTH_MOTION: the pose comes from the 4 ms pushes (controller.cpp)
    }
    float d_x = 0, d_y = 0;
    SDL_GetRelativeMouseState(&d_x, &d_y);
    controller->UpdateAcceleration(constant_down_accel);
    float gyro_from_mouse[3] = {-d_y / 100, -d_x / 100, 0.0f};
    if (mouse_gyro_roll_mode) {
        gyro_from_mouse[1] = 0.0f;
        gyro_from_mouse[2] = -d_x / 100;
    }
    controller->UpdateGyro(gyro_from_mouse);
}

void EmulateTouchpad(GameController* controller, u32 interval) {
    float x, y;
    SDL_MouseButtonFlags mouse_buttons = SDL_GetMouseState(&x, &y);
    controller->SetTouchpadState(0, (mouse_buttons & SDL_BUTTON_LMASK) != 0,
                                 std::clamp(x / g_window->GetWidth(), 0.0f, 1.0f),
                                 std::clamp(y / g_window->GetHeight(), 0.0f, 1.0f));
    controller->Button(Libraries::Pad::OrbisPadButtonDataOffset::TouchPad,
                       (mouse_buttons & SDL_BUTTON_RMASK) != 0);
}

static bool TouchpadBlockFix() {
    static const bool block_fix = [] {
        const char* v = std::getenv("SHADPS4_TOUCHPAD_BLOCK_FIX");
        return v && v[0] == '1';
    }();
    return block_fix;
}

bool MouseEventsBelongToPad() {
    // 2026-10-17 (user: keyboard O = R2 ignored while holding the touchpad in the pad's upper
    // right): the click landed on the FPS counter (top right). ImGui started a window drag
    // (ActiveId != 0 -> WantCaptureKeyboard), so every keyboard key-down went to ImGui while the
    // button was held; a real pad's R2 still passed.
    return TouchpadBlockFix() && mouse_mode != MouseMode::Off;
}

void ApplyMouseInputBlockers() {
    // SHADPS4_TOUCHPAD_BLOCK_FIX=1 (2026-10-14, input study): block only the MOUSE buttons. The
    // ids 1 / 3 are also a pad's LEFT-Y / RIGHT-Y axes and Circle / Triangle, which touchpad mode
    // used to swallow.
    const bool block_fix = TouchpadBlockFix();
    switch (mouse_mode) {
    case MouseMode::Touchpad:
        for (auto& k : pressed_keys) {
            if (block_fix && k.first.input.type != InputType::KeyboardMouse) {
                continue;
            }
            if (k.first.input.sdl_id == SDL_BUTTON_LEFT ||
                k.first.input.sdl_id == SDL_BUTTON_RIGHT) {
                k.second = true;
            }
        }
        break;
    default:
        break;
    }
}

Uint32 MousePolling(void* param, Uint32 id, Uint32 interval) {
    auto* controller = (GameController*)param;
    switch (mouse_mode) {
    case MouseMode::Joystick:
        EmulateJoystick(controller, interval);
        break;
    case MouseMode::Gyro:
        EmulateGyro(controller, interval);
        break;
    case MouseMode::Touchpad:
        EmulateTouchpad(controller, interval);
        break;

    default:
        break;
    }
    return interval;
}

} // namespace Input
