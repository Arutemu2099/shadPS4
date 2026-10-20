// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "input/pad_sensor_mode.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <string>

#include <SDL3/SDL_gamepad.h>
#include <SDL3/SDL_timer.h>

#include "common/logging/log.h"
#include "core/devtools/layer.h"
#include "input/controller.h"

namespace Input::PadSensorMode {
namespace {

bool Enabled() {
    static const bool on = [] {
        const char* v = std::getenv("SHADPS4_PAD_SENSOR_MODE");
        return !(v && v[0] == '0');
    }();
    return on;
}

/// Degrees of aim per second at a full push of the right stick (SHADPS4_PAD_SENSOR_AIM_DEG).
float AimRadPerSecond() {
    static const float rad = [] {
        const char* v = std::getenv("SHADPS4_PAD_SENSOR_AIM_DEG");
        const float deg = v ? static_cast<float>(std::atof(v)) : 90.0f;
        return (deg > 0.0f ? deg : 90.0f) * 3.14159265f / 180.0f;
    }();
    return rad;
}

// Shared between the event thread (FilterEvent) and the SDL timer thread (Tick): atomics only.
std::atomic<bool> g_on{false};
std::atomic<bool> g_emulate_gyro{false};
std::atomic<bool> g_emulate_touch{false};
std::atomic<float> g_right_x{0.0f}, g_right_y{0.0f};
std::atomic<float> g_left_x{0.0f}, g_left_y{0.0f};

// Event thread only (SDL opens and closes the pads there too).
bool g_l3{}, g_r3{};
/// The stick click that completed an L3 + R3 toggle: its press and release stay with us. The first
/// one's press already reached the game; its release passes too, so no binding is left held.
int g_chord_button{-1};
u64 g_a_down_ms{};
SDL_Gamepad* g_mode_pad{}; ///< the pad the mode was entered with (another pad = the mode ends)

// Timer thread only.
bool g_finger_down{};
float g_finger_x{}, g_finger_y{};
u64 g_last_tick_ns{};

bool IsController0(SDL_JoystickID which, GameController* controller0) {
    return controller0 != nullptr && controller0->m_sdl_gamepad != nullptr &&
           SDL_GetGamepadFromID(which) == controller0->m_sdl_gamepad;
}

void SetMode(bool on, GameController* controller0) {
    if (on) {
        SDL_Gamepad* pad = controller0->m_sdl_gamepad;
        const bool has_gyro = pad != nullptr && SDL_GamepadHasSensor(pad, SDL_SENSOR_GYRO);
        const bool has_touch = pad != nullptr && SDL_GetNumGamepadTouchpads(pad) > 0;
        if (has_gyro && has_touch) {
            return; // a DualSense / DualShock 4 has both: nothing to stand in for
        }
        g_emulate_gyro.store(!has_gyro, std::memory_order_relaxed);
        g_emulate_touch.store(!has_touch, std::memory_order_relaxed);
        g_right_x.store(0.0f);
        g_right_y.store(0.0f);
        g_left_x.store(0.0f);
        g_left_y.store(0.0f);
        if (!has_gyro) {
            controller0->SetPadMotion(true); // a fresh, level pose in the middle
        }
        g_mode_pad = pad;
        g_on.store(true, std::memory_order_release);
        std::string text = "Controller sensor mode:";
        if (!has_gyro) {
            text += " right stick aims, hold A = tilt, tap A = rattle;";
        }
        if (!has_touch) {
            text += " left stick = touchpad finger;";
        }
        text += " L3+R3 to leave";
        Overlay::ShowNotice(text.c_str(), 6.0);
        LOG_INFO(Input, "PadSensorMode: on (motion sensor {}, touchpad {})",
                 has_gyro ? "the pad's own" : "from the right stick + A",
                 has_touch ? "the pad's own" : "from the left stick");
    } else {
        g_on.store(false, std::memory_order_release);
        g_mode_pad = nullptr;
        if (g_emulate_gyro.exchange(false, std::memory_order_relaxed)) {
            g_synth_pad_tilt.store(false, std::memory_order_relaxed);
            g_synth_rattle_until.store(0, std::memory_order_relaxed);
            controller0->SetPadMotion(false); // tilt and aim reset
        }
        g_emulate_touch.store(false, std::memory_order_relaxed); // Tick lifts the finger
        Overlay::ShowNotice("Controller sensor mode off");
        LOG_INFO(Input, "PadSensorMode: off");
    }
}

/// A push of the right stick (-1..1) as an aim rate (-1..1): linear past a small dead zone.
float AimRate(float v) {
    constexpr float dead = 0.12f;
    const float a = std::abs(v);
    if (a <= dead) {
        return 0.0f;
    }
    return std::copysign(std::min((a - dead) / (1.0f - dead), 1.0f), v);
}

} // namespace

bool FilterEvent(SDL_Event& event, GameController* controller0) {
    if (!Enabled()) {
        return false;
    }
    // Another pad took controller 0's place (an Xbox pad swapped for a DualSense): the mode ends,
    // so the new pad's own sensors and buttons work again.
    if (g_on.load(std::memory_order_acquire) &&
        (controller0 == nullptr || controller0->m_sdl_gamepad != g_mode_pad)) {
        SetMode(false, controller0);
    }
    switch (event.type) {
    case SDL_EVENT_GAMEPAD_AXIS_MOTION: {
        if (!g_on.load(std::memory_order_acquire) || !IsController0(event.gaxis.which, controller0)) {
            return false;
        }
        const float v = std::clamp(static_cast<float>(event.gaxis.value) / 32767.0f, -1.0f, 1.0f);
        const bool gyro = g_emulate_gyro.load(std::memory_order_relaxed);
        const bool touch = g_emulate_touch.load(std::memory_order_relaxed);
        switch (event.gaxis.axis) {
        case SDL_GAMEPAD_AXIS_RIGHTX:
        case SDL_GAMEPAD_AXIS_RIGHTY:
            if (!gyro) {
                return false;
            }
            (event.gaxis.axis == SDL_GAMEPAD_AXIS_RIGHTX ? g_right_x : g_right_y).store(v);
            event.gaxis.value = 0; // the game's camera stick rests
            return false;
        case SDL_GAMEPAD_AXIS_LEFTX:
        case SDL_GAMEPAD_AXIS_LEFTY:
            if (!touch) {
                return false;
            }
            (event.gaxis.axis == SDL_GAMEPAD_AXIS_LEFTX ? g_left_x : g_left_y).store(v);
            event.gaxis.value = 0; // Delsin does not walk while the stick is the finger
            return false;
        default:
            return false;
        }
    }
    case SDL_EVENT_GAMEPAD_BUTTON_DOWN:
    case SDL_EVENT_GAMEPAD_BUTTON_UP: {
        if (!IsController0(event.gbutton.which, controller0)) {
            return false;
        }
        const bool down = event.type == SDL_EVENT_GAMEPAD_BUTTON_DOWN;
        const u8 button = event.gbutton.button;
        if (button == SDL_GAMEPAD_BUTTON_LEFT_STICK || button == SDL_GAMEPAD_BUTTON_RIGHT_STICK) {
            const bool both_before = g_l3 && g_r3;
            (button == SDL_GAMEPAD_BUTTON_LEFT_STICK ? g_l3 : g_r3) = down;
            if (down && !both_before && g_l3 && g_r3) {
                const bool was_on = g_on.load(std::memory_order_acquire);
                SetMode(!was_on, controller0);
                if (g_on.load(std::memory_order_acquire) != was_on) {
                    g_chord_button = button;
                    return true;
                }
            }
            if (button == g_chord_button) {
                if (!down) {
                    g_chord_button = -1;
                }
                return true;
            }
            return false;
        }
        if (button == SDL_GAMEPAD_BUTTON_SOUTH && g_on.load(std::memory_order_acquire) &&
            g_emulate_gyro.load(std::memory_order_relaxed)) {
            // Keyboard Z's rule (sdl_window): hold = the pad on its side; released within 250 ms
            // = a 600 ms rattle burst.
            g_synth_pad_tilt.store(down, std::memory_order_relaxed);
            const u64 now = SDL_GetTicks();
            if (down) {
                g_a_down_ms = now;
            } else if (now - g_a_down_ms < 250) {
                g_synth_rattle_until.store(now + 600, std::memory_order_relaxed);
            }
            return true;
        }
        return false;
    }
    default:
        return false;
    }
}

Uint32 Tick(void* param, Uint32 id, Uint32 interval) {
    auto* controller0 = static_cast<GameController*>(param);
    const u64 now_ns = SDL_GetTicksNS();
    const float dt = g_last_tick_ns != 0
                         ? std::min(static_cast<float>(now_ns - g_last_tick_ns) / 1e9f, 0.1f)
                         : 0.0f;
    g_last_tick_ns = now_ns;
    if (!Enabled() || controller0 == nullptr) {
        return interval;
    }
    const bool on = g_on.load(std::memory_order_acquire);
    if (on && g_emulate_gyro.load(std::memory_order_relaxed) && dt > 0.0f) {
        // Into the pose's own input, in its units (the mouse's pixels): the same pose code turns
        // it into the orientation, the rotation rate and gravity.
        const float px = AimRadPerSecond() * dt / g_synth_rad_per_px;
        const float ax = AimRate(g_right_x.load(std::memory_order_relaxed));
        const float ay = AimRate(g_right_y.load(std::memory_order_relaxed));
        if (ax != 0.0f) {
            g_synth_dx.fetch_add(ax * px, std::memory_order_relaxed);
        }
        if (ay != 0.0f) {
            g_synth_dy.fetch_add(ay * px, std::memory_order_relaxed);
        }
    }
    // The left-stick finger: down while the stick is pushed, where it points.
    bool want_down = false;
    float x = g_finger_x, y = g_finger_y;
    if (on && g_emulate_touch.load(std::memory_order_relaxed)) {
        const float lx = g_left_x.load(std::memory_order_relaxed);
        const float ly = g_left_y.load(std::memory_order_relaxed);
        if (lx * lx + ly * ly > 0.25f * 0.25f) {
            want_down = true;
            x = std::clamp(0.5f + lx * 0.5f, 0.0f, 1.0f);
            y = std::clamp(0.5f + ly * 0.5f, 0.0f, 1.0f);
        }
    }
    if (want_down != g_finger_down || (want_down && (x != g_finger_x || y != g_finger_y))) {
        controller0->SetTouchpadState(0, want_down, x, y);
        g_finger_down = want_down;
        g_finger_x = x;
        g_finger_y = y;
    }
    return interval;
}

} // namespace Input::PadSensorMode
