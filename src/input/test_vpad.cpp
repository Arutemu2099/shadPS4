// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// SHADPS4_TEST_VPAD=<file> (test switch, 2026-10-20): an SDL virtual gamepad with an Xbox layout
// and NO motion sensor and NO touchpad, so an agent can play the controller sensor mode
// (input/pad_sensor_mode.h) without a real pad. Its state is read from <file> whenever the file
// changes (checked every 30 ms): one "name value" pair per line, # = comment.
//   sticks lx ly rx ry: -1..1 (y down = +1, as SDL); triggers lt rt: 0..1;
//   buttons a b x y back start guide l3 r3 lb rb up down left right: 0 / 1.
// Names not in the file keep their last value. Not for players.

#include "input/test_vpad.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include <SDL3/SDL_gamepad.h>
#include <SDL3/SDL_joystick.h>
#include <SDL3/SDL_timer.h>

#include "common/logging/log.h"

namespace Input::TestVpad {
namespace {

std::string g_path;
SDL_Joystick* g_joystick{};
std::filesystem::file_time_type g_last_write{};

struct Name {
    const char* name;
    bool axis;
    int index;
};
constexpr Name kNames[] = {
    {"lx", true, SDL_GAMEPAD_AXIS_LEFTX},       {"ly", true, SDL_GAMEPAD_AXIS_LEFTY},
    {"rx", true, SDL_GAMEPAD_AXIS_RIGHTX},      {"ry", true, SDL_GAMEPAD_AXIS_RIGHTY},
    {"lt", true, SDL_GAMEPAD_AXIS_LEFT_TRIGGER}, {"rt", true, SDL_GAMEPAD_AXIS_RIGHT_TRIGGER},
    {"a", false, SDL_GAMEPAD_BUTTON_SOUTH},     {"b", false, SDL_GAMEPAD_BUTTON_EAST},
    {"x", false, SDL_GAMEPAD_BUTTON_WEST},      {"y", false, SDL_GAMEPAD_BUTTON_NORTH},
    {"back", false, SDL_GAMEPAD_BUTTON_BACK},   {"guide", false, SDL_GAMEPAD_BUTTON_GUIDE},
    {"start", false, SDL_GAMEPAD_BUTTON_START}, {"l3", false, SDL_GAMEPAD_BUTTON_LEFT_STICK},
    {"r3", false, SDL_GAMEPAD_BUTTON_RIGHT_STICK},
    {"lb", false, SDL_GAMEPAD_BUTTON_LEFT_SHOULDER},
    {"rb", false, SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER},
    {"up", false, SDL_GAMEPAD_BUTTON_DPAD_UP},  {"down", false, SDL_GAMEPAD_BUTTON_DPAD_DOWN},
    {"left", false, SDL_GAMEPAD_BUTTON_DPAD_LEFT},
    {"right", false, SDL_GAMEPAD_BUTTON_DPAD_RIGHT},
};

void Apply(const std::string& name, float value) {
    for (const auto& n : kNames) {
        if (name != n.name) {
            continue;
        }
        if (n.axis) {
            const bool trigger =
                n.index == SDL_GAMEPAD_AXIS_LEFT_TRIGGER || n.index == SDL_GAMEPAD_AXIS_RIGHT_TRIGGER;
            const float v = std::clamp(value, trigger ? 0.0f : -1.0f, 1.0f);
            SDL_SetJoystickVirtualAxis(g_joystick, n.index, static_cast<Sint16>(v * 32767.0f));
        } else {
            SDL_SetJoystickVirtualButton(g_joystick, n.index, value != 0.0f);
        }
        return;
    }
}

Uint32 SDLCALL Poll(void*, SDL_TimerID, Uint32 interval) {
    std::error_code ec;
    const auto t = std::filesystem::last_write_time(g_path, ec);
    if (ec || t == g_last_write) {
        return interval;
    }
    g_last_write = t;
    std::ifstream in(g_path);
    std::string line;
    std::string applied;
    while (std::getline(in, line)) {
        if (const auto hash = line.find('#'); hash != std::string::npos) {
            line.resize(hash);
        }
        std::istringstream ls(line);
        std::string name;
        float value = 0.0f;
        if (ls >> name >> value) {
            Apply(name, value);
            applied += name + "=" + std::to_string(value).substr(0, 5) + " ";
        }
    }
    LOG_INFO(Input, "TestVpad: {}", applied);
    return interval;
}

} // namespace

void Start() {
    const char* env = std::getenv("SHADPS4_TEST_VPAD");
    if (!env || !*env) {
        return;
    }
    g_path = env;
    SDL_VirtualJoystickDesc desc;
    SDL_INIT_INTERFACE(&desc);
    desc.type = SDL_JOYSTICK_TYPE_GAMEPAD;
    desc.naxes = SDL_GAMEPAD_AXIS_COUNT;
    desc.nbuttons = SDL_GAMEPAD_BUTTON_DPAD_RIGHT + 1;
    desc.axis_mask = (1u << SDL_GAMEPAD_AXIS_COUNT) - 1;
    desc.button_mask = (1u << (SDL_GAMEPAD_BUTTON_DPAD_RIGHT + 1)) - 1;
    desc.vendor_id = 0x045e; // looks like an Xbox pad to SDL's mapping
    desc.product_id = 0x02ea;
    desc.name = "shadPS4 test pad (no gyro, no touchpad)";
    const SDL_JoystickID id = SDL_AttachVirtualJoystick(&desc);
    if (id == 0) {
        LOG_ERROR(Input, "TestVpad: SDL_AttachVirtualJoystick failed: {}", SDL_GetError());
        return;
    }
    g_joystick = SDL_OpenJoystick(id);
    if (!g_joystick) {
        LOG_ERROR(Input, "TestVpad: SDL_OpenJoystick failed: {}", SDL_GetError());
        return;
    }
    SDL_AddTimer(30, &Poll, nullptr);
    LOG_WARNING(Input, "TestVpad: TEST MODE - virtual pad {} driven by {}", id, g_path);
}

} // namespace Input::TestVpad
