// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <SDL3/SDL_events.h>
#include <SDL3/SDL_stdinc.h>

namespace Input {
class GameController;
}

// Controller sensor mode (v1.1, user 2026-10-20; GitHub #10, idea from PR #9): Second Son reads
// the DualShock 4's motion sensor (the spray can) and its touchpad (fingerprint scanners, doors). A
// pad that lacks one of them gets the missing part from its sticks while the mode is on:
//  - L3 + R3 together turns the mode on / off (both presses are kept from the game); a notice says
//    what each stick does.
//  - No motion sensor: the RIGHT stick aims the can (the same synthetic pose the mouse drives with
//    SHADPS4_SYNTH_MOTION), A / Cross works like keyboard Z: hold = hold the pad sideways, a quick
//    tap = a short rattle. The right stick and A are kept from the game meanwhile. R2 still sprays.
//  - No touchpad: the LEFT stick is a finger on the touchpad - down while the stick is pushed, at
//    the stick's position (push up = finger at the top). The left stick is kept from the game.
//  - D-pad up / View (Back) stay the touchpad's left / right halves at all times (input_handler).
// A pad with both (DualSense, DualShock 4) never enters the mode; L3 + R3 reach the game as usual.
// A Switch Pro pad (real gyro, no touchpad) keeps its own gyro and gets the left-stick finger.
// Controller 0 only. SHADPS4_PAD_SENSOR_MODE=0 turns the feature off.
namespace Input::PadSensorMode {

/// WindowSDL::OnGamepadEvent, before the bindings: true = the event was used here (dropped).
/// A stick the mode uses is passed on centred, so the game's stick stops instead of sticking.
bool FilterEvent(SDL_Event& event, GameController* controller0);

/// After a pad was added or removed (event thread): the mode ends when controller 0's pad is not
/// the one it began with (unplugged, or swapped for a DualSense).
void PadsChanged(GameController* controller0);

/// 4 ms SDL timer (WindowSDL::InitTimers): stick aim into the pose, the stick finger.
Uint32 Tick(void* controller0, Uint32 id, Uint32 interval);

} // namespace Input::PadSensorMode
