// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <optional>
#include <string>
#include <vector>
#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"
#include "video_core/renderer_vulkan/vk_dlss.h"

// DLSS 5 Neural Rendering on top of SHADPS4_DLSS=1 (LATER-PLAN 9; private build, 2026-10-18).
// The same deal as DLSS 4.5: the player's own files are looked for, and anything missing or any
// failure leaves the frame exactly as DLSS 4.5 alone makes it. Looked for, first match wins:
// SHADPS4_DLSS5_DIR, dlss5\ next to the EXE, and ..\..\dlss5 (a test bundle in artifacts\).
// The folder holds the model (nvngx_dlssnr.dll, not public yet) and the forwarder
// nvngx.dll_dlssnr.dll built from github.com/Skynizz/optiscaler-dlss5 (GPL-3, dlss5\forwarder-src):
// the model only accepts calls from a module whose path contains "nvngx.dll", and the forwarder
// is that module. SHADPS4_DLSS5=0 never loads them.
//
// Order per frame (Rasterizer::DlssResolve): the game's 1080p tonemapped frame (no HUD yet) ->
// DLSS 5 -> DLSS 4.5 (DLAA or the upscale to the window) -> the HUD on top. So the model never
// sees or touches the HUD.
// Keys: End = DLSS 5 on / off, Page Up = the DLSS 5 panel (Home stays DLSS 4.5 on / off).
namespace Vulkan {
class Instance;
}

namespace Vulkan::Dlss5 {

/// What the panel edits. The model reads its tuning once, when the feature is built, so a change
/// here rebuilds the feature on the next frame (Apply).
struct Settings {
    bool enabled = true;         ///< End
    int style = 0;               ///< 0 standard, 1 natural, 2 cinematic (the model's own profiles)
    float intensity = 1.0f;      ///< overall strength of the model's edit
    float local_structure = 1.0f; ///< fine detail / surface structure
    float local_tone = 1.0f;     ///< local light and shade
    bool skin_follows = true;    ///< skin uses the local-structure value (the model's default)
    float skin_structure = 1.0f; ///< used when skin_follows is off
    bool auto_mask = true;       ///< the model decides where it may edit
    int preset = 0;              ///< 0 = the model's default
    bool ui_correction = false;  ///< off: the frame it gets has no HUD in it
    int mv_mode = 0;             ///< motion vectors: 0 pixels (default), 1 normalised, 2 none
};

enum class State : int {
    Off,      ///< SHADPS4_DLSS != 1 or SHADPS4_DLSS5=0
    NotFound, ///< no folder with both DLLs
    Starting, ///< found; waiting for the Vulkan device
    Ready,
    Failed, ///< start-up failed (see message); DLSS 4.5 runs alone
};

struct Status {
    State state = State::Off;
    std::string message;
    std::string folder;
    float gpu_ms_avg = 0.0f; ///< the model's GPU time per frame, last ~2 s
    float gpu_ms_max = 0.0f;
    u64 frames = 0; ///< frames the model ran on
    u32 width = 0;
    u32 height = 0;
    bool running = false; ///< ran on the last DLSS frame
};

/// SHADPS4_DLSS=1, SHADPS4_DLSS5 not "0" and both DLLs found (decided once).
bool Wanted();
/// Device extensions the model uses (Streamline's DLSS list normally has them already).
const std::vector<const char*>& DeviceExtensions();
/// vk_instance, after the device exists: loads the forwarder and the model, checks the parameter
/// block layout, NVSDK_NGX_VULKAN_Init_Ext. Any failure: State::Failed, DLSS 4.5 alone.
void DeviceCreated(vk::Instance instance, vk::PhysicalDevice physical_device, vk::Device device);
/// The device came up without the needed extensions (or the DLSS ones were refused).
void Disable(const char* why);
/// Instance destruction, GPU idle, before Streamline goes.
void Shutdown();

Status GetStatus();
Settings GetSettings();
/// Any thread (the panel): takes effect on the next frame and is saved to dlss5-settings.ini.
void SetSettings(const Settings& settings);
Settings DefaultSettings();
/// End key: returns the new on / off state.
bool ToggleEnabled();

/// Rasterizer::DlssResolve, before Dlss::Evaluate, on the same command buffer: copies `color`
/// (the game's frame, in `color.layout`, left in that layout) into the model's input, runs the
/// model, and returns the image DLSS 4.5 should take as its colour input. Nothing = DLSS 5 off or
/// it failed this frame (DLSS 4.5 takes `color` as before).
std::optional<Dlss::EvalImage> Apply(const Instance& instance, vk::CommandBuffer cmdbuf,
                                     const Dlss::EvalImage& color, const Dlss::EvalImage& motion,
                                     const Dlss::EvalImage& depth, bool reset);


} // namespace Vulkan::Dlss5
