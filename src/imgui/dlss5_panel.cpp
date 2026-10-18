// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <cstdio>
#include <string>
#include <imgui.h>

#include "imgui/dlss5_panel.h"
#include "imgui/imgui_layer.h"
#include "video_core/renderer_vulkan/vk_dlss5.h"

namespace Dlss5Panel {

namespace {

namespace D5 = Vulkan::Dlss5;

std::atomic<bool> g_open{false};

const ImVec4 Green{0.45f, 0.85f, 0.45f, 1.0f};
const ImVec4 Yellow{0.95f, 0.80f, 0.35f, 1.0f};
const ImVec4 Red{0.95f, 0.45f, 0.40f, 1.0f};
const ImVec4 Grey{0.65f, 0.65f, 0.68f, 1.0f};
const ImVec4 Accent{0.26f, 0.59f, 0.98f, 1.0f};
constexpr float LabelWidth = 150.0f; ///< room left of each slider for its name

class Panel final : public ImGui::Layer {
public:
    void Draw() override;
    std::atomic<bool> closing{false}; ///< set by either thread; the layer removal is queued

private:
    /// Saved; a model setting rebuilds the model.
    void Commit() {
        D5::SetSettings(edit);
    }
    /// A model slider: applies when let go (each apply rebuilds the model, a short hitch).
    void ModelSlider(const char* label, float* value, const char* tip);
    /// A blend slider: live while dragged (no rebuild), saved when let go.
    void LiveSlider(const char* label, float* value, float min, float max, const char* format,
                    const char* tip);
    /// One of a row of `count` buttons choosing `*value`.
    void Choice(const char* label, int* value, int choice, int index, int count, const char* tip);
    void DrawStatus(const D5::Status& status);

    D5::Settings edit{};
};

Panel g_panel; // one object for the whole run: adds and removes are queued in order

void Panel::ModelSlider(const char* label, float* value, const char* tip) {
    ImGui::SetNextItemWidth(-LabelWidth);
    ImGui::SliderFloat(label, value, 0.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp);
    if (ImGui::IsItemDeactivatedAfterEdit()) {
        Commit();
    }
    ImGui::SetItemTooltip("%s", tip);
}

void Panel::LiveSlider(const char* label, float* value, float min, float max, const char* format,
                       const char* tip) {
    ImGui::SetNextItemWidth(-LabelWidth);
    if (ImGui::SliderFloat(label, value, min, max, format, ImGuiSliderFlags_AlwaysClamp)) {
        D5::SetSettings(edit, false, false);
    }
    if (ImGui::IsItemDeactivatedAfterEdit()) {
        D5::SetSettings(edit);
    }
    ImGui::SetItemTooltip("%s", tip);
}

void Panel::Choice(const char* label, int* value, int choice, int index, int count,
                   const char* tip) {
    if (index > 0) {
        ImGui::SameLine();
    }
    const bool selected = *value == choice;
    if (selected) {
        ImGui::PushStyleColor(ImGuiCol_Button, Accent);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, Accent);
    }
    const float spacing = ImGui::GetStyle().ItemSpacing.x;
    const float w = index == count - 1
                        ? ImGui::GetContentRegionAvail().x
                        : (ImGui::GetContentRegionAvail().x - spacing * (count - 1 - index)) /
                              static_cast<float>(count - index);
    if (ImGui::Button(label, ImVec2(w, 0)) && !selected) {
        *value = choice;
        Commit();
    }
    if (selected) {
        ImGui::PopStyleColor(2);
    }
    ImGui::SetItemTooltip("%s", tip);
}

void Panel::DrawStatus(const D5::Status& status) {
    ImVec4 colour = Grey;
    std::string line;
    switch (status.state) {
    case D5::State::Off:
        line = "Needs DLSS (SHADPS4_DLSS=1).";
        break;
    case D5::State::NotFound:
        line = status.message;
        break;
    case D5::State::Starting:
        colour = Yellow;
        line = "Starting...";
        break;
    case D5::State::Failed:
        colour = Red;
        line = status.message;
        break;
    case D5::State::Ready:
        if (!edit.enabled) {
            colour = Yellow;
            line = "Off - press End or tick the box below.";
        } else if (status.video_running) {
            colour = Green;
            line = "Running on a video cutscene";
        } else if (status.running) {
            colour = Green;
            line = "Running on the game's " + std::to_string(status.width) + "x" +
                   std::to_string(status.height) + " frame" +
                   (status.passes_running > 1
                        ? ", " + std::to_string(status.passes_running) + " passes"
                        : std::string{});
        } else if (status.message == "Running" || status.message == "Ready") {
            colour = Yellow;
            line = "Waiting for DLSS (Home turns DLSS on).";
        } else {
            colour = Red;
            line = status.message;
        }
        break;
    }
    ImGui::PushStyleColor(ImGuiCol_Text, colour);
    ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + 470.0f);
    ImGui::TextWrapped("%s", line.c_str());
    ImGui::PopTextWrapPos();
    ImGui::PopStyleColor();
    if (status.state == D5::State::Ready && status.running && status.gpu_ms_avg > 0.0f) {
        ImGui::Text("GPU cost: %.2f ms per frame (peak %.2f)", status.gpu_ms_avg,
                    status.gpu_ms_max);
        ImGui::SetItemTooltip("Time DLSS 5 takes on the graphics card each game frame, all passes "
                              "included.\nAt 120 FPS a whole frame has 8.3 ms.");
    }
    if (status.state == D5::State::Ready && status.vram_budget_mb > 0) {
        const bool over = status.vram_used_mb > status.vram_budget_mb;
        if (over) {
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.45f, 0.35f, 1.0f));
        }
        ImGui::Text("Graphics memory: %.1f / %.1f GB (%u model copies)",
                    status.vram_used_mb / 1024.0f, status.vram_budget_mb / 1024.0f,
                    status.features_alive);
        if (over) {
            ImGui::PopStyleColor();
        }
        ImGui::SetItemTooltip("The whole game's use of the graphics card's memory. Over the limit "
                              "(red), the game stutters.\nEach model pass is one copy; a changed "
                              "setting builds new copies and frees the old ones a moment later.");
    }
}

void Panel::Draw() {
    if (closing) {
        return;
    }
    const D5::Status status = D5::GetStatus();
    // Follow outside changes (End key) unless a control is being used.
    if (!ImGui::IsAnyItemActive()) {
        edit = D5::GetSettings();
    }

    const ImGuiViewport* vp = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(ImVec2(vp->WorkPos.x + 24.0f, vp->WorkPos.y + 90.0f),
                            ImGuiCond_FirstUseEver);
    // Fixed width, height follows the content (auto-resize alone would let the sliders, which
    // size themselves from the window, shrink it frame by frame).
    ImGui::SetNextWindowSizeConstraints(ImVec2(510.0f, 0.0f), ImVec2(510.0f, 100000.0f));
    ImGui::SetNextWindowBgAlpha(0.88f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 8.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 4.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(16.0f, 14.0f));
    ImGui::PushStyleColor(ImGuiCol_SliderGrab, Accent);
    ImGui::PushStyleColor(ImGuiCol_SliderGrabActive, Accent);
    bool keep_open = true;
    if (ImGui::Begin("DLSS 5 - Neural Rendering###dlss5panel", &keep_open,
                     ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_AlwaysAutoResize |
                         ImGuiWindowFlags_NoSavedSettings)) {
        ImGui::SetWindowFontScale(1.15f);
        DrawStatus(status);

        const bool usable = status.state == D5::State::Ready;
        ImGui::BeginDisabled(!usable);
        ImGui::Spacing();
        if (ImGui::Checkbox("DLSS 5 on  (End)", &edit.enabled)) {
            Commit();
        }
        ImGui::SameLine(250.0f);
        if (ImGui::Checkbox("Video cutscenes too (experimental)", &edit.video)) {
            Commit();
        }
        ImGui::SetItemTooltip("Most story cutscenes are videos with no depth or motion data, so "
                              "DLSS 5 can only guess:\non them it mostly acts like colour "
                              "correction. Off by default. Subtitles are part of\nthe video, so the "
                              "model sees them too.
        if (status.video_frames > 0) {
            ImGui::TextDisabled("Video frames so far: %llu",
                                static_cast<unsigned long long>(status.video_frames));
        }

        // ---- the model: what it is built with (each change rebuilds it) ----
        ImGui::SeparatorText("Model  (a change rebuilds it - short hitch)");
        ImGui::TextUnformatted("Network");
        Choice("Default", &edit.preset, 0, 0, 3,
               "The DLL's shipping default (in the newer file most likely B).");
        Choice("A: first model", &edit.preset, 1, 1, 3,
               "Preset A: the first release's network (NBA 2K27). Reported to add more\n"
               "screen-space lighting; measured here about 5.5 ms per frame.");
        Choice("B: newer model", &edit.preset, 2, 2, 3,
               "Preset B: the newer network - a subtler look; measured here about 3.4 ms.");
        ImGui::TextUnformatted("Look");
        Choice("Standard", &edit.style, 0, 0, 3, "The model's default look.");
        Choice("Natural", &edit.style, 1, 1, 3, "Keeps closer to the game.");
        Choice("Cinematic", &edit.style, 2, 2, 3, "Pushes light and contrast further.");
        ModelSlider("Intensity", &edit.intensity,
                    "How strongly the model edits overall (read when it is built).");
        ModelSlider("Detail", &edit.local_structure,
                    "Fine surface detail the model adds (skin pores, fabric, stone).");
        ModelSlider("Light and shade", &edit.local_tone,
                    "Local lighting changes: contact shadows, highlights, depth of shading.");
        if (ImGui::Checkbox("Skin follows Detail", &edit.skin_follows)) {
            Commit();
        }
        ImGui::SetItemTooltip("On: faces get the same detail strength as everything else (the "
                              "model's default).\nOff: set skin on its own below.");
        if (!edit.skin_follows) {
            ModelSlider("Skin", &edit.skin_structure, "Detail strength on skin and faces only.");
        }

        // ---- the blend: how its answer goes onto the game's frame (live) ----
        ImGui::SeparatorText("Blend  (live)");
        LiveSlider("Light change", &edit.transfer, 0.0f, 1.5f, "%.2f",
                   "How much of the model's light and detail reaches the picture.\n0 = the "
                   "game's frame, 1 = all of it, above 1 = exaggerated.");
        LiveSlider("Colour", &edit.colour, 0.0f, 1.0f, "%.2f",
                   "How much of the model's colour change reaches the picture.\nLower it if skin "
                   "or a scene shifts colour (grey or washed-out skin).");
        LiveSlider("Saturation", &edit.saturation, 0.5f, 1.5f, "%.2f",
                   "Colour intensity of the result. Above 1 if it looks washed out.");
        LiveSlider("Max light change", &edit.max_ratio, 1.1f, 8.0f,
                   edit.max_ratio >= 7.99f ? "off" : "%.1fx",
                   "The most any pixel may get brighter or darker (x or 1/x).\nKeeps the model "
                   "from restyling a light source or crushing a shadow.");

        // ---- layers ----
        ImGui::SeparatorText("Layers");
        ImGui::SetNextItemWidth(-LabelWidth);
        ImGui::SliderInt("Model passes", &edit.passes, 1, 3, "%d", ImGuiSliderFlags_AlwaysClamp);
        if (ImGui::IsItemDeactivatedAfterEdit()) {
            Commit();
        }
        ImGui::SetItemTooltip("Runs the model again on its own answer (each pass its own model),\n"
                              "composed once onto the game's frame: detail builds up, colour does "
                              "not.\nEach pass costs a full model run (see GPU cost). 1 = normal.");

        // ---- compare ----
        ImGui::SeparatorText("Compare");
        if (ImGui::Checkbox("Split screen (left = game)", &edit.compare)) {
            Commit();
        }
        if (edit.compare) {
            LiveSlider("Split", &edit.split, 0.0f, 1.0f, "%.2f",
                       "Where the line sits: left of it the game's frame, right of it DLSS 5.");
        }

        // ---- advanced ----
        if (ImGui::CollapsingHeader("Advanced")) {
            if (ImGui::Checkbox("Auto mask", &edit.auto_mask)) {
                Commit();
            }
            ImGui::SetItemTooltip("Let the model decide which parts of the picture it may edit "
                                  "(default on).");
            ImGui::SetNextItemWidth(160.0f);
            ImGui::SliderInt("Preset number", &edit.preset, 0, 15,
                             edit.preset == 0 ? "default" : "%d", ImGuiSliderFlags_AlwaysClamp);
            if (ImGui::IsItemDeactivatedAfterEdit()) {
                Commit();
            }
            ImGui::SetItemTooltip("The same choice as Network, by number (0 default, 1 A, 2 B, "
                                  "...). A number this DLL\ndoes not have falls back to its "
                                  "default network (the model's own rule).");
            if (ImGui::Checkbox("UI correction", &edit.ui_correction)) {
                Commit();
            }
            ImGui::SetItemTooltip("For games whose HUD is inside the picture. Here the HUD is "
                                  "added after DLSS, so leave it off.");
            static const char* const mv_names[] = {"Pixels (default)", "Normalised", "None"};
            ImGui::SetNextItemWidth(220.0f);
            if (ImGui::Combo("Motion vectors", &edit.mv_mode, mv_names, 3)) {
                Commit();
            }
            ImGui::SetItemTooltip("How the game's motion is passed to the model. Only for "
                                  "testing: if moving shots smear or flicker, try another.");
            ImGui::SetNextItemWidth(-LabelWidth);
            ImGui::SliderFloat("Video depth", &edit.video_depth, 0.0f, 1.0f, "%.2f",
                               ImGuiSliderFlags_AlwaysClamp);
            if (ImGui::IsItemDeactivatedAfterEdit()) {
                Commit();
            }
            ImGui::SetItemTooltip("Videos have no depth, so the model gets one flat value "
                                  "(0 = far ... 1 = near).\nTry it if video cutscenes change "
                                  "too little or too much.");
            if (ImGui::Checkbox("Video: fresh every frame", &edit.video_fresh)) {
                Commit();
            }
            ImGui::SetItemTooltip("Off (default): the model keeps its history through a video - "
                                  "steady, a moving shot may\nghost a little. On: every frame "
                                  "starts fresh - no ghosting, but details flicker.");
        }

        ImGui::Spacing();
        if (ImGui::Button("Defaults")) {
            const bool on = edit.enabled;
            edit = D5::DefaultSettings();
            edit.enabled = on;
            D5::SetSettings(edit, true); // also the way out after the model refused a setting
        }
        ImGui::SetItemTooltip("Every setting back to its default (DLSS 5 stays on / off).");
        ImGui::EndDisabled();
        ImGui::SameLine();
        if (ImGui::Button("Close  (Page Up)")) {
            keep_open = false;
        }

        ImGui::Spacing();
        ImGui::PushStyleColor(ImGuiCol_Text, Grey);
        ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + 470.0f);
        ImGui::TextWrapped("Order: game frame -> DLSS 5 -> DLSS 4.5 -> HUD, so menus and text are "
                           "never touched. Home = DLSS 4.5 on / off (DLSS 5 needs it).");
        if (!status.folder.empty()) {
            ImGui::TextWrapped("Files: %s", status.folder.c_str());
        }
        ImGui::PopTextWrapPos();
        ImGui::PopStyleColor();
    }
    ImGui::End();
    ImGui::PopStyleColor(2);
    ImGui::PopStyleVar(3);

    if (!keep_open && !closing.exchange(true)) {
        ImGui::Layer::RemoveLayer(this); // before IsOpen() turns false (Page Up race)
        g_open.store(false, std::memory_order_release);
    }
}

} // namespace

void Toggle() {
    if (g_open.exchange(false, std::memory_order_acq_rel)) {
        if (!g_panel.closing.exchange(true)) {
            ImGui::Layer::RemoveLayer(&g_panel);
        }
        return;
    }
    g_panel.closing.store(false);
    ImGui::Layer::AddLayer(&g_panel);
    g_open.store(true, std::memory_order_release);
}

bool IsOpen() {
    return g_open.load(std::memory_order_acquire);
}

} // namespace Dlss5Panel
