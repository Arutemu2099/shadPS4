// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
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

class Panel final : public ImGui::Layer {
public:
    void Draw() override;
    std::atomic<bool> closing{false}; ///< set by either thread; the layer removal is queued

private:
    void Commit() {
        D5::SetSettings(edit);
    }
    /// A slider that applies when it is let go (each apply rebuilds the model).
    void Slider(const char* label, float* value, const char* tip);
    void StyleButton(const char* label, int style);

    D5::Settings edit{};
};

Panel g_panel; // one object for the whole run: adds and removes are queued in order

void Panel::Slider(const char* label, float* value, const char* tip) {
    ImGui::SetNextItemWidth(-ImGui::CalcTextSize("Light and shade").x - 24.0f);
    ImGui::SliderFloat(label, value, 0.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp);
    if (ImGui::IsItemDeactivatedAfterEdit()) {
        Commit();
    }
    ImGui::SetItemTooltip("%s", tip);
}

void Panel::StyleButton(const char* label, int style) {
    const bool selected = edit.style == style;
    if (selected) {
        ImGui::PushStyleColor(ImGuiCol_Button, Accent);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, Accent);
    }
    const float w = (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x * 2) / 3.0f;
    if (ImGui::Button(label, ImVec2(style == 2 ? ImGui::GetContentRegionAvail().x : w, 0)) &&
        !selected) {
        edit.style = style;
        Commit();
    }
    if (selected) {
        ImGui::PopStyleColor(2);
    }
}

void Panel::Draw() {
    if (closing) {
        return;
    }
    const D5::Status status = D5::GetStatus();
    // Follow outside changes (End key) unless a slider is being dragged.
    if (!ImGui::IsAnyItemActive()) {
        edit = D5::GetSettings();
    }

    const ImGuiViewport* vp = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(ImVec2(vp->WorkPos.x + 24.0f, vp->WorkPos.y + 110.0f),
                            ImGuiCond_FirstUseEver);
    // Fixed width, height follows the content (auto-resize alone would let the sliders, which
    // size themselves from the window, shrink it frame by frame).
    ImGui::SetNextWindowSizeConstraints(ImVec2(470.0f, 0.0f), ImVec2(470.0f, 100000.0f));
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

        // ---- status ----
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
            } else if (status.running) {
                colour = Green;
                line = "Running on the game's " + std::to_string(status.width) + "x" +
                       std::to_string(status.height) + " frame";
            } else {
                colour = status.message == "Running" || status.message == "Ready" ? Yellow : Red;
                line = status.message == "Running" || status.message == "Ready"
                           ? "Waiting for DLSS (Home turns DLSS on)."
                           : status.message;
            }
            break;
        }
        ImGui::PushStyleColor(ImGuiCol_Text, colour);
        ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + 430.0f);
        ImGui::TextWrapped("%s", line.c_str());
        ImGui::PopTextWrapPos();
        ImGui::PopStyleColor();
        if (status.state == D5::State::Ready && status.running && status.gpu_ms_avg > 0.0f) {
            ImGui::Text("GPU cost: %.2f ms per frame (peak %.2f)", status.gpu_ms_avg,
                        status.gpu_ms_max);
            ImGui::SetItemTooltip("Time the model takes on the graphics card each frame.\n"
                                  "At 120 FPS a whole frame has 8.3 ms.");
        }

        const bool usable = status.state == D5::State::Ready;
        ImGui::BeginDisabled(!usable);

        ImGui::Spacing();
        if (ImGui::Checkbox("DLSS 5 on  (End)", &edit.enabled)) {
            Commit();
        }

        // ---- look ----
        ImGui::SeparatorText("Look");
        StyleButton("Standard", 0);
        ImGui::SameLine();
        StyleButton("Natural", 1);
        ImGui::SameLine();
        StyleButton("Cinematic", 2);
        ImGui::SetItemTooltip("The model's own looks: Standard is its default, Natural keeps "
                              "closer to the game,\nCinematic pushes light and contrast further.");

        // ---- strength ----
        ImGui::SeparatorText("Strength");
        Slider("Intensity", &edit.intensity,
               "How much of the model's edit reaches the picture overall.");
        Slider("Detail", &edit.local_structure,
               "Fine surface detail the model adds (skin pores, fabric, stone).");
        Slider("Light and shade", &edit.local_tone,
               "Local lighting changes: contact shadows, highlights, depth of shading.");
        if (ImGui::Checkbox("Skin follows Detail", &edit.skin_follows)) {
            Commit();
        }
        ImGui::SetItemTooltip("On: faces get the same detail strength as everything else (the "
                              "model's default).\nOff: set skin on its own below.");
        if (!edit.skin_follows) {
            Slider("Skin", &edit.skin_structure, "Detail strength on skin and faces only.");
        }

        // ---- advanced ----
        if (ImGui::CollapsingHeader("Advanced")) {
            if (ImGui::Checkbox("Auto mask", &edit.auto_mask)) {
                Commit();
            }
            ImGui::SetItemTooltip("Let the model decide which parts of the picture it may edit "
                                  "(default on).");
            ImGui::SetNextItemWidth(160.0f);
            ImGui::SliderInt("Preset", &edit.preset, 0, 15, edit.preset == 0 ? "default" : "%d",
                             ImGuiSliderFlags_AlwaysClamp);
            if (ImGui::IsItemDeactivatedAfterEdit()) {
                Commit();
            }
            ImGui::SetItemTooltip("The model's internal preset. 0 = its default. A value it does "
                                  "not know is refused (the status line says so).");
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
        }

        ImGui::Spacing();
        if (ImGui::Button("Defaults")) {
            const bool on = edit.enabled;
            edit = D5::DefaultSettings();
            edit.enabled = on;
            Commit();
        }
        ImGui::EndDisabled();
        ImGui::SameLine();
        if (ImGui::Button("Close  (Page Up)")) {
            keep_open = false;
        }

        ImGui::Spacing();
        ImGui::PushStyleColor(ImGuiCol_Text, Grey);
        ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + 430.0f);
        ImGui::TextWrapped("Order: game frame -> DLSS 5 -> DLSS 4.5 -> HUD, so menus and text "
                           "are never touched. Each change rebuilds the model (a short hitch). "
                           "Home = DLSS 4.5 on / off (DLSS 5 needs it).");
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
        g_open.store(false, std::memory_order_release);
        ImGui::Layer::RemoveLayer(this);
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
