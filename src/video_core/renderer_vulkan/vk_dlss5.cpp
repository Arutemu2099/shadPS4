// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// DLSS 5 Neural Rendering (vk_dlss5.h). Facts this rests on, read from the fork's source
// (github.com/Skynizz/optiscaler-dlss5 6bfc4389, OptiScaler/dlssnr; LATER-PLAN 9, 2026-10-18):
// - the model (NGX feature 18) has a native Vulkan surface: NVSDK_NGX_VULKAN_Init_Ext /
//   CreateFeature / EvaluateFeature / ReleaseFeature, reached through the forwarder's
//   dlssnr_vk_probe / _init / _create / _evaluate / _release;
// - inputs colour, depth, motion vectors, one output; the tuning is read at create;
// - the device needs VK_NVX_binary_import and VK_NVX_image_view_handle (plus push descriptors and
//   buffer device address, which the emulator enables anyway).

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>

#include <fmt/format.h>

#include "common/logging/log.h"
#include "video_core/renderer_vulkan/vk_dlss5.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_platform.h"
#include "video_core/texture_cache/image.h"

namespace Vulkan::Dlss5 {

namespace {

constexpr const char* ForwarderName = "nvngx.dll_dlssnr.dll";
constexpr const char* ModelName = "nvngx_dlssnr.dll";
constexpr const char* SettingsName = "dlss5-settings.ini";
constexpr int NgxSuccess = 1;
constexpr int NgxFail = static_cast<int>(0xBAD00000u);
constexpr int NgxSdkVersion = 0x15;
constexpr int FeatureId = 18;
constexpr u64 RetireAfter = 64; ///< frames before a replaced feature / image is freed
constexpr u32 TimingSlots = 8;

const std::vector<const char*> g_extensions = {"VK_NVX_binary_import",
                                               "VK_NVX_image_view_handle"};

// ---- the NGX parameter block -------------------------------------------------------------

// The NGX parameter block as the model sees it. The SDK declares eight Set overloads, eight Get
// overloads and Reset (nvsdk_ngx_params.h; order ULL, float, double, uint, int, ID3D11Resource*,
// ID3D12Resource*, void*), and the model, built with MSVC, calls them through the vtable. MSVC
// groups overloads in REVERSE declaration order, so the slots are the ones below - the fork
// measured exactly this on the driver's own block (resource setter 0 / getter 8, uint written
// through 3 = Set(int), float setter 6). Distinct names give this order on any compiler; SelfTest()
// checks the slots the forwarder writes through before anything uses the block. No virtual
// destructor: it would add slots the model does not expect.
class NgxParameters final {
public:
    enum class Kind : u8 { U64, F32, F64, U32, I32, Ptr };
    struct Value {
        Kind kind;
        union {
            u64 u64v;
            float f32;
            double f64;
            u32 u32v;
            s32 i32;
            void* ptr;
        };
    };

    // Slots 0-7: Set(void*), Set(ID3D12Resource*), Set(ID3D11Resource*), Set(int),
    // Set(unsigned int), Set(double), Set(float), Set(unsigned long long).
    virtual void SetPtr(const char* name, void* v) {
        Value x{};
        x.kind = Kind::Ptr;
        x.ptr = v;
        Put(name, x);
    }
    virtual void SetD3D12(const char* name, void* v) {
        SetPtr(name, v);
    }
    virtual void SetD3D11(const char* name, void* v) {
        SetPtr(name, v);
    }
    virtual void SetI32(const char* name, int v) {
        Value x{};
        x.kind = Kind::I32;
        x.i32 = v;
        Put(name, x);
    }
    virtual void SetU32(const char* name, unsigned int v) {
        Value x{};
        x.kind = Kind::U32;
        x.u32v = v;
        Put(name, x);
    }
    virtual void SetF64(const char* name, double v) {
        Value x{};
        x.kind = Kind::F64;
        x.f64 = v;
        Put(name, x);
    }
    virtual void SetF32(const char* name, float v) {
        Value x{};
        x.kind = Kind::F32;
        x.f32 = v;
        Put(name, x);
    }
    virtual void SetU64(const char* name, unsigned long long v) {
        Value x{};
        x.kind = Kind::U64;
        x.u64v = v;
        Put(name, x);
    }

    // Slots 8-15: the Gets in the same order; 16: Reset.
    virtual int GetPtr(const char* name, void** out) const {
        return Pointer(name, out);
    }
    virtual int GetD3D12(const char* name, void** out) const {
        return Pointer(name, out);
    }
    virtual int GetD3D11(const char* name, void** out) const {
        return Pointer(name, out);
    }
    virtual int GetI32(const char* name, int* out) const {
        return Number(name, out);
    }
    virtual int GetU32(const char* name, unsigned int* out) const {
        return Number(name, out);
    }
    virtual int GetF64(const char* name, double* out) const {
        return Number(name, out);
    }
    virtual int GetF32(const char* name, float* out) const {
        return Number(name, out);
    }
    virtual int GetU64(const char* name, unsigned long long* out) const {
        return Number(name, out);
    }
    virtual void Reset() {
        values.clear();
    }

    const Value* Find(const char* name) const {
        if (!name) {
            return nullptr;
        }
        const auto it = values.find(name);
        return it == values.end() ? nullptr : &it->second;
    }
    void Clear() {
        values.clear();
    }

private:
    void Put(const char* name, const Value& v) {
        if (name) {
            values[name] = v;
        }
    }

    template <typename T>
    int Number(const char* name, T* out) const {
        const Value* v = Find(name);
        if (!v || !out) {
            NoteMissing(name);
            return NgxFail;
        }
        switch (v->kind) {
        case Kind::U64:
            *out = static_cast<T>(v->u64v);
            break;
        case Kind::F32:
            *out = static_cast<T>(v->f32);
            break;
        case Kind::F64:
            *out = static_cast<T>(v->f64);
            break;
        case Kind::U32:
            *out = static_cast<T>(v->u32v);
            break;
        case Kind::I32:
            *out = static_cast<T>(v->i32);
            break;
        case Kind::Ptr:
            if constexpr (std::is_integral_v<T> && sizeof(T) == 8) {
                *out = static_cast<T>(reinterpret_cast<uintptr_t>(v->ptr));
                break;
            }
            return NgxFail;
        }
        return NgxSuccess;
    }

    int Pointer(const char* name, void** out) const {
        const Value* v = Find(name);
        if (!v || !out) {
            NoteMissing(name);
            return NgxFail;
        }
        if (v->kind == Kind::Ptr) {
            *out = v->ptr;
        } else if (v->kind == Kind::U64) {
            *out = reinterpret_cast<void*>(static_cast<uintptr_t>(v->u64v));
        } else {
            return NgxFail;
        }
        return NgxSuccess;
    }

    // What the model asks for that nobody set: the first answer to "why did create fail".
    void NoteMissing(const char* name) const {
        if (!name || missing.size() >= 64 || !missing.insert(name).second) {
            return;
        }
        LOG_INFO(Render_Vulkan, "DLSS 5: the model asked for {} (not set)", name);
    }

    std::unordered_map<std::string, Value> values;
    mutable std::unordered_set<std::string> missing;
};

// ---- the NGX Vulkan resource (nvsdk_ngx_vk.h layout) --------------------------------------

struct NgxImageViewInfoVk {
    VkImageView view;
    VkImage image;
    VkImageSubresourceRange range;
    VkFormat format;
    unsigned int width;
    unsigned int height;
};
struct NgxBufferInfoVk {
    VkBuffer buffer;
    unsigned int size;
};
struct NgxResourceVk {
    union {
        NgxImageViewInfoVk image;
        NgxBufferInfoVk buffer;
    } resource;
    int type; ///< 0 = image view
    bool read_write;
};
static_assert(sizeof(NgxImageViewInfoVk) == 48 && sizeof(NgxResourceVk) == 56);

NgxResourceVk MakeResource(const Dlss::EvalImage& img, bool read_write) {
    NgxResourceVk r{};
    r.resource.image.view = static_cast<VkImageView>(img.view);
    r.resource.image.image = static_cast<VkImage>(img.image);
    r.resource.image.range = VkImageSubresourceRange{
        .aspectMask = img.depth ? VkImageAspectFlags{VK_IMAGE_ASPECT_DEPTH_BIT}
                                : VkImageAspectFlags{VK_IMAGE_ASPECT_COLOR_BIT},
        .baseMipLevel = 0,
        .levelCount = 1,
        .baseArrayLayer = 0,
        .layerCount = 1,
    };
    r.resource.image.format = static_cast<VkFormat>(img.format);
    r.resource.image.width = img.width;
    r.resource.image.height = img.height;
    r.type = 0;
    r.read_write = read_write;
    return r;
}

// ---- the forwarder ------------------------------------------------------------------------

using PFN_Probe = int (*)(const wchar_t*);
using PFN_Init = int (*)(const wchar_t*, const wchar_t*, void*, void*, void*, int);
using PFN_Create = void* (*)(void*, void*, unsigned int, unsigned int, int, float, int, float, float,
                             float, int, int);
using PFN_Evaluate = int (*)(void*, void*, void*, void*, void*, void*, void*, unsigned int,
                             unsigned int, unsigned int, unsigned int, int, int, float, int, float,
                             float, float, int, float, float);
using PFN_Release = void (*)(void*);
using PFN_SetFloatSlot = void (*)(int);
using PFN_ProbeFloat = void (*)(void*, const char*, float, int);

struct Forwarder {
#ifdef _WIN32
    HMODULE module{};
#endif
    PFN_Probe probe{};
    PFN_Init init{};
    PFN_Create create{};
    PFN_Evaluate evaluate{};
    PFN_Release release{};
    PFN_SetFloatSlot set_float_slot{};
    PFN_ProbeFloat probe_float{};
    const int* last_init{};
    const int* last_create{};
};

struct OwnedImage {
    VideoCore::UniqueImage image;
    vk::UniqueImageView view;
};

constexpr vk::ImageUsageFlags OwnedUsage =
    vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled |
    vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eTransferSrc;
constexpr vk::Format OwnedFormat = vk::Format::eR16G16B16A16Sfloat;
constexpr vk::ImageSubresourceRange ColorRange{
    .aspectMask = vk::ImageAspectFlagBits::eColor,
    .levelCount = 1,
    .layerCount = 1,
};

// Everything below runs on the thread that records the DLSS resolve, except the settings and
// the status, which the panel reads and writes from the ImGui thread (under g_mutex).
struct Context {
    Forwarder fwd;
    NgxParameters params;
    std::filesystem::path dir;
    vk::Device device;
    float timestamp_period = 0.0f;

    void* feature = nullptr;
    u32 feature_gen = ~0u;
    u32 failed_gen = ~0u; ///< the model refused these settings; wait for a change
    u32 eval_failures = 0;
    bool was_off = true;
    u64 applies = 0;
    std::deque<std::pair<u64, void*>> retired_features;
    std::deque<std::pair<u64, std::unique_ptr<OwnedImage>>> retired_images;

    std::unique_ptr<OwnedImage> input;  ///< the game's frame, RGBA16F (sRGB-encoded values)
    std::unique_ptr<OwnedImage> output; ///< the model's answer, RGBA16F
    u32 width = 0;
    u32 height = 0;

    VkQueryPool query_pool = VK_NULL_HANDLE;
    bool slot_written[TimingSlots]{};
    float window_sum = 0.0f;
    float window_max = 0.0f;
    u32 window_count = 0;
};

Context g;
std::mutex g_mutex; ///< g_settings, g_status
Settings g_settings;
Status g_status;
std::atomic<u32> g_generation{1};
std::atomic<State> g_state{State::Off};

void SetStatus(State state, std::string message) {
    std::scoped_lock lock{g_mutex};
    g_status.state = state;
    g_status.message = std::move(message);
    g_state.store(state, std::memory_order_release);
}

void SetMessage(std::string message) {
    std::scoped_lock lock{g_mutex};
    g_status.message = std::move(message);
}

std::filesystem::path ExeDir() {
#ifdef _WIN32
    std::wstring buf(MAX_PATH, L'\0');
    for (;;) {
        const DWORD n = GetModuleFileNameW(nullptr, buf.data(), static_cast<DWORD>(buf.size()));
        if (n == 0) {
            return {};
        }
        if (n < buf.size()) {
            buf.resize(n);
            break;
        }
        buf.resize(buf.size() * 2);
    }
    return std::filesystem::path{buf}.parent_path();
#else
    return {};
#endif
}

bool HasBoth(const std::filesystem::path& dir) {
    std::error_code ec;
    return !dir.empty() && std::filesystem::is_regular_file(dir / ForwarderName, ec) &&
           std::filesystem::is_regular_file(dir / ModelName, ec);
}

// ---- settings file (key = value, one per line) ---------------------------------------------

void LoadSettingsFile() {
    std::ifstream in{g.dir / SettingsName};
    if (!in) {
        return;
    }
    Settings s = DefaultSettings();
    std::string line;
    while (std::getline(in, line)) {
        const auto eq = line.find('=');
        if (eq == std::string::npos || line.starts_with('#')) {
            continue;
        }
        std::string key = line.substr(0, eq);
        std::string value = line.substr(eq + 1);
        key.erase(std::remove(key.begin(), key.end(), ' '), key.end());
        value.erase(std::remove(value.begin(), value.end(), ' '), value.end());
        const float f = static_cast<float>(std::atof(value.c_str()));
        const int i = std::atoi(value.c_str());
        if (key == "enabled") {
            s.enabled = i != 0;
        } else if (key == "style") {
            s.style = std::clamp(i, 0, 2);
        } else if (key == "intensity") {
            s.intensity = std::clamp(f, 0.0f, 1.0f);
        } else if (key == "local_structure") {
            s.local_structure = std::clamp(f, 0.0f, 1.0f);
        } else if (key == "local_tone") {
            s.local_tone = std::clamp(f, 0.0f, 1.0f);
        } else if (key == "skin_follows") {
            s.skin_follows = i != 0;
        } else if (key == "skin_structure") {
            s.skin_structure = std::clamp(f, 0.0f, 1.0f);
        } else if (key == "auto_mask") {
            s.auto_mask = i != 0;
        } else if (key == "preset") {
            s.preset = std::clamp(i, 0, 15);
        } else if (key == "ui_correction") {
            s.ui_correction = i != 0;
        } else if (key == "mv_mode") {
            s.mv_mode = std::clamp(i, 0, 2);
        }
    }
    std::scoped_lock lock{g_mutex};
    g_settings = s;
}

void SaveSettingsFile(const Settings& s) {
    if (g.dir.empty()) {
        return;
    }
    std::ofstream out{g.dir / SettingsName, std::ios::trunc};
    if (!out) {
        return;
    }
    out << "# DLSS 5 panel settings (Page Up in game). Deleting this file restores the defaults.\n"
        << "enabled = " << (s.enabled ? 1 : 0) << "\n"
        << "style = " << s.style << "\n"
        << "intensity = " << s.intensity << "\n"
        << "local_structure = " << s.local_structure << "\n"
        << "local_tone = " << s.local_tone << "\n"
        << "skin_follows = " << (s.skin_follows ? 1 : 0) << "\n"
        << "skin_structure = " << s.skin_structure << "\n"
        << "auto_mask = " << (s.auto_mask ? 1 : 0) << "\n"
        << "preset = " << s.preset << "\n"
        << "ui_correction = " << (s.ui_correction ? 1 : 0) << "\n"
        << "mv_mode = " << s.mv_mode << "\n";
}

// ---- start-up -----------------------------------------------------------------------------

template <typename T>
bool Resolve(T& out, const char* name) {
#ifdef _WIN32
    out = reinterpret_cast<T>(GetProcAddress(g.fwd.module, name));
#endif
    if (!out) {
        LOG_ERROR(Render_Vulkan, "DLSS 5: the forwarder has no {}", name);
    }
    return out != nullptr;
}

// Proves the vtable slots the forwarder writes through land on the right setters of our block,
// and finds the float one (the forwarder's own block probe; see the NgxParameters comment).
bool SelfTest() {
    NgxParameters& p = g.params;
    p.Clear();
    void** vt = *reinterpret_cast<void***>(&p);
    using SetU64 = void (*)(void*, const char*, unsigned long long);
    using SetU32 = void (*)(void*, const char*, unsigned int);
    reinterpret_cast<SetU64>(vt[0])(&p, "selftest.ptr", 0x1122334455667788ull);
    reinterpret_cast<SetU32>(vt[3])(&p, "selftest.uint", 7u);
    const auto* ptr = p.Find("selftest.ptr");
    const auto* uint = p.Find("selftest.uint");
    const bool ptr_ok = ptr && ptr->kind == NgxParameters::Kind::Ptr &&
                        reinterpret_cast<uintptr_t>(ptr->ptr) == 0x1122334455667788ull;
    const bool uint_ok = uint && uint->kind == NgxParameters::Kind::I32 && uint->i32 == 7;
    int float_slot = -1;
    for (int slot = 0; slot < 8 && float_slot < 0; ++slot) {
        char name[32];
        std::snprintf(name, sizeof(name), "selftest.float%d", slot);
        g.fwd.probe_float(&p, name, 1.25f, slot);
        const auto* f = p.Find(name);
        if (f && f->kind == NgxParameters::Kind::F32 && f->f32 == 1.25f) {
            float_slot = slot;
        }
    }
    p.Clear();
    LOG_INFO(Render_Vulkan, "DLSS 5: parameter block slots: 0 {}, 3 {}, float {}",
             ptr_ok ? "pointer (ok)" : "WRONG", uint_ok ? "int (ok)" : "WRONG", float_slot);
    // 6 is where MSVC puts Set(float) in this declaration; anything else means the model would
    // call the wrong getters too.
    if (!ptr_ok || !uint_ok || float_slot != 6) {
        return false;
    }
    g.fwd.set_float_slot(float_slot);
    return true;
}

// ---- per frame helpers --------------------------------------------------------------------

std::unique_ptr<OwnedImage> MakeImage(const Instance& instance, u32 width, u32 height,
                                      const char* name) {
    auto t = std::make_unique<OwnedImage>();
    t->image = VideoCore::UniqueImage(g.device, instance.GetAllocator());
    t->image.Create(vk::ImageCreateInfo{
        .imageType = vk::ImageType::e2D,
        .format = OwnedFormat,
        .extent{width, height, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .usage = OwnedUsage,
        .initialLayout = vk::ImageLayout::eUndefined,
    });
    SetObjectName(g.device, static_cast<vk::Image>(t->image), "{}", name);
    t->view = Check<"create dlss5 view">(g.device.createImageViewUnique({
        .image = t->image,
        .viewType = vk::ImageViewType::e2D,
        .format = OwnedFormat,
        .subresourceRange = ColorRange,
    }));
    return t;
}

Dlss::EvalImage AsEval(const OwnedImage& img, u32 width, u32 height) {
    return Dlss::EvalImage{
        .image = img.image,
        .view = img.view.get(),
        .format = OwnedFormat,
        .layout = vk::ImageLayout::eGeneral,
        .usage = OwnedUsage,
        .width = width,
        .height = height,
        .depth = false,
    };
}

void RetireFeature() {
    if (g.feature) {
        g.retired_features.emplace_back(g.applies, g.feature);
        g.feature = nullptr;
    }
}

void FreeRetired(bool all) {
    while (!g.retired_features.empty() &&
           (all || g.retired_features.front().first + RetireAfter <= g.applies)) {
        g.fwd.release(g.retired_features.front().second);
        g.retired_features.pop_front();
    }
    while (!g.retired_images.empty() &&
           (all || g.retired_images.front().first + RetireAfter <= g.applies)) {
        g.retired_images.pop_front();
    }
}

void ReadTiming() {
    if (g.query_pool == VK_NULL_HANDLE) {
        return;
    }
    // The oldest slot: written TimingSlots - 1 frames ago, long executed.
    const u32 slot = static_cast<u32>((g.applies + 1) % TimingSlots);
    if (!g.slot_written[slot]) {
        return;
    }
    u64 data[4]{};
    const VkResult r = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetQueryPoolResults(
        static_cast<VkDevice>(g.device), g.query_pool, slot * 2, 2, sizeof(data), data,
        sizeof(u64) * 2, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
    if ((r != VK_SUCCESS && r != VK_NOT_READY) || data[1] == 0 || data[3] == 0) {
        return;
    }
    g.slot_written[slot] = false;
    const float ms =
        static_cast<float>(data[2] - data[0]) * g.timestamp_period / 1'000'000.0f;
    g.window_sum += ms;
    g.window_max = std::max(g.window_max, ms);
    if (++g.window_count >= 120) {
        const float avg = g.window_sum / static_cast<float>(g.window_count);
        {
            std::scoped_lock lock{g_mutex};
            g_status.gpu_ms_avg = avg;
            g_status.gpu_ms_max = g.window_max;
        }
        if ((g.applies / 120) % 5 == 0) {
            LOG_INFO(Render_Vulkan, "DLSS 5: model GPU time {:.2f} ms avg, {:.2f} ms max ({} frames)",
                     avg, g.window_max, g.applies);
        }
        g.window_sum = 0.0f;
        g.window_max = 0.0f;
        g.window_count = 0;
    }
}

void SetRunning(bool running) {
    std::scoped_lock lock{g_mutex};
    g_status.running = running;
    g_status.frames = g.applies;
    g_status.width = g.width;
    g_status.height = g.height;
}

} // namespace

Settings DefaultSettings() {
    return Settings{};
}

bool Wanted() {
    static const bool wanted = [] {
        if (Dlss::Mode() != 1) {
            return false;
        }
        const char* env = std::getenv("SHADPS4_DLSS5");
        if (env && env[0] == '0') {
            LOG_INFO(Render_Vulkan, "DLSS 5: off (SHADPS4_DLSS5=0)");
            return false;
        }
        std::vector<std::filesystem::path> candidates;
        if (const char* dir = std::getenv("SHADPS4_DLSS5_DIR"); dir && dir[0]) {
            candidates.emplace_back(dir);
        }
        const auto exe = ExeDir();
        candidates.push_back(exe / "dlss5");
        candidates.push_back(exe / ".." / ".." / "dlss5");
        for (const auto& c : candidates) {
            if (HasBoth(c)) {
                std::error_code ec;
                g.dir = std::filesystem::weakly_canonical(c, ec);
                if (ec) {
                    g.dir = c;
                }
                break;
            }
        }
        if (g.dir.empty()) {
            LOG_INFO(Render_Vulkan, "DLSS 5: not found - DLSS 4.5 alone (put {} and {} in {})",
                     ModelName, ForwarderName, (exe / "dlss5").string());
            SetStatus(State::NotFound, "Not found: put nvngx_dlssnr.dll and nvngx.dll_dlssnr.dll "
                                       "in the dlss5 folder next to shadPS4.exe.");
            return false;
        }
        {
            std::scoped_lock lock{g_mutex};
            g_status.folder = g.dir.string();
        }
        LoadSettingsFile();
        LOG_INFO(Render_Vulkan, "DLSS 5: found in {} (unofficial model - private use)",
                 g.dir.string());
        SetStatus(State::Starting, "Starting...");
        return true;
    }();
    return wanted;
}

const std::vector<const char*>& DeviceExtensions() {
    return g_extensions;
}

void Disable(const char* why) {
    if (Wanted() && g_state.load(std::memory_order_acquire) == State::Starting) {
        LOG_WARNING(Render_Vulkan, "DLSS 5: off - {}", why);
        SetStatus(State::Failed, std::string{"Off: "} + why);
    }
}

void DeviceCreated(vk::Instance instance, vk::PhysicalDevice physical_device, vk::Device device) {
#ifdef _WIN32
    if (!Wanted() || g_state.load(std::memory_order_acquire) != State::Starting) {
        return;
    }
    const auto fail = [](const std::string& why) {
        LOG_ERROR(Render_Vulkan, "DLSS 5: {} - DLSS 4.5 alone", why);
        SetStatus(State::Failed, why);
    };
    const std::filesystem::path forwarder = g.dir / ForwarderName;
    const std::filesystem::path model = g.dir / ModelName;
    g.fwd.module = LoadLibraryExW(forwarder.wstring().c_str(), nullptr,
                                  LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!g.fwd.module) {
        fail(fmt::format("could not load {} (error {})", forwarder.string(), GetLastError()));
        return;
    }
    bool ok = Resolve(g.fwd.probe, "dlssnr_vk_probe") && Resolve(g.fwd.init, "dlssnr_vk_init") &&
              Resolve(g.fwd.create, "dlssnr_vk_create") &&
              Resolve(g.fwd.evaluate, "dlssnr_vk_evaluate") &&
              Resolve(g.fwd.release, "dlssnr_vk_release") &&
              Resolve(g.fwd.set_float_slot, "dlssnr_call_set_float_slot") &&
              Resolve(g.fwd.probe_float, "dlssnr_call_probe_float") &&
              Resolve(g.fwd.last_init, "dlssnr_vk_last_init") &&
              Resolve(g.fwd.last_create, "dlssnr_vk_last_create");
    if (!ok) {
        fail("the forwarder is missing entry points (wrong build?)");
        return;
    }
    if (!SelfTest()) {
        fail("the parameter block layout check failed");
        return;
    }
    LOG_INFO(Render_Vulkan, "DLSS 5: loading the model ({})", model.string());
    const int probe = g.fwd.probe(model.wstring().c_str());
    if (probe != 15) {
        fail(fmt::format("the model's Vulkan entry points are incomplete (probe {})", probe));
        return;
    }
    std::error_code ec;
    const std::filesystem::path data = g.dir / "ngx-data";
    std::filesystem::create_directories(data, ec);
    const int init = g.fwd.init(model.wstring().c_str(), data.wstring().c_str(),
                                static_cast<VkInstance>(instance),
                                static_cast<VkPhysicalDevice>(physical_device),
                                static_cast<VkDevice>(device), NgxSdkVersion);
    if (init != 1) {
        fail(fmt::format("the model would not start on this GPU (init {:#x})",
                         static_cast<u32>(init)));
        return;
    }
    g.device = device;
    g.timestamp_period = physical_device.getProperties().limits.timestampPeriod;
    if (g.timestamp_period > 0.0f) {
        const VkQueryPoolCreateInfo info{
            .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            .queryType = VK_QUERY_TYPE_TIMESTAMP,
            .queryCount = TimingSlots * 2,
        };
        if (VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateQueryPool(static_cast<VkDevice>(device), &info,
                                                            nullptr, &g.query_pool) != VK_SUCCESS) {
            g.query_pool = VK_NULL_HANDLE;
        }
    }
    LOG_INFO(Render_Vulkan, "DLSS 5: ready (End = on / off, Page Up = panel)");
    SetStatus(State::Ready, "Ready");
#else
    (void)instance;
    (void)physical_device;
    (void)device;
#endif
}

void Shutdown() {
    if (g_state.load(std::memory_order_acquire) != State::Ready) {
        return;
    }
    RetireFeature();
    FreeRetired(true);
    g.input.reset();
    g.output.reset();
    if (g.query_pool != VK_NULL_HANDLE) {
        VULKAN_HPP_DEFAULT_DISPATCHER.vkDestroyQueryPool(static_cast<VkDevice>(g.device),
                                                         g.query_pool, nullptr);
        g.query_pool = VK_NULL_HANDLE;
    }
    SetStatus(State::Failed, "Shut down");
    LOG_INFO(Render_Vulkan, "DLSS 5: shut down after {} frames", g.applies);
}

Status GetStatus() {
    std::scoped_lock lock{g_mutex};
    return g_status;
}

Settings GetSettings() {
    std::scoped_lock lock{g_mutex};
    return g_settings;
}

void SetSettings(const Settings& settings) {
    bool tuning_changed = false;
    {
        std::scoped_lock lock{g_mutex};
        const Settings& o = g_settings;
        // Everything but on / off is read when the feature is built: a change rebuilds it.
        tuning_changed = o.style != settings.style || o.intensity != settings.intensity ||
                         o.local_structure != settings.local_structure ||
                         o.local_tone != settings.local_tone ||
                         o.skin_follows != settings.skin_follows ||
                         o.skin_structure != settings.skin_structure ||
                         o.auto_mask != settings.auto_mask || o.preset != settings.preset ||
                         o.ui_correction != settings.ui_correction || o.mv_mode != settings.mv_mode;
        g_settings = settings;
    }
    if (tuning_changed) {
        g_generation.fetch_add(1, std::memory_order_acq_rel);
    }
    SaveSettingsFile(settings);
}

bool ToggleEnabled() {
    Settings s = GetSettings();
    s.enabled = !s.enabled;
    SetSettings(s);
    return s.enabled;
}

std::optional<Dlss::EvalImage> Apply(const Instance& instance, vk::CommandBuffer cmdbuf,
                                     const Dlss::EvalImage& color, const Dlss::EvalImage& motion,
                                     const Dlss::EvalImage& depth, bool reset) {
    if (g_state.load(std::memory_order_acquire) != State::Ready) {
        return std::nullopt;
    }
    ++g.applies;
    FreeRetired(false);
    ReadTiming();

    const u32 gen = g_generation.load(std::memory_order_acquire);
    const Settings s = GetSettings();
    if (!s.enabled) {
        g.was_off = true;
        SetRunning(false);
        return std::nullopt;
    }
    if (gen == g.failed_gen) {
        SetRunning(false);
        return std::nullopt; // the model refused these settings; the panel shows why
    }

    const u32 w = color.width;
    const u32 h = color.height;
    if (!g.input || w != g.width || h != g.height) {
        RetireFeature();
        if (g.input) {
            g.retired_images.emplace_back(g.applies, std::move(g.input));
        }
        if (g.output) {
            g.retired_images.emplace_back(g.applies, std::move(g.output));
        }
        g.input = MakeImage(instance, w, h, "DLSS 5 input");
        g.output = MakeImage(instance, w, h, "DLSS 5 output");
        g.width = w;
        g.height = h;
        LOG_INFO(Render_Vulkan, "DLSS 5: images {}x{} (input format {})", w, h,
                 vk::to_string(color.format));
    }
    if (g.feature && g.feature_gen != gen) {
        RetireFeature();
    }
    const float skin = s.skin_follows ? -1.0f : s.skin_structure;
    if (!g.feature) {
        g.feature = g.fwd.create(static_cast<VkCommandBuffer>(cmdbuf), &g.params, w, h, s.preset,
                                 s.intensity, s.style, s.local_structure, s.local_tone, skin,
                                 s.auto_mask ? 1 : 0, s.ui_correction ? 1 : 0);
        if (!g.feature) {
            const int code = g.fwd.last_create ? *g.fwd.last_create : 0;
            LOG_ERROR(Render_Vulkan, "DLSS 5: the model would not build a feature ({:#x}) at {}x{}",
                      static_cast<u32>(code), w, h);
            g.failed_gen = gen;
            SetMessage(fmt::format("The model refused these settings (code {:#x}). Change a "
                                   "setting or press Defaults.",
                                   static_cast<u32>(code)));
            SetRunning(false);
            return std::nullopt;
        }
        g.feature_gen = gen;
        g.eval_failures = 0;
        reset = true;
        LOG_INFO(Render_Vulkan,
                 "DLSS 5: feature up at {}x{} (style {}, intensity {:.2f}, structure {:.2f}, "
                 "tone {:.2f}, skin {:.2f}, auto mask {}, preset {})",
                 w, h, s.style, s.intensity, s.local_structure, s.local_tone, skin, s.auto_mask,
                 s.preset);
        SetMessage("Running");
    }
    if (g.was_off) {
        reset = true;
        g.was_off = false;
    }

    // The game's frame -> the model's input (RGBA8 UNORM holding sRGB-encoded values -> RGBA16F,
    // same values); the game's image goes back to its layout.
    const vk::ImageSubresourceRange color_range = ColorRange;
    const std::array<vk::ImageMemoryBarrier2, 3> before{{
        {
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
            .oldLayout = color.layout,
            .newLayout = vk::ImageLayout::eTransferSrcOptimal,
            .image = color.image,
            .subresourceRange = color_range,
        },
        {
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eNone,
            .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .oldLayout = vk::ImageLayout::eUndefined, // fully overwritten
            .newLayout = vk::ImageLayout::eTransferDstOptimal,
            .image = g.input->image,
            .subresourceRange = ColorRange,
        },
        {
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eNone,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask = vk::AccessFlagBits2::eShaderWrite |
                             vk::AccessFlagBits2::eShaderRead |
                             vk::AccessFlagBits2::eTransferWrite,
            .oldLayout = vk::ImageLayout::eUndefined, // the model writes all of it
            .newLayout = vk::ImageLayout::eGeneral,
            .image = g.output->image,
            .subresourceRange = ColorRange,
        },
    }};
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .imageMemoryBarrierCount = static_cast<u32>(before.size()),
        .pImageMemoryBarriers = before.data(),
    });
    vk::ImageBlit blit{};
    blit.srcSubresource = vk::ImageSubresourceLayers{
        .aspectMask = vk::ImageAspectFlagBits::eColor, .mipLevel = 0, .baseArrayLayer = 0,
        .layerCount = 1};
    blit.srcOffsets[1] = vk::Offset3D{static_cast<s32>(w), static_cast<s32>(h), 1};
    blit.dstSubresource = blit.srcSubresource;
    blit.dstOffsets[1] = blit.srcOffsets[1];
    cmdbuf.blitImage(color.image, vk::ImageLayout::eTransferSrcOptimal, g.input->image,
                     vk::ImageLayout::eTransferDstOptimal, blit, vk::Filter::eNearest);
    const std::array<vk::ImageMemoryBarrier2, 2> after_copy{{
        {
            .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .srcAccessMask = vk::AccessFlagBits2::eNone,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
            .oldLayout = vk::ImageLayout::eTransferSrcOptimal,
            .newLayout = color.layout,
            .image = color.image,
            .subresourceRange = color_range,
        },
        {
            .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
            .oldLayout = vk::ImageLayout::eTransferDstOptimal,
            .newLayout = vk::ImageLayout::eGeneral,
            .image = g.input->image,
            .subresourceRange = ColorRange,
        },
    }};
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .imageMemoryBarrierCount = static_cast<u32>(after_copy.size()),
        .pImageMemoryBarriers = after_copy.data(),
    });

    const u32 slot = static_cast<u32>(g.applies % TimingSlots);
    if (g.query_pool != VK_NULL_HANDLE) {
        cmdbuf.resetQueryPool(vk::QueryPool{g.query_pool}, slot * 2, 2);
        cmdbuf.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, vk::QueryPool{g.query_pool},
                               slot * 2);
    }
    NgxResourceVk res_color = MakeResource(AsEval(*g.input, w, h), false);
    NgxResourceVk res_depth = MakeResource(depth, false);
    NgxResourceVk res_motion = MakeResource(motion, false);
    NgxResourceVk res_output = MakeResource(AsEval(*g.output, w, h), true);
    // The game's vectors are screen UV, current minus previous (vk_dlss.cpp): the model wants
    // the offset to the previous position in pixels -> -size (DLSS SR's NGX scale, study 10-14).
    const float mv_x = s.mv_mode == 0 ? -static_cast<float>(w) : s.mv_mode == 1 ? -1.0f : 0.0f;
    const float mv_y = s.mv_mode == 0 ? -static_cast<float>(h) : s.mv_mode == 1 ? -1.0f : 0.0f;
    const int result = g.fwd.evaluate(
        static_cast<VkCommandBuffer>(cmdbuf), g.feature, &g.params, &res_color, &res_depth,
        &res_motion, &res_output, w, h, depth.width, depth.height, /*depth inverted*/ 1,
        reset ? 1 : 0, s.intensity, s.style, s.local_structure, s.local_tone, skin,
        s.auto_mask ? 1 : 0, mv_x, mv_y);
    if (g.query_pool != VK_NULL_HANDLE) {
        cmdbuf.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, vk::QueryPool{g.query_pool},
                               slot * 2 + 1);
        g.slot_written[slot] = true;
    }
    if (result != 1) {
        if (++g.eval_failures <= 8) {
            LOG_ERROR(Render_Vulkan, "DLSS 5: evaluate failed ({:#x})", static_cast<u32>(result));
        }
        if (g.eval_failures >= 8) {
            RetireFeature();
            g.failed_gen = gen;
            SetMessage(fmt::format("The model stopped (evaluate {:#x}). Change a setting to retry.",
                                   static_cast<u32>(result)));
        }
        SetRunning(false);
        return std::nullopt;
    }
    g.eval_failures = 0;
    const vk::ImageMemoryBarrier2 written{
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
        .oldLayout = vk::ImageLayout::eGeneral,
        .newLayout = vk::ImageLayout::eGeneral,
        .image = g.output->image,
        .subresourceRange = ColorRange,
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &written,
    });
    if (g.applies == 1 || (g.applies % 1800) == 0) {
        LOG_INFO(Render_Vulkan, "DLSS 5: running ({} frames)", g.applies);
    }
    SetRunning(true);
    return AsEval(*g.output, w, h);
}

} // namespace Vulkan::Dlss5
