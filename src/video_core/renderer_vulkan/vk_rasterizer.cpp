// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <set>
#include <chrono>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/thread.h"
#include "common/debug.h"
#include "common/buffer_sample_probe.h"
#include "common/draw_sample_probe.h"
#include "common/fault_monitor.h"
#include "common/rdtsc.h"
#include "common/scope_exit.h"
#include "core/debug_state.h"
#include "core/emulator_settings.h"
#include "core/memory.h"
#include "shader_recompiler/runtime_info.h"
#include "video_core/amdgpu/split_hazard.h"
#include "video_core/host_shaders/neo_id_derive_comp.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_hle.h"
#include "video_core/texture_cache/image_view.h"
#include "video_core/texture_cache/texture_cache.h"

#ifdef MemoryBarrier
#undef MemoryBarrier
#endif

namespace Vulkan {

// Shape-only diagnostic; enable for one run, then compile it out for FPS comparisons.
#ifndef SHADPS4_BINDING_PROBE
#define SHADPS4_BINDING_PROBE 0
#endif
#if SHADPS4_BINDING_PROBE
static void ProbeBindingChanges(const Pipeline* pipeline,
                                std::span<const Shader::BoundStage> bound_stages) {
    if (pipeline->IsCompute()) {
        return;
    }
    struct PreviousStage {
        const Shader::Info* info{};
        size_t num_buffers{}, num_images{};
        std::array<AmdGpu::Buffer, Shader::NUM_BUFFERS> buffers{};
        std::array<AmdGpu::Image, Shader::NUM_IMAGES> images{};
    };
    static std::array<PreviousStage, Shader::MaxStageTypes> previous{};
    static u64 draws{}, stages{}, same_program{}, buffers{}, same_buffers{}, special{}, images{},
        same_images{}, written{};
    ++draws;
    for (const auto& bound : bound_stages) {
        const auto* stage = bound.info;
        if (!stage) {
            continue;
        }
        auto& prev = previous[static_cast<u32>(stage->l_stage)];
        ++stages;
        same_program += prev.info == stage;
        size_t i = 0;
        for (const auto& desc : stage->buffers) {
            const auto sharp = desc.GetSharp(bound);
            if (desc.IsSpecial()) {
                ++special;
            } else {
                ++buffers;
                written += desc.is_written;
                same_buffers += prev.info && i < prev.num_buffers &&
                                std::memcmp(&sharp, &prev.buffers[i], sizeof(sharp)) == 0;
            }
            prev.buffers[i++] = sharp;
        }
        prev.num_buffers = i;
        i = 0;
        for (const auto& desc : stage->images) {
            const auto sharp = desc.GetSharp(bound);
            ++images;
            same_images += prev.info && i < prev.num_images &&
                           std::memcmp(&sharp, &prev.images[i], sizeof(sharp)) == 0;
            prev.images[i++] = sharp;
        }
        prev.num_images = i;
        prev.info = stage;
    }
    if (draws >= 200000) {
        LOG_INFO(Render_Vulkan,
                 "BindingChanges: draws={} buffers/draw={:.2f} raw_same_buffers={:.1f}% "
                 "special/draw={:.2f} written_buffers={:.1f}% images/draw={:.2f} "
                 "raw_same_images={:.1f}% same_stage_program={:.1f}%",
                 draws, double(buffers) / draws, buffers ? 100.0 * same_buffers / buffers : 0.0,
                 double(special) / draws, buffers ? 100.0 * written / buffers : 0.0,
                 double(images) / draws, images ? 100.0 * same_images / images : 0.0,
                 stages ? 100.0 * same_program / stages : 0.0);
        draws = stages = same_program = buffers = same_buffers = special = images = same_images =
            written = 0;
    }
    // Equal descriptors do not prove equal contents or cache state. This probe
    // never skips synchronization, resource lookup, barriers, or descriptor writes.
}
#endif

static Shader::PushData MakeUserData(const AmdGpu::Regs& regs) {
    // TODO(roamic): Add support for multiple viewports and geometry shaders when ViewportIndex
    // is encountered and implemented in the recompiler.
    Shader::PushData push_data{};
    push_data.xoffset = regs.viewport_control.xoffset_enable ? regs.viewports[0].xoffset : 0.f;
    push_data.xscale = regs.viewport_control.xscale_enable ? regs.viewports[0].xscale : 1.f;
    push_data.yoffset = regs.viewport_control.yoffset_enable ? regs.viewports[0].yoffset : 0.f;
    push_data.yscale = regs.viewport_control.yscale_enable ? regs.viewports[0].yscale : 1.f;
    return push_data;
}

Rasterizer::Rasterizer(const Instance& instance_, Scheduler& scheduler_,
                       AmdGpu::Liverpool* liverpool_)
    : instance{instance_}, scheduler{scheduler_},
      native_bindings{NativeBindings::Create(instance_, scheduler_)}, page_manager{this},
      buffer_cache{instance, scheduler, liverpool_, texture_cache, page_manager},
      texture_cache{instance, scheduler, liverpool_, buffer_cache, page_manager},
      liverpool{liverpool_}, memory{Core::Memory::Instance()},
      pipeline_cache{instance, scheduler, liverpool} {
    if (!EmulatorSettings.IsNullGPU()) {
        liverpool->BindRasterizer(this);
    }
    memory->SetRasterizer(this);
    build_set_writes = !native_bindings || native_bindings->IsVerifying();
    const char* memo_stats = std::getenv("SHADPS4_BIND_MEMO_STATS");
    bind_memo_stats = memo_stats && memo_stats[0] == '1';
    if (bind_memo_stats) {
        LOG_INFO(Render_Vulkan, "BindMemoStats: enabled (counters only, behaviour unchanged)");
    }
}

Rasterizer::~Rasterizer() {
    scheduler.DrainRecording();
}

void Rasterizer::CpSync() {
    scheduler.EndRendering();
    auto command_access = scheduler.AcquireCommandBuffer();
    auto cmdbuf = command_access.Handle();

    const vk::MemoryBarrier ib_barrier{
        .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
        .dstAccessMask = vk::AccessFlagBits::eIndirectCommandRead,
    };
    cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                           vk::PipelineStageFlagBits::eDrawIndirect,
                           vk::DependencyFlagBits::eByRegion, ib_barrier, {}, {});
}

Rasterizer::DrawClass Rasterizer::ClassifyDraw(const AmdGpu::Regs& regs) {
    if (regs.color_control.mode == AmdGpu::ColorControl::OperationMode::EliminateFastClear) {
        return DrawClass::EliminateFastClear;
    }
    if (regs.color_control.mode == AmdGpu::ColorControl::OperationMode::FmaskDecompress) {
        return DrawClass::FmaskDecompress;
    }
    if (regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Resolve) {
        return DrawClass::Resolve;
    }
    if (regs.primitive_type == AmdGpu::PrimitiveType::None) {
        return DrawClass::PrimitiveNone;
    }

    const bool cb_disabled =
        regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Disable;
    const auto depth_copy =
        regs.depth_render_override.force_z_dirty && regs.depth_render_override.force_z_valid &&
        regs.depth_buffer.DepthValid() && regs.depth_buffer.DepthWriteValid() &&
        regs.depth_buffer.DepthAddress() != regs.depth_buffer.DepthWriteAddress();
    const auto stencil_copy =
        regs.depth_render_override.force_stencil_dirty &&
        regs.depth_render_override.force_stencil_valid && regs.depth_buffer.StencilValid() &&
        regs.depth_buffer.StencilWriteValid() &&
        regs.depth_buffer.StencilAddress() != regs.depth_buffer.StencilWriteAddress();
    if (cb_disabled && (depth_copy || stencil_copy)) {
        return depth_copy && stencil_copy ? DrawClass::DepthStencilCopy
               : depth_copy               ? DrawClass::DepthCopy
                                          : DrawClass::StencilCopy;
    }

    return DrawClass::Draw;
}

void Rasterizer::ExecuteFilteredDraw(DrawClass draw_class) {
    switch (draw_class) {
    case DrawClass::Draw:
        break;
    case DrawClass::EliminateFastClear:
        // Clears the render target if FCE is launched before any draws
        EliminateFastClear();
        break;
    case DrawClass::FmaskDecompress:
        // TODO: check for a valid MRT1 to promote the draw to the resolve pass.
        LOG_TRACE(Render_Vulkan, "FMask decompression pass skipped");
        ScopedMarkerInsert("FmaskDecompress");
        break;
    case DrawClass::Resolve:
        LOG_TRACE(Render_Vulkan, "Resolve pass");
        Resolve();
        break;
    case DrawClass::PrimitiveNone:
        LOG_TRACE(Render_Vulkan, "Primitive type 'None' skipped");
        ScopedMarkerInsert("PrimitiveTypeNone");
        break;
    case DrawClass::DepthCopy:
    case DrawClass::StencilCopy:
    case DrawClass::DepthStencilCopy:
        // Games may disable color buffer and enable force depth/stencil dirty and valid to
        // do a copy from one depth-stencil surface to another, without a pixel shader.
        // We need to detect this case and perform the copy, otherwise it will have no effect.
        LOG_TRACE(Render_Vulkan, "Performing depth-stencil override copy");
        DepthStencilCopy(draw_class != DrawClass::StencilCopy,
                         draw_class != DrawClass::DepthCopy);
        break;
    }
}

bool Rasterizer::FilterDraw() {
    // The classifier reads the front's registers (it runs where the pipeline lookup runs); the
    // actions read the back's through BackRegs().
    const DrawClass draw_class = ClassifyDraw(liverpool->regs);
    if (draw_class == DrawClass::Draw) {
        return true;
    }
    ExecuteFilteredDraw(draw_class);
    return false;
}

static bool SamePreparedTarget(const VideoCore::TextureCache::ImageDesc& a,
                               const VideoCore::TextureCache::ImageDesc& b) {
    const auto& x = a.info;
    const auto& y = b.info;
    if (a.type != b.type || a.view_info != b.view_info || x.pixel_format != y.pixel_format ||
        x.type != y.type || x.resources != y.resources || x.size != y.size ||
        x.num_bits != y.num_bits || x.num_samples != y.num_samples || x.pitch != y.pitch ||
        x.tile_mode != y.tile_mode || x.array_mode != y.array_mode ||
        x.guest_address != y.guest_address || x.guest_size != y.guest_size ||
        x.bank_swizzle != y.bank_swizzle || x.alt_tile != y.alt_tile ||
        x.stencil_addr != y.stencil_addr || x.stencil_size != y.stencil_size ||
        x.meta_info.cmask_addr != y.meta_info.cmask_addr ||
        x.meta_info.fmask_addr != y.meta_info.fmask_addr ||
        x.meta_info.htile_addr != y.meta_info.htile_addr ||
        x.meta_info.htile_clear_mask != y.meta_info.htile_clear_mask ||
        x.props.is_volume != y.props.is_volume || x.props.is_tiled != y.props.is_tiled ||
        x.props.is_pow2 != y.props.is_pow2 || x.props.is_block != y.props.is_block ||
        x.props.is_depth != y.props.is_depth || x.props.has_stencil != y.props.has_stencil) {
        return false;
    }
    for (u32 mip = 0; mip < x.resources.levels; ++mip) {
        const auto& xm = x.mips_layout[mip];
        const auto& ym = y.mips_layout[mip];
        if (xm.size != ym.size || xm.pitch != ym.pitch || xm.height != ym.height ||
            xm.offset != ym.offset) {
            return false;
        }
    }
    return true;
}

void Rasterizer::PrepareRenderState(const GraphicsPipeline* pipeline) {
    static const bool prepared_targets = [] {
        const char* value = std::getenv("SHADPS4_PREPARED_TARGETS");
        return value && value[0] == '1';
    }();
    static const bool verify_targets = [] {
        const char* value = std::getenv("SHADPS4_PREPARED_TARGETS_VERIFY");
        return value && value[0] == '1';
    }();
    const auto verify = [](bool hit, const auto& prepared, const auto& reference) {
        ASSERT(SamePreparedTarget(prepared, reference));
        static thread_local u64 calls{}, hits{};
        hits += hit;
        if (++calls % 500000 == 0) {
            LOG_INFO(Render_Vulkan, "PreparedTargetVerify: calls={} hits={}", calls, hits);
        }
    };
    // Prefetch render targets to handle overlaps with bound textures (e.g. mipgen)
    const auto& key = pipeline->GetGraphicsKey();
    const auto& regs = liverpool->BackRegs();
    if (regs.color_control.degamma_enable) {
        LOG_WARNING(Render_Vulkan, "Color buffers require gamma correction");
    }

    const bool skip_cb_binding =
        regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Disable;
    for (s32 cb = 0; cb < std::bit_width(key.mrt_mask); ++cb) {
        auto& [image_id, desc] = cb_descs[cb];
        const auto& col_buf = regs.color_buffers[cb];
        const u32 target_mask = regs.color_target_mask.GetMask(cb);
        if (skip_cb_binding || !col_buf || !target_mask || (key.mrt_mask & (1 << cb)) == 0) {
            image_id = {};
            continue;
        }
        const auto& hint = liverpool->BackCbExtent(cb);
        if (prepared_targets) {
            auto& entry = prepared_color_targets[cb];
            const bool hit = entry.valid &&
                             std::memcmp(&entry.buffer, &col_buf, sizeof(col_buf)) == 0 &&
                             std::memcmp(&entry.extent, &hint, sizeof(hint)) == 0;
            if (!hit) {
                entry.buffer = col_buf;
                entry.extent = hint;
                std::construct_at(&entry.value, col_buf, hint);
                entry.valid = true;
            }
            if (verify_targets) {
                verify(hit, entry.value, VideoCore::TextureCache::ImageDesc{col_buf, hint});
            }
            desc = entry.value;
        } else {
            std::construct_at(&desc, col_buf, hint);
        }
        image_id = bound_images.emplace_back(texture_cache.FindImage(desc));
        auto& image = texture_cache.GetImage(image_id);
        image.binding.is_target = 1u;
    }

    if ((regs.depth_control.depth_enable && regs.depth_buffer.DepthValid()) ||
        (regs.depth_control.stencil_enable && regs.depth_buffer.StencilValid())) {
        const auto htile_address = regs.depth_htile_data_base.GetAddress();
        const auto& hint = liverpool->BackDbExtent();
        auto& [image_id, desc] = db_desc;
        if (prepared_targets) {
            auto& entry = prepared_depth_target;
            const bool hit =
                entry.valid && entry.htile_address == htile_address &&
                std::memcmp(&entry.buffer, &regs.depth_buffer, sizeof(entry.buffer)) == 0 &&
                std::memcmp(&entry.view, &regs.depth_view, sizeof(entry.view)) == 0 &&
                std::memcmp(&entry.control, &regs.depth_control, sizeof(entry.control)) == 0 &&
                std::memcmp(&entry.extent, &hint, sizeof(hint)) == 0;
            if (!hit) {
                entry.buffer = regs.depth_buffer;
                entry.view = regs.depth_view;
                entry.control = regs.depth_control;
                entry.extent = hint;
                entry.htile_address = htile_address;
                std::construct_at(&entry.value, regs.depth_buffer, regs.depth_view,
                                  regs.depth_control, htile_address, hint);
                entry.valid = true;
            }
            if (verify_targets) {
                verify(hit, entry.value,
                       VideoCore::TextureCache::ImageDesc{regs.depth_buffer, regs.depth_view,
                                                          regs.depth_control, htile_address, hint});
            }
            desc = entry.value;
        } else {
            std::construct_at(&desc, regs.depth_buffer, regs.depth_view, regs.depth_control,
                              htile_address, hint);
        }
        image_id = bound_images.emplace_back(texture_cache.FindImage(desc));
        auto& image = texture_cache.GetImage(image_id);
        image.binding.is_target = 1u;
    } else {
        db_desc.first = {};
    }
}

static std::pair<u32, u32> GetDrawOffsets(
    const AmdGpu::Regs& regs, const Shader::BoundStage& info,
    const Shader::Gcn::FetchShaderData* fetch_shader) {
    u32 vertex_offset = regs.index_offset;
    u32 instance_offset = 0;
    if (fetch_shader) {
        if (vertex_offset == 0 && fetch_shader->vertex_offset_sgpr != -1) {
            vertex_offset = info.user_data[fetch_shader->vertex_offset_sgpr];
        }
        if (fetch_shader->instance_offset_sgpr != -1) {
            instance_offset = info.user_data[fetch_shader->instance_offset_sgpr];
        }
    }
    return {vertex_offset, instance_offset};
}

void Rasterizer::EliminateFastClear() {
    auto& col_buf = liverpool->BackRegs().color_buffers[0];
    if (!col_buf || !col_buf.info.fast_clear) {
        return;
    }
    VideoCore::TextureCache::ImageDesc desc(col_buf, liverpool->BackCbExtent(0));
    const auto image_id = texture_cache.FindImage(desc);
    const auto& image_view = texture_cache.FindRenderTarget(image_id, desc);
    if (!texture_cache.IsMetaCleared(col_buf.CmaskAddress(), col_buf.view.slice_start)) {
        return;
    }
    for (u32 slice = col_buf.view.slice_start; slice <= col_buf.view.slice_max; ++slice) {
        texture_cache.TouchMeta(col_buf.CmaskAddress(), slice, false);
    }
    auto& image = texture_cache.GetImage(image_id);
    const auto clear_value = LiverpoolToVK::ColorBufferClearValue(col_buf);

    ScopeMarkerBegin(fmt::format("EliminateFastClear:MRT={:#x}:M={:#x}", col_buf.Address(),
                                 col_buf.CmaskAddress()));
    image.Clear(clear_value, desc.view_info.range);
    ScopeMarkerEnd();
}

// SHADPS4_REG_SCAN=<hex>[,<hex>...] (diagnostic): every 512th draw, scan all registers for the
// given 32-bit values (e.g. a surface address >> 8) and log each register index that holds one,
// once. Finds which register points at a surface the emulator does not model (Pro ID buffer).
void Rasterizer::ScanRegistersForValue() {
    static const std::vector<u32> targets = [] {
        std::vector<u32> out;
        const char* value = std::getenv("SHADPS4_REG_SCAN");
        if (!value) {
            return out;
        }
        std::string_view list{value};
        while (!list.empty()) {
            const size_t comma = list.find(',');
            const auto item = list.substr(0, comma);
            out.push_back(u32(std::strtoull(std::string(item).c_str(), nullptr, 16)));
            list = comma == std::string_view::npos ? std::string_view{} : list.substr(comma + 1);
        }
        return out;
    }();
    if (targets.empty()) {
        return;
    }
    static u64 draws = 0;
    if ((draws++ & 511) != 0) {
        return;
    }
    static std::unordered_set<u64> seen;
    const auto& regs = liverpool->BackRegs().reg_array;
    for (u32 i = 0; i < regs.size(); ++i) {
        for (const u32 target : targets) {
            if (regs[i] == target && seen.insert((u64(i) << 32) | target).second) {
                LOG_INFO(Render_Vulkan, "REG_SCAN: value {:#x} in register word {:#x} (draw {})",
                         target, i, draws);
            }
        }
    }
}

struct Rasterizer::NeoIdDerive {
    vk::UniqueDescriptorSetLayout set_layout;
    vk::UniquePipelineLayout layout;
    vk::UniquePipeline pipeline;
    vk::Image depth_image{};
    vk::ImageView depth_view{};
    vk::Image id_image{};
    vk::ImageView id_view{};
    // Views are never destroyed while possibly in flight: superseded ones are kept (rare, small).
    std::vector<vk::UniqueImageView> views;
    u64 runs{};
};

void Rasterizer::DeriveNeoIdBuffer(VideoCore::Image& id_image) {
    static const s64 parity_override = [] {
        const char* value = std::getenv("SHADPS4_NEO_IDBUF_PARITY");
        return value ? static_cast<s64>(std::strtoll(value, nullptr, 0)) : s64{-1};
    }();
    static const u32 bucket_mask = [] {
        const char* value = std::getenv("SHADPS4_NEO_IDBUF_MASK");
        return value ? static_cast<u32>(std::strtoull(value, nullptr, 0)) : 0xFFFFF000u;
    }();
    if (!neo_depth_addr) {
        return;
    }
    const auto depth_id = texture_cache.FindDepthImageBySize(neo_depth_addr, neo_depth_size);
    if (!depth_id) {
        return;
    }
    auto& depth = texture_cache.GetImage(depth_id);
    if (depth.info.num_samples != 2 || depth.info.size.width != id_image.info.size.width ||
        depth.info.size.height != id_image.info.size.height) {
        return;
    }
    const auto device = instance.GetDevice();
    if (!neo_id_derive) {
        neo_id_derive = std::make_shared<NeoIdDerive>();
        auto& d = *neo_id_derive;
        const std::array<vk::DescriptorSetLayoutBinding, 2> bindings = {{
            {.binding = 0,
             .descriptorType = vk::DescriptorType::eSampledImage,
             .descriptorCount = 1,
             .stageFlags = vk::ShaderStageFlagBits::eCompute},
            {.binding = 1,
             .descriptorType = vk::DescriptorType::eStorageImage,
             .descriptorCount = 1,
             .stageFlags = vk::ShaderStageFlagBits::eCompute},
        }};
        const vk::DescriptorSetLayoutCreateInfo set_ci{
            .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
            .bindingCount = 2,
            .pBindings = bindings.data(),
        };
        d.set_layout = Check(device.createDescriptorSetLayoutUnique(set_ci));
        const vk::PushConstantRange push_range{
            .stageFlags = vk::ShaderStageFlagBits::eCompute, .offset = 0, .size = 16};
        const vk::PipelineLayoutCreateInfo layout_ci{
            .setLayoutCount = 1,
            .pSetLayouts = &*d.set_layout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &push_range,
        };
        d.layout = Check(device.createPipelineLayoutUnique(layout_ci));
        const auto module =
            Compile(HostShaders::NEO_ID_DERIVE_COMP, vk::ShaderStageFlagBits::eCompute, device);
        const vk::ComputePipelineCreateInfo pipeline_ci{
            .stage = {.stage = vk::ShaderStageFlagBits::eCompute, .module = module, .pName = "main"},
            .layout = *d.layout,
        };
        d.pipeline = Check(device.createComputePipelineUnique({}, pipeline_ci));
        device.destroyShaderModule(module);
        LOG_INFO(Render_Vulkan, "NeoIdDerive: pipeline created");
    }
    auto& d = *neo_id_derive;
    if (d.depth_image != depth.GetImage()) {
        d.depth_image = depth.GetImage();
        const vk::ImageViewCreateInfo view_ci{
            .image = d.depth_image,
            .viewType = vk::ImageViewType::e2D,
            .format = depth.info.pixel_format,
            .subresourceRange = {vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1},
        };
        d.views.push_back(Check(device.createImageViewUnique(view_ci)));
        d.depth_view = *d.views.back();
    }
    if (d.id_image != id_image.GetImage()) {
        d.id_image = id_image.GetImage();
        const vk::ImageViewCreateInfo view_ci{
            .image = d.id_image,
            .viewType = vk::ImageViewType::e2D,
            .format = vk::Format::eR32Uint,
            .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
        };
        d.views.push_back(Check(device.createImageViewUnique(view_ci)));
        d.id_view = *d.views.back();
    }
    const u32 s0_x = neo_depth_x0y0 & 0xF;
    const u32 parity = parity_override >= 0 ? static_cast<u32>(parity_override & 1)
                                            : ((s0_x != 0 && s0_x < 8) ? 1u : 0u);
    const std::array<u32, 4> push{id_image.info.size.width, id_image.info.size.height, parity,
                                  bucket_mask};

    scheduler.EndRendering();
    depth.Transit(vk::ImageLayout::eGeneral, vk::AccessFlagBits2::eShaderRead, {});
    id_image.Transit(vk::ImageLayout::eGeneral, vk::AccessFlagBits2::eShaderWrite, {});
    const vk::DescriptorImageInfo depth_info{.imageView = d.depth_view,
                                             .imageLayout = vk::ImageLayout::eGeneral};
    const vk::DescriptorImageInfo id_info{.imageView = d.id_view,
                                          .imageLayout = vk::ImageLayout::eGeneral};
    const std::array<vk::WriteDescriptorSet, 2> writes = {{
        {.dstBinding = 0,
         .descriptorCount = 1,
         .descriptorType = vk::DescriptorType::eSampledImage,
         .pImageInfo = &depth_info},
        {.dstBinding = 1,
         .descriptorCount = 1,
         .descriptorType = vk::DescriptorType::eStorageImage,
         .pImageInfo = &id_info},
    }};
    {
        auto command_access = scheduler.AcquireCommandBuffer();
        const auto cmdbuf = command_access.Handle();
        cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, *d.pipeline);
        cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, *d.layout, 0, writes);
        cmdbuf.pushConstants(*d.layout, vk::ShaderStageFlagBits::eCompute, 0,
                             static_cast<u32>(sizeof(push)), push.data());
        cmdbuf.dispatch((push[0] + 7) / 8, (push[1] + 7) / 8, 1);
    }
    ++d.runs;
    if (d.runs <= 20 || (d.runs & 1023) == 0) {
        LOG_INFO(Render_Vulkan,
                 "NeoIdDerive: depth {:#x} {}x{} 2s -> id {:#x} parity {} (x0y0 {:#x}) mask {:#x} "
                 "run {}",
                 neo_depth_addr, depth.info.size.width, depth.info.size.height,
                 id_image.info.guest_address, parity, neo_depth_x0y0, bucket_mask, d.runs);
    }
}

void Rasterizer::Draw(bool is_indexed, u32 index_offset) {
    RENDERER_TRACE;
    Common::SampleCommandProcessorCore();
    ScanRegistersForValue();

    auto sample = Common::DrawSampleProbe::Begin();
    const auto t_entry = Common::FaultMonitor::Now();
    scheduler.PopPendingOperations();
    if (native_bindings) {
        // The only point where the heap backend may flush or wait; nothing is assembled yet.
        native_bindings->BeginDraw();
    }

    if (!FilterDraw()) {
        return;
    }

    AmdGpu::SplitHazard::NoteDraw(liverpool->regs.ps_program, liverpool->regs.vs_program);
    if (SkipEmptyDraw(liverpool->regs)) {
        return;
    }

    // Scoped timers: the stack scanner reports inclusive frequencies with false positives and
    // cannot apportion cost here. See Common::FaultMonitor for why these exist.
    const auto t_pipeline = Common::FaultMonitor::Now();
    sample.BeginPhase(Common::DrawSampleProbe::Phase::Pipeline);
    const GraphicsPipeline* pipeline = pipeline_cache.GetGraphicsPipeline();
    sample.EndPhase(Common::DrawSampleProbe::Phase::Pipeline);
    const auto pipeline_ns = Common::FaultMonitor::SinceNs(t_pipeline);
    if (!pipeline) {
        return;
    }
    BindStageData(pipeline);
    DrawTail(pipeline, is_indexed, index_offset, sample, t_entry, t_pipeline, pipeline_ns);
}

bool Rasterizer::SkipEmptyDraw(const AmdGpu::Regs& r) {
    // SHADPS4_EMPTY_DRAWS=1 counts direct draws with zero indices or zero instances (they render
    // nothing but still bind every resource); =2 also skips them. Clears and resolves are
    // handled in FilterDraw before this point. Register reads and statics only (front-safe).
    static const int empty_draws = [] {
        const char* value = std::getenv("SHADPS4_EMPTY_DRAWS");
        return value ? value[0] - '0' : 0;
    }();
    if (empty_draws > 0) {
        static u64 draws = 0, empty = 0;
        const bool is_empty = r.num_indices == 0 || r.num_instances.NumInstances() == 0;
        ++draws;
        empty += is_empty;
        if ((draws & 0xfffff) == 0) {
            LOG_INFO(Render_Vulkan, "EMPTY_DRAWS: {} of {} direct draws ({:.2f}%)", empty, draws,
                     100.0 * empty / draws);
        }
        if (is_empty && empty_draws == 2) {
            return true;
        }
    }
    return false;
}

void Rasterizer::PrepareDrawFront(SplitDraw& draw) {
    // CP split, front thread: the part of Draw/DrawIndirect before PopPendingOperations that
    // reads only the front's registers and the pipeline cache, in the same order as the inline
    // path. Only order change: the lookup now runs before the back's PopPendingOperations and
    // native_bindings->BeginDraw instead of after them.
    // SPLIT-TODO: confirm the pipeline constructor does not depend on BeginDraw (heap backend).
    draw.draw_class = ClassifyDraw(liverpool->regs);
    if (draw.draw_class != DrawClass::Draw) {
        return;
    }
    if (!draw.args.is_indirect) {
        AmdGpu::SplitHazard::NoteDraw(liverpool->regs.ps_program, liverpool->regs.vs_program);
        draw.skip = SkipEmptyDraw(liverpool->regs);
        if (draw.skip) {
            return;
        }
    }
    draw.pipeline = pipeline_cache.GetGraphicsPipeline();
}

const ComputePipeline* Rasterizer::PrepareDispatchFront() {
    // CP split, front thread: reads the front's current queue cs registers (GetCsRegs).
    return pipeline_cache.GetComputePipeline();
}

void Rasterizer::ExecuteDraw(const SplitDraw& draw,
                             const std::array<Shader::BoundStage, Shader::MaxStageTypes>& stages) {
    // CP split, back thread: Draw/DrawIndirect with the front's classification, lookup and
    // per-stage data. Registers come from the back's mirror through BackRegs().
    RENDERER_TRACE;
    // Same prologue as the inline Draw/DrawIndirect, minus what the front already did.
    const auto begin = [&] {
        scheduler.PopPendingOperations();
        if (native_bindings) {
            native_bindings->BeginDraw();
        }
        if (draw.draw_class != DrawClass::Draw) {
            ExecuteFilteredDraw(draw.draw_class);
            return false;
        }
        if (draw.skip || !draw.pipeline) {
            return false;
        }
        bound_stages = stages;
        return true;
    };
    if (draw.args.is_indirect) {
        if (begin()) {
            DrawIndirectTail(draw.pipeline, draw.args.is_indexed, draw.args.arg_address,
                             draw.args.offset, draw.args.stride, draw.args.max_count,
                             draw.args.count_address);
        }
        return;
    }
    Common::SampleCommandProcessorCore();
    ScanRegistersForValue();
    auto sample = Common::DrawSampleProbe::Begin();
    const auto t_entry = Common::FaultMonitor::Now();
    if (begin()) {
        // The lookup ran on the front: the Pipeline phase and pipeline_ns are 0 here.
        DrawTail(draw.pipeline, draw.args.is_indexed, draw.args.index_offset, sample, t_entry,
                 t_entry, 0);
    }
}

void Rasterizer::ExecuteDispatch(const ComputePipeline* pipeline,
                                 const std::array<Shader::BoundStage, Shader::MaxStageTypes>& stages,
                                 bool is_indirect, VAddr address, u32 offset, u32 size) {
    // CP split, back thread: DispatchDirect/DispatchIndirect after the front's lookup. The caller
    // points Liverpool::BackCsRegs() at the record's ComputeProgram snapshot.
    RENDERER_TRACE;
    const auto t_dispatch = Common::FaultMonitor::Now();
    SCOPE_EXIT {
        Common::FaultMonitor::CountDispatch(Common::FaultMonitor::SinceNs(t_dispatch));
    };
    scheduler.PopPendingOperations();
    if (native_bindings) {
        native_bindings->BeginDraw();
    }
    if (!pipeline) {
        return;
    }
    bound_stages = stages;
    if (is_indirect) {
        DispatchIndirectTail(pipeline, address, offset, size);
    } else {
        DispatchDirectTail(pipeline);
    }
}

void Rasterizer::DrawTail(const GraphicsPipeline* pipeline, bool is_indexed, u32 index_offset,
                          Common::DrawSampleProbe& sample,
                          std::chrono::steady_clock::time_point t_entry,
                          std::chrono::steady_clock::time_point t_pipeline, u64 pipeline_ns) {
    const auto& regs = liverpool->BackRegs();
    const auto t_prepare = Common::FaultMonitor::Now();
    sample.BeginPhase(Common::DrawSampleProbe::Phase::Prepare);
    PrepareRenderState(pipeline);
    sample.EndPhase(Common::DrawSampleProbe::Phase::Prepare);
    const auto t_bindres = Common::FaultMonitor::Now();
    sample.BeginPhase(Common::DrawSampleProbe::Phase::BindResources);
    const bool bound = BindResources(pipeline);
    sample.EndPhase(Common::DrawSampleProbe::Phase::BindResources);
    const auto bindres_ns = Common::FaultMonitor::SinceNs(t_bindres);
    if (!bound) {
        return;
    }
    const auto t_begin = Common::FaultMonitor::Now();
    sample.BeginPhase(Common::DrawSampleProbe::Phase::BuildRenderingState);
    {
        // Diagnostic SHADPS4_NEO_REG_DIAG=1: raw depth / AA / CB0 registers, logged once per
        // distinct combination (max 60), to see how PS4 Pro checkerboard programs the DB.
        static const bool neo_reg_diag = [] {
            const char* value = std::getenv("SHADPS4_NEO_REG_DIAG");
            return value && value[0] == '1';
        }();
        if (neo_reg_diag) {
            // Sample-location registers (PA_SC_AA_SAMPLE_LOCS_PIXEL_X0Y0_0 .. X1Y1_0) and
            // PA_SC_AA_CONFIG: log each change with a timestamp (first 200), to see whether the
            // Pro checkerboard alternates sample positions per frame.
            const auto& ra = liverpool->BackRegs().reg_array;
            constexpr u32 base = AmdGpu::Regs::ContextRegWordOffset;
            const std::array<u32, 5> locs{ra[base + 0x2F8], ra[base + 0x2FE], ra[base + 0x302],
                                          ra[base + 0x306], ra[base + 0x30A]};
            static std::array<u32, 5> last_locs{};
            static u32 loc_lines = 0;
            if (locs != last_locs && loc_lines < 200) {
                ++loc_lines;
                last_locs = locs;
                const auto us = std::chrono::duration_cast<std::chrono::microseconds>(
                                    std::chrono::steady_clock::now().time_since_epoch())
                                    .count();
                LOG_WARNING(Render_Vulkan,
                            "NeoSampleLocs: t={}us AA_CONFIG={:#x} X0Y0={:#010x} X1Y0={:#010x} "
                            "X0Y1={:#010x} X1Y1={:#010x} rt0={}x{}",
                            us, locs[0], locs[1], locs[2], locs[3], locs[4],
                            liverpool->BackRegs().color_buffers[0].Pitch(),
                            liverpool->BackRegs().color_buffers[0].Height());
            }
        }
        if (neo_reg_diag && liverpool->BackRegs().depth_buffer.DepthValid()) {
            static constexpr std::array<u32, 16> offsets{
                0x000, 0x003, 0x010, 0x011, 0x012, 0x014, 0x016, 0x017,
                0x201, 0x203, 0x292, 0x293, 0x2F8, 0x2F9, 0x31D, 0x31C};
            std::array<u32, 16> values{};
            for (size_t i = 0; i < offsets.size(); ++i) {
                values[i] = liverpool->BackRegs().reg_array[AmdGpu::Regs::ContextRegWordOffset +
                                                            offsets[i]];
            }
            static std::set<std::array<u32, 16>> seen;
            if (seen.size() < 60 && seen.insert(values).second) {
                LOG_WARNING(Render_Vulkan,
                            "NeoRegs: DB_RENDER_CONTROL={:#x} DB_RENDER_OVERRIDE={:#x} "
                            "Z_INFO={:#x} STENCIL_INFO={:#x} Z_READ_BASE={:#x} Z_WRITE_BASE={:#x} "
                            "DEPTH_SIZE={:#x} DEPTH_SLICE={:#x} DB_EQAA={:#x} "
                            "DB_SHADER_CONTROL={:#x} PA_SC_MODE_CNTL_0={:#x} MODE_CNTL_1={:#x} "
                            "PA_SC_AA_CONFIG={:#x} AA_MASK?={:#x} CB0_ATTRIB={:#x} CB0_INFO={:#x}",
                            values[0], values[1], values[2], values[3], values[4], values[5],
                            values[6], values[7], values[8], values[9], values[10], values[11],
                            values[12], values[13], values[14], values[15]);
            }
        }
    }
    const auto state = BeginRendering(pipeline);
    sample.EndPhase(Common::DrawSampleProbe::Phase::BuildRenderingState);
    const auto t_vtxbuf = Common::FaultMonitor::Now();
    Common::FaultMonitor::CountDraw(pipeline_ns, bindres_ns,
                                    Common::FaultMonitor::DiffNs(t_begin, t_vtxbuf));

    sample.BeginPhase(Common::DrawSampleProbe::Phase::VertexIndexBuffers);
    buffer_cache.BindVertexBuffers(*pipeline, GetBoundStage(Shader::LogicalStage::Vertex),
                                   buffer_barriers);
    if (is_indexed) {
        buffer_cache.BindIndexBuffer(index_offset, buffer_barriers);
    }
    sample.EndPhase(Common::DrawSampleProbe::Phase::VertexIndexBuffers);
    const auto t_descwrite = Common::FaultMonitor::Now();

    sample.BeginPhase(Common::DrawSampleProbe::Phase::PipelineResources);
    EmitPipelineResources(pipeline);
    sample.EndPhase(Common::DrawSampleProbe::Phase::PipelineResources);
    const auto t_dynstate = Common::FaultMonitor::Now();
    sample.BeginPhase(Common::DrawSampleProbe::Phase::DynamicState);
    UpdateDynamicState(pipeline, is_indexed);
    sample.EndPhase(Common::DrawSampleProbe::Phase::DynamicState);
    const auto t_schedbegin = Common::FaultMonitor::Now();
    sample.BeginPhase(Common::DrawSampleProbe::Phase::SchedulerBeginRendering);
    scheduler.BeginRendering(state);
    sample.EndPhase(Common::DrawSampleProbe::Phase::SchedulerBeginRendering);
    const auto t_emit = Common::FaultMonitor::Now();

    sample.BeginPhase(Common::DrawSampleProbe::Phase::ActualDraw);
    const auto& vs_info = GetBoundStage(Shader::LogicalStage::Vertex);
    const auto& fetch_shader = pipeline->GetFetchShader();
    const auto [vertex_offset, instance_offset] = GetDrawOffsets(regs, vs_info, fetch_shader);

    scheduler.Record([handle = pipeline->Handle(), is_indexed, count = regs.num_indices,
                      instances = regs.num_instances.NumInstances(), vertex_offset,
                      instance_offset](vk::CommandBuffer cmdbuf) {
        cmdbuf.bindPipeline(vk::PipelineBindPoint::eGraphics, handle);
        if (is_indexed) {
            cmdbuf.drawIndexed(count, instances, 0, s32(vertex_offset), instance_offset);
        } else {
            cmdbuf.draw(count, instances, vertex_offset, instance_offset);
        }
    });
    DebugState.IncDrawCall();
    scheduler.NoteDraw();

    sample.EndPhase(Common::DrawSampleProbe::Phase::ActualDraw);
    sample.BeginPhase(Common::DrawSampleProbe::Phase::ResetImageBindings);
    ResetBindings();
    sample.EndPhase(Common::DrawSampleProbe::Phase::ResetImageBindings);
    sample.Finish();

    // SHADPS4_FLUSH_EVERY_DRAWS=N: hand recorded work to the GPU every N draws. Guest readback
    // faults wait for everything queued before their copy; measured 2026-09-23 Game:Main spent
    // 51% of its wall time in that wait while the CP sat 41% idle, so earlier submission trades
    // spare CP time for shorter readback latency. 0 (default) keeps the old cadence.
    static const u32 flush_every_draws = [] {
        const char* value = std::getenv("SHADPS4_FLUSH_EVERY_DRAWS");
        return value ? static_cast<u32>(std::strtoul(value, nullptr, 10)) : 0u;
    }();
    if (flush_every_draws != 0) {
        static thread_local u32 draws_since_flush = 0;
        if (++draws_since_flush >= flush_every_draws) {
            draws_since_flush = 0;
            scheduler.Flush();
        }
    }
    // SHADPS4_READBACK_EARLY_FLUSH=1: this draw wrote into memory the guest recently read
    // back; submit now so the next readback of it waits less (see BufferCache::NoteGpuWrite).
    if (buffer_cache.TakeReadbackFlushRequest()) {
        static thread_local u32 early_flushes = 0;
        ++early_flushes;
        scheduler.Flush();
    }

    Common::FaultMonitor::CountDrawTail(Common::FaultMonitor::DiffNs(t_entry, t_pipeline) +
                                            Common::FaultMonitor::DiffNs(t_prepare, t_bindres),
                                        Common::FaultMonitor::DiffNs(t_vtxbuf, t_descwrite),
                                        Common::FaultMonitor::DiffNs(t_descwrite, t_dynstate),
                                        Common::FaultMonitor::DiffNs(t_dynstate, t_schedbegin),
                                        Common::FaultMonitor::DiffNs(t_schedbegin, t_emit),
                                        Common::FaultMonitor::SinceNs(t_emit));
}

void Rasterizer::DrawIndirect(bool is_indexed, VAddr arg_address, u32 offset, u32 stride,
                              u32 max_count, VAddr count_address) {
    RENDERER_TRACE;

    scheduler.PopPendingOperations();
    if (native_bindings) {
        native_bindings->BeginDraw();
    }

    if (!FilterDraw()) {
        return;
    }

    const GraphicsPipeline* pipeline = pipeline_cache.GetGraphicsPipeline();
    if (!pipeline) {
        return;
    }
    BindStageData(pipeline);
    DrawIndirectTail(pipeline, is_indexed, arg_address, offset, stride, max_count, count_address);
}

void Rasterizer::DrawIndirectTail(const GraphicsPipeline* pipeline, bool is_indexed,
                                  VAddr arg_address, u32 offset, u32 stride, u32 max_count,
                                  VAddr count_address) {
    PrepareRenderState(pipeline);
    if (!BindResources(pipeline)) {
        return;
    }
    const auto state = BeginRendering(pipeline);

    buffer_cache.BindVertexBuffers(*pipeline, GetBoundStage(Shader::LogicalStage::Vertex),
                                   buffer_barriers);
    if (is_indexed) {
        buffer_cache.BindIndexBuffer(0, buffer_barriers);
    }

    const auto& [buffer, base] =
        buffer_cache.ObtainBuffer(arg_address + offset, stride * max_count, false);

    VideoCore::Buffer* count_buffer{};
    u32 count_base{};
    if (count_address != 0) {
        std::tie(count_buffer, count_base) = buffer_cache.ObtainBuffer(count_address, 4, false);
    }

    if (auto barrier = buffer->GetBarrier(vk::AccessFlagBits2::eIndirectCommandRead,
                                          vk::PipelineStageFlagBits2::eDrawIndirect)) {
        buffer_barriers.emplace_back(*barrier);
    }
    if (count_buffer) {
        if (auto barrier = count_buffer->GetBarrier(vk::AccessFlagBits2::eIndirectCommandRead,
                                                    vk::PipelineStageFlagBits2::eDrawIndirect)) {
            buffer_barriers.emplace_back(*barrier);
        }
    }

    EmitPipelineResources(pipeline);
    UpdateDynamicState(pipeline, is_indexed);
    scheduler.BeginRendering(state);

    // We can safely ignore both SGPR UD indices and results of fetch shader parsing, as vertex and
    // instance offsets will be automatically applied by Vulkan from indirect args buffer.

    ASSERT((is_indexed ? sizeof(VkDrawIndexedIndirectCommand) : sizeof(VkDrawIndirectCommand)) ==
           stride);
    scheduler.Record([handle = pipeline->Handle(), buffer = buffer->Handle(), base,
                      count_handle = count_buffer ? count_buffer->Handle() : vk::Buffer{},
                      has_count = count_address != 0, count_base, max_count, stride,
                      is_indexed](vk::CommandBuffer cmdbuf) {
        cmdbuf.bindPipeline(vk::PipelineBindPoint::eGraphics, handle);
        if (is_indexed) {
            if (has_count) {
                cmdbuf.drawIndexedIndirectCount(buffer, base, count_handle, count_base, max_count,
                                                stride);
            } else {
                cmdbuf.drawIndexedIndirect(buffer, base, max_count, stride);
            }
        } else {
            if (has_count) {
                cmdbuf.drawIndirectCount(buffer, base, count_handle, count_base, max_count, stride);
            } else {
                cmdbuf.drawIndirect(buffer, base, max_count, stride);
            }
        }
    });
    DebugState.IncDrawCall();

    ResetBindings();
}

void Rasterizer::DispatchDirect() {
    RENDERER_TRACE;

    // 526 dispatches per forest frame, and they sit in the 27% of the frame that Draw's timers do
    // not cover. SCOPE_EXIT so the several early returns below are still counted.
    const auto t_dispatch = Common::FaultMonitor::Now();
    SCOPE_EXIT {
        Common::FaultMonitor::CountDispatch(Common::FaultMonitor::SinceNs(t_dispatch));
    };

    scheduler.PopPendingOperations();
    if (native_bindings) {
        native_bindings->BeginDraw();
    }

    const ComputePipeline* pipeline = pipeline_cache.GetComputePipeline();
    if (!pipeline) {
        return;
    }
    BindStageData(pipeline);
    DispatchDirectTail(pipeline);
}

void Rasterizer::DispatchDirectTail(const ComputePipeline* pipeline) {
    const auto& cs_program = liverpool->BackCsRegs();
    const auto& cs = GetBoundStage(Shader::LogicalStage::Compute);
    if (ExecuteShaderHLE(cs, liverpool->BackRegs(), cs_program, *this)) {
        return;
    }

    if (!BindResources(pipeline)) {
        return;
    }

    scheduler.EndRendering();
    EmitPipelineResources(pipeline);

    scheduler.Record([handle = pipeline->Handle(), x = cs_program.dim_x, y = cs_program.dim_y,
                      z = cs_program.dim_z](vk::CommandBuffer cmdbuf) {
        cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, handle);
        cmdbuf.dispatch(x, y, z);
    });
    DebugState.IncDispatch();

    ResetBindings();
    if (buffer_cache.TakeReadbackFlushRequest()) {
        scheduler.Flush();
    }
}

void Rasterizer::DispatchIndirect(VAddr address, u32 offset, u32 size) {
    RENDERER_TRACE;

    const auto t_dispatch = Common::FaultMonitor::Now();
    SCOPE_EXIT {
        Common::FaultMonitor::CountDispatch(Common::FaultMonitor::SinceNs(t_dispatch));
    };

    scheduler.PopPendingOperations();
    if (native_bindings) {
        native_bindings->BeginDraw();
    }

    const ComputePipeline* pipeline = pipeline_cache.GetComputePipeline();
    if (!pipeline) {
        return;
    }
    BindStageData(pipeline);
    DispatchIndirectTail(pipeline, address, offset, size);
}

void Rasterizer::DispatchIndirectTail(const ComputePipeline* pipeline, VAddr address, u32 offset,
                                      u32 size) {
    if (!BindResources(pipeline)) {
        return;
    }

    const auto [buffer, base] = buffer_cache.ObtainBuffer(address + offset, size, false);

    if (auto barrier = buffer->GetBarrier(vk::AccessFlagBits2::eIndirectCommandRead,
                                          vk::PipelineStageFlagBits2::eDrawIndirect)) {
        buffer_barriers.emplace_back(*barrier);
    }

    scheduler.EndRendering();
    EmitPipelineResources(pipeline);

    scheduler.Record(
        [handle = pipeline->Handle(), buffer = buffer->Handle(), base](vk::CommandBuffer cmdbuf) {
            cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, handle);
            cmdbuf.dispatchIndirect(buffer, base);
        });
    DebugState.IncDispatch();

    ResetBindings();
}

u64 Rasterizer::Flush() {
    const u64 current_tick = scheduler.CurrentTick();
    SubmitInfo info{};
    scheduler.Flush(info);
    return current_tick;
}

void Rasterizer::Finish() {
    scheduler.Finish();
}

void Rasterizer::DeferToGpu(Common::UniqueFunction<void>&& func, bool flush) {
    // Queued at the current tick first, then that tick is submitted (or, lazily, left to the
    // next flush: every SHADPS4_FLUSH_EVERY_DRAWS draws, a CP memory wait or CP idle).
    scheduler.DeferPriorityOperation(std::move(func));
    if (flush) {
        Flush();
    } else {
        lazy_fence_tick = scheduler.CurrentTick();
    }
}

void Rasterizer::FlushLazyFences() {
    if (lazy_fence_tick != 0 && lazy_fence_tick == scheduler.CurrentTick()) {
        Flush();
    }
    lazy_fence_tick = 0;
}

void Rasterizer::OnSubmit() {
    {
        // Diagnostic: SHADPS4_DUMP_IMAGES=addr[,addr...] (hex) with SHADPS4_DUMP_IMAGES_AT_S=N
        // dumps those surfaces at the first submit N s after the first one, twice (a frame apart).
        static const std::vector<VAddr> dump_addresses = [] {
            std::vector<VAddr> out;
            if (const char* value = std::getenv("SHADPS4_DUMP_IMAGES")) {
                std::string list{value};
                size_t pos = 0;
                while (pos < list.size()) {
                    const size_t comma = list.find(',', pos);
                    out.push_back(std::strtoull(list.substr(pos, comma - pos).c_str(), nullptr, 16));
                    pos = comma == std::string::npos ? list.size() : comma + 1;
                }
            }
            return out;
        }();
        if (!dump_addresses.empty()) {
            static const double at_s = [] {
                const char* value = std::getenv("SHADPS4_DUMP_IMAGES_AT_S");
                return value ? std::atof(value) : 120.0;
            }();
            static const auto first = std::chrono::steady_clock::now();
            static u32 dumps_done = 0;
            if (dumps_done < 2 &&
                std::chrono::duration<double>(std::chrono::steady_clock::now() - first).count() >=
                    at_s + dumps_done * 0.05) {
                texture_cache.DebugDumpImages(dump_addresses, dumps_done++);
            }
        }
    }
    if (fault_process_pending) {
        fault_process_pending = false;
        buffer_cache.ProcessFaultBuffer();
    }
    texture_cache.ProcessDownloadImages();
    texture_cache.RunGarbageCollector();
    buffer_cache.RunGarbageCollector();
}

bool Rasterizer::BindResources(const Pipeline* pipeline) {
    if (!pending_stencil_writebacks.empty()) {
        FlushStencilWritebacks();
    }
    if (IsComputeImageCopy(pipeline) || IsComputeMetaClear(pipeline) ||
        IsComputeImageClear(pipeline)) {
        return false;
    }

#if SHADPS4_BINDING_PROBE
    ProbeBindingChanges(pipeline, BoundStages(pipeline));
#endif

    set_write_index = 0;
    set_writes.clear();
    buffer_barriers.clear();
    buffer_infos.clear();
    image_infos.clear();

    bool uses_dma = false;

    // Bind resource buffers and textures.
    Shader::Backend::Bindings binding{};
    push_data = MakeUserData(liverpool->BackRegs());
    u8 memo_draw_bits = 3;
    for (const auto& bound : BoundStages(pipeline)) {
        const auto* stage = bound.info;
        if (!stage) {
            continue;
        }
        if (build_set_writes) {
            set_writes.resize(set_writes.size() + stage->buffers.size() + stage->images.size() +
                              stage->samplers.size());
        }
        bound.PushUd(binding, push_data);
        if (bind_memo_stats) {
            // Runs the same BindBuffers/BindTextures, with the memo's key and checks around them.
            memo_draw_bits &= BindStageMemoStats(pipeline, bound, binding);
        } else {
            // Split measured because BindResources is 6.8 us of the 8.0 us spent per draw - 85% -
            // and a stack scan could not tell these two apart from anything else.
            const auto t_buf = Common::FaultMonitor::Now();
            BindBuffers(bound, binding, push_data);
            const auto t_tex = Common::FaultMonitor::Now();
            BindTextures(bound, binding);
            Common::FaultMonitor::CountBindSplit(Common::FaultMonitor::DiffNs(t_buf, t_tex),
                                                 Common::FaultMonitor::SinceNs(t_tex));
        }
        uses_dma |= stage->uses_dma;
    }
    native_bound_count = binding.unified;
    if (bind_memo_stats) {
        BindMemoEndDraw(pipeline->IsCompute(), memo_draw_bits);
    }

    if (uses_dma) {
        // We only use fault buffer for DMA right now.
        Common::RecursiveSharedLock lock{mapped_ranges_mutex};
        for (auto& range : mapped_ranges) {
            buffer_cache.SynchronizeBuffersInRange(range.lower(), range.upper() - range.lower());
        }
        fault_process_pending = true;
    }

    return true;
}

namespace {

// SHADPS4_BIND_MEMO_STATS state, CP thread only. One record per (pipeline, stage), which is where
// the memo itself would keep it. Generations are the values read before the full path that
// produced the outputs, so a bump during that path makes the next occurrence miss.
struct BindMemoRecord {
    struct Buf {
        VideoCore::BufferId id{};
        VAddr addr{};
        u64 size{};
        vk::Buffer handle{};
        vk::DeviceSize offset{};
        vk::DeviceSize range{};
        u8 adjust{};
        bool memo{}; // read-only guest binding: a hit would skip it
    };
    struct Img {
        VideoCore::ImageId id{};
        u64 uid{};
        const VideoCore::Image::BackingImage* backing{};
        vk::ImageView view{};
        vk::ImageLayout layout{};
        VideoCore::Image::State state{}; // after the full path's Transit
    };
    bool valid{};
    std::vector<AmdGpu::Buffer> vsharps;
    std::vector<AmdGpu::Image> tsharps;
    std::vector<AmdGpu::Sampler> ssharps;
    u64 border{};
    u64 buf_gen{};
    u64 buf_tex_gen{};
    bool buf_excluded{};
    bool buf_texel{};
    std::vector<Buf> bufs;
    u64 tex_gen{};
    u64 smp_gen{};
    bool tex_excluded{};
    std::vector<Img> imgs;
    std::vector<vk::Sampler> samplers;
};

enum BindMemoBufMiss : u32 { BufExcluded, BufGen, BufTexGen, BufDirtyPage, NumBufMiss };
enum BindMemoTexMiss : u32 {
    TexExcluded,
    TexGen,
    TexSamplerGen,
    TexUid,
    TexBacking,
    TexDirtyImage,
    TexFlags,
    NumTexMiss
};

// Hit work that is not replayed: GetBarrier per skipped buffer, the sampler LRU touch per
// sampler. Guesses, subtracted so the estimate is not flattered by leaving them out.
constexpr double BindMemoAllowBufNs = 5.0;
constexpr double BindMemoAllowSamplerNs = 5.0;

struct BindMemoStats {
    std::unordered_map<u64, BindMemoRecord> records;
    // Key under test and dry-replay scratch; capacity is kept so timed spans do not allocate.
    std::vector<AmdGpu::Buffer> vsharps;
    std::vector<AmdGpu::Image> tsharps;
    std::vector<AmdGpu::Sampler> ssharps;
    std::vector<vk::DescriptorBufferInfo> replay_buffers;
    std::vector<vk::DescriptorImageInfo> replay_images;
    std::array<u8, Shader::NUM_BUFFERS> replay_adjust{};
    u64 stage_counter{};
    bool started{};
    // Window origin.
    u64 frames0{}, buf_gen0{}, tex_gen0{}, smp_gen0{}, tsc0{};
    std::chrono::steady_clock::time_point clock0{};
    // Window counters. Stage types: 0 vs, 1 ps, 2 cs, 3 other.
    u64 binds{}, stages{}, draws{}, draw_key{}, draw_would{}, dma_stages{};
    std::array<u64, 4> type_stages{}, type_key{}, type_would{};
    u64 buf_key{}, buf_would{}, tex_key{}, tex_would{}, transit_needed{};
    std::array<u64, NumBufMiss> buf_miss{};
    std::array<u64, NumTexMiss> tex_miss{};
    u64 slot_buf{}, slot_buf_key{}, slot_img{}, slot_img_key{}, slot_smp{}, slot_smp_key{};
    u64 mis_buf{}, mis_upload{}, mis_img{}, mis_smp{}, layout_diff{};
    u64 memo_bufs_on_hit{}, samplers_on_hit{};
    // Timed stages (1 in 16), TSC ticks. t_empty is two back-to-back reads, subtracted from
    // every span average so the fenced reads themselves are not counted as work.
    u64 timed{}, t_empty{}, t_key_buf{}, t_key_tex{};
    u64 t_check_buf_n{}, t_check_buf{}, t_check_tex_n{}, t_check_tex{};
    u64 t_replay_buf_n{}, t_replay_buf{}, t_replay_tex_n{}, t_replay_tex{};
    u64 t_buf_hit_n{}, t_buf_hit{}, t_fixed_hit{}, t_buf_miss_n{}, t_buf_miss{};
    u64 t_tex_hit_n{}, t_tex_hit{}, t_tex_miss_n{}, t_tex_miss{};
};

BindMemoStats& GetBindMemoStats() {
    static BindMemoStats stats;
    return stats;
}

template <typename T>
bool SameBytes(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() &&
           (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0);
}

template <typename T>
bool SameSlot(const std::vector<T>& a, const std::vector<T>& b, size_t i) {
    return i < a.size() && i < b.size() && std::memcmp(&a[i], &b[i], sizeof(T)) == 0;
}

} // Anonymous namespace

u8 Rasterizer::BindStageMemoStats(const Pipeline* pipeline, const Shader::BoundStage& bound,
                                  Shader::Backend::Bindings& binding) {
    const auto& stage = *bound.info;
    auto& st = GetBindMemoStats();
    if (!st.started) {
        st.started = true;
        st.frames0 = Common::FaultMonitor::presented_frames.load(std::memory_order_relaxed);
        st.buf_gen0 = buffer_cache.StructGeneration();
        st.tex_gen0 = texture_cache.StructGeneration();
        st.smp_gen0 = texture_cache.SamplerGeneration();
        st.clock0 = std::chrono::steady_clock::now();
        st.tsc0 = Common::FencedRDTSC();
    }
    const bool timed = (++st.stage_counter & 15) == 0;
    const auto tsc = [timed] { return timed ? Common::FencedRDTSC() : u64{0}; };
    // The lookup stays outside the timed spans: the memo would own its record.
    const u64 record_key =
        (static_cast<u64>(reinterpret_cast<uintptr_t>(pipeline)) << 3) | u64(stage.l_stage);
    auto& rec = st.records[record_key];
    const u32 type = stage.l_stage == Shader::LogicalStage::Vertex     ? 0
                     : stage.l_stage == Shader::LogicalStage::Fragment ? 1
                     : stage.l_stage == Shader::LogicalStage::Compute  ? 2
                                                                       : 3;

    const u64 t_empty = tsc();
    // Key, buffer half: every guest V#, read as BindBuffers reads it.
    const u64 t_key = tsc();
    st.vsharps.clear();
    for (const auto& desc : stage.buffers) {
        st.vsharps.push_back(desc.IsSpecial() ? AmdGpu::Buffer{} : desc.GetSharp(bound));
    }
    const bool buf_key = rec.valid && SameBytes(st.vsharps, rec.vsharps);
    const u64 t_key_tex = tsc();
    // Key, texture half: every T#, every S# after BindTextures' aniso adjustment, border base.
    st.tsharps.clear();
    for (const auto& desc : stage.images) {
        st.tsharps.push_back(desc.GetSharp(bound));
    }
    st.ssharps.clear();
    for (const auto& sampler : stage.samplers) {
        auto ssharp = sampler.GetSharp(bound);
        if (sampler.disable_aniso) {
            const auto& tsharp = stage.images[sampler.associated_image].GetSharp(bound);
            if (tsharp.base_level == 0 && tsharp.last_level == 0) {
                ssharp.max_aniso.Assign(AmdGpu::AnisoRatio::One);
            }
        }
        st.ssharps.push_back(ssharp);
    }
    const u64 border = liverpool->BackRegs().ta_bc_base.base_addr;
    const bool tex_key = rec.valid && border == rec.border && SameBytes(st.tsharps, rec.tsharps) &&
                         SameBytes(st.ssharps, rec.ssharps);

    // Validation, buffer half (design section 3): structure generation, texel aliasing, exact
    // CPU-dirty pages of every skipped range. Written and special bindings run in full anyway.
    const u64 t_check_buf = tsc();
    const u64 buf_gen = buffer_cache.StructGeneration();
    const u64 tex_gen = texture_cache.StructGeneration();
    const u64 smp_gen = texture_cache.SamplerGeneration();
    s32 buf_miss = -1;
    if (buf_key) {
        if (rec.buf_excluded || stage.uses_dma || native_bindings) {
            buf_miss = BufExcluded;
        } else if (buf_gen != rec.buf_gen) {
            buf_miss = BufGen;
        } else if (rec.buf_texel && tex_gen != rec.buf_tex_gen) {
            buf_miss = BufTexGen;
        } else {
            for (const auto& b : rec.bufs) {
                if (b.memo && b.id && buffer_cache.IsRegionCpuDirtyRelaxed(b.addr, b.size)) {
                    buf_miss = BufDirtyPage;
                    break;
                }
            }
        }
    }
    const bool buf_would = buf_key && buf_miss < 0;
    // Validation, texture half: image structure and sampler generations, then per image the
    // identity, backing, UpdateImageLockFree predicate and binding flags. A state difference is
    // not a miss: the memo would call Transit.
    const u64 t_check_tex = tsc();
    s32 tex_miss = -1;
    bool transit = false;
    if (tex_key) {
        if (rec.tex_excluded || native_bindings) {
            tex_miss = TexExcluded;
        } else if (tex_gen != rec.tex_gen) {
            tex_miss = TexGen;
        } else if (!rec.samplers.empty() && smp_gen != rec.smp_gen) {
            tex_miss = TexSamplerGen;
        } else {
            for (const auto& img : rec.imgs) {
                if (!img.id) {
                    continue;
                }
                auto& image = texture_cache.PeekImage(img.id);
                if (image.image_uid != img.uid) {
                    tex_miss = TexUid;
                } else if (image.backing != img.backing) {
                    tex_miss = TexBacking;
                } else if (!texture_cache.ImageUpToDateRelaxed(img.id)) {
                    tex_miss = TexDirtyImage;
                } else if (image.binding.is_target || image.binding.force_general ||
                           image.binding.needs_rebind) {
                    tex_miss = TexFlags;
                }
                if (tex_miss >= 0) {
                    break;
                }
                const auto& backing = *image.backing;
                transit |= !(backing.state == img.state) || !backing.subresource_runs.empty();
            }
        }
    }
    const bool tex_would = tex_key && tex_miss < 0;
    const u64 t_checked = tsc();

    // Dry replay of the stored outputs into scratch, timed only: what a hit writes instead.
    u64 replay_buf_ticks = 0;
    u64 replay_tex_ticks = 0;
    if (timed && buf_would) {
        const u64 t0 = tsc();
        st.replay_buffers.clear();
        for (size_t i = 0; i < rec.bufs.size(); ++i) {
            const auto& b = rec.bufs[i];
            st.replay_buffers.emplace_back(b.handle, b.offset, b.range);
            st.replay_adjust[i] = b.adjust;
        }
        replay_buf_ticks = tsc() - t0;
    }
    if (timed && tex_would) {
        const u64 t0 = tsc();
        st.replay_images.clear();
        for (const auto& img : rec.imgs) {
            if (img.id) {
                static_cast<void>(texture_cache.GetImage(img.id)); // the LRU touch a hit keeps
            }
            st.replay_images.emplace_back(vk::Sampler{}, img.view, img.layout);
        }
        for (const auto sampler : rec.samplers) {
            st.replay_images.emplace_back(sampler, vk::ImageView{}, vk::ImageLayout::eGeneral);
        }
        replay_tex_ticks = tsc() - t0;
    }

    // The unchanged full path.
    const size_t buf_begin = buffer_infos.size();
    const size_t img_begin = image_infos.size();
    const u32 buf_binding = binding.buffer;
    bind_memo_active = true;
    bind_memo_timing = timed;
    bind_memo_fixed_ticks = 0;
    bind_memo_redirected = 0;
    bind_memo_uploaded.fill(0);
    const u64 t_full = tsc();
    BindBuffers(bound, binding, push_data);
    const u64 t_full_tex = tsc();
    BindTextures(bound, binding);
    const u64 t_done = tsc();
    bind_memo_active = false;
    bind_memo_timing = false;

    if (timed) {
        ++st.timed;
        st.t_empty += t_key - t_empty;
        st.t_key_buf += t_key_tex - t_key;
        st.t_key_tex += t_check_buf - t_key_tex;
        if (buf_key) {
            ++st.t_check_buf_n;
            st.t_check_buf += t_check_tex - t_check_buf;
        }
        if (tex_key) {
            ++st.t_check_tex_n;
            st.t_check_tex += t_checked - t_check_tex;
        }
        if (buf_would) {
            ++st.t_replay_buf_n;
            st.t_replay_buf += replay_buf_ticks;
            ++st.t_buf_hit_n;
            st.t_buf_hit += t_full_tex - t_full;
            st.t_fixed_hit += bind_memo_fixed_ticks;
        } else {
            ++st.t_buf_miss_n;
            st.t_buf_miss += t_full_tex - t_full;
        }
        if (tex_would) {
            ++st.t_replay_tex_n;
            st.t_replay_tex += replay_tex_ticks;
            ++st.t_tex_hit_n;
            st.t_tex_hit += t_done - t_full_tex;
        } else {
            ++st.t_tex_miss_n;
            st.t_tex_miss += t_done - t_full_tex;
        }
    }

    ++st.stages;
    ++st.type_stages[type];
    st.type_key[type] += buf_key && tex_key;
    st.type_would[type] += buf_would && tex_would;
    st.dma_stages += stage.uses_dma;
    st.buf_key += buf_key;
    st.buf_would += buf_would;
    st.tex_key += tex_key;
    st.tex_would += tex_would;
    st.transit_needed += tex_would && transit;
    if (buf_miss >= 0) {
        ++st.buf_miss[static_cast<u32>(buf_miss)];
    }
    if (tex_miss >= 0) {
        ++st.tex_miss[static_cast<u32>(tex_miss)];
    }
    // Per-slot repeats against the previous occurrence (the per-binding fallback's key).
    for (size_t i = 0; i < stage.buffers.size(); ++i) {
        if (!stage.buffers[i].IsSpecial()) {
            ++st.slot_buf;
            st.slot_buf_key += rec.valid && SameSlot(st.vsharps, rec.vsharps, i);
        }
    }
    for (size_t i = 0; i < st.tsharps.size(); ++i) {
        ++st.slot_img;
        st.slot_img_key += rec.valid && SameSlot(st.tsharps, rec.tsharps, i);
    }
    for (size_t i = 0; i < st.ssharps.size(); ++i) {
        ++st.slot_smp;
        st.slot_smp_key +=
            rec.valid && border == rec.border && SameSlot(st.ssharps, rec.ssharps, i);
    }

    // Soundness: on a predicted hit the full path must have produced exactly the stored outputs
    // and uploaded nothing. Anything else means the invalidation list is incomplete.
    if (buf_would) {
        if (rec.bufs.size() != stage.buffers.size()) {
            ++st.mis_buf;
        } else {
            for (size_t i = 0; i < rec.bufs.size(); ++i) {
                const auto& b = rec.bufs[i];
                if (!b.memo) {
                    continue;
                }
                const auto& info = buffer_infos[buf_begin + i];
                const auto buffer_id = std::get<0>(buffer_bindings[i]);
                st.mis_buf += buffer_id != b.id || info.buffer != b.handle ||
                              info.offset != b.offset || info.range != b.range ||
                              push_data.buf_offsets[buf_binding + i] != b.adjust;
                st.mis_upload += bind_memo_uploaded[i] != 0;
                ++st.memo_bufs_on_hit;
            }
        }
    }
    if (tex_would) {
        const size_t num_images = image_bindings.size();
        if (num_images != rec.imgs.size() || stage.samplers.size() != rec.samplers.size()) {
            ++st.mis_img;
        } else {
            for (size_t i = 0; i < num_images; ++i) {
                const auto& info = image_infos[img_begin + i];
                const auto& img = rec.imgs[i];
                st.mis_img += image_bindings[i].first != img.id || info.imageView != img.view;
                st.layout_diff += info.imageLayout != img.layout;
            }
            for (size_t i = 0; i < rec.samplers.size(); ++i) {
                st.mis_smp += image_infos[img_begin + num_images + i].sampler != rec.samplers[i];
            }
            st.samplers_on_hit += rec.samplers.size();
        }
    }

    // Record this occurrence (outside every timed span).
    rec.valid = true;
    rec.vsharps = st.vsharps;
    rec.tsharps = st.tsharps;
    rec.ssharps = st.ssharps;
    rec.border = border;
    rec.buf_gen = buf_gen;
    rec.buf_tex_gen = tex_gen;
    rec.tex_gen = tex_gen;
    rec.smp_gen = smp_gen;
    rec.buf_excluded = false;
    rec.buf_texel = false;
    rec.bufs.resize(stage.buffers.size());
    for (size_t i = 0; i < stage.buffers.size(); ++i) {
        const auto& desc = stage.buffers[i];
        const auto& [buffer_id, vsharp, size] = buffer_bindings[i];
        const auto& info = buffer_infos[buf_begin + i];
        auto& b = rec.bufs[i];
        b.id = buffer_id;
        b.addr = vsharp.base_address;
        b.size = size;
        b.handle = info.buffer;
        b.offset = info.offset;
        b.range = info.range;
        b.adjust = push_data.buf_offsets[buf_binding + i];
        b.memo = !desc.IsSpecial() && !desc.is_written;
        if (b.memo && buffer_id) {
            // Stream copies and host mirrors are not the cache buffer; a texel range aliasing
            // an image is re-tiled on every bind (SynchronizeBufferFromImage).
            if (info.buffer != buffer_cache.GetBuffer(buffer_id).Handle()) {
                rec.buf_excluded = true;
            }
            if (desc.is_formatted) {
                rec.buf_texel = true;
                if (texture_cache.FindImageFromRange(vsharp.base_address, size)) {
                    rec.buf_excluded = true;
                }
            }
        }
    }
    rec.tex_excluded = native_bindings != nullptr;
    for (const auto& desc : stage.images) {
        rec.tex_excluded |= desc.mip_fallback_mode != Shader::MipStorageFallbackMode::None;
    }
    rec.imgs.resize(image_bindings.size());
    for (size_t i = 0; i < image_bindings.size(); ++i) {
        const auto& [image_id, desc] = image_bindings[i];
        const auto& info = image_infos[img_begin + i];
        auto& img = rec.imgs[i];
        img = {};
        img.id = image_id;
        img.view = info.imageView;
        img.layout = info.imageLayout;
        if (!image_id) {
            continue;
        }
        auto& image = texture_cache.PeekImage(image_id);
        img.uid = image.image_uid;
        img.backing = image.backing;
        img.state = image.backing->state;
        // Storage images, depth/stencil redirects and the Pro ID buffer stay on the full path.
        rec.tex_excluded |= desc.type == VideoCore::TextureCache::BindingType::Storage ||
                            ((bind_memo_redirected >> (i & 63)) & 1) != 0 ||
                            (!stencil_scratches.empty() && IsStencilScratch(image_id)) ||
                            (image.info.num_samples == 2 &&
                             image.info.pixel_format == vk::Format::eR32Uint);
    }
    rec.samplers.resize(stage.samplers.size());
    for (size_t i = 0; i < stage.samplers.size(); ++i) {
        rec.samplers[i] = image_infos[img_begin + image_bindings.size() + i].sampler;
    }
    return u8(buf_key && tex_key) | u8((buf_would && tex_would) << 1);
}

void Rasterizer::BindMemoEndDraw(bool is_compute, u8 draw_bits) {
    auto& st = GetBindMemoStats();
    ++st.binds;
    if (!is_compute) {
        ++st.draws;
        st.draw_key += (draw_bits & 1) != 0;
        st.draw_would += (draw_bits & 2) != 0;
    }
    if (st.binds < (u64{1} << 18)) {
        return;
    }
    const u64 tsc_now = Common::FencedRDTSC();
    const auto clock_now = std::chrono::steady_clock::now();
    const double wall_ns = double(
        std::chrono::duration_cast<std::chrono::nanoseconds>(clock_now - st.clock0).count());
    const double ns_per_tick = tsc_now > st.tsc0 ? wall_ns / double(tsc_now - st.tsc0) : 0.0;
    const u64 frames_now = Common::FaultMonitor::presented_frames.load(std::memory_order_relaxed);
    const u64 frames = frames_now - st.frames0;
    const double per_frame = frames ? 1.0 / double(frames) : 0.0;
    const u64 buf_gen = buffer_cache.StructGeneration();
    const u64 tex_gen = texture_cache.StructGeneration();
    const u64 smp_gen = texture_cache.SamplerGeneration();
    const auto pct = [](u64 part, u64 whole) {
        return whole ? 100.0 * double(part) / double(whole) : 0.0;
    };
    const double empty_ns = st.timed ? double(st.t_empty) * ns_per_tick / double(st.timed) : 0.0;
    const auto avg_ns = [ns_per_tick, empty_ns](u64 ticks, u64 n) {
        return n ? std::max(0.0, double(ticks) * ns_per_tick / double(n) - empty_ns) : 0.0;
    };
    const double full_buf_hit = avg_ns(st.t_buf_hit, st.t_buf_hit_n);
    // Summed per always-full binding inside BindBuffers, so no single-span correction here.
    const double fixed_buf_hit =
        st.t_buf_hit_n ? double(st.t_fixed_hit) * ns_per_tick / double(st.t_buf_hit_n) : 0.0;
    const double full_buf_miss = avg_ns(st.t_buf_miss, st.t_buf_miss_n);
    const double full_tex_hit = avg_ns(st.t_tex_hit, st.t_tex_hit_n);
    const double full_tex_miss = avg_ns(st.t_tex_miss, st.t_tex_miss_n);
    const double key_buf = avg_ns(st.t_key_buf, st.timed);
    const double key_tex = avg_ns(st.t_key_tex, st.timed);
    const double check_buf = avg_ns(st.t_check_buf, st.t_check_buf_n);
    const double check_tex = avg_ns(st.t_check_tex, st.t_check_tex_n);
    const double replay_buf = avg_ns(st.t_replay_buf, st.t_replay_buf_n);
    const double replay_tex = avg_ns(st.t_replay_tex, st.t_replay_tex_n);
    // Per predicted hit: the half's full path minus what a hit still runs (always-full buffer
    // bindings, the checks, the replay, the unreplayed allowances).
    const double allow_buf =
        st.buf_would ? BindMemoAllowBufNs * double(st.memo_bufs_on_hit) / double(st.buf_would)
                     : 0.0;
    const double allow_tex =
        st.tex_would ? BindMemoAllowSamplerNs * double(st.samplers_on_hit) / double(st.tex_would)
                     : 0.0;
    const double save_buf_hit = full_buf_hit - fixed_buf_hit - check_buf - replay_buf - allow_buf;
    const double save_tex_hit = full_tex_hit - check_tex - replay_tex - allow_tex;
    // Paid on every stage (key), on every key hit that fails validation (check) and on every
    // miss (record update, costed like a replay).
    const double gross_buf = double(st.buf_would) * save_buf_hit;
    const double gross_tex = double(st.tex_would) * save_tex_hit;
    const double cost_buf = double(st.stages) * key_buf +
                            double(st.buf_key - st.buf_would) * check_buf +
                            double(st.stages - st.buf_would) * replay_buf;
    const double cost_tex = double(st.stages) * key_tex +
                            double(st.tex_key - st.tex_would) * check_tex +
                            double(st.stages - st.tex_would) * replay_tex;
    const double net_buf_us = (gross_buf - cost_buf) * per_frame / 1000.0;
    const double net_tex_us = (gross_tex - cost_tex) * per_frame / 1000.0;
    const u64 mismatches = st.mis_buf + st.mis_upload + st.mis_img + st.mis_smp;

    LOG_INFO(Render_Vulkan,
             "BindMemoStats: binds={} draws={} frames={} stages={} "
             "stage_key(vs/ps/cs/other)={:.1f}/{:.1f}/{:.1f}/{:.1f}% "
             "stage_would(vs/ps/cs/other)={:.1f}/{:.1f}/{:.1f}/{:.1f}% "
             "whole_draw(key/would)={:.1f}/{:.1f}% slot_key(buf/img/smp)={:.1f}/{:.1f}/{:.1f}% "
             "dma_stages={}",
             st.binds, st.draws, frames, st.stages, pct(st.type_key[0], st.type_stages[0]),
             pct(st.type_key[1], st.type_stages[1]), pct(st.type_key[2], st.type_stages[2]),
             pct(st.type_key[3], st.type_stages[3]), pct(st.type_would[0], st.type_stages[0]),
             pct(st.type_would[1], st.type_stages[1]), pct(st.type_would[2], st.type_stages[2]),
             pct(st.type_would[3], st.type_stages[3]), pct(st.draw_key, st.draws),
             pct(st.draw_would, st.draws), pct(st.slot_buf_key, st.slot_buf),
             pct(st.slot_img_key, st.slot_img), pct(st.slot_smp_key, st.slot_smp),
             st.dma_stages);
    LOG_INFO(Render_Vulkan,
             "BindMemoStats: buf key={:.1f}% would={:.1f}% "
             "miss(excluded/buf_gen/tex_gen/dirty_page)={}/{}/{}/{} tex key={:.1f}% "
             "would={:.1f}% transit_needed={} "
             "miss(excluded/tex_gen/smp_gen/uid/backing/dirty_image/flags)={}/{}/{}/{}/{}/{}/{} "
             "mismatches={} (buf={} upload={} img={} smp={}) layout_diff={} "
             "gens/frame(buf/tex/smp)={:.2f}/{:.3f}/{:.3f}",
             pct(st.buf_key, st.stages), pct(st.buf_would, st.stages), st.buf_miss[BufExcluded],
             st.buf_miss[BufGen], st.buf_miss[BufTexGen], st.buf_miss[BufDirtyPage],
             pct(st.tex_key, st.stages), pct(st.tex_would, st.stages), st.transit_needed,
             st.tex_miss[TexExcluded], st.tex_miss[TexGen], st.tex_miss[TexSamplerGen],
             st.tex_miss[TexUid], st.tex_miss[TexBacking], st.tex_miss[TexDirtyImage],
             st.tex_miss[TexFlags], mismatches, st.mis_buf, st.mis_upload, st.mis_img,
             st.mis_smp, st.layout_diff, double(buf_gen - st.buf_gen0) * per_frame,
             double(tex_gen - st.tex_gen0) * per_frame, double(smp_gen - st.smp_gen0) * per_frame);
    LOG_INFO(Render_Vulkan,
             "BindMemoCost: timed={} tsc_read_ns={:.1f} ns full_buf(hit/miss)={:.0f}/{:.0f} "
             "fixed_buf_hit={:.0f} "
             "full_tex(hit/miss)={:.0f}/{:.0f} key(buf/tex)={:.1f}/{:.1f} "
             "check(buf/tex)={:.1f}/{:.1f} replay(buf/tex)={:.1f}/{:.1f} "
             "save_per_hit(buf/tex)={:.0f}/{:.0f} gross_us_frame(buf/tex)={:.0f}/{:.0f} "
             "est_save_us_frame(buf/tex/total)={:.0f}/{:.0f}/{:.0f}",
             st.timed, empty_ns, full_buf_hit, full_buf_miss, fixed_buf_hit, full_tex_hit,
             full_tex_miss,
             key_buf, key_tex, check_buf, check_tex, replay_buf, replay_tex, save_buf_hit,
             save_tex_hit, gross_buf * per_frame / 1000.0, gross_tex * per_frame / 1000.0,
             net_buf_us, net_tex_us, net_buf_us + net_tex_us);

    // Cumulative-and-reset: the next window starts here. Records are kept.
    auto records = std::move(st.records);
    const u64 stage_counter = st.stage_counter;
    st = {};
    st.records = std::move(records);
    st.stage_counter = stage_counter;
    st.started = true;
    st.frames0 = frames_now;
    st.buf_gen0 = buf_gen;
    st.tex_gen0 = tex_gen;
    st.smp_gen0 = smp_gen;
    st.clock0 = clock_now;
    st.tsc0 = tsc_now;
}

void Rasterizer::EmitPipelineResources(const Pipeline* pipeline) {
    if (!native_bindings) {
        pipeline->BindResources(set_writes, buffer_barriers, push_data);
        return;
    }
    // The mode is global, so every guest pipeline here was created with heap mappings. A
    // count mismatch would mean the draw-time binding walk disagrees with pipeline creation.
    ASSERT_MSG(pipeline->UsesNativeBindings() &&
                   native_bound_count == pipeline->NativeBindingCount(),
               "Native binding count {} does not match pipeline {}", native_bound_count,
               pipeline->NativeBindingCount());
    const std::span<const u32> indices{native_indices.data(), native_bound_count};
    if (native_bindings->IsVerifying()) {
        native_bindings->Verify(set_writes, indices);
    }
    pipeline->BindResourcesNative(*native_bindings, indices, buffer_barriers, push_data);
}

bool Rasterizer::IsComputeMetaClear(const Pipeline* pipeline) {
    if (!pipeline->IsCompute()) {
        return false;
    }

    // Most of the time when a metadata is updated with a shader it gets cleared. It means
    // we can skip the whole dispatch and update the tracked state instead. Also, it is not
    // intended to be consumed and in such rare cases (e.g. HTile introspection, CRAA) we
    // will need its full emulation anyways.
    const auto& info = pipeline->GetStage(Shader::LogicalStage::Compute);
    const auto& bound = GetBoundStage(Shader::LogicalStage::Compute);

    // Assume if a shader reads metadata, it is a copy shader.
    for (const auto& desc : info.buffers) {
        const VAddr address = desc.GetSharp(bound).base_address;
        if (!desc.IsSpecial() && !desc.is_written && texture_cache.IsMeta(address)) {
            return false;
        }
    }

    // Metadata surfaces are tiled and thus need address calculation to be written properly.
    // If a shader wants to encode HTILE, for example, from a depth image it will have to compute
    // proper tile address from dispatch invocation id. This address calculation contains an xor
    // operation so use it as a heuristic for metadata writes that are probably not clears.
    if (!info.has_bitwise_xor) {
        // Assume if a shader writes metadata without address calculation, it is a clear shader.
        for (const auto& desc : info.buffers) {
            const VAddr address = desc.GetSharp(bound).base_address;
            if (!desc.IsSpecial() && desc.is_written && texture_cache.ClearMeta(address)) {
                // Assume all slices were updates
                LOG_TRACE(Render_Vulkan, "Metadata update skipped");
                return true;
            }
        }
    }
    return false;
}

bool Rasterizer::IsComputeImageCopy(const Pipeline* pipeline) {
    if (!pipeline->IsCompute()) {
        return false;
    }

    // Ensure shader only has 2 bound buffers
    const auto& cs_pgm = liverpool->BackCsRegs();
    const auto& info = pipeline->GetStage(Shader::LogicalStage::Compute);
    const auto& bound = GetBoundStage(Shader::LogicalStage::Compute);
    if (cs_pgm.num_thread_x.full != 64 || info.buffers.size() != 2 || !info.images.empty()) {
        return false;
    }

    // Those 2 buffers must both be formatted. One must be source and another destination.
    const auto& desc0 = info.buffers[0];
    const auto& desc1 = info.buffers[1];
    if (!desc0.is_formatted || !desc1.is_formatted || desc0.is_written == desc1.is_written) {
        return false;
    }

    // Buffers must have the same size and each thread of the dispatch must copy 1 dword of data
    const AmdGpu::Buffer buf0 = desc0.GetSharp(bound);
    const AmdGpu::Buffer buf1 = desc1.GetSharp(bound);
    if (buf0.GetSize() != buf1.GetSize() || cs_pgm.dim_x != (buf0.GetSize() / 256)) {
        return false;
    }

    // Find images the buffer alias
    const auto image0_id = texture_cache.FindImageFromRange(buf0.base_address, buf0.GetSize());
    if (!image0_id) {
        return false;
    }
    const auto image1_id =
        texture_cache.FindImageFromRange(buf1.base_address, buf1.GetSize(), false);
    if (!image1_id) {
        return false;
    }

    // Image copy must be valid
    VideoCore::Image& image0 = texture_cache.GetImage(image0_id);
    VideoCore::Image& image1 = texture_cache.GetImage(image1_id);
    if (image0.info.guest_size != image1.info.guest_size ||
        image0.info.pitch != image1.info.pitch || image0.info.guest_size != buf0.GetSize() ||
        image0.info.num_bits != image1.info.num_bits) {
        return false;
    }

    // Perform image copy
    VideoCore::Image& src_image = desc0.is_written ? image1 : image0;
    VideoCore::Image& dst_image = desc0.is_written ? image0 : image1;
    if (instance.IsMaintenance8Supported() ||
        src_image.info.props.is_depth == dst_image.info.props.is_depth) {
        dst_image.CopyImage(src_image);
    } else {
        const auto& copy_buffer =
            buffer_cache.GetUtilityBuffer(VideoCore::MemoryUsage::DeviceLocal);
        dst_image.CopyImageWithBuffer(src_image, copy_buffer.Handle(), 0);
    }
    dst_image.flags |= VideoCore::ImageFlagBits::GpuModified;
    dst_image.flags &= ~VideoCore::ImageFlagBits::Dirty;
    return true;
}

bool Rasterizer::IsComputeImageClear(const Pipeline* pipeline) {
    if (!pipeline->IsCompute()) {
        return false;
    }

    // Ensure shader only has 2 bound buffers
    const auto& cs_pgm = liverpool->BackCsRegs();
    const auto& info = pipeline->GetStage(Shader::LogicalStage::Compute);
    const auto& bound = GetBoundStage(Shader::LogicalStage::Compute);
    if (cs_pgm.num_thread_x.full != 64 || info.buffers.size() != 2 || !info.images.empty()) {
        return false;
    }

    // From those 2 buffers, first must hold the clear vector and second the image being cleared
    const auto& desc0 = info.buffers[0];
    const auto& desc1 = info.buffers[1];
    if (desc0.is_formatted || !desc1.is_formatted || desc0.is_written || !desc1.is_written) {
        return false;
    }

    // First buffer must have size of vec4 and second the size of a single layer
    const AmdGpu::Buffer buf0 = desc0.GetSharp(bound);
    const AmdGpu::Buffer buf1 = desc1.GetSharp(bound);
    const u32 buf1_bpp = AmdGpu::NumBitsPerBlock(buf1.GetDataFmt());
    if (buf0.GetSize() != 16 || (cs_pgm.dim_x * 128ULL * (buf1_bpp / 8)) != buf1.GetSize()) {
        return false;
    }

    // Find image the buffer alias
    const auto image1_id =
        texture_cache.FindImageFromRange(buf1.base_address, buf1.GetSize(), false);
    if (!image1_id) {
        return false;
    }

    // Image clear must be valid
    VideoCore::Image& image1 = texture_cache.GetImage(image1_id);
    if (image1.info.guest_size != buf1.GetSize() || image1.info.num_bits != buf1_bpp ||
        image1.info.props.is_depth) {
        return false;
    }

    // Perform image clear
    const float* values = reinterpret_cast<float*>(buf0.base_address);
    const vk::ClearValue clear = {
        .color = {.float32 = std::array<float, 4>{values[0], values[1], values[2], values[3]}},
    };
    const VideoCore::SubresourceRange range = {
        .base =
            {
                .level = 0,
                .layer = 0,
            },
        .extent = image1.info.resources,
    };
    image1.Clear(clear, range);
    image1.flags |= VideoCore::ImageFlagBits::GpuModified;
    image1.flags &= ~VideoCore::ImageFlagBits::Dirty;
    return true;
}

void Rasterizer::BindBuffers(const Shader::BoundStage& bound, Shader::Backend::Bindings& binding,
                             Shader::PushData& push_data) {
    const auto& stage = *bound.info;
    buffer_bindings.clear();

    // Pass 1 (FindBuffer) vs pass 2 (ObtainBuffer -> SynchronizeBuffer) timed separately: this
    // function is 62% of per-draw cost and a stack scan is not trustworthy for apportioning it.
    const auto t_find = Common::FaultMonitor::Now();
    for (const auto& desc : stage.buffers) {
        const auto vsharp = desc.GetSharp(bound);
        if (!desc.IsSpecial() && vsharp.base_address != 0 && vsharp.GetSize() > 0) {
            const u64 size = memory->ClampRangeSize(vsharp.base_address, vsharp.GetSize());
            const auto buffer_id = buffer_cache.FindBuffer(vsharp.base_address, size);
            buffer_bindings.emplace_back(buffer_id, vsharp, size);
        } else {
            buffer_bindings.emplace_back(VideoCore::BufferId{}, vsharp, 0);
        }
    }
    const u64 find_ns = Common::FaultMonitor::SinceNs(t_find);
    u64 obtain_ns = 0;

    // Second pass to re-bind buffers that were updated after binding
    for (u32 i = 0; i < buffer_bindings.size(); i++) {
        const auto& [buffer_id, vsharp, size] = buffer_bindings[i];
        const auto& desc = stage.buffers[i];
        // SHADPS4_BIND_MEMO_STATS: bindings a memo hit would still run in full are timed.
        const bool memo_fixed = bind_memo_timing && (desc.IsSpecial() || desc.is_written);
        const u64 memo_t0 = memo_fixed ? Common::FencedRDTSC() : 0;
        const bool is_storage = desc.IsStorage(vsharp);
        const u32 alignment =
            is_storage ? instance.StorageMinAlignment() : instance.UniformMinAlignment();
        // The buffer object behind this binding's legacy payload; null for null descriptors.
        const VideoCore::Buffer* bound_buffer = nullptr;
        // Buffer is not from the cache, either a special buffer or unbound.
        if (!buffer_id) {
            if (desc.buffer_type == Shader::BufferType::GdsBuffer) {
                const auto* gds_buf = buffer_cache.GetGdsBuffer();
                buffer_infos.emplace_back(gds_buf->Handle(), 0, gds_buf->SizeBytes());
                bound_buffer = gds_buf;
            } else if (desc.buffer_type == Shader::BufferType::Flatbuf) {
                auto& vk_buffer = buffer_cache.GetUtilityBuffer(VideoCore::MemoryUsage::Stream);
                const u32 ubo_size = bound.flattened_ud_buf.size() * sizeof(u32);
                const u64 offset =
                    vk_buffer.Copy(bound.flattened_ud_buf.data(), ubo_size, alignment);
                buffer_infos.emplace_back(vk_buffer.Handle(), offset, ubo_size);
                bound_buffer = &vk_buffer;
            } else if (desc.buffer_type == Shader::BufferType::ClipPlanes) {
                // Permutations compiled without enabled planes never read the buffer, so the
                // declared binding is satisfied with a null descriptor instead of a copy.
                if (liverpool->BackRegs().clipper_control.user_clip_plane_enable == 0) {
                    buffer_infos.emplace_back(VK_NULL_HANDLE, 0, VK_WHOLE_SIZE);
                } else {
                    auto& vk_buffer = buffer_cache.GetUtilityBuffer(VideoCore::MemoryUsage::Stream);
                    std::array<float, AmdGpu::NUM_CLIP_PLANES * 4> planes{};
                    for (u32 i = 0; i < AmdGpu::NUM_CLIP_PLANES; ++i) {
                        const auto& plane = liverpool->BackRegs().clip_user_data[i];
                        planes[i * 4 + 0] = std::bit_cast<float>(plane.data_x);
                        planes[i * 4 + 1] = std::bit_cast<float>(plane.data_y);
                        planes[i * 4 + 2] = std::bit_cast<float>(plane.data_z);
                        planes[i * 4 + 3] = std::bit_cast<float>(plane.data_w);
                    }
                    const u32 ubo_size = static_cast<u32>(sizeof(planes));
                    const u64 offset = vk_buffer.Copy(planes.data(), ubo_size, alignment);
                    buffer_infos.emplace_back(vk_buffer.Handle(), offset, ubo_size);
                    bound_buffer = &vk_buffer;
                }
            } else if (desc.buffer_type == Shader::BufferType::BdaPagetable) {
                const auto* bda_buffer = buffer_cache.GetBdaPageTableBuffer();
                buffer_infos.emplace_back(bda_buffer->Handle(), 0, bda_buffer->SizeBytes());
                bound_buffer = bda_buffer;
            } else if (desc.buffer_type == Shader::BufferType::FaultBuffer) {
                const auto* fault_buffer = buffer_cache.GetFaultBuffer();
                buffer_infos.emplace_back(fault_buffer->Handle(), 0, fault_buffer->SizeBytes());
                bound_buffer = fault_buffer;
            } else if (desc.buffer_type == Shader::BufferType::SharedMemory) {
                auto& lds_buffer = buffer_cache.GetUtilityBuffer(VideoCore::MemoryUsage::Stream);
                const auto& cs_program = liverpool->BackCsRegs();
                const auto lds_size = cs_program.SharedMemSize() * cs_program.NumWorkgroups();
                const auto [data, offset] = lds_buffer.Map(lds_size, alignment);
                std::memset(data, 0, lds_size);
                buffer_infos.emplace_back(lds_buffer.Handle(), offset, lds_size);
                bound_buffer = &lds_buffer;
            } else {
                buffer_infos.emplace_back(VK_NULL_HANDLE, 0, VK_WHOLE_SIZE);
            }
        } else {
            const auto t_obtain = Common::FaultMonitor::Now();
            const u64 memo_uploads = bind_memo_active ? buffer_cache.SyncUploadBytesTotal() : 0;
            const auto [vk_buffer, offset] = buffer_cache.ObtainBuffer(
                vsharp.base_address, size, desc.is_written, desc.is_formatted, buffer_id,
                is_storage ? Common::BufferSampleCaller::Storage
                           : Common::BufferSampleCaller::Uniform);
            obtain_ns += Common::FaultMonitor::SinceNs(t_obtain);
            if (bind_memo_active) {
                bind_memo_uploaded[i] = buffer_cache.SyncUploadBytesTotal() != memo_uploads;
            }
            const u32 offset_aligned = Common::AlignDown(offset, alignment);
            const u32 adjust = offset - offset_aligned;
            ASSERT(adjust % 4 == 0);
            push_data.AddOffset(binding.buffer, adjust);
            buffer_infos.emplace_back(vk_buffer->Handle(), offset_aligned, size + adjust);
            bound_buffer = vk_buffer;
            if (auto barrier =
                    vk_buffer->GetBarrier(desc.is_written ? vk::AccessFlagBits2::eShaderWrite
                                                          : vk::AccessFlagBits2::eShaderRead,
                                          vk::PipelineStageFlagBits2::eAllCommands)) {
                buffer_barriers.emplace_back(*barrier);
            }
            if (desc.is_written && VideoCore::TextureCache::BufferImageAliasDiag()) {
                texture_cache.DiagBufferWriteOverImages(vsharp.base_address, size,
                                                        desc.is_formatted);
            }
            // SHADPS4_RAW_WRITE_INVALIDATE=1 (port of upstream #5092): raw storage-buffer writes
            // can also make an aliased cached image stale, not only formatted ones.
            static const bool raw_write_invalidate = [] {
                const char* value = std::getenv("SHADPS4_RAW_WRITE_INVALIDATE");
                return value && value[0] == '1';
            }();
            if (desc.is_written && (desc.is_formatted || raw_write_invalidate)) {
                texture_cache.InvalidateMemoryFromGPU(vsharp.base_address, size);
            }
        }

        const u32 unified_binding = binding.unified++;
        if (native_bindings) {
            // Encode exactly the handle generation, offset and range recorded just above. A
            // later merge leaves this payload alone in the legacy path, and so does this.
            const auto& info = buffer_infos.back();
            native_indices[unified_binding] = native_bindings->BufferIndex(
                info.buffer, bound_buffer ? bound_buffer->BufferDeviceAddress() : 0, info.offset,
                info.range, is_storage);
        }
        if (build_set_writes) {
            auto& set_write = set_writes[set_write_index++];
            set_write.dstSet = VK_NULL_HANDLE;
            set_write.dstBinding = unified_binding;
            set_write.dstArrayElement = 0;
            set_write.descriptorCount = 1;
            set_write.descriptorType = is_storage ? vk::DescriptorType::eStorageBuffer
                                                  : vk::DescriptorType::eUniformBuffer;
            set_write.pBufferInfo = &buffer_infos.back();
        }
        ++binding.buffer;
        if (memo_fixed) {
            bind_memo_fixed_ticks += Common::FencedRDTSC() - memo_t0;
        }
    }
    Common::FaultMonitor::CountBindBufferPass(find_ns, obtain_ns);
}

VideoCore::ImageId Rasterizer::StencilScratchFor(VideoCore::ImageId depth_id,
                                                 const VideoCore::ImageInfo& info) {
    const u64 depth_uid = texture_cache.GetImage(depth_id).image_uid;
    for (auto& entry : stencil_scratches) {
        if (entry.depth_id == depth_id && entry.depth_uid == depth_uid) {
            return entry.scratch_id;
        }
    }
    // The depth image can be recreated at the same address (same surface): reuse the scratch.
    for (auto& entry : stencil_scratches) {
        const auto& scratch = texture_cache.GetImage(entry.scratch_id);
        if (scratch.info.guest_address == info.guest_address &&
            scratch.info.size == info.size && scratch.info.pixel_format == info.pixel_format) {
            entry.depth_id = depth_id;
            entry.depth_uid = depth_uid;
            return entry.scratch_id;
        }
    }
    const VideoCore::ImageId scratch_id = texture_cache.InsertScratchImage(info);
    stencil_scratches.push_back({depth_id, depth_uid, scratch_id});
    const auto& depth = texture_cache.GetImage(depth_id);
    const auto& scratch = texture_cache.GetImage(scratch_id);
    LOG_WARNING(Render_Vulkan,
                "StencilStorage: scratch {} ({}x{} {}) for depth {:#x} ({}x{} {}, {} samples)",
                scratch_id.index, scratch.info.size.width, scratch.info.size.height,
                vk::to_string(scratch.info.pixel_format), depth.info.guest_address,
                depth.info.size.width, depth.info.size.height,
                vk::to_string(depth.info.pixel_format), depth.info.num_samples);
    return scratch_id;
}

bool Rasterizer::IsStencilScratch(VideoCore::ImageId id) const {
    return std::ranges::any_of(stencil_scratches,
                               [id](const auto& entry) { return entry.scratch_id == id; });
}

void Rasterizer::CopyStencilPlane(VideoCore::ImageId src_id, VideoCore::ImageId dst_id) {
    auto& src = texture_cache.GetImage(src_id);
    auto& dst = texture_cache.GetImage(dst_id);
    const auto aspect_of = [](const VideoCore::Image& image) {
        return (image.aspect_mask & vk::ImageAspectFlagBits::eStencil)
                   ? vk::ImageAspectFlagBits::eStencil
                   : vk::ImageAspectFlagBits::eColor;
    };
    const vk::ImageCopy region{
        .srcSubresource = {aspect_of(src), 0, 0, 1},
        .dstSubresource = {aspect_of(dst), 0, 0, 1},
        .extent = {std::min(src.info.size.width, dst.info.size.width),
                   std::min(src.info.size.height, dst.info.size.height), 1},
    };
    scheduler.EndRendering();
    src.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, {});
    dst.Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite, {});
    auto command_access = scheduler.AcquireCommandBuffer();
    command_access.Handle().copyImage(src.GetImage(), src.backing->state.layout, dst.GetImage(),
                                      dst.backing->state.layout, region);
}

void Rasterizer::FlushStencilWritebacks() {
    for (const auto& [scratch_id, depth_id] : pending_stencil_writebacks) {
        CopyStencilPlane(scratch_id, depth_id);
    }
    pending_stencil_writebacks.clear();
}

void Rasterizer::BindTextures(const Shader::BoundStage& bound,
                              Shader::Backend::Bindings& binding) {
    const auto& stage = *bound.info;
    u32 lookup_count{};
    image_bindings.clear();
    const u32 first_image_idx = image_infos.size();
    // For loading/storing to explicit mip levels, when no native instruction support, bind an array
    // of descriptors consecutively, 1 for each mip level. The shader can index this with LOD
    // operand.
    // This array holds the size of each consecutive array with the number of bindings consumed.
    // This is currently always 1 for anything other than mip fallback arrays.
    boost::container::small_vector<u32, 8> image_descriptor_array_sizes;

    const auto t_findimage = Common::FaultMonitor::Now();

    static const bool cache_decoded_images = [] {
        const char* value = std::getenv("SHADPS4_DECODE_IMAGE_CACHE");
        return !value || value[0] != '0';
    }();
    static const bool verify_decoded_images = [] {
        const char* value = std::getenv("SHADPS4_DECODE_IMAGE_VERIFY");
        return value && value[0] == '1';
    }();
    for (size_t image_slot = 0; image_slot < stage.images.size(); ++image_slot) {
        const auto& image_desc = stage.images[image_slot];
        const auto tsharp = image_desc.GetSharp(bound);
        if (texture_cache.IsMeta(tsharp.Address())) {
            LOG_WARNING(Render_Vulkan, "Unexpected metadata read by a shader (texture)");
        }

        if (tsharp.Address() == 0 || tsharp.GetDataFmt() == AmdGpu::DataFormat::FormatInvalid) {
            image_bindings.emplace_back(std::piecewise_construct, std::tuple{}, std::tuple{});
            image_descriptor_array_sizes.push_back(1);
            continue;
        }

        const Shader::MipStorageFallbackMode mip_fallback_mode = image_desc.mip_fallback_mode;
        const u32 num_bindings = image_desc.NumBindings(bound);

        const VideoCore::TextureCache::ImageDesc* decoded = nullptr;
        if (cache_decoded_images) {
            auto& entry = decoded_image_descs[static_cast<u32>(stage.l_stage)][image_slot];
            // These are the only ImageResource fields read by the ImageInfo and
            // ImageViewInfo constructors. Neo mode is fixed for the guest process.
            // Cache only the pure descriptor decode, before FindImage modifies its
            // view range. Image IDs, contents, bindings and barriers remain live.
            const u8 flags =
                image_desc.is_depth | (image_desc.is_array << 1) | (image_desc.is_written << 2);
            const bool hit = entry.valid && entry.flags == flags &&
                             std::memcmp(&entry.sharp, &tsharp, sizeof(tsharp)) == 0;
            if (!hit) {
                entry.sharp = tsharp;
                entry.flags = flags;
                entry.value = VideoCore::TextureCache::ImageDesc{tsharp, image_desc};
                entry.valid = true;
            }
            decoded = &entry.value;
            if (verify_decoded_images) {
                const VideoCore::TextureCache::ImageDesc reference{tsharp, image_desc};
                const auto& a = entry.value.info;
                const auto& b = reference.info;
                ASSERT(entry.value.type == reference.type &&
                       entry.value.view_info == reference.view_info &&
                       a.pixel_format == b.pixel_format && a.type == b.type &&
                       a.resources == b.resources && a.size == b.size && a.num_bits == b.num_bits &&
                       a.num_samples == b.num_samples && a.pitch == b.pitch &&
                       a.tile_mode == b.tile_mode && a.array_mode == b.array_mode &&
                       a.guest_address == b.guest_address && a.guest_size == b.guest_size &&
                       a.bank_swizzle == b.bank_swizzle && a.alt_tile == b.alt_tile &&
                       a.props.is_depth == b.props.is_depth &&
                       a.props.is_block == b.props.is_block && a.props.is_pow2 == b.props.is_pow2 &&
                       a.props.is_volume == b.props.is_volume &&
                       a.props.is_tiled == b.props.is_tiled);
                ASSERT(std::memcmp(a.mips_layout.data(), b.mips_layout.data(),
                                   a.resources.levels * sizeof(VideoCore::ImageInfo::MipInfo)) ==
                       0);
                static u64 calls{}, hits{};
                hits += hit;
                if (++calls >= 500000) {
                    LOG_INFO(Render_Vulkan, "DecodeImageVerify: calls={} hits={} ({:.1f}%)", calls,
                             hits, 100.0 * hits / calls);
                    calls = hits = 0;
                }
            }
        }

        for (auto i = 0; i < num_bindings; i++) {
            auto& [image_id, desc] =
                decoded ? image_bindings.emplace_back(VideoCore::ImageId{}, *decoded)
                        : image_bindings.emplace_back(std::piecewise_construct, std::tuple{},
                                                      std::tuple{tsharp, image_desc});

            if (mip_fallback_mode == Shader::MipStorageFallbackMode::ConstantIndex) {
                ASSERT(num_bindings == 1);
                desc.view_info.range.base.level += image_desc.constant_mip_index;
                desc.view_info.range.extent.levels = 1;
            } else if (mip_fallback_mode == Shader::MipStorageFallbackMode::DynamicIndex) {
                desc.view_info.range.base.level += i;
                desc.view_info.range.extent.levels = 1;
            }

            image_id = texture_cache.FindImage(desc);
            ++lookup_count;
            auto* image = &texture_cache.GetImage(image_id);
            {
                // Diagnostic SHADPS4_TRACE_SHADER=<hex pgm hash>: log texture bindings of that
                // shader (first 16 binds): requested T# vs the cache image it resolved to.
                static const std::vector<u64> trace_hashes = [] {
                    std::vector<u64> out;
                    if (const char* value = std::getenv("SHADPS4_TRACE_SHADER")) {
                        char* cursor = const_cast<char*>(value);
                        while (*cursor) {
                            out.push_back(std::strtoull(cursor, &cursor, 16));
                            if (*cursor == ',') {
                                ++cursor;
                            } else {
                                break;
                            }
                        }
                    }
                    return out;
                }();
                static u32 traced = 0;
                const u64 trace_hash = stage.pgm_hash;
                if (!trace_hashes.empty() && traced < 48 &&
                    std::ranges::find(trace_hashes, trace_hash) != trace_hashes.end()) {
                    ++traced;
                    LOG_WARNING(Render_Vulkan,
                                "TraceShader {:#x}: slot {} type {} T# {:#x} {}x{} {} -> image "
                                "{:#x} {}x{} {} samples {} uid {} flags {:#x}",
                                trace_hash, image_slot, static_cast<u32>(desc.type),
                                desc.info.guest_address, desc.info.size.width,
                                desc.info.size.height, vk::to_string(desc.info.pixel_format),
                                image->info.guest_address, image->info.size.width,
                                image->info.size.height, vk::to_string(image->info.pixel_format),
                                image->info.num_samples, image->image_uid,
                                static_cast<u32>(image->flags));
                }
            }
            if (auto depth_image_id = texture_cache.GetAssociatedDepth(*image)) {
                // SHADPS4_BIND_MEMO_STATS: redirected bindings are excluded from the memo.
                bind_memo_redirected |= u64{1} << ((image_bindings.size() - 1) & 63);
                static const bool stencil_storage = [] {
                    const char* value = std::getenv("SHADPS4_STENCIL_STORAGE");
                    return value && value[0] == '1';
                }();
                if (stencil_storage &&
                    desc.type == VideoCore::TextureCache::BindingType::Storage) {
                    // The placeholder's association is whichever depth surface bound it last;
                    // prefer the depth image at that address whose extent matches this view.
                    const VAddr depth_addr = texture_cache.GetImage(depth_image_id).info.guest_address;
                    if (const auto sized = texture_cache.FindDepthImageBySize(depth_addr,
                                                                             desc.info.size)) {
                        depth_image_id = sized;
                    }
                    // Vulkan cannot bind a D32S8 stencil aspect as a storage image. Write an R8
                    // scratch seeded from the stencil plane; copied back before the next bind.
                    image_id = StencilScratchFor(depth_image_id, desc.info);
                    image = &texture_cache.GetImage(image_id);
                    const std::pair writeback{image_id, depth_image_id};
                    if (std::ranges::find(pending_stencil_writebacks, writeback) ==
                        pending_stencil_writebacks.end()) {
                        CopyStencilPlane(depth_image_id, image_id);
                        pending_stencil_writebacks.push_back(writeback);
                    }
                } else {
                    // If this image has an associated depth image, it's a stencil attachment.
                    // Redirect the access to the actual depth-stencil buffer.
                    image_id = depth_image_id;
                    image = &texture_cache.GetImage(image_id);
                }
            } else if (image_desc.is_written &&
                       desc.type == VideoCore::TextureCache::BindingType::Storage) {
                texture_cache.NoteStorageWrite(*image);
            }
            if (image->binding.is_bound) {
                // The image is already bound. In case if it is about to be used as storage we
                // need to force general layout on it.
                image->binding.force_general |= image_desc.is_written;
            }
            image->binding.is_bound = 1u;
        }

        image_descriptor_array_sizes.push_back(num_bindings);
    }

    if (Common::BufferSampleProbe::Enabled()) {
        struct LookupShape {
            u64 stages{}, nonempty{}, lookups{};
        };
        static thread_local LookupShape shape;
        ++shape.stages;
        shape.nonempty += lookup_count != 0;
        shape.lookups += lookup_count;
        if ((shape.stages & 0xfffff) == 0) {
            LOG_INFO(Render_Vulkan,
                     "TextureLookupShape: stages={} nonempty={} lookups={} (cumulative)",
                     shape.stages, shape.nonempty, shape.lookups);
        }
    }
    const auto t_view = Common::FaultMonitor::Now();
    u64 findtex_ns = 0;
    u64 transit_ns = 0;

    // Second pass to re-bind images that were updated after binding
    for (auto& [image_id, desc] : image_bindings) {
        bool is_storage = desc.type == VideoCore::TextureCache::BindingType::Storage;
        if (!image_id) {
            image_infos.emplace_back(VK_NULL_HANDLE, VK_NULL_HANDLE, vk::ImageLayout::eGeneral);
            if (native_bindings) {
                native_image_indices[image_infos.size() - 1] =
                    NativeBindings::NullImageIndex(is_storage);
            }
        } else {
            const bool is_scratch = !stencil_scratches.empty() && IsStencilScratch(image_id);
            if (auto& old_image = texture_cache.GetImage(image_id);
                !is_scratch && old_image.binding.needs_rebind) {
                old_image.binding = {};
                image_id = texture_cache.FindImage(desc);
            }

            bound_images.emplace_back(image_id);

            auto& image = texture_cache.GetImage(image_id);
            const auto t_ft = Common::FaultMonitor::Now();
            auto& image_view = is_scratch ? image.FindView(desc.view_info)
                                          : texture_cache.FindTexture(image_id, desc);
            {
                // Experiment SHADPS4_NEO_IDBUF_FILL=<u32>: the PS4 Pro hardware ID buffer (a
                // 2-sample R32Uint surface written by the rasterizer) is not emulated; fill it with
                // a constant before a shader samples it, to test whether it gates the checkerboard
                // resolve. Not a fix.
                static const s64 idbuf_fill = [] {
                    const char* value = std::getenv("SHADPS4_NEO_IDBUF_FILL");
                    return value ? static_cast<s64>(std::strtoll(value, nullptr, 0)) : s64{-1};
                }();
                static const bool idbuf_derive = [] {
                    const char* value = std::getenv("SHADPS4_NEO_IDBUF_DERIVE");
                    return value && value[0] == '1';
                }();
                if (idbuf_derive && !is_storage && image.info.num_samples == 2 &&
                    image.info.pixel_format == vk::Format::eR32Uint) {
                    DeriveNeoIdBuffer(image);
                } else if (idbuf_fill >= 0 && !is_storage && image.info.num_samples == 2 &&
                           image.info.pixel_format == vk::Format::eR32Uint) {
                    scheduler.EndRendering();
                    image.Transit(vk::ImageLayout::eTransferDstOptimal,
                                  vk::AccessFlagBits2::eTransferWrite, {});
                    const u32 fill = static_cast<u32>(idbuf_fill);
                    vk::ClearColorValue value{};
                    value.setUint32({fill, fill, fill, fill});
                    const vk::ImageSubresourceRange range{vk::ImageAspectFlagBits::eColor, 0, 1, 0,
                                                          1};
                    auto command_access = scheduler.AcquireCommandBuffer();
                    command_access.Handle().clearColorImage(image.GetImage(),
                                                            image.backing->state.layout, value,
                                                            range);
                }
            }
            const auto t_tr = Common::FaultMonitor::Now();
            findtex_ns += Common::FaultMonitor::DiffNs(t_ft, t_tr);

            // The image is either bound as storage in a separate descriptor or bound as render
            // target in feedback loop. Depth images are excluded because they can't be bound as
            // storage and feedback loop doesn't make sense for them
            if ((image.binding.force_general || image.binding.is_target) &&
                !image.info.props.is_depth) {
                image.Transit(instance.IsAttachmentFeedbackLoopLayoutSupported() &&
                                      image.binding.is_target
                                  ? vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT
                                  : vk::ImageLayout::eGeneral,
                              vk::AccessFlagBits2::eShaderRead |
                                  (image.info.props.is_depth
                                       ? vk::AccessFlagBits2::eDepthStencilAttachmentWrite
                                       : vk::AccessFlagBits2::eColorAttachmentWrite |
                                             vk::AccessFlagBits2::eColorAttachmentRead),
                              {});
            } else {
                if (is_storage) {
                    image.Transit(vk::ImageLayout::eGeneral,
                                  vk::AccessFlagBits2::eShaderRead |
                                      vk::AccessFlagBits2::eShaderWrite,
                                  desc.view_info.range);
                } else {
                    const auto new_layout = image.info.props.is_depth
                                                ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
                                                : vk::ImageLayout::eShaderReadOnlyOptimal;
                    image.Transit(new_layout, vk::AccessFlagBits2::eShaderRead,
                                  desc.view_info.range);
                }
            }
            transit_ns += Common::FaultMonitor::SinceNs(t_tr);
            image.usage.storage |= is_storage;
            image.usage.texture |= !is_storage;

            image_infos.emplace_back(VK_NULL_HANDLE, *image_view.image_view,
                                     image.backing->state.layout);
            if (native_bindings) {
                // Same view object and the layout observed at this moment, as recorded above.
                const auto& info = image_infos.back();
                native_image_indices[image_infos.size() - 1] = native_bindings->ImageIndex(
                    image_view.native, info.imageView, info.imageLayout, is_storage);
            }
        }
    }

    Common::FaultMonitor::CountBindTexView(findtex_ns, transit_ns);
    const auto t_sampler_start = Common::FaultMonitor::Now();

    u32 image_info_idx = first_image_idx;
    u32 image_binding_idx = 0;
    for (u32 array_size : image_descriptor_array_sizes) {
        const auto& [_, desc] = image_bindings[image_binding_idx];
        const bool is_storage = desc.type == VideoCore::TextureCache::BindingType::Storage;
        if (native_bindings) {
            for (u32 element = 0; element < array_size; ++element) {
                native_indices[binding.unified + element] =
                    native_image_indices[image_info_idx + element];
            }
        }
        if (build_set_writes) {
            auto& set_write = set_writes[set_write_index++];
            set_write.dstSet = VK_NULL_HANDLE;
            set_write.dstBinding = binding.unified;
            set_write.dstArrayElement = 0;
            set_write.descriptorCount = array_size;
            set_write.descriptorType =
                is_storage ? vk::DescriptorType::eStorageImage : vk::DescriptorType::eSampledImage;
            set_write.pImageInfo = &image_infos[image_info_idx];
        }

        image_info_idx += array_size;
        image_binding_idx += array_size;
        binding.unified += array_size;
    }

    for (const auto& sampler : stage.samplers) {
        auto ssharp = sampler.GetSharp(bound);
        if (sampler.disable_aniso) {
            const auto& tsharp = stage.images[sampler.associated_image].GetSharp(bound);
            if (tsharp.base_level == 0 && tsharp.last_level == 0) {
                ssharp.max_aniso.Assign(AmdGpu::AnisoRatio::One);
            }
        }
        u32 native_sampler_index{};
        const auto vk_sampler =
            texture_cache.GetSampler(ssharp, liverpool->BackRegs().ta_bc_base,
                                     native_bindings ? &native_sampler_index : nullptr);
        image_infos.emplace_back(vk_sampler, VK_NULL_HANDLE, vk::ImageLayout::eGeneral);
        const u32 unified_binding = binding.unified++;
        if (native_bindings) {
            native_indices[unified_binding] = native_sampler_index;
        }
        if (build_set_writes) {
            auto& set_write = set_writes[set_write_index++];
            set_write.dstSet = VK_NULL_HANDLE;
            set_write.dstBinding = unified_binding;
            set_write.dstArrayElement = 0;
            set_write.descriptorCount = 1;
            set_write.descriptorType = vk::DescriptorType::eSampler;
            set_write.pImageInfo = &image_infos.back();
        }
    }

    // The descriptor-write loop between t_view's end and t_sampler_start is bookkeeping only and
    // is folded into the sampler span; if sampler time looks large with a high memo hit rate, that
    // loop is where to look next rather than GetSampler.
    Common::FaultMonitor::CountBindTexSplit(Common::FaultMonitor::DiffNs(t_findimage, t_view),
                                            Common::FaultMonitor::DiffNs(t_view, t_sampler_start),
                                            Common::FaultMonitor::SinceNs(t_sampler_start));
}

RenderState Rasterizer::BeginRendering(const GraphicsPipeline* pipeline) {
    attachment_feedback_loop = false;
    const auto& regs = liverpool->BackRegs();
    const auto& key = pipeline->GetGraphicsKey();
    RenderState state{};
    state.width = instance.GetMaxFramebufferWidth();
    state.height = instance.GetMaxFramebufferHeight();
    state.num_layers = std::numeric_limits<u16>::max();
    state.num_color_attachments = std::bit_width(key.mrt_mask);
    for (auto cb = 0u; cb < state.num_color_attachments; ++cb) {
        auto& [image_id, desc] = cb_descs[cb];
        if (!image_id) {
            state.color_attachments[cb] = {};
            continue;
        }
        auto* image = &texture_cache.GetImage(image_id);
        if (image->binding.needs_rebind) {
            image_id = bound_images.emplace_back(texture_cache.FindImage(desc));
            image = &texture_cache.GetImage(image_id);
        }
        texture_cache.UpdateImage(image_id);
        image->SetBackingSamples(key.color_samples[cb]);
        const auto& image_view = texture_cache.FindRenderTarget(image_id, desc);
        const auto slice = image_view.info.range.base.layer;
        const auto mip = image_view.info.range.base.level;

        const auto& col_buf = regs.color_buffers[cb];
        const bool is_clear = texture_cache.IsMetaCleared(col_buf.CmaskAddress(), slice);
        texture_cache.TouchMeta(col_buf.CmaskAddress(), slice, false);

        if (image->binding.is_bound) {
            ASSERT_MSG(!image->binding.force_general,
                       "Having image both as storage and render target is unsupported");
            image->Transit(instance.IsAttachmentFeedbackLoopLayoutSupported()
                               ? vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT
                               : vk::ImageLayout::eGeneral,
                           vk::AccessFlagBits2::eColorAttachmentWrite, {});
            attachment_feedback_loop = true;
        } else {
            image->Transit(vk::ImageLayout::eColorAttachmentOptimal,
                           vk::AccessFlagBits2::eColorAttachmentWrite |
                               vk::AccessFlagBits2::eColorAttachmentRead,
                           desc.view_info.range);
        }

        state.width = std::min<u32>(state.width, std::max(image->info.size.width >> mip, 1u));
        state.height = std::min<u32>(state.height, std::max(image->info.size.height >> mip, 1u));
        state.num_layers = std::min<u32>(state.num_layers, image_view.info.range.extent.layers);

        const auto clear_value =
            is_clear ? LiverpoolToVK::ColorBufferClearValue(col_buf) : vk::ClearValue{};
        auto& attachment = state.color_attachments[cb];
        attachment.image_view = *image_view.image_view;
        attachment.image_layout = image->backing->state.layout;
        attachment.clear_value = clear_value.color.uint32;
        attachment.is_clear = is_clear;

        image->usage.render_target = 1u;
    }
    for (u32 cb = state.num_color_attachments; cb < state.color_attachments.size(); ++cb) {
        state.color_attachments[cb] = {};
    }

    if (auto image_id = db_desc.first; image_id) {
        auto& desc = db_desc.second;
        const auto htile_address = regs.depth_htile_data_base.GetAddress();
        const auto t_fd = Common::FaultMonitor::Now();
        const auto& image_view = texture_cache.FindDepthTarget(image_id, desc);
        Common::FaultMonitor::CountFindDepthTarget(Common::FaultMonitor::SinceNs(t_fd));
        auto& image = texture_cache.GetImage(image_id);
        if (image.info.num_samples == 2) {
            // Pro scene depth (2 samples): source for SHADPS4_NEO_IDBUF_DERIVE, with the sample
            // locations active while it is rasterized (PA_SC_AA_SAMPLE_LOCS_PIXEL_X0Y0_0).
            neo_depth_addr = image.info.guest_address;
            neo_depth_size = image.info.size;
            neo_depth_x0y0 = regs.reg_array[0xA2FE];
        }

        const auto slice = image_view.info.range.base.layer;
        const bool is_depth_clear =
            (regs.depth_render_control.depth_clear_enable && regs.depth_control.depth_enable &&
             regs.depth_control.depth_write_enable) ||
            texture_cache.IsMetaCleared(htile_address, slice);
        const bool is_stencil_clear = regs.depth_render_control.stencil_clear_enable;
        texture_cache.TouchMeta(htile_address, slice, false);
        ASSERT(desc.view_info.range.extent.levels == 1 && !image.binding.needs_rebind);

        const bool has_stencil = image.info.props.has_stencil;
        // Stencil writes can be enabled while depth writes are off.
        const bool stencil_write =
            has_stencil && regs.depth_control.stencil_enable && !desc.view_info.is_storage;
        const auto new_layout = desc.view_info.is_storage
                                    ? has_stencil ? vk::ImageLayout::eDepthStencilAttachmentOptimal
                                                  : vk::ImageLayout::eDepthAttachmentOptimal
                                : stencil_write
                                    ? vk::ImageLayout::eDepthReadOnlyStencilAttachmentOptimal
                                : has_stencil ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
                                              : vk::ImageLayout::eDepthReadOnlyOptimal;
        image.Transit(new_layout,
                      vk::AccessFlagBits2::eDepthStencilAttachmentWrite |
                          vk::AccessFlagBits2::eDepthStencilAttachmentRead,
                      desc.view_info.range);

        state.width = std::min<u32>(state.width, image.info.size.width);
        state.height = std::min<u32>(state.height, image.info.size.height);
        state.num_layers = std::min<u32>(state.num_layers, image_view.info.range.extent.layers);

        auto& attachment = state.depth_stencil_attachment;
        attachment.image_view = *image_view.image_view;
        attachment.image_layout = image.backing->state.layout;
        attachment.clear_value = {};

        if (regs.depth_buffer.DepthValid()) {
            attachment.clear_value[0] = is_depth_clear ? std::bit_cast<u32>(regs.depth_clear) : 0u;
            attachment.has_depth = true;
            attachment.depth_clear = is_depth_clear;
        }
        if (regs.depth_buffer.StencilValid()) {
            attachment.clear_value[1] = is_stencil_clear ? regs.stencil_clear : 0u;
            attachment.has_stencil = true;
            attachment.stencil_clear = is_stencil_clear;
        } else if (regs.depth_buffer.DepthValid() &&
                   LiverpoolToVK::PromoteDepthToStencil(regs.depth_buffer.z_info.format,
                                                        regs.depth_buffer.stencil_info.format)) {
            // Promoted D32S8: attach the stencil aspect (loaded, never cleared) so the attachment
            // matches the pipeline's stencil format.
            attachment.has_stencil = true;
            attachment.stencil_clear = false;
        }

        image.usage.depth_target = true;
    } else {
        state.depth_stencil_attachment = {};
    }

    if (state.num_layers == std::numeric_limits<u16>::max()) {
        state.num_layers = 1;
    }

    // Diagnostic: a render pass with no usable attachment at all.
    // A null color attachment on its own is normal - MRT gaps, or the guest disabling colour
    // writes - but a pass where every colour attachment is null AND there is no depth/stencil
    // attachment renders into nothing. That is legal Vulkan and raises no validation error, so
    // it is invisible unless we say so here. Deduplicated by signature; each distinct case is
    // reported once per run.
    {
        bool any_color = false;
        for (u32 cb = 0; cb < state.num_color_attachments; ++cb) {
            if (state.color_attachments[cb].image_view) {
                any_color = true;
                break;
            }
        }
        if (!any_color && !state.depth_stencil_attachment.image_view) {
            u64 sig = u64(key.mrt_mask) << 1;
            for (const auto h : key.stage_hashes) {
                sig = sig * 0x100000001b3ULL ^ u64(h);
            }
            static std::unordered_set<u64> reported;
            if (reported.insert(sig).second) {
                const bool cb_disabled =
                    regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Disable;
                std::string detail;
                for (u32 cb = 0; cb < state.num_color_attachments; ++cb) {
                    const auto& col_buf = regs.color_buffers[cb];
                    detail += fmt::format(
                        " [cb{} valid={} addr={:#x} dfmt={} nfmt={} write_mask={:#x} in_mrt={}]",
                        cb, bool(col_buf), col_buf.Address(), u32(col_buf.GetDataFmt()),
                        u32(col_buf.GetNumberFmt()), regs.color_target_mask.GetMask(cb),
                        (key.mrt_mask >> cb) & 1);
                }
                std::string stages;
                for (u32 i = 0; i < key.stage_hashes.size(); ++i) {
                    if (key.stage_hashes[i] != 0) {
                        stages += fmt::format(" stage{}={:#x}", i, key.stage_hashes[i]);
                    }
                }
                // NOTE: this is usually the guest's intent, not a fault - Second Son issues
                // these with color_writes_disabled=true and write_mask=0. Reported so the case
                // stays visible, but do NOT read it as an error on its own. See
                // notes/renderdoc-capture-01-timbers.md.
                LOG_WARNING(Render_Vulkan,
                            "Attachment-less draw (usually guest intent, see notes): "
                            "mrt_mask={:#x} "
                            "color_writes_disabled={} depth_en={} depth_valid={} stencil_en={} "
                            "stencil_valid={}{} shaders:{}",
                            key.mrt_mask, cb_disabled, bool(regs.depth_control.depth_enable),
                            regs.depth_buffer.DepthValid(), bool(regs.depth_control.stencil_enable),
                            regs.depth_buffer.StencilValid(), detail, stages);
            }
        }
    }

    return state;
}

void Rasterizer::Resolve() {
    const auto& mrt0_hint = liverpool->BackCbExtent(0);
    const auto& mrt1_hint = liverpool->BackCbExtent(1);
    VideoCore::TextureCache::ImageDesc mrt0_desc{liverpool->BackRegs().color_buffers[0], mrt0_hint};
    VideoCore::TextureCache::ImageDesc mrt1_desc{liverpool->BackRegs().color_buffers[1], mrt1_hint};
    auto& mrt0_image = texture_cache.GetImage(texture_cache.FindImage(mrt0_desc, true));
    auto& mrt1_image = texture_cache.GetImage(texture_cache.FindImage(mrt1_desc, true));

    ScopeMarkerBegin(fmt::format("Resolve:MRT0={:#x}:MRT1={:#x}",
                                 liverpool->BackRegs().color_buffers[0].Address(),
                                 liverpool->BackRegs().color_buffers[1].Address()));
    mrt1_image.Resolve(mrt0_image, mrt0_desc.view_info.range, mrt1_desc.view_info.range);
    ScopeMarkerEnd();
}

void Rasterizer::DepthStencilCopy(bool is_depth, bool is_stencil) {
    auto& regs = liverpool->BackRegs();

    auto read_desc = VideoCore::TextureCache::ImageDesc(
        regs.depth_buffer, regs.depth_view, regs.depth_control,
        regs.depth_htile_data_base.GetAddress(), liverpool->BackDbExtent(), false);
    auto write_desc = VideoCore::TextureCache::ImageDesc(
        regs.depth_buffer, regs.depth_view, regs.depth_control,
        regs.depth_htile_data_base.GetAddress(), liverpool->BackDbExtent(), true);

    auto& read_image = texture_cache.GetImage(texture_cache.FindImage(read_desc));
    auto& write_image = texture_cache.GetImage(texture_cache.FindImage(write_desc));

    VideoCore::SubresourceRange sub_range;
    sub_range.base.layer = liverpool->BackRegs().depth_view.slice_start;
    sub_range.extent.layers = liverpool->BackRegs().depth_view.NumSlices() - sub_range.base.layer;

    ScopeMarkerBegin(fmt::format(
        "DepthStencilCopy:DR={:#x}:SR={:#x}:DW={:#x}:SW={:#x}", regs.depth_buffer.DepthAddress(),
        regs.depth_buffer.StencilAddress(), regs.depth_buffer.DepthWriteAddress(),
        regs.depth_buffer.StencilWriteAddress()));

    read_image.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead,
                       sub_range);
    write_image.Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite,
                        sub_range);

    auto aspect_mask = vk::ImageAspectFlags(0);
    if (is_depth) {
        aspect_mask |= vk::ImageAspectFlagBits::eDepth;
    }
    if (is_stencil) {
        aspect_mask |= vk::ImageAspectFlagBits::eStencil;
    }

    vk::ImageCopy region = {
        .srcSubresource =
            {
                .aspectMask = aspect_mask,
                .mipLevel = 0,
                .baseArrayLayer = sub_range.base.layer,
                .layerCount = sub_range.extent.layers,
            },
        .srcOffset = {0, 0, 0},
        .dstSubresource =
            {
                .aspectMask = aspect_mask,
                .mipLevel = 0,
                .baseArrayLayer = sub_range.base.layer,
                .layerCount = sub_range.extent.layers,
            },
        .dstOffset = {0, 0, 0},
        .extent = {write_image.info.size.width, write_image.info.size.height, 1},
    };
    scheduler.AcquireCommandBuffer().Handle().copyImage(
        read_image.GetImage(), Vulkan::GuestLayout(vk::ImageLayout::eTransferSrcOptimal),
        write_image.GetImage(), Vulkan::GuestLayout(vk::ImageLayout::eTransferDstOptimal), region);

    ScopeMarkerEnd();
}

void Rasterizer::FillBuffer(VAddr address, u32 num_bytes, u32 value, bool is_gds) {
    if (!is_gds) {
        AmdGpu::SplitHazard::NoteWrite(address, num_bytes);
    }
    buffer_cache.FillBuffer(address, num_bytes, value, is_gds);
}

void Rasterizer::CopyBuffer(VAddr dst, VAddr src, u32 num_bytes, bool dst_gds, bool src_gds) {
    if (!dst_gds) {
        AmdGpu::SplitHazard::NoteWrite(dst, num_bytes);
    }
    buffer_cache.CopyBuffer(dst, src, num_bytes, dst_gds, src_gds);
}

u32 Rasterizer::ReadDataFromGds(u32 gds_offset) {
    auto* gds_buf = buffer_cache.GetGdsBuffer();
    u32 value;
    std::memcpy(&value, gds_buf->mapped_data.data() + gds_offset, sizeof(u32));
    return value;
}

bool Rasterizer::InvalidateMemory(VAddr addr, u64 size, VideoCore::InvalidateOrigin origin) {
    if (!IsMapped(addr, size)) {
        // Not GPU mapped memory, can skip invalidation logic entirely.
        return false;
    }
    buffer_cache.InvalidateMemory(addr, size, origin);
    texture_cache.InvalidateMemory(addr, size);
    return true;
}

bool Rasterizer::ReadMemory(VAddr addr, u64 size) {
    if (!IsMapped(addr, size)) {
        // Not GPU mapped memory, can skip invalidation logic entirely.
        return false;
    }
    buffer_cache.ReadMemory(addr, size);
    return true;
}

void Rasterizer::ProcessDownloadImages() {
    texture_cache.ProcessDownloadImages();
}

bool Rasterizer::IsMapped(VAddr addr, u64 size) {
    if (size == 0) {
        // There is no memory, so not mapped.
        return false;
    }
    if (static_cast<u64>(addr) > std::numeric_limits<u64>::max() - size) {
        // Memory range wrapped the address space, cannot be mapped.
        return false;
    }
    const auto range = decltype(mapped_ranges)::interval_type::right_open(addr, addr + size);

    Common::RecursiveSharedLock lock{mapped_ranges_mutex};
    return boost::icl::contains(mapped_ranges, range);
}

void Rasterizer::MapMemory(VAddr addr, u64 size) {
    {
        std::scoped_lock lock{mapped_ranges_mutex};
        mapped_ranges += decltype(mapped_ranges)::interval_type::right_open(addr, addr + size);
    }
    page_manager.OnGpuMap(addr, size);
    buffer_cache.EndGpuMap(addr, size);
}

void Rasterizer::UnmapMemory(VAddr addr, u64 size) {
    buffer_cache.BeginGpuUnmap(addr, size);
    buffer_cache.InvalidateMemory(addr, size);
    texture_cache.UnmapMemory(addr, size);
    page_manager.OnGpuUnmap(addr, size);
    {
        std::scoped_lock lock{mapped_ranges_mutex};
        mapped_ranges -= decltype(mapped_ranges)::interval_type::right_open(addr, addr + size);
    }
    // SHADPS4_BIND_MEMO_STATS: BeginGpuUnmap bumped before the unmap; bump again after it, so a
    // record taken while it was in progress cannot validate against the final mapping.
    buffer_cache.BumpStructGeneration();
}

void Rasterizer::UpdateDynamicState(const GraphicsPipeline* pipeline, const bool is_indexed) const {
    // SHADPS4_VIEWPORT_MEMO=1: viewports and scissors are a pure function of context registers
    // (and device limits), so skip recomputing them while no SET_CONTEXT_REG changed a byte.
    // =2 recomputes anyway and checks the tracked values did not change.
    static const int viewport_memo = [] {
        const char* value = std::getenv("SHADPS4_VIEWPORT_MEMO");
        return value ? value[0] - '0' : 0;
    }();
    const u64 epoch = liverpool->BackContextEpoch();
    if (viewport_memo == 1 && epoch == viewport_epoch) {
        ++viewport_memo_skips;
    } else if (viewport_memo == 2 && epoch == viewport_epoch) {
        const auto& ds = scheduler.GetDynamicState();
        const auto viewports = ds.viewports;
        const auto scissors = ds.scissors;
        UpdateViewportScissorState();
        const auto& after = scheduler.GetDynamicState();
        const bool same =
            viewports.size() == after.viewports.size() && scissors.size() == after.scissors.size() &&
            std::memcmp(viewports.data(), after.viewports.data(),
                        viewports.size() * sizeof(viewports[0])) == 0 &&
            std::memcmp(scissors.data(), after.scissors.data(),
                        scissors.size() * sizeof(scissors[0])) == 0;
        ++viewport_memo_skips;
        if (!same) {
            ++viewport_memo_mismatches;
        }
        if ((viewport_memo_skips & 0xfffff) == 0) {
            LOG_INFO(Render_Vulkan, "VIEWPORT_MEMO verify: skippable {} mismatches {}",
                     viewport_memo_skips, viewport_memo_mismatches);
        }
    } else {
        UpdateViewportScissorState();
        viewport_epoch = epoch;
    }
    UpdateDepthStencilState();

    // Bounded read-only diagnostic: distinguish PS4 test/operation stencil values
    // before selecting any correction for the captured hair composite failure.
    static const bool hair_stencil_trace = [] {
        const char* value = std::getenv("SHADPS4_HAIR_STENCIL_TRACE");
        return value && std::string{value} == "1";
    }();
    if (hair_stencil_trace) {
        const auto* fs = pipeline->GetStages()[int(Shader::LogicalStage::Fragment)];
        if (fs && (fs->pgm_hash == 0xe59a987f || fs->pgm_hash == 0x1540dfa9 ||
                   fs->pgm_hash == 0x86bb04f5 || fs->pgm_hash == 0x6f5258e8)) {
            static std::unordered_set<std::string> seen;
            const auto& regs = liverpool->BackRegs();
            const auto state = fmt::format(
                "fs={:#x} control={:#x} ops={:#x} front={:#x} back={:#x} shader={:#x} "
                "render={:#x} stencil_valid={} depth_addr={:#x}",
                fs->pgm_hash, std::bit_cast<u32>(regs.depth_control),
                std::bit_cast<u32>(regs.stencil_control), std::bit_cast<u32>(regs.stencil_ref_front),
                std::bit_cast<u32>(regs.stencil_ref_back),
                std::bit_cast<u32>(regs.depth_shader_control),
                std::bit_cast<u32>(regs.depth_render_control), regs.depth_buffer.StencilValid(),
                regs.depth_buffer.DepthAddress());
            if (seen.size() < 128 && seen.insert(state).second) {
                LOG_INFO(Render_Vulkan, "HairStencil: {}", state);
            }
        }
    }
    UpdatePrimitiveState(is_indexed);
    UpdateRasterizationState();
    UpdateColorBlendingState(pipeline);

    scheduler.CommitDynamicState();
}

void Rasterizer::UpdateViewportScissorState() const {
    const auto& regs = liverpool->BackRegs();

    const auto combined_scissor_value_tl = [](s16 scr, s16 win, s16 gen, s16 win_offset) {
        return std::max({scr, s16(win + win_offset), s16(gen + win_offset)});
    };
    const auto combined_scissor_value_br = [](s16 scr, s16 win, s16 gen, s16 win_offset) {
        return std::min({scr, s16(win + win_offset), s16(gen + win_offset)});
    };
    const bool enable_offset = !regs.window_scissor.window_offset_disable;

    AmdGpu::Scissor scsr{};
    scsr.top_left_x = combined_scissor_value_tl(
        regs.screen_scissor.top_left_x, s16(regs.window_scissor.top_left_x),
        s16(regs.generic_scissor.top_left_x),
        enable_offset ? regs.window_offset.window_x_offset : 0);
    scsr.top_left_y = combined_scissor_value_tl(
        regs.screen_scissor.top_left_y, s16(regs.window_scissor.top_left_y),
        s16(regs.generic_scissor.top_left_y),
        enable_offset ? regs.window_offset.window_y_offset : 0);
    scsr.bottom_right_x = combined_scissor_value_br(
        regs.screen_scissor.bottom_right_x, regs.window_scissor.bottom_right_x,
        regs.generic_scissor.bottom_right_x,
        enable_offset ? regs.window_offset.window_x_offset : 0);
    scsr.bottom_right_y = combined_scissor_value_br(
        regs.screen_scissor.bottom_right_y, regs.window_scissor.bottom_right_y,
        regs.generic_scissor.bottom_right_y,
        enable_offset ? regs.window_offset.window_y_offset : 0);

    boost::container::static_vector<vk::Viewport, AmdGpu::NUM_VIEWPORTS> viewports;
    boost::container::static_vector<vk::Rect2D, AmdGpu::NUM_VIEWPORTS> scissors;

    if (regs.polygon_control.enable_window_offset &&
        (regs.window_offset.window_x_offset != 0 || regs.window_offset.window_y_offset != 0)) {
        LOG_ERROR(Render_Vulkan,
                  "PA_SU_SC_MODE_CNTL.VTX_WINDOW_OFFSET_ENABLE support is not yet implemented.");
    }

    const auto& vp_ctl = regs.viewport_control;
    for (u32 i = 0; i < AmdGpu::NUM_VIEWPORTS; i++) {
        const auto& vp = regs.viewports[i];
        const auto& vp_d = regs.viewport_depths[i];
        if (vp.xscale == 0) {
            continue;
        }

        const auto zoffset = vp_ctl.zoffset_enable ? vp.zoffset : 0.f;
        const auto zscale = vp_ctl.zscale_enable ? vp.zscale : 1.f;

        vk::Viewport viewport{};

        // https://gitlab.freedesktop.org/mesa/mesa/-/blob/209a0ed/src/amd/vulkan/radv_pipeline_graphics.c#L688-689
        // https://gitlab.freedesktop.org/mesa/mesa/-/blob/209a0ed/src/amd/vulkan/radv_cmd_buffer.c#L3103-3109
        // When the clip space is ranged [-1...1], the zoffset is centered.
        // By reversing the above viewport calculations, we get the following:
        if (regs.clipper_control.clip_space == AmdGpu::ClipSpace::MinusWToW) {
            viewport.minDepth = zoffset - zscale;
            viewport.maxDepth = zoffset + zscale;
        } else {
            viewport.minDepth = zoffset;
            viewport.maxDepth = zoffset + zscale;
        }

        if (!instance.IsDepthRangeUnrestrictedSupported()) {
            // Unrestricted depth range not supported by device. Restrict to valid range.
            viewport.minDepth = std::max(viewport.minDepth, 0.f);
            viewport.maxDepth = std::min(viewport.maxDepth, 1.f);
        }

        if (regs.IsClipDisabled()) {
            // In case if clipping is disabled we patch the shader to convert vertex position
            // from screen space coordinates to NDC by defining a render space as full hardware
            // window range [0..16383, 0..16383] and setting the viewport to its size.
            viewport.x = 0.f;
            viewport.y = 0.f;
            viewport.width = float(std::min<u32>(instance.GetMaxViewportWidth(), 16_KB));
            viewport.height = float(std::min<u32>(instance.GetMaxViewportHeight(), 16_KB));
        } else {
            const auto xoffset = vp_ctl.xoffset_enable ? vp.xoffset : 0.f;
            const auto xscale = vp_ctl.xscale_enable ? vp.xscale : 1.f;
            const auto yoffset = vp_ctl.yoffset_enable ? vp.yoffset : 0.f;
            const auto yscale = vp_ctl.yscale_enable ? vp.yscale : 1.f;

            viewport.x = xoffset - xscale;
            viewport.y = yoffset - yscale;
            viewport.width = xscale * 2.0f;
            viewport.height = yscale * 2.0f;
        }

        viewports.push_back(viewport);

        auto vp_scsr = scsr;
        if (regs.mode_control.vport_scissor_enable) {
            vp_scsr.top_left_x =
                std::max(vp_scsr.top_left_x, s16(regs.viewport_scissors[i].top_left_x));
            vp_scsr.top_left_y =
                std::max(vp_scsr.top_left_y, s16(regs.viewport_scissors[i].top_left_y));
            vp_scsr.bottom_right_x = std::min(AmdGpu::Scissor::Clamp(vp_scsr.bottom_right_x),
                                              regs.viewport_scissors[i].bottom_right_x);
            vp_scsr.bottom_right_y = std::min(AmdGpu::Scissor::Clamp(vp_scsr.bottom_right_y),
                                              regs.viewport_scissors[i].bottom_right_y);
        }
        scissors.push_back({
            .offset = {vp_scsr.top_left_x, vp_scsr.top_left_y},
            .extent = {vp_scsr.GetWidth(), vp_scsr.GetHeight()},
        });
    }

    if (viewports.empty()) {
        // Vulkan requires providing at least one viewport.
        constexpr vk::Viewport empty_viewport = {
            .x = -1.0f,
            .y = -1.0f,
            .width = 1.0f,
            .height = 1.0f,
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        };
        constexpr vk::Rect2D empty_scissor = {
            .offset = {0, 0},
            .extent = {1, 1},
        };
        viewports.push_back(empty_viewport);
        scissors.push_back(empty_scissor);
    }

    auto& dynamic_state = scheduler.GetDynamicState();
    dynamic_state.SetViewports(viewports);
    dynamic_state.SetScissors(scissors);
}

void Rasterizer::UpdateDepthStencilState() const {
    const auto& regs = liverpool->BackRegs();
    auto& dynamic_state = scheduler.GetDynamicState();

    const auto depth_test_enabled =
        regs.depth_control.depth_enable && regs.depth_buffer.DepthValid();
    dynamic_state.SetDepthTestEnabled(depth_test_enabled);
    if (depth_test_enabled) {
        dynamic_state.SetDepthWriteEnabled(regs.depth_control.depth_write_enable &&
                                           !regs.depth_render_control.depth_clear_enable);
        dynamic_state.SetDepthCompareOp(LiverpoolToVK::CompareOp(regs.depth_control.depth_func));
    }

    const auto depth_bounds_test_enabled = regs.depth_control.depth_bounds_enable;
    dynamic_state.SetDepthBoundsTestEnabled(depth_bounds_test_enabled);
    if (depth_bounds_test_enabled) {
        dynamic_state.SetDepthBounds(regs.depth_bounds_min, regs.depth_bounds_max);
    }

    const auto depth_bias_enabled = regs.polygon_control.NeedsBias();
    dynamic_state.SetDepthBiasEnabled(depth_bias_enabled);
    if (depth_bias_enabled) {
        const bool front = regs.polygon_control.enable_polygon_offset_front;
        dynamic_state.SetDepthBias(
            front ? regs.poly_offset.front_offset : regs.poly_offset.back_offset,
            regs.poly_offset.depth_bias,
            (front ? regs.poly_offset.front_scale : regs.poly_offset.back_scale) / 16.f);
    }

    const auto stencil_test_enabled =
        regs.depth_control.stencil_enable && regs.depth_buffer.StencilValid();
    dynamic_state.SetStencilTestEnabled(stencil_test_enabled);
    if (stencil_test_enabled) {
        const StencilOps front_ops{
            .fail_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_fail_front),
            .pass_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_zpass_front),
            .depth_fail_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_zfail_front),
            .compare_op = LiverpoolToVK::CompareOp(regs.depth_control.stencil_ref_func),
        };
        const StencilOps back_ops = regs.depth_control.backface_enable ? StencilOps{
            .fail_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_fail_back),
            .pass_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_zpass_back),
            .depth_fail_op = LiverpoolToVK::StencilOp(regs.stencil_control.stencil_zfail_back),
            .compare_op = LiverpoolToVK::CompareOp(regs.depth_control.stencil_bf_func),
        } : front_ops;
        dynamic_state.SetStencilOps(front_ops, back_ops);

        const bool stencil_clear = regs.depth_render_control.stencil_clear_enable;
        const auto front = regs.stencil_ref_front;
        const auto back =
            regs.depth_control.backface_enable ? regs.stencil_ref_back : regs.stencil_ref_front;
        static const bool replace_op_reference = [] {
            const char* value = std::getenv("SHADPS4_STENCIL_REPLACE_OP");
            return value && std::string{value} == "1";
        }();
        const auto reference = [&](const AmdGpu::StencilRefMask& face, AmdGpu::CompareFunc compare,
                                   AmdGpu::StencilFunc fail, AmdGpu::StencilFunc pass,
                                   AmdGpu::StencilFunc depth_fail) -> u32 {
            // PS4 ReplaceOp writes stencil_op_val, independently of stencil_test_val.
            // Vulkan Replace uses one reference for both. With an unconditional test
            // and Keep for the other outcomes, changing that reference is equivalent.
            // Other combinations and shader-exported references need separate handling.
            if (replace_op_reference && compare == AmdGpu::CompareFunc::Always &&
                fail == AmdGpu::StencilFunc::Keep && pass == AmdGpu::StencilFunc::ReplaceOp &&
                depth_fail == AmdGpu::StencilFunc::Keep &&
                !regs.depth_shader_control.stencil_test_val_export_enable &&
                !regs.depth_shader_control.stencil_op_val_export_enable) {
                return face.stencil_op_val;
            }
            return face.stencil_test_val;
        };
        const u32 front_reference =
            reference(front, regs.depth_control.stencil_ref_func,
                      regs.stencil_control.stencil_fail_front, regs.stencil_control.stencil_zpass_front,
                      regs.stencil_control.stencil_zfail_front);
        const u32 back_reference = regs.depth_control.backface_enable
                                      ? reference(back, regs.depth_control.stencil_bf_func,
                                                  regs.stencil_control.stencil_fail_back,
                                                  regs.stencil_control.stencil_zpass_back,
                                                  regs.stencil_control.stencil_zfail_back)
                                      : front_reference;
        if (replace_op_reference && (front_reference != front.stencil_test_val ||
                                     back_reference != back.stencil_test_val)) {
            static bool reported{};
            if (!reported) {
                LOG_INFO(Render_Vulkan,
                         "Stencil ReplaceOp correction active: front test={} op={} reference={} "
                         "write_mask={:#x}; back test={} op={} reference={} write_mask={:#x}",
                         front.stencil_test_val, front.stencil_op_val, front_reference,
                         front.stencil_write_mask, back.stencil_test_val, back.stencil_op_val,
                         back_reference, back.stencil_write_mask);
                reported = true;
            }
        }
        dynamic_state.SetStencilReferences(front_reference, back_reference);
        dynamic_state.SetStencilWriteMasks(!stencil_clear ? front.stencil_write_mask : 0U,
                                           !stencil_clear ? back.stencil_write_mask : 0U);
        dynamic_state.SetStencilCompareMasks(front.stencil_mask, back.stencil_mask);
    }
}

void Rasterizer::UpdatePrimitiveState(const bool is_indexed) const {
    const auto& regs = liverpool->BackRegs();
    auto& dynamic_state = scheduler.GetDynamicState();

    const auto is_list_topology = [](const AmdGpu::PrimitiveType type) {
        const auto topology = LiverpoolToVK::PrimitiveType(type);
        return topology == vk::PrimitiveTopology::ePointList ||
               topology == vk::PrimitiveTopology::eLineList ||
               topology == vk::PrimitiveTopology::eTriangleList ||
               topology == vk::PrimitiveTopology::eLineListWithAdjacency ||
               topology == vk::PrimitiveTopology::eTriangleListWithAdjacency;
    };
    const auto is_patch_list_topology = [](const AmdGpu::PrimitiveType type) {
        // Quad and rect lists are emulated using tessellation.
        return type == AmdGpu::PrimitiveType::PatchPrimitive ||
               type == AmdGpu::PrimitiveType::QuadList || type == AmdGpu::PrimitiveType::RectList;
    };

    const auto prim_restart =
        (regs.enable_primitive_restart & 1) != 0 &&
        (instance.IsListRestartSupported() || !is_list_topology(regs.primitive_type)) &&
        (instance.IsPatchListRestartSupported() || !is_patch_list_topology(regs.primitive_type));
    ASSERT_MSG(!is_indexed || !prim_restart || regs.primitive_restart_index == 0xFFFF ||
                   regs.primitive_restart_index == 0xFFFFFFFF,
               "Primitive restart index other than -1 is not supported yet");

    const auto cull_mode = LiverpoolToVK::IsPrimitiveCulled(regs.primitive_type)
                               ? LiverpoolToVK::CullMode(regs.polygon_control.CullingMode())
                               : vk::CullModeFlagBits::eNone;
    const auto front_face = LiverpoolToVK::FrontFace(regs.polygon_control.front_face);

    dynamic_state.SetPrimitiveRestartEnabled(prim_restart);
    dynamic_state.SetRasterizerDiscardEnabled(regs.clipper_control.dx_rasterization_kill);
    dynamic_state.SetCullMode(cull_mode);
    dynamic_state.SetFrontFace(front_face);
}

void Rasterizer::UpdateRasterizationState() const {
    const auto& regs = liverpool->BackRegs();
    auto& dynamic_state = scheduler.GetDynamicState();
    dynamic_state.SetLineWidth(regs.line_control.Width());
}

void Rasterizer::UpdateColorBlendingState(const GraphicsPipeline* pipeline) const {
    const auto& regs = liverpool->BackRegs();
    auto& dynamic_state = scheduler.GetDynamicState();
    dynamic_state.SetBlendConstants(regs.blend_constants);
    dynamic_state.SetColorWriteMasks(pipeline->GetGraphicsKey().write_masks);
    dynamic_state.SetAttachmentFeedbackLoopEnabled(attachment_feedback_loop);
}

void Rasterizer::ScopeMarkerBegin(const std::string_view& str, bool from_guest) {
    if ((from_guest && !EmulatorSettings.IsVkGuestMarkersEnabled()) ||
        (!from_guest && !EmulatorSettings.IsVkHostMarkersEnabled())) {
        return;
    }
    auto command_access = scheduler.AcquireCommandBuffer();
    const auto cmdbuf = command_access.Handle();
    cmdbuf.beginDebugUtilsLabelEXT(vk::DebugUtilsLabelEXT{
        .pLabelName = str.data(),
    });
}

void Rasterizer::ScopeMarkerEnd(bool from_guest) {
    if ((from_guest && !EmulatorSettings.IsVkGuestMarkersEnabled()) ||
        (!from_guest && !EmulatorSettings.IsVkHostMarkersEnabled())) {
        return;
    }
    auto command_access = scheduler.AcquireCommandBuffer();
    const auto cmdbuf = command_access.Handle();
    cmdbuf.endDebugUtilsLabelEXT();
}

void Rasterizer::ScopedMarkerInsert(const std::string_view& str, bool from_guest) {
    if ((from_guest && !EmulatorSettings.IsVkGuestMarkersEnabled()) ||
        (!from_guest && !EmulatorSettings.IsVkHostMarkersEnabled())) {
        return;
    }
    auto command_access = scheduler.AcquireCommandBuffer();
    const auto cmdbuf = command_access.Handle();
    cmdbuf.insertDebugUtilsLabelEXT(vk::DebugUtilsLabelEXT{
        .pLabelName = str.data(),
    });
}

void Rasterizer::ScopedMarkerInsertColor(const std::string_view& str, const u32 color,
                                         bool from_guest) {
    if ((from_guest && !EmulatorSettings.IsVkGuestMarkersEnabled()) ||
        (!from_guest && !EmulatorSettings.IsVkHostMarkersEnabled())) {
        return;
    }
    auto command_access = scheduler.AcquireCommandBuffer();
    const auto cmdbuf = command_access.Handle();
    cmdbuf.insertDebugUtilsLabelEXT(vk::DebugUtilsLabelEXT{
        .pLabelName = str.data(),
        .color = std::array<f32, 4>(
            {(f32)((color >> 16) & 0xff) / 255.0f, (f32)((color >> 8) & 0xff) / 255.0f,
             (f32)(color & 0xff) / 255.0f, (f32)((color >> 24) & 0xff) / 255.0f})});
}

} // namespace Vulkan
