// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstdlib>
#include <string>
#include <unordered_set>

#include "common/debug.h"
#include "common/buffer_sample_probe.h"
#include "common/draw_sample_probe.h"
#include "common/fault_monitor.h"
#include "common/scope_exit.h"
#include "core/debug_state.h"
#include "core/emulator_settings.h"
#include "core/memory.h"
#include "shader_recompiler/runtime_info.h"
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
static void ProbeBindingChanges(const Pipeline* pipeline) {
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
    for (const auto* stage : pipeline->GetStages()) {
        if (!stage) {
            continue;
        }
        auto& prev = previous[static_cast<u32>(stage->l_stage)];
        ++stages;
        same_program += prev.info == stage;
        size_t i = 0;
        for (const auto& desc : stage->buffers) {
            const auto sharp = desc.GetSharp(*stage);
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
            const auto sharp = desc.GetSharp(*stage);
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

bool Rasterizer::FilterDraw() {
    const auto& regs = liverpool->regs;
    if (regs.color_control.mode == AmdGpu::ColorControl::OperationMode::EliminateFastClear) {
        // Clears the render target if FCE is launched before any draws
        EliminateFastClear();
        return false;
    }
    if (regs.color_control.mode == AmdGpu::ColorControl::OperationMode::FmaskDecompress) {
        // TODO: check for a valid MRT1 to promote the draw to the resolve pass.
        LOG_TRACE(Render_Vulkan, "FMask decompression pass skipped");
        ScopedMarkerInsert("FmaskDecompress");
        return false;
    }
    if (regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Resolve) {
        LOG_TRACE(Render_Vulkan, "Resolve pass");
        Resolve();
        return false;
    }
    if (regs.primitive_type == AmdGpu::PrimitiveType::None) {
        LOG_TRACE(Render_Vulkan, "Primitive type 'None' skipped");
        ScopedMarkerInsert("PrimitiveTypeNone");
        return false;
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
        // Games may disable color buffer and enable force depth/stencil dirty and valid to
        // do a copy from one depth-stencil surface to another, without a pixel shader.
        // We need to detect this case and perform the copy, otherwise it will have no effect.
        LOG_TRACE(Render_Vulkan, "Performing depth-stencil override copy");
        DepthStencilCopy(depth_copy, stencil_copy);
        return false;
    }

    return true;
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
    const auto& regs = liverpool->regs;
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
        const auto& hint = liverpool->last_cb_extent[cb];
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
        const auto& hint = liverpool->last_db_extent;
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
    const AmdGpu::Regs& regs, const Shader::Info& info,
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
    auto& col_buf = liverpool->regs.color_buffers[0];
    if (!col_buf || !col_buf.info.fast_clear) {
        return;
    }
    VideoCore::TextureCache::ImageDesc desc(col_buf, liverpool->last_cb_extent[0]);
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

void Rasterizer::Draw(bool is_indexed, u32 index_offset) {
    RENDERER_TRACE;

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

    const auto& regs = liverpool->regs;
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
    const auto state = BeginRendering(pipeline);
    sample.EndPhase(Common::DrawSampleProbe::Phase::BuildRenderingState);
    const auto t_vtxbuf = Common::FaultMonitor::Now();
    Common::FaultMonitor::CountDraw(pipeline_ns, bindres_ns,
                                    Common::FaultMonitor::DiffNs(t_begin, t_vtxbuf));

    sample.BeginPhase(Common::DrawSampleProbe::Phase::VertexIndexBuffers);
    buffer_cache.BindVertexBuffers(*pipeline, buffer_barriers);
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
    const auto& vs_info = pipeline->GetStage(Shader::LogicalStage::Vertex);
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

    PrepareRenderState(pipeline);
    if (!BindResources(pipeline)) {
        return;
    }
    const auto state = BeginRendering(pipeline);

    buffer_cache.BindVertexBuffers(*pipeline, buffer_barriers);
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

    const auto& cs_program = liverpool->GetCsRegs();
    const ComputePipeline* pipeline = pipeline_cache.GetComputePipeline();
    if (!pipeline) {
        return;
    }

    const auto& cs = pipeline->GetStage(Shader::LogicalStage::Compute);
    if (ExecuteShaderHLE(cs, liverpool->regs, cs_program, *this)) {
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

    const auto& cs_program = liverpool->GetCsRegs();
    const ComputePipeline* pipeline = pipeline_cache.GetComputePipeline();
    if (!pipeline) {
        return;
    }

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

void Rasterizer::OnSubmit() {
    if (fault_process_pending) {
        fault_process_pending = false;
        buffer_cache.ProcessFaultBuffer();
    }
    texture_cache.ProcessDownloadImages();
    texture_cache.RunGarbageCollector();
    buffer_cache.RunGarbageCollector();
}

bool Rasterizer::BindResources(const Pipeline* pipeline) {
    if (IsComputeImageCopy(pipeline) || IsComputeMetaClear(pipeline) ||
        IsComputeImageClear(pipeline)) {
        return false;
    }

#if SHADPS4_BINDING_PROBE
    ProbeBindingChanges(pipeline);
#endif

    set_write_index = 0;
    set_writes.clear();
    buffer_barriers.clear();
    buffer_infos.clear();
    image_infos.clear();

    bool uses_dma = false;

    // Bind resource buffers and textures.
    Shader::Backend::Bindings binding{};
    push_data = MakeUserData(liverpool->regs);
    for (const auto* stage : pipeline->GetStages()) {
        if (!stage) {
            continue;
        }
        if (build_set_writes) {
            set_writes.resize(set_writes.size() + stage->buffers.size() + stage->images.size() +
                              stage->samplers.size());
        }
        stage->PushUd(binding, push_data);
        // Split measured because BindResources is 6.8 us of the 8.0 us spent per draw - 85% - and
        // a stack scan could not tell these two apart from anything else.
        const auto t_buf = Common::FaultMonitor::Now();
        BindBuffers(*stage, binding, push_data);
        const auto t_tex = Common::FaultMonitor::Now();
        BindTextures(*stage, binding);
        Common::FaultMonitor::CountBindSplit(Common::FaultMonitor::DiffNs(t_buf, t_tex),
                                             Common::FaultMonitor::SinceNs(t_tex));
        uses_dma |= stage->uses_dma;
    }
    native_bound_count = binding.unified;

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

    // Assume if a shader reads metadata, it is a copy shader.
    for (const auto& desc : info.buffers) {
        const VAddr address = desc.GetSharp(info).base_address;
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
            const VAddr address = desc.GetSharp(info).base_address;
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
    const auto& cs_pgm = liverpool->GetCsRegs();
    const auto& info = pipeline->GetStage(Shader::LogicalStage::Compute);
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
    const AmdGpu::Buffer buf0 = desc0.GetSharp(info);
    const AmdGpu::Buffer buf1 = desc1.GetSharp(info);
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
    const auto& cs_pgm = liverpool->GetCsRegs();
    const auto& info = pipeline->GetStage(Shader::LogicalStage::Compute);
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
    const AmdGpu::Buffer buf0 = desc0.GetSharp(info);
    const AmdGpu::Buffer buf1 = desc1.GetSharp(info);
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

void Rasterizer::BindBuffers(const Shader::Info& stage, Shader::Backend::Bindings& binding,
                             Shader::PushData& push_data) {
    buffer_bindings.clear();

    // Pass 1 (FindBuffer) vs pass 2 (ObtainBuffer -> SynchronizeBuffer) timed separately: this
    // function is 62% of per-draw cost and a stack scan is not trustworthy for apportioning it.
    const auto t_find = Common::FaultMonitor::Now();
    for (const auto& desc : stage.buffers) {
        const auto vsharp = desc.GetSharp(stage);
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
                const u32 ubo_size = stage.flattened_ud_buf.size() * sizeof(u32);
                const u64 offset =
                    vk_buffer.Copy(stage.flattened_ud_buf.data(), ubo_size, alignment);
                buffer_infos.emplace_back(vk_buffer.Handle(), offset, ubo_size);
                bound_buffer = &vk_buffer;
            } else if (desc.buffer_type == Shader::BufferType::ClipPlanes) {
                // Permutations compiled without enabled planes never read the buffer, so the
                // declared binding is satisfied with a null descriptor instead of a copy.
                if (liverpool->regs.clipper_control.user_clip_plane_enable == 0) {
                    buffer_infos.emplace_back(VK_NULL_HANDLE, 0, VK_WHOLE_SIZE);
                } else {
                    auto& vk_buffer = buffer_cache.GetUtilityBuffer(VideoCore::MemoryUsage::Stream);
                    std::array<float, AmdGpu::NUM_CLIP_PLANES * 4> planes{};
                    for (u32 i = 0; i < AmdGpu::NUM_CLIP_PLANES; ++i) {
                        const auto& plane = liverpool->regs.clip_user_data[i];
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
                const auto& cs_program = liverpool->GetCsRegs();
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
            const auto [vk_buffer, offset] = buffer_cache.ObtainBuffer(
                vsharp.base_address, size, desc.is_written, desc.is_formatted, buffer_id,
                is_storage ? Common::BufferSampleCaller::Storage
                           : Common::BufferSampleCaller::Uniform);
            obtain_ns += Common::FaultMonitor::SinceNs(t_obtain);
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
            if (desc.is_written && desc.is_formatted) {
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
    }
    Common::FaultMonitor::CountBindBufferPass(find_ns, obtain_ns);
}

void Rasterizer::BindTextures(const Shader::Info& stage, Shader::Backend::Bindings& binding) {
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
        const auto tsharp = image_desc.GetSharp(stage);
        if (texture_cache.IsMeta(tsharp.Address())) {
            LOG_WARNING(Render_Vulkan, "Unexpected metadata read by a shader (texture)");
        }

        if (tsharp.Address() == 0 || tsharp.GetDataFmt() == AmdGpu::DataFormat::FormatInvalid) {
            image_bindings.emplace_back(std::piecewise_construct, std::tuple{}, std::tuple{});
            image_descriptor_array_sizes.push_back(1);
            continue;
        }

        const Shader::MipStorageFallbackMode mip_fallback_mode = image_desc.mip_fallback_mode;
        const u32 num_bindings = image_desc.NumBindings(stage);

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
            if (auto depth_image_id = texture_cache.GetAssociatedDepth(*image)) {
                // If this image has an associated depth image, it's a stencil attachment.
                // Redirect the access to the actual depth-stencil buffer.
                image_id = depth_image_id;
                image = &texture_cache.GetImage(image_id);
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
            if (auto& old_image = texture_cache.GetImage(image_id);
                old_image.binding.needs_rebind) {
                old_image.binding = {};
                image_id = texture_cache.FindImage(desc);
            }

            bound_images.emplace_back(image_id);

            auto& image = texture_cache.GetImage(image_id);
            const auto t_ft = Common::FaultMonitor::Now();
            auto& image_view = texture_cache.FindTexture(image_id, desc);
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
        auto ssharp = sampler.GetSharp(stage);
        if (sampler.disable_aniso) {
            const auto& tsharp = stage.images[sampler.associated_image].GetSharp(stage);
            if (tsharp.base_level == 0 && tsharp.last_level == 0) {
                ssharp.max_aniso.Assign(AmdGpu::AnisoRatio::One);
            }
        }
        u32 native_sampler_index{};
        const auto vk_sampler = texture_cache.GetSampler(
            ssharp, liverpool->regs.ta_bc_base, native_bindings ? &native_sampler_index : nullptr);
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
    const auto& regs = liverpool->regs;
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
    const auto& mrt0_hint = liverpool->last_cb_extent[0];
    const auto& mrt1_hint = liverpool->last_cb_extent[1];
    VideoCore::TextureCache::ImageDesc mrt0_desc{liverpool->regs.color_buffers[0], mrt0_hint};
    VideoCore::TextureCache::ImageDesc mrt1_desc{liverpool->regs.color_buffers[1], mrt1_hint};
    auto& mrt0_image = texture_cache.GetImage(texture_cache.FindImage(mrt0_desc, true));
    auto& mrt1_image = texture_cache.GetImage(texture_cache.FindImage(mrt1_desc, true));

    ScopeMarkerBegin(fmt::format("Resolve:MRT0={:#x}:MRT1={:#x}",
                                 liverpool->regs.color_buffers[0].Address(),
                                 liverpool->regs.color_buffers[1].Address()));
    mrt1_image.Resolve(mrt0_image, mrt0_desc.view_info.range, mrt1_desc.view_info.range);
    ScopeMarkerEnd();
}

void Rasterizer::DepthStencilCopy(bool is_depth, bool is_stencil) {
    auto& regs = liverpool->regs;

    auto read_desc = VideoCore::TextureCache::ImageDesc(
        regs.depth_buffer, regs.depth_view, regs.depth_control,
        regs.depth_htile_data_base.GetAddress(), liverpool->last_db_extent, false);
    auto write_desc = VideoCore::TextureCache::ImageDesc(
        regs.depth_buffer, regs.depth_view, regs.depth_control,
        regs.depth_htile_data_base.GetAddress(), liverpool->last_db_extent, true);

    auto& read_image = texture_cache.GetImage(texture_cache.FindImage(read_desc));
    auto& write_image = texture_cache.GetImage(texture_cache.FindImage(write_desc));

    VideoCore::SubresourceRange sub_range;
    sub_range.base.layer = liverpool->regs.depth_view.slice_start;
    sub_range.extent.layers = liverpool->regs.depth_view.NumSlices() - sub_range.base.layer;

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
        read_image.GetImage(), vk::ImageLayout::eTransferSrcOptimal, write_image.GetImage(),
        vk::ImageLayout::eTransferDstOptimal, region);

    ScopeMarkerEnd();
}

void Rasterizer::FillBuffer(VAddr address, u32 num_bytes, u32 value, bool is_gds) {
    buffer_cache.FillBuffer(address, num_bytes, value, is_gds);
}

void Rasterizer::CopyBuffer(VAddr dst, VAddr src, u32 num_bytes, bool dst_gds, bool src_gds) {
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
}

void Rasterizer::UpdateDynamicState(const GraphicsPipeline* pipeline, const bool is_indexed) const {
    UpdateViewportScissorState();
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
            const auto& regs = liverpool->regs;
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
    const auto& regs = liverpool->regs;

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
    const auto& regs = liverpool->regs;
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
    const auto& regs = liverpool->regs;
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
    const auto& regs = liverpool->regs;
    auto& dynamic_state = scheduler.GetDynamicState();
    dynamic_state.SetLineWidth(regs.line_control.Width());
}

void Rasterizer::UpdateColorBlendingState(const GraphicsPipeline* pipeline) const {
    const auto& regs = liverpool->regs;
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
