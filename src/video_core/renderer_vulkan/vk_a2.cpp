// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "common/logging/log.h"
#include "shader_recompiler/native_emit.h"
#include "video_core/amdgpu/cp_split.h"
#include "video_core/amdgpu/native_rec.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/native_census.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/vk_a2.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_master_semaphore.h"

// Architecture A step A2, the shadow translator (v1). Everything here runs on a WORKER thread.
// What it reproduces: NativeOpen (shadow, !colour) + NativeRingPrepare's table and command math
// for a run of ring-described draws, and the generic draw tail's command-buffer state (pipeline,
// push constants, push descriptors, vertex input, index / vertex buffers, dynamic state) - as
// VALUES in a StretchCmds that the back replays in stream order (Replay below).
//
// Thread-safety review - every non-local object this file reads, and why that is safe here:
//  - Config: Mode / NumWorkers (own statics), VideoCore::g_native_pass, g_native_coherent,
//    AmdGpu::NativeRec::g_gpu_mode, Shader::g_native_ring / g_native_pt / g_native_seg: all
//    `inline const` initialised at load, never written.
//  - GraphicsPipeline: native_variant (atomic, acquire; the front stores it once), and the
//    variant's immutable fields (Handle - may wait a deferred compile, GetLayout, GetGraphicsKey,
//    GetFetchShader, native_*_binding) plus the base's stage Infos' persistent lists (buffers,
//    ud_mask), which only the compile thread writes before the pipeline is published.
//  - DrawIn::stages / vi: the Draw record's blocks in the CpSplit ring, valid while the worker
//    runs on this DCB (the caller's guarantee); read-only.
//  - NativeRec: g_rings / g_unions / g_gpu_handles (atomics, acquire), g_job_next (relaxed, a
//    lifetime bound only); Record and RunUnion through their stamp seqlocks exactly as
//    NativeRingPrepare reads them (fields first, acquire fence, stamp re-check).
//  - BufferCache: A2LookupBuffer (page_table find_acquire + stable SlotVector; see its comment),
//    StructGeneration (atomic), GetBdaPageTableBuffer / GetFaultBuffer handles and device
//    addresses (set at construction, never change).
//  - Instance: device limits and feature flags (immutable after creation).
//  - Scheduler: GetMasterSemaphore()->KnownGpuTick() (atomic load) for arena reuse only.
//  - The arena: this worker's own buffer (created here, host-coherent); the GPU reads a region
//    only after the back published the stretch (happens-after this thread's writes through the
//    Stretch handoff), and the region is reused only after that batch's tick passed.
//  - Nothing here records into the scheduler, touches the stream buffer, the texture cache,
//    barriers or any Rasterizer member. Memory validity (CPU-dirty pages) is the submission's
//    coherence point (SHADPS4_NATIVE_COHERENT, required); GPU-written index / vertex data is
//    covered by the back's conservative per-stretch global barrier rule (architecture-a sec. 6).

namespace Vulkan::A2 {

namespace NR = AmdGpu::NativeRec;

namespace {

constexpr u32 MaxDraws = 64;                              // Rasterizer::NativeMaxDraws
constexpr u32 CmdBytes = 20;                              // VkDrawIndexedIndirectCommand
constexpr u32 TableBytes = Shader::NativeTableDwords * 4; // the per-batch table
constexpr u64 ArenaBytes = u64{8} << 20;                  // per worker
constexpr u64 MaxSpan = u64{64} << 20;                    // NativeRingPrepare's union bound

constexpr u64 AlignUp(u64 v, u64 a) {
    return (v + a - 1) & ~(a - 1);
}

/// SHADPS4_STENCIL_REPLACE_OP (vk_rasterizer.cpp g_replace_op_reference: exactly "1").
const bool g_replace_op = [] {
    const char* v = std::getenv("SHADPS4_STENCIL_REPLACE_OP");
    return v && std::strcmp(v, "1") == 0;
}();

constexpr std::array<const char*, NumRefusals> RefusalNames = {
    "off",           "not indexed",     "indirect",       "no variant",      "variant kind",
    "no vi",         "no job",          "no ring",        "stamp",           "lifetime",
    "record fields", "inherited slot",  "union",          "slot kind",       "guest lookup",
    "index lookup",  "vertex lookup",   "record vs regs", "instance offset", "vertex offset",
    "restart index", "arena",           "stages",         "layout",          "no cmd (once)",
};

/// vk_rasterizer.cpp GetDrawOffsets, verbatim.
std::pair<u32, u32> GetDrawOffsets(const AmdGpu::Regs& regs, const Shader::BoundStage& info,
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

/// vk_rasterizer.cpp MakeUserData, verbatim.
Shader::PushData MakePushData(const AmdGpu::Regs& regs) {
    Shader::PushData push_data{};
    push_data.xoffset = regs.viewport_control.xoffset_enable ? regs.viewports[0].xoffset : 0.f;
    push_data.xscale = regs.viewport_control.xscale_enable ? regs.viewports[0].xscale : 1.f;
    push_data.yoffset = regs.viewport_control.yoffset_enable ? regs.viewports[0].yoffset : 0.f;
    push_data.yscale = regs.viewport_control.yscale_enable ? regs.viewports[0].yscale : 1.f;
    return push_data;
}

/// buffer_cache.cpp BuildVertexInputArrays, verbatim (it sits in that file's namespace).
void BuildVertexInputArrays(const AmdGpu::CpSplit::VertexInputsPayload& vi, SubBatch& out) {
    const auto* attribs = AmdGpu::CpSplit::VertexInputAttribs(vi);
    out.vi_count = vi.count;
    for (u32 i = 0; i < vi.count; ++i) {
        const auto& desc = attribs[i].desc;
        out.vi_attrs[i] = vk::VertexInputAttributeDescription2EXT{
            .location = desc.location,
            .binding = desc.location,
            .format = static_cast<vk::Format>(desc.vk_format),
            .offset = 0,
        };
        out.vi_bindings[i] = vk::VertexInputBindingDescription2EXT{
            .binding = desc.location,
            .stride = desc.stride,
            .inputRate = desc.instanced ? vk::VertexInputRate::eInstance
                                        : vk::VertexInputRate::eVertex,
            .divisor = desc.divisor,
        };
    }
}

// Rasterizer::UpdateViewportScissorState / UpdateDepthStencilState / UpdatePrimitiveState /
// UpdateRasterizationState / UpdateColorBlendingState as free functions of (regs, instance,
// pipeline): pure functions of the register mirror and device limits (the viewport memo and
// the hair trace are left out; attachment_feedback_loop is false: a shadow pass samples no
// bound target).

void ComputeViewportScissor(DynamicState& dynamic_state, const Instance& instance,
                            const AmdGpu::Regs& regs) {
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

    Viewports viewports;
    Scissors scissors;

    const auto& vp_ctl = regs.viewport_control;
    for (u32 i = 0; i < AmdGpu::NUM_VIEWPORTS; i++) {
        const auto& vp = regs.viewports[i];
        if (vp.xscale == 0) {
            continue;
        }

        const auto zoffset = vp_ctl.zoffset_enable ? vp.zoffset : 0.f;
        const auto zscale = vp_ctl.zscale_enable ? vp.zscale : 1.f;

        vk::Viewport viewport{};
        if (regs.clipper_control.clip_space == AmdGpu::ClipSpace::MinusWToW) {
            viewport.minDepth = zoffset - zscale;
            viewport.maxDepth = zoffset + zscale;
        } else {
            viewport.minDepth = zoffset;
            viewport.maxDepth = zoffset + zscale;
        }
        if (!instance.IsDepthRangeUnrestrictedSupported()) {
            viewport.minDepth = std::max(viewport.minDepth, 0.f);
            viewport.maxDepth = std::min(viewport.maxDepth, 1.f);
        }
        if (regs.IsClipDisabled()) {
            viewport.x = 0.f;
            viewport.y = 0.f;
            viewport.width = float(std::min<u32>(instance.GetMaxViewportWidth(), 16384u));
            viewport.height = float(std::min<u32>(instance.GetMaxViewportHeight(), 16384u));
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
    dynamic_state.SetViewports(viewports);
    dynamic_state.SetScissors(scissors);
}

void ComputeDepthStencil(DynamicState& dynamic_state, const AmdGpu::Regs& regs) {
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
        const auto reference = [&](const AmdGpu::StencilRefMask& face, AmdGpu::CompareFunc compare,
                                   AmdGpu::StencilFunc fail, AmdGpu::StencilFunc pass,
                                   AmdGpu::StencilFunc depth_fail) -> u32 {
            if (g_replace_op && compare == AmdGpu::CompareFunc::Always &&
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
        dynamic_state.SetStencilReferences(front_reference, back_reference);
        dynamic_state.SetStencilWriteMasks(!stencil_clear ? front.stencil_write_mask : 0U,
                                           !stencil_clear ? back.stencil_write_mask : 0U);
        dynamic_state.SetStencilCompareMasks(front.stencil_mask, back.stencil_mask);
    }
}

/// False: primitive restart with an index other than -1 (the back ASSERTs; a worker refuses).
bool ComputePrimitive(DynamicState& dynamic_state, const Instance& instance,
                      const AmdGpu::Regs& regs) {
    const auto is_list_topology = [](const AmdGpu::PrimitiveType type) {
        const auto topology = LiverpoolToVK::PrimitiveType(type);
        return topology == vk::PrimitiveTopology::ePointList ||
               topology == vk::PrimitiveTopology::eLineList ||
               topology == vk::PrimitiveTopology::eTriangleList ||
               topology == vk::PrimitiveTopology::eLineListWithAdjacency ||
               topology == vk::PrimitiveTopology::eTriangleListWithAdjacency;
    };
    const auto is_patch_list_topology = [](const AmdGpu::PrimitiveType type) {
        return type == AmdGpu::PrimitiveType::PatchPrimitive ||
               type == AmdGpu::PrimitiveType::QuadList || type == AmdGpu::PrimitiveType::RectList;
    };
    const auto prim_restart =
        (regs.enable_primitive_restart & 1) != 0 &&
        (instance.IsListRestartSupported() || !is_list_topology(regs.primitive_type)) &&
        (instance.IsPatchListRestartSupported() || !is_patch_list_topology(regs.primitive_type));
    if (prim_restart && regs.primitive_restart_index != 0xFFFF &&
        regs.primitive_restart_index != 0xFFFFFFFF) {
        return false;
    }
    const auto cull_mode = LiverpoolToVK::IsPrimitiveCulled(regs.primitive_type)
                               ? LiverpoolToVK::CullMode(regs.polygon_control.CullingMode())
                               : vk::CullModeFlagBits::eNone;
    const auto front_face = LiverpoolToVK::FrontFace(regs.polygon_control.front_face);
    dynamic_state.SetPrimitiveRestartEnabled(prim_restart);
    dynamic_state.SetRasterizerDiscardEnabled(regs.clipper_control.dx_rasterization_kill);
    dynamic_state.SetCullMode(cull_mode);
    dynamic_state.SetFrontFace(front_face);
    return true;
}

bool ComputeDynamicState(DynamicState& ds, const Instance& instance, const AmdGpu::Regs& regs,
                         const GraphicsPipeline* pipeline) {
    ComputeViewportScissor(ds, instance, regs);
    ComputeDepthStencil(ds, regs);
    if (!ComputePrimitive(ds, instance, regs)) {
        return false;
    }
    ds.SetLineWidth(regs.line_control.Width());
    ds.SetBlendConstants(regs.blend_constants);
    ds.SetColorWriteMasks(pipeline->GetGraphicsKey().write_masks);
    ds.SetAttachmentFeedbackLoopEnabled(false);
    return true;
}

/// The push-constant bytes the native shaders read: the viewport scale / offset floats. The
/// ud_regs are per draw (the VS reads them from the record, the FS has none) and buf_offsets
/// are zero, so they are not a batching key.
bool PushKeySame(const Shader::PushData& a, const Shader::PushData& b) {
    static_assert(offsetof(Shader::PushData, ud_regs) == 4 * sizeof(float));
    return std::memcmp(&a, &b, offsetof(Shader::PushData, ud_regs)) == 0;
}

bool AnyDirty(const DynamicState& ds) {
    // The dirty bits are a bit-field struct whose bytes are only ever memset (Invalidate /
    // ClearDirty) or bit-set: compare them with zero.
    constexpr size_t Bytes = sizeof(DynamicState::dirty_state);
    std::array<u8, Bytes> zero{};
    return std::memcmp(&ds.dirty_state, zero.data(), Bytes) != 0;
}

void ClearDirty(DynamicState& ds) {
    std::memset(&ds.dirty_state, 0, sizeof(ds.dirty_state));
}

} // namespace

int Mode() {
    static const int mode = [] {
        const char* v = std::getenv("SHADPS4_A2");
        return v && (v[0] == '1' || v[0] == '2') && v[1] == 0 ? v[0] - '0' : 0;
    }();
    return mode;
}

u32 NumWorkers() {
    static const u32 n = [] {
        const char* v = std::getenv("SHADPS4_A2_WORKERS");
        const long w = v ? std::strtol(v, nullptr, 10) : 3;
        return static_cast<u32>(std::clamp<long>(w <= 0 ? 3 : w, 1, 8));
    }();
    return n;
}

const char* RefusalName(u32 reason) {
    return reason < NumRefusals ? RefusalNames[reason] : "?";
}

void Replay(const StretchCmds& cmds, vk::CommandBuffer cmdbuf, const Instance& instance) {
    for (const auto& b : cmds.batches) {
        if (b.emit_pipeline) {
            cmdbuf.bindPipeline(vk::PipelineBindPoint::eGraphics, b.pipeline);
        }
        if (b.emit_push) {
            cmdbuf.pushConstants(b.layout, AllGraphicsStageBits, 0u, sizeof(b.push), &b.push);
        }
        if (b.emit_descriptors && b.num_writes != 0) {
            std::array<vk::DescriptorBufferInfo, Shader::NUM_BUFFERS + 4> infos;
            std::array<vk::WriteDescriptorSet, Shader::NUM_BUFFERS + 4> writes;
            for (u32 i = 0; i < b.num_writes; ++i) {
                const auto& w = b.writes[i];
                infos[i] = vk::DescriptorBufferInfo{w.buffer, w.offset, w.range};
                writes[i] = vk::WriteDescriptorSet{
                    .dstSet = VK_NULL_HANDLE,
                    .dstBinding = w.binding,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = vk::DescriptorType::eStorageBuffer,
                    .pBufferInfo = &infos[i],
                };
            }
            cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eGraphics, b.layout, 0,
                                        vk::ArrayProxy<const vk::WriteDescriptorSet>{
                                            b.num_writes, writes.data()});
        }
        if (b.emit_vertex_input) {
            cmdbuf.setVertexInputEXT(
                vk::ArrayProxy<const vk::VertexInputBindingDescription2EXT>{b.vi_count,
                                                                             b.vi_bindings.data()},
                vk::ArrayProxy<const vk::VertexInputAttributeDescription2EXT>{b.vi_count,
                                                                               b.vi_attrs.data()});
        }
        if (b.emit_index) {
            cmdbuf.bindIndexBuffer(b.ib, 0, b.ib_type);
        }
        if (b.emit_vertex_buffers && b.vb_n != 0) {
            cmdbuf.bindVertexBuffers(0, b.vb_n, b.vb.data(), b.vb_off.data());
        }
        DynamicState dyn = b.dyn; // Commit clears dirty bits: work on a copy
        dyn.Commit(instance, cmdbuf);
        cmdbuf.drawIndexedIndirectCount(b.arena, b.cmd_off, b.arena, b.count_off, b.n, CmdBytes);
    }
}

struct StretchTranslator::Impl {
    explicit Impl(StretchTranslator* owner_, const WorkerContext& ctx_);

    bool Take(const AmdGpu::Regs& regs, const DrawIn& draw);
    std::unique_ptr<Stretch> End();

    /// A region of `bytes` from the arena ring at `align`; never waits. False = none free.
    /// `start` = its unmasked start (the fence bookkeeping), `masked_off` its buffer offset.
    bool AllocRegion(u64 bytes, u64& start, u64& masked_off, u8*& ptr);

    StretchTranslator* owner;
    WorkerContext ctx;
    std::unique_ptr<VideoCore::Buffer> arena_buffer; ///< the Arena's VkBuffer (owned here)
    bool enabled{};
    bool cmd_warned{};
    u64 align{256};
    u64 region_bytes{};
    vk::Buffer dummy{}; ///< the always-valid Guest stand-in (BDA page table buffer)
    u64 pt_fallback{}, fault_addr{};

    // The open stretch.
    std::unique_ptr<Stretch> stretch;
    std::shared_ptr<StretchCmds> cmds;
    u64 stretch_start{};
    u64 last_desc_id{};
    DynamicState dyn{}; ///< tracked state of the stretch (dirty bits cleared per sub-batch)

    // The open sub-batch and its keys.
    bool batch_open{};
    u32* count_ptr{};
    u8* cmd_ptr{};
    const GraphicsPipeline* b_native{};
    u32 b_slot{}, b_next_idx{}, b_union{}, b_head{};
    u32 b_push_inh{}, b_inh_vs{}, b_inh_ps{};
    std::array<u32, 16> b_ud_vs{}, b_ud_ps{};
    u64 b_ib{};
    u32 b_index_size{};
    u32 b_vb_n{};
    std::array<u64, 16> b_vb{}, b_vcpu{}, b_vb_off{};
    std::array<u32, 16> b_stride{};
    std::array<u8, 16> b_inst{};
    u64 b_guest_mask{};
    std::array<VAddr, Shader::NUM_BUFFERS> b_glo{}, b_ghi{};
    Shader::PushData b_push{};
};

StretchTranslator::Impl::Impl(StretchTranslator* owner_, const WorkerContext& ctx_)
    : owner{owner_}, ctx{ctx_} {
    const char* why = nullptr;
    if (Mode() == 0) {
        why = "SHADPS4_A2 off";
    } else if (VideoCore::g_native_pass != 1) {
        why = "SHADPS4_NATIVE_PASS != 1";
    } else if (NR::g_gpu_mode != 1 || !Shader::g_native_ring) {
        why = "SHADPS4_NATIVE_GPU_DIRECT != 1";
    } else if (Shader::g_native_pt != 1) {
        why = "SHADPS4_NATIVE_PT != 1";
    } else if (Shader::g_native_seg != 0) {
        why = "SHADPS4_NATIVE_SEG on (list-addressed records)";
    } else if (VideoCore::g_native_coherent == 0) {
        why = "SHADPS4_NATIVE_COHERENT off";
    } else if (!ctx.instance || !ctx.scheduler || !ctx.buffer_cache || !ctx.arena) {
        why = "context incomplete";
    } else if (!ctx.instance->IsVertexInputDynamicState()) {
        why = "no dynamic vertex input";
    }
    if (!why) {
        align = std::max<u64>(ctx.instance->StorageMinAlignment(), 16);
        region_bytes = AlignUp(TableBytes + u64(MaxDraws) * CmdBytes + 16, align);
        if (!ctx.arena->buffer) {
            // Host-visible, coherent (Upload prefers HOST_COHERENT), storage + indirect, with a
            // device address as the Arena interface asks (its own dedicated allocation).
            arena_buffer = std::make_unique<VideoCore::Buffer>(
                *ctx.instance, *ctx.scheduler, VideoCore::MemoryUsage::Upload, 0,
                vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eIndirectBuffer |
                    vk::BufferUsageFlagBits::eShaderDeviceAddress,
                ArenaBytes);
            if (!arena_buffer->is_coherent || arena_buffer->mapped_data.empty()) {
                why = "arena memory not host-coherent / not mapped";
            } else {
                ctx.arena->buffer = arena_buffer->Handle();
                ctx.arena->mapped = arena_buffer->mapped_data.data();
                ctx.arena->device_address = arena_buffer->BufferDeviceAddress();
                ctx.arena->size = ArenaBytes;
                ctx.arena->head = 0;
            }
        }
    }
    if (!why) {
        const auto* pt = ctx.buffer_cache->GetBdaPageTableBuffer();
        dummy = pt->Handle();
        pt_fallback = pt->BufferDeviceAddress();
        fault_addr = ctx.buffer_cache->GetFaultBuffer()->BufferDeviceAddress();
        enabled = true;
        LOG_INFO(Render_Vulkan, "A2 worker {}: translator on (arena {} MB, region {} B, align {})",
                 ctx.worker_index, ArenaBytes >> 20, region_bytes, align);
    } else {
        LOG_INFO(Render_Vulkan, "A2 worker {}: translator off ({})", ctx.worker_index, why);
    }
}

bool StretchTranslator::Impl::AllocRegion(u64 bytes, u64& start, u64& masked_off, u8*& ptr) {
    Arena& ar = *ctx.arena;
    u64 a = AlignUp(ar.head, align);
    if ((a % ar.size) + bytes > ar.size) {
        a += ar.size - (a % ar.size); // never straddle the wrap: pad to the next lap
    }
    const u64 end = a + bytes;
    // The open stretch's own regions must not alias (a stretch longer than the arena).
    if (stretch && end - stretch_start > ar.size) {
        return false;
    }
    // Every published region one lap back must be past its batch tick; never wait (the back
    // may be blocked on this worker's DCB).
    if (end > ar.size) {
        const u64 reuse_below = end - ar.size;
        u64 known = 0;
        bool known_read = false;
        while (!ar.fences.empty() && ar.fences.front().start < reuse_below) {
            const u64 tick = ar.fences.front().tick->load(std::memory_order_acquire);
            if (tick == 0) {
                return false; // not published yet
            }
            if (!known_read) {
                known = ctx.scheduler->GetMasterSemaphore()->KnownGpuTick();
                known_read = true;
            }
            if (known < tick) {
                return false; // the GPU has not passed it (the back refreshes the tick)
            }
            ar.fences.erase(ar.fences.begin());
        }
    }
    ar.head = end;
    start = a;
    masked_off = a % ar.size;
    ptr = ar.mapped + masked_off;
    return true;
}

bool StretchTranslator::Impl::Take(const AmdGpu::Regs& regs, const DrawIn& draw) {
    const auto refuse = [&](u32 why) {
        ++owner->refusals[why];
        return false;
    };
    if (!enabled) {
        return refuse(RefuseOff);
    }
    if (!draw.is_indexed) {
        return refuse(RefuseNotIndexed);
    }
    if (draw.is_indirect) {
        return refuse(RefuseIndirect);
    }
    const GraphicsPipeline* base = draw.pipeline;
    const GraphicsPipeline* native =
        base ? base->native_variant.load(std::memory_order_acquire) : nullptr;
    if (!native) {
        return refuse(RefuseNoVariant);
    }
    if (native->native_records_binding == GraphicsPipeline::NoNativeRecords ||
        native->native_table_binding == GraphicsPipeline::NoNativeRecords ||
        native->native_pt_debug_binding != GraphicsPipeline::NoNativeRecords) {
        return refuse(RefuseVariantKind);
    }
    if (!draw.vi || draw.vi->count > 16) {
        return refuse(RefuseNoVi);
    }
    if (!draw.stages) {
        return refuse(RefuseStages);
    }
    const Shader::BoundStage& vs = (*draw.stages)[u32(Shader::LogicalStage::Vertex)];
    if (!vs.info) {
        return refuse(RefuseStages);
    }
    if (draw.job_tag == 0) {
        return refuse(RefuseNoJob);
    }
    const u32 slot = draw.job_tag >> 28;
    if (slot > NR::MaxRecordSlot) {
        return refuse(RefuseNoRing);
    }
    const NR::Record* ring = NR::g_rings[slot].load(std::memory_order_acquire);
    const NR::RunUnion* unions = NR::g_unions[slot].load(std::memory_order_acquire);
    const u64 ring_handle = NR::g_gpu_handles[slot].load(std::memory_order_acquire);
    if (!ring || !unions || ring_handle == 0) {
        return refuse(RefuseNoRing);
    }
    constexpr u32 mask = NR::RingSize - 1;
    const u32 idx = draw.job_idx;
    // Ring lifetime (NativeRingPrepare fail 2): the job thread must not be a quarter ring ahead.
    if (NR::g_job_next[slot].load(std::memory_order_relaxed) - idx > NR::RingSize / 4) {
        return refuse(RefuseLifetime);
    }
    // The job record (seqlock: fields, acquire fence, stamp).
    const NR::Record& r = ring[idx & mask];
    const u32 run_k = r.run_k, head_idx = r.head_idx, gpu_ok = r.gpu_ok;
    const u32 union_idx = r.union_idx, inst_known = r.inst_known, instances = r.instances;
    const u32 index_count = r.index_count, io_ud = r.io_ud, vcount = r.vcount;
    const u64 index_addr_rec = r.index_addr;
    const std::array<u64, 16> vaddr = r.vaddr;
    const void* cmd = r.cmd;
    const u32 push_inh = r.push_inh, inh_vs = r.inh_vs, inh_ps = r.inh_ps;
    const u64 inh_slots = r.inh_slots;
    std::atomic_thread_fence(std::memory_order_acquire);
    if (r.stamp.load(std::memory_order_relaxed) != idx + 1 || !gpu_ok) {
        return refuse(RefuseStamp);
    }
    if (draw.job_cmd) {
        if (cmd != draw.job_cmd) {
            return refuse(RefuseStamp);
        }
    } else if (!cmd_warned) {
        cmd_warned = true;
        ++owner->refusals[RefuseNoCmd];
    }
    if (!inst_known || io_ud != 0 || index_count == 0 || (push_inh >> Shader::NUM_USER_DATA_REGS) != 0) {
        return refuse(RefuseRecordFields);
    }
    if (inh_slots != 0) {
        // v1 (as NativeDirectBind fail 3): an inherited V#'s table words need the opener's cache
        // offsets; refuse rather than reproduce them.
        return refuse(RefuseInheritedSlot);
    }

    // The mirror against the record (NativePrepConfirm refuse 12 / 13, Opus G2 #3).
    const bool is_index16 = regs.index_buffer_type.index_type == AmdGpu::IndexType::Index16;
    const u32 index_size = is_index16 ? 2u : 4u;
    const VAddr index_addr =
        regs.index_base_address.Address<VAddr>() + u64(draw.index_offset) * index_size;
    if (index_addr != index_addr_rec || regs.num_indices != index_count ||
        regs.num_instances.NumInstances() != instances) {
        return refuse(RefuseRecordRegs);
    }
    const auto* attribs = AmdGpu::CpSplit::VertexInputAttribs(*draw.vi);
    const u32 vb_n = draw.vi->count;
    if (vcount != vb_n) {
        return refuse(RefuseRecordRegs);
    }
    std::array<VAddr, 16> vaddr_vi{};
    for (u32 k = 0; k < vb_n; ++k) {
        const auto& a = attribs[k];
        vaddr_vi[k] = (a.base_address != 0 && a.size != 0) ? a.base_address : 0;
        if (vaddr_vi[k] != vaddr[k]) {
            return refuse(RefuseRecordRegs);
        }
    }
    const auto [vertex_offset, instance_offset] = GetDrawOffsets(regs, vs, base->GetFetchShader());
    if (instance_offset != 0) {
        return refuse(RefuseInstanceOffset); // Opus code review #2 (drawIndirectFirstInstance)
    }

    // Set 0 in BindResources order (LogicalStage index order over the stages): every binding
    // must be fillable without the caches - Guest V#s (never read under PT: the dummy buffer),
    // unused clip planes (a null descriptor); then the records SSBO (the job ring) and the table.
    Shader::Backend::Bindings bnd{};
    Shader::PushData push = MakePushData(regs);
    std::array<SubBatch::Write, Shader::NUM_BUFFERS + 4> writes{};
    u32 num_writes = 0;
    u64 guest_mask = 0;
    std::array<VAddr, Shader::NUM_BUFFERS> o_addr{};
    u32 bi = 0;
    for (u32 s = 0; s < Shader::MaxStageTypes - 1; ++s) {
        const Shader::BoundStage& bound = (*draw.stages)[s];
        const Shader::Info* info = bound.info;
        if (!info) {
            continue;
        }
        if (!info->images.empty() || !info->samplers.empty()) {
            return refuse(RefuseSlotKind);
        }
        if (bnd.user_data + info->ud_mask.NumRegs() > Shader::NUM_USER_DATA_REGS) {
            return refuse(RefuseLayout);
        }
        bound.PushUd(bnd, push);
        for (const auto& desc : info->buffers) {
            if (bi >= Shader::NUM_BUFFERS || num_writes >= Shader::NUM_BUFFERS) {
                return refuse(RefuseLayout);
            }
            if (desc.buffer_type == Shader::BufferType::ClipPlanes) {
                if (regs.clipper_control.user_clip_plane_enable != 0) {
                    return refuse(RefuseSlotKind); // needs a stream-buffer copy of the planes
                }
                writes[num_writes++] = {bnd.unified++, vk::Buffer{}, 0, VK_WHOLE_SIZE};
                ++bnd.buffer;
                ++bi;
                continue;
            }
            if (desc.buffer_type != Shader::BufferType::Guest || desc.is_written ||
                desc.is_formatted) {
                return refuse(RefuseSlotKind);
            }
            const auto vsharp = desc.GetSharp(bound);
            o_addr[bi] = vsharp.base_address;
            guest_mask |= u64{1} << bi;
            writes[num_writes++] = {bnd.unified++, dummy, 0, VK_WHOLE_SIZE};
            ++bnd.buffer;
            ++bi;
        }
    }
    if (native->native_records_binding != bnd.unified ||
        native->native_table_binding != bnd.unified + 1 || guest_mask == 0) {
        return refuse(RefuseLayout);
    }
    push.buf_offsets = {}; // whole-range bindings: the record carries each draw's offset

    // Index buffer: the whole index range in one cached buffer, bound at offset 0.
    const VideoCore::Buffer* ib = ctx.buffer_cache->A2LookupBuffer(index_addr);
    if (!ib || !ib->IsInBounds(index_addr, u64(index_count) * index_size)) {
        return refuse(RefuseIndexLookup);
    }
    const u64 ib_off = index_addr - ib->CpuAddr();
    if (ib_off % index_size != 0) {
        return refuse(RefuseRecordRegs); // NativeRingPrepare why 11
    }
    const u64 ib_handle = std::bit_cast<u64>(static_cast<VkBuffer>(ib->Handle()));
    // Vertex streams: each non-null stream's buffer and its offset in it.
    std::array<u64, 16> vb_handle{}, vcpu{}, voff{};
    for (u32 k = 0; k < vb_n; ++k) {
        if (vaddr_vi[k] == 0) {
            continue;
        }
        const VideoCore::Buffer* vb = ctx.buffer_cache->A2LookupBuffer(vaddr_vi[k]);
        if (!vb) {
            return refuse(RefuseVertexLookup);
        }
        vb_handle[k] = std::bit_cast<u64>(static_cast<VkBuffer>(vb->Handle()));
        vcpu[k] = vb->CpuAddr();
        voff[k] = vaddr_vi[k] - vcpu[k];
    }

    // Dynamic state of this draw on top of the tracked state (a copy: a refusal below must not
    // change what the open sub-batch recorded). The first draw of a stretch starts from an
    // invalidated state, so its sub-batch records everything.
    DynamicState cand = dyn;
    if (!stretch) {
        cand.Invalidate();
    }
    if (!ComputeDynamicState(cand, *ctx.instance, regs, native)) {
        return refuse(RefuseRestartIndex);
    }

    // Can the draw continue the open sub-batch? Keys: the variant, the next ring entry of the
    // same job slot and run, the inherited set and the inherited registers' values, the same
    // index / vertex VkBuffers and strides, the same push constants, vertex description and
    // dynamic state, every Guest V# inside the buffers the opener's table bases name, the
    // vertex rebase quotient rule (below), fewer than MaxDraws draws.
    const u32 head = run_k == 0 ? idx : head_idx;
    bool cont = batch_open && b_native == native && b_slot == slot && idx == b_next_idx &&
                cmds->batches.back().n < MaxDraws && union_idx == b_union && head == b_head &&
                push_inh == b_push_inh && inh_vs == b_inh_vs && inh_ps == b_inh_ps &&
                ib_handle == b_ib && index_size == b_index_size && vb_n == b_vb_n &&
                draw.vi->desc_id == last_desc_id && PushKeySame(push, b_push) &&
                guest_mask == b_guest_mask && !AnyDirty(cand);
    for (u32 m = inh_vs; cont && m != 0; m &= m - 1) {
        const u32 i = static_cast<u32>(std::countr_zero(m));
        cont = i < 16 && regs.vs_program.user_data[i] == b_ud_vs[i];
    }
    for (u32 m = inh_ps; cont && m != 0; m &= m - 1) {
        const u32 i = static_cast<u32>(std::countr_zero(m));
        cont = i < 16 && regs.ps_program.user_data[i] == b_ud_ps[i];
    }
    for (u32 k = 0; cont && k < vb_n; ++k) {
        cont = vb_handle[k] == b_vb[k] && attribs[k].desc.stride == b_stride[k] &&
               u8(attribs[k].desc.instanced) == b_inst[k];
    }
    for (u64 m = guest_mask; cont && m != 0; m &= m - 1) {
        const u32 b = static_cast<u32>(std::countr_zero(m));
        cont = o_addr[b] >= b_glo[b] && o_addr[b] < b_ghi[b];
    }
    // The member's vertexOffset against the opener's rebased bindings (NativeRingPrepare's
    // d / q loop): one common quotient over the per-vertex streams; instance-rate and
    // stride-0 streams may not move. A failure ends the sub-batch, not the stretch.
    s64 vo = 0;
    if (cont) {
        s64 q = 0;
        bool q_set = false;
        bool ok = true;
        for (u32 k = 0; ok && k < vb_n; ++k) {
            if (vb_handle[k] == 0) {
                continue; // null here and (same handles checked) null at the opener
            }
            const s64 d = s64(voff[k]) - s64(b_vb_off[k]);
            if (attribs[k].desc.instanced || attribs[k].desc.stride == 0) {
                ok = d == 0;
                continue;
            }
            const s64 stride = s64(attribs[k].desc.stride);
            if (d % stride != 0) {
                ok = false;
                break;
            }
            const s64 qa = d / stride;
            if (q_set && qa != q) {
                ok = false;
                break;
            }
            q = qa;
            q_set = true;
        }
        vo = s64(s32(vertex_offset)) + q;
        cont = ok && vo >= 0 && vo <= INT32_MAX;
    }

    // Open a sub-batch (and the stretch if none is open).
    if (!cont) {
        // Rebase (NativeOpen's !colour block): the per-vertex strided streams to the lowest
        // offset of their stride residue if they share one quotient, else the full offsets.
        std::array<u64, 16> bo = voff;
        u64 q0 = 0;
        {
            bool ok = true, any = false;
            u64 qq = 0;
            for (u32 k = 0; k < vb_n; ++k) {
                if (vb_handle[k] == 0 || attribs[k].desc.instanced || attribs[k].desc.stride == 0) {
                    continue;
                }
                const u64 qk = voff[k] / attribs[k].desc.stride;
                if (any && qk != qq) {
                    ok = false;
                }
                qq = qk;
                any = true;
            }
            if (ok && any && qq <= 0x7fffffffull) {
                for (u32 k = 0; k < vb_n; ++k) {
                    if (vb_handle[k] == 0 || attribs[k].desc.instanced ||
                        attribs[k].desc.stride == 0) {
                        continue;
                    }
                    bo[k] = voff[k] % attribs[k].desc.stride;
                }
                q0 = qq;
            }
        }
        vo = s64(s32(vertex_offset)) + s64(q0);
        if (vo < 0 || vo > INT32_MAX) {
            return refuse(RefuseVertexOffset);
        }
        // The run union (NativeRingPrepare): per Guest slot the buffer holding the run's range
        // (its CPU address low 32 bits is the table base), or the opener's own V# when every
        // draw of the run binds the same one. Seqlock as the back reads it.
        const NR::RunUnion& u = unions[union_idx & NR::UnionMask];
        const u32 un = u.n.load(std::memory_order_acquire);
        if (u.stamp.load(std::memory_order_relaxed) != head + 1 || un <= run_k) {
            return refuse(RefuseUnion);
        }
        const u64 u_guest = u.guest_mask, u_eq = u.eq_mask;
        std::array<u64, Shader::NUM_BUFFERS> lo{}, hi{}, first{};
        for (u64 m = guest_mask; m != 0; m &= m - 1) {
            const u32 b = static_cast<u32>(std::countr_zero(m));
            lo[b] = u.lo[b];
            hi[b] = u.hi[b];
            first[b] = u.first_addr[b];
        }
        std::atomic_thread_fence(std::memory_order_acquire);
        if (u.stamp.load(std::memory_order_relaxed) != head + 1) {
            return refuse(RefuseUnion);
        }
        std::array<u32, Shader::NUM_BUFFERS> base{};
        std::array<VAddr, Shader::NUM_BUFFERS> glo{}, ghi{};
        for (u64 m = guest_mask; m != 0; m &= m - 1) {
            const u32 b = static_cast<u32>(std::countr_zero(m));
            if (!(u_guest & (u64{1} << b))) {
                return refuse(RefuseUnion);
            }
            const VideoCore::Buffer* buf = nullptr;
            if ((u_eq & (u64{1} << b)) && first[b] == o_addr[b]) {
                buf = ctx.buffer_cache->A2LookupBuffer(o_addr[b]);
            } else if (lo[b] != 0 && hi[b] > lo[b] && hi[b] - lo[b] <= MaxSpan) {
                buf = ctx.buffer_cache->A2LookupBuffer(lo[b]);
                if (buf && !buf->IsInBounds(lo[b], hi[b] - lo[b])) {
                    buf = nullptr;
                }
            }
            if (!buf || o_addr[b] < buf->CpuAddr() ||
                o_addr[b] - buf->CpuAddr() >= buf->SizeBytes()) {
                return refuse(RefuseGuestLookup);
            }
            base[b] = static_cast<u32>(buf->CpuAddr());
            glo[b] = buf->CpuAddr();
            ghi[b] = buf->CpuAddr() + buf->SizeBytes();
        }
        // The arena region: table, commands, count. The last check that can refuse.
        u64 region_start = 0;
        u64 region_off = 0;
        u8* region = nullptr;
        if (!AllocRegion(region_bytes, region_start, region_off, region)) {
            return refuse(RefuseArena);
        }
        const bool first_batch = !stretch;
        if (first_batch) {
            stretch = std::make_unique<Stretch>();
            stretch->first_seq = draw.seq;
            stretch->tick = std::make_shared<std::atomic<u64>>(0);
            stretch->buf_gen = ctx.buffer_cache->StructGeneration();
            cmds = std::make_shared<StretchCmds>();
            cmds->batches.reserve(8);
            stretch->cmds = cmds;
            stretch_start = region_start;
            last_desc_id = 0;
            b_native = nullptr;
            batch_open = false;
        }
        // The table (NativeRingPrepare + NativeOpen's ring-mode words; SegMat 0, no inherited
        // slots: InhSlots / Size / PtLo / PtHi stay 0).
        auto* table = reinterpret_cast<u32*>(region);
        std::memset(table, 0, TableBytes);
        table[Shader::NativeTableFirst] = idx & mask;
        table[Shader::NativeTableMask] = mask;
        table[Shader::NativeTablePtFallback] = static_cast<u32>(pt_fallback);
        table[Shader::NativeTablePtFallback + 1] = static_cast<u32>(pt_fallback >> 32);
        table[Shader::NativeTableFaultAddr] = static_cast<u32>(fault_addr);
        table[Shader::NativeTableFaultAddr + 1] = static_cast<u32>(fault_addr >> 32);
        table[Shader::NativeTablePushInh] = push_inh;
        for (u32 pm = push_inh; pm != 0; pm &= pm - 1) {
            const u32 pi = static_cast<u32>(std::countr_zero(pm));
            table[Shader::NativeTableUd + pi] = push.ud_regs[pi];
        }
        for (u64 m = guest_mask; m != 0; m &= m - 1) {
            const u32 b = static_cast<u32>(std::countr_zero(m));
            table[Shader::NativeTableBase + b] = base[b];
        }
        const u64 cmd_off = region_off + TableBytes;
        const u64 count_off = cmd_off + u64(MaxDraws) * CmdBytes;
        cmd_ptr = region + TableBytes;
        count_ptr = reinterpret_cast<u32*>(region + TableBytes + u64(MaxDraws) * CmdBytes);
        *count_ptr = 0;

        // The sub-batch record: what Replay emits before its indirect draw.
        SubBatch& sb = cmds->batches.emplace_back();
        const bool ib_changed = first_batch || ib_handle != b_ib || index_size != b_index_size;
        bool vb_changed = first_batch || vb_n != b_vb_n;
        for (u32 k = 0; !vb_changed && k < vb_n; ++k) {
            vb_changed = vb_handle[k] != b_vb[k] || bo[k] != b_vb_off[k];
        }
        sb.emit_pipeline = first_batch || native != b_native;
        sb.emit_push = first_batch || !PushKeySame(push, b_push);
        sb.emit_descriptors = true; // the table is a new region every sub-batch
        sb.emit_vertex_input = first_batch || draw.vi->desc_id != last_desc_id;
        sb.emit_index = ib_changed;
        sb.emit_vertex_buffers = vb_changed;
        sb.pipeline = native->Handle();
        sb.layout = native->GetLayout();
        sb.push = push;
        writes[num_writes++] = {native->native_records_binding,
                                vk::Buffer{std::bit_cast<VkBuffer>(ring_handle)}, 0, VK_WHOLE_SIZE};
        writes[num_writes++] = {native->native_table_binding, ctx.arena->buffer, region_off,
                                TableBytes};
        sb.num_writes = num_writes;
        sb.writes = writes;
        BuildVertexInputArrays(*draw.vi, sb);
        sb.ib = vk::Buffer{std::bit_cast<VkBuffer>(ib_handle)};
        sb.ib_type = is_index16 ? vk::IndexType::eUint16 : vk::IndexType::eUint32;
        sb.vb_n = vb_n;
        for (u32 k = 0; k < vb_n; ++k) {
            sb.vb[k] = vk::Buffer{std::bit_cast<VkBuffer>(vb_handle[k])};
            sb.vb_off[k] = vb_handle[k] != 0 ? bo[k] : 0;
        }
        sb.dyn = cand; // dirty bits = what changed (everything for the first sub-batch)
        sb.arena = ctx.arena->buffer;
        sb.cmd_off = cmd_off;
        sb.count_off = count_off;
        sb.n = 0;
        ClearDirty(cand);

        // The keys of the now-open sub-batch.
        batch_open = true;
        b_native = native;
        b_slot = slot;
        b_union = union_idx;
        b_head = head;
        b_push_inh = push_inh;
        b_inh_vs = inh_vs;
        b_inh_ps = inh_ps;
        b_ud_vs = regs.vs_program.user_data;
        b_ud_ps = regs.ps_program.user_data;
        b_ib = ib_handle;
        b_index_size = index_size;
        b_vb_n = vb_n;
        b_vb = vb_handle;
        b_vcpu = vcpu;
        b_vb_off = bo;
        for (u32 k = 0; k < 16; ++k) {
            b_stride[k] = k < vb_n ? attribs[k].desc.stride : 0;
            b_inst[k] = k < vb_n ? u8(attribs[k].desc.instanced) : 0;
        }
        b_guest_mask = guest_mask;
        b_glo = glo;
        b_ghi = ghi;
        b_push = push;
        last_desc_id = draw.vi->desc_id;
        ++owner->sub_batches;
    }

    // The draw's indirect command (NativeRingPrepare's member command: the GPU reads its record
    // at ring entry first + n).
    SubBatch& sb = cmds->batches.back();
    const u32 n = sb.n;
    const u32 cmdw[5] = {index_count, instances, static_cast<u32>(ib_off / index_size),
                         static_cast<u32>(s32(vo)), 0u};
    std::memcpy(cmd_ptr + u64(n) * CmdBytes, cmdw, CmdBytes);
    sb.n = n + 1;
    *count_ptr = n + 1;
    b_next_idx = idx + 1;
    dyn = cand;
    stretch->last_seq = draw.seq;
    ++stretch->draws;
    ++owner->taken;
    return true;
}

std::unique_ptr<Stretch> StretchTranslator::Impl::End() {
    if (!stretch) {
        return nullptr;
    }
    stretch->arena_end = ctx.arena->head;
    ctx.arena->fences.push_back({stretch_start, ctx.arena->head, stretch->tick});
    batch_open = false;
    b_native = nullptr;
    cmd_ptr = nullptr;
    count_ptr = nullptr;
    cmds.reset();
    return std::move(stretch);
}

StretchTranslator::StretchTranslator(const WorkerContext& ctx)
    : impl{std::make_unique<Impl>(this, ctx)} {}

StretchTranslator::~StretchTranslator() = default;

bool StretchTranslator::TakeDraw(const AmdGpu::Regs& regs, const DrawIn& draw) {
    return impl->Take(regs, draw);
}

std::unique_ptr<Stretch> StretchTranslator::EndStretch() {
    return impl->End();
}

} // namespace Vulkan::A2
