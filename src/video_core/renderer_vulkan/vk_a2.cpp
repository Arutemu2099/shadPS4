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

// Architecture A step A2, the stretch translator: v1 shadow draws, v2 colour draws (bindless
// textures + pulled vertices, "deferred material resolve"). Everything here runs on a WORKER
// thread except Replay (the recording lane).
// What it reproduces: NativeOpen (shadow and colour) + NativeRingPrepare's table and command
// math for a run of ring-described draws, and the generic draw tail's command-buffer state
// (pipeline, push constants, push descriptors, vertex input, index / vertex buffers, dynamic
// state) - as VALUES in a StretchCmds that the back replays in stream order (Replay below).
// Colour: the bindless slot words are the back's (BindlessResolve needs the texture cache), so
// the worker only keys each draw's material (raw T# / S# words as NativeImageKey) and leaves a
// SlotWrite per draw; the back resolves every material of the stretch once at publish and
// Replay writes the slot words into the job ring records as Rasterizer::BindlessRingWrite does.
//
// Thread-safety review - every non-local object this file reads, and why that is safe here:
//  - Config: Mode / NumWorkers (own statics), VideoCore::g_native_pass, g_native_coherent,
//    g_native_colour, g_native_colour_diag, AmdGpu::NativeRec::g_gpu_mode, Shader::g_native_ring /
//    g_native_pt / g_native_seg: all `const` / `inline const` initialised at load, never written.
//  - GraphicsPipeline: native_variant (atomic, acquire; the front stores it once), the
//    native_colour slots (key acquire, then variant acquire: the front claims the key, the
//    compile thread publishes the variant after setting native_vertex_binding /
//    native_table_binding; native_bindless is a constructor argument), and the variant's
//    immutable fields (Handle - may wait a deferred compile, GetLayout, GetGraphicsKey,
//    GetFetchShader, native_*_binding) plus the base's stage Infos' persistent lists (buffers,
//    images, samplers, ud_mask), which only the compile thread writes before the pipeline is
//    published.
//  - DrawIn::stages / vi: the Draw record's blocks in the CpSplit ring, valid while the worker
//    runs on this DCB (the caller's guarantee); read-only. The image / sampler sharps come from
//    the stage's flat buffer span inside that block (ReadStages output), as NativeImageKey's.
//  - NativeRec: g_rings / g_unions / g_gpu_handles (atomics, acquire), g_job_next (relaxed, a
//    lifetime bound only); Record (img_n / img_h1 / img_h2 included) and RunUnion through their
//    stamp seqlocks exactly as NativeRingPrepare reads them (fields first, acquire fence, stamp
//    re-check). g_gpu_recs (acquire) is written by Replay ONLY, on the recording lane: the slot
//    words 128-159 of a record, which the job thread never writes (GpuRecJobDwords) and the back
//    writes the same way (BindlessRingWrite); the back has published the stretch (slot words
//    resolved) before it records the Replay closure.
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
//    covered by the back's conservative per-stretch global barrier rule (architecture-a sec. 6);
//    a colour variant's VS reads the vertex buffer as STORAGE (the back's open adds a whole-buffer
//    vertex barrier when the stream is GPU-written), so that rule must cover storage reads too.
//  - What the worker does NOT check for a colour draw (the back does, per material, at
//    publish): NativeColourGates (tex_bind_memo / bind_memo_stats / pending stencil writebacks),
//    attachment_feedback_loop, images that are bound targets or force_general, the texture-cache
//    side of BindlessResolve (FindImage, layouts, sampler generation / border colour).

namespace Vulkan::A2 {

namespace NR = AmdGpu::NativeRec;

namespace {

constexpr u32 MaxDraws = 64;                              // Rasterizer::NativeMaxDraws
constexpr u32 CmdBytes = 20;                              // VkDrawIndexedIndirectCommand
constexpr u32 TableBytes = Shader::NativeTableDwords * 4; // the per-batch table
constexpr u32 ClipBytes = 96; // AmdGpu::NUM_CLIP_PLANES x 4 words (the back's planes copy)
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
    "colour gates",  "colour stages",   "colour images",  "colour key hash", "colour vertex",
    "colour stream", "materials",
};
static_assert(RefusalNames.size() == NumRefusals && NumRefusals <= 32);

/// The raw T# / S# words of a colour draw (Rasterizer::NativeImageKey, 512 words at most).
struct ImageKey {
    u32 n{};
    std::array<u32, 512> w{};
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
    // Colour draws: the back resolved every material of the stretch (mat_slots, bindless_set)
    // before recording this closure; write each draw's slot words into its job ring record
    // exactly as Rasterizer::BindlessRingWrite does (write-combined ring memory, 128-159).
    for (const auto& sw : cmds.slot_writes) {
        u32* gr = NR::g_gpu_recs[sw.tag >> 28].load(std::memory_order_acquire);
        if (gr && sw.mat < MaxMaterials) {
            std::memcpy(gr + size_t(sw.idx & (NR::RingSize - 1)) * NR::GpuRecDwords +
                            Shader::NativeRecordImgOffset,
                        cmds.mat_slots[sw.mat].data(), sizeof(u32) * 2 * Shader::NativeBindlessMax);
        }
    }
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
        if (b.bindless && cmds.bindless_set) {
            // The global image / sampler arrays (set 1) with the variant's layout, as the
            // back's native colour batch binds them (vk_rasterizer DrawTail, bl_set).
            cmdbuf.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, b.layout, 1,
                                      cmds.bindless_set, {});
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
    bool colour_on{}; ///< colour draws may be taken (SHADPS4_NATIVE_COLOUR=1, no diag 1 / 4096)
    bool cmd_warned{};
    u64 align{256};
    u64 region_bytes{};
    u64 clip_off_in_region{}; ///< the sub-batch's clip planes (after table, commands, count)
    vk::Buffer dummy{}; ///< the always-valid Guest stand-in (BDA page table buffer)
    u64 pt_fallback{}, fault_addr{};

    // The open stretch.
    std::unique_ptr<Stretch> stretch;
    std::shared_ptr<StretchCmds> cmds;
    u64 stretch_start{};
    u64 last_desc_id{};
    DynamicState dyn{}; ///< tracked state of the stretch (dirty bits cleared per sub-batch)
    /// The stretch's materials (index = Stretch::materials index): the key the dedup compares
    /// and the base pipeline it was seen with (another base's FS may decode the same T# with
    /// other descriptor bits: a different material).
    std::array<ImageKey, MaxMaterials> mat_keys{};
    std::array<const GraphicsPipeline*, MaxMaterials> mat_base{};

    // The open sub-batch and its keys.
    bool batch_open{};
    bool b_colour{};
    u64 b_vbuf{};  ///< colour: the one vertex VkBuffer of the opener
    u32 b_vmask{}; ///< colour: the opener's non-null pulled locations
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
    // Inherited Guest slots (the opener's V# for every member) and clip planes.
    u64 b_inh_slots{};
    std::array<VAddr, Shader::NUM_BUFFERS> b_iaddr{};
    std::array<u64, Shader::NUM_BUFFERS> b_isize{};
    std::array<u32, Shader::NUM_BUFFERS> b_ioff{}, b_isz{};
    bool b_clip{};
    std::array<u32, 24> b_planes{};
    u64 b_fmt_mask{};
    std::array<u32, Shader::NUM_BUFFERS> b_tbase{}; ///< the opener's table bases (checks)
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
        clip_off_in_region = AlignUp(TableBytes + u64(MaxDraws) * CmdBytes + 16, align);
        region_bytes = clip_off_in_region + AlignUp(ClipBytes, align);
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
        // Colour draws: SHADPS4_NATIVE_COLOUR=1; diag bit 1 (the back takes no colour joins) and
        // bit 4096 (the compare SSBO shifts the variant's table binding) keep them on the back.
        colour_on = VideoCore::g_native_colour && (VideoCore::g_native_colour_diag & 4097u) == 0;
        LOG_INFO(Render_Vulkan,
                 "A2 worker {}: translator on (arena {} MB, region {} B, align {}, colour {})",
                 ctx.worker_index, ArenaBytes >> 20, region_bytes, align, colour_on ? "on" : "off");
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
        owner->last_refusal = why;
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
    // Colour (the back's DrawTail rule): a pipeline without a shadow variant may have a colour
    // variant per vertex description (NativeColourVariant by NativeVertexKey); it must be the
    // bindless + pulled-vertex kind (NativeDirectCandidate).
    bool colour = false;
    if (!native) {
        if (!base || !colour_on) {
            return refuse(RefuseNoVariant);
        }
        if (!draw.vi || draw.vi->count > 16) {
            return refuse(RefuseNoVi);
        }
        native = base->NativeColourVariant(VideoCore::NativeVertexKey(
            AmdGpu::CpSplit::VertexInputAttribs(*draw.vi), draw.vi->count));
        if (!native) {
            return refuse(RefuseNoVariant);
        }
        colour = true;
        if (!native->native_bindless ||
            native->native_vertex_binding == GraphicsPipeline::NoNativeRecords) {
            return refuse(RefuseColourGates);
        }
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
    if (colour) {
        // The colour variant numbers its bindings over {Fragment, Vertex} only (pipeline cache),
        // and BindlessResolve needs the fragment stage: no other stage may be bound.
        if (!(*draw.stages)[u32(Shader::LogicalStage::Fragment)].info) {
            return refuse(RefuseColourStages);
        }
        for (u32 s = 0; s < Shader::MaxStageTypes - 1; ++s) {
            if (s != u32(Shader::LogicalStage::Fragment) && s != u32(Shader::LogicalStage::Vertex) &&
                (*draw.stages)[s].info) {
                return refuse(RefuseColourStages);
            }
        }
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
    const u32 img_n_rec = r.img_n;
    const u64 img_h1_rec = r.img_h1, img_h2_rec = r.img_h2;
    std::atomic_thread_fence(std::memory_order_acquire);
    if (const u32 stamp = r.stamp.load(std::memory_order_relaxed); stamp != idx + 1 || !gpu_ok) {
        ++owner->sub[stamp != idx + 1 ? (s32(stamp - (idx + 1)) < 0 ? 12 : 13) : 14];
        return refuse(RefuseStamp);
    }
    if (draw.job_cmd) {
        if (cmd != draw.job_cmd) {
            ++owner->sub[15];
            return refuse(RefuseStamp);
        }
    } else if (!cmd_warned) {
        cmd_warned = true;
        ++owner->refusals[RefuseNoCmd];
    }
    if (io_ud != 0 || index_count == 0 || (push_inh >> Shader::NUM_USER_DATA_REGS) != 0) {
        ++owner->sub[io_ud != 0 ? 1 : index_count == 0 ? 2 : 3];
        return refuse(RefuseRecordFields);
    }
    // The record knows the instance count only after a NUM_INSTANCES packet in the same DCB; the
    // mirror always has it (as the generic draw path reads it).
    const u32 n_inst = inst_known ? instances : regs.num_instances.NumInstances();
    owner->sub[0] += inst_known ? 0 : 1; // census: instance count taken from the mirror
    // Inherited Guest slots (l3e): every member binds the opener's V# through the table (Base =
    // 0 - offset, Size, the InhSlots bit, PtLo / PtHi), as NativeRingPrepare does; resolved in the
    // union loop below.

    // The mirror against the record (NativePrepConfirm refuse 12 / 13, Opus G2 #3).
    const bool is_index16 = regs.index_buffer_type.index_type == AmdGpu::IndexType::Index16;
    const u32 index_size = is_index16 ? 2u : 4u;
    const VAddr index_addr =
        regs.index_base_address.Address<VAddr>() + u64(draw.index_offset) * index_size;
    if (index_addr != index_addr_rec || regs.num_indices != index_count ||
        regs.num_instances.NumInstances() != n_inst) {
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
    // unused clip planes (a null descriptor); then the records SSBO (the job ring), (colour) the
    // vertex SSBO, and the table. Colour: images and samplers (fragment stage only) take binding
    // numbers but no set-0 write (the bindless PS samples set 1 at the record's slots); their raw
    // T# / S# words form the draw's material key exactly as NativeImageKey builds it (stage index
    // order, images then samplers).
    Shader::Backend::Bindings bnd{};
    Shader::PushData push = MakePushData(regs);
    std::array<SubBatch::Write, Shader::NUM_BUFFERS + 4> writes{};
    u32 num_writes = 0;
    u64 guest_mask = 0;
    std::array<VAddr, Shader::NUM_BUFFERS> o_addr{};
    std::array<u64, Shader::NUM_BUFFERS> o_vsize{};
    bool clip_on = false;
    u32 clip_wi = 0;
    u64 fmt_mask = 0;
    std::array<u32, 24> planes{};
    ImageKey key{};
    u32 bi = 0;
    for (u32 s = 0; s < Shader::MaxStageTypes - 1; ++s) {
        const Shader::BoundStage& bound = (*draw.stages)[s];
        const Shader::Info* info = bound.info;
        if (!info) {
            continue;
        }
        if (!colour && (!info->images.empty() || !info->samplers.empty())) {
            ++owner->sub[4];
            return refuse(RefuseSlotKind);
        }
        if (colour && s != u32(Shader::LogicalStage::Fragment) &&
            (!info->images.empty() || !info->samplers.empty())) {
            return refuse(RefuseColourStages); // BindlessResolve resolves the FS only
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
                    // As BindResources: a copy of the planes (the opener's region, below).
                    clip_on = true;
                    clip_wi = num_writes;
                    for (u32 i = 0; i < AmdGpu::NUM_CLIP_PLANES; ++i) {
                        const auto& plane = regs.clip_user_data[i];
                        planes[i * 4 + 0] = plane.data_x;
                        planes[i * 4 + 1] = plane.data_y;
                        planes[i * 4 + 2] = plane.data_z;
                        planes[i * 4 + 3] = plane.data_w;
                    }
                }
                writes[num_writes++] = {bnd.unified++, vk::Buffer{}, 0, VK_WHOLE_SIZE};
                ++bnd.buffer;
                ++bi;
                continue;
            }
            // Formatted (typed) reads on colour draws: lowered to raw loads in the variant (the
            // pipeline cache allows them for colour only); the back syncs the range at publish.
            const bool fmt_ok = colour && desc.is_formatted &&
                                desc.buffer_type == Shader::BufferType::Guest && !desc.is_written;
            if (desc.buffer_type != Shader::BufferType::Guest || desc.is_written ||
                (desc.is_formatted && !fmt_ok)) {
                ++owner->sub[desc.buffer_type != Shader::BufferType::Guest ? 6
                             : desc.is_written                             ? 7
                                                                           : 8];
                return refuse(RefuseSlotKind);
            }
            const auto vsharp = desc.GetSharp(bound);
            if (colour && (desc.inline_cbuf || vsharp.base_address == 0 || vsharp.GetSize() == 0)) {
                ++owner->sub[9];
                return refuse(RefuseSlotKind); // NativeDirectBind fail 5 / 6
            }
            o_addr[bi] = vsharp.base_address;
            o_vsize[bi] = vsharp.GetSize();
            guest_mask |= u64{1} << bi;
            if (fmt_ok) {
                fmt_mask |= u64{1} << bi;
                bool known = false;
                if (stretch) {
                    for (const auto& [ta, ts] : stretch->texel) {
                        known = known || (ta == o_addr[bi] && ts == o_vsize[bi]);
                    }
                }
                if (!known && stretch && stretch->texel.size() >= 16) {
                    ++owner->sub[21];
                    return refuse(RefuseSlotKind);
                }
            }
            writes[num_writes++] = {bnd.unified++, dummy, 0, VK_WHOLE_SIZE};
            ++bnd.buffer;
            ++bi;
        }
        if (colour) {
            // BindlessResolve's own pure checks (bl_fail 0 / 1) and the numbering rule: the
            // variant counts one binding per image (Info::AddBindings), NativeDirectBind one per
            // mip level under the DynamicIndex fallback - refuse that mode rather than guess.
            if (info->images.size() > Shader::NativeBindlessMax ||
                info->samplers.size() > Shader::NativeBindlessMax) {
                return refuse(RefuseColourImages);
            }
            for (const auto& desc : info->images) {
                if (desc.is_written || desc.is_atomic ||
                    desc.mip_fallback_mode == Shader::MipStorageFallbackMode::DynamicIndex) {
                    return refuse(RefuseColourImages);
                }
                const auto tsharp = desc.GetSharp(bound);
                if (tsharp.Address() == 0 ||
                    tsharp.GetDataFmt() == AmdGpu::DataFormat::FormatInvalid) {
                    return refuse(RefuseColourImages);
                }
                constexpr u32 words = static_cast<u32>(sizeof(tsharp) / sizeof(u32));
                if (key.n + words > key.w.size()) {
                    return refuse(RefuseColourImages);
                }
                std::memcpy(&key.w[key.n], &tsharp, sizeof(tsharp));
                key.n += words;
            }
            for (const auto& desc : info->samplers) {
                const auto ssharp = desc.GetSharp(bound);
                constexpr u32 words = static_cast<u32>(sizeof(ssharp) / sizeof(u32));
                if (key.n + words > key.w.size()) {
                    return refuse(RefuseColourImages);
                }
                std::memcpy(&key.w[key.n], &ssharp, sizeof(ssharp));
                key.n += words;
            }
            bnd.unified += static_cast<u32>(info->images.size() + info->samplers.size());
        }
    }
    if (colour) {
        if (native->native_records_binding != bnd.unified ||
            native->native_vertex_binding != bnd.unified + 1 ||
            native->native_table_binding != bnd.unified + 2 || guest_mask == 0) {
            return refuse(RefuseLayout);
        }
        // The job record's image count / key hashes must agree with the stages (NativeRingPrepare
        // fail 13), unless a T# / S# sat in inherited registers (zeros in the job's key).
        if ((img_n_rec & ~NR::ImgInhBit) != key.n) {
            return refuse(RefuseColourKeyHash);
        }
        if (!(img_n_rec & NR::ImgInhBit)) {
            u64 h1 = 0, h2 = 0;
            NR::ImageKeyHash(key.w.data(), key.n, h1, h2);
            if (h1 != img_h1_rec || h2 != img_h2_rec) {
                return refuse(RefuseColourKeyHash);
            }
        }
    } else if (native->native_records_binding != bnd.unified ||
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
    // Vertex streams: each non-null stream's buffer and its offset in it. Colour (NativeOpen's
    // pulled-vertex rule): per-vertex attributes at locations < 16 only, every non-null stream
    // inside ONE VkBuffer, offsets below 2^32, at least one stream (the back binds a
    // stream-buffer stand-in when all are null: not reproduced, refused); each stream's whole
    // [base, base + size) inside the buffer (NativeRingPrepare fail 7 checks the run's span).
    std::array<u64, 16> vb_handle{}, vcpu{}, voff{};
    u64 vbuf = 0;
    u32 vmask = 0;
    for (u32 k = 0; k < vb_n; ++k) {
        const u32 loc = attribs[k].desc.location;
        if (colour && (attribs[k].desc.instanced || loc >= Shader::NativeMaxPulledLocations)) {
            return refuse(RefuseColourVertex);
        }
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
        if (colour) {
            if ((vbuf != 0 && vb_handle[k] != vbuf) || voff[k] > 0xffffffffull) {
                return refuse(RefuseColourVertex);
            }
            if (!vb->IsInBounds(vaddr_vi[k], attribs[k].size)) {
                return refuse(RefuseColourStream); // NativeRingPrepare fail 7 (the run's span)
            }
            vbuf = vb_handle[k];
            vmask |= 1u << loc;
        }
    }
    if (colour && vbuf == 0) {
        return refuse(RefuseColourVertex);
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

    // Colour: the draw's material in the stretch's list (dedup by base pipeline + key words), or
    // a new one if the list has room. The list itself changes only when the draw is taken.
    s32 mat = -1;
    if (colour) {
        const u32 nm = stretch ? static_cast<u32>(stretch->materials.size()) : 0;
        for (u32 m = 0; m < nm; ++m) {
            if (mat_base[m] == base && mat_keys[m].n == key.n &&
                std::memcmp(mat_keys[m].w.data(), key.w.data(), sizeof(u32) * key.n) == 0) {
                mat = static_cast<s32>(m);
                break;
            }
        }
        if (mat < 0 && nm >= MaxMaterials) {
            return refuse(RefuseMaterials);
        }
    }

    // Can the draw continue the open sub-batch? Keys: the variant, the next ring entry of the
    // same job slot and run, the inherited set and the inherited registers' values, the same
    // index / vertex VkBuffers and strides, the same push constants, vertex description and
    // dynamic state, every Guest V# inside the buffers the opener's table bases name, the
    // vertex rebase quotient rule (below), fewer than MaxDraws draws. Colour: the same vertex
    // VkBuffer and the same set of non-null locations (NativeJoin's `had` rule); the material
    // may differ (each draw's record gets its own slot words).
    const u32 head = run_k == 0 ? idx : head_idx;
    bool cont = batch_open && b_native == native && b_colour == colour && b_slot == slot &&
                idx == b_next_idx && cmds->batches.back().n < MaxDraws && union_idx == b_union &&
                head == b_head && push_inh == b_push_inh && inh_vs == b_inh_vs &&
                inh_ps == b_inh_ps && ib_handle == b_ib && index_size == b_index_size &&
                vb_n == b_vb_n && draw.vi->desc_id == last_desc_id && PushKeySame(push, b_push) &&
                guest_mask == b_guest_mask && !AnyDirty(cand) &&
                (!colour || (vbuf == b_vbuf && vmask == b_vmask)) && inh_slots == b_inh_slots &&
                clip_on == b_clip && (!clip_on || planes == b_planes) && fmt_mask == b_fmt_mask;
    for (u64 m = fmt_mask; cont && m != 0; m &= m - 1) {
        // The back's member rule: a formatted slot binds the opener's own V#.
        const u32 b = static_cast<u32>(std::countr_zero(m));
        cont = o_addr[b] == b_iaddr[b] && o_vsize[b] == b_isize[b];
    }
    for (u64 m = inh_slots; cont && m != 0; m &= m - 1) {
        const u32 b = static_cast<u32>(std::countr_zero(m));
        cont = o_addr[b] == b_iaddr[b] && o_vsize[b] == b_isize[b];
    }
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
    // stride-0 streams may not move. A failure ends the sub-batch, not the stretch. Colour: no
    // rebase (the record carries each location's offset), vertexOffset as is.
    s64 vo = 0;
    if (cont && colour) {
        vo = s64(s32(vertex_offset));
        cont = vo >= 0 && vo <= INT32_MAX;
    } else if (cont) {
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
        // Colour: no rebase (NativeOpen binds the opener's offsets as they are).
        std::array<u64, 16> bo = voff;
        u64 q0 = 0;
        if (!colour) {
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
        std::array<u32, Shader::NUM_BUFFERS> i_off{}, i_size{};
        for (u64 m = guest_mask; m != 0; m &= m - 1) {
            const u32 b = static_cast<u32>(std::countr_zero(m));
            if (inh_slots & (u64{1} << b)) {
                // The opener's own V# (no union): one cached buffer must hold all of it.
                const u64 vsz = o_vsize[b];
                if (vsz == 0 || vsz >= (u64{1} << 30)) {
                    ++owner->sub[16];
                    return refuse(RefuseInheritedSlot);
                }
                const VideoCore::Buffer* ibuf = ctx.buffer_cache->A2LookupBuffer(o_addr[b]);
                if (!ibuf || !ibuf->IsInBounds(o_addr[b], vsz)) {
                    ++owner->sub[17];
                    return refuse(RefuseInheritedSlot);
                }
                const u64 ioff = o_addr[b] - ibuf->CpuAddr();
                if (ioff > 0xffffffffull) {
                    ++owner->sub[18];
                    return refuse(RefuseInheritedSlot);
                }
                i_off[b] = static_cast<u32>(ioff);
                i_size[b] = static_cast<u32>(vsz);
                glo[b] = ibuf->CpuAddr();
                ghi[b] = ibuf->CpuAddr() + ibuf->SizeBytes();
                continue;
            }
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
                ++owner->sub[!buf ? 10 : 11];
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
        // The table (NativeRingPrepare + NativeOpen's ring-mode words; SegMat 0; InhSlots / Size /
        // PtLo / PtHi only for inherited slots, as NativeRingPrepare writes them).
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
            if (inh_slots & (u64{1} << b)) {
                table[Shader::NativeTableBase + b] = 0u - i_off[b];
                table[Shader::NativeTableSize + b] = i_size[b];
                table[Shader::NativeTableInhSlots + b / 32] |= 1u << (b % 32);
                table[Shader::NativeTablePtLo + b] = static_cast<u32>(o_addr[b]);
                table[Shader::NativeTablePtHi + b / 4] |=
                    static_cast<u32>((o_addr[b] >> 32) & 0xffu) << ((b % 4) * 8);
                continue;
            }
            table[Shader::NativeTableBase + b] = base[b];
        }
        if (colour) {
            // NativeRingPrepare's per-location bases: the vertex VkBuffer's CPU address (the GPU
            // record holds each stream's guest address; record - base = the offset). SegMat
            // stays 0: the slots come from the record words.
            for (u32 k = 0; k < vb_n; ++k) {
                if (vb_handle[k] != 0) {
                    table[Shader::NativeTableVbase + attribs[k].desc.location] =
                        static_cast<u32>(vcpu[k]);
                }
            }
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
        sb.bindless = colour;
        sb.pipeline = native->Handle();
        sb.layout = native->GetLayout();
        sb.push = push;
        writes[num_writes++] = {native->native_records_binding,
                                vk::Buffer{std::bit_cast<VkBuffer>(ring_handle)}, 0, VK_WHOLE_SIZE};
        if (colour) {
            // The whole vertex VkBuffer (NativeOpen's colour write, before the table).
            writes[num_writes++] = {native->native_vertex_binding,
                                    vk::Buffer{std::bit_cast<VkBuffer>(vbuf)}, 0, VK_WHOLE_SIZE};
        }
        writes[num_writes++] = {native->native_table_binding, ctx.arena->buffer, region_off,
                                TableBytes};
        if (clip_on) {
            std::memcpy(region + clip_off_in_region, planes.data(), ClipBytes);
            writes[clip_wi].buffer = ctx.arena->buffer;
            writes[clip_wi].offset = region_off + clip_off_in_region;
            writes[clip_wi].range = ClipBytes;
        }
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
        b_colour = colour;
        b_vbuf = vbuf;
        b_vmask = vmask;
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
        b_inh_slots = inh_slots;
        b_iaddr = o_addr;
        b_isize = o_vsize;
        b_ioff = i_off;
        b_isz = i_size;
        b_clip = clip_on;
        b_planes = planes;
        b_fmt_mask = fmt_mask;
        b_tbase = base;
        last_desc_id = draw.vi->desc_id;
        ++owner->sub_batches;
    }

    // The draw's indirect command (NativeRingPrepare's member command: the GPU reads its record
    // at ring entry first + n).
    SubBatch& sb = cmds->batches.back();
    const u32 n = sb.n;
    const u32 cmdw[5] = {index_count, n_inst, static_cast<u32>(ib_off / index_size),
                         static_cast<u32>(s32(vo)), 0u};
    std::memcpy(cmd_ptr + u64(n) * CmdBytes, cmdw, CmdBytes);
    sb.n = n + 1;
    *count_ptr = n + 1;
    if (colour) {
        // The draw's material (a new one is appended now: nothing above can refuse any more)
        // and its slot write (Replay fills record (tag, idx) from mat_slots[mat]).
        if (mat < 0) {
            mat = static_cast<s32>(stretch->materials.size());
            stretch->materials.push_back({base, draw.rec_pos});
            mat_keys[mat] = key;
            mat_base[mat] = base;
        }
        cmds->slot_writes.push_back({draw.job_tag, idx, static_cast<u32>(mat)});
    }
    if (inh_slots != 0) {
        ++owner->sub[19];
        if (Mode() == 2) {
            for (u64 m = inh_slots; m != 0; m &= m - 1) {
                const u32 b = static_cast<u32>(std::countr_zero(m));
                cmds->inh_checks.push_back({draw.seq, b_iaddr[b], b_isize[b], b_ioff[b], b_isz[b]});
            }
        }
    }
    if (fmt_mask != 0) {
        ++owner->sub[22];
        for (u64 m = fmt_mask; m != 0; m &= m - 1) {
            const u32 b = static_cast<u32>(std::countr_zero(m));
            bool known = false;
            for (const auto& [ta, ts] : stretch->texel) {
                known = known || (ta == o_addr[b] && ts == o_vsize[b]);
            }
            if (!known) {
                stretch->texel.emplace_back(o_addr[b], o_vsize[b]);
            }
            if (Mode() == 2) {
                cmds->fmt_checks.push_back({draw.seq, o_addr[b], b_tbase[b]});
            }
        }
    }
    if (clip_on) {
        ++owner->sub[20];
        if (Mode() == 2) {
            cmds->clip_checks.push_back(
                {draw.seq, regs.clipper_control.user_clip_plane_enable, b_planes});
        }
    }
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
