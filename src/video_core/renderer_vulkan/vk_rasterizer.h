// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <chrono>

#include "common/recursive_lock.h"
#include "common/shared_first_mutex.h"
#include "common/unique_function.h"
#include "video_core/amdgpu/cb_db_extent.h"
#include "video_core/amdgpu/cp_split.h"
#include "video_core/amdgpu/regs.h"
#include "video_core/amdgpu/regs_color.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/page_manager.h"
#include "video_core/renderer_vulkan/vk_native_bindings.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/texture_cache/texture_cache.h"

namespace AmdGpu {
struct Liverpool;
}

namespace Core {
class MemoryManager;
}

namespace Common {
class DrawSampleProbe;
}

namespace Vulkan {

class Scheduler;
class RenderState;
class GraphicsPipeline;
struct TexMemoRecord;

class Rasterizer {
public:
    explicit Rasterizer(const Instance& instance, Scheduler& scheduler,
                        AmdGpu::Liverpool* liverpool);
    ~Rasterizer();

    [[nodiscard]] Scheduler& GetScheduler() noexcept {
        return scheduler;
    }

    [[nodiscard]] VideoCore::BufferCache& GetBufferCache() noexcept {
        return buffer_cache;
    }

    [[nodiscard]] VideoCore::TextureCache& GetTextureCache() noexcept {
        return texture_cache;
    }

    /// What a draw packet does, decided from registers alone (CP front/back split, step 3).
    /// Everything except Draw replaces the draw with a back-side action or a marker.
    enum class DrawClass : u8 {
        Draw,               ///< An ordinary draw.
        EliminateFastClear, ///< CB mode EliminateFastClear: EliminateFastClear() instead.
        FmaskDecompress,    ///< CB mode FmaskDecompress: skipped (marker only).
        Resolve,            ///< CB mode Resolve: Resolve() instead.
        PrimitiveNone,      ///< Primitive type None: skipped (marker only).
        DepthCopy,          ///< CB disabled + forced depth dirty/valid: DepthStencilCopy(1, 0).
        StencilCopy,        ///< Same for stencil: DepthStencilCopy(0, 1).
        DepthStencilCopy,   ///< Both: DepthStencilCopy(1, 1).
    };
    /// Pure: reads only `regs`, touches no cache, scheduler or Rasterizer member, so the CP front
    /// can call it.
    static DrawClass ClassifyDraw(const AmdGpu::Regs& regs);
    /// The back-side action for a class other than Draw (texture cache work or a marker).
    void ExecuteFilteredDraw(DrawClass draw_class);

    /// CP front/back split (SHADPS4_CP_SPLIT, video_core/amdgpu/cp_split.h): one draw as the
    /// front hands it to the back. Trivially copyable; it travels inside a ring record.
    struct SplitDraw {
        AmdGpu::CpSplit::DrawArgs args{};
        DrawClass draw_class{DrawClass::Draw};
        bool skip{};                         ///< SHADPS4_EMPTY_DRAWS=2 dropped it on the front.
        const GraphicsPipeline* pipeline{};  ///< Null: the lookup failed or was not reached.
    };
    /// Front thread: classification, hazard note, empty-draw skip and pipeline lookup, reading
    /// the front's registers. Touches no cache, the scheduler or any per-draw member.
    void PrepareDrawFront(SplitDraw& draw);
    /// Front thread: compute pipeline lookup against the current queue's cs registers.
    const ComputePipeline* PrepareDispatchFront();
    /// Back thread: runs a draw the front prepared, with the per-stage user data it copied
    /// (indexed by LogicalStage). Registers come from Liverpool::BackRegs().
    void ExecuteDraw(const SplitDraw& draw,
                     const std::array<Shader::BoundStage, Shader::MaxStageTypes>& stages);
    /// Back thread: runs a dispatch the front looked up. The caller points
    /// Liverpool::BackCsRegs() at the record's ComputeProgram snapshot.
    void ExecuteDispatch(const ComputePipeline* pipeline,
                         const std::array<Shader::BoundStage, Shader::MaxStageTypes>& stages,
                         bool is_indirect, VAddr address, u32 offset, u32 size);

    void Draw(bool is_indexed, u32 index_offset = 0);
    void ScanRegistersForValue();
    /// SHADPS4_NEO_IDBUF_DERIVE=1: fill the Pro ID buffer from the last 2-sample scene depth.
    void DeriveNeoIdBuffer(VideoCore::Image& id_image);
    struct NeoIdDerive;
    std::shared_ptr<NeoIdDerive> neo_id_derive;
    VAddr neo_depth_addr{};
    VideoCore::Extent3D neo_depth_size{};
    u32 neo_depth_x0y0{};
    void DrawIndirect(bool is_indexed, VAddr arg_address, u32 offset, u32 size, u32 max_count,
                      VAddr count_address);

    void DispatchDirect();
    void DispatchIndirect(VAddr address, u32 offset, u32 size);

    void ScopeMarkerBegin(const std::string_view& str, bool from_guest = false);
    void ScopeMarkerEnd(bool from_guest = false);
    void ScopedMarkerInsert(const std::string_view& str, bool from_guest = false);
    void ScopedMarkerInsertColor(const std::string_view& str, const u32 color,
                                 bool from_guest = false);

    void FillBuffer(VAddr address, u32 num_bytes, u32 value, bool is_gds);
    void CopyBuffer(VAddr dst, VAddr src, u32 num_bytes, bool dst_gds, bool src_gds);
    u32 ReadDataFromGds(u32 gsd_offset);
    bool InvalidateMemory(VAddr addr, u64 size,
                          VideoCore::InvalidateOrigin origin = VideoCore::InvalidateOrigin::Explicit);
    bool ReadMemory(VAddr addr, u64 size);
    void ProcessDownloadImages();
    bool IsMapped(VAddr addr, u64 size);
    void MapMemory(VAddr addr, u64 size);
    void UnmapMemory(VAddr addr, u64 size);

    void CpSync();
    u64 Flush();
    void Finish();
    /// SHADPS4_FENCE_AT_GPU: run `func` on the scheduler's priority thread once the host GPU has
    /// finished everything recorded so far; submits the current batch so that happens.
    void DeferToGpu(Common::UniqueFunction<void>&& func, bool flush);
    /// Mode 2 (lazy): submits the batch holding deferred fences if it is still being recorded.
    void FlushLazyFences();
    u64 lazy_fence_tick{};
    void OnSubmit();

    PipelineCache& GetPipelineCache() {
        return pipeline_cache;
    }

    template <typename Func>
    void ForEachMappedRangeInRange(VAddr addr, u64 size, Func&& func) {
        const auto range = decltype(mapped_ranges)::interval_type::right_open(addr, addr + size);
        Common::RecursiveSharedLock lock{mapped_ranges_mutex};
        for (const auto& mapped_range : (mapped_ranges & range)) {
            func(mapped_range);
        }
    }

private:
    void PrepareRenderState(const GraphicsPipeline* pipeline);
    RenderState BeginRendering(const GraphicsPipeline* pipeline);
    void Resolve();
    void DepthStencilCopy(bool is_depth, bool is_stencil);
    void EliminateFastClear();

    void UpdateDynamicState(const GraphicsPipeline* pipeline, bool is_indexed) const;
    void UpdateViewportScissorState() const;
    mutable u64 viewport_epoch{0};
    mutable u64 viewport_memo_skips{0};
    mutable u64 viewport_memo_mismatches{0};
    void UpdateDepthStencilState() const;
    void UpdatePrimitiveState(bool is_indexed) const;
    void UpdateRasterizationState() const;
    void UpdateColorBlendingState(const GraphicsPipeline* pipeline) const;

    bool FilterDraw();
    /// SHADPS4_EMPTY_DRAWS counting and (=2) skipping; registers and statics only.
    static bool SkipEmptyDraw(const AmdGpu::Regs& regs);

    // Everything after the pipeline lookup, shared by the inline path and the CP split's back.
    // bound_stages must already hold the draw's stage data.
    void DrawTail(const GraphicsPipeline* pipeline, bool is_indexed, u32 index_offset,
                  Common::DrawSampleProbe& sample, std::chrono::steady_clock::time_point t_entry,
                  std::chrono::steady_clock::time_point t_pipeline, u64 pipeline_ns);
    void DrawIndirectTail(const GraphicsPipeline* pipeline, bool is_indexed, VAddr arg_address,
                          u32 offset, u32 stride, u32 max_count, VAddr count_address);
    void DispatchDirectTail(const ComputePipeline* pipeline);
    void DispatchIndirectTail(const ComputePipeline* pipeline, VAddr address, u32 offset,
                              u32 size);

    void BindBuffers(const Shader::BoundStage& bound, Shader::Backend::Bindings& binding,
                     Shader::PushData& push_data);
    void BindTextures(const Shader::BoundStage& bound, Shader::Backend::Bindings& binding);
    bool BindResources(const Pipeline* pipeline);

    /// SHADPS4_BIND_MEMO_STATS=1 (diagnostic, default off; notes/bind-memo-stats-2026-09-25.md):
    /// evaluates a per-stage binding memo next to the unchanged full path and logs its hit rate,
    /// miss reasons, output mismatches and the time a hit would save. Never skips anything.
    /// Reads user data and the flattened buffer only through `bound`.
    /// Returns bit 0 = both key halves hit, bit 1 = both halves would hit.
    u8 BindStageMemoStats(const Pipeline* pipeline, const Shader::BoundStage& bound,
                          Shader::Backend::Bindings& binding);
    void BindMemoEndDraw(bool is_compute, u8 draw_bits);
    bool bind_memo_stats{};  // read once in the constructor
    bool bind_memo_active{}; // inside a stats stage: BindBuffers records uploads per binding
    bool bind_memo_timing{}; // inside a timed stats stage: BindBuffers times always-full bindings
    u64 bind_memo_fixed_ticks{};
    u64 bind_memo_redirected{}; // image bindings BindTextures redirected to another image
    std::array<u8, Shader::NUM_BUFFERS> bind_memo_uploaded{};

    /// SHADPS4_TEX_BIND_MEMO (default off; notes/bind-memo-stats-2026-09-25.md): the texture
    /// half of the binding memo. =1: when a stage's T#/S#/border key and every validity check
    /// match the record kept on the pipeline, replays the recorded views and samplers instead of
    /// BindTextures, keeping what a hit must still do (binding flags, LRU touches, bound_images,
    /// a layout transition when the state differs, descriptor writes). =2 verify: runs the hit,
    /// rolls its outputs back, runs BindTextures and compares (log + counter, never a crash).
    void BindTexturesMemo(const Pipeline* pipeline, const Shader::BoundStage& bound,
                          Shader::Backend::Bindings& binding);
    /// A hit's outputs; returns the number of Transit calls it made.
    u32 EmitTexMemoHit(const TexMemoRecord& rec, Shader::Backend::Bindings& binding);
    void RecordTexMemo(TexMemoRecord& rec, const Shader::Info& stage, u64 tex_gen, u64 smp_gen,
                       u64 border, size_t img_begin, u32 sw_begin);
    u8 tex_bind_memo{};        // 0 off, 1 memo, 2 verify; read once in the constructor
    bool bind_memo_rebound{};  // BindTextures re-found a needs_rebind image

    /// Fills bound_stages from the pipeline lookup result; called right after the lookup.
    void BindStageData(const Pipeline* pipeline) {
        const auto stages = pipeline->GetStages();
        const u32 first = pipeline->IsCompute() ? u32(Shader::LogicalStage::Compute) : 0u;
        for (u32 i = 0; i < stages.size(); ++i) {
            bound_stages[first + i] = Shader::BoundStage{stages[i]};
        }
    }
    /// The bound_stages entries of `pipeline`, in Pipeline::GetStages() order.
    std::span<const Shader::BoundStage> BoundStages(const Pipeline* pipeline) const {
        const auto all = std::span{bound_stages};
        return pipeline->IsCompute() ? all.last(1) : all.first(Shader::MaxStageTypes - 1);
    }
    const Shader::BoundStage& GetBoundStage(Shader::LogicalStage stage) const {
        return bound_stages[u32(stage)];
    }

    /// SHADPS4_STENCIL_STORAGE=1: shader storage writes into a depth buffer's stencil plane
    /// (PS4 Pro checkerboard mask) go to an R8 scratch image that is seeded from and copied back
    /// into the depth image's stencil aspect, instead of an invalid storage view of D32S8.
    VideoCore::ImageId StencilScratchFor(VideoCore::ImageId depth_id,
                                         const VideoCore::ImageInfo& info);
    bool IsStencilScratch(VideoCore::ImageId id) const;
    void CopyStencilPlane(VideoCore::ImageId src_id, VideoCore::ImageId dst_id);
    void FlushStencilWritebacks();
    struct StencilScratch {
        VideoCore::ImageId depth_id;
        u64 depth_uid;
        VideoCore::ImageId scratch_id;
    };
    std::vector<StencilScratch> stencil_scratches;
    boost::container::small_vector<std::pair<VideoCore::ImageId, VideoCore::ImageId>, 2>
        pending_stencil_writebacks; // (scratch, depth)

    /// Emits the bindings assembled by BindResources through the pipeline's binding model.
    void EmitPipelineResources(const Pipeline* pipeline);

    void ResetBindings() {
        for (auto& image_id : bound_images) {
            texture_cache.GetImage(image_id).binding = {};
        }
        bound_images.clear();
    }

    bool IsComputeMetaClear(const Pipeline* pipeline);
    bool IsComputeImageCopy(const Pipeline* pipeline);
    bool IsComputeImageClear(const Pipeline* pipeline);

private:
    friend class VideoCore::BufferCache;

    const Instance& instance;
    Scheduler& scheduler;
    // Declared before the caches: created before any buffer, view, sampler or pipeline, and
    // destroyed after them so object-owned heap slots can still be released. Null in legacy.
    std::unique_ptr<NativeBindings> native_bindings;
    VideoCore::PageManager page_manager;
    VideoCore::BufferCache buffer_cache;
    VideoCore::TextureCache texture_cache;
    AmdGpu::Liverpool* liverpool;
    Core::MemoryManager* memory;
    boost::icl::interval_set<VAddr> mapped_ranges;
    Common::SharedFirstMutex mapped_ranges_mutex;
    PipelineCache pipeline_cache;

    using RenderTargetInfo = std::pair<VideoCore::ImageId, VideoCore::TextureCache::ImageDesc>;
    std::array<RenderTargetInfo, AmdGpu::NUM_COLOR_BUFFERS> cb_descs;
    std::pair<VideoCore::ImageId, VideoCore::TextureCache::ImageDesc> db_desc;
    // Pure preparation inputs/results only. Live image identities, uploads and layouts
    // are resolved separately; FindImage may mutate the working descriptor's view range.
    struct PreparedColorTarget {
        AmdGpu::ColorBuffer buffer{};
        AmdGpu::CbDbExtent extent{};
        VideoCore::TextureCache::ImageDesc value{};
        bool valid{};
    };
    struct PreparedDepthTarget {
        AmdGpu::DepthBuffer buffer{};
        AmdGpu::DepthView view{};
        AmdGpu::DepthControl control{};
        AmdGpu::CbDbExtent extent{};
        VAddr htile_address{};
        VideoCore::TextureCache::ImageDesc value{};
        bool valid{};
    };
    std::array<PreparedColorTarget, AmdGpu::NUM_COLOR_BUFFERS> prepared_color_targets{};
    PreparedDepthTarget prepared_depth_target{};
    boost::container::static_vector<vk::DescriptorImageInfo, Shader::NUM_IMAGES> image_infos;
    boost::container::static_vector<vk::DescriptorBufferInfo, Shader::NUM_BUFFERS> buffer_infos;
    boost::container::static_vector<VideoCore::ImageId, Shader::NUM_IMAGES> bound_images;

    // Heap slot per unified binding, filled from the same resolved buffer_infos/image_infos
    // entries the legacy writes point at. Legacy writes are skipped unless verifying.
    std::array<u32, NativeMaxBindingsPerDraw> native_indices{};
    std::array<u32, Shader::NUM_IMAGES> native_image_indices{};
    u32 native_bound_count{};
    bool build_set_writes{true};

    u32 set_write_index{};
    Pipeline::DescriptorWrites set_writes;
    Pipeline::BufferBarriers buffer_barriers;
    Shader::PushData push_data;
    // Per-draw user data of each stage, indexed like Pipeline::stages (by LogicalStage). The back
    // path reads user data only through these (CP front/back split, step 2).
    std::array<Shader::BoundStage, Shader::MaxStageTypes> bound_stages{};

    using BufferBindingInfo = std::tuple<VideoCore::BufferId, AmdGpu::Buffer, u64>;
    boost::container::static_vector<BufferBindingInfo, Shader::NUM_BUFFERS> buffer_bindings;
    using ImageBindingInfo = std::pair<VideoCore::ImageId, VideoCore::TextureCache::ImageDesc>;
    boost::container::static_vector<ImageBindingInfo, Shader::NUM_IMAGES> image_bindings;
    struct DecodedImageDesc {
        AmdGpu::Image sharp{};
        u8 flags{};
        bool valid{};
        VideoCore::TextureCache::ImageDesc value{};
    };
    std::array<std::array<DecodedImageDesc, Shader::NUM_IMAGES>, Shader::MaxStageTypes>
        decoded_image_descs{};
    bool fault_process_pending{};
    bool attachment_feedback_loop{};

};

} // namespace Vulkan
