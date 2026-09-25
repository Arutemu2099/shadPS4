// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/recursive_lock.h"
#include "common/shared_first_mutex.h"
#include "common/unique_function.h"
#include "video_core/amdgpu/cb_db_extent.h"
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

namespace Vulkan {

class Scheduler;
class RenderState;
class GraphicsPipeline;

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

    void BindBuffers(const Shader::BoundStage& bound, Shader::Backend::Bindings& binding,
                     Shader::PushData& push_data);
    void BindTextures(const Shader::BoundStage& bound, Shader::Backend::Bindings& binding);
    bool BindResources(const Pipeline* pipeline);

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
