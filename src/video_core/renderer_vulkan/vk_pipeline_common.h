// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "shader_recompiler/profile.h"
#include "shader_recompiler/runtime_info.h"
#include "video_core/renderer_vulkan/vk_common.h"
#include "video_core/renderer_vulkan/vk_native_bindings_api.h"

#include <array>
#include <memory>
#include <span>
#include <vector>
#include <boost/container/small_vector.hpp>

namespace Shader {
struct Info;
struct PushData;
} // namespace Shader

namespace Vulkan {

static constexpr auto AllGraphicsStageBits =
    vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eTessellationControl |
    vk::ShaderStageFlagBits::eTessellationEvaluation | vk::ShaderStageFlagBits::eGeometry |
    vk::ShaderStageFlagBits::eFragment;

class Instance;
class Scheduler;
class DescriptorHeap;
class NativeBindings;

class Pipeline {
public:
    Pipeline(const Instance& instance, Scheduler& scheduler, DescriptorHeap& desc_heap,
             const Shader::Profile& profile, vk::PipelineCache pipeline_cache,
             bool is_compute = false);
    virtual ~Pipeline();

    vk::Pipeline Handle() const noexcept {
        return *pipeline;
    }

    vk::PipelineLayout GetLayout() const noexcept {
        return *pipeline_layout;
    }

    auto GetStages() const {
        static_assert(static_cast<u32>(Shader::LogicalStage::Compute) == Shader::MaxStageTypes - 1);
        if (is_compute) {
            return std::span{stages.cend() - 1, stages.cend()};
        } else {
            return std::span{stages.cbegin(), stages.cend() - 1};
        }
    }

    const Shader::Info& GetStage(Shader::LogicalStage stage) const noexcept {
        return *stages[u32(stage)];
    }

    bool IsCompute() const {
        return is_compute;
    }

    using DescriptorWrites = std::vector<vk::WriteDescriptorSet>;
    using BufferBarriers = boost::container::small_vector<vk::BufferMemoryBarrier2, 16>;

    void BindResources(DescriptorWrites& set_writes, const BufferBarriers& buffer_barriers,
                       const Shader::PushData& push_data) const;

    /// True when the pipeline was created with descriptor heap mappings and no legacy layout.
    bool UsesNativeBindings() const noexcept {
        return uses_native_bindings;
    }

    /// Number of unified bindings, i.e. the length of the per-draw heap index array.
    u32 NativeBindingCount() const noexcept {
        return native_binding_count;
    }

    /// Descriptor heap emission: barriers, heap binds when invalidated, then one push-data
    /// block carrying shader push constants plus direct indices or an indirect index address.
    void BindResourcesNative(NativeBindings& native, std::span<const u32> indices,
                             const BufferBarriers& buffer_barriers,
                             const Shader::PushData& push_data) const;

    /// SHADPS4_TEX_BIND_MEMO (vk_rasterizer.cpp): texture-binding memo records per logical stage
    /// (one, or SHADPS4_TEX_MEMO_WAYS), owned by the pipeline so a record can never outlive, or be reused for, a different
    /// Shader::Info. Type-erased (the record type is private to vk_rasterizer.cpp); touched only
    /// by the thread that binds resources.
    mutable std::array<std::shared_ptr<void>, Shader::MaxStageTypes> tex_bind_memo{};

protected:
    [[nodiscard]] std::string GetDebugString() const;

    void RecordBufferBarriers(const BufferBarriers& buffer_barriers) const;

    /// Builds one heap mapping per legacy binding, grouped by the owning shader stage.
    void BuildNativeMappings(std::span<const vk::DescriptorSetLayoutBinding> bindings);

    /// Mapping info to chain into the given stage's create info, or null if it binds nothing.
    /// Pointers stay valid for the lifetime of the pipeline object.
    const NativeApi::ShaderDescriptorSetAndBindingMappingInfo* NativeMappingInfo(
        vk::ShaderStageFlagBits stage, const void* next) const;

    const Instance& instance;
    Scheduler& scheduler;
    DescriptorHeap& desc_heap;
    const Shader::Profile& profile;
    vk::UniquePipeline pipeline;
    vk::UniquePipelineLayout pipeline_layout;
    vk::UniqueDescriptorSetLayout desc_layout;
    std::array<const Shader::Info*, Shader::MaxStageTypes> stages{};
    bool uses_push_descriptors{};
    bool is_compute;
    bool uses_native_bindings{};
    bool uses_direct_push_index{};
    u32 native_binding_count{};
    struct NativeStageMappings {
        vk::ShaderStageFlags stage{};
        std::vector<NativeApi::DescriptorSetAndBindingMapping> mappings;
        mutable NativeApi::ShaderDescriptorSetAndBindingMappingInfo info{};
    };
    std::vector<NativeStageMappings> native_stage_mappings;
};

} // namespace Vulkan
