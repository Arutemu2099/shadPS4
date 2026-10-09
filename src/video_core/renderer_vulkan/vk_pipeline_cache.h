// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <unordered_set>
#include <variant>
#include <tsl/robin_map.h>
#include "shader_recompiler/prepared_specialization.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/recompiler.h"
#include "shader_recompiler/specialization.h"
#include "video_core/renderer_vulkan/prepared_graphics_state.h"
#include "video_core/renderer_vulkan/vk_compute_pipeline.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_resource_pool.h"

template <>
struct std::hash<vk::ShaderModule> {
    std::size_t operator()(const vk::ShaderModule& module) const noexcept {
        return std::hash<size_t>{}(reinterpret_cast<size_t>((VkShaderModule)module));
    }
};

namespace AmdGpu {
class Liverpool;
}

namespace Serialization {
struct Archive;
}

namespace Shader {
struct Info;
}

namespace Vulkan {

class Instance;
class Scheduler;
class ShaderCache;

struct Program {
    struct Module {
        vk::ShaderModule module;
        Shader::StageSpecialization spec;
        // Immutable and independently owned: module-vector growth/replacement
        // cannot invalidate metadata retained by a pipeline or a current draw.
        std::shared_ptr<const Shader::Gcn::FetchShaderData> fetch_shader;
    };
    static constexpr size_t MaxPermutations = 8;
    using ModuleList = boost::container::small_vector<Module, MaxPermutations>;

    Shader::Info info;
    ModuleList modules{};
    // MRU guards only; guest user data and descriptors still refresh before each probe.
    // The secondary is allocated only for the opt-in two-entry experiment.
    std::unique_ptr<Shader::PreparedSpecialization> prepared;
    std::unique_ptr<Shader::PreparedSpecialization> prepared_secondary;
    // Bumped by every module-list or module-handle mutation. Prepared graphics pipeline
    // entries record it, since a new/inserted permutation can change first-match selection.
    u64 generation{};
    // SHADPS4_PRECOMPILE: built before the game dispatched it; first live use counted once.
    bool precompiled{};
    bool precompiled_seen{};

    Program() = default;
    Program(Shader::Stage stage, Shader::LogicalStage l_stage, Shader::ShaderParams params)
        : info{stage, l_stage, params} {}

    void InvalidatePrepared() {
        ++generation;
        if (prepared) {
            prepared->valid = false;
        }
        if (prepared_secondary) {
            prepared_secondary->valid = false;
        }
    }

    void AddPermut(vk::ShaderModule module, Shader::StageSpecialization&& spec) {
        InvalidatePrepared();
        auto fetch = spec.fetch_shader_data
                         ? std::make_shared<const Shader::Gcn::FetchShaderData>(*spec.fetch_shader_data)
                         : nullptr;
        modules.emplace_back(module, std::move(spec), std::move(fetch));
    }

    void InsertPermut(vk::ShaderModule module, Shader::StageSpecialization&& spec,
                      size_t perm_idx) {
        InvalidatePrepared();
        modules.resize(std::max(modules.size(), perm_idx + 1)); // <-- beware of realloc
        auto fetch = spec.fetch_shader_data
                         ? std::make_shared<const Shader::Gcn::FetchShaderData>(*spec.fetch_shader_data)
                         : nullptr;
        modules[perm_idx] = {module, std::move(spec), std::move(fetch)};
    }
};

class PipelineCache {
public:
    explicit PipelineCache(const Instance& instance, Scheduler& scheduler,
                           AmdGpu::Liverpool* liverpool);
    ~PipelineCache();

    void WarmUp();
    // SHADPS4_PRECOMPILE (vk_pipeline_precompile.cpp): run by WarmUp on the main thread.
    void PrecompileFromExtract();
    // SHADPS4_PRECOMPILE, Fable finding 1: a precompiled program whose first live dispatch matched
    // none of its guessed permutations leaves program_cache with its compute pipelines; both are
    // parked (a pool compile of such a pipeline may still be running), never destroyed early.
    void RetirePrecompiled(u64 hash);
    void Sync();

    bool LoadComputePipeline(Serialization::Archive& ar);
    // The second half of LoadComputePipeline (WarmUp runs it later for precompiled metas).
    bool BuildComputePipeline(const ComputePipelineKey& key,
                              ComputePipeline::SerializationSupport& sdata,
                              std::vector<u8>&& meta_blob);
    bool LoadGraphicsPipeline(Serialization::Archive& ar);
    bool LoadPipelineStage(Serialization::Archive& ar, size_t stage);

    const GraphicsPipeline* GetGraphicsPipeline();

    /// SHADPS4_NATIVE_PASS (front / lookup thread, right after GetGraphicsPipeline returned
    /// `base` for the current registers): decides once whether `base` can draw native batches
    /// and, if so, builds its native variant (vertex shader re-emitted to read per-draw records,
    /// layout + one records SSBO) and publishes it in base->native_variant.
    void PrepareNativeVariant(const GraphicsPipeline* base);

    /// SHADPS4_NATIVE_COLOUR (front / lookup thread, right after the front derived the draw's
    /// vertex inputs): decides once whether `base` (a textured pipeline) can draw native batches
    /// and, for each new vertex description, translates its native VS + PS here and queues the
    /// driver compile on the native compiler thread, which publishes it in base->native_colour.
    void PrepareNativeColour(const GraphicsPipeline* base,
                             const AmdGpu::CpSplit::VertexInputsDerived& vi);

    const ComputePipeline* GetComputePipeline();

    using Result = std::tuple<const Shader::Info*, vk::ShaderModule,
                              std::shared_ptr<const Shader::Gcn::FetchShaderData>, u64>;
    Result GetProgram(Shader::Stage stage, Shader::LogicalStage l_stage,
                      const Shader::ShaderParams& params, Shader::Backend::Bindings& binding);

    std::optional<vk::ShaderModule> ReplaceShader(vk::ShaderModule module,
                                                  std::span<const u32> spv_code);

    static std::string GetShaderName(Shader::Stage stage, u64 hash,
                                     std::optional<size_t> perm = {});

    auto& GetProfile() const {
        return profile;
    }

private:
    bool RefreshGraphicsKey();
    bool RefreshGraphicsStages();
    bool RefreshComputeKey();

    // SHADPS4_PREPARED_PIPELINES: one successful original VS+PS selection, reused while its
    // exact register snapshot, both per-stage specialization guards, both program
    // generations and the table epoch still hold. Command-processor thread only.
    struct PreparedPipelineEntry {
        bool valid{};
        u64 epoch{};
        u64 last_use{}; // SHADPS4_PREPARED_LRU: prepared_clock at the last hit or seed
        PreparedGraphicsState state;
        Program* fs_program{};
        Program* vs_program{};
        u64 fs_generation{};
        u64 vs_generation{};
        // Their captured RuntimeInfo bytes/permutation also restore the cache outputs.
        Shader::PreparedSpecialization fs_guard;
        Shader::PreparedSpecialization vs_guard;
        const GraphicsPipeline* pipeline{};
        vk::ShaderModule fs_module;
        vk::ShaderModule vs_module;
        std::shared_ptr<const Shader::Gcn::FetchShaderData> fetch_shader;
    };
    struct StageSelection {
        Program* program{};
        size_t permutation{};
        bool compiled{};
    };
    static constexpr size_t PreparedPipelineSets = 1024;
    static constexpr size_t PreparedPipelineMaxWays = 32;
    /// SHADPS4_PREPARED_WAYS=<4..32> (default 4): associativity of the prepared-pipeline table.
    static size_t PreparedPipelineWays();

    PreparedPipelineEntry* TryPreparedPipeline(PreparedGraphicsState& state, bool& eligible);
    void SeedPreparedPipeline(const PreparedGraphicsState& state, const GraphicsPipeline* pipeline);
    void VerifyPreparedPipeline(PreparedPipelineEntry& entry,
                                const GraphicsPipeline* pipeline, bool is_new);
    void NoteStageSelection(Shader::LogicalStage l_stage, Program* program, size_t permutation,
                            bool compiled);

    void DumpShader(std::span<const u32> code, u64 hash, Shader::Stage stage, size_t perm_idx,
                    std::string_view ext);
    std::optional<std::vector<u32>> GetShaderPatch(u64 hash, Shader::Stage stage, size_t perm_idx,
                                                   std::string_view ext);
    vk::ShaderModule CompileModule(Shader::Info& info, Shader::RuntimeInfo& runtime_info,
                                   const std::span<const u32>& code, size_t perm_idx,
                                   Shader::Backend::Bindings& binding);
    const Shader::RuntimeInfo& BuildRuntimeInfo(Shader::Stage stage, Shader::LogicalStage l_stage);

    [[nodiscard]] bool IsPipelineCacheDirty() const {
        return num_new_pipelines > 0;
    }

    // SHADPS4_ASYNC_PIPELINES (default off): a new graphics key is handed to one compiler thread
    // and the lookup returns null (the draw is skipped) until a later lookup publishes the
    // finished pipeline. graphics_pipelines keeps its single mutator: only the lookup thread
    // inserts (a null entry marks a pending key) and assigns. Compute pipelines stay synchronous.
    struct AsyncJob;
    struct AsyncCompiler;
    void QueueAsyncPipeline(u64 pipeline_hash);
    void PublishAsyncPipelines();
    void NoteCompiledPipeline(u64 hash, double translate_ms, double create_ms, u32 queued_ahead,
                              bool async);

private:
    const Instance& instance;
    Scheduler& scheduler;
    AmdGpu::Liverpool* liverpool;
    DescriptorHeap desc_heap;
    vk::UniquePipelineCache pipeline_cache;
    vk::UniquePipelineLayout pipeline_layout;
    Shader::Profile profile{};
    Shader::Pools pools;
    tsl::robin_map<size_t, std::unique_ptr<Program>> program_cache;
    tsl::robin_map<ComputePipelineKey, std::unique_ptr<ComputePipeline>> compute_pipelines;
    std::vector<std::unique_ptr<Program>> retired_programs;          ///< RetirePrecompiled
    std::vector<std::unique_ptr<ComputePipeline>> retired_pipelines; ///< RetirePrecompiled
    // SHADPS4_PRECOMPILE, Opus review 2026-10-07 fix A. A retire leaves the guessed (marked)
    // metas of the program's higher permutation indices in the cache next to the live rebuild's.
    // Programs retired in any earlier launch (sidecar precompile-retired.txt, read by WarmUp):
    // their marked metas are not loaded. All other marked compute metas load after the unmarked
    // ones (deferred), so a live meta, never a guessed one, creates a program's shared Info.
    std::unordered_set<u64> precompile_retired;
    struct DeferredComputeLoad {
        ComputePipelineKey key{};
        ComputePipeline::SerializationSupport sdata{};
        std::vector<u8> meta;
    };
    std::vector<DeferredComputeLoad>* defer_marked_compute{}; ///< set only during WarmUp's pass
    u32 retired_metas_skipped{};
    tsl::robin_map<GraphicsPipelineKey, std::unique_ptr<GraphicsPipeline>> graphics_pipelines;
    std::vector<std::unique_ptr<GraphicsPipeline>> native_variants; // SHADPS4_NATIVE_PASS
    std::array<Shader::RuntimeInfo, MaxShaderStages> runtime_infos{};
    std::array<const Shader::Info*, MaxShaderStages> infos{};
    std::array<vk::ShaderModule, MaxShaderStages> modules{};
    std::shared_ptr<const Shader::Gcn::FetchShaderData> fetch_shader;
    GraphicsPipelineKey graphics_key{};
    ComputePipelineKey compute_key{};
    u32 num_new_pipelines{}; // new pipelines added to the cache since the game start

    std::unique_ptr<AsyncCompiler> async_compiler; // started by the first async job
    struct NativeColourJob;
    struct NativeColourCompiler;
    std::chrono::steady_clock::time_point native_colour_last{};
    u32 native_colour_logged{};
    u32 async_pending{};         // queued, not yet published; lookup thread only
    double lookup_translate_ms{}; // GetProgram's module creation time within this lookup
    // PipelineCompile / PipelineCompileSummary log lines (both modes). skipped_draws counts
    // lookups that returned null because the key's compile was pending.
    struct CompileStats {
        u64 n{};
        u64 skipped_draws{};
        double translate_ms_total{};
        double create_ms_total{};
        double max_create_ms{};
    } compile_stats{};

    // Allocated on first use when SHADPS4_PREPARED_PIPELINES=1. The epoch invalidates every
    // entry at once whenever a pipeline is erased or a module handle is replaced.
    std::vector<PreparedPipelineEntry> prepared_pipelines;
    // SHADPS4_PREPARED_TAGS=1: state hash per way, written at seeding, so a lookup only
    // touches the (large, ~1 KB apart) entries whose hash matches. A stale tag only costs the
    // full compare the lookup did before.
    std::vector<u64> prepared_tags;
    std::array<StageSelection, MaxShaderStages> stage_selections{};
    u64 prepared_pipeline_epoch{1};
    u32 prepared_pipeline_victim{};
    u64 prepared_clock{};

    // Only if Config::collectShadersForDebug()
    tsl::robin_map<vk::ShaderModule,
                   std::vector<std::variant<GraphicsPipelineKey, ComputePipelineKey>>>
        module_related_pipelines;

    // SHADPS4_NATIVE_COLOUR: last member, so it stops (and destroys the variants it owns)
    // before anything else of the cache goes.
    std::unique_ptr<NativeColourCompiler> native_colour_compiler;
};

} // namespace Vulkan
