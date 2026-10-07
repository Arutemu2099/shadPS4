// SPDX-FileCopyrightText: Copyright 2025-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <thread>
#include <vector>

#include "shader_recompiler/spirv_variants.h"
#include "common/serdes.h"
#include "core/emulator_settings.h"
#include "shader_recompiler/frontend/fetch_shader.h"
#include "shader_recompiler/info.h"
#include "video_core/cache_storage.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"

namespace Serialization {
/* You should increment versions below once corresponding serialization scheme is changed. */
static constexpr u32 ShaderBinaryVersion = 4u; // GDS append index computed at run time (d6857fbc)
static constexpr u32 ShaderMetaVersion = 4u; // ImageSpecialization gained is_block_1d
static constexpr u32 PipelineKeyVersion = 4u; // SerializationSupport gained image_binding_counts
} // namespace Serialization

namespace Vulkan {

u32 GetShaderBinaryVersion() {
    return Serialization::ShaderBinaryVersion;
}

void RegisterPipelineData(const ComputePipelineKey& key,
                          ComputePipeline::SerializationSupport& sdata) {
    if (!Storage::DataBase::Instance().IsOpened()) {
        return;
    }

    Serialization::Archive ar{};
    Serialization::Writer pldata{ar};

    pldata.Write(Serialization::PipelineKeyVersion);
    pldata.Write(u32{1}); // compute

    key.Serialize(ar);
    sdata.Serialize(ar);

    Storage::DataBase::Instance().Save(Storage::BlobType::PipelineKey,
                                       fmt::format("c_{:#018x}", key.value), ar.TakeOff());
}

void RegisterPipelineData(const GraphicsPipelineKey& key, u64 hash,
                          GraphicsPipeline::SerializationSupport& sdata) {
    if (!Storage::DataBase::Instance().IsOpened()) {
        return;
    }

    Serialization::Archive ar{};
    Serialization::Writer pldata{ar};

    pldata.Write(Serialization::PipelineKeyVersion);
    pldata.Write(u32{0}); // graphics

    key.Serialize(ar);
    sdata.Serialize(ar);

    Storage::DataBase::Instance().Save(Storage::BlobType::PipelineKey,
                                       fmt::format("g_{:#018x}", hash), ar.TakeOff());
}

void RegisterShaderMeta(const Shader::Info& info,
                        const std::optional<Shader::Gcn::FetchShaderData>& fetch_shader_data,
                        const Shader::StageSpecialization& spec, size_t perm_hash,
                        size_t perm_idx) {
    if (!Storage::DataBase::Instance().IsOpened()) {
        return;
    }

    Serialization::Archive ar;
    Serialization::Writer meta{ar};

    meta.Write(Serialization::ShaderMetaVersion);
    meta.Write(Serialization::ShaderBinaryVersion);

    meta.Write(perm_hash);
    meta.Write(perm_idx);

    spec.Serialize(ar);
    info.Serialize(ar);

    Storage::DataBase::Instance().Save(Storage::BlobType::ShaderMeta,
                                       fmt::format("{:#018x}", perm_hash), ar.TakeOff());
}

void RegisterShaderBinary(std::vector<u32>&& spv, u64 pgm_hash, size_t perm_idx) {
    if (!Storage::DataBase::Instance().IsOpened()) {
        return;
    }

    Storage::DataBase::Instance().Save(Storage::BlobType::ShaderBinary,
                                       fmt::format("{:#018x}_{}", pgm_hash, perm_idx),
                                       std::move(spv));
}

bool SameBindingLayout(const Shader::Info& a, const Shader::Info& b) {
    if (a.uses_dma != b.uses_dma || a.buffers.size() != b.buffers.size() ||
        a.images.size() != b.images.size() || a.samplers.size() != b.samplers.size() ||
        a.fmasks.size() != b.fmasks.size()) {
        return false;
    }
    for (size_t i = 0; i < a.buffers.size(); ++i) {
        if (a.buffers[i].buffer_type != b.buffers[i].buffer_type) {
            return false;
        }
    }
    for (size_t i = 0; i < a.images.size(); ++i) {
        if (a.images[i].is_written != b.images[i].is_written) {
            return false;
        }
    }
    return true;
}

bool LoadShaderMeta(Serialization::Archive& ar, Shader::Info& info,
                    std::optional<Shader::Gcn::FetchShaderData>& fetch_shader_data,
                    Shader::StageSpecialization& spec, size_t& perm_idx) {
    Serialization::Reader meta{ar};

    u32 meta_version{};
    meta.Read(meta_version);
    if (meta_version != Serialization::ShaderMetaVersion) {
        return false;
    }

    u32 binary_version{};
    meta.Read(binary_version);
    // Version 3 -> 4 changed only shaders that append / consume through GDS (the counter index
    // was baked at compile time, d6857fbc). A version-3 entry of any other shader is still exact,
    // so the shared cache folder (cache_storage.cpp Open) stays warm across the bump.
    const bool upgradable = binary_version == 3u && Serialization::ShaderBinaryVersion == 4u;
    if (binary_version != Serialization::ShaderBinaryVersion && !upgradable) {
        return false;
    }

    u64 perm_hash_ar{};
    meta.Read(perm_hash_ar);
    meta.Read(perm_idx);

    spec.Deserialize(ar);
    info.Deserialize(ar);
    if (binary_version != Serialization::ShaderBinaryVersion) {
        for (const auto& buffer : info.buffers) {
            if (buffer.buffer_type == Shader::BufferType::GdsBuffer) {
                return false; // recompiled with the fix, stored as version 4
            }
        }
    }

    fetch_shader_data = spec.fetch_shader_data;
    return true;
}

void ComputePipelineKey::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer key{ar};
    key.Write(value);
}

bool ComputePipelineKey::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader key{ar};
    key.Read(value);
    return true;
}

void ComputePipeline::SerializationSupport::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer sdata{ar};

    sdata.Write(image_binding_counts);
}

bool ComputePipeline::SerializationSupport::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader sdata{ar};

    sdata.Read(image_binding_counts);
    return true;
}

bool PipelineCache::LoadComputePipeline(Serialization::Archive& ar) {
    // A prior preload may have returned midway through loading its stages.
    infos.fill(nullptr);
    modules.fill(nullptr);
    fetch_shader.reset();
    compute_key.Deserialize(ar);

    ComputePipeline::SerializationSupport sdata{};
    sdata.Deserialize(ar);

    std::vector<u8> meta_blob;
    Storage::DataBase::Instance().Load(Storage::BlobType::ShaderMeta,
                                       fmt::format("{:#018x}", compute_key.value), meta_blob);
    if (meta_blob.empty()) {
        return false;
    }

    Serialization::Archive meta_ar{std::move(meta_blob)};

    if (!LoadPipelineStage(meta_ar, 0)) {
        return false;
    }

    const auto [it, is_new] = compute_pipelines.try_emplace(compute_key);
    ASSERT(is_new);

    it.value() =
        std::make_unique<ComputePipeline>(instance, scheduler, desc_heap, profile, *pipeline_cache,
                                          compute_key, *infos[0], modules[0], sdata, true);

    infos.fill(nullptr);
    modules.fill(nullptr);

    return true;
}

void GraphicsPipelineKey::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer key{ar};

    key.Write(this, sizeof(*this));
}

bool GraphicsPipelineKey::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader key{ar};

    key.Read(this, sizeof(*this));
    return true;
}

void GraphicsPipeline::SerializationSupport::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer sdata{ar};

    sdata.Write(&vertex_attributes, sizeof(vertex_attributes));
    sdata.Write(&vertex_bindings, sizeof(vertex_bindings));
    sdata.Write(&divisors, sizeof(divisors));
    sdata.Write(multisampling);
    sdata.Write(tcs);
    sdata.Write(tes);
    sdata.Write(image_binding_counts);
}

bool GraphicsPipeline::SerializationSupport::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader sdata{ar};

    sdata.Read(&vertex_attributes, sizeof(vertex_attributes));
    sdata.Read(&vertex_bindings, sizeof(vertex_bindings));
    sdata.Read(&divisors, sizeof(divisors));
    sdata.Read(multisampling);
    sdata.Read(tcs);
    sdata.Read(tes);
    sdata.Read(image_binding_counts);
    return true;
}

namespace {
/// SHADPS4_PARALLEL_WARMUP=1 (default off, 2026-10-07): the boot preload constructs the cached
/// graphics pipelines on several threads after loading them in order. The constructor holds the
/// driver compile, ~30 ms per pipeline the driver's own cache does not hold (measured s176): with
/// ~5,700 cached graphics pipelines that was the minutes-long black screen of a build's first
/// launch. Everything the constructor reads (Infos, modules, fetch data) is complete before the
/// threads start and nothing else runs (the game has not started).
struct PreloadGraphicsJob {
    GraphicsPipelineKey key{};
    std::array<const Shader::Info*, MaxShaderStages> infos{};
    std::array<Shader::RuntimeInfo, MaxShaderStages> runtime_infos{};
    std::array<vk::ShaderModule, MaxShaderStages> modules{};
    std::shared_ptr<const Shader::Gcn::FetchShaderData> fetch_shader;
    GraphicsPipeline::SerializationSupport sdata{};
    std::unique_ptr<GraphicsPipeline> pipeline;
};
/// Set only while WarmUp runs with SHADPS4_PARALLEL_WARMUP (single-threaded loading phase).
std::vector<PreloadGraphicsJob>* g_preload_jobs = nullptr;

bool ParallelWarmUpEnabled() {
    const char* value = std::getenv("SHADPS4_PARALLEL_WARMUP");
    return value && value[0] == '1';
}
} // Anonymous namespace

bool PipelineCache::LoadGraphicsPipeline(Serialization::Archive& ar) {
    infos.fill(nullptr);
    modules.fill(nullptr);
    fetch_shader.reset();
    graphics_key.Deserialize(ar);

    GraphicsPipeline::SerializationSupport sdata{};
    sdata.Deserialize(ar);

    for (int stage_idx = 0; stage_idx < MaxShaderStages; ++stage_idx) {
        const auto& hash = graphics_key.stage_hashes[stage_idx];
        if (!hash) {
            continue;
        }

        std::vector<u8> meta_blob;
        Storage::DataBase::Instance().Load(Storage::BlobType::ShaderMeta,
                                           fmt::format("{:#018x}", hash), meta_blob);
        if (meta_blob.empty()) {
            return false;
        }

        Serialization::Archive meta_ar{std::move(meta_blob)};

        if (!LoadPipelineStage(meta_ar, stage_idx)) {
            return false;
        }
    }

    if (g_preload_jobs) {
        // SHADPS4_PARALLEL_WARMUP: constructed by WarmUp's threads, inserted after them.
        auto& job = g_preload_jobs->emplace_back();
        job.key = graphics_key;
        job.infos = infos;
        job.runtime_infos = runtime_infos;
        job.modules = modules;
        job.fetch_shader = fetch_shader;
        job.sdata = std::move(sdata);
        infos.fill(nullptr);
        modules.fill(nullptr);
        fetch_shader.reset();
        return true;
    }

    const auto [it, is_new] = graphics_pipelines.try_emplace(graphics_key);
    ASSERT(is_new);

    it.value() = std::make_unique<GraphicsPipeline>(
        instance, scheduler, desc_heap, profile, graphics_key, *pipeline_cache, infos,
        runtime_infos, fetch_shader, modules, sdata, true);

    infos.fill(nullptr);
    modules.fill(nullptr);
    fetch_shader.reset();

    return true;
}

bool PipelineCache::LoadPipelineStage(Serialization::Archive& ar, size_t stage) {
    auto program = std::make_unique<Program>();
    Shader::StageSpecialization spec{};
    spec.info = &program->info;
    size_t perm_idx{};
    std::optional<Shader::Gcn::FetchShaderData> loaded_fetch;
    if (!LoadShaderMeta(ar, program->info, loaded_fetch, spec, perm_idx)) {
        return false;
    }
    // Preserve loader scratch state on later failure; on success use the module's
    // immutable owner below. Serialized values and cache format stay unchanged.
    fetch_shader = loaded_fetch
                       ? std::make_shared<const Shader::Gcn::FetchShaderData>(*loaded_fetch)
                       : nullptr;

    std::vector<u32> spv{};
    Storage::DataBase::Instance().Load(Storage::BlobType::ShaderBinary,
                                       fmt::format("{:#018x}_{}", program->info.pgm_hash, perm_idx),
                                       spv);
    if (spv.empty()) {
        return false;
    }

    // Permutation hash depends on shader variation index. To prevent collisions, we need insert it
    // at the exact position rather than append

    vk::ShaderModule module{};

    auto [it_pgm, new_program] = program_cache.try_emplace(program->info.pgm_hash);
    if (new_program) {
        module = CompileSPV(spv, instance.GetDevice());
        it_pgm.value() = std::move(program);
    } else {
        const auto& it = std::ranges::find(it_pgm.value()->modules, spec, &Program::Module::spec);
        if (it != it_pgm.value()->modules.end()) {
            // A matching permutation is valid only at its original index. A different index means
            // the store holds entries from more than one cache generation, so this pipeline is
            // left to compile at runtime.
            const auto idx = std::distance(it_pgm.value()->modules.begin(), it);
            if (perm_idx != idx) {
                LOG_WARNING(Render_Vulkan,
                            "Cached permutation {} of {}_{:x} conflicts with index {}, skipping "
                            "preload",
                            perm_idx, program->info.stage, program->info.pgm_hash, idx);
                return false;
            }
            module = it->module;
        } else {
            // Every permutation is bound through the program's first Info. A cached permutation
            // from another cache generation can have other bindings (DMA buffers shift every
            // image), so it is left to compile at runtime, where it is pinned to that Info.
            if (!SameBindingLayout(program->info, it_pgm.value()->info)) {
                LOG_WARNING(Render_Vulkan,
                            "Cached permutation {} of {}_{:x} has a different binding layout "
                            "than its program, skipping preload",
                            perm_idx, program->info.stage, program->info.pgm_hash);
                return false;
            }
            module = CompileSPV(spv, instance.GetDevice());
        }
    }
    it_pgm.value()->InsertPermut(module, std::move(spec), perm_idx);
    fetch_shader = it_pgm.value()->modules[perm_idx].fetch_shader;

    infos[stage] = &it_pgm.value()->info;
    modules[stage] = module;

    return true;
}

void PipelineCache::WarmUp() {
    if (!EmulatorSettings.IsPipelineCacheEnabled()) {
        return;
    }
    if (Shader::SpirvVariantActive()) {
        LOG_WARNING(Render, "A SPIR-V variant flag is set: the pipeline cache is not used this run");
        return;
    }
    // SHADPS4_PIPELINE_CACHE_SKIP=1: neither preload nor save the on-disk cache this run, so
    // every pipeline compiles at first use (a test bed for SHADPS4_ASYNC_PIPELINES that needs no
    // new area). Unlike the SPIR-V variant flags it changes nothing in the shaders; the store
    // stays unopened, so Register* and Sync are no-ops.
    static const bool skip_cache = [] {
        const char* value = std::getenv("SHADPS4_PIPELINE_CACHE_SKIP");
        return value && value[0] == '1';
    }();
    if (skip_cache) {
        LOG_WARNING(Render, "SHADPS4_PIPELINE_CACHE_SKIP: the pipeline cache is neither read nor "
                            "written this run");
        return;
    }

    Storage::DataBase::Instance().Open(instance.IsNativeBindingsActive());

    // Check if cache is compatible
    std::vector<u8> profile_data{};
    Storage::DataBase::Instance().Load(Storage::BlobType::ShaderProfile, "profile", profile_data);
    if (profile_data.empty()) {
        Storage::DataBase::Instance().FinishPreload();

        profile_data.resize(sizeof(profile));
        std::memcpy(profile_data.data(), &profile, sizeof(profile));
        Storage::DataBase::Instance().Save(Storage::BlobType::ShaderProfile, "profile",
                                           std::move(profile_data));
        return;
    }
    if (profile_data.size() != sizeof(Shader::Profile)) {
        LOG_WARNING(Render,
                    "Pipeline cache profile has unexpected size ({} != {}). Ignoring the cache",
                    profile_data.size(), sizeof(Shader::Profile));
        Storage::DataBase::Instance().Close();
        return;
    }

    Shader::Profile cached_profile{};
    std::memcpy(&cached_profile, profile_data.data(), sizeof(cached_profile));
    if (cached_profile != profile) {
        LOG_WARNING(Render,
                    "Pipeline cache isn't compatible with current system. Ignoring the cache");
        Storage::DataBase::Instance().Close();
        return;
    }

    u32 num_pipelines{};
    u32 num_total_pipelines{};

    const auto t_warmup = std::chrono::steady_clock::now();
    std::vector<PreloadGraphicsJob> preload_jobs;
    if (ParallelWarmUpEnabled()) {
        g_preload_jobs = &preload_jobs;
    }
    Storage::DataBase::Instance().ForEachBlob(
        Storage::BlobType::PipelineKey, [&](std::vector<u8>&& data) {
            ++num_total_pipelines;

            Serialization::Archive ar{std::move(data)};
            Serialization::Reader pldata{ar};

            u32 version{};
            pldata.Read(version);
            if (version != Serialization::PipelineKeyVersion) {
                return;
            }

            u32 is_compute{};
            pldata.Read(is_compute);

            bool result{};
            if (is_compute) {
                result = LoadComputePipeline(ar);
            } else {
                result = LoadGraphicsPipeline(ar);
            }

            if (result) {
                ++num_pipelines;
            }
        });
    g_preload_jobs = nullptr;

    if (!preload_jobs.empty()) {
        const auto t_build = std::chrono::steady_clock::now();
        const char* env = std::getenv("SHADPS4_WARMUP_THREADS");
        const u32 hw = std::max(2u, std::thread::hardware_concurrency());
        const u32 num_threads = std::clamp<u32>(
            env ? static_cast<u32>(std::strtoul(env, nullptr, 10)) : hw / 2, 1u, 16u);
        std::atomic<size_t> next{0};
        {
            std::vector<std::jthread> threads;
            for (u32 t = 0; t < num_threads; ++t) {
                threads.emplace_back([&] {
                    for (size_t i = next.fetch_add(1); i < preload_jobs.size();
                         i = next.fetch_add(1)) {
                        auto& job = preload_jobs[i];
                        job.pipeline = std::make_unique<GraphicsPipeline>(
                            instance, scheduler, desc_heap, profile, job.key, *pipeline_cache,
                            job.infos, job.runtime_infos, job.fetch_shader, job.modules, job.sdata,
                            true);
                    }
                });
            }
        } // joins
        for (auto& job : preload_jobs) {
            const auto [it, is_new] = graphics_pipelines.try_emplace(job.key);
            ASSERT(is_new);
            it.value() = std::move(job.pipeline);
        }
        LOG_INFO(Render, "WarmUp: {} graphics pipelines built on {} threads in {} ms",
                 preload_jobs.size(), num_threads,
                 std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::steady_clock::now() - t_build)
                     .count());
    }
    LOG_INFO(Render, "WarmUp: total {} ms",
             std::chrono::duration_cast<std::chrono::milliseconds>(
                 std::chrono::steady_clock::now() - t_warmup)
                 .count());

    LOG_INFO(Render, "Preloaded {} pipelines", num_pipelines);
    if (num_total_pipelines > num_pipelines) {
        LOG_WARNING(Render, "{} stale pipelines were found. Consider re-generating the cache",
                    num_total_pipelines - num_pipelines);
    }

    Storage::DataBase::Instance().FinishPreload();
}

void PipelineCache::Sync() {
    Storage::DataBase::Instance().Close();
}

} // namespace Vulkan

namespace Shader {

void Info::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer info{ar};

    info.Write(this, sizeof(InfoPersistent));
    info.Write(flattened_ud_buf);
    srt_info.Serialize(ar);
}

bool Info::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader info{ar};

    info.Read(this, sizeof(Shader::InfoPersistent));
    info.Read(flattened_ud_buf);

    return srt_info.Deserialize(ar);
}

void Gcn::FetchShaderData::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer fetch{ar};
    ar.Grow(6 + attributes.size() * sizeof(VertexAttribute));

    fetch.Write(size);
    fetch.Write(vertex_offset_sgpr);
    fetch.Write(instance_offset_sgpr);
    fetch.Write(attributes);
}

bool Gcn::FetchShaderData::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader fetch{ar};

    fetch.Read(size);
    fetch.Read(vertex_offset_sgpr);
    fetch.Read(instance_offset_sgpr);
    fetch.Read(attributes);

    return true;
}

void PersistentSrtInfo::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer srt{ar};

    srt.Write(this, sizeof(*this));
    if (walker_func_size) {
        srt.Write(reinterpret_cast<void*>(walker_func), walker_func_size);
    }
}

bool PersistentSrtInfo::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader srt{ar};

    srt.Read(this, sizeof(*this));

    if (walker_func_size) {
        walker_func = RegisterWalkerCode(ar.CurrPtr(), walker_func_size);
        ar.Advance(walker_func_size);
    }

    return true;
}

void StageSpecialization::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer spec{ar};

    spec.Write(start);
    spec.Write(runtime_info);

    spec.Write(bitset.to_string());

    if (fetch_shader_data) {
        spec.Write(sizeof(*fetch_shader_data));
        fetch_shader_data->Serialize(ar);
    } else {
        spec.Write(size_t{0});
    }

    spec.Write(vs_attribs);
    spec.Write(buffers);
    spec.Write(images);
    spec.Write(fmasks);
    spec.Write(samplers);
}

bool StageSpecialization::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader spec{ar};

    spec.Read(start);
    spec.Read(runtime_info);

    std::string bits{};
    spec.Read(bits);
    bitset = std::bitset<MaxStageResources>(bits);

    u64 fetch_data_size{};
    spec.Read(fetch_data_size);

    if (fetch_data_size) {
        Gcn::FetchShaderData fetch_data;
        fetch_data.Deserialize(ar);
        fetch_shader_data = fetch_data;
    }

    spec.Read(vs_attribs);
    spec.Read(buffers);
    spec.Read(images);
    spec.Read(fmasks);
    spec.Read(samplers);

    return true;
}

} // namespace Shader
