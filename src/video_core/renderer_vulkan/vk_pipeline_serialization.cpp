// SPDX-FileCopyrightText: Copyright 2025-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <thread>
#include <vector>

#include "shader_recompiler/spirv_variants.h"
#include "common/hash.h"
#include "common/serdes.h"
#include "core/emulator_settings.h"
#include "shader_recompiler/frontend/fetch_shader.h"
#include "shader_recompiler/info.h"
#include "video_core/cache_storage.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_pipeline_compile_pool.h"
#include "video_core/renderer_vulkan/vk_pipeline_serialization.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"

namespace Serialization {
/* You should increment versions below once corresponding serialization scheme is changed. */
static constexpr u32 ShaderBinaryVersion = 4u; // GDS append index computed at run time (d6857fbc)
static constexpr u32 ShaderMetaVersion = 4u; // ImageSpecialization gained is_block_1d
static constexpr u32 PipelineKeyVersion = 4u; // SerializationSupport gained image_binding_counts
// SHADPS4_PRECOMPILE: metas of permutations built from guessed contexts carry this bit, so builds
// whose compute key lacks num_user_data / LDS size (before 2026-10-07) reject them as stale.
static constexpr u32 PrecompiledMetaBit = 0x80000000u;
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
                        size_t perm_idx, bool precompiled) {
    if (!Storage::DataBase::Instance().IsOpened()) {
        return;
    }
    if (info.nodma_pinned || Shader::NoDmaPinnedHash(info.pgm_hash)) {
        return; // q9b: a NODMA module never enters the shader cache
    }

    Serialization::Archive ar;
    Serialization::Writer meta{ar};

    meta.Write(Serialization::ShaderMetaVersion |
               (precompiled ? Serialization::PrecompiledMetaBit : 0u));
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
    if (Shader::NoDmaPinnedHash(pgm_hash)) {
        return; // q9b
    }

    Storage::DataBase::Instance().Save(Storage::BlobType::ShaderBinary,
                                       fmt::format("{:#018x}_{}", pgm_hash, perm_idx),
                                       std::move(spv));
}

bool SameBindingLayout(const Shader::Info& a, const Shader::Info& b) {
    // ud_mask: dispatch pushes user data by the program's shared mask (Fable finding 1).
    if (a.uses_dma != b.uses_dma || a.buffers.size() != b.buffers.size() ||
        a.images.size() != b.images.size() || a.samplers.size() != b.samplers.size() ||
        a.fmasks.size() != b.fmasks.size() || a.ud_mask.mask != b.ud_mask.mask) {
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
                    Shader::StageSpecialization& spec, size_t& perm_idx, bool* precompiled) {
    Serialization::Reader meta{ar};

    u32 meta_version{};
    meta.Read(meta_version);
    if ((meta_version & ~Serialization::PrecompiledMetaBit) != Serialization::ShaderMetaVersion) {
        return false;
    }
    if (precompiled) {
        *precompiled = (meta_version & Serialization::PrecompiledMetaBit) != 0;
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
    if (Shader::NoDmaMode() && info.stage == Shader::Stage::Compute && info.uses_dma) {
        return false; // q9b: translate it (the NODMA rule may apply)
    }
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
    // SHADPS4_PRECOMPILE fix A: a guessed (marked) meta loads after every unmarked one. The
    // version word written first by RegisterShaderMeta carries the marker.
    if (defer_marked_compute && meta_blob.size() >= sizeof(u32)) {
        u32 meta_version{};
        std::memcpy(&meta_version, meta_blob.data(), sizeof(meta_version));
        if ((meta_version & Serialization::PrecompiledMetaBit) != 0) {
            defer_marked_compute->push_back({compute_key, std::move(sdata), std::move(meta_blob)});
            return true; // counted as loaded; WarmUp corrects the count if it fails later
        }
    }
    return BuildComputePipeline(compute_key, sdata, std::move(meta_blob));
}

bool PipelineCache::BuildComputePipeline(const ComputePipelineKey& key,
                                         ComputePipeline::SerializationSupport& sdata,
                                         std::vector<u8>&& meta_blob) {
    infos.fill(nullptr);
    modules.fill(nullptr);
    fetch_shader.reset();
    compute_key = key;

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

/// SHADPS4_CACHE_REBUILD (2026-10-16): 1 rebuild missing SPIR-V from the game's files, 2 VERIFY
/// cached SPIR-V against a rebuild, 3 both (LoadPipelineStage).
int CacheRebuildMode() {
    static const int mode = [] {
        const char* value = std::getenv("SHADPS4_CACHE_REBUILD");
        return value ? std::atoi(value) : 0;
    }();
    return mode;
}

/// vs-formats.bin key: shader hashes have a zero upper half (measured), perm < 256.
u64 VsFormatKey(u64 pgm_hash, size_t perm_idx) {
    return (pgm_hash << 8) | (perm_idx & 0xff);
}

constexpr u32 VsFormatsMagic = 0x4d465356; // "VSFM"
constexpr u32 VsFormatsVersion = 2; // codes per attribute semantic
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
    bool marked = false;
    if (!LoadShaderMeta(ar, program->info, loaded_fetch, spec, perm_idx, &marked)) {
        return false;
    }
    if (marked && precompile_retired.contains(program->info.pgm_hash)) {
        // Fix A: a guess left over from a program retired after a live miss in an earlier launch.
        ++retired_metas_skipped;
        return false;
    }
    // SHADPS4_PRECOMPILE (Fable review 2026-10-07 finding 5): keep the marker across launches, so
    // the first live dispatch still reports PrecompiledUse and a MISS still retires the program.
    // A program counts as precompiled only while every loaded permutation is (fix A, below).
    program->precompiled = marked;
    // Preserve loader scratch state on later failure; on success use the module's
    // immutable owner below. Serialized values and cache format stay unchanged.
    fetch_shader = loaded_fetch
                       ? std::make_shared<const Shader::Gcn::FetchShaderData>(*loaded_fetch)
                       : nullptr;

    std::vector<u32> spv{};
    Storage::DataBase::Instance().Load(Storage::BlobType::ShaderBinary,
                                       fmt::format("{:#018x}_{}", program->info.pgm_hash, perm_idx),
                                       spv);
    // SHADPS4_CACHE_REBUILD (2026-10-16): =1 a stage whose SPIR-V is missing is made again from the
    // game's own code + this meta (RebuildStageSpv) and saved; =2 VERIFY: every stage that has
    // SPIR-V is made again and compared word for word (nothing changes). =3 both.
    bool rebuilt_known = false;
    if (const int rebuild = CacheRebuildMode(); rebuild != 0) {
        const size_t si = std::min<size_t>(static_cast<size_t>(program->info.stage), 7);
        auto& rs = rebuild_stats;
        // Bit 4 TRACE (s1297 stopped inside WarmUp without a log line): the stage about to be
        // rebuilt, written and flushed to the cache folder's rebuild-inprogress.txt first.
        if (rebuild & 4) {
            static std::ofstream trace{
                Storage::DataBase::Instance().SidecarPath("rebuild-inprogress.txt"),
                std::ios::binary | std::ios::trunc};
            static u64 traced = 0;
            trace.seekp(0);
            trace << fmt::format("{} {:#018x} perm {} marked {} spv {} n {}            \n",
                                 program->info.stage, program->info.pgm_hash, perm_idx,
                                 marked ? 1 : 0, spv.size(), ++traced);
            trace.flush();
        }
        // A permutation another pipeline already rebuilt (or loaded) needs no SPIR-V: the module
        // is taken from program_cache below.
        const auto it_known = program_cache.find(program->info.pgm_hash);
        const bool in_cache = it_known != program_cache.end() &&
                              std::ranges::find(it_known.value()->modules, spec,
                                                &Program::Module::spec) !=
                                  it_known.value()->modules.end();
        const bool known = spv.empty() && in_cache;
        if (known && (rebuild & 1)) {
            ++rs.reused[si];
            rebuilt_known = true;
        } else if (spv.empty() && (rebuild & 1)) {
            bool same_layout = false;
            const auto it_fmt = vs_formats.find(VsFormatKey(program->info.pgm_hash, perm_idx));
            spv = RebuildStageSpv(program->info, loaded_fetch, spec, &same_layout,
                                  it_fmt != vs_formats.end() ? &it_fmt->second : nullptr);
            if (spv.empty()) {
                ++rs.no_code[si];
                return false;
            }
            if (!same_layout) {
                ++rs.layout_mismatch[si];
                return false;
            }
            ++rs.rebuilt[si];
            RegisterShaderBinary(std::vector<u32>(spv), program->info.pgm_hash, perm_idx);
        } else if (!spv.empty() && (rebuild & 2) && !in_cache) { // each permutation once
            bool same_layout = false;
            const u64 fmt_key = VsFormatKey(program->info.pgm_hash, perm_idx);
            const auto it_fmt = vs_formats.find(fmt_key);
            const auto again = RebuildStageSpv(program->info, loaded_fetch, spec, &same_layout,
                                               it_fmt != vs_formats.end() ? &it_fmt->second
                                                                          : nullptr);
            std::vector<u8> found;
            if (!again.empty() && again != spv && same_layout && (rebuild & 8) &&
                program->info.stage == Shader::Stage::Vertex) {
                found = AnnotateVsConversions(program->info, loaded_fetch, spec, spv);
                if (found.empty()) {
                    ++rs.annotate_failed[si];
                } else {
                    ++rs.annotated[si];
                }
            }
            if (!found.empty()) {
                vs_formats[fmt_key] = std::move(found);
                vs_formats_dirty = true;
            } else if (again.empty()) {
                ++rs.no_code[si];
            } else if (again == spv && same_layout) {
                ++rs.equal[si];
            } else {
                ++rs.differ[si];
                // The shader meta of every differing permutation (file name = HashCombine(hash,
                // perm), as GetProgram saves it): a seed for players drops these, so they compile
                // at play time from the real V#s instead of a rebuild that may decode differently.
                {
                    std::ofstream list{Storage::DataBase::Instance().SidecarPath("rebuild-differ.txt"),
                                       std::ios::app};
                    list << fmt::format("{:#018x} {} {:#x} {}\n",
                                        HashCombine(program->info.pgm_hash, perm_idx),
                                        program->info.stage, program->info.pgm_hash, perm_idx);
                }
                u32 differs = 0;
                for (const u32 n : rs.differ) {
                    differs += n;
                }
                if (differs <= 3) {
                    // Both SPIR-V blobs next to the cache, for spirv-dis + diff.
                    for (const auto& [suffix, words] :
                         {std::pair<const char*, const std::vector<u32>*>{"cached", &spv},
                          std::pair<const char*, const std::vector<u32>*>{"rebuilt", &again}}) {
                        std::ofstream out{Storage::DataBase::Instance().SidecarPath(fmt::format(
                                              "rebuild-diff-{:016x}-{}-{}.spv",
                                              program->info.pgm_hash, perm_idx, suffix)),
                                          std::ios::binary | std::ios::trunc};
                        out.write(reinterpret_cast<const char*>(words->data()),
                                  words->size() * sizeof(u32));
                    }
                }
                if (differs <= 16) {
                    size_t first = 0;
                    while (first < spv.size() && first < again.size() && spv[first] == again[first]) {
                        ++first;
                    }
                    LOG_WARNING(Render_Vulkan,
                                "CacheRebuild verify: {} {:#x} perm {} differs: {} vs cached {} "
                                "words, first difference at word {}, layout {}",
                                program->info.stage, program->info.pgm_hash, perm_idx, again.size(),
                                spv.size(), first, same_layout ? "same" : "DIFFERENT");
                }
            }
        }
    }
    if (spv.empty() && !rebuilt_known) {
        return false;
    }
    DumpOneShader(spv, program->info.pgm_hash, program->info.stage, perm_idx, "cached");

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
        // Fix A: one live (unmarked) permutation makes the program a live one: no PrecompiledUse
        // line and no retire on its first dispatch. Set only for a permutation actually inserted;
        // a live meta skipped above for its layout leaves a guessed program to retire at its
        // first miss.
        it_pgm.value()->precompiled = it_pgm.value()->precompiled && marked;
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
    Storage::DataBase::Instance().Prefetch();

    // Check if cache is compatible
    std::vector<u8> profile_data{};
    Storage::DataBase::Instance().Load(Storage::BlobType::ShaderProfile, "profile", profile_data);
    if (profile_data.empty()) {
        std::vector<u8> fresh(sizeof(profile));
        std::memcpy(fresh.data(), &profile, sizeof(profile));
        // SHADPS4_CACHE_REBUILD=1 (2026-10-16): a folder seeded with pipeline keys + shader metas
        // but no profile and no SPIR-V (a shipped set) goes on to the preload below, which rebuilds
        // every stage from the game's own files for this machine's profile (LoadPipelineStage).
        if ((CacheRebuildMode() & 1) == 0) {
            Storage::DataBase::Instance().FinishPreload();
            Storage::DataBase::Instance().Save(Storage::BlobType::ShaderProfile, "profile",
                                               std::move(fresh));
            // SHADPS4_PRECOMPILE (2026-10-16, notes/study-shader-stutter-2026-10-16.md sec. 4
            // step 3): a fresh cache folder (a new player's first launch) has nothing to preload
            // but still gets the pre-compile pass; this branch used to return, so the pass (and a
            // context table) took effect only from the second launch on. Returns at once when the
            // flag is off.
            precompile_retired = ReadPrecompileHashes(PrecompileRetiredFile);
            PrecompileFromExtract();
            return;
        }
        profile_data = fresh;
        Storage::DataBase::Instance().Save(Storage::BlobType::ShaderProfile, "profile",
                                           std::move(fresh));
        LOG_INFO(Render, "CacheRebuild: no profile in the cache folder; this machine's saved, "
                         "stages without SPIR-V are rebuilt from the game's files");
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
    // SHADPS4_CACHE_REBUILD: the vertex-attribute conversions of vs permutations (ANNOTATE output,
    // shipped with a seed folder).
    if (CacheRebuildMode() != 0) {
        std::ifstream in{Storage::DataBase::Instance().SidecarPath("vs-formats.bin"),
                         std::ios::binary};
        u32 magic{}, version{}, count{};
        in.read(reinterpret_cast<char*>(&magic), 4);
        in.read(reinterpret_cast<char*>(&version), 4);
        in.read(reinterpret_cast<char*>(&count), 4);
        if (in && magic == VsFormatsMagic && version == VsFormatsVersion && count < (1u << 22)) {
            for (u32 e = 0; e < count && in; ++e) {
                u64 key{};
                u8 n{};
                in.read(reinterpret_cast<char*>(&key), 8);
                in.read(reinterpret_cast<char*>(&n), 1);
                std::vector<u8> codes(n);
                in.read(reinterpret_cast<char*>(codes.data()), n);
                if (in) {
                    vs_formats[key] = std::move(codes);
                }
            }
        }
        LOG_INFO(Render, "CacheRebuild: vs-formats.bin entries {}", vs_formats.size());
    }
    // SHADPS4_PRECOMPILE fix A (any build of this branch, whatever the flag: the cache may hold
    // marked metas from an earlier PRECOMPILE launch).
    precompile_retired = ReadPrecompileHashes(PrecompileRetiredFile);
    std::vector<DeferredComputeLoad> deferred_marked;
    defer_marked_compute = &deferred_marked;
    Storage::DataBase::Instance().ForEachBlob(
        Storage::BlobType::PipelineKey, [&](std::vector<u8>&& data) {
            ++num_total_pipelines;
            if (CacheRebuildMode() != 0) {
                ShowBootProgress("pipelines read", num_total_pipelines, 0);
            }

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
    defer_marked_compute = nullptr;
    // Fix A: guessed permutations after every live one (counted as loaded above).
    u32 deferred_failed = 0;
    for (auto& load : deferred_marked) {
        if (!BuildComputePipeline(load.key, load.sdata, std::move(load.meta))) {
            --num_pipelines;
            ++deferred_failed;
        }
    }
    if (!deferred_marked.empty() || !precompile_retired.empty()) {
        LOG_INFO(Render, "WarmUp: precompiled compute pipelines loaded last {} (not loaded {}), "
                         "retired programs {} (their guessed metas skipped {})",
                 deferred_marked.size(), deferred_failed, precompile_retired.size(),
                 retired_metas_skipped);
    }
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
    if (CacheRebuildMode() != 0) {
        // SHADPS4_PRECOMPILE_WAIT=1 with a rebuild: the boot also waits for the compute pipelines
        // the preload deferred to the compile pool (cold driver: ~17 k x ~50 ms), so the first
        // effects never meet them still queued.
        const char* wait_env = std::getenv("SHADPS4_PRECOMPILE_WAIT");
        if ((CacheRebuildMode() & 1) && wait_env && wait_env[0] == '1') {
            const auto t_wait = std::chrono::steady_clock::now();
            while (const size_t left =
                       PipelineCompilePool::WaitIdleFor(std::chrono::milliseconds{250})) {
                ShowBootProgress("pipelines left", left, 0);
            }
            LOG_INFO(Render, "CacheRebuild: waited {} ms for the compile pool to drain",
                     std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::steady_clock::now() - t_wait)
                         .count());
        }
        ShowBootProgress({}, 0, ~0ull);
        const auto& rs = rebuild_stats;
        for (size_t s = 0; s < 8; ++s) {
            if (rs.equal[s] + rs.differ[s] + rs.no_code[s] + rs.rebuilt[s] + rs.reused[s] +
                rs.layout_mismatch[s]) {
                LOG_INFO(Render,
                         "CacheRebuild: stage {}: verify equal={} differ={} annotated={} "
                         "annotate_failed={} | rebuilt={} reused={} layout_mismatch={} | no code "
                         "in the game index={}",
                         static_cast<Shader::Stage>(s), rs.equal[s], rs.differ[s],
                         rs.annotated[s], rs.annotate_failed[s], rs.rebuilt[s], rs.reused[s],
                         rs.layout_mismatch[s], rs.no_code[s]);
            }
        }
        if (vs_formats_dirty) {
            std::ofstream out{Storage::DataBase::Instance().SidecarPath("vs-formats.bin"),
                              std::ios::binary | std::ios::trunc};
            const u32 count = static_cast<u32>(vs_formats.size());
            out.write(reinterpret_cast<const char*>(&VsFormatsMagic), 4);
            out.write(reinterpret_cast<const char*>(&VsFormatsVersion), 4);
            out.write(reinterpret_cast<const char*>(&count), 4);
            for (const auto& [key, codes] : vs_formats) {
                const u8 n = static_cast<u8>(std::min<size_t>(codes.size(), 255));
                out.write(reinterpret_cast<const char*>(&key), 8);
                out.write(reinterpret_cast<const char*>(&n), 1);
                out.write(reinterpret_cast<const char*>(codes.data()), n);
            }
            LOG_INFO(Render, "CacheRebuild: vs-formats.bin written, {} entries", count);
        }
    }
    if (num_total_pipelines > num_pipelines) {
        LOG_WARNING(Render, "{} stale pipelines were found. Consider re-generating the cache",
                    num_total_pipelines - num_pipelines);
    }

    PrecompileFromExtract();

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
