// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// SHADPS4_PRECOMPILE (default off, 2026-10-07): translate compute shaders from the game's own
// shader list (SHADPS4_PRECOMPILE_DIR = a folder of <hash:016x>.bin files plus manifest.csv, made
// from the user's game dump) instead of waiting for the game's first dispatch of each one. Runs
// inside WarmUp on the main thread, before the game starts, so nothing else touches the caches.
// Bits of the value:
//   2 VERIFY: re-translate every cached compute permutation from the list's code with the context
//     it was cached with (its RuntimeInfo and flattened user data) and compare the SPIR-V word for
//     word with the cached binary. Changes nothing.
//   1 COMPILE: every listed compute program the cache does not hold gets up to
//     SHADPS4_PRECOMPILE_GUESSES (default 4) permutations, one per context learned from cached
//     programs of the same code shape (resource list), most frequent first. Each is registered,
//     saved to the cache and its pipeline created like a preloaded one. Every guessed input is
//     part of the permutation key (RuntimeInfo now keys num_user_data and the LDS size), so a
//     wrong guess is a miss at the game's dispatch, never a wrong shader.
//   4 HOLDOUT: every cached program is treated as unseen: its guesses come from the other
//     programs' contexts only, and each guess's spec is compared with the program's real cached
//     permutations (and, on a spec match, the SPIR-V with the cached SPIR-V). Changes nothing;
//     predicts the hit rate and catches unkeyed inputs that change the shader.
// Programs that use images are skipped (their T# cannot be guessed from buffers), as are shapes
// no cached program has.
// Record: notes/precompile-build-2026-10-07.md.

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/hash.h"
#include "common/logging/log.h"
#include "common/serdes.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/frontend/decode.h"
#include "shader_recompiler/info.h"
#include "shader_recompiler/recompiler.h"
#include "shader_recompiler/runtime_info.h"
#include "video_core/cache_storage.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_pipeline_serialization.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"

namespace Vulkan {

namespace {

using Clock = std::chrono::steady_clock;
using UserData = std::array<u32, Shader::ShaderParams::NumShaderUserData>;

double MsSince(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

/// Four copies read as a valid V# at any dword offset: dst_sel xyzw, data format 14 (32_32_32_32),
/// number format 7 (float), type 0, no swizzle / add_tid, nonzero records, stride 119.
constexpr u32 FillerDword = 0x00077FACu;
constexpr u32 FlatFillerDwords = 1024;

struct ExtractEntry {
    std::filesystem::path bin;
    u32 code_length{};
};

int PrecompileMode() {
    const char* value = std::getenv("SHADPS4_PRECOMPILE");
    return value ? std::atoi(value) : 0;
}

u32 PrecompileGuesses() {
    const char* value = std::getenv("SHADPS4_PRECOMPILE_GUESSES");
    const int n = value ? std::atoi(value) : 4;
    return static_cast<u32>(std::clamp(n, 1, 7)); // Program::MaxPermutations is 8
}

/// Compute rows of manifest.csv: shader_hash (hex, as BinaryInfo::shader_hash), stage, code_length.
std::unordered_map<u64, ExtractEntry> ReadManifest(const std::filesystem::path& dir) {
    std::unordered_map<u64, ExtractEntry> entries;
    std::ifstream file{dir / "manifest.csv"};
    std::string line;
    if (!std::getline(file, line)) {
        return entries;
    }
    auto split = [](const std::string& text) {
        std::vector<std::string> fields;
        size_t start = 0;
        while (true) {
            const size_t comma = text.find(',', start);
            fields.push_back(text.substr(start, comma - start));
            if (comma == std::string::npos) {
                return fields;
            }
            start = comma + 1;
        }
    };
    const auto header = split(line);
    auto column = [&](std::string_view name) {
        for (size_t i = 0; i < header.size(); ++i) {
            if (header[i] == name) {
                return static_cast<int>(i);
            }
        }
        return -1;
    };
    const int c_hash = column("shader_hash");
    const int c_stage = column("stage");
    const int c_length = column("code_length");
    if (c_hash < 0 || c_stage < 0 || c_length < 0) {
        LOG_ERROR(Render_Vulkan, "Precompile: manifest.csv lacks shader_hash / stage / code_length");
        return entries;
    }
    const int needed = std::max({c_hash, c_stage, c_length});
    while (std::getline(file, line)) {
        const auto fields = split(line);
        if (static_cast<int>(fields.size()) <= needed || fields[c_stage] != "cs") {
            continue;
        }
        const u64 hash = std::strtoull(fields[c_hash].c_str(), nullptr, 16);
        const u32 length = static_cast<u32>(std::strtoul(fields[c_length].c_str(), nullptr, 10));
        if (length == 0 || (length & 3) != 0) {
            continue;
        }
        entries[hash] = {dir / (fields[c_hash] + ".bin"), length};
    }
    return entries;
}

std::vector<u32> ReadCode(const ExtractEntry& entry) {
    std::vector<u32> code(entry.code_length / sizeof(u32));
    std::ifstream file{entry.bin, std::ios::binary};
    file.read(reinterpret_cast<char*>(code.data()), entry.code_length);
    if (file.gcount() != static_cast<std::streamsize>(entry.code_length)) {
        code.clear();
    }
    return code;
}

/// True when the code has an image instruction (decoding only, no translation).
bool UsesImages(const std::vector<u32>& code) {
    Shader::Gcn::GcnCodeSlice slice(code.data(), code.data() + code.size());
    Shader::Gcn::GcnDecodeContext decoder;
    while (!slice.atEnd()) {
        if (decoder.decodeInstruction(slice).encoding == Shader::Gcn::InstEncoding::MIMG) {
            return true;
        }
    }
    return false;
}

bool IsGuessableBuffer(const Shader::BufferResource& buffer) {
    return buffer.buffer_type == Shader::BufferType::Guest && !buffer.inline_cbuf;
}

/// The code-derived resource list. 0 = not handled (images, samplers or fmasks).
u64 ShapeHash(const Shader::Info& info) {
    if (!info.images.empty() || !info.samplers.empty() || !info.fmasks.empty()) {
        return 0;
    }
    u64 hash = HashCombine(0x5052454350ull, info.buffers.size());
    for (const auto& buffer : info.buffers) {
        hash = HashCombine(hash, static_cast<u64>(buffer.buffer_type));
        hash = HashCombine(hash, static_cast<u64>(buffer.used_types));
        hash = HashCombine(hash, (u64(buffer.is_written) << 1) | u64(buffer.is_formatted));
        hash = HashCombine(hash, IsGuessableBuffer(buffer) ? u64(buffer.sharp_idx) : ~0ull);
    }
    return hash;
}

struct Context {
    Shader::RuntimeInfo runtime_info{};
    std::vector<std::pair<u32, std::array<u32, 4>>> sharps; // sharp_idx, V#
    u32 count{};
};

struct ShapeContexts {
    std::unordered_map<u64, Context> by_key;
    std::vector<const Context*> ranked;
};

/// The context of a cached permutation whose guest buffers are all valid, or nullopt.
std::optional<std::pair<u64, Context>> LearnContext(const Shader::Info& info,
                                                    const Shader::StageSpecialization& spec,
                                                    u32& add_tid_seen) {
    Context ctx{};
    ctx.runtime_info = spec.runtime_info;
    const auto& rt = ctx.runtime_info;
    u64 key = HashCombine(u64(rt.num_user_data), u64(rt.cs_info.shared_memory_size));
    for (const u32 v : rt.cs_info.workgroup_size) {
        key = HashCombine(key, u64(v));
    }
    for (const bool v : rt.cs_info.tgid_enable) {
        key = HashCombine(key, u64(v));
    }
    for (const auto& buffer : info.buffers) {
        if (!IsGuessableBuffer(buffer)) {
            continue;
        }
        if (buffer.sharp_idx + 4 > info.flattened_ud_buf.size()) {
            return std::nullopt;
        }
        std::array<u32, 4> sharp{};
        std::copy_n(info.flattened_ud_buf.begin() + buffer.sharp_idx, 4, sharp.begin());
        AmdGpu::Buffer vsharp{};
        std::memcpy(&vsharp, sharp.data(), sizeof(vsharp));
        if (!vsharp.Valid() || !vsharp) {
            return std::nullopt; // an empty buffer at that dispatch: learn from a full one
        }
        add_tid_seen += vsharp.add_tid_enable;
        // Base address bits (dword 0, dword 1 [11:0]) are not part of any key.
        key = HashCombine(key, u64(buffer.sharp_idx));
        key = HashCombine(key, u64(sharp[1] & ~0xfffu));
        key = HashCombine(key, u64(sharp[3]));
        ctx.sharps.emplace_back(buffer.sharp_idx, sharp);
    }
    return std::make_pair(key, std::move(ctx));
}

struct RealPerm {
    std::unique_ptr<Shader::Info> info;
    Shader::StageSpecialization spec;
    size_t perm_idx{};
    u64 shape{};
    u64 ctx_key{}; // 0 = not learned
    bool full{};   // every guest buffer non-empty at the dispatch it was made for
};

} // Anonymous namespace

void NotePrecompiledUse(const Program& program, const Shader::StageSpecialization& live, bool hit) {
    static u32 hits = 0, misses = 0;
    hit ? ++hits : ++misses;
    const auto& rt = live.runtime_info;
    u32 first_stride = 0;
    if (!live.buffers.empty()) {
        first_stride = live.buffers[0].stride;
    }
    u32 guess_stride = 0;
    if (!program.modules.empty() && !program.modules[0].spec.buffers.empty()) {
        guess_stride = program.modules[0].spec.buffers[0].stride;
    }
    LOG_INFO(Render_Vulkan,
             "PrecompiledUse: cs_{:#x} {} (hits={} misses={}) live nud={} lds={} wg={}x{}x{} "
             "stride0={} | guess0 nud={} lds={} stride0={} perms={}",
             program.info.pgm_hash, hit ? "hit" : "MISS", hits, misses, rt.num_user_data,
             rt.cs_info.shared_memory_size, rt.cs_info.workgroup_size[0],
             rt.cs_info.workgroup_size[1], rt.cs_info.workgroup_size[2], first_stride,
             program.modules.empty() ? 0u : program.modules[0].spec.runtime_info.num_user_data,
             program.modules.empty()
                 ? 0u
                 : program.modules[0].spec.runtime_info.cs_info.shared_memory_size,
             guess_stride, program.modules.size());
    if (hit || misses > 64) {
        return;
    }
    for (size_t m = 0; m < program.modules.size(); ++m) {
        const auto& guess = program.modules[m].spec;
        std::string why;
        if (!(guess.runtime_info == live.runtime_info)) {
            why = "runtime_info";
        } else if (guess.buffers.size() != live.buffers.size()) {
            why = fmt::format("buffer count {} vs {}", guess.buffers.size(), live.buffers.size());
        } else {
            for (size_t i = 0; i < live.buffers.size(); ++i) {
                if (!live.bitset[i] || guess.buffers[i] == live.buffers[i]) {
                    continue;
                }
                const auto& g = guess.buffers[i];
                const auto& l = live.buffers[i];
                why = fmt::format("buffer {}: stride {} vs {}, formatted {} vs {}, fmt {}/{} vs {}/{}, "
                                  "swizzle {} vs {}",
                                  i, u32(g.stride), u32(l.stride), u32(g.is_formatted),
                                  u32(l.is_formatted), u32(g.data_format), u32(g.num_format),
                                  u32(l.data_format), u32(l.num_format), u32(g.swizzle_enable),
                                  u32(l.swizzle_enable));
                break;
            }
            if (why.empty()) {
                why = "other (images / samplers / fmasks / start bindings)";
            }
        }
        LOG_INFO(Render_Vulkan, "PrecompiledUse:   cs_{:#x} guess {} differs: {}",
                 program.info.pgm_hash, m, why);
    }
}

void PipelineCache::PrecompileFromExtract() {
    const int mode = PrecompileMode();
    if (mode == 0) {
        return;
    }
    const char* dir_env = std::getenv("SHADPS4_PRECOMPILE_DIR");
    if (!dir_env || !*dir_env) {
        LOG_WARNING(Render_Vulkan, "Precompile: SHADPS4_PRECOMPILE_DIR is not set");
        return;
    }
    const bool verify = (mode & 2) != 0;
    const bool compile = (mode & 1) != 0;
    const bool holdout = (mode & 4) != 0;
    const bool learn = compile || holdout;
    const auto t_start = Clock::now();
    const auto manifest = ReadManifest(dir_env);
    LOG_INFO(Render_Vulkan, "Precompile: mode {} dir {} compute shaders in the list {} ({:.0f} ms)",
             mode, dir_env, manifest.size(), MsSince(t_start));
    if (manifest.empty()) {
        return;
    }

    // One pass over the cached metas: verify, and learn the contexts per shape.
    std::unordered_set<u64> cached_programs;
    std::unordered_map<u64, ShapeContexts> shapes;
    std::unordered_map<u64, std::vector<RealPerm>> reals; // HOLDOUT only
    u32 metas = 0, unloadable = 0, compute = 0, no_code = 0, no_spv = 0, matched = 0, differed = 0;
    u32 learned = 0, unlearnable = 0, add_tid_seen = 0;
    double verify_ms = 0.0;
    Storage::DataBase::Instance().ForEachBlob(
        Storage::BlobType::ShaderMeta, [&](std::vector<u8>&& data) {
            ++metas;
            Serialization::Archive ar{std::move(data)};
            Shader::Info cached{};
            Shader::StageSpecialization spec{};
            spec.info = &cached;
            std::optional<Shader::Gcn::FetchShaderData> fetch;
            size_t perm_idx{};
            if (!LoadShaderMeta(ar, cached, fetch, spec, perm_idx)) {
                ++unloadable;
                return;
            }
            if (cached.stage != Shader::Stage::Compute) {
                return;
            }
            ++compute;
            cached_programs.insert(cached.pgm_hash);
            u64 shape = 0, ctx_key = 0;
            if (learn) {
                shape = ShapeHash(cached);
                auto ctx = shape ? LearnContext(cached, spec, add_tid_seen) : std::nullopt;
                if (ctx) {
                    ctx_key = ctx->first;
                    auto& entry = shapes[shape].by_key[ctx->first];
                    if (entry.count == 0) {
                        entry = std::move(ctx->second);
                    }
                    ++entry.count;
                    ++learned;
                } else {
                    ++unlearnable;
                }
            }
            if (holdout && shape != 0) {
                auto& real = reals[cached.pgm_hash].emplace_back();
                real.info = std::make_unique<Shader::Info>(cached);
                real.spec = spec;
                real.spec.info = real.info.get();
                real.perm_idx = perm_idx;
                real.shape = shape;
                real.ctx_key = ctx_key;
                real.full = ctx_key != 0; // LearnContext accepts only all-non-empty buffers
            }
            if (!verify) {
                return;
            }
            const auto it = manifest.find(cached.pgm_hash);
            if (it == manifest.end()) {
                ++no_code;
                return;
            }
            std::vector<u32> cached_spv;
            Storage::DataBase::Instance().Load(
                Storage::BlobType::ShaderBinary,
                fmt::format("{:#018x}_{}", cached.pgm_hash, perm_idx), cached_spv);
            const auto code = ReadCode(it->second);
            if (cached_spv.empty() || code.empty()) {
                ++no_spv;
                return;
            }

            const auto t_translate = Clock::now();
            UserData user_data{};
            std::copy_n(cached.flattened_ud_buf.begin(),
                        std::min(user_data.size(), cached.flattened_ud_buf.size()),
                        user_data.begin());
            const Shader::ShaderParams params{
                .user_data = user_data,
                .code = code,
                .hash = cached.pgm_hash,
            };
            Shader::Info info{Shader::Stage::Compute, Shader::LogicalStage::Compute, params};
            // As GetProgram: a permutation's DMA use is pinned to its program's.
            info.pinned_uses_dma = cached.uses_dma;
            auto runtime_info = spec.runtime_info;
            Shader::Backend::Bindings binding = spec.start;
            Shader::g_synthetic_flat = &cached.flattened_ud_buf;
            const auto program = Shader::TranslateProgram(code, pools, info, runtime_info, profile);
            const auto spv =
                Shader::Backend::SPIRV::EmitSPIRV(profile, runtime_info, program, binding);
            Shader::g_synthetic_flat = nullptr;
            verify_ms += MsSince(t_translate);

            if (spv == cached_spv) {
                ++matched;
                return;
            }
            ++differed;
            if (differed <= 12) {
                size_t first = 0;
                while (first < spv.size() && first < cached_spv.size() &&
                       spv[first] == cached_spv[first]) {
                    ++first;
                }
                LOG_WARNING(Render_Vulkan,
                            "Precompile verify: cs_{:#x} perm {} differs: {} vs cached {} words, "
                            "first difference at word {}",
                            cached.pgm_hash, perm_idx, spv.size(), cached_spv.size(), first);
            }
        });
    if (verify) {
        LOG_INFO(Render_Vulkan,
                 "Precompile verify: metas={} unloadable={} compute={} no_code={} no_spv={} "
                 "matched={} differed={} translate_ms={:.0f} total_ms={:.0f}",
                 metas, unloadable, compute, no_code, no_spv, matched, differed, verify_ms,
                 MsSince(t_start));
    }
    if (!learn) {
        return;
    }
    for (auto& [shape, entry] : shapes) {
        for (const auto& [key, ctx] : entry.by_key) {
            entry.ranked.push_back(&ctx);
        }
        std::ranges::stable_sort(entry.ranked, [](const Context* a, const Context* b) {
            return a->count > b->count;
        });
    }
    LOG_INFO(Render_Vulkan,
             "Precompile learn: compute metas={} learned={} unlearnable={} shapes={} "
             "add_tid_sharps={} programs_cached={}",
             compute, learned, unlearnable, shapes.size(), add_tid_seen, cached_programs.size());

    const u32 guesses = PrecompileGuesses();
    const std::vector<u32> filler(FlatFillerDwords, FillerDword);
    // Pass 1, the shape: every user SGPR from user data, filler sharps everywhere.
    auto probe_shape = [&](const std::vector<u32>& code, u64 hash, size_t& flat_size) {
        UserData user_data{};
        user_data.fill(FillerDword);
        const Shader::ShaderParams params{.user_data = user_data, .code = code, .hash = hash};
        Shader::Info probe{Shader::Stage::Compute, Shader::LogicalStage::Compute, params};
        Shader::RuntimeInfo probe_rt{};
        probe_rt.Initialize(Shader::Stage::Compute);
        probe_rt.num_user_data = Shader::ShaderParams::NumShaderUserData;
        probe_rt.cs_info.workgroup_size = {64, 1, 1};
        probe_rt.cs_info.tgid_enable = {true, false, false};
        probe_rt.cs_info.shared_memory_size = static_cast<u32>(32_KB);
        Shader::g_synthetic_flat = &filler;
        static_cast<void>(Shader::TranslateProgram(code, pools, probe, probe_rt, profile));
        Shader::g_synthetic_flat = nullptr;
        flat_size = probe.flattened_ud_buf.size();
        return ShapeHash(probe);
    };
    auto make_flat = [&](const Context& ctx, size_t flat_size) {
        std::vector<u32> flat(std::max<size_t>(FlatFillerDwords, flat_size), FillerDword);
        for (const auto& [idx, sharp] : ctx.sharps) {
            if (idx + 4 <= flat.size()) {
                std::copy(sharp.begin(), sharp.end(), flat.begin() + idx);
            }
        }
        return flat;
    };

    if (holdout) {
        u32 tested = 0, image_users = 0, probe_mismatch = 0, no_context = 0, missed = 0;
        u32 spv_equal = 0, spv_diff = 0, spv_equal_partial = 0, spv_diff_partial = 0;
        std::array<u32, 8> hit_at{};
        const auto t_holdout = Clock::now();
        for (auto& [hash, perms] : reals) {
            std::ranges::stable_partition(perms, [](const RealPerm& r) { return r.full; });
            const auto it_code = manifest.find(hash);
            if (it_code == manifest.end()) {
                continue;
            }
            const auto code = ReadCode(it_code->second);
            if (code.empty()) {
                continue;
            }
            if (UsesImages(code)) {
                ++image_users;
                continue;
            }
            size_t flat_size = 0;
            const u64 shape = probe_shape(code, hash, flat_size);
            if (shape != perms.front().shape) {
                ++probe_mismatch;
                continue;
            }
            const auto it_shape = shapes.find(shape);
            if (it_shape == shapes.end()) {
                ++no_context;
                continue;
            }
            // The program's own contributions leave the counts: rank what the others taught.
            std::unordered_map<u64, u32> own;
            for (const auto& real : perms) {
                if (real.ctx_key) {
                    ++own[real.ctx_key];
                }
            }
            std::vector<std::pair<u32, const Context*>> ranked;
            for (const auto& [key, ctx] : it_shape->second.by_key) {
                const auto it_own = own.find(key);
                const u32 own_count = it_own == own.end() ? 0u : it_own->second;
                if (ctx.count > own_count) {
                    ranked.emplace_back(ctx.count - own_count, &ctx);
                }
            }
            if (ranked.empty()) {
                ++no_context;
                continue;
            }
            std::ranges::stable_sort(ranked, [](const auto& a, const auto& b) {
                return a.first > b.first;
            });
            ++tested;
            bool hit = false;
            for (u32 rank = 0; rank < guesses && rank < ranked.size() && !hit; ++rank) {
                const Context& ctx = *ranked[rank].second;
                const auto flat = make_flat(ctx, flat_size);
                UserData ud{};
                std::copy_n(flat.begin(), ud.size(), ud.begin());
                const Shader::ShaderParams params{.user_data = ud, .code = code, .hash = hash};
                Shader::Info guess{Shader::Stage::Compute, Shader::LogicalStage::Compute, params};
                auto runtime_info = ctx.runtime_info;
                Shader::Backend::Bindings binding{};
                Shader::g_synthetic_flat = &flat;
                const auto ir =
                    Shader::TranslateProgram(code, pools, guess, runtime_info, profile);
                const auto spv =
                    Shader::Backend::SPIRV::EmitSPIRV(profile, runtime_info, ir, binding);
                Shader::g_synthetic_flat = nullptr;
                const Shader::StageSpecialization spec{guess, runtime_info, profile, {}};
                for (const auto& real : perms) {
                    if (!(spec == real.spec)) {
                        continue;
                    }
                    hit = true;
                    ++hit_at[rank];
                    std::vector<u32> cached_spv;
                    Storage::DataBase::Instance().Load(
                        Storage::BlobType::ShaderBinary,
                        fmt::format("{:#018x}_{}", hash, real.perm_idx), cached_spv);
                    if (!real.full) {
                        ++(spv == cached_spv ? spv_equal_partial : spv_diff_partial);
                    } else if (spv == cached_spv) {
                        ++spv_equal;
                    } else {
                        ++spv_diff;
                        if (spv_diff <= 12) {
                            size_t first = 0;
                            while (first < spv.size() && first < cached_spv.size() &&
                                   spv[first] == cached_spv[first]) {
                                ++first;
                            }
                            LOG_WARNING(Render_Vulkan,
                                        "Precompile holdout: cs_{:#x} rank {} matches perm {} by "
                                        "spec but its SPIR-V differs ({} vs {} words, first at {})",
                                        hash, rank, real.perm_idx, spv.size(), cached_spv.size(),
                                        first);
                        }
                    }
                    break;
                }
            }
            missed += !hit;
        }
        LOG_INFO(Render_Vulkan,
                 "Precompile holdout: programs={} tested={} image_users={} probe_mismatch={} "
                 "no_context={} hit@1={} @2={} @3={} @4={} @5+={} missed={} full: spv_equal={} "
                 "spv_diff={} | empty-buffer perms: spv_equal={} spv_diff={} guesses={} ms={:.0f}",
                 reals.size(), tested, image_users, probe_mismatch, no_context, hit_at[0],
                 hit_at[1], hit_at[2], hit_at[3], hit_at[4] + hit_at[5] + hit_at[6] + hit_at[7],
                 missed, spv_equal, spv_diff, spv_equal_partial, spv_diff_partial, guesses,
                 MsSince(t_holdout));
    }
    if (!compile) {
        return;
    }

    // COMPILE: every listed program the cache does not hold.
    u32 listed_new = 0, image_users = 0, no_shape = 0, drifted = 0, programs_built = 0;
    u32 permutations = 0, duplicate_specs = 0, layout_mismatch = 0;
    double translate_ms = 0.0;
    const auto t_compile = Clock::now();
    for (const auto& [hash, entry] : manifest) {
        if (cached_programs.contains(hash) || program_cache.contains(hash)) {
            continue;
        }
        ++listed_new;
        const auto code = ReadCode(entry);
        if (code.empty()) {
            continue;
        }
        if (UsesImages(code)) {
            ++image_users;
            continue;
        }
        const auto t_translate = Clock::now();

        size_t flat_size = 0;
        const u64 shape = probe_shape(code, hash, flat_size);
        const auto it_shape = shape ? shapes.find(shape) : shapes.end();
        if (it_shape == shapes.end()) {
            ++no_shape;
            translate_ms += MsSince(t_translate);
            continue;
        }

        // Pass 2, one permutation per learned context, most frequent first.
        std::unique_ptr<Program> program;
        std::vector<std::vector<u32>> flats;
        std::vector<UserData> uds;
        const size_t max_attempts = size_t{guesses} * 2;
        flats.reserve(max_attempts);
        uds.reserve(max_attempts);
        for (const Context* ctx : it_shape->second.ranked) {
            if ((program && program->modules.size() >= guesses) || uds.size() >= max_attempts) {
                break;
            }
            auto& flat = flats.emplace_back(make_flat(*ctx, flat_size));
            auto& ud = uds.emplace_back();
            std::copy_n(flat.begin(), ud.size(), ud.begin());
            const Shader::ShaderParams params{.user_data = ud, .code = code, .hash = hash};
            auto runtime_info = ctx->runtime_info;
            Shader::g_synthetic_flat = &flat;

            vk::ShaderModule module{};
            Shader::StageSpecialization spec{};
            if (!program) {
                program = std::make_unique<Program>(Shader::Stage::Compute,
                                                    Shader::LogicalStage::Compute, params);
                Shader::Backend::Bindings binding{};
                const auto ir =
                    Shader::TranslateProgram(code, pools, program->info, runtime_info, profile);
                auto spv = Shader::Backend::SPIRV::EmitSPIRV(profile, runtime_info, ir, binding);
                if (ShapeHash(program->info) != shape) {
                    ++drifted;
                    program.reset();
                    break;
                }
                module = CompileSPV(spv, instance.GetDevice());
                RegisterShaderBinary(std::move(spv), hash, 0);
                spec = Shader::StageSpecialization(program->info, runtime_info, profile, {});
            } else {
                // As GetProgram for a new permutation: the spec from the program's Info refreshed
                // with this context, the module from a fresh Info pinned to the program's DMA use.
                program->info.user_data = ud;
                program->info.RefreshFlatBuf();
                spec = Shader::StageSpecialization(program->info, runtime_info, profile, {});
                if (std::ranges::find(program->modules, spec, &Program::Module::spec) !=
                    program->modules.end()) {
                    ++duplicate_specs;
                    continue;
                }
                Shader::Info new_info{Shader::Stage::Compute, Shader::LogicalStage::Compute,
                                      params};
                new_info.pinned_uses_dma = program->info.uses_dma;
                Shader::Backend::Bindings binding{};
                const auto ir =
                    Shader::TranslateProgram(code, pools, new_info, runtime_info, profile);
                auto spv = Shader::Backend::SPIRV::EmitSPIRV(profile, runtime_info, ir, binding);
                if (!SameBindingLayout(new_info, program->info)) {
                    ++layout_mismatch;
                    continue;
                }
                module = CompileSPV(spv, instance.GetDevice());
                RegisterShaderBinary(std::move(spv), hash, program->modules.size());
            }
            const size_t perm_idx = program->modules.size();
            const u64 perm_hash = HashCombine(hash, perm_idx);
            RegisterShaderMeta(program->info, std::nullopt, spec, perm_hash, perm_idx, true);
            program->AddPermut(module, std::move(spec));

            const ComputePipelineKey key{perm_hash};
            ComputePipeline::SerializationSupport sdata{};
            const auto [it_pl, is_new] = compute_pipelines.try_emplace(key);
            if (is_new) {
                it_pl.value() = std::make_unique<ComputePipeline>(
                    instance, scheduler, desc_heap, profile, *pipeline_cache, key, program->info,
                    module, sdata, true);
                RegisterPipelineData(key, sdata);
            }
            ++permutations;
        }
        Shader::g_synthetic_flat = nullptr;
        translate_ms += MsSince(t_translate);
        if (!program || program->modules.empty()) {
            continue;
        }
        // Guest user data and the code address are refreshed by GetProgram before any use.
        program->info.user_data = {};
        program->precompiled = true;
        program_cache.try_emplace(hash, std::move(program));
        ++programs_built;
    }
    LOG_INFO(Render_Vulkan,
             "Precompile compile: listed_new={} image_users={} no_shape={} drifted={} "
             "programs={} permutations={} duplicate_specs={} layout_mismatch={} guesses={} "
             "translate_ms={:.0f} compile_phase_ms={:.0f} total_ms={:.0f}",
             listed_new, image_users, no_shape, drifted, programs_built, permutations,
             duplicate_specs, layout_mismatch, guesses, translate_ms, MsSince(t_compile),
             MsSince(t_start));
}

} // namespace Vulkan
