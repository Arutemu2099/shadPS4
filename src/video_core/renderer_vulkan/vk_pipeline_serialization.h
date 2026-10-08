// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "shader_recompiler/frontend/fetch_shader.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"

namespace Vulkan {

// The on-disk cache is named per shader binary version (cache_storage.cpp Open).
u32 GetShaderBinaryVersion();

void RegisterPipelineData(const ComputePipelineKey& key,
                          ComputePipeline::SerializationSupport& sdata);
void RegisterPipelineData(const GraphicsPipelineKey& key, u64 hash,
                          GraphicsPipeline::SerializationSupport& sdata);
void RegisterShaderMeta(const Shader::Info& info,
                        const std::optional<Shader::Gcn::FetchShaderData>& fetch_shader_data,
                        const Shader::StageSpecialization& spec, size_t perm_hash, size_t perm_idx,
                        bool precompiled = false);
void RegisterShaderBinary(std::vector<u32>&& spv, u64 pgm_hash, size_t perm_idx);
// SHADPS4_PRECOMPILE: the game's first dispatch of a pre-compiled program, matched or not.
void NotePrecompiledUse(const Program& program, const Shader::StageSpecialization& live, bool hit);
// `precompiled` (optional): set when the meta carries the SHADPS4_PRECOMPILE marker.
bool LoadShaderMeta(Serialization::Archive& ar, Shader::Info& info,
                    std::optional<Shader::Gcn::FetchShaderData>& fetch_shader_data,
                    Shader::StageSpecialization& spec, size_t& perm_idx,
                    bool* precompiled = nullptr);
// True when two translations of one program give the same descriptor bindings (buffer kinds and
// order, images, samplers, fmasks). Permutations share one Info, so they must agree on these.
bool SameBindingLayout(const Shader::Info& a, const Shader::Info& b);

} // namespace Vulkan
