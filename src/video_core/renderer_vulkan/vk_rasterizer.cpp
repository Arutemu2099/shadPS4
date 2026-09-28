// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <immintrin.h>
#include <set>
#include <chrono>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/thread.h"
#include "common/debug.h"
#include "common/buffer_sample_probe.h"
#include "common/draw_sample_probe.h"
#include "common/fault_monitor.h"
#include "common/lean.h"
#include "common/rdtsc.h"
#include "common/scope_exit.h"
#include "core/debug_state.h"
#include "core/emulator_settings.h"
#include "core/memory.h"
#include "shader_recompiler/runtime_info.h"
#include "video_core/amdgpu/split_hazard.h"
#include "video_core/buffer_cache/dirty_epoch.h"
#include "video_core/host_shaders/neo_id_derive_comp.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_hle.h"
#include "video_core/texture_cache/image_view.h"
#include "video_core/texture_cache/texture_cache.h"

#ifdef MemoryBarrier
#undef MemoryBarrier
#endif

namespace AmdGpu {
extern thread_local u64 g_back_idle_cyc; // liverpool.cpp, SHADPS4_DRAW_COST
}

namespace Vulkan {

namespace {
// Per-draw / per-binding environment flags, read once at startup. A function-local
// `static const` flag costs a guard check on every call (under the MSVC ABI: load the guard,
// _tls_index, gs:[0x58], the thread's TLS block, compare with _Init_thread_epoch); guard lines
// drew ~2.5% of the command processor's stack samples (a19, 2026-09-25). Values unchanged.
bool EnvIs1(const char* name) {
    const char* value = std::getenv(name);
    return value && value[0] == '1';
}
bool EnvExactly1(const char* name) {
    const char* value = std::getenv(name);
    return value && std::string_view{value} == "1";
}
const bool g_prepared_targets = EnvIs1("SHADPS4_PREPARED_TARGETS");
const bool g_verify_targets = EnvIs1("SHADPS4_PREPARED_TARGETS_VERIFY");
const bool g_raw_write_invalidate = EnvIs1("SHADPS4_RAW_WRITE_INVALIDATE");
const bool g_cache_decoded_images = [] {
    const char* value = std::getenv("SHADPS4_DECODE_IMAGE_CACHE");
    return !value || value[0] != '0';
}();
const bool g_verify_decoded_images = EnvIs1("SHADPS4_DECODE_IMAGE_VERIFY");
const std::vector<u64> g_trace_hashes = [] {
    std::vector<u64> out;
    if (const char* value = std::getenv("SHADPS4_TRACE_SHADER")) {
        char* cursor = const_cast<char*>(value);
        while (*cursor) {
            out.push_back(std::strtoull(cursor, &cursor, 16));
            if (*cursor == ',') {
                ++cursor;
            } else {
                break;
            }
        }
    }
    return out;
}();
const s64 g_idbuf_fill = [] {
    const char* value = std::getenv("SHADPS4_NEO_IDBUF_FILL");
    return value ? static_cast<s64>(std::strtoll(value, nullptr, 0)) : s64{-1};
}();
const bool g_idbuf_derive = EnvIs1("SHADPS4_NEO_IDBUF_DERIVE");
const int g_viewport_memo = [] {
    const char* value = std::getenv("SHADPS4_VIEWPORT_MEMO");
    return value ? value[0] - '0' : 0;
}();
const bool g_hair_stencil_trace = EnvExactly1("SHADPS4_HAIR_STENCIL_TRACE");
const bool g_replace_op_reference = EnvExactly1("SHADPS4_STENCIL_REPLACE_OP");

// SHADPS4_DRAW_COST=1 (diagnostic, default off): rdtsc cycles of each DrawTail phase per graphics
// pipeline on the thread that executes draws (the CP back with the split), logged every 2^20
// draws as the top pipelines by total cost plus a depth-only/no-image class total. Answers which
// kinds of draws the CP time goes to (frame map 2026-09-27). Direct draws only (DrawTail).
const bool g_draw_cost = EnvIs1("SHADPS4_DRAW_COST");
enum DrawCostPhase : u32 {
    DcPrepare,
    DcBindBuffers,
    DcBindTextures,
    DcBindOther,
    DcBeginRendering,
    DcVertexIndex,
    DcEmitResources,
    DcRecordTail,
    DcCount
};
constexpr std::array<const char*, DcCount> kDrawCostNames{
    "prep", "buf", "tex", "bind_other", "begin", "vtx_idx", "emit", "tail"};
thread_local u64 g_dc_bind_buf = 0;
thread_local u64 g_dc_bind_tex = 0;
// Buffer bindings (BindBuffers, draws and dispatches): [0] read-only, [1] written; phases:
// 0 find pass (per stage, split evenly), 1 sync walk, 2 sync rest, 3 other ObtainBuffer work,
// 4 barrier + written-branch invalidation + descriptor info; plus counts and special bindings.
thread_local std::array<std::array<u64, 5>, 2> g_bb_cyc{};
thread_local std::array<u64, 2> g_bb_n{};
thread_local u64 g_bb_special_n = 0;
// Repeat census for read bindings (DRAW_COST): same (buffer, base, size) already bound since the
// last scheduler tick change (= submission), and whether the dirty epoch is unchanged since then.
thread_local u64 g_rep_total = 0;
thread_local u64 g_rep_hit = 0;
thread_local u64 g_rep_hit_clean = 0;
thread_local u64 g_rep_distinct = 0;
void NoteReadBindingRepeat(u64 tick, u32 buffer_id, u64 base, u64 size) {
    thread_local u64 last_tick = 0;
    thread_local std::unordered_map<u64, u64> seen; // key -> dirty epoch at first bind
    if (tick != last_tick) {
        g_rep_distinct += seen.size();
        seen.clear();
        last_tick = tick;
    }
    const u64 key = (base * 0x9E3779B97F4A7C15ULL) ^ (size << 20) ^ buffer_id;
    const u64 epoch = VideoCore::g_dirty_epoch.load(std::memory_order_relaxed);
    ++g_rep_total;
    const auto [it, inserted] = seen.try_emplace(key, epoch);
    if (!inserted) {
        ++g_rep_hit;
        if (it->second == epoch) {
            ++g_rep_hit_clean;
        } else {
            it->second = epoch;
        }
    }
}
thread_local u64 g_dc_disp_cyc = 0; // ExecuteDispatch on the back (direct + indirect)
thread_local u64 g_dc_disp_n = 0;
// Inside BindTextures (DRAW_COST only): [0] first pass (FindImage per binding), [1] FindTexture
// (view lookup, refresh/upload), [2] Transit (layout changes, may end rendering), plus the single
// slowest FindTexture+Transit image of the draw. DrawTail logs the split for draws whose texture
// phase exceeds ~50 us (first 40 such draws).
thread_local std::array<u64, 3> g_dc_tex_sub{};
struct SlowImage {
    u64 cyc{};
    u64 addr{};
    u32 w{};
    u32 h{};
    u32 layers{};
    u32 fmt{};
    bool depth{};
    bool tiled{};
    u32 layout_before{};
};
thread_local SlowImage g_dc_slow_img{};

struct DrawCostStats {
    u64 n = 0;
    std::array<u64, DcCount> cyc{};
};

bool IsDepthOnlyNoImage(const GraphicsPipeline* pipeline) {
    // mrt_mask is the shader's export mask (0x1 even for the empty depth-only shader); the
    // attachment count is what the capture shows as "no colour output".
    if (pipeline->GetGraphicsKey().num_color_attachments != 0) {
        return false;
    }
    for (const auto* stage : pipeline->GetStages()) {
        if (stage && !stage->images.empty()) {
            return false;
        }
    }
    return true;
}

void NoteDrawCost(const GraphicsPipeline* pipeline, const std::array<u64, DcCount>& cyc) {
    thread_local std::unordered_map<const GraphicsPipeline*, DrawCostStats> stats;
    thread_local u64 draws = 0;
    thread_local u64 win_tsc = 0;
    thread_local std::chrono::steady_clock::time_point win_start{};
    if (win_tsc == 0) {
        win_tsc = __rdtsc();
        win_start = std::chrono::steady_clock::now();
    }
    auto& s = stats[pipeline];
    ++s.n;
    for (u32 i = 0; i < DcCount; ++i) {
        s.cyc[i] += cyc[i];
    }
    if ((++draws & 0xFFFFF) != 0) {
        return;
    }
    const u64 now_tsc = __rdtsc();
    const double wall_ns = double(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                      std::chrono::steady_clock::now() - win_start)
                                      .count());
    const double ns_per_cyc = wall_ns / double(now_tsc - win_tsc);
    std::vector<std::pair<const GraphicsPipeline*, const DrawCostStats*>> order;
    order.reserve(stats.size());
    u64 all_cyc = 0;
    u64 depth_cyc = 0;
    u64 depth_n = 0;
    std::array<u64, DcCount> all_phase{};
    std::array<u64, DcCount> depth_phase{};
    for (const auto& [p, st] : stats) {
        order.emplace_back(p, &st);
        const bool depth_only = IsDepthOnlyNoImage(p);
        for (u32 i = 0; i < DcCount; ++i) {
            all_cyc += st.cyc[i];
            all_phase[i] += st.cyc[i];
            if (depth_only) {
                depth_cyc += st.cyc[i];
                depth_phase[i] += st.cyc[i];
            }
        }
        if (depth_only) {
            depth_n += st.n;
        }
    }
    const auto total_of = [](const DrawCostStats& st) {
        u64 t = 0;
        for (const u64 c : st.cyc) {
            t += c;
        }
        return t;
    };
    std::sort(order.begin(), order.end(),
              [&](const auto& a, const auto& b) { return total_of(*a.second) > total_of(*b.second); });
    const auto phases = [&](const std::array<u64, DcCount>& ph, u64 n) {
        std::string out;
        for (u32 i = 0; i < DcCount; ++i) {
            out += fmt::format(" {}={:.0f}", kDrawCostNames[i],
                               n ? double(ph[i]) * ns_per_cyc / double(n) : 0.0);
        }
        return out;
    };
    LOG_INFO(Render_Vulkan,
             "DrawCost: window {:.0f} ms, {} draws, draw time {:.1f}% of wall, {:.0f} ns/draw;{}",
             wall_ns / 1e6, draws, 100.0 * double(all_cyc) * ns_per_cyc / wall_ns,
             double(all_cyc) * ns_per_cyc / double(draws), phases(all_phase, draws));
    LOG_INFO(Render_Vulkan,
             "DrawCost: back idle (WaitForWork) {:.1f}% of wall; dispatches {} = {:.1f}% of wall, "
             "{:.0f} ns each",
             100.0 * double(AmdGpu::g_back_idle_cyc) * ns_per_cyc / wall_ns, g_dc_disp_n,
             100.0 * double(g_dc_disp_cyc) * ns_per_cyc / wall_ns,
             g_dc_disp_n ? double(g_dc_disp_cyc) * ns_per_cyc / double(g_dc_disp_n) : 0.0);
    for (u32 w = 0; w < 2; ++w) {
        const double n = double(std::max<u64>(g_bb_n[w], 1));
        const auto& c = g_bb_cyc[w];
        LOG_INFO(Render_Vulkan,
                 "DrawCost: buffer bindings {}: {} ({:.0f}/draw) ns each: find={:.0f} walk={:.0f} "
                 "sync_rest={:.0f} obtain_other={:.0f} barrier_written={:.0f} total={:.0f}; "
                 "{:.1f}% of wall",
                 w ? "written" : "read", g_bb_n[w], double(g_bb_n[w]) / double(draws),
                 double(c[0]) * ns_per_cyc / n, double(c[1]) * ns_per_cyc / n,
                 double(c[2]) * ns_per_cyc / n, double(c[3]) * ns_per_cyc / n,
                 double(c[4]) * ns_per_cyc / n,
                 double(c[0] + c[1] + c[2] + c[3] + c[4]) * ns_per_cyc / n,
                 100.0 * double(c[0] + c[1] + c[2] + c[3] + c[4]) * ns_per_cyc / wall_ns);
    }
    LOG_INFO(Render_Vulkan, "DrawCost: special (non-cache) buffer bindings {}", g_bb_special_n);
    LOG_INFO(Render_Vulkan,
             "DrawCost: read-binding repeats within a submission: {} of {} ({:.1f}%), with no dirty-epoch "
             "change since the previous bind {} ({:.1f}%); distinct per submission (summed) {}",
             g_rep_hit, g_rep_total, 100.0 * double(g_rep_hit) / double(std::max<u64>(g_rep_total, 1)),
             g_rep_hit_clean, 100.0 * double(g_rep_hit_clean) / double(std::max<u64>(g_rep_total, 1)),
             g_rep_distinct);
    g_rep_total = g_rep_hit = g_rep_hit_clean = g_rep_distinct = 0;
    g_bb_cyc = {};
    g_bb_n = {};
    g_bb_special_n = 0;
    AmdGpu::g_back_idle_cyc = 0;
    g_dc_disp_cyc = 0;
    g_dc_disp_n = 0;
    LOG_INFO(Render_Vulkan,
             "DrawCost: depth-only/no-image class: {} draws ({:.1f}%), {:.1f}% of draw time, "
             "{:.0f} ns/draw;{}",
             depth_n, 100.0 * double(depth_n) / double(draws),
             all_cyc ? 100.0 * double(depth_cyc) / double(all_cyc) : 0.0,
             depth_n ? double(depth_cyc) * ns_per_cyc / double(depth_n) : 0.0,
             phases(depth_phase, depth_n));
    for (size_t k = 0; k < std::min<size_t>(order.size(), 30); ++k) {
        const auto* p = order[k].first;
        const auto& st = *order[k].second;
        u64 vs = 0;
        u64 fs = 0;
        for (const auto* stage : p->GetStages()) {
            if (!stage) {
                continue;
            }
            if (stage->l_stage == Shader::LogicalStage::Vertex) {
                vs = stage->pgm_hash;
            } else if (stage->l_stage == Shader::LogicalStage::Fragment) {
                fs = stage->pgm_hash;
            }
        }
        LOG_INFO(Render_Vulkan,
                 "DrawCost: #{} fs_{:08x} vs_{:08x} mrt={:#x} depth_only={} n={} share={:.1f}% "
                 "ns/draw={:.0f};{}",
                 k, fs, vs, p->GetGraphicsKey().mrt_mask, IsDepthOnlyNoImage(p) ? 1 : 0, st.n,
                 all_cyc ? 100.0 * double(total_of(st)) / double(all_cyc) : 0.0,
                 double(total_of(st)) * ns_per_cyc / double(st.n), phases(st.cyc, st.n));
    }
    stats.clear();
    draws = 0;
    win_tsc = __rdtsc();
    win_start = std::chrono::steady_clock::now();
}

// SHADPS4_DRAW_COST=1, dispatches: per compute pipeline, total cycles and the BindResources
// buffer/texture split (from BindResources' own counters); top 20 every 2^18 dispatches.
struct DispatchCostStats {
    u64 n = 0;
    u64 total = 0;
    u64 buf = 0;
    u64 tex = 0;
};
void NoteDispatchCost(const ComputePipeline* pipeline, u64 total, u64 buf, u64 tex) {
    thread_local std::unordered_map<const ComputePipeline*, DispatchCostStats> stats;
    thread_local u64 dispatches = 0;
    thread_local u64 win_tsc = 0;
    thread_local std::chrono::steady_clock::time_point win_start{};
    if (win_tsc == 0) {
        win_tsc = __rdtsc();
        win_start = std::chrono::steady_clock::now();
    }
    auto& st = stats[pipeline];
    ++st.n;
    st.total += total;
    st.buf += buf;
    st.tex += tex;
    if ((++dispatches & 0x3FFFF) != 0) {
        return;
    }
    const double wall_ns = double(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                      std::chrono::steady_clock::now() - win_start)
                                      .count());
    const double ns_per_cyc = wall_ns / double(__rdtsc() - win_tsc);
    std::vector<std::pair<const ComputePipeline*, const DispatchCostStats*>> order;
    u64 all = 0;
    for (const auto& [p, v] : stats) {
        order.emplace_back(p, &v);
        all += v.total;
    }
    std::sort(order.begin(), order.end(),
              [](const auto& a, const auto& b) { return a.second->total > b.second->total; });
    LOG_INFO(Render_Vulkan, "DispatchCost: window {:.0f} ms, {} dispatches, {:.1f}% of wall, {:.0f} ns each",
             wall_ns / 1e6, dispatches, 100.0 * double(all) * ns_per_cyc / wall_ns,
             double(all) * ns_per_cyc / double(dispatches));
    for (size_t k = 0; k < std::min<size_t>(order.size(), 20); ++k) {
        const auto* p = order[k].first;
        const auto& v = *order[k].second;
        u64 cs = 0;
        for (const auto* stage : p->GetStages()) {
            if (stage) {
                cs = stage->pgm_hash;
            }
        }
        LOG_INFO(Render_Vulkan,
                 "DispatchCost: #{} cs_{:08x} n={} share={:.1f}% ns/disp={:.0f} buf={:.0f} tex={:.0f} "
                 "other={:.0f} buffers={} images={}",
                 k, cs, v.n, all ? 100.0 * double(v.total) / double(all) : 0.0,
                 double(v.total) * ns_per_cyc / double(v.n), double(v.buf) * ns_per_cyc / double(v.n),
                 double(v.tex) * ns_per_cyc / double(v.n),
                 double(v.total - v.buf - v.tex) * ns_per_cyc / double(v.n),
                 p->GetStages()[0] ? p->GetStages()[0]->buffers.size() : 0,
                 p->GetStages()[0] ? p->GetStages()[0]->images.size() : 0);
    }
    stats.clear();
    dispatches = 0;
    win_tsc = __rdtsc();
    win_start = std::chrono::steady_clock::now();
}

// SHADPS4_MERGE_PROBE=1 (diagnostic, default off): for each direct draw that uses the same
// graphics pipeline as the previous direct draw, record which inputs differ from that draw.
// Bits: user-data regs, flattened user data, buffer V# base, buffer V# size/stride, image T#,
// vertex V# base, index base, index offset (VGT index_offset / vertex offset), index count,
// instance count, render-target change (a BeginRendering in between). Per pipeline: draws,
// same-pipeline repeats, top diff masks, OR of differing user-data regs and buffer slots.
// Logged every 2^20 draws. Answers whether consecutive draws can merge into one call.
const bool g_merge_probe = EnvIs1("SHADPS4_MERGE_PROBE");
enum MergeDiffBit : u32 {
    MdUserData = 1u << 0,
    MdFlatUd = 1u << 1,
    MdBufBase = 1u << 2,
    MdBufShape = 1u << 3,
    MdImage = 1u << 4,
    MdVertexBase = 1u << 5,
    MdIndexBase = 1u << 6,
    MdIndexOffset = 1u << 7,
    MdCount = 1u << 8,
    MdInstances = 1u << 9,
    MdTarget = 1u << 10,
};
constexpr std::array<const char*, 11> kMergeDiffNames{
    "ud", "flat", "bufbase", "bufshape", "img", "vbase", "ibase", "ioff", "count", "inst", "rt"};

struct MergeSig {
    const GraphicsPipeline* pipeline{};
    std::vector<u32> ud;
    std::vector<u32> flat;
    std::vector<u64> buf_base;
    std::vector<u64> buf_shape;
    std::vector<u64> img;
    std::vector<u64> vbase;
    u64 ibase{};
    u32 ioff{};
    u32 count{};
    u32 inst{};
    u64 rt_epoch{};
};

struct MergeStats {
    u64 n{};
    u64 repeats{};
    u32 ud_regs{};  // OR of differing user-data register indices (VS in low 16, FS in high 16)
    u64 buf_slots{}; // OR of differing buffer slots (VS slots 0-31, FS slots 32-63)
    std::unordered_map<u32, u64> masks;
};

u64 g_merge_rt_epoch = 0; // bumped by BeginRendering changes (draw thread only)

void NoteMergeProbe(const GraphicsPipeline* pipeline, std::span<const Shader::BoundStage> stages,
                    const AmdGpu::Regs& regs, bool is_indexed, u32 index_offset) {
    thread_local MergeSig prev;
    thread_local MergeSig cur;
    thread_local std::unordered_map<const GraphicsPipeline*, MergeStats> stats;
    thread_local u64 draws = 0;
    cur.pipeline = pipeline;
    cur.ud.clear();
    cur.flat.clear();
    cur.buf_base.clear();
    cur.buf_shape.clear();
    cur.img.clear();
    cur.vbase.clear();
    u32 ud_diff = 0;
    u64 slot_diff = 0;
    const bool comparable = prev.pipeline == pipeline;
    u32 stage_no = 0;
    for (const auto& bound : stages) {
        const auto* info = bound.info;
        if (!info) {
            continue;
        }
        const bool is_vs = info->l_stage == Shader::LogicalStage::Vertex;
        const u32 ud_begin = static_cast<u32>(cur.ud.size());
        for (u32 i = 0; i < bound.user_data.size() && i < 16; ++i) {
            cur.ud.push_back(bound.user_data[i]);
        }
        cur.flat.insert(cur.flat.end(), bound.flattened_ud_buf.begin(),
                        bound.flattened_ud_buf.end());
        const u32 buf_begin = static_cast<u32>(cur.buf_base.size());
        for (const auto& desc : info->buffers) {
            const auto vsharp = desc.GetSharp(bound);
            cur.buf_base.push_back(vsharp.base_address);
            cur.buf_shape.push_back(u64(vsharp.GetSize()) << 32 | vsharp.GetStride());
        }
        for (const auto& desc : info->images) {
            const auto tsharp = desc.GetSharp(bound);
            u64 h = 0;
            const auto* words = reinterpret_cast<const u64*>(&tsharp);
            for (size_t w = 0; w < sizeof(tsharp) / sizeof(u64); ++w) {
                h = h * 0x9E3779B97F4A7C15ULL ^ words[w];
            }
            cur.img.push_back(h);
        }
        if (is_vs) {
            if (const auto* fetch = pipeline->GetFetchShader()) {
                for (const auto& attrib : fetch->attributes) {
                    cur.vbase.push_back(attrib.GetSharp(bound).base_address);
                }
            }
        }
        if (comparable) {
            for (u32 i = ud_begin; i < cur.ud.size() && i < prev.ud.size(); ++i) {
                if (cur.ud[i] != prev.ud[i]) {
                    ud_diff |= 1u << ((i - ud_begin) + (is_vs ? 0 : 16));
                }
            }
            for (u32 i = buf_begin; i < cur.buf_base.size() && i < prev.buf_base.size(); ++i) {
                if (cur.buf_base[i] != prev.buf_base[i] || cur.buf_shape[i] != prev.buf_shape[i]) {
                    slot_diff |= u64{1} << (((i - buf_begin) & 31) + (is_vs ? 0 : 32));
                }
            }
        }
        ++stage_no;
    }
    cur.ibase = is_indexed ? regs.index_base_address.Address() : 0;
    cur.ioff = index_offset;
    cur.count = regs.num_indices;
    cur.inst = regs.num_instances.NumInstances();
    cur.rt_epoch = g_merge_rt_epoch;
    auto& st = stats[pipeline];
    ++st.n;
    if (comparable) {
        u32 mask = 0;
        mask |= cur.ud != prev.ud ? MdUserData : 0;
        mask |= cur.flat != prev.flat ? MdFlatUd : 0;
        mask |= cur.buf_base != prev.buf_base ? MdBufBase : 0;
        mask |= cur.buf_shape != prev.buf_shape ? MdBufShape : 0;
        mask |= cur.img != prev.img ? MdImage : 0;
        mask |= cur.vbase != prev.vbase ? MdVertexBase : 0;
        mask |= cur.ibase != prev.ibase ? MdIndexBase : 0;
        mask |= cur.ioff != prev.ioff ? MdIndexOffset : 0;
        mask |= cur.count != prev.count ? MdCount : 0;
        mask |= cur.inst != prev.inst ? MdInstances : 0;
        mask |= cur.rt_epoch != prev.rt_epoch ? MdTarget : 0;
        ++st.repeats;
        ++st.masks[mask];
        st.ud_regs |= ud_diff;
        st.buf_slots |= slot_diff;
    }
    std::swap(prev, cur);
    if ((++draws & 0xFFFFF) != 0) {
        return;
    }
    std::vector<std::pair<const GraphicsPipeline*, const MergeStats*>> order;
    for (const auto& [p, s] : stats) {
        order.emplace_back(p, &s);
    }
    std::sort(order.begin(), order.end(),
              [](const auto& a, const auto& b) { return a.second->n > b.second->n; });
    const auto mask_name = [](u32 mask) {
        if (mask == 0) {
            return std::string("none");
        }
        std::string out;
        for (u32 b = 0; b < kMergeDiffNames.size(); ++b) {
            if (mask & (1u << b)) {
                out += out.empty() ? "" : "+";
                out += kMergeDiffNames[b];
            }
        }
        return out;
    };
    for (size_t k = 0; k < std::min<size_t>(order.size(), 16); ++k) {
        const auto* p = order[k].first;
        const auto& s = *order[k].second;
        u64 vs = 0;
        u64 fs = 0;
        for (const auto* stage : p->GetStages()) {
            if (stage && stage->l_stage == Shader::LogicalStage::Vertex) {
                vs = stage->pgm_hash;
            } else if (stage && stage->l_stage == Shader::LogicalStage::Fragment) {
                fs = stage->pgm_hash;
            }
        }
        std::vector<std::pair<u32, u64>> masks(s.masks.begin(), s.masks.end());
        std::sort(masks.begin(), masks.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        std::string top;
        for (size_t m = 0; m < std::min<size_t>(masks.size(), 5); ++m) {
            top += fmt::format(" {}:{:.1f}%", mask_name(masks[m].first),
                               100.0 * double(masks[m].second) / double(s.repeats));
        }
        LOG_INFO(Render_Vulkan,
                 "MergeProbe: #{} fs_{:08x} vs_{:08x} n={} repeats={} ({:.1f}%) ud_regs={:#010x} "
                 "buf_slots={:#018x} masks:{}",
                 k, fs, vs, s.n, s.repeats, 100.0 * double(s.repeats) / double(s.n), s.ud_regs,
                 s.buf_slots, top);
    }
    stats.clear();
    draws = 0;
}
} // namespace

// SHADPS4_FLUSH_EVERY_OPS=N: submit once N draws + dispatches have been recorded since the last
// submission of any kind (a fault flush resets the count too). A readback copy runs after
// everything submitted before it; measured 2026-09-27 in the fire scene's slow mode, readbacks
// queued behind ~5 ms batches. SHADPS4_FLUSH_EVERY_DRAWS counts draws only and is not reset by
// other flushes. 0 (default) = off. Runs on the command processor thread that records.
static void MaybeFlushEveryOps(Scheduler& scheduler) {
    static const u32 flush_every_ops = [] {
        const char* value = std::getenv("SHADPS4_FLUSH_EVERY_OPS");
        return value ? static_cast<u32>(std::strtoul(value, nullptr, 10)) : 0u;
    }();
    if (flush_every_ops != 0 && scheduler.BatchOps() >= flush_every_ops) {
        scheduler.Flush();
    }
}

// Shape-only diagnostic; enable for one run, then compile it out for FPS comparisons.
#ifndef SHADPS4_BINDING_PROBE
#define SHADPS4_BINDING_PROBE 0
#endif
#if SHADPS4_BINDING_PROBE
static void ProbeBindingChanges(const Pipeline* pipeline,
                                std::span<const Shader::BoundStage> bound_stages) {
    if (pipeline->IsCompute()) {
        return;
    }
    struct PreviousStage {
        const Shader::Info* info{};
        size_t num_buffers{}, num_images{};
        std::array<AmdGpu::Buffer, Shader::NUM_BUFFERS> buffers{};
        std::array<AmdGpu::Image, Shader::NUM_IMAGES> images{};
    };
    static std::array<PreviousStage, Shader::MaxStageTypes> previous{};
    static u64 draws{}, stages{}, same_program{}, buffers{}, same_buffers{}, special{}, images{},
        same_images{}, written{};
    ++draws;
    for (const auto& bound : bound_stages) {
        const auto* stage = bound.info;
        if (!stage) {
            continue;
        }
        auto& prev = previous[static_cast<u32>(stage->l_stage)];
        ++stages;
        same_program += prev.info == stage;
        size_t i = 0;
        for (const auto& desc : stage->buffers) {
            const auto sharp = desc.GetSharp(bound);
            if (desc.IsSpecial()) {
                ++special;
            } else {
                ++buffers;
                written += desc.is_written;
                same_buffers += prev.info && i < prev.num_buffers &&
                                std::memcmp(&sharp, &prev.buffers[i], sizeof(sharp)) == 0;
            }
            prev.buffers[i++] = sharp;
        }
        prev.num_buffers = i;
        i = 0;
        for (const auto& desc : stage->images) {
            const auto sharp = desc.GetSharp(bound);
            ++images;
            same_images += prev.info && i < prev.num_images &&
                           std::memcmp(&sharp, &prev.images[i], sizeof(sharp)) == 0;
            prev.images[i++] = sharp;
        }
        prev.num_images = i;
        prev.info = stage;
    }
    if (draws >= 200000) {
        LOG_INFO(Render_Vulkan,
                 "BindingChanges: draws={} buffers/draw={:.2f} raw_same_buffers={:.1f}% "
                 "special/draw={:.2f} written_buffers={:.1f}% images/draw={:.2f} "
                 "raw_same_images={:.1f}% same_stage_program={:.1f}%",
                 draws, double(buffers) / draws, buffers ? 100.0 * same_buffers / buffers : 0.0,
                 double(special) / draws, buffers ? 100.0 * written / buffers : 0.0,
                 double(images) / draws, images ? 100.0 * same_images / images : 0.0,
                 stages ? 100.0 * same_program / stages : 0.0);
        draws = stages = same_program = buffers = same_buffers = special = images = same_images =
            written = 0;
    }
    // Equal descriptors do not prove equal contents or cache state. This probe
    // never skips synchronization, resource lookup, barriers, or descriptor writes.
}
#endif

static Shader::PushData MakeUserData(const AmdGpu::Regs& regs) {
    // TODO(roamic): Add support for multiple viewports and geometry shaders when ViewportIndex
    // is encountered and implemented in the recompiler.
    Shader::PushData push_data{};
    push_data.xoffset = regs.viewport_control.xoffset_enable ? regs.viewports[0].xoffset : 0.f;
    push_data.xscale = regs.viewport_control.xscale_enable ? regs.viewports[0].xscale : 1.f;
    push_data.yoffset = regs.viewport_control.yoffset_enable ? regs.viewports[0].yoffset : 0.f;
    push_data.yscale = regs.viewport_control.yscale_enable ? regs.viewports[0].yscale : 1.f;
    return push_data;
}

Rasterizer::Rasterizer(const Instance& instance_, Scheduler& scheduler_,
                       AmdGpu::Liverpool* liverpool_)
    : instance{instance_}, scheduler{scheduler_},
      native_bindings{NativeBindings::Create(instance_, scheduler_)}, page_manager{this},
      buffer_cache{instance, scheduler, liverpool_, texture_cache, page_manager},
      texture_cache{instance, scheduler, liverpool_, buffer_cache, page_manager},
      liverpool{liverpool_}, memory{Core::Memory::Instance()},
      pipeline_cache{instance, scheduler, liverpool} {
    if (!EmulatorSettings.IsNullGPU()) {
        liverpool->BindRasterizer(this);
    }
    memory->SetRasterizer(this);
    build_set_writes = !native_bindings || native_bindings->IsVerifying();
    const char* memo_stats = std::getenv("SHADPS4_BIND_MEMO_STATS");
    bind_memo_stats = memo_stats && memo_stats[0] == '1';
    if (bind_memo_stats) {
        LOG_INFO(Render_Vulkan, "BindMemoStats: enabled (counters only, behaviour unchanged)");
    }
    if (const char* value = std::getenv("SHADPS4_TEX_BIND_MEMO");
        value && (value[0] == '1' || value[0] == '2')) {
        if (bind_memo_stats) {
            LOG_WARNING(Render_Vulkan, "TexBindMemo: off, SHADPS4_BIND_MEMO_STATS takes precedence");
        } else if (native_bindings) {
            LOG_WARNING(Render_Vulkan, "TexBindMemo: off, not supported with native bindings");
        } else {
            tex_bind_memo = static_cast<u8>(value[0] - '0');
            LOG_INFO(Render_Vulkan, "TexBindMemo: enabled, mode={}",
                     tex_bind_memo == 2 ? "verify (full path + compare)" : "memo");
        }
    }
}

Rasterizer::~Rasterizer() {
    scheduler.DrainRecording();
}

void Rasterizer::CpSync() {
    scheduler.EndRendering();
    auto command_access = scheduler.AcquireCommandBuffer();
    auto cmdbuf = command_access.Handle();

    const vk::MemoryBarrier ib_barrier{
        .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
        .dstAccessMask = vk::AccessFlagBits::eIndirectCommandRead,
    };
    cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                           vk::PipelineStageFlagBits::eDrawIndirect,
                           vk::DependencyFlagBits::eByRegion, ib_barrier, {}, {});
}

Rasterizer::DrawClass Rasterizer::ClassifyDraw(const AmdGpu::Regs& regs) {
    if (regs.color_control.mode == AmdGpu::ColorControl::OperationMode::EliminateFastClear) {
        return DrawClass::EliminateFastClear;
    }
    if (regs.color_control.mode == AmdGpu::ColorControl::OperationMode::FmaskDecompress) {
        return DrawClass::FmaskDecompress;
    }
    if (regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Resolve) {
        return DrawClass::Resolve;
    }
    if (regs.primitive_type == AmdGpu::PrimitiveType::None) {
        return DrawClass::PrimitiveNone;
    }

    const bool cb_disabled =
        regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Disable;
    const auto depth_copy =
        regs.depth_render_override.force_z_dirty && regs.depth_render_override.force_z_valid &&
        regs.depth_buffer.DepthValid() && regs.depth_buffer.DepthWriteValid() &&
        regs.depth_buffer.DepthAddress() != regs.depth_buffer.DepthWriteAddress();
    const auto stencil_copy =
        regs.depth_render_override.force_stencil_dirty &&
        regs.depth_render_override.force_stencil_valid && regs.depth_buffer.StencilValid() &&
        regs.depth_buffer.StencilWriteValid() &&
        regs.depth_buffer.StencilAddress() != regs.depth_buffer.StencilWriteAddress();
    if (cb_disabled && (depth_copy || stencil_copy)) {
        return depth_copy && stencil_copy ? DrawClass::DepthStencilCopy
               : depth_copy               ? DrawClass::DepthCopy
                                          : DrawClass::StencilCopy;
    }

    return DrawClass::Draw;
}

void Rasterizer::ExecuteFilteredDraw(DrawClass draw_class) {
    switch (draw_class) {
    case DrawClass::Draw:
        break;
    case DrawClass::EliminateFastClear:
        // Clears the render target if FCE is launched before any draws
        EliminateFastClear();
        break;
    case DrawClass::FmaskDecompress:
        // TODO: check for a valid MRT1 to promote the draw to the resolve pass.
        LOG_TRACE(Render_Vulkan, "FMask decompression pass skipped");
        ScopedMarkerInsert("FmaskDecompress");
        break;
    case DrawClass::Resolve:
        LOG_TRACE(Render_Vulkan, "Resolve pass");
        Resolve();
        break;
    case DrawClass::PrimitiveNone:
        LOG_TRACE(Render_Vulkan, "Primitive type 'None' skipped");
        ScopedMarkerInsert("PrimitiveTypeNone");
        break;
    case DrawClass::DepthCopy:
    case DrawClass::StencilCopy:
    case DrawClass::DepthStencilCopy:
        // Games may disable color buffer and enable force depth/stencil dirty and valid to
        // do a copy from one depth-stencil surface to another, without a pixel shader.
        // We need to detect this case and perform the copy, otherwise it will have no effect.
        LOG_TRACE(Render_Vulkan, "Performing depth-stencil override copy");
        DepthStencilCopy(draw_class != DrawClass::StencilCopy,
                         draw_class != DrawClass::DepthCopy);
        break;
    }
}

bool Rasterizer::FilterDraw() {
    // The classifier reads the front's registers (it runs where the pipeline lookup runs); the
    // actions read the back's through BackRegs().
    const DrawClass draw_class = ClassifyDraw(liverpool->regs);
    if (draw_class == DrawClass::Draw) {
        return true;
    }
    ExecuteFilteredDraw(draw_class);
    return false;
}

static bool SamePreparedTarget(const VideoCore::TextureCache::ImageDesc& a,
                               const VideoCore::TextureCache::ImageDesc& b) {
    const auto& x = a.info;
    const auto& y = b.info;
    if (a.type != b.type || a.view_info != b.view_info || x.pixel_format != y.pixel_format ||
        x.type != y.type || x.resources != y.resources || x.size != y.size ||
        x.num_bits != y.num_bits || x.num_samples != y.num_samples || x.pitch != y.pitch ||
        x.tile_mode != y.tile_mode || x.array_mode != y.array_mode ||
        x.guest_address != y.guest_address || x.guest_size != y.guest_size ||
        x.bank_swizzle != y.bank_swizzle || x.alt_tile != y.alt_tile ||
        x.stencil_addr != y.stencil_addr || x.stencil_size != y.stencil_size ||
        x.meta_info.cmask_addr != y.meta_info.cmask_addr ||
        x.meta_info.fmask_addr != y.meta_info.fmask_addr ||
        x.meta_info.htile_addr != y.meta_info.htile_addr ||
        x.meta_info.htile_clear_mask != y.meta_info.htile_clear_mask ||
        x.props.is_volume != y.props.is_volume || x.props.is_tiled != y.props.is_tiled ||
        x.props.is_pow2 != y.props.is_pow2 || x.props.is_block != y.props.is_block ||
        x.props.is_depth != y.props.is_depth || x.props.has_stencil != y.props.has_stencil) {
        return false;
    }
    for (u32 mip = 0; mip < x.resources.levels; ++mip) {
        const auto& xm = x.mips_layout[mip];
        const auto& ym = y.mips_layout[mip];
        if (xm.size != ym.size || xm.pitch != ym.pitch || xm.height != ym.height ||
            xm.offset != ym.offset) {
            return false;
        }
    }
    return true;
}

void Rasterizer::PrepareRenderState(const GraphicsPipeline* pipeline) {
    const bool prepared_targets = g_prepared_targets;
    const bool verify_targets = g_verify_targets;
    const auto verify = [](bool hit, const auto& prepared, const auto& reference) {
        ASSERT(SamePreparedTarget(prepared, reference));
        static thread_local u64 calls{}, hits{};
        hits += hit;
        if (++calls % 500000 == 0) {
            LOG_INFO(Render_Vulkan, "PreparedTargetVerify: calls={} hits={}", calls, hits);
        }
    };
    // Prefetch render targets to handle overlaps with bound textures (e.g. mipgen)
    const auto& key = pipeline->GetGraphicsKey();
    const auto& regs = liverpool->BackRegs();
    if (regs.color_control.degamma_enable) {
        LOG_WARNING(Render_Vulkan, "Color buffers require gamma correction");
    }

    const bool skip_cb_binding =
        regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Disable;
    for (s32 cb = 0; cb < std::bit_width(key.mrt_mask); ++cb) {
        auto& [image_id, desc] = cb_descs[cb];
        const auto& col_buf = regs.color_buffers[cb];
        const u32 target_mask = regs.color_target_mask.GetMask(cb);
        if (skip_cb_binding || !col_buf || !target_mask || (key.mrt_mask & (1 << cb)) == 0) {
            image_id = {};
            continue;
        }
        const auto& hint = liverpool->BackCbExtent(cb);
        if (prepared_targets) {
            auto& entry = prepared_color_targets[cb];
            const bool hit = entry.valid &&
                             std::memcmp(&entry.buffer, &col_buf, sizeof(col_buf)) == 0 &&
                             std::memcmp(&entry.extent, &hint, sizeof(hint)) == 0;
            if (!hit) {
                entry.buffer = col_buf;
                entry.extent = hint;
                std::construct_at(&entry.value, col_buf, hint);
                entry.valid = true;
            }
            if (verify_targets) {
                verify(hit, entry.value, VideoCore::TextureCache::ImageDesc{col_buf, hint});
            }
            desc = entry.value;
        } else {
            std::construct_at(&desc, col_buf, hint);
        }
        image_id = bound_images.emplace_back(texture_cache.FindImage(desc));
        auto& image = texture_cache.GetImage(image_id);
        image.binding.is_target = 1u;
    }

    if ((regs.depth_control.depth_enable && regs.depth_buffer.DepthValid()) ||
        (regs.depth_control.stencil_enable && regs.depth_buffer.StencilValid())) {
        const auto htile_address = regs.depth_htile_data_base.GetAddress();
        const auto& hint = liverpool->BackDbExtent();
        auto& [image_id, desc] = db_desc;
        if (prepared_targets) {
            auto& entry = prepared_depth_target;
            const bool hit =
                entry.valid && entry.htile_address == htile_address &&
                std::memcmp(&entry.buffer, &regs.depth_buffer, sizeof(entry.buffer)) == 0 &&
                std::memcmp(&entry.view, &regs.depth_view, sizeof(entry.view)) == 0 &&
                std::memcmp(&entry.control, &regs.depth_control, sizeof(entry.control)) == 0 &&
                std::memcmp(&entry.extent, &hint, sizeof(hint)) == 0;
            if (!hit) {
                entry.buffer = regs.depth_buffer;
                entry.view = regs.depth_view;
                entry.control = regs.depth_control;
                entry.extent = hint;
                entry.htile_address = htile_address;
                std::construct_at(&entry.value, regs.depth_buffer, regs.depth_view,
                                  regs.depth_control, htile_address, hint);
                entry.valid = true;
            }
            if (verify_targets) {
                verify(hit, entry.value,
                       VideoCore::TextureCache::ImageDesc{regs.depth_buffer, regs.depth_view,
                                                          regs.depth_control, htile_address, hint});
            }
            desc = entry.value;
        } else {
            std::construct_at(&desc, regs.depth_buffer, regs.depth_view, regs.depth_control,
                              htile_address, hint);
        }
        image_id = bound_images.emplace_back(texture_cache.FindImage(desc));
        auto& image = texture_cache.GetImage(image_id);
        image.binding.is_target = 1u;
    } else {
        db_desc.first = {};
    }
}

static std::pair<u32, u32> GetDrawOffsets(
    const AmdGpu::Regs& regs, const Shader::BoundStage& info,
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

void Rasterizer::EliminateFastClear() {
    auto& col_buf = liverpool->BackRegs().color_buffers[0];
    if (!col_buf || !col_buf.info.fast_clear) {
        return;
    }
    VideoCore::TextureCache::ImageDesc desc(col_buf, liverpool->BackCbExtent(0));
    const auto image_id = texture_cache.FindImage(desc);
    const auto& image_view = texture_cache.FindRenderTarget(image_id, desc);
    if (!texture_cache.IsMetaCleared(col_buf.CmaskAddress(), col_buf.view.slice_start)) {
        return;
    }
    for (u32 slice = col_buf.view.slice_start; slice <= col_buf.view.slice_max; ++slice) {
        texture_cache.TouchMeta(col_buf.CmaskAddress(), slice, false);
    }
    auto& image = texture_cache.GetImage(image_id);
    const auto clear_value = LiverpoolToVK::ColorBufferClearValue(col_buf);

    ScopeMarkerBegin(fmt::format("EliminateFastClear:MRT={:#x}:M={:#x}", col_buf.Address(),
                                 col_buf.CmaskAddress()));
    image.Clear(clear_value, desc.view_info.range);
    ScopeMarkerEnd();
}

// SHADPS4_REG_SCAN=<hex>[,<hex>...] (diagnostic): every 512th draw, scan all registers for the
// given 32-bit values (e.g. a surface address >> 8) and log each register index that holds one,
// once. Finds which register points at a surface the emulator does not model (Pro ID buffer).
void Rasterizer::ScanRegistersForValue() {
    static const std::vector<u32> targets = [] {
        std::vector<u32> out;
        const char* value = std::getenv("SHADPS4_REG_SCAN");
        if (!value) {
            return out;
        }
        std::string_view list{value};
        while (!list.empty()) {
            const size_t comma = list.find(',');
            const auto item = list.substr(0, comma);
            out.push_back(u32(std::strtoull(std::string(item).c_str(), nullptr, 16)));
            list = comma == std::string_view::npos ? std::string_view{} : list.substr(comma + 1);
        }
        return out;
    }();
    if (targets.empty()) {
        return;
    }
    static u64 draws = 0;
    if ((draws++ & 511) != 0) {
        return;
    }
    static std::unordered_set<u64> seen;
    const auto& regs = liverpool->BackRegs().reg_array;
    for (u32 i = 0; i < regs.size(); ++i) {
        for (const u32 target : targets) {
            if (regs[i] == target && seen.insert((u64(i) << 32) | target).second) {
                LOG_INFO(Render_Vulkan, "REG_SCAN: value {:#x} in register word {:#x} (draw {})",
                         target, i, draws);
            }
        }
    }
}

struct Rasterizer::NeoIdDerive {
    vk::UniqueDescriptorSetLayout set_layout;
    vk::UniquePipelineLayout layout;
    vk::UniquePipeline pipeline;
    vk::Image depth_image{};
    vk::ImageView depth_view{};
    vk::Image id_image{};
    vk::ImageView id_view{};
    // Views are never destroyed while possibly in flight: superseded ones are kept (rare, small).
    std::vector<vk::UniqueImageView> views;
    u64 runs{};
};

void Rasterizer::DeriveNeoIdBuffer(VideoCore::Image& id_image) {
    static const s64 parity_override = [] {
        const char* value = std::getenv("SHADPS4_NEO_IDBUF_PARITY");
        return value ? static_cast<s64>(std::strtoll(value, nullptr, 0)) : s64{-1};
    }();
    static const u32 bucket_mask = [] {
        const char* value = std::getenv("SHADPS4_NEO_IDBUF_MASK");
        return value ? static_cast<u32>(std::strtoull(value, nullptr, 0)) : 0xFFFFF000u;
    }();
    if (!neo_depth_addr) {
        return;
    }
    const auto depth_id = texture_cache.FindDepthImageBySize(neo_depth_addr, neo_depth_size);
    if (!depth_id) {
        return;
    }
    auto& depth = texture_cache.GetImage(depth_id);
    if (depth.info.num_samples != 2 || depth.info.size.width != id_image.info.size.width ||
        depth.info.size.height != id_image.info.size.height) {
        return;
    }
    const auto device = instance.GetDevice();
    if (!neo_id_derive) {
        neo_id_derive = std::make_shared<NeoIdDerive>();
        auto& d = *neo_id_derive;
        const std::array<vk::DescriptorSetLayoutBinding, 2> bindings = {{
            {.binding = 0,
             .descriptorType = vk::DescriptorType::eSampledImage,
             .descriptorCount = 1,
             .stageFlags = vk::ShaderStageFlagBits::eCompute},
            {.binding = 1,
             .descriptorType = vk::DescriptorType::eStorageImage,
             .descriptorCount = 1,
             .stageFlags = vk::ShaderStageFlagBits::eCompute},
        }};
        const vk::DescriptorSetLayoutCreateInfo set_ci{
            .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
            .bindingCount = 2,
            .pBindings = bindings.data(),
        };
        d.set_layout = Check(device.createDescriptorSetLayoutUnique(set_ci));
        const vk::PushConstantRange push_range{
            .stageFlags = vk::ShaderStageFlagBits::eCompute, .offset = 0, .size = 16};
        const vk::PipelineLayoutCreateInfo layout_ci{
            .setLayoutCount = 1,
            .pSetLayouts = &*d.set_layout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &push_range,
        };
        d.layout = Check(device.createPipelineLayoutUnique(layout_ci));
        const auto module =
            Compile(HostShaders::NEO_ID_DERIVE_COMP, vk::ShaderStageFlagBits::eCompute, device);
        const vk::ComputePipelineCreateInfo pipeline_ci{
            .stage = {.stage = vk::ShaderStageFlagBits::eCompute, .module = module, .pName = "main"},
            .layout = *d.layout,
        };
        d.pipeline = Check(device.createComputePipelineUnique({}, pipeline_ci));
        device.destroyShaderModule(module);
        LOG_INFO(Render_Vulkan, "NeoIdDerive: pipeline created");
    }
    auto& d = *neo_id_derive;
    if (d.depth_image != depth.GetImage()) {
        d.depth_image = depth.GetImage();
        const vk::ImageViewCreateInfo view_ci{
            .image = d.depth_image,
            .viewType = vk::ImageViewType::e2D,
            .format = depth.info.pixel_format,
            .subresourceRange = {vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1},
        };
        d.views.push_back(Check(device.createImageViewUnique(view_ci)));
        d.depth_view = *d.views.back();
    }
    if (d.id_image != id_image.GetImage()) {
        d.id_image = id_image.GetImage();
        const vk::ImageViewCreateInfo view_ci{
            .image = d.id_image,
            .viewType = vk::ImageViewType::e2D,
            .format = vk::Format::eR32Uint,
            .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
        };
        d.views.push_back(Check(device.createImageViewUnique(view_ci)));
        d.id_view = *d.views.back();
    }
    const u32 s0_x = neo_depth_x0y0 & 0xF;
    const u32 parity = parity_override >= 0 ? static_cast<u32>(parity_override & 1)
                                            : ((s0_x != 0 && s0_x < 8) ? 1u : 0u);
    const std::array<u32, 4> push{id_image.info.size.width, id_image.info.size.height, parity,
                                  bucket_mask};

    scheduler.EndRendering();
    depth.Transit(vk::ImageLayout::eGeneral, vk::AccessFlagBits2::eShaderRead, {});
    id_image.Transit(vk::ImageLayout::eGeneral, vk::AccessFlagBits2::eShaderWrite, {});
    const vk::DescriptorImageInfo depth_info{.imageView = d.depth_view,
                                             .imageLayout = vk::ImageLayout::eGeneral};
    const vk::DescriptorImageInfo id_info{.imageView = d.id_view,
                                          .imageLayout = vk::ImageLayout::eGeneral};
    const std::array<vk::WriteDescriptorSet, 2> writes = {{
        {.dstBinding = 0,
         .descriptorCount = 1,
         .descriptorType = vk::DescriptorType::eSampledImage,
         .pImageInfo = &depth_info},
        {.dstBinding = 1,
         .descriptorCount = 1,
         .descriptorType = vk::DescriptorType::eStorageImage,
         .pImageInfo = &id_info},
    }};
    {
        auto command_access = scheduler.AcquireCommandBuffer();
        const auto cmdbuf = command_access.Handle();
        cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, *d.pipeline);
        cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, *d.layout, 0, writes);
        cmdbuf.pushConstants(*d.layout, vk::ShaderStageFlagBits::eCompute, 0,
                             static_cast<u32>(sizeof(push)), push.data());
        cmdbuf.dispatch((push[0] + 7) / 8, (push[1] + 7) / 8, 1);
    }
    ++d.runs;
    if (d.runs <= 20 || (d.runs & 1023) == 0) {
        LOG_INFO(Render_Vulkan,
                 "NeoIdDerive: depth {:#x} {}x{} 2s -> id {:#x} parity {} (x0y0 {:#x}) mask {:#x} "
                 "run {}",
                 neo_depth_addr, depth.info.size.width, depth.info.size.height,
                 id_image.info.guest_address, parity, neo_depth_x0y0, bucket_mask, d.runs);
    }
}

void Rasterizer::Draw(bool is_indexed, u32 index_offset) {
    RENDERER_TRACE;
    Common::SampleCommandProcessorCore();
    ScanRegistersForValue();

    auto sample = Common::DrawSampleProbe::Begin();
    const auto t_entry = Common::FaultMonitor::Now();
    scheduler.PopPendingOperations();
    if (native_bindings) {
        // The only point where the heap backend may flush or wait; nothing is assembled yet.
        native_bindings->BeginDraw();
    }

    if (!FilterDraw()) {
        return;
    }

    AmdGpu::SplitHazard::NoteDraw(liverpool->regs.ps_program, liverpool->regs.vs_program);
    if (SkipEmptyDraw(liverpool->regs)) {
        return;
    }

    // Scoped timers: the stack scanner reports inclusive frequencies with false positives and
    // cannot apportion cost here. See Common::FaultMonitor for why these exist.
    const auto t_pipeline = Common::FaultMonitor::Now();
    sample.BeginPhase(Common::DrawSampleProbe::Phase::Pipeline);
    const GraphicsPipeline* pipeline = pipeline_cache.GetGraphicsPipeline();
    sample.EndPhase(Common::DrawSampleProbe::Phase::Pipeline);
    const auto pipeline_ns = Common::FaultMonitor::SinceNs(t_pipeline);
    if (!pipeline) {
        return;
    }
    BindStageData(pipeline);
    DrawTail(pipeline, is_indexed, index_offset, sample, t_entry, t_pipeline, pipeline_ns);
}

bool Rasterizer::SkipEmptyDraw(const AmdGpu::Regs& r) {
    // SHADPS4_EMPTY_DRAWS=1 counts direct draws with zero indices or zero instances (they render
    // nothing but still bind every resource); =2 also skips them. Clears and resolves are
    // handled in FilterDraw before this point. Register reads and statics only (front-safe).
    static const int empty_draws = [] {
        const char* value = std::getenv("SHADPS4_EMPTY_DRAWS");
        return value ? value[0] - '0' : 0;
    }();
    if (empty_draws > 0) {
        static u64 draws = 0, empty = 0;
        const bool is_empty = r.num_indices == 0 || r.num_instances.NumInstances() == 0;
        ++draws;
        empty += is_empty;
        if ((draws & 0xfffff) == 0) {
            LOG_INFO(Render_Vulkan, "EMPTY_DRAWS: {} of {} direct draws ({:.2f}%)", empty, draws,
                     100.0 * empty / draws);
        }
        if (is_empty && empty_draws == 2) {
            return true;
        }
    }
    return false;
}

void Rasterizer::PrepareDrawFront(SplitDraw& draw) {
    // CP split, front thread: the part of Draw/DrawIndirect before PopPendingOperations that
    // reads only the front's registers and the pipeline cache, in the same order as the inline
    // path. Only order change: the lookup now runs before the back's PopPendingOperations and
    // native_bindings->BeginDraw instead of after them.
    // SPLIT-TODO: confirm the pipeline constructor does not depend on BeginDraw (heap backend).
    draw.draw_class = ClassifyDraw(liverpool->regs);
    if (draw.draw_class != DrawClass::Draw) {
        return;
    }
    if (!draw.args.is_indirect) {
        AmdGpu::SplitHazard::NoteDraw(liverpool->regs.ps_program, liverpool->regs.vs_program);
        draw.skip = SkipEmptyDraw(liverpool->regs);
        if (draw.skip) {
            return;
        }
    }
    draw.pipeline = pipeline_cache.GetGraphicsPipeline();
}

const ComputePipeline* Rasterizer::PrepareDispatchFront() {
    // CP split, front thread: reads the front's current queue cs registers (GetCsRegs).
    return pipeline_cache.GetComputePipeline();
}

const GraphicsPipeline* Rasterizer::RelookupDrawFront() {
    // Liverpool::ExecuteDrawRecord (VERIFY): the front is parked, so its registers and the
    // pipeline cache are stable and published to this thread (see DrawRelookup).
    return pipeline_cache.GetGraphicsPipeline();
}

void Rasterizer::ExecuteDraw(const SplitDraw& draw,
                             const std::array<Shader::BoundStage, Shader::MaxStageTypes>& stages) {
    // CP split, back thread: Draw/DrawIndirect with the front's classification, lookup and
    // per-stage data. Registers come from the back's mirror through BackRegs().
    // With run-ahead the front runs concurrently: nothing here may read front state (regs,
    // Info::user_data/flattened_ud_buf/pgm_base, pipeline-cache tables). Indirect args are read
    // here, in stream order, so they need no front drain.
    // SPLIT-TODO(run-ahead): FaultMonitor's single-writer counters must keep one writer each
    // (checked: draw/dispatch counts are back-only, WaitRegMem/CpWait/CpActive front-only; the
    // other Count* sites are unchecked).
    RENDERER_TRACE;
    // Same prologue as the inline Draw/DrawIndirect, minus what the front already did.
    const auto begin = [&] {
        scheduler.PopPendingOperations();
        if (native_bindings) {
            native_bindings->BeginDraw();
        }
        if (draw.draw_class != DrawClass::Draw) {
            ExecuteFilteredDraw(draw.draw_class);
            return false;
        }
        if (draw.skip || !draw.pipeline) {
            return false;
        }
        bound_stages = stages;
        if (VideoCore::BufferCache::CensusEnabled()) [[unlikely]] {
            u64 id = 0;
            for (const auto* stage : draw.pipeline->GetStages()) {
                if (stage && stage->l_stage == Shader::LogicalStage::Fragment) {
                    id = stage->pgm_hash;
                }
            }
            buffer_cache.SetWriterTag(1, id);
        }
        return true;
    };
    if (draw.args.is_indirect) {
        if (begin()) {
            DrawIndirectTail(draw.pipeline, draw.args.is_indexed, draw.args.arg_address,
                             draw.args.offset, draw.args.stride, draw.args.max_count,
                             draw.args.count_address);
        }
        return;
    }
    Common::SampleCommandProcessorCore();
    ScanRegistersForValue();
    auto sample = Common::DrawSampleProbe::Begin();
    const auto t_entry = Common::FaultMonitor::Now();
    if (begin()) {
        // The lookup ran on the front: the Pipeline phase and pipeline_ns are 0 here.
        DrawTail(draw.pipeline, draw.args.is_indexed, draw.args.index_offset, sample, t_entry,
                 t_entry, 0);
    }
}

void Rasterizer::ExecuteDispatch(const ComputePipeline* pipeline,
                                 const std::array<Shader::BoundStage, Shader::MaxStageTypes>& stages,
                                 bool is_indirect, VAddr address, u32 offset, u32 size) {
    // CP split, back thread: DispatchDirect/DispatchIndirect after the front's lookup. The caller
    // points Liverpool::BackCsRegs() at the record's ComputeProgram snapshot.
    RENDERER_TRACE;
    const auto t_dispatch = Common::FaultMonitor::Now();
    const u64 dc_disp_t0 = g_draw_cost ? __rdtsc() : 0;
    if (g_draw_cost) {
        g_dc_bind_buf = 0; // an HLE dispatch skips BindResources, which resets these
        g_dc_bind_tex = 0;
    }
    SCOPE_EXIT {
        Common::FaultMonitor::CountDispatch(Common::FaultMonitor::SinceNs(t_dispatch));
        if (g_draw_cost) {
            const u64 cyc = __rdtsc() - dc_disp_t0;
            g_dc_disp_cyc += cyc;
            ++g_dc_disp_n;
            if (pipeline) {
                NoteDispatchCost(pipeline, cyc, g_dc_bind_buf, g_dc_bind_tex);
            }
        }
    };
    const bool asc_stream = scheduler.HasAscStream();
    if (asc_stream) {
        // Pending operations, HLE shaders, stencil writebacks and buffer creation/joins (pass 1)
        // record on the main stream; BindBuffers moves an eligible dispatch to the ASC stream.
        scheduler.SwitchStream(0);
    }
    scheduler.PopPendingOperations();
    if (native_bindings) {
        native_bindings->BeginDraw();
    }
    if (!pipeline) {
        return;
    }
    bound_stages = stages;
    dispatch_target = asc_stream && dispatch_queue != 0 && AscCandidate(pipeline) ? 1u : 0u;
    if (VideoCore::BufferCache::CensusEnabled()) [[unlikely]] {
        u64 id = 0;
        for (const auto* stage : pipeline->GetStages()) {
            if (stage) {
                id = stage->pgm_hash;
            }
        }
        buffer_cache.SetWriterTag(dispatch_queue == 0 ? 2 : 3, id);
    }
    dispatch_on_asc = false;
    // An indirect dispatch reads its argument buffer: BindBuffers checks it with the bindings.
    buffer_cache.SetAscIndirectArgs(is_indirect ? address + offset : 0, is_indirect ? size : 0);
    const u32 ts = dispatch_target;
    // SHADPS4_GUEST_DISPATCH_ORDER: continue the previous dispatch's run only if the guest issued
    // no sync packet or draw in between (same nonzero guest run) and nothing was recorded since
    // that dispatch ended (no copy, upload, readback, pending operation or borrowed buffer).
    // With the ASC stream the state is per command stream (barriers order one stream only).
    if (dispatch_run != 0 && scheduler.IsAsyncRecording()) {
        const bool same_guest = dispatch_run == (asc_stream ? s_prev_run[ts] : prev_dispatch_run);
        const u64 seq = asc_stream ? scheduler.RecordSeqOf(ts) : scheduler.RecordSeq();
        // A flush between the two dispatches may run pending operations without moving the
        // record sequence (Opus review 2026-09-27, ASC_YIELD): require the same batch too.
        const bool cont =
            asc_stream ? same_guest && seq == s_prev_end_seq[ts] &&
                             StreamBatchId(ts) == s_prev_end_batch[ts]
                       : same_guest && seq == prev_dispatch_end_seq &&
                             scheduler.CurrentTick() == prev_dispatch_end_tick;
        const u64 prev_eff = asc_stream ? s_prev_effective[ts] : prev_effective_run;
        cur_dispatch_run = cont ? prev_eff : ++effective_run_counter;
        // Counters (log only): continued runs vs breaks by cause, and recorded commands between.
        static u64 n_cont = 0, n_guest = 0, n_seq = 0, seq_gap = 0, n_all = 0;
        ++n_all;
        n_cont += cont;
        n_guest += !same_guest;
        if (same_guest && !cont) {
            ++n_seq;
            seq_gap += seq - (asc_stream ? s_prev_end_seq[ts] : prev_dispatch_end_seq);
        }
        if ((n_all & 0xffff) == 0) {
            LOG_INFO(Render_Vulkan,
                     "GuestDispatchOrder: dispatches={} continued={} new_guest_run={} "
                     "recorded_between={} (avg {:.1f} records)",
                     n_all, n_cont, n_guest, n_seq, n_seq ? double(seq_gap) / double(n_seq) : 0.0);
        }
    } else {
        cur_dispatch_run = 0;
    }
    const u64 tick_before_tail = scheduler.CurrentTick();
    if (is_indirect) {
        DispatchIndirectTail(pipeline, address, offset, size);
    } else {
        DispatchDirectTail(pipeline);
    }
    // A flush inside the tail runs pending operations after the dispatch; the next dispatch's
    // sequence check could not see what they recorded (Opus review 2026-09-27): end the run.
    const bool flushed = scheduler.CurrentTick() != tick_before_tail;
    if (asc_stream) {
        // dispatch_on_asc: BindBuffers really moved it (HLE / skipped dispatches never get there).
        const u32 s = dispatch_on_asc ? 1u : 0u;
        s_prev_run[s] = (cur_dispatch_run != 0 && !flushed) ? dispatch_run : 0;
        s_prev_effective[s] = cur_dispatch_run;
        s_prev_end_seq[s] = scheduler.RecordSeqOf(s);
        s_prev_end_batch[s] = StreamBatchId(s);
        if (s == 1) {
            // Stamp the dispatch's ASC accesses with the segment that holds its tail.
            const u64 asc_value = scheduler.OpenAscValue();
            ASSERT(asc_value != 0 || buffer_cache.pending_asc.empty());
            buffer_cache.StampPendingAsc(asc_value);
            ++asc_dispatches;
            if ((asc_dispatches & 0xffff) == 0) {
                LOG_INFO(Render_Vulkan,
                         "AscStream: dispatches={} demoted_to_main={} reasons[struct/written/"
                         "read/args/gds/texel_image]={}/{}/{}/{}/{}/{} main_flushes={}",
                         asc_dispatches, asc_demoted, asc_demote_reason[1], asc_demote_reason[2],
                         asc_demote_reason[3], asc_demote_reason[4], asc_demote_reason[5],
                         asc_demote_reason[6], asc_flushes);
            }
        }
        dispatch_target = 0;
        dispatch_on_asc = false;
        cur_dispatch_run = 0;
        buffer_cache.SetAscIndirectArgs(0, 0);
        // Nothing between records may land in the ASC stream (markers, CP-self work).
        scheduler.SwitchStream(0);
        return;
    }
    prev_dispatch_run = (cur_dispatch_run != 0 && !flushed) ? dispatch_run : 0;
    prev_effective_run = cur_dispatch_run;
    prev_dispatch_end_seq = scheduler.RecordSeq();
    prev_dispatch_end_tick = scheduler.CurrentTick();
    cur_dispatch_run = 0;
}

u64 Rasterizer::StreamBatchId(u32 stream) const {
    return stream == 1 ? scheduler.OpenAscValue() : scheduler.CurrentTick();
}

bool Rasterizer::AscCandidate(const ComputePipeline* pipeline) const {
    // Buffers only: no images, samplers, texel buffers (may alias images), DMA / fault / BDA
    // buffers. HLE shaders and image copy / clear shaders never reach BindResources.
    // Descriptor sets from the heap are recycled by the main timeline (asc-i1-relax design R0
    // table): ASC pipelines must push their descriptors.
    if (!pipeline->UsesPushDescriptors()) {
        static std::atomic<u64> no_push{0};
        const u64 n = no_push.fetch_add(1) + 1;
        if ((n & (n - 1)) == 0) {
            LOG_INFO(Render_Vulkan, "AscCandidate: {} dispatches rejected (heap descriptors)", n);
        }
        return false;
    }
    const auto* info = bound_stages[static_cast<u32>(Shader::LogicalStage::Compute)].info;
    if (info && VideoCore::BufferCache::CensusEnabled()) [[unlikely]] {
        // SHADPS4_ASC_V2_CENSUS: why an async-queue pipeline is not eligible, once per shader.
        static std::unordered_set<u64> logged;
        if (logged.insert(info->pgm_hash).second) {
            std::string bufs;
            for (const auto& desc : info->buffers) {
                bufs += fmt::format(" [type={} formatted={} written={}]",
                                    static_cast<u32>(desc.buffer_type), desc.is_formatted,
                                    desc.is_written);
            }
            LOG_INFO(Render_Vulkan, "AscCensusPipe: cs_{:08x} images={} samplers={} uses_dma={} "
                                    "buffers:{}",
                     info->pgm_hash, info->images.size(), info->samplers.size(), info->uses_dma,
                     bufs);
        }
    }
    if (!info || !info->images.empty() || !info->samplers.empty() || info->uses_dma) {
        return false;
    }
    // SHADPS4_ASC_TEXEL=1: formatted (texel) buffers are allowed; BindBuffers keeps a dispatch on
    // main only if a read texel binding starts where a cached image does (the one case whose
    // SynchronizeBufferFromImage records an image->buffer copy). Census 2026-09-28: most particle
    // shaders bind one formatted buffer, and the smoke sort's formatted output on main blocked
    // 38% of the control-block copies from the ASC path.
    static const bool asc_texel = [] {
        const char* value = std::getenv("SHADPS4_ASC_TEXEL");
        return value && value[0] == '1';
    }();
    for (const auto& desc : info->buffers) {
        if (desc.is_formatted && !asc_texel) {
            return false;
        }
        if (desc.IsSpecial() && desc.buffer_type != Shader::BufferType::GdsBuffer &&
            desc.buffer_type != Shader::BufferType::Flatbuf &&
            desc.buffer_type != Shader::BufferType::SharedMemory) {
            return false;
        }
    }
    return true;
}

void Rasterizer::DrawTail(const GraphicsPipeline* pipeline, bool is_indexed, u32 index_offset,
                          Common::DrawSampleProbe& sample,
                          std::chrono::steady_clock::time_point t_entry,
                          std::chrono::steady_clock::time_point t_pipeline, u64 pipeline_ns) {
    const auto& regs = liverpool->BackRegs();
    std::array<u64, DcCount> dc{};
    u64 dc_t = g_draw_cost ? __rdtsc() : 0;
    const auto dc_lap = [&](DrawCostPhase phase) {
        if (g_draw_cost) {
            const u64 now = __rdtsc();
            dc[phase] += now - dc_t;
            dc_t = now;
        }
    };
    const auto t_prepare = Common::FaultMonitor::Now();
    sample.BeginPhase(Common::DrawSampleProbe::Phase::Prepare);
    PrepareRenderState(pipeline);
    sample.EndPhase(Common::DrawSampleProbe::Phase::Prepare);
    dc_lap(DcPrepare);
    const auto t_bindres = Common::FaultMonitor::Now();
    sample.BeginPhase(Common::DrawSampleProbe::Phase::BindResources);
    const bool bound = BindResources(pipeline);
    sample.EndPhase(Common::DrawSampleProbe::Phase::BindResources);
    const auto bindres_ns = Common::FaultMonitor::SinceNs(t_bindres);
    if (!bound) {
        return;
    }
    if (g_draw_cost) {
        dc_lap(DcBindOther);
        dc[DcBindBuffers] = g_dc_bind_buf;
        dc[DcBindTextures] = g_dc_bind_tex;
        dc[DcBindOther] -= std::min(dc[DcBindOther], g_dc_bind_buf + g_dc_bind_tex);
    }
    const auto t_begin = Common::FaultMonitor::Now();
    sample.BeginPhase(Common::DrawSampleProbe::Phase::BuildRenderingState);
    {
        // Diagnostic SHADPS4_NEO_REG_DIAG=1: raw depth / AA / CB0 registers, logged once per
        // distinct combination (max 60), to see how PS4 Pro checkerboard programs the DB.
        static const bool neo_reg_diag = [] {
            const char* value = std::getenv("SHADPS4_NEO_REG_DIAG");
            return value && value[0] == '1';
        }();
        if (neo_reg_diag) {
            // Sample-location registers (PA_SC_AA_SAMPLE_LOCS_PIXEL_X0Y0_0 .. X1Y1_0) and
            // PA_SC_AA_CONFIG: log each change with a timestamp (first 200), to see whether the
            // Pro checkerboard alternates sample positions per frame.
            const auto& ra = liverpool->BackRegs().reg_array;
            constexpr u32 base = AmdGpu::Regs::ContextRegWordOffset;
            const std::array<u32, 5> locs{ra[base + 0x2F8], ra[base + 0x2FE], ra[base + 0x302],
                                          ra[base + 0x306], ra[base + 0x30A]};
            static std::array<u32, 5> last_locs{};
            static u32 loc_lines = 0;
            if (locs != last_locs && loc_lines < 200) {
                ++loc_lines;
                last_locs = locs;
                const auto us = std::chrono::duration_cast<std::chrono::microseconds>(
                                    std::chrono::steady_clock::now().time_since_epoch())
                                    .count();
                LOG_WARNING(Render_Vulkan,
                            "NeoSampleLocs: t={}us AA_CONFIG={:#x} X0Y0={:#010x} X1Y0={:#010x} "
                            "X0Y1={:#010x} X1Y1={:#010x} rt0={}x{}",
                            us, locs[0], locs[1], locs[2], locs[3], locs[4],
                            liverpool->BackRegs().color_buffers[0].Pitch(),
                            liverpool->BackRegs().color_buffers[0].Height());
            }
        }
        if (neo_reg_diag && liverpool->BackRegs().depth_buffer.DepthValid()) {
            static constexpr std::array<u32, 16> offsets{
                0x000, 0x003, 0x010, 0x011, 0x012, 0x014, 0x016, 0x017,
                0x201, 0x203, 0x292, 0x293, 0x2F8, 0x2F9, 0x31D, 0x31C};
            std::array<u32, 16> values{};
            for (size_t i = 0; i < offsets.size(); ++i) {
                values[i] = liverpool->BackRegs().reg_array[AmdGpu::Regs::ContextRegWordOffset +
                                                            offsets[i]];
            }
            static std::set<std::array<u32, 16>> seen;
            if (seen.size() < 60 && seen.insert(values).second) {
                LOG_WARNING(Render_Vulkan,
                            "NeoRegs: DB_RENDER_CONTROL={:#x} DB_RENDER_OVERRIDE={:#x} "
                            "Z_INFO={:#x} STENCIL_INFO={:#x} Z_READ_BASE={:#x} Z_WRITE_BASE={:#x} "
                            "DEPTH_SIZE={:#x} DEPTH_SLICE={:#x} DB_EQAA={:#x} "
                            "DB_SHADER_CONTROL={:#x} PA_SC_MODE_CNTL_0={:#x} MODE_CNTL_1={:#x} "
                            "PA_SC_AA_CONFIG={:#x} AA_MASK?={:#x} CB0_ATTRIB={:#x} CB0_INFO={:#x}",
                            values[0], values[1], values[2], values[3], values[4], values[5],
                            values[6], values[7], values[8], values[9], values[10], values[11],
                            values[12], values[13], values[14], values[15]);
            }
        }
    }
    const auto state = BeginRendering(pipeline);
    sample.EndPhase(Common::DrawSampleProbe::Phase::BuildRenderingState);
    dc_lap(DcBeginRendering);
    if (g_merge_probe) {
        thread_local RenderState last_state{};
        if (!(state == last_state)) {
            ++g_merge_rt_epoch;
            last_state = state;
        }
        NoteMergeProbe(pipeline, BoundStages(pipeline), regs, is_indexed, index_offset);
    }
    const auto t_vtxbuf = Common::FaultMonitor::Now();
    Common::FaultMonitor::CountDraw(pipeline_ns, bindres_ns,
                                    Common::FaultMonitor::DiffNs(t_begin, t_vtxbuf));

    sample.BeginPhase(Common::DrawSampleProbe::Phase::VertexIndexBuffers);
    buffer_cache.BindVertexBuffers(*pipeline, GetBoundStage(Shader::LogicalStage::Vertex),
                                   buffer_barriers);
    if (is_indexed) {
        buffer_cache.BindIndexBuffer(index_offset, buffer_barriers);
    }
    sample.EndPhase(Common::DrawSampleProbe::Phase::VertexIndexBuffers);
    dc_lap(DcVertexIndex);
    const auto t_descwrite = Common::FaultMonitor::Now();

    sample.BeginPhase(Common::DrawSampleProbe::Phase::PipelineResources);
    EmitPipelineResources(pipeline);
    sample.EndPhase(Common::DrawSampleProbe::Phase::PipelineResources);
    dc_lap(DcEmitResources);
    const auto t_dynstate = Common::FaultMonitor::Now();
    sample.BeginPhase(Common::DrawSampleProbe::Phase::DynamicState);
    UpdateDynamicState(pipeline, is_indexed);
    sample.EndPhase(Common::DrawSampleProbe::Phase::DynamicState);
    const auto t_schedbegin = Common::FaultMonitor::Now();
    sample.BeginPhase(Common::DrawSampleProbe::Phase::SchedulerBeginRendering);
    scheduler.BeginRendering(state);
    sample.EndPhase(Common::DrawSampleProbe::Phase::SchedulerBeginRendering);
    const auto t_emit = Common::FaultMonitor::Now();

    sample.BeginPhase(Common::DrawSampleProbe::Phase::ActualDraw);
    const auto& vs_info = GetBoundStage(Shader::LogicalStage::Vertex);
    const auto& fetch_shader = pipeline->GetFetchShader();
    const auto [vertex_offset, instance_offset] = GetDrawOffsets(regs, vs_info, fetch_shader);

    scheduler.Record([handle = pipeline->Handle(), is_indexed, count = regs.num_indices,
                      instances = regs.num_instances.NumInstances(), vertex_offset,
                      instance_offset](vk::CommandBuffer cmdbuf) {
        if (!GraphicsBindTracker::Enabled() || GraphicsBindTracker::NeedBind(cmdbuf, handle)) {
            cmdbuf.bindPipeline(vk::PipelineBindPoint::eGraphics, handle);
        }
        if (is_indexed) {
            cmdbuf.drawIndexed(count, instances, 0, s32(vertex_offset), instance_offset);
        } else {
            cmdbuf.draw(count, instances, vertex_offset, instance_offset);
        }
    });
    DebugState.IncDrawCall();
    scheduler.NoteDraw();

    sample.EndPhase(Common::DrawSampleProbe::Phase::ActualDraw);
    sample.BeginPhase(Common::DrawSampleProbe::Phase::ResetImageBindings);
    ResetBindings();
    sample.EndPhase(Common::DrawSampleProbe::Phase::ResetImageBindings);
    sample.Finish();
    if (g_draw_cost) {
        dc_lap(DcRecordTail);
        NoteDrawCost(pipeline, dc);
        // ~50 us at ~3.5 GHz TSC: log where a slow texture phase went (first 5 per shader).
        static thread_local std::unordered_map<u64, u32> slow_logged;
        u64 fs = 0;
        if (dc[DcBindTextures] > 175000) {
            for (const auto* stage : pipeline->GetStages()) {
                if (stage && stage->l_stage == Shader::LogicalStage::Fragment) {
                    fs = stage->pgm_hash;
                }
            }
        }
        if (dc[DcBindTextures] > 175000 && slow_logged[fs]++ < 5) {
            const auto& si = g_dc_slow_img;
            LOG_INFO(Render_Vulkan,
                     "DrawCostSlowTex: fs_{:08x} tex={} cyc first_pass={} find_texture={} "
                     "transit={} | slowest image {:#x} {}x{}x{} fmt={} depth={} tiled={} "
                     "layout_before={} cyc={}",
                     fs, dc[DcBindTextures], g_dc_tex_sub[0], g_dc_tex_sub[1], g_dc_tex_sub[2],
                     si.addr, si.w, si.h, si.layers, vk::to_string(vk::Format(si.fmt)), si.depth,
                     si.tiled, vk::to_string(vk::ImageLayout(si.layout_before)), si.cyc);
        }
    }

    // SHADPS4_FLUSH_EVERY_DRAWS=N: hand recorded work to the GPU every N draws. Guest readback
    // faults wait for everything queued before their copy; measured 2026-09-23 Game:Main spent
    // 51% of its wall time in that wait while the CP sat 41% idle, so earlier submission trades
    // spare CP time for shorter readback latency. 0 (default) keeps the old cadence.
    static const u32 flush_every_draws = [] {
        const char* value = std::getenv("SHADPS4_FLUSH_EVERY_DRAWS");
        return value ? static_cast<u32>(std::strtoul(value, nullptr, 10)) : 0u;
    }();
    if (flush_every_draws != 0) {
        static thread_local u32 draws_since_flush = 0;
        if (++draws_since_flush >= flush_every_draws) {
            draws_since_flush = 0;
            scheduler.Flush();
        }
    }
    // SHADPS4_READBACK_EARLY_FLUSH=1: this draw wrote into memory the guest recently read
    // back; submit now so the next readback of it waits less (see BufferCache::NoteGpuWrite).
    if (buffer_cache.TakeReadbackFlushRequest()) {
        static thread_local u32 early_flushes = 0;
        ++early_flushes;
        scheduler.Flush();
    }
    MaybeFlushEveryOps(scheduler);

    Common::FaultMonitor::CountDrawTail(Common::FaultMonitor::DiffNs(t_entry, t_pipeline) +
                                            Common::FaultMonitor::DiffNs(t_prepare, t_bindres),
                                        Common::FaultMonitor::DiffNs(t_vtxbuf, t_descwrite),
                                        Common::FaultMonitor::DiffNs(t_descwrite, t_dynstate),
                                        Common::FaultMonitor::DiffNs(t_dynstate, t_schedbegin),
                                        Common::FaultMonitor::DiffNs(t_schedbegin, t_emit),
                                        Common::FaultMonitor::SinceNs(t_emit));
}

void Rasterizer::DrawIndirect(bool is_indexed, VAddr arg_address, u32 offset, u32 stride,
                              u32 max_count, VAddr count_address) {
    RENDERER_TRACE;

    scheduler.PopPendingOperations();
    if (native_bindings) {
        native_bindings->BeginDraw();
    }

    if (!FilterDraw()) {
        return;
    }

    const GraphicsPipeline* pipeline = pipeline_cache.GetGraphicsPipeline();
    if (!pipeline) {
        return;
    }
    BindStageData(pipeline);
    DrawIndirectTail(pipeline, is_indexed, arg_address, offset, stride, max_count, count_address);
}

void Rasterizer::DrawIndirectTail(const GraphicsPipeline* pipeline, bool is_indexed,
                                  VAddr arg_address, u32 offset, u32 stride, u32 max_count,
                                  VAddr count_address) {
    PrepareRenderState(pipeline);
    if (!BindResources(pipeline)) {
        return;
    }
    const auto state = BeginRendering(pipeline);

    buffer_cache.BindVertexBuffers(*pipeline, GetBoundStage(Shader::LogicalStage::Vertex),
                                   buffer_barriers);
    if (is_indexed) {
        buffer_cache.BindIndexBuffer(0, buffer_barriers);
    }

    const auto& [buffer, base] =
        buffer_cache.ObtainBuffer(arg_address + offset, stride * max_count, false);

    VideoCore::Buffer* count_buffer{};
    u32 count_base{};
    if (count_address != 0) {
        std::tie(count_buffer, count_base) = buffer_cache.ObtainBuffer(count_address, 4, false);
    }

    if (auto barrier = buffer->GetBarrier(vk::AccessFlagBits2::eIndirectCommandRead,
                                          vk::PipelineStageFlagBits2::eDrawIndirect)) {
        buffer_barriers.emplace_back(*barrier);
    }
    if (count_buffer) {
        if (auto barrier = count_buffer->GetBarrier(vk::AccessFlagBits2::eIndirectCommandRead,
                                                    vk::PipelineStageFlagBits2::eDrawIndirect)) {
            buffer_barriers.emplace_back(*barrier);
        }
    }

    EmitPipelineResources(pipeline);
    UpdateDynamicState(pipeline, is_indexed);
    scheduler.BeginRendering(state);

    // We can safely ignore both SGPR UD indices and results of fetch shader parsing, as vertex and
    // instance offsets will be automatically applied by Vulkan from indirect args buffer.

    ASSERT((is_indexed ? sizeof(VkDrawIndexedIndirectCommand) : sizeof(VkDrawIndirectCommand)) ==
           stride);
    scheduler.Record([handle = pipeline->Handle(), buffer = buffer->Handle(), base,
                      count_handle = count_buffer ? count_buffer->Handle() : vk::Buffer{},
                      has_count = count_address != 0, count_base, max_count, stride,
                      is_indexed](vk::CommandBuffer cmdbuf) {
        if (!GraphicsBindTracker::Enabled() || GraphicsBindTracker::NeedBind(cmdbuf, handle)) {
            cmdbuf.bindPipeline(vk::PipelineBindPoint::eGraphics, handle);
        }
        if (is_indexed) {
            if (has_count) {
                cmdbuf.drawIndexedIndirectCount(buffer, base, count_handle, count_base, max_count,
                                                stride);
            } else {
                cmdbuf.drawIndexedIndirect(buffer, base, max_count, stride);
            }
        } else {
            if (has_count) {
                cmdbuf.drawIndirectCount(buffer, base, count_handle, count_base, max_count, stride);
            } else {
                cmdbuf.drawIndirect(buffer, base, max_count, stride);
            }
        }
    });
    DebugState.IncDrawCall();
    scheduler.NoteOp();

    ResetBindings();
    MaybeFlushEveryOps(scheduler);
}

void Rasterizer::DispatchDirect() {
    RENDERER_TRACE;

    // 526 dispatches per forest frame, and they sit in the 27% of the frame that Draw's timers do
    // not cover. SCOPE_EXIT so the several early returns below are still counted.
    const auto t_dispatch = Common::FaultMonitor::Now();
    SCOPE_EXIT {
        Common::FaultMonitor::CountDispatch(Common::FaultMonitor::SinceNs(t_dispatch));
    };

    scheduler.PopPendingOperations();
    if (native_bindings) {
        native_bindings->BeginDraw();
    }

    const ComputePipeline* pipeline = pipeline_cache.GetComputePipeline();
    if (!pipeline) {
        return;
    }
    BindStageData(pipeline);
    DispatchDirectTail(pipeline);
}

void Rasterizer::DispatchDirectTail(const ComputePipeline* pipeline) {
    const auto& cs_program = liverpool->BackCsRegs();
    const auto& cs = GetBoundStage(Shader::LogicalStage::Compute);
    if (ExecuteShaderHLE(cs, liverpool->BackRegs(), cs_program, *this)) {
        return;
    }

    if (!BindResources(pipeline)) {
        return;
    }

    scheduler.EndRendering();
    EmitPipelineResources(pipeline);

    scheduler.Record([handle = pipeline->Handle(), x = cs_program.dim_x, y = cs_program.dim_y,
                      z = cs_program.dim_z](vk::CommandBuffer cmdbuf) {
        cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, handle);
        cmdbuf.dispatch(x, y, z);
    });
    DebugState.IncDispatch();
    scheduler.NoteDispatch();

    ResetBindings();
    if (buffer_cache.TakeReadbackFlushRequest()) {
        scheduler.Flush();
    }
    MaybeFlushEveryOps(scheduler);
}

void Rasterizer::DispatchIndirect(VAddr address, u32 offset, u32 size) {
    RENDERER_TRACE;

    const auto t_dispatch = Common::FaultMonitor::Now();
    SCOPE_EXIT {
        Common::FaultMonitor::CountDispatch(Common::FaultMonitor::SinceNs(t_dispatch));
    };

    scheduler.PopPendingOperations();
    if (native_bindings) {
        native_bindings->BeginDraw();
    }

    const ComputePipeline* pipeline = pipeline_cache.GetComputePipeline();
    if (!pipeline) {
        return;
    }
    BindStageData(pipeline);
    DispatchIndirectTail(pipeline, address, offset, size);
}

void Rasterizer::DispatchIndirectTail(const ComputePipeline* pipeline, VAddr address, u32 offset,
                                      u32 size) {
    if (!BindResources(pipeline)) {
        return;
    }

    const auto [buffer, base] = buffer_cache.ObtainBuffer(address + offset, size, false);

    if (auto barrier = buffer->GetBarrier(vk::AccessFlagBits2::eIndirectCommandRead,
                                          vk::PipelineStageFlagBits2::eDrawIndirect)) {
        buffer_barriers.emplace_back(*barrier);
    }

    scheduler.EndRendering();
    EmitPipelineResources(pipeline);

    scheduler.Record(
        [handle = pipeline->Handle(), buffer = buffer->Handle(), base](vk::CommandBuffer cmdbuf) {
            cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, handle);
            cmdbuf.dispatchIndirect(buffer, base);
        });
    DebugState.IncDispatch();
    scheduler.NoteDispatch();

    ResetBindings();
    MaybeFlushEveryOps(scheduler);
}

u64 Rasterizer::Flush() {
    const u64 current_tick = scheduler.CurrentTick();
    SubmitInfo info{};
    scheduler.Flush(info);
    return current_tick;
}

void Rasterizer::Finish() {
    scheduler.Finish();
}

void Rasterizer::DeferToGpu(Common::UniqueFunction<void>&& func, bool flush) {
    // Queued at the current tick first, then that tick is submitted (or, lazily, left to the
    // next flush: every SHADPS4_FLUSH_EVERY_DRAWS draws, a CP memory wait or CP idle).
    scheduler.DeferPriorityOperation(std::move(func));
    if (flush) {
        Flush();
    } else {
        lazy_fence_tick = scheduler.CurrentTick();
    }
}

void Rasterizer::FlushLazyFences() {
    if (lazy_fence_tick != 0 && lazy_fence_tick == scheduler.CurrentTick()) {
        Flush();
    }
    lazy_fence_tick = 0;
}

void Rasterizer::OnSubmit() {
    {
        // Diagnostic: SHADPS4_DUMP_IMAGES=addr[,addr...] (hex) with SHADPS4_DUMP_IMAGES_AT_S=N
        // dumps those surfaces at the first submit N s after the first one, twice (a frame apart).
        static const std::vector<VAddr> dump_addresses = [] {
            std::vector<VAddr> out;
            if (const char* value = std::getenv("SHADPS4_DUMP_IMAGES")) {
                std::string list{value};
                size_t pos = 0;
                while (pos < list.size()) {
                    const size_t comma = list.find(',', pos);
                    out.push_back(std::strtoull(list.substr(pos, comma - pos).c_str(), nullptr, 16));
                    pos = comma == std::string::npos ? list.size() : comma + 1;
                }
            }
            return out;
        }();
        if (!dump_addresses.empty()) {
            static const double at_s = [] {
                const char* value = std::getenv("SHADPS4_DUMP_IMAGES_AT_S");
                return value ? std::atof(value) : 120.0;
            }();
            static const auto first = std::chrono::steady_clock::now();
            static u32 dumps_done = 0;
            if (dumps_done < 2 &&
                std::chrono::duration<double>(std::chrono::steady_clock::now() - first).count() >=
                    at_s + dumps_done * 0.05) {
                texture_cache.DebugDumpImages(dump_addresses, dumps_done++);
            }
        }
    }
    if (fault_process_pending) {
        fault_process_pending = false;
        buffer_cache.ProcessFaultBuffer();
    }
    texture_cache.ProcessDownloadImages();
    texture_cache.RunGarbageCollector();
    buffer_cache.RunGarbageCollector();
}

bool Rasterizer::BindResources(const Pipeline* pipeline) {
    if (!pending_stencil_writebacks.empty()) {
        FlushStencilWritebacks();
    }
    if (IsComputeImageCopy(pipeline) || IsComputeMetaClear(pipeline) ||
        IsComputeImageClear(pipeline)) {
        return false;
    }

#if SHADPS4_BINDING_PROBE
    ProbeBindingChanges(pipeline, BoundStages(pipeline));
#endif

    set_write_index = 0;
    set_writes.clear();
    buffer_barriers.clear();
    buffer_infos.clear();
    image_infos.clear();
    g_dc_bind_buf = 0;
    g_dc_bind_tex = 0;
    g_dc_tex_sub = {};
    g_dc_slow_img = {};

    bool uses_dma = false;
    binding_compute = pipeline->IsCompute();

    // Bind resource buffers and textures.
    Shader::Backend::Bindings binding{};
    push_data = MakeUserData(liverpool->BackRegs());
    u8 memo_draw_bits = 3;
    for (const auto& bound : BoundStages(pipeline)) {
        const auto* stage = bound.info;
        if (!stage) {
            continue;
        }
        if (build_set_writes) {
            set_writes.resize(set_writes.size() + stage->buffers.size() + stage->images.size() +
                              stage->samplers.size());
        }
        bound.PushUd(binding, push_data);
        if (bind_memo_stats) {
            // Runs the same BindBuffers/BindTextures, with the memo's key and checks around them.
            memo_draw_bits &= BindStageMemoStats(pipeline, bound, binding);
        } else {
            // Split measured because BindResources is 6.8 us of the 8.0 us spent per draw - 85% -
            // and a stack scan could not tell these two apart from anything else.
            const auto t_buf = Common::FaultMonitor::Now();
            if (VideoCore::BufferCache::ReadProducerStatsEnabled()) [[unlikely]] {
                buffer_cache.SetDiagProducer(stage->pgm_hash);
            }
            const u64 dc0 = g_draw_cost ? __rdtsc() : 0;
            BindBuffers(bound, binding, push_data);
            const u64 dc1 = g_draw_cost ? __rdtsc() : 0;
            const auto t_tex = Common::FaultMonitor::Now();
            if (tex_bind_memo) {
                BindTexturesMemo(pipeline, bound, binding);
            } else {
                BindTextures(bound, binding);
            }
            if (g_draw_cost) {
                g_dc_bind_buf += dc1 - dc0;
                g_dc_bind_tex += __rdtsc() - dc1;
            }
            Common::FaultMonitor::CountBindSplit(Common::FaultMonitor::DiffNs(t_buf, t_tex),
                                                 Common::FaultMonitor::SinceNs(t_tex));
        }
        uses_dma |= stage->uses_dma;
    }
    native_bound_count = binding.unified;
    if (VideoCore::BufferCache::ReadProducerStatsEnabled()) [[unlikely]] {
        buffer_cache.SetDiagProducer(0); // Later written bindings (DMA, HLE, fills): untracked.
    }
    if (bind_memo_stats) {
        BindMemoEndDraw(pipeline->IsCompute(), memo_draw_bits);
    }

    if (uses_dma) {
        // We only use fault buffer for DMA right now.
        Common::RecursiveSharedLock lock{mapped_ranges_mutex};
        for (auto& range : mapped_ranges) {
            buffer_cache.SynchronizeBuffersInRange(range.lower(), range.upper() - range.lower());
        }
        fault_process_pending = true;
    }

    return true;
}

namespace {

// SHADPS4_BIND_MEMO_STATS state, CP thread only. One record per (pipeline, stage), which is where
// the memo itself would keep it. Generations are the values read before the full path that
// produced the outputs, so a bump during that path makes the next occurrence miss.
struct BindMemoRecord {
    struct Buf {
        VideoCore::BufferId id{};
        VAddr addr{};
        u64 size{};
        vk::Buffer handle{};
        vk::DeviceSize offset{};
        vk::DeviceSize range{};
        u8 adjust{};
        bool memo{}; // read-only guest binding: a hit would skip it
    };
    struct Img {
        VideoCore::ImageId id{};
        u64 uid{};
        const VideoCore::Image::BackingImage* backing{};
        vk::ImageView view{};
        vk::ImageLayout layout{};
        VideoCore::Image::State state{}; // after the full path's Transit
    };
    bool valid{};
    std::vector<AmdGpu::Buffer> vsharps;
    std::vector<AmdGpu::Image> tsharps;
    std::vector<AmdGpu::Sampler> ssharps;
    u64 border{};
    u64 buf_gen{};
    u64 buf_tex_gen{};
    bool buf_excluded{};
    bool buf_texel{};
    std::vector<Buf> bufs;
    u64 tex_gen{};
    u64 smp_gen{};
    bool tex_excluded{};
    std::vector<Img> imgs;
    std::vector<vk::Sampler> samplers;
};

enum BindMemoBufMiss : u32 { BufExcluded, BufGen, BufTexGen, BufDirtyPage, NumBufMiss };
enum BindMemoTexMiss : u32 {
    TexExcluded,
    TexGen,
    TexSamplerGen,
    TexUid,
    TexBacking,
    TexDirtyImage,
    TexFlags,
    NumTexMiss
};

// Hit work that is not replayed: GetBarrier per skipped buffer, the sampler LRU touch per
// sampler. Guesses, subtracted so the estimate is not flattered by leaving them out.
constexpr double BindMemoAllowBufNs = 5.0;
constexpr double BindMemoAllowSamplerNs = 5.0;

struct BindMemoStats {
    std::unordered_map<u64, BindMemoRecord> records;
    // Key under test and dry-replay scratch; capacity is kept so timed spans do not allocate.
    std::vector<AmdGpu::Buffer> vsharps;
    std::vector<AmdGpu::Image> tsharps;
    std::vector<AmdGpu::Sampler> ssharps;
    std::vector<vk::DescriptorBufferInfo> replay_buffers;
    std::vector<vk::DescriptorImageInfo> replay_images;
    std::array<u8, Shader::NUM_BUFFERS> replay_adjust{};
    u64 stage_counter{};
    bool started{};
    // Window origin.
    u64 frames0{}, buf_gen0{}, tex_gen0{}, smp_gen0{}, tsc0{};
    std::chrono::steady_clock::time_point clock0{};
    // Window counters. Stage types: 0 vs, 1 ps, 2 cs, 3 other.
    u64 binds{}, stages{}, draws{}, draw_key{}, draw_would{}, dma_stages{};
    std::array<u64, 4> type_stages{}, type_key{}, type_would{};
    u64 buf_key{}, buf_would{}, tex_key{}, tex_would{}, transit_needed{};
    std::array<u64, NumBufMiss> buf_miss{};
    std::array<u64, NumTexMiss> tex_miss{};
    u64 slot_buf{}, slot_buf_key{}, slot_img{}, slot_img_key{}, slot_smp{}, slot_smp_key{};
    u64 mis_buf{}, mis_upload{}, mis_img{}, mis_smp{}, layout_diff{};
    u64 memo_bufs_on_hit{}, samplers_on_hit{};
    // Timed stages (1 in 16), TSC ticks. t_empty is two back-to-back reads, subtracted from
    // every span average so the fenced reads themselves are not counted as work.
    u64 timed{}, t_empty{}, t_key_buf{}, t_key_tex{};
    u64 t_check_buf_n{}, t_check_buf{}, t_check_tex_n{}, t_check_tex{};
    u64 t_replay_buf_n{}, t_replay_buf{}, t_replay_tex_n{}, t_replay_tex{};
    u64 t_buf_hit_n{}, t_buf_hit{}, t_fixed_hit{}, t_buf_miss_n{}, t_buf_miss{};
    u64 t_tex_hit_n{}, t_tex_hit{}, t_tex_miss_n{}, t_tex_miss{};
};

BindMemoStats& GetBindMemoStats() {
    static BindMemoStats stats;
    return stats;
}

template <typename T>
bool SameBytes(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() &&
           (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0);
}

template <typename T>
bool SameSlot(const std::vector<T>& a, const std::vector<T>& b, size_t i) {
    return i < a.size() && i < b.size() && std::memcmp(&a[i], &b[i], sizeof(T)) == 0;
}

} // Anonymous namespace

u8 Rasterizer::BindStageMemoStats(const Pipeline* pipeline, const Shader::BoundStage& bound,
                                  Shader::Backend::Bindings& binding) {
    const auto& stage = *bound.info;
    auto& st = GetBindMemoStats();
    if (!st.started) {
        st.started = true;
        st.frames0 = Common::FaultMonitor::presented_frames.load(std::memory_order_relaxed);
        st.buf_gen0 = buffer_cache.StructGeneration();
        st.tex_gen0 = texture_cache.StructGeneration();
        st.smp_gen0 = texture_cache.SamplerGeneration();
        st.clock0 = std::chrono::steady_clock::now();
        st.tsc0 = Common::FencedRDTSC();
    }
    const bool timed = (++st.stage_counter & 15) == 0;
    const auto tsc = [timed] { return timed ? Common::FencedRDTSC() : u64{0}; };
    // The lookup stays outside the timed spans: the memo would own its record.
    const u64 record_key =
        (static_cast<u64>(reinterpret_cast<uintptr_t>(pipeline)) << 3) | u64(stage.l_stage);
    auto& rec = st.records[record_key];
    const u32 type = stage.l_stage == Shader::LogicalStage::Vertex     ? 0
                     : stage.l_stage == Shader::LogicalStage::Fragment ? 1
                     : stage.l_stage == Shader::LogicalStage::Compute  ? 2
                                                                       : 3;

    const u64 t_empty = tsc();
    // Key, buffer half: every guest V#, read as BindBuffers reads it.
    const u64 t_key = tsc();
    st.vsharps.clear();
    for (const auto& desc : stage.buffers) {
        st.vsharps.push_back(desc.IsSpecial() ? AmdGpu::Buffer{} : desc.GetSharp(bound));
    }
    const bool buf_key = rec.valid && SameBytes(st.vsharps, rec.vsharps);
    const u64 t_key_tex = tsc();
    // Key, texture half: every T#, every S# after BindTextures' aniso adjustment, border base.
    st.tsharps.clear();
    for (const auto& desc : stage.images) {
        st.tsharps.push_back(desc.GetSharp(bound));
    }
    st.ssharps.clear();
    for (const auto& sampler : stage.samplers) {
        auto ssharp = sampler.GetSharp(bound);
        if (sampler.disable_aniso) {
            const auto& tsharp = stage.images[sampler.associated_image].GetSharp(bound);
            if (tsharp.base_level == 0 && tsharp.last_level == 0) {
                ssharp.max_aniso.Assign(AmdGpu::AnisoRatio::One);
            }
        }
        st.ssharps.push_back(ssharp);
    }
    const u64 border = liverpool->BackRegs().ta_bc_base.base_addr;
    const bool tex_key = rec.valid && border == rec.border && SameBytes(st.tsharps, rec.tsharps) &&
                         SameBytes(st.ssharps, rec.ssharps);

    // Validation, buffer half (design section 3): structure generation, texel aliasing, exact
    // CPU-dirty pages of every skipped range. Written and special bindings run in full anyway.
    const u64 t_check_buf = tsc();
    const u64 buf_gen = buffer_cache.StructGeneration();
    const u64 tex_gen = texture_cache.StructGeneration();
    const u64 smp_gen = texture_cache.SamplerGeneration();
    s32 buf_miss = -1;
    if (buf_key) {
        if (rec.buf_excluded || stage.uses_dma || native_bindings) {
            buf_miss = BufExcluded;
        } else if (buf_gen != rec.buf_gen) {
            buf_miss = BufGen;
        } else if (rec.buf_texel && tex_gen != rec.buf_tex_gen) {
            buf_miss = BufTexGen;
        } else {
            for (const auto& b : rec.bufs) {
                if (b.memo && b.id && buffer_cache.IsRegionCpuDirtyRelaxed(b.addr, b.size)) {
                    buf_miss = BufDirtyPage;
                    break;
                }
            }
        }
    }
    const bool buf_would = buf_key && buf_miss < 0;
    // Validation, texture half: image structure and sampler generations, then per image the
    // identity, backing, UpdateImageLockFree predicate and binding flags. A state difference is
    // not a miss: the memo would call Transit.
    const u64 t_check_tex = tsc();
    s32 tex_miss = -1;
    bool transit = false;
    if (tex_key) {
        if (rec.tex_excluded || native_bindings) {
            tex_miss = TexExcluded;
        } else if (tex_gen != rec.tex_gen) {
            tex_miss = TexGen;
        } else if (!rec.samplers.empty() && smp_gen != rec.smp_gen) {
            tex_miss = TexSamplerGen;
        } else {
            for (const auto& img : rec.imgs) {
                if (!img.id) {
                    continue;
                }
                auto& image = texture_cache.PeekImage(img.id);
                if (image.image_uid != img.uid) {
                    tex_miss = TexUid;
                } else if (image.backing != img.backing) {
                    tex_miss = TexBacking;
                } else if (!texture_cache.ImageUpToDateRelaxed(img.id)) {
                    tex_miss = TexDirtyImage;
                } else if (image.binding.is_target || image.binding.force_general ||
                           image.binding.needs_rebind) {
                    tex_miss = TexFlags;
                }
                if (tex_miss >= 0) {
                    break;
                }
                const auto& backing = *image.backing;
                transit |= !(backing.state == img.state) || !backing.subresource_runs.empty();
            }
        }
    }
    const bool tex_would = tex_key && tex_miss < 0;
    const u64 t_checked = tsc();

    // Dry replay of the stored outputs into scratch, timed only: what a hit writes instead.
    u64 replay_buf_ticks = 0;
    u64 replay_tex_ticks = 0;
    if (timed && buf_would) {
        const u64 t0 = tsc();
        st.replay_buffers.clear();
        for (size_t i = 0; i < rec.bufs.size(); ++i) {
            const auto& b = rec.bufs[i];
            st.replay_buffers.emplace_back(b.handle, b.offset, b.range);
            st.replay_adjust[i] = b.adjust;
        }
        replay_buf_ticks = tsc() - t0;
    }
    if (timed && tex_would) {
        const u64 t0 = tsc();
        st.replay_images.clear();
        for (const auto& img : rec.imgs) {
            if (img.id) {
                static_cast<void>(texture_cache.GetImage(img.id)); // the LRU touch a hit keeps
            }
            st.replay_images.emplace_back(vk::Sampler{}, img.view, img.layout);
        }
        for (const auto sampler : rec.samplers) {
            st.replay_images.emplace_back(sampler, vk::ImageView{}, vk::ImageLayout::eGeneral);
        }
        replay_tex_ticks = tsc() - t0;
    }

    // The unchanged full path.
    const size_t buf_begin = buffer_infos.size();
    const size_t img_begin = image_infos.size();
    const u32 buf_binding = binding.buffer;
    bind_memo_active = true;
    bind_memo_timing = timed;
    bind_memo_fixed_ticks = 0;
    bind_memo_redirected = 0;
    bind_memo_uploaded.fill(0);
    const u64 t_full = tsc();
    BindBuffers(bound, binding, push_data);
    const u64 t_full_tex = tsc();
    BindTextures(bound, binding);
    const u64 t_done = tsc();
    bind_memo_active = false;
    bind_memo_timing = false;

    if (timed) {
        ++st.timed;
        st.t_empty += t_key - t_empty;
        st.t_key_buf += t_key_tex - t_key;
        st.t_key_tex += t_check_buf - t_key_tex;
        if (buf_key) {
            ++st.t_check_buf_n;
            st.t_check_buf += t_check_tex - t_check_buf;
        }
        if (tex_key) {
            ++st.t_check_tex_n;
            st.t_check_tex += t_checked - t_check_tex;
        }
        if (buf_would) {
            ++st.t_replay_buf_n;
            st.t_replay_buf += replay_buf_ticks;
            ++st.t_buf_hit_n;
            st.t_buf_hit += t_full_tex - t_full;
            st.t_fixed_hit += bind_memo_fixed_ticks;
        } else {
            ++st.t_buf_miss_n;
            st.t_buf_miss += t_full_tex - t_full;
        }
        if (tex_would) {
            ++st.t_replay_tex_n;
            st.t_replay_tex += replay_tex_ticks;
            ++st.t_tex_hit_n;
            st.t_tex_hit += t_done - t_full_tex;
        } else {
            ++st.t_tex_miss_n;
            st.t_tex_miss += t_done - t_full_tex;
        }
    }

    ++st.stages;
    ++st.type_stages[type];
    st.type_key[type] += buf_key && tex_key;
    st.type_would[type] += buf_would && tex_would;
    st.dma_stages += stage.uses_dma;
    st.buf_key += buf_key;
    st.buf_would += buf_would;
    st.tex_key += tex_key;
    st.tex_would += tex_would;
    st.transit_needed += tex_would && transit;
    if (buf_miss >= 0) {
        ++st.buf_miss[static_cast<u32>(buf_miss)];
    }
    if (tex_miss >= 0) {
        ++st.tex_miss[static_cast<u32>(tex_miss)];
    }
    // Per-slot repeats against the previous occurrence (the per-binding fallback's key).
    for (size_t i = 0; i < stage.buffers.size(); ++i) {
        if (!stage.buffers[i].IsSpecial()) {
            ++st.slot_buf;
            st.slot_buf_key += rec.valid && SameSlot(st.vsharps, rec.vsharps, i);
        }
    }
    for (size_t i = 0; i < st.tsharps.size(); ++i) {
        ++st.slot_img;
        st.slot_img_key += rec.valid && SameSlot(st.tsharps, rec.tsharps, i);
    }
    for (size_t i = 0; i < st.ssharps.size(); ++i) {
        ++st.slot_smp;
        st.slot_smp_key +=
            rec.valid && border == rec.border && SameSlot(st.ssharps, rec.ssharps, i);
    }

    // Soundness: on a predicted hit the full path must have produced exactly the stored outputs
    // and uploaded nothing. Anything else means the invalidation list is incomplete.
    if (buf_would) {
        if (rec.bufs.size() != stage.buffers.size()) {
            ++st.mis_buf;
        } else {
            for (size_t i = 0; i < rec.bufs.size(); ++i) {
                const auto& b = rec.bufs[i];
                if (!b.memo) {
                    continue;
                }
                const auto& info = buffer_infos[buf_begin + i];
                const auto buffer_id = std::get<0>(buffer_bindings[i]);
                st.mis_buf += buffer_id != b.id || info.buffer != b.handle ||
                              info.offset != b.offset || info.range != b.range ||
                              push_data.buf_offsets[buf_binding + i] != b.adjust;
                st.mis_upload += bind_memo_uploaded[i] != 0;
                ++st.memo_bufs_on_hit;
            }
        }
    }
    if (tex_would) {
        const size_t num_images = image_bindings.size();
        if (num_images != rec.imgs.size() || stage.samplers.size() != rec.samplers.size()) {
            ++st.mis_img;
        } else {
            for (size_t i = 0; i < num_images; ++i) {
                const auto& info = image_infos[img_begin + i];
                const auto& img = rec.imgs[i];
                st.mis_img += image_bindings[i].first != img.id || info.imageView != img.view;
                st.layout_diff += info.imageLayout != img.layout;
            }
            for (size_t i = 0; i < rec.samplers.size(); ++i) {
                st.mis_smp += image_infos[img_begin + num_images + i].sampler != rec.samplers[i];
            }
            st.samplers_on_hit += rec.samplers.size();
        }
    }

    // Record this occurrence (outside every timed span).
    rec.valid = true;
    rec.vsharps = st.vsharps;
    rec.tsharps = st.tsharps;
    rec.ssharps = st.ssharps;
    rec.border = border;
    rec.buf_gen = buf_gen;
    rec.buf_tex_gen = tex_gen;
    rec.tex_gen = tex_gen;
    rec.smp_gen = smp_gen;
    rec.buf_excluded = false;
    rec.buf_texel = false;
    rec.bufs.resize(stage.buffers.size());
    for (size_t i = 0; i < stage.buffers.size(); ++i) {
        const auto& desc = stage.buffers[i];
        const auto& [buffer_id, vsharp, size] = buffer_bindings[i];
        const auto& info = buffer_infos[buf_begin + i];
        auto& b = rec.bufs[i];
        b.id = buffer_id;
        b.addr = vsharp.base_address;
        b.size = size;
        b.handle = info.buffer;
        b.offset = info.offset;
        b.range = info.range;
        b.adjust = push_data.buf_offsets[buf_binding + i];
        b.memo = !desc.IsSpecial() && !desc.is_written;
        if (b.memo && buffer_id) {
            // Stream copies and host mirrors are not the cache buffer; a texel range aliasing
            // an image is re-tiled on every bind (SynchronizeBufferFromImage).
            if (info.buffer != buffer_cache.GetBuffer(buffer_id).Handle()) {
                rec.buf_excluded = true;
            }
            if (desc.is_formatted) {
                rec.buf_texel = true;
                if (texture_cache.FindImageFromRange(vsharp.base_address, size)) {
                    rec.buf_excluded = true;
                }
            }
        }
    }
    rec.tex_excluded = native_bindings != nullptr;
    for (const auto& desc : stage.images) {
        rec.tex_excluded |= desc.mip_fallback_mode != Shader::MipStorageFallbackMode::None;
    }
    rec.imgs.resize(image_bindings.size());
    for (size_t i = 0; i < image_bindings.size(); ++i) {
        const auto& [image_id, desc] = image_bindings[i];
        const auto& info = image_infos[img_begin + i];
        auto& img = rec.imgs[i];
        img = {};
        img.id = image_id;
        img.view = info.imageView;
        img.layout = info.imageLayout;
        if (!image_id) {
            continue;
        }
        auto& image = texture_cache.PeekImage(image_id);
        img.uid = image.image_uid;
        img.backing = image.backing;
        img.state = image.backing->state;
        // Storage images, depth/stencil redirects and the Pro ID buffer stay on the full path.
        rec.tex_excluded |= desc.type == VideoCore::TextureCache::BindingType::Storage ||
                            ((bind_memo_redirected >> (i & 63)) & 1) != 0 ||
                            (!stencil_scratches.empty() && IsStencilScratch(image_id)) ||
                            (image.info.num_samples == 2 &&
                             image.info.pixel_format == vk::Format::eR32Uint);
    }
    rec.samplers.resize(stage.samplers.size());
    for (size_t i = 0; i < stage.samplers.size(); ++i) {
        rec.samplers[i] = image_infos[img_begin + image_bindings.size() + i].sampler;
    }
    return u8(buf_key && tex_key) | u8((buf_would && tex_would) << 1);
}

void Rasterizer::BindMemoEndDraw(bool is_compute, u8 draw_bits) {
    auto& st = GetBindMemoStats();
    ++st.binds;
    if (!is_compute) {
        ++st.draws;
        st.draw_key += (draw_bits & 1) != 0;
        st.draw_would += (draw_bits & 2) != 0;
    }
    if (st.binds < (u64{1} << 18)) {
        return;
    }
    const u64 tsc_now = Common::FencedRDTSC();
    const auto clock_now = std::chrono::steady_clock::now();
    const double wall_ns = double(
        std::chrono::duration_cast<std::chrono::nanoseconds>(clock_now - st.clock0).count());
    const double ns_per_tick = tsc_now > st.tsc0 ? wall_ns / double(tsc_now - st.tsc0) : 0.0;
    const u64 frames_now = Common::FaultMonitor::presented_frames.load(std::memory_order_relaxed);
    const u64 frames = frames_now - st.frames0;
    const double per_frame = frames ? 1.0 / double(frames) : 0.0;
    const u64 buf_gen = buffer_cache.StructGeneration();
    const u64 tex_gen = texture_cache.StructGeneration();
    const u64 smp_gen = texture_cache.SamplerGeneration();
    const auto pct = [](u64 part, u64 whole) {
        return whole ? 100.0 * double(part) / double(whole) : 0.0;
    };
    const double empty_ns = st.timed ? double(st.t_empty) * ns_per_tick / double(st.timed) : 0.0;
    const auto avg_ns = [ns_per_tick, empty_ns](u64 ticks, u64 n) {
        return n ? std::max(0.0, double(ticks) * ns_per_tick / double(n) - empty_ns) : 0.0;
    };
    const double full_buf_hit = avg_ns(st.t_buf_hit, st.t_buf_hit_n);
    // Summed per always-full binding inside BindBuffers, so no single-span correction here.
    const double fixed_buf_hit =
        st.t_buf_hit_n ? double(st.t_fixed_hit) * ns_per_tick / double(st.t_buf_hit_n) : 0.0;
    const double full_buf_miss = avg_ns(st.t_buf_miss, st.t_buf_miss_n);
    const double full_tex_hit = avg_ns(st.t_tex_hit, st.t_tex_hit_n);
    const double full_tex_miss = avg_ns(st.t_tex_miss, st.t_tex_miss_n);
    const double key_buf = avg_ns(st.t_key_buf, st.timed);
    const double key_tex = avg_ns(st.t_key_tex, st.timed);
    const double check_buf = avg_ns(st.t_check_buf, st.t_check_buf_n);
    const double check_tex = avg_ns(st.t_check_tex, st.t_check_tex_n);
    const double replay_buf = avg_ns(st.t_replay_buf, st.t_replay_buf_n);
    const double replay_tex = avg_ns(st.t_replay_tex, st.t_replay_tex_n);
    // Per predicted hit: the half's full path minus what a hit still runs (always-full buffer
    // bindings, the checks, the replay, the unreplayed allowances).
    const double allow_buf =
        st.buf_would ? BindMemoAllowBufNs * double(st.memo_bufs_on_hit) / double(st.buf_would)
                     : 0.0;
    const double allow_tex =
        st.tex_would ? BindMemoAllowSamplerNs * double(st.samplers_on_hit) / double(st.tex_would)
                     : 0.0;
    const double save_buf_hit = full_buf_hit - fixed_buf_hit - check_buf - replay_buf - allow_buf;
    const double save_tex_hit = full_tex_hit - check_tex - replay_tex - allow_tex;
    // Paid on every stage (key), on every key hit that fails validation (check) and on every
    // miss (record update, costed like a replay).
    const double gross_buf = double(st.buf_would) * save_buf_hit;
    const double gross_tex = double(st.tex_would) * save_tex_hit;
    const double cost_buf = double(st.stages) * key_buf +
                            double(st.buf_key - st.buf_would) * check_buf +
                            double(st.stages - st.buf_would) * replay_buf;
    const double cost_tex = double(st.stages) * key_tex +
                            double(st.tex_key - st.tex_would) * check_tex +
                            double(st.stages - st.tex_would) * replay_tex;
    const double net_buf_us = (gross_buf - cost_buf) * per_frame / 1000.0;
    const double net_tex_us = (gross_tex - cost_tex) * per_frame / 1000.0;
    const u64 mismatches = st.mis_buf + st.mis_upload + st.mis_img + st.mis_smp;

    LOG_INFO(Render_Vulkan,
             "BindMemoStats: binds={} draws={} frames={} stages={} "
             "stage_key(vs/ps/cs/other)={:.1f}/{:.1f}/{:.1f}/{:.1f}% "
             "stage_would(vs/ps/cs/other)={:.1f}/{:.1f}/{:.1f}/{:.1f}% "
             "whole_draw(key/would)={:.1f}/{:.1f}% slot_key(buf/img/smp)={:.1f}/{:.1f}/{:.1f}% "
             "dma_stages={}",
             st.binds, st.draws, frames, st.stages, pct(st.type_key[0], st.type_stages[0]),
             pct(st.type_key[1], st.type_stages[1]), pct(st.type_key[2], st.type_stages[2]),
             pct(st.type_key[3], st.type_stages[3]), pct(st.type_would[0], st.type_stages[0]),
             pct(st.type_would[1], st.type_stages[1]), pct(st.type_would[2], st.type_stages[2]),
             pct(st.type_would[3], st.type_stages[3]), pct(st.draw_key, st.draws),
             pct(st.draw_would, st.draws), pct(st.slot_buf_key, st.slot_buf),
             pct(st.slot_img_key, st.slot_img), pct(st.slot_smp_key, st.slot_smp),
             st.dma_stages);
    LOG_INFO(Render_Vulkan,
             "BindMemoStats: buf key={:.1f}% would={:.1f}% "
             "miss(excluded/buf_gen/tex_gen/dirty_page)={}/{}/{}/{} tex key={:.1f}% "
             "would={:.1f}% transit_needed={} "
             "miss(excluded/tex_gen/smp_gen/uid/backing/dirty_image/flags)={}/{}/{}/{}/{}/{}/{} "
             "mismatches={} (buf={} upload={} img={} smp={}) layout_diff={} "
             "gens/frame(buf/tex/smp)={:.2f}/{:.3f}/{:.3f}",
             pct(st.buf_key, st.stages), pct(st.buf_would, st.stages), st.buf_miss[BufExcluded],
             st.buf_miss[BufGen], st.buf_miss[BufTexGen], st.buf_miss[BufDirtyPage],
             pct(st.tex_key, st.stages), pct(st.tex_would, st.stages), st.transit_needed,
             st.tex_miss[TexExcluded], st.tex_miss[TexGen], st.tex_miss[TexSamplerGen],
             st.tex_miss[TexUid], st.tex_miss[TexBacking], st.tex_miss[TexDirtyImage],
             st.tex_miss[TexFlags], mismatches, st.mis_buf, st.mis_upload, st.mis_img,
             st.mis_smp, st.layout_diff, double(buf_gen - st.buf_gen0) * per_frame,
             double(tex_gen - st.tex_gen0) * per_frame, double(smp_gen - st.smp_gen0) * per_frame);
    LOG_INFO(Render_Vulkan,
             "BindMemoCost: timed={} tsc_read_ns={:.1f} ns full_buf(hit/miss)={:.0f}/{:.0f} "
             "fixed_buf_hit={:.0f} "
             "full_tex(hit/miss)={:.0f}/{:.0f} key(buf/tex)={:.1f}/{:.1f} "
             "check(buf/tex)={:.1f}/{:.1f} replay(buf/tex)={:.1f}/{:.1f} "
             "save_per_hit(buf/tex)={:.0f}/{:.0f} gross_us_frame(buf/tex)={:.0f}/{:.0f} "
             "est_save_us_frame(buf/tex/total)={:.0f}/{:.0f}/{:.0f}",
             st.timed, empty_ns, full_buf_hit, full_buf_miss, fixed_buf_hit, full_tex_hit,
             full_tex_miss,
             key_buf, key_tex, check_buf, check_tex, replay_buf, replay_tex, save_buf_hit,
             save_tex_hit, gross_buf * per_frame / 1000.0, gross_tex * per_frame / 1000.0,
             net_buf_us, net_tex_us, net_buf_us + net_tex_us);

    // Cumulative-and-reset: the next window starts here. Records are kept.
    auto records = std::move(st.records);
    const u64 stage_counter = st.stage_counter;
    st = {};
    st.records = std::move(records);
    st.stage_counter = stage_counter;
    st.started = true;
    st.frames0 = frames_now;
    st.buf_gen0 = buf_gen;
    st.tex_gen0 = tex_gen;
    st.smp_gen0 = smp_gen;
    st.clock0 = clock_now;
    st.tsc0 = tsc_now;
}

// SHADPS4_TEX_BIND_MEMO: the texture half of the binding memo (notes/binding-memo-design-
// 2026-09-25.md sections 3-4). Key and validity checks are the stats mode's above (measured
// 2026-09-25: texture key hit 94.5%, 0 mismatches), plus a live depth-association check and a
// perfect-match requirement at record time. One record per (pipeline, logical stage), owned by
// the Pipeline. Binding thread only.
struct TexMemoRecord {
    struct Img {
        VideoCore::ImageId id{};
        u64 uid{};
        const VideoCore::Image::BackingImage* backing{};
        vk::ImageView view{};
        vk::DescriptorType type{};
        // Backing state after the recorded full path. With a full-range view and no subresource
        // runs this state is a fixed point of the stage's Transit calls (same layout, access and
        // stage every time), so a hit skips Transit only when the live state equals it.
        VideoCore::Image::State state{};
        VideoCore::SubresourceRange range{}; // the range the full path passed to Transit
        bool full_range{};
    };
    bool valid{};
    bool excluded{};
    u64 border{};
    u64 tex_gen{};
    u64 smp_gen{};
    std::vector<AmdGpu::Image> tsharps;
    std::vector<AmdGpu::Sampler> ssharps;
    std::vector<Img> imgs;
    std::vector<vk::Sampler> samplers;
    std::vector<size_t> sampler_lru;
    u64 last_use{}; ///< SHADPS4_TEX_MEMO_WAYS: LRU stamp of this way.
};

/// SHADPS4_TEX_MEMO_WAYS=N (1-8, default 1): each (pipeline, stage) keeps N records, so a shader
/// that alternates between materials hits on each of them instead of missing on the key.
struct TexMemoSet {
    static constexpr u32 MaxWays = 8;
    std::array<TexMemoRecord, MaxWays> ways{};
};

static u32 TexMemoWays() {
    static const u32 ways = [] {
        const char* value = std::getenv("SHADPS4_TEX_MEMO_WAYS");
        const u32 v = value ? static_cast<u32>(std::strtoul(value, nullptr, 10)) : 1u;
        return std::clamp<u32>(v, 1u, TexMemoSet::MaxWays);
    }();
    return ways;
}

namespace {

enum TexMemoMiss : u32 {
    TmKey,
    TmExcluded,
    TmTexGen,
    TmSmpGen,
    TmUid,
    TmBacking,
    TmDirtyImage,
    TmFlags,
    TmAssoc,
    TmRefused,
    NumTexMemoMiss
};

struct TexMemoState {
    // Key under test; capacity is kept.
    std::vector<AmdGpu::Image> tsharps;
    std::vector<AmdGpu::Sampler> ssharps;
    // Verify scratch: a hit's outputs, captured before they are rolled back.
    struct Write {
        u32 binding;
        u32 count;
        vk::DescriptorType type;
        const vk::DescriptorImageInfo* info;
    };
    std::vector<vk::DescriptorImageInfo> hit_infos;
    std::vector<Write> hit_writes;
    std::vector<VideoCore::ImageId> hit_bound;
    std::vector<std::pair<VideoCore::Image::State, size_t>> hit_states;
    // Window counters (reset at every log line) and run totals.
    bool started{};
    u64 frames0{};
    u64 stages{}, empty_stages{}, key_hits{}, hits{}, hit_transits{}, hit_images{}, hit_samplers{};
    std::array<u64, NumTexMemoMiss> miss{};
    u64 verified{}, mismatched{};
    u64 mis_shape{}, mis_id{}, mis_view{}, mis_layout{}, mis_sampler{}, mis_write{}, mis_bound{},
        mis_state{};
    u64 total_stages{}, total_hits{}, total_verified{}, total_mismatched{};
    u32 logged{};
};

TexMemoState& GetTexMemoState() {
    static TexMemoState state;
    return state;
}

void LogTexMemo(TexMemoState& st, u8 mode) {
    const u64 frames_now = Common::FaultMonitor::presented_frames.load(std::memory_order_relaxed);
    const u64 frames = frames_now - st.frames0;
    const auto pct = [](u64 part, u64 whole) {
        return whole ? 100.0 * double(part) / double(whole) : 0.0;
    };
    const auto per = [](u64 part, u64 whole) { return whole ? double(part) / double(whole) : 0.0; };
    const u64 tex_stages = st.stages - st.empty_stages;
    LOG_INFO(Render_Vulkan,
             "TexBindMemo: mode={} frames={} stages={} tex_stages={} key={:.1f}% hit={:.1f}% "
             "hit_all={:.1f}% hits/frame={:.0f} transits/hit={:.3f} images/hit={:.2f} "
             "samplers/hit={:.2f} miss(key/excluded/tex_gen/smp_gen/uid/backing/dirty_image/"
             "flags/assoc/refused)={}/{}/{}/{}/{}/{}/{}/{}/{}/{} verified={} mismatches={} "
             "(shape={} id={} view={} layout={} sampler={} write={} bound={} state={}) "
             "total(stages/hits/verified/mismatches)={}/{}/{}/{}",
             mode == 2 ? "verify" : "memo", frames, st.stages, tex_stages,
             pct(st.key_hits, tex_stages), pct(st.hits, tex_stages),
             pct(st.hits + st.empty_stages, st.stages), per(st.hits, frames),
             per(st.hit_transits, st.hits), per(st.hit_images, st.hits),
             per(st.hit_samplers, st.hits), st.miss[TmKey], st.miss[TmExcluded],
             st.miss[TmTexGen], st.miss[TmSmpGen], st.miss[TmUid], st.miss[TmBacking],
             st.miss[TmDirtyImage], st.miss[TmFlags], st.miss[TmAssoc], st.miss[TmRefused],
             st.verified, st.mismatched, st.mis_shape, st.mis_id, st.mis_view, st.mis_layout,
             st.mis_sampler, st.mis_write, st.mis_bound, st.mis_state, st.total_stages,
             st.total_hits, st.total_verified, st.total_mismatched);
    st.frames0 = frames_now;
    st.stages = st.empty_stages = st.key_hits = st.hits = st.hit_transits = st.hit_images =
        st.hit_samplers = 0;
    st.miss = {};
    st.verified = st.mismatched = 0;
    st.mis_shape = st.mis_id = st.mis_view = st.mis_layout = st.mis_sampler = st.mis_write =
        st.mis_bound = st.mis_state = 0;
}

} // Anonymous namespace

void Rasterizer::BindTexturesMemo(const Pipeline* pipeline, const Shader::BoundStage& bound,
                                  Shader::Backend::Bindings& binding) {
    const auto& stage = *bound.info;
    auto& st = GetTexMemoState();
    if (!st.started) {
        st.started = true;
        st.frames0 = Common::FaultMonitor::presented_frames.load(std::memory_order_relaxed);
    }
    if (st.stages >= (u64{1} << 18)) {
        LogTexMemo(st, tex_bind_memo);
    }
    ++st.stages;
    ++st.total_stages;
    if (stage.images.empty() && stage.samplers.empty()) {
        // Nothing to skip; BindTextures only walks empty lists.
        ++st.empty_stages;
        BindTextures(bound, binding);
        return;
    }
    auto& slot = pipeline->tex_bind_memo[static_cast<u32>(stage.l_stage)];
    if (!slot) {
        slot = std::make_shared<TexMemoSet>();
    }
    auto& set = *static_cast<TexMemoSet*>(slot.get());
    const u32 ways = TexMemoWays();

    // Key: every T#, every S# after BindTextures' aniso adjustment, the border colour base.
    st.tsharps.clear();
    for (const auto& desc : stage.images) {
        st.tsharps.push_back(desc.GetSharp(bound));
    }
    st.ssharps.clear();
    for (const auto& sampler : stage.samplers) {
        auto ssharp = sampler.GetSharp(bound);
        if (sampler.disable_aniso) {
            const auto& tsharp = stage.images[sampler.associated_image].GetSharp(bound);
            if (tsharp.base_level == 0 && tsharp.last_level == 0) {
                ssharp.max_aniso.Assign(AmdGpu::AnisoRatio::One);
            }
        }
        st.ssharps.push_back(ssharp);
    }
    const u64 border = liverpool->BackRegs().ta_bc_base.base_addr;

    // The way whose key matches, else the least recently used (or an empty) way, which then
    // misses on the key below and is re-recorded by the full path.
    TexMemoRecord* found = nullptr;
    TexMemoRecord* victim = &set.ways[0];
    for (u32 w = 0; w < ways; ++w) {
        auto& cand = set.ways[w];
        if (cand.valid && border == cand.border && SameBytes(st.tsharps, cand.tsharps) &&
            SameBytes(st.ssharps, cand.ssharps)) {
            found = &cand;
            break;
        }
        if (!cand.valid ? victim->valid : (victim->valid && cand.last_use < victim->last_use)) {
            victim = &cand;
        }
    }
    auto& rec = found ? *found : *victim;
    rec.last_use = st.total_stages;

    // Validation (design section 3). Image checks and the LRU touches a hit keeps run inside
    // one memo section, so a foreign structural change cannot interleave with them; the
    // structure generation is re-read inside it for the same reason.
    s32 miss = -1;
    if (!rec.valid || border != rec.border || !SameBytes(st.tsharps, rec.tsharps) ||
        !SameBytes(st.ssharps, rec.ssharps)) {
        miss = TmKey;
    } else if (rec.excluded) {
        miss = TmExcluded;
    } else {
        const auto section = texture_cache.EnterMemoSection();
        if (section == VideoCore::TextureCache::MemoSection::Refused) {
            miss = TmRefused;
        } else {
            if (texture_cache.StructGeneration() != rec.tex_gen) {
                miss = TmTexGen;
            } else {
                for (const auto& img : rec.imgs) {
                    if (!img.id) {
                        continue;
                    }
                    const auto& image = texture_cache.PeekImage(img.id);
                    if (image.image_uid != img.uid) {
                        miss = TmUid;
                    } else if (image.backing != img.backing) {
                        miss = TmBacking;
                    } else if (!texture_cache.ImageUpToDateRelaxed(img.id)) {
                        miss = TmDirtyImage;
                    } else if (image.binding.is_target || image.binding.force_general ||
                               image.binding.needs_rebind) {
                        miss = TmFlags;
                    } else if (image.depth_id) {
                        // AssociateDepth bumps no generation; the full path would redirect.
                        miss = TmAssoc;
                    }
                    if (miss >= 0) {
                        break;
                    }
                }
            }
            if (miss < 0) {
                // FindImage and UpdateImage fast paths: access tick and LRU touch.
                for (const auto& img : rec.imgs) {
                    if (img.id) {
                        texture_cache.TouchImageAccess(texture_cache.PeekImage(img.id));
                    }
                }
            }
            texture_cache.LeaveMemoSection(section);
        }
        // GetSampler's LRU touch, so the sampler GC cannot free a sampler this draw uses.
        if (miss < 0 && !texture_cache.TouchSamplers(rec.sampler_lru, rec.smp_gen)) {
            miss = TmSmpGen;
        }
    }
    st.key_hits += miss != static_cast<s32>(TmKey);
    if (miss >= 0) {
        ++st.miss[static_cast<u32>(miss)];
    }

    if (miss < 0 && tex_bind_memo == 1) {
        st.hit_transits += EmitTexMemoHit(rec, binding);
        ++st.hits;
        ++st.total_hits;
        st.hit_images += rec.imgs.size();
        st.hit_samplers += rec.samplers.size();
        return;
    }

    const size_t img_begin = image_infos.size();
    const u32 sw_begin = set_write_index;
    if (miss < 0) {
        // Verify: run the hit for real, capture its outputs, roll the outputs back and run the
        // full path over the same slots. Binding flags, LRU touches and any Transit the hit made
        // stay; the full path repeats them idempotently when the hit was exact, and redoes
        // whatever the hit skipped wrongly (then caught below as a state mismatch).
        const u32 unified_begin = binding.unified;
        const size_t bound_begin = bound_images.size();
        st.hit_transits += EmitTexMemoHit(rec, binding);
        ++st.hits;
        ++st.total_hits;
        st.hit_images += rec.imgs.size();
        st.hit_samplers += rec.samplers.size();
        st.hit_infos.assign(image_infos.begin() + img_begin, image_infos.end());
        st.hit_writes.clear();
        for (u32 i = sw_begin; i < set_write_index; ++i) {
            const auto& w = set_writes[i];
            st.hit_writes.push_back({w.dstBinding, w.descriptorCount, w.descriptorType,
                                     w.pImageInfo});
        }
        st.hit_bound.assign(bound_images.begin() + bound_begin, bound_images.end());
        const u32 hit_unified = binding.unified;
        st.hit_states.clear();
        for (const auto& img : rec.imgs) {
            if (img.id) {
                const auto& backing = *texture_cache.PeekImage(img.id).backing;
                st.hit_states.emplace_back(backing.state, backing.subresource_runs.size());
            }
        }
        image_infos.erase(image_infos.begin() + img_begin, image_infos.end());
        set_write_index = sw_begin;
        binding.unified = unified_begin;
        bound_images.erase(bound_images.begin() + bound_begin, bound_images.end());

        const u64 tex_gen = texture_cache.StructGeneration();
        const u64 smp_gen = texture_cache.SamplerGeneration();
        bind_memo_redirected = 0;
        bind_memo_rebound = false;
        BindTextures(bound, binding);

        u64 shape = 0, id = 0, view = 0, layout = 0, sampler = 0, write = 0, bound_diff = 0,
            state = 0;
        const size_t num_infos = image_infos.size() - img_begin;
        if (num_infos != st.hit_infos.size() ||
            set_write_index - sw_begin != st.hit_writes.size() || binding.unified != hit_unified ||
            image_bindings.size() != rec.imgs.size()) {
            shape = 1;
        } else {
            for (size_t i = 0; i < num_infos; ++i) {
                const auto& full = image_infos[img_begin + i];
                const auto& hit = st.hit_infos[i];
                view += full.imageView != hit.imageView;
                layout += full.imageLayout != hit.imageLayout;
                sampler += full.sampler != hit.sampler;
            }
            for (size_t i = 0; i < st.hit_writes.size(); ++i) {
                const auto& full = set_writes[sw_begin + i];
                const auto& hit = st.hit_writes[i];
                write += full.dstBinding != hit.binding || full.descriptorCount != hit.count ||
                         full.descriptorType != hit.type || full.pImageInfo != hit.info;
            }
            for (size_t i = 0; i < rec.imgs.size(); ++i) {
                id += image_bindings[i].first != rec.imgs[i].id;
            }
        }
        bound_diff = bound_images.size() - bound_begin != st.hit_bound.size() ||
                     !std::equal(st.hit_bound.begin(), st.hit_bound.end(),
                                 bound_images.begin() + bound_begin);
        size_t k = 0;
        for (const auto& img : rec.imgs) {
            if (!img.id) {
                continue;
            }
            const auto& backing = *texture_cache.PeekImage(img.id).backing;
            state += !(backing.state == st.hit_states[k].first) ||
                     backing.subresource_runs.size() != st.hit_states[k].second;
            ++k;
        }
        ++st.verified;
        ++st.total_verified;
        if (shape | id | view | layout | sampler | write | bound_diff | state) {
            ++st.mismatched;
            ++st.total_mismatched;
            st.mis_shape += shape != 0;
            st.mis_id += id != 0;
            st.mis_view += view != 0;
            st.mis_layout += layout != 0;
            st.mis_sampler += sampler != 0;
            st.mis_write += write != 0;
            st.mis_bound += bound_diff != 0;
            st.mis_state += state != 0;
            if (st.logged < 32) {
                ++st.logged;
                LOG_ERROR(Render_Vulkan,
                          "TexBindMemo verify MISMATCH #{}: pgm={:#x} stage={} images={} "
                          "samplers={} shape={} id={} view={} layout={} sampler={} write={} "
                          "bound={} state={}",
                          st.total_mismatched, stage.pgm_hash, static_cast<u32>(stage.l_stage),
                          rec.imgs.size(), rec.samplers.size(), shape, id, view, layout, sampler,
                          write, bound_diff, state);
            }
            // Re-record from the full path so one hole is not reported on every draw.
            RecordTexMemo(rec, stage, tex_gen, smp_gen, border, img_begin, sw_begin);
        }
        return;
    }

    // Miss: the unchanged full path, then record it. Generations are read first, so a bump
    // during BindTextures makes the next occurrence miss.
    const u64 tex_gen = texture_cache.StructGeneration();
    const u64 smp_gen = texture_cache.SamplerGeneration();
    bind_memo_redirected = 0;
    bind_memo_rebound = false;
    BindTextures(bound, binding);
    RecordTexMemo(rec, stage, tex_gen, smp_gen, border, img_begin, sw_begin);
}

u32 Rasterizer::EmitTexMemoHit(const TexMemoRecord& rec, Shader::Backend::Bindings& binding) {
    // BindTextures pass 1: binding flags. No memoized image is written, so force_general stays.
    for (const auto& img : rec.imgs) {
        if (img.id) {
            texture_cache.PeekImage(img.id).binding.is_bound = 1u;
        }
    }
    // Pass 2: bound_images, state, usage and image infos, in binding order.
    const u32 first_image_idx = image_infos.size();
    u32 transits = 0;
    for (const auto& img : rec.imgs) {
        if (!img.id) {
            image_infos.emplace_back(VK_NULL_HANDLE, VK_NULL_HANDLE, vk::ImageLayout::eGeneral);
            continue;
        }
        bound_images.emplace_back(img.id);
        auto& image = texture_cache.PeekImage(img.id);
        const auto& backing = *image.backing;
        if (!img.full_range || !backing.subresource_runs.empty() || !(backing.state == img.state)) {
            // Exactly the full path's call for a sampled image with no feedback/general flags.
            const auto new_layout = image.info.props.is_depth
                                        ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
                                        : vk::ImageLayout::eShaderReadOnlyOptimal;
            image.Transit(new_layout, vk::AccessFlagBits2::eShaderRead, img.range);
            ++transits;
        }
        image.usage.texture |= 1u;
        image_infos.emplace_back(VK_NULL_HANDLE, img.view, image.backing->state.layout);
    }
    // Descriptor writes: one per image (array size 1; mip-fallback arrays are never memoized),
    // then one per sampler, advancing the unified binding exactly as BindTextures does.
    u32 image_info_idx = first_image_idx;
    for (const auto& img : rec.imgs) {
        if (build_set_writes) {
            auto& set_write = set_writes[set_write_index++];
            set_write.dstSet = VK_NULL_HANDLE;
            set_write.dstBinding = binding.unified;
            set_write.dstArrayElement = 0;
            set_write.descriptorCount = 1;
            set_write.descriptorType = img.type;
            set_write.pImageInfo = &image_infos[image_info_idx];
        }
        ++image_info_idx;
        ++binding.unified;
    }
    for (const auto sampler : rec.samplers) {
        image_infos.emplace_back(sampler, VK_NULL_HANDLE, vk::ImageLayout::eGeneral);
        const u32 unified_binding = binding.unified++;
        if (build_set_writes) {
            auto& set_write = set_writes[set_write_index++];
            set_write.dstSet = VK_NULL_HANDLE;
            set_write.dstBinding = unified_binding;
            set_write.dstArrayElement = 0;
            set_write.descriptorCount = 1;
            set_write.descriptorType = vk::DescriptorType::eSampler;
            set_write.pImageInfo = &image_infos.back();
        }
    }
    return transits;
}

void Rasterizer::RecordTexMemo(TexMemoRecord& rec, const Shader::Info& stage, u64 tex_gen,
                               u64 smp_gen, u64 border, size_t img_begin, u32 sw_begin) {
    const auto& st = GetTexMemoState();
    rec.valid = true;
    rec.tsharps = st.tsharps;
    rec.ssharps = st.ssharps;
    rec.border = border;
    rec.tex_gen = tex_gen;
    rec.smp_gen = smp_gen;
    const size_t num_images = image_bindings.size();
    const size_t num_samplers = stage.samplers.size();
    // Stage-wide exclusions: native bindings, stencil redirects and scratch (Pro paths), images
    // re-found through needs_rebind, mip-fallback arrays, storage images, and any shape that is
    // not one descriptor per image and per sampler.
    rec.excluded = !build_set_writes || native_bindings != nullptr || bind_memo_redirected != 0 ||
                   bind_memo_rebound || num_images != stage.images.size() ||
                   image_infos.size() - img_begin != num_images + num_samplers ||
                   set_write_index - sw_begin != num_images + num_samplers;
    for (const auto& desc : stage.images) {
        rec.excluded |=
            desc.is_written || desc.mip_fallback_mode != Shader::MipStorageFallbackMode::None;
    }
    if (rec.excluded) {
        rec.imgs.clear();
        rec.samplers.clear();
        rec.sampler_lru.clear();
        return;
    }
    rec.imgs.resize(num_images);
    for (size_t i = 0; i < num_images; ++i) {
        const auto& [image_id, desc] = image_bindings[i];
        const auto& info = image_infos[img_begin + i];
        auto& img = rec.imgs[i];
        img = {};
        img.id = image_id;
        img.view = info.imageView;
        img.type = set_writes[sw_begin + i].descriptorType;
        img.range = desc.view_info.range;
        if (!image_id) {
            continue;
        }
        auto& image = texture_cache.PeekImage(image_id);
        img.uid = image.image_uid;
        img.backing = image.backing;
        img.state = image.backing->state;
        img.full_range = img.range.base == VideoCore::SubresourceBase{} &&
                         img.range.extent == image.info.resources;
        // Per-image exclusions: anything but a plain sampled image, a live depth association,
        // stencil scratch, the Pro ID buffer, and bindings FindImage reached through overlap
        // resolution (only perfect matches are stable while the structure generation is).
        rec.excluded |= desc.type != VideoCore::TextureCache::BindingType::Texture ||
                        img.type != vk::DescriptorType::eSampledImage ||
                        static_cast<bool>(image.depth_id) ||
                        (!stencil_scratches.empty() && IsStencilScratch(image_id)) ||
                        (image.info.num_samples == 2 &&
                         image.info.pixel_format == vk::Format::eR32Uint) ||
                        !texture_cache.IsPerfectMatch(desc.info, image_id);
    }
    rec.samplers.resize(num_samplers);
    rec.sampler_lru.resize(num_samplers);
    for (size_t i = 0; i < num_samplers; ++i) {
        rec.samplers[i] = image_infos[img_begin + num_images + i].sampler;
        rec.sampler_lru[i] = texture_cache.SamplerLruId(st.ssharps[i]);
        rec.excluded |= rec.sampler_lru[i] == ~size_t{0} ||
                        set_writes[sw_begin + num_images + i].descriptorType !=
                            vk::DescriptorType::eSampler;
    }
}

void Rasterizer::EmitPipelineResources(const Pipeline* pipeline) {
    if (!native_bindings) {
        pipeline->BindResources(set_writes, buffer_barriers, push_data);
        return;
    }
    // The mode is global, so every guest pipeline here was created with heap mappings. A
    // count mismatch would mean the draw-time binding walk disagrees with pipeline creation.
    ASSERT_MSG(pipeline->UsesNativeBindings() &&
                   native_bound_count == pipeline->NativeBindingCount(),
               "Native binding count {} does not match pipeline {}", native_bound_count,
               pipeline->NativeBindingCount());
    const std::span<const u32> indices{native_indices.data(), native_bound_count};
    if (native_bindings->IsVerifying()) {
        native_bindings->Verify(set_writes, indices);
    }
    pipeline->BindResourcesNative(*native_bindings, indices, buffer_barriers, push_data);
}

bool Rasterizer::IsComputeMetaClear(const Pipeline* pipeline) {
    if (!pipeline->IsCompute()) {
        return false;
    }

    // Most of the time when a metadata is updated with a shader it gets cleared. It means
    // we can skip the whole dispatch and update the tracked state instead. Also, it is not
    // intended to be consumed and in such rare cases (e.g. HTile introspection, CRAA) we
    // will need its full emulation anyways.
    const auto& info = pipeline->GetStage(Shader::LogicalStage::Compute);
    const auto& bound = GetBoundStage(Shader::LogicalStage::Compute);

    // Assume if a shader reads metadata, it is a copy shader.
    for (const auto& desc : info.buffers) {
        const VAddr address = desc.GetSharp(bound).base_address;
        if (!desc.IsSpecial() && !desc.is_written && texture_cache.IsMeta(address)) {
            return false;
        }
    }

    // Metadata surfaces are tiled and thus need address calculation to be written properly.
    // If a shader wants to encode HTILE, for example, from a depth image it will have to compute
    // proper tile address from dispatch invocation id. This address calculation contains an xor
    // operation so use it as a heuristic for metadata writes that are probably not clears.
    if (!info.has_bitwise_xor) {
        // Assume if a shader writes metadata without address calculation, it is a clear shader.
        for (const auto& desc : info.buffers) {
            const VAddr address = desc.GetSharp(bound).base_address;
            if (!desc.IsSpecial() && desc.is_written && texture_cache.ClearMeta(address)) {
                // Assume all slices were updates
                LOG_TRACE(Render_Vulkan, "Metadata update skipped");
                return true;
            }
        }
    }
    return false;
}

bool Rasterizer::IsComputeImageCopy(const Pipeline* pipeline) {
    if (!pipeline->IsCompute()) {
        return false;
    }

    // Ensure shader only has 2 bound buffers
    const auto& cs_pgm = liverpool->BackCsRegs();
    const auto& info = pipeline->GetStage(Shader::LogicalStage::Compute);
    const auto& bound = GetBoundStage(Shader::LogicalStage::Compute);
    if (cs_pgm.num_thread_x.full != 64 || info.buffers.size() != 2 || !info.images.empty()) {
        return false;
    }

    // Those 2 buffers must both be formatted. One must be source and another destination.
    const auto& desc0 = info.buffers[0];
    const auto& desc1 = info.buffers[1];
    if (!desc0.is_formatted || !desc1.is_formatted || desc0.is_written == desc1.is_written) {
        return false;
    }

    // Buffers must have the same size and each thread of the dispatch must copy 1 dword of data
    const AmdGpu::Buffer buf0 = desc0.GetSharp(bound);
    const AmdGpu::Buffer buf1 = desc1.GetSharp(bound);
    if (buf0.GetSize() != buf1.GetSize() || cs_pgm.dim_x != (buf0.GetSize() / 256)) {
        return false;
    }

    // Find images the buffer alias
    const auto image0_id = texture_cache.FindImageFromRange(buf0.base_address, buf0.GetSize());
    if (!image0_id) {
        return false;
    }
    const auto image1_id =
        texture_cache.FindImageFromRange(buf1.base_address, buf1.GetSize(), false);
    if (!image1_id) {
        return false;
    }

    // Image copy must be valid
    VideoCore::Image& image0 = texture_cache.GetImage(image0_id);
    VideoCore::Image& image1 = texture_cache.GetImage(image1_id);
    if (image0.info.guest_size != image1.info.guest_size ||
        image0.info.pitch != image1.info.pitch || image0.info.guest_size != buf0.GetSize() ||
        image0.info.num_bits != image1.info.num_bits) {
        return false;
    }

    // Perform image copy
    VideoCore::Image& src_image = desc0.is_written ? image1 : image0;
    VideoCore::Image& dst_image = desc0.is_written ? image0 : image1;
    if (instance.IsMaintenance8Supported() ||
        src_image.info.props.is_depth == dst_image.info.props.is_depth) {
        dst_image.CopyImage(src_image);
    } else {
        const auto& copy_buffer =
            buffer_cache.GetUtilityBuffer(VideoCore::MemoryUsage::DeviceLocal);
        dst_image.CopyImageWithBuffer(src_image, copy_buffer.Handle(), 0);
    }
    dst_image.flags |= VideoCore::ImageFlagBits::GpuModified;
    dst_image.flags &= ~VideoCore::ImageFlagBits::Dirty;
    return true;
}

bool Rasterizer::IsComputeImageClear(const Pipeline* pipeline) {
    if (!pipeline->IsCompute()) {
        return false;
    }

    // Ensure shader only has 2 bound buffers
    const auto& cs_pgm = liverpool->BackCsRegs();
    const auto& info = pipeline->GetStage(Shader::LogicalStage::Compute);
    const auto& bound = GetBoundStage(Shader::LogicalStage::Compute);
    if (cs_pgm.num_thread_x.full != 64 || info.buffers.size() != 2 || !info.images.empty()) {
        return false;
    }

    // From those 2 buffers, first must hold the clear vector and second the image being cleared
    const auto& desc0 = info.buffers[0];
    const auto& desc1 = info.buffers[1];
    if (desc0.is_formatted || !desc1.is_formatted || desc0.is_written || !desc1.is_written) {
        return false;
    }

    // First buffer must have size of vec4 and second the size of a single layer
    const AmdGpu::Buffer buf0 = desc0.GetSharp(bound);
    const AmdGpu::Buffer buf1 = desc1.GetSharp(bound);
    const u32 buf1_bpp = AmdGpu::NumBitsPerBlock(buf1.GetDataFmt());
    if (buf0.GetSize() != 16 || (cs_pgm.dim_x * 128ULL * (buf1_bpp / 8)) != buf1.GetSize()) {
        return false;
    }

    // Find image the buffer alias
    const auto image1_id =
        texture_cache.FindImageFromRange(buf1.base_address, buf1.GetSize(), false);
    if (!image1_id) {
        return false;
    }

    // Image clear must be valid
    VideoCore::Image& image1 = texture_cache.GetImage(image1_id);
    if (image1.info.guest_size != buf1.GetSize() || image1.info.num_bits != buf1_bpp ||
        image1.info.props.is_depth) {
        return false;
    }

    // Perform image clear
    const float* values = reinterpret_cast<float*>(buf0.base_address);
    const vk::ClearValue clear = {
        .color = {.float32 = std::array<float, 4>{values[0], values[1], values[2], values[3]}},
    };
    const VideoCore::SubresourceRange range = {
        .base =
            {
                .level = 0,
                .layer = 0,
            },
        .extent = image1.info.resources,
    };
    image1.Clear(clear, range);
    image1.flags |= VideoCore::ImageFlagBits::GpuModified;
    image1.flags &= ~VideoCore::ImageFlagBits::Dirty;
    return true;
}

void Rasterizer::BindBuffers(const Shader::BoundStage& bound, Shader::Backend::Bindings& binding,
                             Shader::PushData& push_data) {
    const auto& stage = *bound.info;
    buffer_bindings.clear();
    const u64 struct_gen_before = buffer_cache.StructGeneration();

    // Pass 1 (FindBuffer) vs pass 2 (ObtainBuffer -> SynchronizeBuffer) timed separately: this
    // function is 62% of per-draw cost and a stack scan is not trustworthy for apportioning it.
    const auto t_find = Common::FaultMonitor::Now();
    const u64 bb_find_t0 = g_draw_cost ? __rdtsc() : 0;
    for (const auto& desc : stage.buffers) {
        const auto vsharp = desc.GetSharp(bound);
        if (!desc.IsSpecial() && vsharp.base_address != 0 && vsharp.GetSize() > 0) {
            const u64 size = memory->ClampRangeSize(vsharp.base_address, vsharp.GetSize());
            const auto buffer_id = buffer_cache.FindBuffer(vsharp.base_address, size);
            buffer_bindings.emplace_back(buffer_id, vsharp, size);
        } else {
            buffer_bindings.emplace_back(VideoCore::BufferId{}, vsharp, 0);
        }
    }
    const u64 find_ns = Common::FaultMonitor::SinceNs(t_find);
    if (binding_compute && dispatch_target == 1 && VideoCore::BufferCache::CensusEnabled())
        [[unlikely]] {
        for (u32 i = 0; i < buffer_bindings.size(); ++i) {
            const auto& [buffer_id, vsharp, size] = buffer_bindings[i];
            if (size != 0) {
                buffer_cache.CensusAscBinding(buffer_id, vsharp.base_address, size,
                                              stage.buffers[i].is_written);
            }
        }
        if (VideoCore::BufferCache::RangeShadowEnabled()) {
            // Step 2 shadow: per-Buffer open-batch conflicts vs real range overlaps (RAW: main wrote
            // a range this dispatch touches; WAR: main read a range it writes).
            struct Stats {
                u64 dispatches = 0, per_buffer = 0, range = 0;
                std::array<u64, 256> by_kind{};
                std::unordered_map<u64, u64> by_id;
            };
            static Stats st;
            const u64 now = scheduler.CurrentTick();
            bool pb = false;
            u8 first_kind = 0;
            u64 first_id = 0;
            for (u32 i = 0; i < buffer_bindings.size(); ++i) {
                const auto& [buffer_id, vsharp, size] = buffer_bindings[i];
                if (!buffer_id || size == 0) {
                    continue;
                }
                pb |= buffer_cache.GetBuffer(buffer_id).main_access_tick >= now;
                if (first_kind == 0) {
                    u64 id = 0;
                    first_kind = buffer_cache.RangeConflict(buffer_id, vsharp.base_address, size,
                                                            stage.buffers[i].is_written, now, id);
                    first_id = id;
                }
            }
            ++st.dispatches;
            st.per_buffer += pb;
            if (first_kind != 0) {
                ++st.range;
                ++st.by_kind[first_kind];
                ++st.by_id[(u64(first_kind) << 56) ^ first_id];
            }
            if ((st.dispatches & 0xffff) == 0) {
                std::string kinds;
                for (size_t k = 0; k < st.by_kind.size(); ++k) {
                    if (st.by_kind[k] != 0) {
                        kinds += fmt::format(" k{}={}", k, st.by_kind[k]);
                    }
                }
                std::vector<std::pair<u64, u64>> top(st.by_id.begin(), st.by_id.end());
                std::sort(top.begin(), top.end(),
                          [](const auto& a, const auto& b) { return a.second > b.second; });
                std::string ids;
                for (size_t k = 0; k < std::min<size_t>(top.size(), 8); ++k) {
                    ids += fmt::format(" k{}:{:08x}={}", top[k].first >> 56,
                                       top[k].first & 0x00ffffffffffffffULL, top[k].second);
                }
                LOG_INFO(Render_Vulkan,
                         "AscRangeShadow: candidates={} per_buffer_open_conflicts={} "
                         "range_open_conflicts={} kinds:{} top:{}",
                         st.dispatches, st.per_buffer, st.range, kinds, ids);
            }
        }
    }
    if (binding_compute && scheduler.HasAscStream()) {
        // SHADPS4_ASC_STREAM: the buffers are known now. A candidate goes to the ASC stream only if
        // no buffer it binds was touched by the main batch still being recorded (an ASC segment
        // may only wait on main batches already flushed) and pass 1 created or joined nothing.
        u32 target = dispatch_target;
        u64 wait_main = 0;
        if (target == 1) {
            const u64 now = scheduler.CurrentTick();
            bool ok = buffer_cache.StructGeneration() == struct_gen_before;
            u32 reason = ok ? 0 : 1;
            // SHADPS4_ASC_TEXEL: a read texel binding over a cached image needs an image->buffer
            // copy (images are main-stream only). Checked unconditionally: mode 2's flush must not
            // send such a dispatch to the ASC stream (Opus review 2026-09-28).
            bool texel_block = false;
            for (u32 i = 0; i < buffer_bindings.size(); ++i) {
                const auto& desc = stage.buffers[i];
                const auto& [buffer_id, vsharp, size] = buffer_bindings[i];
                if (desc.is_formatted && !desc.is_written && buffer_id && size != 0 &&
                    texture_cache.FindImageFromRange(vsharp.base_address, size)) {
                    texel_block = true;
                    break;
                }
            }
            static const bool asc_range = VideoCore::BufferCache::AscRangeEnabled();
            for (u32 i = 0; i < buffer_bindings.size(); ++i) {
                if (!ok) {
                    break;
                }
                const auto& [buffer_id, vsharp, size] = buffer_bindings[i];
                if (!buffer_id) {
                    continue;
                }
                const auto& buffer = buffer_cache.GetBuffer(buffer_id);
                if (asc_range) {
                    // SHADPS4_ASC_RANGE: conflict only if the open batch touched this binding's
                    // range (RAW / WAR / upload WAR); otherwise wait on the newest flushed batch
                    // that touched the Buffer (never on the open one: the ASC segment is
                    // submitted before it, I1).
                    buffer_cache.MarkAscBound(buffer_id);
                    u64 id = 0;
                    const u8 kind = buffer_cache.RangeConflict(
                        buffer_id, vsharp.base_address, size, stage.buffers[i].is_written, now, id);
                    ok = kind == 0;
                    if (!ok) {
                        reason = buffer.main_write_tick >= now ? 2 : 3;
                    }
                    // Step 3 (I2 precision): a read binding only depends on flushed main WRITES
                    // (RAW); a written one also on flushed reads (WAR) - the heap is read by every
                    // graphics batch, so access ticks made every segment chase main.
                    // A read binding over CPU-dirty pages uploads on the ASC stream (a write):
                    // it keeps the access-based wait (WAR vs flushed main reads).
                    const bool written =
                        stage.buffers[i].is_written ||
                        buffer_cache.IsRegionCpuModified(vsharp.base_address, size);
                    const u64 last = written ? buffer.main_access_tick : buffer.main_write_tick;
                    const u64 prev = written ? buffer.prev_access_tick : buffer.prev_write_tick;
                    const u64 flushed = last < now ? last : prev;
                    wait_main = std::max(wait_main, flushed);
                    if (VideoCore::BufferCache::CensusEnabled()) [[unlikely]] {
                        // Would range precision drop this flushed-batch wait? (I2 census)
                        static u64 waits = 0, logged = 0, avoidable = 0;
                        if (flushed != 0) {
                            ++waits;
                            const int c = buffer.PrevLogConflict(flushed, vsharp.base_address,
                                                                 size, written);
                            logged += c >= 0;
                            avoidable += c == 0;
                            if ((waits & 0xfffff) == 0) {
                                LOG_INFO(Render_Vulkan,
                                         "AscI2Census: flushed_waits={} with_log={} "
                                         "no_range_overlap={}",
                                         waits, logged, avoidable);
                            }
                        }
                    }
                    continue;
                }
                ok = buffer.main_access_tick < now;
                if (!ok) {
                    reason = buffer.main_write_tick >= now ? 2 : 3;
                }
                wait_main = std::max(wait_main, buffer.main_access_tick);
            }
            if (ok && buffer_cache.asc_indirect_size != 0) {
                const auto args_id = buffer_cache.FindBuffer(buffer_cache.asc_indirect_addr,
                                                             buffer_cache.asc_indirect_size);
                ok = buffer_cache.StructGeneration() == struct_gen_before;
                if (ok) {
                    const auto& args = buffer_cache.GetBuffer(args_id);
                    ok = args.main_access_tick < now;
                    wait_main = std::max(wait_main, args.main_access_tick);
                }
                if (!ok && reason == 0) {
                    reason = 4;
                }
            }
            for (const auto& desc : stage.buffers) {
                if (ok && desc.buffer_type == Shader::BufferType::GdsBuffer) {
                    const u64 gds_tick = buffer_cache.GetGdsBuffer()->main_access_tick;
                    ok = gds_tick < now;
                    wait_main = std::max(wait_main, gds_tick);
                    if (!ok) {
                        reason = 5;
                    }
                }
            }
            // SHADPS4_ASC_STREAM=2: instead of keeping the dispatch on main (whose writes then
            // demote every later dispatch of the span: 89% of demotions, run
            // diag-fire-ascstream-20260928-r), flush the open main batch once. Its submission
            // first submits the open ASC segment (I1); the new segment waits on the flushed batch.
            static const bool flush_mode = [] {
                const char* value = std::getenv("SHADPS4_ASC_STREAM");
                return value && value[0] == '2';
            }();
            // A texel read over an image must stay on main: a flush cannot move its copy.
            if (!ok && reason != 1 && !texel_block && flush_mode) {
                const u64 flushed = scheduler.CurrentTick();
                scheduler.Flush();
                wait_main = std::max(wait_main, flushed);
                ok = true;
                ++asc_flushes;
            }
            if (texel_block) {
                ok = false;
                reason = 6;
            }
            if (!ok) {
                target = 0;
                dispatch_target = 0;
                cur_dispatch_run = 0; // Its run state was the ASC stream's.
                ++asc_demoted;
                // Reasons: 1 pass 1 created/joined, 2 binding written by the open main batch,
                // 3 binding only read by it, 4 indirect args, 5 GDS.
                ++asc_demote_reason[std::min<u32>(reason, 6)];
            }
        }
        scheduler.SwitchStream(target);
        if (target == 1) {
            scheduler.AddAscWaitMain(wait_main);
            dispatch_on_asc = true;
        }
    }
    u64 obtain_ns = 0;
    const u64 bb_find_cyc = g_draw_cost ? __rdtsc() - bb_find_t0 : 0;
    const u64 bb_find_each = buffer_bindings.empty() ? 0 : bb_find_cyc / buffer_bindings.size();

    // Second pass to re-bind buffers that were updated after binding
    for (u32 i = 0; i < buffer_bindings.size(); i++) {
        const auto& [buffer_id, vsharp, size] = buffer_bindings[i];
        const auto& desc = stage.buffers[i];
        // SHADPS4_BIND_MEMO_STATS: bindings a memo hit would still run in full are timed.
        const bool memo_fixed = bind_memo_timing && (desc.IsSpecial() || desc.is_written);
        const u64 memo_t0 = memo_fixed ? Common::FencedRDTSC() : 0;
        const bool is_storage = desc.IsStorage(vsharp);
        const u32 alignment =
            is_storage ? instance.StorageMinAlignment() : instance.UniformMinAlignment();
        // The buffer object behind this binding's legacy payload; null for null descriptors.
        const VideoCore::Buffer* bound_buffer = nullptr;
        // Buffer is not from the cache, either a special buffer or unbound.
        if (!buffer_id) {
            if (desc.buffer_type == Shader::BufferType::GdsBuffer) {
                const auto* gds_buf = buffer_cache.GetGdsBuffer();
                gds_buf->NoteMainAccess(scheduler.CurrentTick(), true);
                if (VideoCore::g_record_stream == 0) {
                    gds_buf->CheckMainVsAsc(gds_buf->CpuAddr(), gds_buf->SizeBytes(), true);
                } else if (VideoCore::BufferCache::AscDepsEnabled()) {
                    buffer_cache.pending_asc.push_back({VideoCore::BufferId{}, gds_buf,
                                                        gds_buf->CpuAddr(), gds_buf->SizeBytes(),
                                                        true});
                }
                buffer_infos.emplace_back(gds_buf->Handle(), 0, gds_buf->SizeBytes());
                bound_buffer = gds_buf;
            } else if (desc.buffer_type == Shader::BufferType::Flatbuf) {
                auto& vk_buffer = buffer_cache.GetUtilityBuffer(VideoCore::MemoryUsage::Stream);
                const u32 ubo_size = bound.flattened_ud_buf.size() * sizeof(u32);
                const u64 offset =
                    vk_buffer.Copy(bound.flattened_ud_buf.data(), ubo_size, alignment);
                buffer_infos.emplace_back(vk_buffer.Handle(), offset, ubo_size);
                bound_buffer = &vk_buffer;
            } else if (desc.buffer_type == Shader::BufferType::ClipPlanes) {
                // Permutations compiled without enabled planes never read the buffer, so the
                // declared binding is satisfied with a null descriptor instead of a copy.
                if (liverpool->BackRegs().clipper_control.user_clip_plane_enable == 0) {
                    buffer_infos.emplace_back(VK_NULL_HANDLE, 0, VK_WHOLE_SIZE);
                } else {
                    auto& vk_buffer = buffer_cache.GetUtilityBuffer(VideoCore::MemoryUsage::Stream);
                    std::array<float, AmdGpu::NUM_CLIP_PLANES * 4> planes{};
                    for (u32 i = 0; i < AmdGpu::NUM_CLIP_PLANES; ++i) {
                        const auto& plane = liverpool->BackRegs().clip_user_data[i];
                        planes[i * 4 + 0] = std::bit_cast<float>(plane.data_x);
                        planes[i * 4 + 1] = std::bit_cast<float>(plane.data_y);
                        planes[i * 4 + 2] = std::bit_cast<float>(plane.data_z);
                        planes[i * 4 + 3] = std::bit_cast<float>(plane.data_w);
                    }
                    const u32 ubo_size = static_cast<u32>(sizeof(planes));
                    const u64 offset = vk_buffer.Copy(planes.data(), ubo_size, alignment);
                    buffer_infos.emplace_back(vk_buffer.Handle(), offset, ubo_size);
                    bound_buffer = &vk_buffer;
                }
            } else if (desc.buffer_type == Shader::BufferType::BdaPagetable) {
                const auto* bda_buffer = buffer_cache.GetBdaPageTableBuffer();
                buffer_infos.emplace_back(bda_buffer->Handle(), 0, bda_buffer->SizeBytes());
                bound_buffer = bda_buffer;
            } else if (desc.buffer_type == Shader::BufferType::FaultBuffer) {
                const auto* fault_buffer = buffer_cache.GetFaultBuffer();
                buffer_infos.emplace_back(fault_buffer->Handle(), 0, fault_buffer->SizeBytes());
                bound_buffer = fault_buffer;
            } else if (desc.buffer_type == Shader::BufferType::SharedMemory) {
                auto& lds_buffer = buffer_cache.GetUtilityBuffer(VideoCore::MemoryUsage::Stream);
                const auto& cs_program = liverpool->BackCsRegs();
                const auto lds_size = cs_program.SharedMemSize() * cs_program.NumWorkgroups();
                const auto [data, offset] = lds_buffer.Map(lds_size, alignment);
                std::memset(data, 0, lds_size);
                buffer_infos.emplace_back(lds_buffer.Handle(), offset, lds_size);
                bound_buffer = &lds_buffer;
            } else {
                buffer_infos.emplace_back(VK_NULL_HANDLE, 0, VK_WHOLE_SIZE);
            }
        } else {
            const auto t_obtain = Common::FaultMonitor::Now();
            const u64 memo_uploads = bind_memo_active ? buffer_cache.SyncUploadBytesTotal() : 0;
            u64 bb_t0 = 0;
            if (g_draw_cost) {
                VideoCore::DiagSyncWalkCycles() = 0;
                VideoCore::DiagSyncRestCycles() = 0;
                bb_t0 = __rdtsc();
            }
            const auto [vk_buffer, offset] = buffer_cache.ObtainBuffer(
                vsharp.base_address, size, desc.is_written, desc.is_formatted, buffer_id,
                is_storage ? Common::BufferSampleCaller::Storage
                           : Common::BufferSampleCaller::Uniform);
            obtain_ns += Common::FaultMonitor::SinceNs(t_obtain);
            const u64 bb_t1 = g_draw_cost ? __rdtsc() : 0;
            if (bind_memo_active) {
                bind_memo_uploaded[i] = buffer_cache.SyncUploadBytesTotal() != memo_uploads;
            }
            const u32 offset_aligned = Common::AlignDown(offset, alignment);
            const u32 adjust = offset - offset_aligned;
            ASSERT(adjust % 4 == 0);
            push_data.AddOffset(binding.buffer, adjust);
            buffer_infos.emplace_back(vk_buffer->Handle(), offset_aligned, size + adjust);
            bound_buffer = vk_buffer;
            if (auto barrier =
                    vk_buffer->GetBarrier(desc.is_written ? vk::AccessFlagBits2::eShaderWrite
                                                          : vk::AccessFlagBits2::eShaderRead,
                                          vk::PipelineStageFlagBits2::eAllCommands, 0,
                                          cur_dispatch_run)) {
                buffer_barriers.emplace_back(*barrier);
            }
            if (desc.is_written && VideoCore::TextureCache::BufferImageAliasDiag()) {
                texture_cache.DiagBufferWriteOverImages(vsharp.base_address, size,
                                                        desc.is_formatted);
            }
            // SHADPS4_RAW_WRITE_INVALIDATE=1 (port of upstream #5092): raw storage-buffer writes
            // can also make an aliased cached image stale, not only formatted ones.
            const bool raw_write_invalidate = g_raw_write_invalidate;
            if (desc.is_written && (desc.is_formatted || raw_write_invalidate)) {
                texture_cache.InvalidateMemoryFromGPU(vsharp.base_address, size);
            }
            if (g_draw_cost) {
                const u32 w = desc.is_written ? 1 : 0;
                const u64 walk = VideoCore::DiagSyncWalkCycles();
                const u64 rest = VideoCore::DiagSyncRestCycles();
                const u64 obtain = bb_t1 - bb_t0;
                g_bb_cyc[w][0] += bb_find_each;
                g_bb_cyc[w][1] += walk;
                g_bb_cyc[w][2] += rest;
                g_bb_cyc[w][3] += obtain > walk + rest ? obtain - walk - rest : 0;
                g_bb_cyc[w][4] += __rdtsc() - bb_t1;
                ++g_bb_n[w];
                if (!w) {
                    NoteReadBindingRepeat(scheduler.CurrentTick(), buffer_id.index,
                                          vsharp.base_address, size);
                }
            }
        }
        if (g_draw_cost && !buffer_id) {
            ++g_bb_special_n;
        }

        const u32 unified_binding = binding.unified++;
        if (native_bindings) {
            // Encode exactly the handle generation, offset and range recorded just above. A
            // later merge leaves this payload alone in the legacy path, and so does this.
            const auto& info = buffer_infos.back();
            native_indices[unified_binding] = native_bindings->BufferIndex(
                info.buffer, bound_buffer ? bound_buffer->BufferDeviceAddress() : 0, info.offset,
                info.range, is_storage);
        }
        if (build_set_writes) {
            auto& set_write = set_writes[set_write_index++];
            set_write.dstSet = VK_NULL_HANDLE;
            set_write.dstBinding = unified_binding;
            set_write.dstArrayElement = 0;
            set_write.descriptorCount = 1;
            set_write.descriptorType = is_storage ? vk::DescriptorType::eStorageBuffer
                                                  : vk::DescriptorType::eUniformBuffer;
            set_write.pBufferInfo = &buffer_infos.back();
        }
        ++binding.buffer;
        if (memo_fixed) {
            bind_memo_fixed_ticks += Common::FencedRDTSC() - memo_t0;
        }
    }
    Common::FaultMonitor::CountBindBufferPass(find_ns, obtain_ns);
}

VideoCore::ImageId Rasterizer::StencilScratchFor(VideoCore::ImageId depth_id,
                                                 const VideoCore::ImageInfo& info) {
    const u64 depth_uid = texture_cache.GetImage(depth_id).image_uid;
    for (auto& entry : stencil_scratches) {
        if (entry.depth_id == depth_id && entry.depth_uid == depth_uid) {
            return entry.scratch_id;
        }
    }
    // The depth image can be recreated at the same address (same surface): reuse the scratch.
    for (auto& entry : stencil_scratches) {
        const auto& scratch = texture_cache.GetImage(entry.scratch_id);
        if (scratch.info.guest_address == info.guest_address &&
            scratch.info.size == info.size && scratch.info.pixel_format == info.pixel_format) {
            entry.depth_id = depth_id;
            entry.depth_uid = depth_uid;
            return entry.scratch_id;
        }
    }
    const VideoCore::ImageId scratch_id = texture_cache.InsertScratchImage(info);
    stencil_scratches.push_back({depth_id, depth_uid, scratch_id});
    const auto& depth = texture_cache.GetImage(depth_id);
    const auto& scratch = texture_cache.GetImage(scratch_id);
    LOG_WARNING(Render_Vulkan,
                "StencilStorage: scratch {} ({}x{} {}) for depth {:#x} ({}x{} {}, {} samples)",
                scratch_id.index, scratch.info.size.width, scratch.info.size.height,
                vk::to_string(scratch.info.pixel_format), depth.info.guest_address,
                depth.info.size.width, depth.info.size.height,
                vk::to_string(depth.info.pixel_format), depth.info.num_samples);
    return scratch_id;
}

bool Rasterizer::IsStencilScratch(VideoCore::ImageId id) const {
    return std::ranges::any_of(stencil_scratches,
                               [id](const auto& entry) { return entry.scratch_id == id; });
}

void Rasterizer::CopyStencilPlane(VideoCore::ImageId src_id, VideoCore::ImageId dst_id) {
    auto& src = texture_cache.GetImage(src_id);
    auto& dst = texture_cache.GetImage(dst_id);
    const auto aspect_of = [](const VideoCore::Image& image) {
        return (image.aspect_mask & vk::ImageAspectFlagBits::eStencil)
                   ? vk::ImageAspectFlagBits::eStencil
                   : vk::ImageAspectFlagBits::eColor;
    };
    const vk::ImageCopy region{
        .srcSubresource = {aspect_of(src), 0, 0, 1},
        .dstSubresource = {aspect_of(dst), 0, 0, 1},
        .extent = {std::min(src.info.size.width, dst.info.size.width),
                   std::min(src.info.size.height, dst.info.size.height), 1},
    };
    scheduler.EndRendering();
    src.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, {});
    dst.Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite, {});
    auto command_access = scheduler.AcquireCommandBuffer();
    command_access.Handle().copyImage(src.GetImage(), src.backing->state.layout, dst.GetImage(),
                                      dst.backing->state.layout, region);
}

void Rasterizer::FlushStencilWritebacks() {
    for (const auto& [scratch_id, depth_id] : pending_stencil_writebacks) {
        CopyStencilPlane(scratch_id, depth_id);
    }
    pending_stencil_writebacks.clear();
}

void Rasterizer::BindTextures(const Shader::BoundStage& bound,
                              Shader::Backend::Bindings& binding) {
    const u64 dc_tex_t0 = g_draw_cost ? __rdtsc() : 0;
    const auto& stage = *bound.info;
    u32 lookup_count{};
    image_bindings.clear();
    const u32 first_image_idx = image_infos.size();
    // For loading/storing to explicit mip levels, when no native instruction support, bind an array
    // of descriptors consecutively, 1 for each mip level. The shader can index this with LOD
    // operand.
    // This array holds the size of each consecutive array with the number of bindings consumed.
    // This is currently always 1 for anything other than mip fallback arrays.
    boost::container::small_vector<u32, 8> image_descriptor_array_sizes;

    const auto t_findimage = Common::FaultMonitor::Now();

    const bool cache_decoded_images = g_cache_decoded_images;
    const bool verify_decoded_images = g_verify_decoded_images;
    for (size_t image_slot = 0; image_slot < stage.images.size(); ++image_slot) {
        const auto& image_desc = stage.images[image_slot];
        const auto tsharp = image_desc.GetSharp(bound);
        // Log-only: a hash lookup per image binding (a19 profile: 11 of 1279 CP samples).
        if (!::Common::Lean() && texture_cache.IsMeta(tsharp.Address())) {
            LOG_WARNING(Render_Vulkan, "Unexpected metadata read by a shader (texture)");
        }

        if (tsharp.Address() == 0 || tsharp.GetDataFmt() == AmdGpu::DataFormat::FormatInvalid) {
            image_bindings.emplace_back(std::piecewise_construct, std::tuple{}, std::tuple{});
            image_descriptor_array_sizes.push_back(1);
            continue;
        }

        const Shader::MipStorageFallbackMode mip_fallback_mode = image_desc.mip_fallback_mode;
        const u32 num_bindings = image_desc.NumBindings(bound);

        const VideoCore::TextureCache::ImageDesc* decoded = nullptr;
        if (cache_decoded_images) {
            auto& entry = decoded_image_descs[static_cast<u32>(stage.l_stage)][image_slot];
            // These are the only ImageResource fields read by the ImageInfo and
            // ImageViewInfo constructors. Neo mode is fixed for the guest process.
            // Cache only the pure descriptor decode, before FindImage modifies its
            // view range. Image IDs, contents, bindings and barriers remain live.
            const u8 flags =
                image_desc.is_depth | (image_desc.is_array << 1) | (image_desc.is_written << 2);
            const bool hit = entry.valid && entry.flags == flags &&
                             std::memcmp(&entry.sharp, &tsharp, sizeof(tsharp)) == 0;
            if (!hit) {
                entry.sharp = tsharp;
                entry.flags = flags;
                entry.value = VideoCore::TextureCache::ImageDesc{tsharp, image_desc};
                entry.valid = true;
            }
            decoded = &entry.value;
            if (verify_decoded_images) {
                const VideoCore::TextureCache::ImageDesc reference{tsharp, image_desc};
                const auto& a = entry.value.info;
                const auto& b = reference.info;
                ASSERT(entry.value.type == reference.type &&
                       entry.value.view_info == reference.view_info &&
                       a.pixel_format == b.pixel_format && a.type == b.type &&
                       a.resources == b.resources && a.size == b.size && a.num_bits == b.num_bits &&
                       a.num_samples == b.num_samples && a.pitch == b.pitch &&
                       a.tile_mode == b.tile_mode && a.array_mode == b.array_mode &&
                       a.guest_address == b.guest_address && a.guest_size == b.guest_size &&
                       a.bank_swizzle == b.bank_swizzle && a.alt_tile == b.alt_tile &&
                       a.props.is_depth == b.props.is_depth &&
                       a.props.is_block == b.props.is_block && a.props.is_pow2 == b.props.is_pow2 &&
                       a.props.is_volume == b.props.is_volume &&
                       a.props.is_tiled == b.props.is_tiled);
                ASSERT(std::memcmp(a.mips_layout.data(), b.mips_layout.data(),
                                   a.resources.levels * sizeof(VideoCore::ImageInfo::MipInfo)) ==
                       0);
                static u64 calls{}, hits{};
                hits += hit;
                if (++calls >= 500000) {
                    LOG_INFO(Render_Vulkan, "DecodeImageVerify: calls={} hits={} ({:.1f}%)", calls,
                             hits, 100.0 * hits / calls);
                    calls = hits = 0;
                }
            }
        }

        for (auto i = 0; i < num_bindings; i++) {
            auto& [image_id, desc] =
                decoded ? image_bindings.emplace_back(VideoCore::ImageId{}, *decoded)
                        : image_bindings.emplace_back(std::piecewise_construct, std::tuple{},
                                                      std::tuple{tsharp, image_desc});

            if (mip_fallback_mode == Shader::MipStorageFallbackMode::ConstantIndex) {
                ASSERT(num_bindings == 1);
                desc.view_info.range.base.level += image_desc.constant_mip_index;
                desc.view_info.range.extent.levels = 1;
            } else if (mip_fallback_mode == Shader::MipStorageFallbackMode::DynamicIndex) {
                desc.view_info.range.base.level += i;
                desc.view_info.range.extent.levels = 1;
            }

            image_id = texture_cache.FindImage(desc);
            ++lookup_count;
            auto* image = &texture_cache.GetImage(image_id);
            {
                // Diagnostic SHADPS4_TRACE_SHADER=<hex pgm hash>: log texture bindings of that
                // shader (first 16 binds): requested T# vs the cache image it resolved to.
                const auto& trace_hashes = g_trace_hashes;
                static u32 traced = 0;
                const u64 trace_hash = stage.pgm_hash;
                if (!trace_hashes.empty() && traced < 48 &&
                    std::ranges::find(trace_hashes, trace_hash) != trace_hashes.end()) {
                    ++traced;
                    LOG_WARNING(Render_Vulkan,
                                "TraceShader {:#x}: slot {} type {} T# {:#x} {}x{} {} -> image "
                                "{:#x} {}x{} {} samples {} uid {} flags {:#x}",
                                trace_hash, image_slot, static_cast<u32>(desc.type),
                                desc.info.guest_address, desc.info.size.width,
                                desc.info.size.height, vk::to_string(desc.info.pixel_format),
                                image->info.guest_address, image->info.size.width,
                                image->info.size.height, vk::to_string(image->info.pixel_format),
                                image->info.num_samples, image->image_uid,
                                static_cast<u32>(image->flags));
                }
            }
            if (auto depth_image_id = texture_cache.GetAssociatedDepth(*image)) {
                // SHADPS4_BIND_MEMO_STATS: redirected bindings are excluded from the memo.
                bind_memo_redirected |= u64{1} << ((image_bindings.size() - 1) & 63);
                static const bool stencil_storage = [] {
                    const char* value = std::getenv("SHADPS4_STENCIL_STORAGE");
                    return value && value[0] == '1';
                }();
                if (stencil_storage &&
                    desc.type == VideoCore::TextureCache::BindingType::Storage) {
                    // The placeholder's association is whichever depth surface bound it last;
                    // prefer the depth image at that address whose extent matches this view.
                    const VAddr depth_addr = texture_cache.GetImage(depth_image_id).info.guest_address;
                    if (const auto sized = texture_cache.FindDepthImageBySize(depth_addr,
                                                                             desc.info.size)) {
                        depth_image_id = sized;
                    }
                    // Vulkan cannot bind a D32S8 stencil aspect as a storage image. Write an R8
                    // scratch seeded from the stencil plane; copied back before the next bind.
                    image_id = StencilScratchFor(depth_image_id, desc.info);
                    image = &texture_cache.GetImage(image_id);
                    const std::pair writeback{image_id, depth_image_id};
                    if (std::ranges::find(pending_stencil_writebacks, writeback) ==
                        pending_stencil_writebacks.end()) {
                        CopyStencilPlane(depth_image_id, image_id);
                        pending_stencil_writebacks.push_back(writeback);
                    }
                } else {
                    // If this image has an associated depth image, it's a stencil attachment.
                    // Redirect the access to the actual depth-stencil buffer.
                    image_id = depth_image_id;
                    image = &texture_cache.GetImage(image_id);
                }
            } else if (image_desc.is_written &&
                       desc.type == VideoCore::TextureCache::BindingType::Storage) {
                texture_cache.NoteStorageWrite(*image);
            }
            if (image->binding.is_bound) {
                // The image is already bound. In case if it is about to be used as storage we
                // need to force general layout on it.
                image->binding.force_general |= image_desc.is_written;
            }
            image->binding.is_bound = 1u;
        }

        image_descriptor_array_sizes.push_back(num_bindings);
    }

    if (Common::BufferSampleProbe::Enabled()) {
        struct LookupShape {
            u64 stages{}, nonempty{}, lookups{};
        };
        static thread_local LookupShape shape;
        ++shape.stages;
        shape.nonempty += lookup_count != 0;
        shape.lookups += lookup_count;
        if ((shape.stages & 0xfffff) == 0) {
            LOG_INFO(Render_Vulkan,
                     "TextureLookupShape: stages={} nonempty={} lookups={} (cumulative)",
                     shape.stages, shape.nonempty, shape.lookups);
        }
    }
    const auto t_view = Common::FaultMonitor::Now();
    u64 findtex_ns = 0;
    u64 transit_ns = 0;
    if (g_draw_cost) {
        g_dc_tex_sub[0] += __rdtsc() - dc_tex_t0;
    }

    // Second pass to re-bind images that were updated after binding
    for (auto& [image_id, desc] : image_bindings) {
        bool is_storage = desc.type == VideoCore::TextureCache::BindingType::Storage;
        if (!image_id) {
            image_infos.emplace_back(VK_NULL_HANDLE, VK_NULL_HANDLE, vk::ImageLayout::eGeneral);
            if (native_bindings) {
                native_image_indices[image_infos.size() - 1] =
                    NativeBindings::NullImageIndex(is_storage);
            }
        } else {
            const bool is_scratch = !stencil_scratches.empty() && IsStencilScratch(image_id);
            if (auto& old_image = texture_cache.GetImage(image_id);
                !is_scratch && old_image.binding.needs_rebind) {
                bind_memo_rebound = true; // SHADPS4_TEX_BIND_MEMO: not memoized
                old_image.binding = {};
                image_id = texture_cache.FindImage(desc);
            }

            bound_images.emplace_back(image_id);

            auto& image = texture_cache.GetImage(image_id);
            const auto t_ft = Common::FaultMonitor::Now();
            const u64 dc_ft0 = g_draw_cost ? __rdtsc() : 0;
            const u32 dc_layout_before =
                g_draw_cost ? static_cast<u32>(image.backing->state.layout) : 0;
            auto& image_view = is_scratch ? image.FindView(desc.view_info)
                                          : texture_cache.FindTexture(image_id, desc);
            const u64 dc_ft1 = g_draw_cost ? __rdtsc() : 0;
            {
                // Experiment SHADPS4_NEO_IDBUF_FILL=<u32>: the PS4 Pro hardware ID buffer (a
                // 2-sample R32Uint surface written by the rasterizer) is not emulated; fill it with
                // a constant before a shader samples it, to test whether it gates the checkerboard
                // resolve. Not a fix.
                const s64 idbuf_fill = g_idbuf_fill;
                const bool idbuf_derive = g_idbuf_derive;
                if (idbuf_derive && !is_storage && image.info.num_samples == 2 &&
                    image.info.pixel_format == vk::Format::eR32Uint) {
                    DeriveNeoIdBuffer(image);
                } else if (idbuf_fill >= 0 && !is_storage && image.info.num_samples == 2 &&
                           image.info.pixel_format == vk::Format::eR32Uint) {
                    scheduler.EndRendering();
                    image.Transit(vk::ImageLayout::eTransferDstOptimal,
                                  vk::AccessFlagBits2::eTransferWrite, {});
                    const u32 fill = static_cast<u32>(idbuf_fill);
                    vk::ClearColorValue value{};
                    value.setUint32({fill, fill, fill, fill});
                    const vk::ImageSubresourceRange range{vk::ImageAspectFlagBits::eColor, 0, 1, 0,
                                                          1};
                    auto command_access = scheduler.AcquireCommandBuffer();
                    command_access.Handle().clearColorImage(image.GetImage(),
                                                            image.backing->state.layout, value,
                                                            range);
                }
            }
            const auto t_tr = Common::FaultMonitor::Now();
            findtex_ns += Common::FaultMonitor::DiffNs(t_ft, t_tr);

            // The image is either bound as storage in a separate descriptor or bound as render
            // target in feedback loop. Depth images are excluded because they can't be bound as
            // storage and feedback loop doesn't make sense for them
            if ((image.binding.force_general || image.binding.is_target) &&
                !image.info.props.is_depth) {
                image.Transit(instance.IsAttachmentFeedbackLoopLayoutSupported() &&
                                      image.binding.is_target
                                  ? vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT
                                  : vk::ImageLayout::eGeneral,
                              vk::AccessFlagBits2::eShaderRead |
                                  (image.info.props.is_depth
                                       ? vk::AccessFlagBits2::eDepthStencilAttachmentWrite
                                       : vk::AccessFlagBits2::eColorAttachmentWrite |
                                             vk::AccessFlagBits2::eColorAttachmentRead),
                              {});
            } else {
                if (is_storage) {
                    image.Transit(vk::ImageLayout::eGeneral,
                                  vk::AccessFlagBits2::eShaderRead |
                                      vk::AccessFlagBits2::eShaderWrite,
                                  desc.view_info.range);
                } else {
                    const auto new_layout = image.info.props.is_depth
                                                ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
                                                : vk::ImageLayout::eShaderReadOnlyOptimal;
                    image.Transit(new_layout, vk::AccessFlagBits2::eShaderRead,
                                  desc.view_info.range);
                }
            }
            transit_ns += Common::FaultMonitor::SinceNs(t_tr);
            if (g_draw_cost) {
                const u64 dc_ft2 = __rdtsc();
                g_dc_tex_sub[1] += dc_ft1 - dc_ft0;
                g_dc_tex_sub[2] += dc_ft2 - dc_ft1;
                if (dc_ft2 - dc_ft0 > g_dc_slow_img.cyc) {
                    g_dc_slow_img = {dc_ft2 - dc_ft0,
                                     image.info.guest_address,
                                     image.info.size.width,
                                     image.info.size.height,
                                     image.info.resources.layers,
                                     static_cast<u32>(image.info.pixel_format),
                                     image.info.props.is_depth != 0,
                                     image.info.props.is_tiled != 0,
                                     dc_layout_before};
                }
            }
            image.usage.storage |= is_storage;
            image.usage.texture |= !is_storage;

            image_infos.emplace_back(VK_NULL_HANDLE, *image_view.image_view,
                                     image.backing->state.layout);
            if (native_bindings) {
                // Same view object and the layout observed at this moment, as recorded above.
                const auto& info = image_infos.back();
                native_image_indices[image_infos.size() - 1] = native_bindings->ImageIndex(
                    image_view.native, info.imageView, info.imageLayout, is_storage);
            }
        }
    }

    Common::FaultMonitor::CountBindTexView(findtex_ns, transit_ns);
    const auto t_sampler_start = Common::FaultMonitor::Now();

    u32 image_info_idx = first_image_idx;
    u32 image_binding_idx = 0;
    for (u32 array_size : image_descriptor_array_sizes) {
        const auto& [_, desc] = image_bindings[image_binding_idx];
        const bool is_storage = desc.type == VideoCore::TextureCache::BindingType::Storage;
        if (native_bindings) {
            for (u32 element = 0; element < array_size; ++element) {
                native_indices[binding.unified + element] =
                    native_image_indices[image_info_idx + element];
            }
        }
        if (build_set_writes) {
            auto& set_write = set_writes[set_write_index++];
            set_write.dstSet = VK_NULL_HANDLE;
            set_write.dstBinding = binding.unified;
            set_write.dstArrayElement = 0;
            set_write.descriptorCount = array_size;
            set_write.descriptorType =
                is_storage ? vk::DescriptorType::eStorageImage : vk::DescriptorType::eSampledImage;
            set_write.pImageInfo = &image_infos[image_info_idx];
        }

        image_info_idx += array_size;
        image_binding_idx += array_size;
        binding.unified += array_size;
    }

    for (const auto& sampler : stage.samplers) {
        auto ssharp = sampler.GetSharp(bound);
        if (sampler.disable_aniso) {
            const auto& tsharp = stage.images[sampler.associated_image].GetSharp(bound);
            if (tsharp.base_level == 0 && tsharp.last_level == 0) {
                ssharp.max_aniso.Assign(AmdGpu::AnisoRatio::One);
            }
        }
        u32 native_sampler_index{};
        const auto vk_sampler =
            texture_cache.GetSampler(ssharp, liverpool->BackRegs().ta_bc_base,
                                     native_bindings ? &native_sampler_index : nullptr);
        image_infos.emplace_back(vk_sampler, VK_NULL_HANDLE, vk::ImageLayout::eGeneral);
        const u32 unified_binding = binding.unified++;
        if (native_bindings) {
            native_indices[unified_binding] = native_sampler_index;
        }
        if (build_set_writes) {
            auto& set_write = set_writes[set_write_index++];
            set_write.dstSet = VK_NULL_HANDLE;
            set_write.dstBinding = unified_binding;
            set_write.dstArrayElement = 0;
            set_write.descriptorCount = 1;
            set_write.descriptorType = vk::DescriptorType::eSampler;
            set_write.pImageInfo = &image_infos.back();
        }
    }

    // The descriptor-write loop between t_view's end and t_sampler_start is bookkeeping only and
    // is folded into the sampler span; if sampler time looks large with a high memo hit rate, that
    // loop is where to look next rather than GetSampler.
    Common::FaultMonitor::CountBindTexSplit(Common::FaultMonitor::DiffNs(t_findimage, t_view),
                                            Common::FaultMonitor::DiffNs(t_view, t_sampler_start),
                                            Common::FaultMonitor::SinceNs(t_sampler_start));
}

RenderState Rasterizer::BeginRendering(const GraphicsPipeline* pipeline) {
    attachment_feedback_loop = false;
    const auto& regs = liverpool->BackRegs();
    const auto& key = pipeline->GetGraphicsKey();
    RenderState state{};
    state.width = instance.GetMaxFramebufferWidth();
    state.height = instance.GetMaxFramebufferHeight();
    state.num_layers = std::numeric_limits<u16>::max();
    state.num_color_attachments = std::bit_width(key.mrt_mask);
    for (auto cb = 0u; cb < state.num_color_attachments; ++cb) {
        auto& [image_id, desc] = cb_descs[cb];
        if (!image_id) {
            state.color_attachments[cb] = {};
            continue;
        }
        auto* image = &texture_cache.GetImage(image_id);
        if (image->binding.needs_rebind) {
            image_id = bound_images.emplace_back(texture_cache.FindImage(desc));
            image = &texture_cache.GetImage(image_id);
        }
        texture_cache.UpdateImage(image_id);
        image->SetBackingSamples(key.color_samples[cb]);
        const auto& image_view = texture_cache.FindRenderTarget(image_id, desc);
        const auto slice = image_view.info.range.base.layer;
        const auto mip = image_view.info.range.base.level;

        const auto& col_buf = regs.color_buffers[cb];
        const bool is_clear = texture_cache.IsMetaCleared(col_buf.CmaskAddress(), slice);
        texture_cache.TouchMeta(col_buf.CmaskAddress(), slice, false);

        if (image->binding.is_bound) {
            ASSERT_MSG(!image->binding.force_general,
                       "Having image both as storage and render target is unsupported");
            image->Transit(instance.IsAttachmentFeedbackLoopLayoutSupported()
                               ? vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT
                               : vk::ImageLayout::eGeneral,
                           vk::AccessFlagBits2::eColorAttachmentWrite, {});
            attachment_feedback_loop = true;
        } else {
            image->Transit(vk::ImageLayout::eColorAttachmentOptimal,
                           vk::AccessFlagBits2::eColorAttachmentWrite |
                               vk::AccessFlagBits2::eColorAttachmentRead,
                           desc.view_info.range);
        }

        state.width = std::min<u32>(state.width, std::max(image->info.size.width >> mip, 1u));
        state.height = std::min<u32>(state.height, std::max(image->info.size.height >> mip, 1u));
        state.num_layers = std::min<u32>(state.num_layers, image_view.info.range.extent.layers);

        const auto clear_value =
            is_clear ? LiverpoolToVK::ColorBufferClearValue(col_buf) : vk::ClearValue{};
        auto& attachment = state.color_attachments[cb];
        attachment.image_view = *image_view.image_view;
        attachment.image_layout = image->backing->state.layout;
        attachment.clear_value = clear_value.color.uint32;
        attachment.is_clear = is_clear;

        image->usage.render_target = 1u;
    }
    for (u32 cb = state.num_color_attachments; cb < state.color_attachments.size(); ++cb) {
        state.color_attachments[cb] = {};
    }

    if (auto image_id = db_desc.first; image_id) {
        auto& desc = db_desc.second;
        const auto htile_address = regs.depth_htile_data_base.GetAddress();
        const auto t_fd = Common::FaultMonitor::Now();
        const auto& image_view = texture_cache.FindDepthTarget(image_id, desc);
        Common::FaultMonitor::CountFindDepthTarget(Common::FaultMonitor::SinceNs(t_fd));
        auto& image = texture_cache.GetImage(image_id);
        if (image.info.num_samples == 2) {
            // Pro scene depth (2 samples): source for SHADPS4_NEO_IDBUF_DERIVE, with the sample
            // locations active while it is rasterized (PA_SC_AA_SAMPLE_LOCS_PIXEL_X0Y0_0).
            neo_depth_addr = image.info.guest_address;
            neo_depth_size = image.info.size;
            neo_depth_x0y0 = regs.reg_array[0xA2FE];
        }

        const auto slice = image_view.info.range.base.layer;
        const bool is_depth_clear =
            (regs.depth_render_control.depth_clear_enable && regs.depth_control.depth_enable &&
             regs.depth_control.depth_write_enable) ||
            texture_cache.IsMetaCleared(htile_address, slice);
        const bool is_stencil_clear = regs.depth_render_control.stencil_clear_enable;
        texture_cache.TouchMeta(htile_address, slice, false);
        ASSERT(desc.view_info.range.extent.levels == 1 && !image.binding.needs_rebind);

        const bool has_stencil = image.info.props.has_stencil;
        // Stencil writes can be enabled while depth writes are off.
        const bool stencil_write =
            has_stencil && regs.depth_control.stencil_enable && !desc.view_info.is_storage;
        const auto new_layout = desc.view_info.is_storage
                                    ? has_stencil ? vk::ImageLayout::eDepthStencilAttachmentOptimal
                                                  : vk::ImageLayout::eDepthAttachmentOptimal
                                : stencil_write
                                    ? vk::ImageLayout::eDepthReadOnlyStencilAttachmentOptimal
                                : has_stencil ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
                                              : vk::ImageLayout::eDepthReadOnlyOptimal;
        image.Transit(new_layout,
                      vk::AccessFlagBits2::eDepthStencilAttachmentWrite |
                          vk::AccessFlagBits2::eDepthStencilAttachmentRead,
                      desc.view_info.range);

        state.width = std::min<u32>(state.width, image.info.size.width);
        state.height = std::min<u32>(state.height, image.info.size.height);
        state.num_layers = std::min<u32>(state.num_layers, image_view.info.range.extent.layers);

        auto& attachment = state.depth_stencil_attachment;
        attachment.image_view = *image_view.image_view;
        attachment.image_layout = image.backing->state.layout;
        attachment.clear_value = {};

        if (regs.depth_buffer.DepthValid()) {
            attachment.clear_value[0] = is_depth_clear ? std::bit_cast<u32>(regs.depth_clear) : 0u;
            attachment.has_depth = true;
            attachment.depth_clear = is_depth_clear;
        }
        if (regs.depth_buffer.StencilValid()) {
            attachment.clear_value[1] = is_stencil_clear ? regs.stencil_clear : 0u;
            attachment.has_stencil = true;
            attachment.stencil_clear = is_stencil_clear;
        } else if (regs.depth_buffer.DepthValid() &&
                   LiverpoolToVK::PromoteDepthToStencil(regs.depth_buffer.z_info.format,
                                                        regs.depth_buffer.stencil_info.format)) {
            // Promoted D32S8: attach the stencil aspect (loaded, never cleared) so the attachment
            // matches the pipeline's stencil format.
            attachment.has_stencil = true;
            attachment.stencil_clear = false;
        }

        image.usage.depth_target = true;
    } else {
        state.depth_stencil_attachment = {};
    }

    if (state.num_layers == std::numeric_limits<u16>::max()) {
        state.num_layers = 1;
    }

    // Diagnostic: a render pass with no usable attachment at all.
    // A null color attachment on its own is normal - MRT gaps, or the guest disabling colour
    // writes - but a pass where every colour attachment is null AND there is no depth/stencil
    // attachment renders into nothing. That is legal Vulkan and raises no validation error, so
    // it is invisible unless we say so here. Deduplicated by signature; each distinct case is
    // reported once per run.
    {
        bool any_color = false;
        for (u32 cb = 0; cb < state.num_color_attachments; ++cb) {
            if (state.color_attachments[cb].image_view) {
                any_color = true;
                break;
            }
        }
        if (!any_color && !state.depth_stencil_attachment.image_view) {
            u64 sig = u64(key.mrt_mask) << 1;
            for (const auto h : key.stage_hashes) {
                sig = sig * 0x100000001b3ULL ^ u64(h);
            }
            static std::unordered_set<u64> reported;
            if (reported.insert(sig).second) {
                const bool cb_disabled =
                    regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Disable;
                std::string detail;
                for (u32 cb = 0; cb < state.num_color_attachments; ++cb) {
                    const auto& col_buf = regs.color_buffers[cb];
                    detail += fmt::format(
                        " [cb{} valid={} addr={:#x} dfmt={} nfmt={} write_mask={:#x} in_mrt={}]",
                        cb, bool(col_buf), col_buf.Address(), u32(col_buf.GetDataFmt()),
                        u32(col_buf.GetNumberFmt()), regs.color_target_mask.GetMask(cb),
                        (key.mrt_mask >> cb) & 1);
                }
                std::string stages;
                for (u32 i = 0; i < key.stage_hashes.size(); ++i) {
                    if (key.stage_hashes[i] != 0) {
                        stages += fmt::format(" stage{}={:#x}", i, key.stage_hashes[i]);
                    }
                }
                // NOTE: this is usually the guest's intent, not a fault - Second Son issues
                // these with color_writes_disabled=true and write_mask=0. Reported so the case
                // stays visible, but do NOT read it as an error on its own. See
                // notes/renderdoc-capture-01-timbers.md.
                LOG_WARNING(Render_Vulkan,
                            "Attachment-less draw (usually guest intent, see notes): "
                            "mrt_mask={:#x} "
                            "color_writes_disabled={} depth_en={} depth_valid={} stencil_en={} "
                            "stencil_valid={}{} shaders:{}",
                            key.mrt_mask, cb_disabled, bool(regs.depth_control.depth_enable),
                            regs.depth_buffer.DepthValid(), bool(regs.depth_control.stencil_enable),
                            regs.depth_buffer.StencilValid(), detail, stages);
            }
        }
    }

    return state;
}

void Rasterizer::Resolve() {
    const auto& mrt0_hint = liverpool->BackCbExtent(0);
    const auto& mrt1_hint = liverpool->BackCbExtent(1);
    VideoCore::TextureCache::ImageDesc mrt0_desc{liverpool->BackRegs().color_buffers[0], mrt0_hint};
    VideoCore::TextureCache::ImageDesc mrt1_desc{liverpool->BackRegs().color_buffers[1], mrt1_hint};
    auto& mrt0_image = texture_cache.GetImage(texture_cache.FindImage(mrt0_desc, true));
    auto& mrt1_image = texture_cache.GetImage(texture_cache.FindImage(mrt1_desc, true));

    ScopeMarkerBegin(fmt::format("Resolve:MRT0={:#x}:MRT1={:#x}",
                                 liverpool->BackRegs().color_buffers[0].Address(),
                                 liverpool->BackRegs().color_buffers[1].Address()));
    mrt1_image.Resolve(mrt0_image, mrt0_desc.view_info.range, mrt1_desc.view_info.range);
    ScopeMarkerEnd();
}

void Rasterizer::DepthStencilCopy(bool is_depth, bool is_stencil) {
    auto& regs = liverpool->BackRegs();

    auto read_desc = VideoCore::TextureCache::ImageDesc(
        regs.depth_buffer, regs.depth_view, regs.depth_control,
        regs.depth_htile_data_base.GetAddress(), liverpool->BackDbExtent(), false);
    auto write_desc = VideoCore::TextureCache::ImageDesc(
        regs.depth_buffer, regs.depth_view, regs.depth_control,
        regs.depth_htile_data_base.GetAddress(), liverpool->BackDbExtent(), true);

    auto& read_image = texture_cache.GetImage(texture_cache.FindImage(read_desc));
    auto& write_image = texture_cache.GetImage(texture_cache.FindImage(write_desc));

    VideoCore::SubresourceRange sub_range;
    sub_range.base.layer = liverpool->BackRegs().depth_view.slice_start;
    sub_range.extent.layers = liverpool->BackRegs().depth_view.NumSlices() - sub_range.base.layer;

    ScopeMarkerBegin(fmt::format(
        "DepthStencilCopy:DR={:#x}:SR={:#x}:DW={:#x}:SW={:#x}", regs.depth_buffer.DepthAddress(),
        regs.depth_buffer.StencilAddress(), regs.depth_buffer.DepthWriteAddress(),
        regs.depth_buffer.StencilWriteAddress()));

    read_image.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead,
                       sub_range);
    write_image.Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite,
                        sub_range);

    auto aspect_mask = vk::ImageAspectFlags(0);
    if (is_depth) {
        aspect_mask |= vk::ImageAspectFlagBits::eDepth;
    }
    if (is_stencil) {
        aspect_mask |= vk::ImageAspectFlagBits::eStencil;
    }

    vk::ImageCopy region = {
        .srcSubresource =
            {
                .aspectMask = aspect_mask,
                .mipLevel = 0,
                .baseArrayLayer = sub_range.base.layer,
                .layerCount = sub_range.extent.layers,
            },
        .srcOffset = {0, 0, 0},
        .dstSubresource =
            {
                .aspectMask = aspect_mask,
                .mipLevel = 0,
                .baseArrayLayer = sub_range.base.layer,
                .layerCount = sub_range.extent.layers,
            },
        .dstOffset = {0, 0, 0},
        .extent = {write_image.info.size.width, write_image.info.size.height, 1},
    };
    scheduler.AcquireCommandBuffer().Handle().copyImage(
        read_image.GetImage(), Vulkan::GuestLayout(vk::ImageLayout::eTransferSrcOptimal),
        write_image.GetImage(), Vulkan::GuestLayout(vk::ImageLayout::eTransferDstOptimal), region);

    ScopeMarkerEnd();
}

void Rasterizer::FillBuffer(VAddr address, u32 num_bytes, u32 value, bool is_gds) {
    if (!is_gds) {
        AmdGpu::SplitHazard::NoteWrite(address, num_bytes);
    }
    buffer_cache.FillBuffer(address, num_bytes, value, is_gds);
}

void Rasterizer::CopyBuffer(VAddr dst, VAddr src, u32 num_bytes, bool dst_gds, bool src_gds) {
    if (!dst_gds) {
        AmdGpu::SplitHazard::NoteWrite(dst, num_bytes);
    }
    buffer_cache.CopyBuffer(dst, src, num_bytes, dst_gds, src_gds);
}

u32 Rasterizer::ReadDataFromGds(u32 gds_offset) {
    auto* gds_buf = buffer_cache.GetGdsBuffer();
    u32 value;
    std::memcpy(&value, gds_buf->mapped_data.data() + gds_offset, sizeof(u32));
    return value;
}

bool Rasterizer::InvalidateMemory(VAddr addr, u64 size, VideoCore::InvalidateOrigin origin) {
    if (!IsMapped(addr, size)) {
        // Not GPU mapped memory, can skip invalidation logic entirely.
        return false;
    }
    // CP split run-ahead: a fault on the front must not download or invalidate ahead of the
    // draws it has posted. No lock is held here. No-op on every other thread.
    liverpool->DrainForFrontFault();
    buffer_cache.InvalidateMemory(addr, size, origin);
    texture_cache.InvalidateMemory(addr, size);
    return true;
}

bool Rasterizer::ReadMemory(VAddr addr, u64 size) {
    if (!IsMapped(addr, size)) {
        // Not GPU mapped memory, can skip invalidation logic entirely.
        return false;
    }
    // CP split run-ahead: see InvalidateMemory. SPLIT-TODO: a drain here waits for posted draws
    // the way today's guest faults wait for the CP's current packet (SendCommand<true>), so it
    // should add no new lock cycle - inferred, not checked at run time.
    liverpool->DrainForFrontFault();
    buffer_cache.ReadMemory(addr, size);
    return true;
}

void Rasterizer::ProcessDownloadImages() {
    texture_cache.ProcessDownloadImages();
}

bool Rasterizer::IsMapped(VAddr addr, u64 size) {
    if (size == 0) {
        // There is no memory, so not mapped.
        return false;
    }
    if (static_cast<u64>(addr) > std::numeric_limits<u64>::max() - size) {
        // Memory range wrapped the address space, cannot be mapped.
        return false;
    }
    const auto range = decltype(mapped_ranges)::interval_type::right_open(addr, addr + size);

    Common::RecursiveSharedLock lock{mapped_ranges_mutex};
    return boost::icl::contains(mapped_ranges, range);
}

void Rasterizer::MapMemory(VAddr addr, u64 size) {
    {
        std::scoped_lock lock{mapped_ranges_mutex};
        mapped_ranges += decltype(mapped_ranges)::interval_type::right_open(addr, addr + size);
    }
    page_manager.OnGpuMap(addr, size);
    buffer_cache.EndGpuMap(addr, size);
}

void Rasterizer::UnmapMemory(VAddr addr, u64 size) {
    buffer_cache.BeginGpuUnmap(addr, size);
    buffer_cache.InvalidateMemory(addr, size);
    texture_cache.UnmapMemory(addr, size);
    page_manager.OnGpuUnmap(addr, size);
    {
        std::scoped_lock lock{mapped_ranges_mutex};
        mapped_ranges -= decltype(mapped_ranges)::interval_type::right_open(addr, addr + size);
    }
    // SHADPS4_BIND_MEMO_STATS: BeginGpuUnmap bumped before the unmap; bump again after it, so a
    // record taken while it was in progress cannot validate against the final mapping.
    buffer_cache.BumpStructGeneration();
}

void Rasterizer::UpdateDynamicState(const GraphicsPipeline* pipeline, const bool is_indexed) const {
    // SHADPS4_VIEWPORT_MEMO=1: viewports and scissors are a pure function of context registers
    // (and device limits), so skip recomputing them while no SET_CONTEXT_REG changed a byte.
    // =2 recomputes anyway and checks the tracked values did not change.
    const int viewport_memo = g_viewport_memo;
    const u64 epoch = liverpool->BackContextEpoch();
    if (viewport_memo == 1 && epoch == viewport_epoch) {
        ++viewport_memo_skips;
    } else if (viewport_memo == 2 && epoch == viewport_epoch) {
        const auto& ds = scheduler.GetDynamicState();
        const auto viewports = ds.viewports;
        const auto scissors = ds.scissors;
        UpdateViewportScissorState();
        const auto& after = scheduler.GetDynamicState();
        const bool same =
            viewports.size() == after.viewports.size() && scissors.size() == after.scissors.size() &&
            std::memcmp(viewports.data(), after.viewports.data(),
                        viewports.size() * sizeof(viewports[0])) == 0 &&
            std::memcmp(scissors.data(), after.scissors.data(),
                        scissors.size() * sizeof(scissors[0])) == 0;
        ++viewport_memo_skips;
        if (!same) {
            ++viewport_memo_mismatches;
        }
        if ((viewport_memo_skips & 0xfffff) == 0) {
            LOG_INFO(Render_Vulkan, "VIEWPORT_MEMO verify: skippable {} mismatches {}",
                     viewport_memo_skips, viewport_memo_mismatches);
        }
    } else {
        UpdateViewportScissorState();
        viewport_epoch = epoch;
    }
    UpdateDepthStencilState();

    // Bounded read-only diagnostic: distinguish PS4 test/operation stencil values
    // before selecting any correction for the captured hair composite failure.
    const bool hair_stencil_trace = g_hair_stencil_trace;
    if (hair_stencil_trace) {
        const auto* fs = pipeline->GetStages()[int(Shader::LogicalStage::Fragment)];
        if (fs && (fs->pgm_hash == 0xe59a987f || fs->pgm_hash == 0x1540dfa9 ||
                   fs->pgm_hash == 0x86bb04f5 || fs->pgm_hash == 0x6f5258e8)) {
            static std::unordered_set<std::string> seen;
            const auto& regs = liverpool->BackRegs();
            const auto state = fmt::format(
                "fs={:#x} control={:#x} ops={:#x} front={:#x} back={:#x} shader={:#x} "
                "render={:#x} stencil_valid={} depth_addr={:#x}",
                fs->pgm_hash, std::bit_cast<u32>(regs.depth_control),
                std::bit_cast<u32>(regs.stencil_control), std::bit_cast<u32>(regs.stencil_ref_front),
                std::bit_cast<u32>(regs.stencil_ref_back),
                std::bit_cast<u32>(regs.depth_shader_control),
                std::bit_cast<u32>(regs.depth_render_control), regs.depth_buffer.StencilValid(),
                regs.depth_buffer.DepthAddress());
            if (seen.size() < 128 && seen.insert(state).second) {
                LOG_INFO(Render_Vulkan, "HairStencil: {}", state);
            }
        }
    }
    UpdatePrimitiveState(is_indexed);
    UpdateRasterizationState();
    UpdateColorBlendingState(pipeline);

    scheduler.CommitDynamicState();
}

void Rasterizer::UpdateViewportScissorState() const {
    const auto& regs = liverpool->BackRegs();

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

    boost::container::static_vector<vk::Viewport, AmdGpu::NUM_VIEWPORTS> viewports;
    boost::container::static_vector<vk::Rect2D, AmdGpu::NUM_VIEWPORTS> scissors;

    if (regs.polygon_control.enable_window_offset &&
        (regs.window_offset.window_x_offset != 0 || regs.window_offset.window_y_offset != 0)) {
        LOG_ERROR(Render_Vulkan,
                  "PA_SU_SC_MODE_CNTL.VTX_WINDOW_OFFSET_ENABLE support is not yet implemented.");
    }

    const auto& vp_ctl = regs.viewport_control;
    for (u32 i = 0; i < AmdGpu::NUM_VIEWPORTS; i++) {
        const auto& vp = regs.viewports[i];
        const auto& vp_d = regs.viewport_depths[i];
        if (vp.xscale == 0) {
            continue;
        }

        const auto zoffset = vp_ctl.zoffset_enable ? vp.zoffset : 0.f;
        const auto zscale = vp_ctl.zscale_enable ? vp.zscale : 1.f;

        vk::Viewport viewport{};

        // https://gitlab.freedesktop.org/mesa/mesa/-/blob/209a0ed/src/amd/vulkan/radv_pipeline_graphics.c#L688-689
        // https://gitlab.freedesktop.org/mesa/mesa/-/blob/209a0ed/src/amd/vulkan/radv_cmd_buffer.c#L3103-3109
        // When the clip space is ranged [-1...1], the zoffset is centered.
        // By reversing the above viewport calculations, we get the following:
        if (regs.clipper_control.clip_space == AmdGpu::ClipSpace::MinusWToW) {
            viewport.minDepth = zoffset - zscale;
            viewport.maxDepth = zoffset + zscale;
        } else {
            viewport.minDepth = zoffset;
            viewport.maxDepth = zoffset + zscale;
        }

        if (!instance.IsDepthRangeUnrestrictedSupported()) {
            // Unrestricted depth range not supported by device. Restrict to valid range.
            viewport.minDepth = std::max(viewport.minDepth, 0.f);
            viewport.maxDepth = std::min(viewport.maxDepth, 1.f);
        }

        if (regs.IsClipDisabled()) {
            // In case if clipping is disabled we patch the shader to convert vertex position
            // from screen space coordinates to NDC by defining a render space as full hardware
            // window range [0..16383, 0..16383] and setting the viewport to its size.
            viewport.x = 0.f;
            viewport.y = 0.f;
            viewport.width = float(std::min<u32>(instance.GetMaxViewportWidth(), 16_KB));
            viewport.height = float(std::min<u32>(instance.GetMaxViewportHeight(), 16_KB));
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
        // Vulkan requires providing at least one viewport.
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

    auto& dynamic_state = scheduler.GetDynamicState();
    dynamic_state.SetViewports(viewports);
    dynamic_state.SetScissors(scissors);
}

void Rasterizer::UpdateDepthStencilState() const {
    const auto& regs = liverpool->BackRegs();
    auto& dynamic_state = scheduler.GetDynamicState();

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
        const bool replace_op_reference = g_replace_op_reference;
        const auto reference = [&](const AmdGpu::StencilRefMask& face, AmdGpu::CompareFunc compare,
                                   AmdGpu::StencilFunc fail, AmdGpu::StencilFunc pass,
                                   AmdGpu::StencilFunc depth_fail) -> u32 {
            // PS4 ReplaceOp writes stencil_op_val, independently of stencil_test_val.
            // Vulkan Replace uses one reference for both. With an unconditional test
            // and Keep for the other outcomes, changing that reference is equivalent.
            // Other combinations and shader-exported references need separate handling.
            if (replace_op_reference && compare == AmdGpu::CompareFunc::Always &&
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
        if (replace_op_reference && (front_reference != front.stencil_test_val ||
                                     back_reference != back.stencil_test_val)) {
            static bool reported{};
            if (!reported) {
                LOG_INFO(Render_Vulkan,
                         "Stencil ReplaceOp correction active: front test={} op={} reference={} "
                         "write_mask={:#x}; back test={} op={} reference={} write_mask={:#x}",
                         front.stencil_test_val, front.stencil_op_val, front_reference,
                         front.stencil_write_mask, back.stencil_test_val, back.stencil_op_val,
                         back_reference, back.stencil_write_mask);
                reported = true;
            }
        }
        dynamic_state.SetStencilReferences(front_reference, back_reference);
        dynamic_state.SetStencilWriteMasks(!stencil_clear ? front.stencil_write_mask : 0U,
                                           !stencil_clear ? back.stencil_write_mask : 0U);
        dynamic_state.SetStencilCompareMasks(front.stencil_mask, back.stencil_mask);
    }
}

void Rasterizer::UpdatePrimitiveState(const bool is_indexed) const {
    const auto& regs = liverpool->BackRegs();
    auto& dynamic_state = scheduler.GetDynamicState();

    const auto is_list_topology = [](const AmdGpu::PrimitiveType type) {
        const auto topology = LiverpoolToVK::PrimitiveType(type);
        return topology == vk::PrimitiveTopology::ePointList ||
               topology == vk::PrimitiveTopology::eLineList ||
               topology == vk::PrimitiveTopology::eTriangleList ||
               topology == vk::PrimitiveTopology::eLineListWithAdjacency ||
               topology == vk::PrimitiveTopology::eTriangleListWithAdjacency;
    };
    const auto is_patch_list_topology = [](const AmdGpu::PrimitiveType type) {
        // Quad and rect lists are emulated using tessellation.
        return type == AmdGpu::PrimitiveType::PatchPrimitive ||
               type == AmdGpu::PrimitiveType::QuadList || type == AmdGpu::PrimitiveType::RectList;
    };

    const auto prim_restart =
        (regs.enable_primitive_restart & 1) != 0 &&
        (instance.IsListRestartSupported() || !is_list_topology(regs.primitive_type)) &&
        (instance.IsPatchListRestartSupported() || !is_patch_list_topology(regs.primitive_type));
    ASSERT_MSG(!is_indexed || !prim_restart || regs.primitive_restart_index == 0xFFFF ||
                   regs.primitive_restart_index == 0xFFFFFFFF,
               "Primitive restart index other than -1 is not supported yet");

    const auto cull_mode = LiverpoolToVK::IsPrimitiveCulled(regs.primitive_type)
                               ? LiverpoolToVK::CullMode(regs.polygon_control.CullingMode())
                               : vk::CullModeFlagBits::eNone;
    const auto front_face = LiverpoolToVK::FrontFace(regs.polygon_control.front_face);

    dynamic_state.SetPrimitiveRestartEnabled(prim_restart);
    dynamic_state.SetRasterizerDiscardEnabled(regs.clipper_control.dx_rasterization_kill);
    dynamic_state.SetCullMode(cull_mode);
    dynamic_state.SetFrontFace(front_face);
}

void Rasterizer::UpdateRasterizationState() const {
    const auto& regs = liverpool->BackRegs();
    auto& dynamic_state = scheduler.GetDynamicState();
    dynamic_state.SetLineWidth(regs.line_control.Width());
}

void Rasterizer::UpdateColorBlendingState(const GraphicsPipeline* pipeline) const {
    const auto& regs = liverpool->BackRegs();
    auto& dynamic_state = scheduler.GetDynamicState();
    dynamic_state.SetBlendConstants(regs.blend_constants);
    dynamic_state.SetColorWriteMasks(pipeline->GetGraphicsKey().write_masks);
    dynamic_state.SetAttachmentFeedbackLoopEnabled(attachment_feedback_loop);
}

void Rasterizer::ScopeMarkerBegin(const std::string_view& str, bool from_guest) {
    if ((from_guest && !EmulatorSettings.IsVkGuestMarkersEnabled()) ||
        (!from_guest && !EmulatorSettings.IsVkHostMarkersEnabled())) {
        return;
    }
    auto command_access = scheduler.AcquireCommandBuffer();
    const auto cmdbuf = command_access.Handle();
    cmdbuf.beginDebugUtilsLabelEXT(vk::DebugUtilsLabelEXT{
        .pLabelName = str.data(),
    });
}

void Rasterizer::ScopeMarkerEnd(bool from_guest) {
    if ((from_guest && !EmulatorSettings.IsVkGuestMarkersEnabled()) ||
        (!from_guest && !EmulatorSettings.IsVkHostMarkersEnabled())) {
        return;
    }
    auto command_access = scheduler.AcquireCommandBuffer();
    const auto cmdbuf = command_access.Handle();
    cmdbuf.endDebugUtilsLabelEXT();
}

void Rasterizer::ScopedMarkerInsert(const std::string_view& str, bool from_guest) {
    if ((from_guest && !EmulatorSettings.IsVkGuestMarkersEnabled()) ||
        (!from_guest && !EmulatorSettings.IsVkHostMarkersEnabled())) {
        return;
    }
    auto command_access = scheduler.AcquireCommandBuffer();
    const auto cmdbuf = command_access.Handle();
    cmdbuf.insertDebugUtilsLabelEXT(vk::DebugUtilsLabelEXT{
        .pLabelName = str.data(),
    });
}

void Rasterizer::ScopedMarkerInsertColor(const std::string_view& str, const u32 color,
                                         bool from_guest) {
    if ((from_guest && !EmulatorSettings.IsVkGuestMarkersEnabled()) ||
        (!from_guest && !EmulatorSettings.IsVkHostMarkersEnabled())) {
        return;
    }
    auto command_access = scheduler.AcquireCommandBuffer();
    const auto cmdbuf = command_access.Handle();
    cmdbuf.insertDebugUtilsLabelEXT(vk::DebugUtilsLabelEXT{
        .pLabelName = str.data(),
        .color = std::array<f32, 4>(
            {(f32)((color >> 16) & 0xff) / 255.0f, (f32)((color >> 8) & 0xff) / 255.0f,
             (f32)(color & 0xff) / 255.0f, (f32)((color >> 24) & 0xff) / 255.0f})});
}

} // namespace Vulkan
