// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// Architecture A, piece 2: the lean worker parser (lean_parse.h). Every handler below is a
// line-for-line mirror of the matching case of Liverpool::ProcessGraphics (liverpool.cpp, the
// type-3 switch) and of Liverpool::EmitDraw / Rasterizer::PrepareDrawFront in split run-ahead
// mode, minus the front's diagnostics. Where the front's path would wait on the back, assert,
// or touch front-only state, the parser stops BEFORE the packet instead (Stop::*), leaving the
// chain state and the channel exactly as the front's would be before that packet.
//
// ---- Thread safety: everything non-local this code reads while the front keeps running -----
//  * ChainState `st`: owned by the caller, handed from the worker of command buffer n to the
//    worker of n + 1 in stream order; one thread at a time. Its register file is a private copy
//    (the front's `regs` is never read). ASC SET_SH_REG writes land in the FRONT's register file
//    (ProcessCompute, liverpool.cpp ~8470), not here: the chain owner must carry them, measured 0
//    while a gfx DCB is open in the city (architecture-a sec. 14, s510).
//  * CpSplit::Channel `out`: the worker's own channel; this thread is its single producer (the
//    channel's producer-side state is unsynchronised by contract). Reserve waits for ring space
//    and PostRunAhead for run-ahead depth exactly as the front's channel does; nothing else waits.
//  * The command buffer: guest memory the guest may not reuse until the submission retires, read
//    by the front the same way. Reads past a packet that the front makes (CB/DB extent NOP hints,
//    the DRAW_INDEX_2 trailing job tag) are bounds-checked against the span here (Stop::Overrun).
//  * Vulkan::PipelineCache::LookupGraphics (piece 1): the ONLY pipeline-cache entry point used.
//    It holds the cache's shared lock for the lookup, reads program_cache / graphics_pipelines
//    and each program's immutable Info half, and writes only the caller's scratch (private Info
//    views, user data and flat buffer copies). Never LookupOrBuildGraphics / PrepareNative* -
//    those write the shared Infos and member state and are left to the front (Stop::PipelineMiss,
//    Stop::NativeUnchecked, Stop::ColourVariantMissing).
//  * GraphicsPipeline fields: GetFetchShader / GetStages / key are immutable after construction.
//    `native_checked` and `native_colour_static` are plain `mutable` bytes the front writes ONCE
//    under the cache's exclusive lock (false -> true, 0 -> 1 or 2); this thread reads them without
//    the lock. A stale read only makes the parser stop where the front would already have moved
//    on (conservative); it can never make it proceed where the front would write. The colour
//    slot keys are atomics (acquire).
//  * Guest memory behind user data: LookupGraphics walks SRT tables (flat buffer) and
//    DeriveVertexInputs reads one V# per fetch attribute through the scratch's user data copy -
//    at parse time, exactly as the front's lookup and FRONT_DERIVE do today (the run-ahead note in
//    RunGfxDma: measured 0 hazards against DMA CPU-path writes, nested tables unmeasured). OPEN
//    RISK for the engineer: a read fault taken here hits the fault handlers, which key their drain
//    on CpSplit::t_is_front; a thread that is neither front nor back is new to them.
//  * Shader::Info immutable half (ud_mask, buffers, images, srt_info, ...): read by the lookup
//    and the derivation; never written after a program is published.
//  * Rasterizer::ClassifyDraw / SkipEmptyDrawRegs / NativeColourVariantReady: pure functions of
//    their arguments plus read-once environment statics.
//  * Globals: VideoCore::g_native_pass / g_native_colour / g_epoch_defer_upload /
//    g_native_coherent, NativeRec::g_mode / g_segb, SplitHazard::Enabled(): inline const or
//    static-init read-once values. LeanViewportMemoEnabled (static bool), LeanKeepsSubmitEpoch
//    (constexpr), LeanNopHasBackEffect (pure over the packet).
//  * NOT touched: any Liverpool member (regs, last_cb/db_extent, context_change_epoch,
//    front_vi_*, cs_state, indirect_args_addr, the posted-write log / span registry), DebugState,
//    NativeRec run state (g_nr_run) and its verify (NativeRecFront / Head / Member), SegB,
//    SplitHazard, the front's counters (regs_generation*, EMPTY_DRAWS, DrawTagFront, FrontCost,
//    DispatchSync) and the program prefetch hint (PrefetchProgramsOnWrite).

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iterator>

#include "common/assert.h"
#include "video_core/amdgpu/lean_front_api.h"
#include "video_core/amdgpu/lean_parse.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/amdgpu/native_rec.h"
#include "video_core/amdgpu/split_hazard.h"
#include "video_core/amdgpu/split_records.h"
#include "video_core/buffer_cache/back_clean.h"
#include "video_core/buffer_cache/submit_epoch.h"
#include "video_core/native_census.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"

namespace AmdGpu::Lean {

namespace {

using Vulkan::Rasterizer;

/// GetStages() of a graphics pipeline spans the logical stages before Compute.
constexpr u32 GraphicsStageCount = Shader::MaxStageTypes - 1;

/// The NOP header the guest places after a CB / DB register block (PM4 NOP, count 0).
constexpr u32 ExtentHintHeader = 0xc0001000u;

bool EnvFlag(const char* name) {
    const char* v = std::getenv(name);
    return v && v[0] == '1';
}

/// Liverpool::JournalField: the field's word offset inside the register file.
template <typename T>
u32 FieldWordOffset(const Regs& regs, const T& field) {
    const auto bytes = reinterpret_cast<const u8*>(&field) - reinterpret_cast<const u8*>(&regs);
    return static_cast<u32>(bytes / sizeof(u32));
}

template <typename T>
void JournalField(CpSplit::Channel& out, const Regs& regs, const T& field) {
    static_assert(sizeof(T) % sizeof(u32) == 0);
    const u32 off = FieldWordOffset(regs, field);
    out.EmitRegWrite(off, &regs.reg_array[off], static_cast<u32>(sizeof(T) / sizeof(u32)));
}

/// Liverpool::JournalExtent (split mode).
void JournalExtent(CpSplit::Channel& out, const ChainState& st, u32 reg_addr) {
    using CR = Liverpool;
    switch (reg_addr) {
    case CR::CbColor0Base:
    case CR::CbColor1Base:
    case CR::CbColor2Base:
    case CR::CbColor3Base:
    case CR::CbColor4Base:
    case CR::CbColor5Base:
    case CR::CbColor6Base:
    case CR::CbColor7Base: {
        const u32 cb = (reg_addr - CR::CbColor0Base) / (CR::CbColor1Base - CR::CbColor0Base);
        out.EmitExtent(cb, st.cb_extent[cb].raw);
        break;
    }
    case CR::CbColor0Cmask:
    case CR::CbColor1Cmask:
    case CR::CbColor2Cmask:
    case CR::CbColor3Cmask:
    case CR::CbColor4Cmask:
    case CR::CbColor5Cmask:
    case CR::CbColor6Cmask:
    case CR::CbColor7Cmask: {
        const u32 cb = (reg_addr - CR::CbColor0Cmask) / (CR::CbColor1Cmask - CR::CbColor0Cmask);
        out.EmitExtent(cb, st.cb_extent[cb].raw);
        break;
    }
    case CR::DbZInfo:
        out.EmitExtent(NUM_COLOR_BUFFERS, st.db_extent.raw);
        break;
    default:
        break;
    }
}

constexpr const char* StopNames[] = {
    "none",          "cut",           "end",       "unsupported",    "overrun",
    "hint-missing",  "bad-set-base",  "nop-back",  "dma-sync",       "dma-guard",
    "pipeline-miss", "native-unchecked", "colour-variant-missing", "config",
};
static_assert(std::size(StopNames) == static_cast<size_t>(Stop::Count));

} // namespace

const char* StopName(Stop stop) {
    const auto i = static_cast<size_t>(stop);
    return i < std::size(StopNames) ? StopNames[i] : "?";
}

/// What DecideDraw settles before anything is emitted (EmitDraw then writes it).
struct Parser::DrawPlan {
    Rasterizer::SplitDraw draw{};
    const char* name{};
    bool has_stages{};
    bool vi_same{}; ///< FRONT_DERIVE: the description equals st.vi_prev (no desc-id bump)
    u32 bytes{};
    u32 vi_bytes{};
    u32 job_tag{};
    u32 job_idx{};
};

Parser::Parser(const Env& env_) : env{env_} {
    // The whole parser is off when the front would take a path this file does not mirror:
    //  - split_verify: EmitDraw's periodic synchronous VERIFY draw and the snapshot block;
    //  - !split_run_ahead (depth 0): every draw record syncs;
    //  - NativeRec::g_segb: SegB::NotePacket on every packet and the segment builder in EmitDraw;
    //  - SHADPS4_DRAW_TAG: g_draw_tag_front writes SplitDrawRecord::pad and front counters;
    //  - SHADPS4_DISPATCH_SYNC_STATS / SHADPS4_GUEST_DISPATCH_ORDER: g_dispatch_sync.Note mutates
    //    per-queue run state on draws (the dispatch records' `run` numbers would diverge);
    //  - SHADPS4_SPLIT_HAZARD_STATS: SplitHazard::NoteDraw keeps front-global page statistics;
    //  - no rasterizer / pipeline cache / split: ProcessGraphics draws inline, no records.
    const bool ok = env.rasterizer && env.pipeline_cache && env.split_enabled &&
                    env.split_run_ahead && !env.split_verify && NativeRec::g_segb == 0 &&
                    !env.draw_tag_front && !EnvFlag("SHADPS4_DISPATCH_SYNC_STATS") &&
                    !EnvFlag("SHADPS4_GUEST_DISPATCH_ORDER") && !SplitHazard::Enabled();
    config_stop = ok ? Stop::None : Stop::Config;
    viewport_memo = LeanViewportMemoEnabled();
    // Liverpool::BumpEpoch in split mode emits a record only under these flags.
    epoch_records = VideoCore::g_epoch_defer_upload || VideoCore::g_native_coherent != 0;
    scratch = std::make_unique<Vulkan::GraphicsLookupScratch>();
    vi = std::make_unique<CpSplit::VertexInputsDerived>();
}

Parser::~Parser() = default;

Stop Parser::DecideDraw(const PM4Header* header, std::span<const u32> dcb, u32 off,
                        const char* name, const CpSplit::DrawArgs& args, const ChainState& st,
                        DrawPlan& plan) {
    // EmitDraw up to (not including) the record, with PrepareDrawFront inlined; the registers
    // already hold this draw packet's writes (as the front's do when EmitDraw runs). Nothing here
    // writes `st` or the channel: a Stop leaves both as they were, and the caller restores the
    // draw packet's register words.
    const Regs& regs = *st.regs;
    plan = {};
    plan.name = name;
    plan.draw.args = args;

    // Rasterizer::PrepareDrawFront.
    plan.draw.draw_class = Rasterizer::ClassifyDraw(regs);
    if (plan.draw.draw_class == Rasterizer::DrawClass::Draw) {
        if (!args.is_indirect) {
            // SplitHazard::NoteDraw: front-global statistics, off by configuration (Config stop).
            plan.draw.skip = Rasterizer::SkipEmptyDrawRegs(regs);
        }
        if (!plan.draw.skip) {
            // Piece 1: read-only, shared-locked, never compiles. Null for every reason the
            // front's GetGraphicsPipeline would return null for AND for every miss it would
            // compile / build on: the front must run that draw.
            plan.draw.pipeline = env.pipeline_cache->LookupGraphics(regs, *scratch);
            if (!plan.draw.pipeline) {
                return Stop::PipelineMiss;
            }
            if (VideoCore::g_native_pass != 0 && !plan.draw.pipeline->native_checked) {
                return Stop::NativeUnchecked; // the front would run PrepareNativeVariant
            }
        }
    }
    plan.has_stages = plan.draw.pipeline &&
                      plan.draw.draw_class == Rasterizer::DrawClass::Draw && !plan.draw.skip;

    plan.bytes = sizeof(SplitDrawRecord);
    if (plan.has_stages) {
        // pipeline->GetStages() in logical-stage order = the scratch's bound stages (piece 1's
        // LookupMatchesFront checks the Info pointers slot by slot).
        for (u32 i = 0; i < GraphicsStageCount; ++i) {
            const auto& slot = scratch->stages[i];
            if (slot.info) {
                plan.bytes += StageBytes(static_cast<u32>(slot.user_data.size()),
                                         static_cast<u32>(slot.flattened_ud_buf.size()));
            }
        }
    }

    // FRONT_DERIVE: the derivation from the scratch's vertex stage (the front builds its
    // BoundStage from the shared Info the lookup just refreshed; same words). The V# reads go
    // through the scratch's user data copy. Not for indirect draws (as EmitDraw).
    if (env.split_front_derive != 0 && plan.has_stages && !args.is_indirect) {
        const auto& vslot = scratch->stages[static_cast<u32>(Shader::LogicalStage::Vertex)];
        Shader::BoundStage vs{};
        vs.info = vslot.info;
        vs.user_data = vslot.user_data;
        vs.flattened_ud_buf = vslot.flattened_ud_buf;
        vs.pgm_base = vslot.pgm_base;
        plan.vi_bytes = plan.draw.pipeline->DeriveVertexInputs(
            *vi, regs.vgt_instance_step_rate_0, regs.vgt_instance_step_rate_1, vs);
        plan.bytes += plan.vi_bytes;
        // The desc-id rule's comparison (applied in EmitDraw, after the decision).
        bool same = vi->header.count == st.vi_prev_count;
        for (u32 i = 0; same && i < vi->header.count; ++i) {
            same = std::memcmp(&vi->attribs[i].desc, &st.vi_prev[i],
                               sizeof(CpSplit::VertexInputDesc)) == 0;
        }
        plan.vi_same = same;
        if (VideoCore::g_native_colour) [[unlikely]] {
            // PrepareNativeColourFront: only when it would return without writing anything.
            if (!Rasterizer::NativeColourVariantReady(plan.draw.pipeline, *vi)) {
                return Stop::ColourVariantMissing;
            }
        }
    }

    // NativeRec >= 2: EmitDraw reads the DRAW_INDEX_2 trailing NOP (the next packet) for the
    // job record of this draw. The front's member path / run state (NativeRecMember, NativeRecHead,
    // g_nr_run) is front-global and NOT reproduced: nr_tag / nr_idx / job_member stay 0.
    if (NativeRec::g_mode >= 2 && header->type3.opcode == PM4ItOpcode::DrawIndex2) {
        const u32* w = reinterpret_cast<const u32*>(header);
        if (off + 8 > dcb.size()) {
            return Stop::Overrun;
        }
        if (w[6] == 0xc0021000u && w[7] == NativeRec::Magic) {
            if (off + 10 > dcb.size()) {
                return Stop::Overrun;
            }
            plan.job_tag = w[8];
            plan.job_idx = w[9];
        }
    }
    return Stop::None;
}

void Parser::EmitDraw(const PM4Header* header, const DrawPlan& plan, ChainState& st,
                      CpSplit::Channel& out) {
    // The record part of Liverpool::EmitDraw for a posted, non-verify, non-member draw.
    auto* rec = out.Reserve<SplitDrawRecord>(CpSplit::RecordType::Draw, plan.bytes);
    rec->draw = plan.draw;
    rec->ctx_epoch = st.ctx_epoch;
    rec->marker = env.host_markers ? plan.name : nullptr;
    rec->cmd_address = header;
    rec->stage_count = 0;
    rec->verify = 0;
    rec->vi_bytes = plan.vi_bytes;
    rec->pad = 0;
    rec->nr_tag = 0;
    rec->nr_idx = 0;
    rec->job_tag = plan.job_tag;
    rec->job_idx = plan.job_idx;
    rec->job_member = 0;
    rec->job_pad = 0;
    rec->segb_id = 0;
    rec->segb_member = 0;
    rec->segb_index_addr = 0;
    rec->segb_index_count = 0;
    rec->segb_instances = 0;
    auto* p = reinterpret_cast<u8*>(rec + 1);
    if (plan.has_stages) {
        for (u32 i = 0; i < GraphicsStageCount; ++i) {
            const auto& slot = scratch->stages[i];
            if (slot.info) {
                p = WriteStage(p, slot.info, slot.pgm_base, slot.user_data, slot.flattened_ud_buf,
                               i);
                ++rec->stage_count;
            }
        }
    }
    if (plan.vi_bytes != 0) {
        // Number the description part: the back skips the setVertexInputEXT of a repeated id.
        if (!plan.vi_same) {
            ++st.vi_desc_id;
            st.vi_prev_count = vi->header.count;
            for (u32 i = 0; i < vi->header.count; ++i) {
                st.vi_prev[i] = vi->attribs[i].desc;
            }
        }
        vi->header.desc_id = st.vi_desc_id;
        p = vi->Write(p);
    }
    out.Commit();
    out.PostRunAhead(CpSplit::PostKind::Draw);
    out.MaybeLog();
}

SpanResult Parser::Parse(std::span<const u32> dcb, u32 cut_dw, ChainState& st,
                         CpSplit::Channel& out) {
    SpanResult r{};
    const auto stop_at = [&](Stop stop, u32 off, u32 opcode) {
        r.stop = stop;
        r.end_dw = off;
        r.stop_opcode = opcode;
        return r;
    };
    if (config_stop != Stop::None) {
        return stop_at(config_stop, 0, 0);
    }
    ASSERT(st.regs);
    Regs& regs = *st.regs;
    const u32 n = static_cast<u32>(dcb.size());
    const u32 limit = std::min(cut_dw, n);
    const u32 num_regs = static_cast<u32>(regs.reg_array.size());
    u32 off = 0;

    for (;;) {
        if (off >= n) {
            return stop_at(Stop::End, n, 0);
        }
        if (off >= limit) {
            return stop_at(Stop::Cut, off, 0);
        }
        const u32* w = dcb.data() + off;
        const auto* header = reinterpret_cast<const PM4Header*>(w);
        if (header->type == 2) {
            // Padding: NextPacket(dcb, 1) and `continue` (no epoch rule).
            ++off;
            ++r.packets;
            continue;
        }
        if (header->type != 3) {
            return stop_at(Stop::Unsupported, off, 0); // type 0 / 1: the front asserts
        }
        const u32 count = header->type3.NumWords();
        const u32 dwords = count + 1;
        const PM4ItOpcode opcode = header->type3.opcode;
        const u32 opcode_u = static_cast<u32>(opcode);
        if (count == 0) {
            // count field 0x3fff: NumWords wraps to 0 and `count - 1` below would wrap too (the
            // front has the same hole); leave such a packet to the front.
            return stop_at(Stop::Unsupported, off, opcode_u);
        }
        if (off + dwords > n) {
            return stop_at(Stop::Overrun, off, opcode_u); // NextPacket's empty-return case
        }
        if (off + dwords > limit) {
            return stop_at(Stop::Cut, off, opcode_u);
        }

        Stop stop = Stop::None;
        switch (opcode) {
        case PM4ItOpcode::Nop: {
            // Flip IRQ / guest markers are back work (RunGfxPacket: a synchronous Packet record);
            // every other NOP (extent hints, job tags, sharp-in-udata notes) is nothing.
            if (LeanNopHasBackEffect(header, env.guest_markers)) {
                stop = Stop::NopBackEffect;
            }
            break;
        }
        case PM4ItOpcode::ContextControl:
        case PM4ItOpcode::AcquireMem: {
            break; // the front ignores both
        }
        case PM4ItOpcode::ClearState: {
            regs.SetDefaults();
            ++st.ctx_epoch;
            out.EmitClearState();
            break;
        }
        case PM4ItOpcode::SetConfigReg: {
            const u32 reg_addr = Regs::ConfigRegWordOffset + (w[1] & 0xffff);
            const u32 words = count - 1;
            if (reg_addr + words > num_regs) {
                stop = Stop::Unsupported;
                break;
            }
            std::memcpy(&regs.reg_array[reg_addr], w + 2, size_t{words} * sizeof(u32));
            out.EmitRegWrite(reg_addr, &regs.reg_array[reg_addr], words);
            break;
        }
        case PM4ItOpcode::SetContextReg: {
            using CR = Liverpool;
            const u32 reg_addr = Regs::ContextRegWordOffset + (w[1] & 0xffff);
            const u32 words = count - 1;
            if (reg_addr + words > num_regs) {
                stop = Stop::Unsupported;
                break;
            }
            // Decide first: the CB / DB extent hint the front reads PAST this packet
            // (payload[k] = w[2 + k] with payload = header + 2): a NOP header at k, the raw
            // extent at k + 1. The front ASSERTs the header word; the parser stops instead.
            u32 hint = ~0u;
            switch (reg_addr) {
            case CR::CbColor0Base:
            case CR::CbColor1Base:
            case CR::CbColor2Base:
            case CR::CbColor3Base:
            case CR::CbColor4Base:
            case CR::CbColor5Base:
            case CR::CbColor6Base:
            case CR::CbColor7Base: {
                const u32 nop_offset = header->type3.count.Value();
                if (nop_offset == 0x0e || nop_offset == 0x0d || nop_offset == 0x0b) {
                    hint = nop_offset;
                }
                break;
            }
            case CR::CbColor0Cmask:
            case CR::CbColor1Cmask:
            case CR::CbColor2Cmask:
            case CR::CbColor3Cmask:
            case CR::CbColor4Cmask:
            case CR::CbColor5Cmask:
            case CR::CbColor6Cmask:
            case CR::CbColor7Cmask: {
                if (header->type3.count == 0x04) {
                    hint = 0x04;
                }
                break;
            }
            case CR::DbZInfo: {
                if (header->type3.count == 8) {
                    hint = 20;
                }
                break;
            }
            default:
                break;
            }
            if (hint != ~0u) {
                if (off + 2 + hint + 2 > n) {
                    stop = Stop::Overrun;
                    break;
                }
                if (w[2 + hint] != ExtentHintHeader) {
                    stop = Stop::HintMissing;
                    break;
                }
            }
            // Apply, in the front's order: epoch on a real change, the words, the journal, the
            // extent, the extent record.
            const size_t ctx_sz = size_t{words} * sizeof(u32);
            if (viewport_memo && std::memcmp(&regs.reg_array[reg_addr], w + 2, ctx_sz) != 0) {
                ++st.ctx_epoch;
            }
            std::memcpy(&regs.reg_array[reg_addr], w + 2, ctx_sz);
            out.EmitRegWrite(reg_addr, &regs.reg_array[reg_addr], words);
            switch (reg_addr) {
            case CR::CbColor0Base:
            case CR::CbColor1Base:
            case CR::CbColor2Base:
            case CR::CbColor3Base:
            case CR::CbColor4Base:
            case CR::CbColor5Base:
            case CR::CbColor6Base:
            case CR::CbColor7Base: {
                const u32 cb = (reg_addr - CR::CbColor0Base) / (CR::CbColor1Base - CR::CbColor0Base);
                st.cb_extent[cb].raw = hint != ~0u ? w[2 + hint + 1] : 0u;
                break;
            }
            case CR::CbColor0Cmask:
            case CR::CbColor1Cmask:
            case CR::CbColor2Cmask:
            case CR::CbColor3Cmask:
            case CR::CbColor4Cmask:
            case CR::CbColor5Cmask:
            case CR::CbColor6Cmask:
            case CR::CbColor7Cmask: {
                if (hint != ~0u) { // else untouched (the front leaves the extent as it was)
                    const u32 cb =
                        (reg_addr - CR::CbColor0Cmask) / (CR::CbColor1Cmask - CR::CbColor0Cmask);
                    st.cb_extent[cb].raw = w[2 + hint + 1];
                }
                break;
            }
            case CR::DbZInfo: {
                st.db_extent.raw = hint != ~0u ? w[2 + hint + 1] : 0u;
                break;
            }
            default:
                break;
            }
            JournalExtent(out, st, reg_addr);
            break;
        }
        case PM4ItOpcode::SetShReg: {
            const u32 reg_off = w[1] & 0xffff;
            const size_t set_size = size_t{count - 1} * sizeof(u32);
            if (reg_off >= 0x200 && reg_off <= (0x200 + sizeof(ComputeProgram) / 4)) {
                // The gfx queue's compute program (dispatch state): no journal, no register
                // words. The front ASSERTs set_size <= sizeof(ComputeProgram) and does not check
                // the end: both become a stop here (the second would write past cs_state).
                const size_t byte_off = size_t{reg_off - 0x200} * sizeof(u32);
                if (set_size > sizeof(ComputeProgram) ||
                    byte_off + set_size > sizeof(ComputeProgram)) {
                    stop = Stop::Unsupported;
                    break;
                }
                std::memcpy(reinterpret_cast<u8*>(&st.cs_state) + byte_off, w + 2, set_size);
            } else {
                const u32 reg_addr = Regs::ShRegWordOffset + reg_off;
                if (reg_addr + (count - 1) > num_regs) {
                    stop = Stop::Unsupported;
                    break;
                }
                std::memcpy(&regs.reg_array[reg_addr], w + 2, set_size);
                out.EmitRegWrite(reg_addr, &regs.reg_array[reg_addr], count - 1);
                // PrefetchProgramsOnWrite (SHADPS4_PGM_PREFETCH): a cache prefetch hint for the
                // front's own next lookup; nothing observable, skipped.
            }
            break;
        }
        case PM4ItOpcode::SetUconfigReg: {
            const u32 reg_addr = Regs::UconfigRegWordOffset + (w[1] & 0xffff);
            const u32 words = count - 1;
            if (reg_addr + words > num_regs) {
                stop = Stop::Unsupported;
                break;
            }
            std::memcpy(&regs.reg_array[reg_addr], w + 2, size_t{words} * sizeof(u32));
            out.EmitRegWrite(reg_addr, &regs.reg_array[reg_addr], words);
            break;
        }
        case PM4ItOpcode::IndexType: {
            const auto* index_type = reinterpret_cast<const PM4CmdDrawIndexType*>(header);
            regs.index_buffer_type.raw = index_type->raw;
            JournalField(out, regs, regs.index_buffer_type);
            break;
        }
        case PM4ItOpcode::NumInstances: {
            const auto* num_instances = reinterpret_cast<const PM4CmdDrawNumInstances*>(header);
            regs.num_instances.num_instances = num_instances->num_instances;
            JournalField(out, regs, regs.num_instances);
            break;
        }
        case PM4ItOpcode::IndexBase: {
            const auto* index_base = reinterpret_cast<const PM4CmdDrawIndexBase*>(header);
            regs.index_base_address.base_addr_lo = index_base->addr_lo;
            regs.index_base_address.base_addr_hi = index_base->addr_hi;
            JournalField(out, regs, regs.index_base_address);
            break;
        }
        case PM4ItOpcode::IndexBufferSize: {
            const auto* index_size = reinterpret_cast<const PM4CmdDrawIndexBufferSize*>(header);
            regs.num_indices = index_size->num_indices;
            JournalField(out, regs, regs.num_indices);
            break;
        }
        case PM4ItOpcode::SetBase: {
            const auto* set_base = reinterpret_cast<const PM4CmdSetBase*>(header);
            if (set_base->base_index.Value() !=
                PM4CmdSetBase::BaseIndex::DrawIndexIndirPatchTable) {
                stop = Stop::BadSetBase; // the front ASSERTs
                break;
            }
            // = set_base->Address<u64>() (no journal: a front-only value for indirect draws).
            st.indirect_args_addr =
                VAddr{set_base->address0} | (VAddr{set_base->address1 & 0xffff} << 32u);
            break;
        }
        case PM4ItOpcode::DmaData: {
            // Only the no-op L2-prefetch DMA (destination 0x3022C; liverpool.cpp A2NoopDma): the
            // back executes its Packet record as a no-op. Front path = RunGfxDma under POST_DMA:
            // a verbatim Packet record, posted. Everything that makes the front wait or log is a
            // stop.
            const auto* dma = reinterpret_cast<const PM4DmaData*>(header);
            if (dma->dst_addr_lo != 0x3022C) {
                stop = Stop::Unsupported;
                break;
            }
            if (!env.split_post_dma) {
                stop = Stop::DmaSync; // RunGfxPacket: synchronous at every depth
                break;
            }
            if (dma->dst_sel == DmaDataDst::Memory || dma->dst_sel == DmaDataDst::MemoryUsingL2) {
                // RunGfxDma's guard against the rest of THIS command buffer (the front's `dcb`
                // span starts at this packet and runs to the buffer's end).
                const VAddr dst = dma->DstAddress<VAddr>();
                const VAddr dst_end = dst + dma->NumBytes();
                const auto lo = reinterpret_cast<VAddr>(w);
                const VAddr hi = lo + size_t{n - off} * sizeof(u32);
                if (dst < hi && lo < dst_end) {
                    stop = Stop::DmaGuard; // the front syncs (PacketDmaGuard)
                    break;
                }
            }
            if (env.split_task_guard != 0 &&
                (dma->dst_sel == DmaDataDst::Memory || dma->dst_sel == DmaDataDst::MemoryUsingL2) &&
                dma->NumBytes() != 0) {
                // NotePostedWrite logs a memory destination in the front's posted-write log; a
                // GDS / register destination is 0 bytes there (NotePostedWrite returns at once).
                stop = Stop::DmaGuard;
                break;
            }
            void* copy = out.Reserve(CpSplit::RecordType::Packet, dwords * sizeof(u32),
                                     env.guest_markers ? u16{CpSplit::GuestMarkers} : u16{0});
            std::memcpy(copy, header, dwords * sizeof(u32));
            out.Commit();
            out.PostRunAhead(CpSplit::PostKind::GfxPacket);
            break;
        }
        case PM4ItOpcode::DrawIndex2: {
            const auto* d = reinterpret_cast<const PM4CmdDrawIndex2*>(header);
            const u32 old_max = regs.max_index_size;
            const auto old_base = regs.index_base_address;
            const u32 old_num = regs.num_indices;
            const u32 old_init = regs.draw_initiator;
            regs.max_index_size = d->max_size;
            regs.index_base_address.base_addr_lo = d->index_base_lo;
            regs.index_base_address.base_addr_hi = d->index_base_hi;
            regs.num_indices = d->index_count;
            regs.draw_initiator = d->draw_initiator;
            DrawPlan plan;
            stop = DecideDraw(header, dcb, off, "DrawIndex2", {.is_indexed = true}, st, plan);
            if (stop != Stop::None) {
                regs.max_index_size = old_max;
                regs.index_base_address = old_base;
                regs.num_indices = old_num;
                regs.draw_initiator = old_init;
                break;
            }
            JournalField(out, regs, regs.max_index_size);
            JournalField(out, regs, regs.index_base_address);
            JournalField(out, regs, regs.num_indices);
            JournalField(out, regs, regs.draw_initiator);
            // DebugState.PushRegsDump (debugger register dumps): not mirrored.
            EmitDraw(header, plan, st, out);
            ++r.draws;
            break;
        }
        case PM4ItOpcode::DrawIndexOffset2: {
            const auto* d = reinterpret_cast<const PM4CmdDrawIndexOffset2*>(header);
            const u32 old_max = regs.max_index_size;
            const u32 old_num = regs.num_indices;
            const u32 old_init = regs.draw_initiator;
            regs.max_index_size = d->max_size;
            regs.num_indices = d->index_count;
            regs.draw_initiator = d->draw_initiator;
            DrawPlan plan;
            stop = DecideDraw(header, dcb, off, "DrawIndexOffset2",
                              {.is_indexed = true, .index_offset = d->index_offset}, st, plan);
            if (stop != Stop::None) {
                regs.max_index_size = old_max;
                regs.num_indices = old_num;
                regs.draw_initiator = old_init;
                break;
            }
            JournalField(out, regs, regs.max_index_size);
            JournalField(out, regs, regs.num_indices);
            JournalField(out, regs, regs.draw_initiator);
            EmitDraw(header, plan, st, out);
            ++r.draws;
            break;
        }
        case PM4ItOpcode::DrawIndexAuto: {
            const auto* d = reinterpret_cast<const PM4CmdDrawIndexAuto*>(header);
            const u32 old_num = regs.num_indices;
            const u32 old_init = regs.draw_initiator;
            regs.num_indices = d->index_count;
            regs.draw_initiator = d->draw_initiator;
            DrawPlan plan;
            stop = DecideDraw(header, dcb, off, "DrawIndexAuto", {}, st, plan);
            if (stop != Stop::None) {
                regs.num_indices = old_num;
                regs.draw_initiator = old_init;
                break;
            }
            JournalField(out, regs, regs.num_indices);
            JournalField(out, regs, regs.draw_initiator);
            EmitDraw(header, plan, st, out);
            ++r.draws;
            break;
        }
        default:
            stop = Stop::Unsupported;
            break;
        }
        if (stop != Stop::None) {
            return stop_at(stop, off, opcode_u);
        }
        // After every packet: Liverpool::BumpEpoch(opcode) unless the opcode keeps the window.
        if (!LeanKeepsSubmitEpoch(opcode) && epoch_records) {
            out.EmitEpochBump(opcode_u);
        }
        off += dwords;
        ++r.packets;
    }
}

} // namespace AmdGpu::Lean
