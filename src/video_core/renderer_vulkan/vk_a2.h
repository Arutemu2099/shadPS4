// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <memory>
#include <vector>

#include "common/types.h"
#include "shader_recompiler/info.h"
#include "video_core/amdgpu/regs.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

/**
 * Architecture A, step A2 (notes/NATIVE-RENDERER.md, notes/architecture-a-2026-10-10.md sec. 6):
 * worker threads translate the NATIVE STRETCHES of the game's top-level command buffers (DCBs)
 * into plain command lists (StretchCmds); the back runs everything else on the old path and
 * replays the lists in stream order (one Scheduler::Record closure per stretch, inside the
 * render pass it began), so the GPU sees the original order.
 *
 * Split of work:
 *  - Liverpool (liverpool.cpp): the DCB table, the worker threads, the record walk with a private
 *    register mirror per worker, the stretch boundaries, the back-side execution of a DCB.
 *  - StretchTranslator (vk_a2.cpp): per draw, decides whether the worker can take it and appends
 *    it to the open stretch's command list. Runs on a WORKER thread: it must not touch the buffer
 *    cache's mutable state, the texture cache, the scheduler (Record / Rc / ticks / sinks), the
 *    stream buffer or any Rasterizer member. Allowed: immutable pipeline data and atomics
 *    (native_variant), the job-record rings (NativeRec, read-only, stamp-checked), the buffer
 *    cache's lock-free lookup (BufferCache::A2LookupBuffer), the worker's own arena, its own
 *    DynamicState, and the master semaphore's known GPU tick (read-only, for arena reuse).
 */
namespace Vulkan {
class GraphicsPipeline;
class Instance;
} // namespace Vulkan
namespace VideoCore {
class BufferCache;
class Buffer;
} // namespace VideoCore
namespace AmdGpu::CpSplit {
struct VertexInputsPayload;
}

namespace Vulkan::A2 {

/// SHADPS4_A2: 0 off (default); 1 workers record native stretches and the back publishes them;
/// 2 verify: workers translate, the back discards every stretch and runs the old path (counts).
int Mode();
/// SHADPS4_A2_WORKERS (default 3, 1-8).
u32 NumWorkers();

/// The draw as the worker loop decoded it from the Draw record (pointers into the CpSplit ring:
/// valid while the worker runs on this DCB).
struct DrawIn {
    const GraphicsPipeline* pipeline{}; ///< the front's lookup (base pipeline)
    bool is_indexed{};
    bool is_indirect{};
    u32 index_offset{};
    const std::array<Shader::BoundStage, Shader::MaxStageTypes>* stages{}; ///< ReadStages output
    const AmdGpu::CpSplit::VertexInputsPayload* vi{}; ///< FRONT_DERIVE block, may be null
    u32 job_tag{}, job_idx{}; ///< the draw's own job record (tag 0: none)
    /// The draw's DRAW_INDEX_2 packet address (the Draw record's cmd_address): the job record's
    /// `cmd` must equal it (NativeRingPrepare's `cmd0 != draw_job_cmd` check). Null: the check
    /// is skipped (stamp + fields only) and counted under RefuseNoCmd once per session.
    const void* job_cmd{};
    u64 seq{};                ///< the Draw record's CpSplit seq
};

/// Per-worker memory the GPU reads (tables, indirect commands, counts): a host-visible coherent
/// ring. Regions are handed out in order; a region is reusable once the GPU passed the tick of the
/// submission batch its stretch was PUBLISHED into (the back stamps Stretch::tick at publish).
struct Arena {
    vk::Buffer buffer{};
    u8* mapped{};
    u64 device_address{};
    u64 size{};
    u64 head{}; ///< next free byte (unmasked)
    /// Reuse fence per ended stretch: [unmasked start, unmasked end, tick] in order, oldest
    /// first. tick 0 = not published yet = not free. The tick object is shared with the Stretch
    /// (the back destroys the Stretch when the DCB is done; the fence keeps the tick alive).
    struct Fence {
        u64 start{};
        u64 end{};
        std::shared_ptr<std::atomic<u64>> tick;
    };
    std::vector<Fence> fences;
};

/// One sub-batch of a stretch: everything one drawIndexedIndirectCount needs, as values. The
/// first sub-batch of a stretch carries the FULL state (pipeline, push constants, descriptors,
/// vertex input, index / vertex buffers, every dynamic state); later ones only what changed
/// (the emit_* flags and the dynamic state's dirty bits say what Replay records).
struct SubBatch {
    bool emit_pipeline{}, emit_push{}, emit_descriptors{}, emit_vertex_input{}, emit_index{},
        emit_vertex_buffers{};
    vk::Pipeline pipeline{};
    vk::PipelineLayout layout{};
    Shader::PushData push{};
    /// Set-0 push-descriptor writes (storage buffers only), in binding order.
    struct Write {
        u32 binding{};
        vk::Buffer buffer{}; ///< VK_NULL_HANDLE: a null descriptor (unused clip planes)
        u64 offset{};
        u64 range{};
    };
    u32 num_writes{};
    std::array<Write, Shader::NUM_BUFFERS + 4> writes{};
    /// setVertexInputEXT arrays (the FRONT_DERIVE block's description part).
    u32 vi_count{};
    std::array<vk::VertexInputAttributeDescription2EXT, 16> vi_attrs{};
    std::array<vk::VertexInputBindingDescription2EXT, 16> vi_bindings{};
    vk::Buffer ib{};
    vk::IndexType ib_type{};
    u32 vb_n{};
    std::array<vk::Buffer, 16> vb{};
    std::array<vk::DeviceSize, 16> vb_off{};
    /// Dynamic state values; its dirty bits are what this sub-batch records (Commit).
    DynamicState dyn{};
    /// The indirect draw: commands and count in the arena buffer.
    vk::Buffer arena{};
    u64 cmd_off{};
    u64 count_off{};
    u32 n{}; ///< draws (maxDrawCount of the indirect count draw)
};

struct StretchCmds {
    std::vector<SubBatch> batches;
};

/// Records `cmds` into `cmdbuf` exactly as the translator's closures would have, in order. Called
/// by the recording lane inside the stretch's render pass; after it the recorder's tracked
/// pipeline / dynamic state / vertex input / descriptor state is unknown (the back invalidates).
void Replay(const StretchCmds& cmds, vk::CommandBuffer cmdbuf, const Instance& instance);

/// One stretch's output: what the back publishes at the stretch's first draw.
struct Stretch {
    u64 first_seq{}, last_seq{}; ///< Draw record seqs of its first and last draw
    u32 draws{};
    std::shared_ptr<StretchCmds> cmds;
    u64 arena_end{};            ///< unmasked arena end of its regions (for the reuse fence)
    /// Set by the back at publish (store release): the batch tick the stretch was recorded into;
    /// verify mode stores 1 right away. 0 = unpublished (its arena regions are not free).
    std::shared_ptr<std::atomic<u64>> tick;
    /// BufferCache::StructGeneration() when the first lookup of the stretch ran: the back refuses
    /// the stretch (old path) if the generation moved before publish (a buffer was deleted or
    /// joined since the worker's lookups).
    u64 buf_gen{};
};

/// Context a translator gets (set up once per worker thread).
struct WorkerContext {
    const Instance* instance{};
    Scheduler* scheduler{};               ///< only GetMasterSemaphore()->KnownGpuTick() is read
    VideoCore::BufferCache* buffer_cache{}; ///< only A2-safe lookups may be called
    Arena* arena{};
    u32 worker_index{};
};

/// Refusal reasons (StretchTranslator::refusals index). RefusalName() gives the log label.
enum Refusal : u32 {
    RefuseOff = 0,       ///< the translator is off (config gates, arena not coherent)
    RefuseNotIndexed,
    RefuseIndirect,
    RefuseNoVariant,     ///< no native shadow variant (yet)
    RefuseVariantKind,   ///< variant without ring table / with the PT=2 debug binding
    RefuseNoVi,          ///< no FRONT_DERIVE block, or more than 16 attributes
    RefuseNoJob,         ///< job_tag 0 (no job record)
    RefuseNoRing,        ///< the job slot has no ring / unions / GPU ring handle
    RefuseStamp,         ///< record stamp / cmd / gpu_ok mismatch
    RefuseLifetime,      ///< the job thread is more than a quarter ring ahead
    RefuseRecordFields,  ///< inst_known 0, io_ud != 0, index_count 0
    RefuseInheritedSlot, ///< a Guest V# sits in inherited registers (v1: not handled)
    RefuseUnion,         ///< the run union is not published / does not cover the draw
    RefuseSlotKind,      ///< a non-Guest binding, images / samplers, or enabled user clip planes
    RefuseGuestLookup,   ///< a Guest slot's union range is not in one cached buffer
    RefuseIndexLookup,   ///< the index range is not in one cached buffer
    RefuseVertexLookup,  ///< a vertex stream's base is not in a cached buffer
    RefuseRecordRegs,    ///< the record disagrees with the mirror (counts, index / vertex addr)
    RefuseInstanceOffset,///< firstInstance != 0
    RefuseVertexOffset,  ///< the rebased vertexOffset leaves [0, INT32_MAX] or no common quotient
    RefuseRestartIndex,  ///< primitive restart with an index other than -1
    RefuseArena,         ///< no free arena region (unpublished or GPU behind) - never waits
    RefuseStages,        ///< no vertex stage data
    RefuseLayout,        ///< the variant's binding numbering does not match the stages
    RefuseNoCmd,         ///< DrawIn::job_cmd null (counted once; the draw is not refused)
    NumRefusals
};
const char* RefusalName(u32 reason);

/// Per-draw translation on a worker. One instance per worker thread.
class StretchTranslator {
public:
    explicit StretchTranslator(const WorkerContext& ctx);
    ~StretchTranslator();

    /// True if the worker can take this draw into the open stretch (opening one if none is open);
    /// it is appended to the stretch's command list then. False: the draw is the back's (old
    /// path); the caller ends the open stretch first (EndStretch) - a refused draw never changes
    /// the list. `regs` = the worker's private mirror at this draw.
    bool TakeDraw(const AmdGpu::Regs& regs, const DrawIn& draw);
    /// Ends the open stretch (flushes the pending sub-batch) and returns it (null if none open).
    std::unique_ptr<Stretch> EndStretch();
    /// Refusal counters by reason (logged by the caller).
    std::array<u64, 32> refusals{};
    /// Draws taken and sub-batches opened so far (logged by the caller).
    u64 taken{}, sub_batches{};

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace Vulkan::A2
