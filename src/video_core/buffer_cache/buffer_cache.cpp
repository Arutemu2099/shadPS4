// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <thread>
#include <immintrin.h>
#include <unordered_map>
#include "common/alignment.h"
#include "common/buffer_sample_probe.h"
#include "common/debug.h"
#include "common/fault_monitor.h"
#include "common/lean.h"
#include "common/readback_offload.h"
#include "common/scope_exit.h"
#include "core/memory.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/buffer_cache/readback_queue.h"
#include "video_core/buffer_cache/memory_tracker.h"
#include "video_core/renderer_vulkan/vk_gpu_batch_timer.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/texture_cache/texture_cache.h"

#include <vk_mem_alloc.h>

namespace VideoCore {

namespace {
// SHADPS4_DRAW_COST=1 (diagnostic): ObtainBufferForImage path and cycle split for calls over
// ~100 us (SynchronizeBuffer: dirty-page walk vs rest); first 8 lines per address.
const bool g_image_buffer_cost = [] {
    const char* value = std::getenv("SHADPS4_DRAW_COST");
    return value != nullptr && value[0] == '1';
}();
thread_local u64 g_sync_walk_cyc = 0;
thread_local u64 g_sync_rest_cyc = 0;
} // namespace

u64& DiagSyncWalkCycles() {
    return g_sync_walk_cyc;
}
u64& DiagSyncRestCycles() {
    return g_sync_rest_cyc;
}

namespace Offload = Common::ReadbackOffload;

namespace {
// Flags checked per binding, per draw or per fault, read once at startup: a function-local
// `static const` costs a guard check (a TLS epoch load chain under the MSVC ABI) on every call.
bool EnvIs1(const char* name) {
    const char* value = std::getenv(name);
    return value && value[0] == '1';
}
const char g_prebatch_mode = [] {
    const char* value = std::getenv("SHADPS4_READBACK_PREBATCH");
    return value && (value[0] == '1' || value[0] == '2') ? value[0] : '0';
}();
const bool g_precise_producer = EnvIs1("SHADPS4_PRECISE_PRODUCER");
const bool g_buffer_traffic_enabled = EnvIs1("SHADPS4_BUFFER_TRAFFIC");
const bool g_fault_write_skip_settle = EnvIs1("SHADPS4_FAULT_WRITE_SKIP_SETTLE");
const bool g_async_submit = EnvIs1("SHADPS4_ASYNC_SUBMIT");
const bool g_readback_early_flush = EnvIs1("SHADPS4_READBACK_EARLY_FLUSH");
const bool g_readback_producer_stats = EnvIs1("SHADPS4_READBACK_PRODUCER_STATS");
/// SHADPS4_WRITE_FAULT_WIDEN=1: a write fault on GPU-modified memory first downloads the same
/// widened window a read fault would (one submit and wait), so neighbouring GPU-patched pages the
/// guest CPU rewrites next no longer each cost their own synchronous download. Train-dip captures
/// (2026-09-26): ~10 consecutive 4 KB record pages, patched by cs_a16fa93a copies and then
/// rewritten by the CPU, cost ~10 extra submits and GPU waits per frame.
/// =2: the window spans every buffer in it (the train records are split into 16 KB buffers), all
/// downloaded with one submit and one wait.
const char g_write_fault_widen = [] {
    const char* value = std::getenv("SHADPS4_WRITE_FAULT_WIDEN");
    return value && (value[0] == '1' || value[0] == '2') ? value[0] : '0';
}();
/// SHADPS4_WRITE_FAULT_OFFLOAD=1 (with WRITE_FAULT_WIDEN=2 and READBACK_OFFLOAD=1, default off): a
/// guest thread's write fault on GPU-modified memory becomes a readback job over the same window
/// the synchronous =2 path downloads. The command processor records the copies and moves on; the
/// faulting thread waits for the job tick, and publication writes the bytes back, clears the
/// window's GPU state and CPU-marks the exact faulting range, before the guest write proceeds.
/// Measured 2026-09-26 (a25 dip anatomy): the dip adds ~2.5 command processor GPU waits per frame.
const bool g_write_fault_offload =
    g_write_fault_widen == '2' && EnvIs1("SHADPS4_WRITE_FAULT_OFFLOAD");
const bool g_range_add_skip = EnvIs1("SHADPS4_RANGE_ADD_SKIP");
const bool g_vb_repeat_stats = EnvIs1("SHADPS4_VB_REPEAT_STATS");
const bool g_written_memo = EnvIs1("SHADPS4_WRITTEN_MEMO");
/// SHADPS4_READ_BIND_MEMO: 0 off, 1 on, 2 verify (a hit still runs the full path and checks that
/// it uploaded nothing; the returned pair is identical by construction); 3 / 4 the same with the
/// buffer's creation serial as the structure guard instead of the global StructGeneration. Off
/// under SHADPS4_EPOCH_DEFER_UPLOAD, which can leave a synchronized read range dirty on purpose.
/// A hit returns before the ASC_SPAN_STATS / BUFFER_TRAFFIC / sample-probe accounting: take those
/// censuses with the memo off (Opus review 2026-09-28).
const int g_read_bind_memo = [] {
    const char* value = std::getenv("SHADPS4_READ_BIND_MEMO");
    const char* defer = std::getenv("SHADPS4_EPOCH_DEFER_UPLOAD");
    if (!value || value[0] < '1' || value[0] > '4' || (defer && defer[0] == '1')) {
        return 0;
    }
    return value[0] - '0';
}();
const bool g_deferred_protect = EnvIs1("SHADPS4_DEFERRED_PROTECT");
} // namespace

/// SHADPS4_READBACK_PREBATCH: 1 = copies of data the recorded batch does not write go ahead of
/// it; 2 = the same, and the batch is still submitted right after, as a fault always did.
/// Measured 2026-09-24: mode 1 alone let batches grow (787 vs 650 draws) and the remaining
/// in-batch readbacks waited 2.9 ms instead of 1.1 ms.
static char PrebatchMode() {
    return g_prebatch_mode;
}

/// SHADPS4_PRECISE_PRODUCER=1 (with SHADPS4_READBACK_QUEUE=1 or 2, prebatch off): a readback
/// copy on the readback queue waits only for the newest batch that actually wrote its bytes,
/// found in a 64-batch write history, instead of the coarse 1 MiB-slot producer ticks (whose
/// floor made nearly every copy wait for the newest submitted batch). Measured 2026-09-24:
/// early copies in queue order waited ~0.7 ms behind unrelated batches.
static bool PreciseProducerEnabled() {
    return g_precise_producer;
}

static bool PrebatchEnabled() {
    return PrebatchMode() != '0';
}

/// SHADPS4_READBACK_FAST_LANE=1 (with SHADPS4_READBACK_QUEUE=1 for the extra queue and
/// SHADPS4_READBACK_PREBATCH=2): a guest readback whose newest GPU writer (64-batch write
/// history) has already completed is copied at once on a second queue instead of behind the
/// unfinished main-queue batches; every other readback keeps the prebatch path. On PS4 the CPU
/// reads finished GPU results in place; measured 2026-09-27 (ASC_YIELD on) such copies waited
/// ~1.4 ms of unrelated GPU work. Use READBACK_QUEUE=3 (compute family): a second graphics-family
/// queue time-slices the graphics engine on NVIDIA.
static bool FastLaneEnabled() {
    static const bool enabled = EnvIs1("SHADPS4_READBACK_FAST_LANE");
    return enabled;
}

/// SHADPS4_BATCH_MIX_SS=1 (diagnostic, inFAMOUS Second Son only, with SHADPS4_READBACK_LATENCY and
/// SHADPS4_GPU_BATCH_TIMING): classifies a read fault by the game's own particle globals (REFERENCE
/// 4.2): the per-frame linear arena that holds the effect control blocks (cursor 0x807d3eb20, limit
/// 0x807d3eb28; the arena's size is not known, so "arena" = the 64 MiB below the limit, checked by
/// the logged samples) and the particle pool ([0x807d3eb40], 0x20000 records of 96 bytes).
static u32 ClassifyReadbackAddr(VAddr addr) {
    using Timer = Vulkan::GpuBatchTimer;
    static const bool enabled = EnvIs1("SHADPS4_BATCH_MIX_SS");
    if (!enabled) {
        return Timer::MixOther;
    }
    // Exact when the faulting instruction is known: the game functions of REFERENCE 4.2
    // (control-block reader 0x3f3ed0, mesh snap 0x3fed00, exposure sum 0x171610; function
    // extents assumed, entry + a few KB).
    if (Offload::t_in_hle_memset) {
        return Timer::MixMemset;
    }
    if (const u64 tagged = Offload::t_read_fault_rip; tagged != 0) {
        const bool write = (tagged >> 63) != 0;
        const u64 rip = tagged & ~(1ULL << 63);
        const u64 off = rip - 0x800000000ULL;
        u32 by_rip = write ? Timer::MixWriteFault : Timer::MixOther;
        if (off >= 0x3f3ed0 && off < 0x3f4800) {
            by_rip = Timer::MixArena;
        } else if (off >= 0x3fed00 && off < 0x400000) {
            by_rip = Timer::MixPool;
        } else if (off >= 0x171610 && off < 0x171a00) {
            by_rip = Timer::MixExposure;
        }
        static std::atomic<u32> other_samples{0};
        if ((by_rip == Timer::MixOther || by_rip == Timer::MixWriteFault) &&
            other_samples.fetch_add(1, std::memory_order_relaxed) < 256) {
            LOG_INFO(Render_Vulkan, "BatchMixOtherRip: write={} rip={:#x} off={:#x} addr={:#x}",
                     write, rip, off, addr);
        }
        return by_rip;
    }
    const auto read = [](u64 a) { return *reinterpret_cast<const volatile u64*>(a); };
    const u64 cursor = read(0x807d3eb20);
    const u64 limit = read(0x807d3eb28);
    const u64 pool = read(0x807d3eb40);
    u32 cls = Timer::MixOther;
    if (pool != 0 && addr >= pool && addr < pool + 0x20000ULL * 96) {
        cls = Timer::MixPool;
    } else if (limit != 0 && addr < limit && addr + (64ULL << 20) >= limit) {
        cls = Timer::MixArena;
    }
    static std::atomic<u32> samples{0};
    if (samples.fetch_add(1, std::memory_order_relaxed) < 48) {
        LOG_INFO(Render_Vulkan,
                 "BatchMixSample: addr={:#x} cls={} cursor={:#x} limit={:#x} pool={:#x} "
                 "limit-addr={:#x} cursor-addr={}",
                 addr, cls, cursor, limit, pool, limit - addr, s64(cursor) - s64(addr));
    }
    return cls;
}

namespace {
/// SHADPS4_READBACK_LATENCY: a thread spinning on the master timeline semaphore that records
/// when each tick was first seen complete. Diagnostic only; it burns one core.
struct TickPoller {
    static constexpr u64 Size = 8192;
    std::array<std::atomic<u64>, Size> ticks{};
    std::array<std::atomic<u64>, Size> done_ns{};
    std::jthread thread;

    /// When `tick` was first observed complete, or 0 if unknown.
    u64 DoneNs(u64 tick) const {
        const u64 i = tick % Size;
        if (ticks[i].load(std::memory_order_acquire) != tick) {
            return 0;
        }
        return done_ns[i].load(std::memory_order_relaxed);
    }
};
TickPoller* g_tick_poller = nullptr;

void StartTickPoller(Vulkan::MasterSemaphore* semaphore) {
    static TickPoller poller;
    g_tick_poller = &poller;
    poller.thread = std::jthread([semaphore](std::stop_token stop) {
        Common::SetCurrentThreadName("shadPS4:TickPoller");
        u64 last = semaphore->QueryCounter();
        while (!stop.stop_requested()) {
            const u64 value = semaphore->QueryCounter();
            if (value > last) {
                const u64 now = Offload::NowNs();
                const u64 first =
                    std::max(last + 1, value > TickPoller::Size ? value - TickPoller::Size + 1 : 1);
                for (u64 t = first; t <= value; ++t) {
                    const u64 i = t % TickPoller::Size;
                    poller.done_ns[i].store(now, std::memory_order_relaxed);
                    poller.ticks[i].store(t, std::memory_order_release);
                }
                last = value;
            } else {
                for (int i = 0; i < 8; ++i) {
                    _mm_pause();
                }
            }
        }
    });
}
} // namespace

static constexpr size_t DataShareBufferSize = 64_KB;
static constexpr size_t StagingBufferSize = 512_MB;
static constexpr size_t DownloadBufferSize = 32_MB;
static constexpr size_t UboStreamBufferSize = 64_MB;
static constexpr size_t DeviceBufferSize = 128_MB;

/**
 * Alignment a guest read fault is widened to before downloading.
 *
 * Windows faults report 8 bytes, while protection is page granular and a nonempty buffer
 * download submits and waits. Combining nearby read faults can reduce those queue drains.
 *
 * Widening stays within the existing buffer and selects only tracked GPU-dirty pages/bytes.
 * Extra bytes are speculative: the guest may not read them. Enlarged windows also preflight
 * staging capacity before dirty-state mutation. Keep the default for unmeasured workloads.
 */
static constexpr u64 DefaultReadFaultWidenSize = 64_KB;

static u64 ReadFaultWidenSize(u64 original_size) {
    static const u64 value = []() -> u64 {
        if (const char* env = std::getenv("SHADPS4_READ_FAULT_WIDEN_SIZE")) {
            char* end = nullptr;
            const u64 parsed = std::strtoull(env, &end, 0);
            if (end != env && *end == '\0' &&
                (parsed == 64_KB || parsed == 256_KB || parsed == 1_MB)) {
                LOG_INFO(Render, "BufferCache: read-fault window overridden to {} bytes", parsed);
                return parsed;
            }
            LOG_WARNING(Render, "Invalid read-fault window override; retaining 64 KiB");
        }
        return DefaultReadFaultWidenSize;
    }();
    // Only tune small read faults. Preserve baseline handling of larger explicit reads.
    return original_size > 0 && original_size <= 4_KB ? value : DefaultReadFaultWidenSize;
}

/**
 * Size ceiling for `ObtainBuffer`'s read-only stream-copy fast path.
 *
 * Upstream compares against `CACHING_PAGESIZE` (16 KB), sending small read-only bindings down a
 * path that copies guest memory into the stream buffer every draw "to reduce renderpass breaks".
 * Measured, that trade loses: the copy runs 6.6 times per draw and moves ~43 MB/frame, while the
 * tracked buffer-cache path it avoids costs about the same per call (~0.37 us vs ~0.35 us) and
 * copies nothing when the range is clean.
 *
 * **Default is 0, i.e. the fast path is off.** Same build, same scene, same draw count:
 *
 *     stream path ON   12.47 fps    ObtainBuffer ~5.2 us/draw
 *     stream path OFF  14.57 fps    ObtainBuffer  3.06 us/draw     +17%
 *
 * The objection to changing the default was memory: every small binding becomes a resident cache
 * entry. Measured head-to-head, 10 minutes idle each - growth is 6.00 MB/min with the path on and
 * 8.26 MB/min with it off, but the OFF arm DECELERATES (10.42 -> 6.00 MB/min across the run) and
 * converges on the ON arm's rate. So the extra ~22 MB is a cache filling once, not a leak. The
 * ~6 MB/min both arms share is a pre-existing leak unrelated to this.
 *
 * Set `SHADPS4_STREAM_MAX_SIZE=16384` to restore upstream behaviour. Both paths are correct - this
 * is a performance knob, not a compatibility one.
 */
static u64 StreamCopyMaxSize() {
    static const u64 value = []() -> u64 {
        if (const char* env = std::getenv("SHADPS4_STREAM_MAX_SIZE")) {
            char* end = nullptr;
            const u64 parsed = std::strtoull(env, &end, 0);
            if (end != env) {
                LOG_INFO(Render, "BufferCache: stream-copy max size overridden to {}", parsed);
                return parsed;
            }
        }
        return 0;
    }();
    return value;
}

namespace {
/// SHADPS4_BUFFER_TRAFFIC=1 (diagnostic, costs a tracker query per binding): buffer bindings
/// per 16 MB guest region and per size class - kind, bytes bound, whether the range had
/// CPU-modified pages at bind time, and bytes uploaded - logged every ~2 s. Sizes plan (b).
struct BufferTraffic {
    struct Region {
        u64 calls[4]{}; // ro, ro_texel, written, other-caller share of ro
        u64 bound{}, dirty{}, uploaded{};
    };
    std::unordered_map<u64, Region> regions;
    u64 size_calls[6]{}, size_dirty[6]{}, size_bound[6]{}, size_uploaded[6]{};
    u64 calls{}, last_frames{};
    std::chrono::steady_clock::time_point last{std::chrono::steady_clock::now()};

    static bool Enabled() {
        return g_buffer_traffic_enabled;
    }

    void Record(VAddr addr, u32 size, bool written, bool texel, bool other_caller, bool dirty,
                u64 uploaded) {
        auto& r = regions[addr >> 24];
        ++r.calls[written ? 2 : texel ? 1 : 0];
        r.calls[3] += (!written && !texel && other_caller);
        r.bound += size;
        r.dirty += dirty;
        r.uploaded += uploaded;
        const int b = size <= 256 ? 0 : size <= 4096 ? 1 : size <= 65536 ? 2
                    : size <= (1u << 20) ? 3 : size <= (16u << 20) ? 4 : 5;
        ++size_calls[b];
        size_dirty[b] += dirty;
        size_bound[b] += size;
        size_uploaded[b] += uploaded;
        if ((++calls & 1023) != 0) {
            return;
        }
        const auto now = std::chrono::steady_clock::now();
        if (now - last < std::chrono::seconds(2)) {
            return;
        }
        const u64 frames_now = Common::FaultMonitor::presented_frames.load();
        const double f = static_cast<double>(std::max<u64>(frames_now - last_frames, 1));
        std::vector<std::pair<u64, Region>> sorted(regions.begin(), regions.end());
        std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
            return a.second.calls[0] + a.second.calls[1] + a.second.calls[2] >
                   b.second.calls[0] + b.second.calls[1] + b.second.calls[2];
        });
        u64 total = 0, total_dirty = 0, total_up = 0;
        for (const auto& [base, reg] : sorted) {
            total += reg.calls[0] + reg.calls[1] + reg.calls[2];
            total_dirty += reg.dirty;
            total_up += reg.uploaded;
        }
        std::string out = fmt::format(
            "BufferTraffic: frames={:.0f} binds/frame={:.0f} dirty/frame={:.0f} "
            "upload={:.0f}kB/frame regions={} | size(<=256,4K,64K,1M,16M,>16M) binds/frame",
            f, total / f, total_dirty / f, total_up / 1024.0 / f, sorted.size());
        for (int i = 0; i < 6; ++i) {
            out += fmt::format(" {:.0f}(d{:.0f},up{:.0f}k)", size_calls[i] / f, size_dirty[i] / f,
                               size_uploaded[i] / 1024.0 / f);
        }
        out += " | top regions base:ro/texel/written(other) dirty up_kB bound_MB per frame:";
        for (size_t i = 0; i < std::min<size_t>(10, sorted.size()); ++i) {
            const auto& reg = sorted[i].second;
            out += fmt::format(" {:#x}:{:.0f}/{:.0f}/{:.0f}({:.0f}) d{:.0f} up{:.0f} b{:.1f}",
                               sorted[i].first << 24, reg.calls[0] / f, reg.calls[1] / f,
                               reg.calls[2] / f, reg.calls[3] / f, reg.dirty / f,
                               reg.uploaded / 1024.0 / f, reg.bound / 1048576.0 / f);
        }
        LOG_INFO(Render, "{}", out);
        regions.clear();
        std::fill(std::begin(size_calls), std::end(size_calls), 0);
        std::fill(std::begin(size_dirty), std::end(size_dirty), 0);
        std::fill(std::begin(size_bound), std::end(size_bound), 0);
        std::fill(std::begin(size_uploaded), std::end(size_uploaded), 0);
        last = now;
        last_frames = frames_now;
    }
};
BufferTraffic g_buffer_traffic;
} // namespace

/// Expands a read fault to the selected window, clipped to the buffer that already covers it.
/// The clip is what keeps this safe: `FindBuffer` has already run on the original range, so
/// buffer creation and merging behaviour is unchanged by the wider download.
static std::pair<VAddr, u64> WidenReadFault(const Buffer& buffer, VAddr device_addr, u64 size,
                                            u64 window) {
    if (!buffer.IsInBounds(device_addr, size)) {
        return {device_addr, size};
    }
    const VAddr begin = std::max(Common::AlignDown(device_addr, window), buffer.CpuAddr());
    const VAddr end = std::min(Common::AlignUp(device_addr + size, window),
                               buffer.CpuAddr() + buffer.SizeBytes());
    return {begin, end - begin};
}

BufferCache::BufferCache(const Vulkan::Instance& instance_, Vulkan::Scheduler& scheduler_,
                         AmdGpu::Liverpool* liverpool_, TextureCache& texture_cache_,
                         PageManager& tracker)
    : instance{instance_}, scheduler{scheduler_}, liverpool{liverpool_},
      memory{Core::Memory::Instance()}, texture_cache{texture_cache_}, page_manager{tracker},
      fault_manager{instance, scheduler, *this, CACHING_PAGEBITS, CACHING_NUMPAGES},
      staging_buffer{instance, scheduler, MemoryUsage::Upload, StagingBufferSize},
      stream_buffer{instance, scheduler, MemoryUsage::Stream, UboStreamBufferSize},
      download_buffer{instance, scheduler, MemoryUsage::Download, DownloadBufferSize},
      device_buffer{instance, scheduler, MemoryUsage::DeviceLocal, DeviceBufferSize},
      gds_buffer{instance, scheduler, MemoryUsage::Stream, 0, AllFlags, DataShareBufferSize},
      bda_pagetable_buffer{instance, scheduler, MemoryUsage::DeviceLocal,
                           0,        AllFlags,  BDA_PAGETABLE_SIZE} {
    if (const char* mirror = std::getenv("SHADPS4_HOST_MIRROR"); mirror && mirror[0] == '1') {
        host_mirror = instance.IsExternalMemoryHostEnabled();
        host_mirror_max = static_cast<u32>(256_MB);
        if (const char* max = std::getenv("SHADPS4_HOST_MIRROR_MAX")) {
            host_mirror_max = static_cast<u32>(std::strtoul(max, nullptr, 0));
        }
        // "lo-hi" (hex ok): mirror every eligible binding inside this guest range, without the
        // per-bind dirty-page query (which walks the range a second time).
        if (const char* texel = std::getenv("SHADPS4_HOST_MIRROR_TEXEL")) {
            host_mirror_texel = texel[0] == '1';
        }
        if (const char* range = std::getenv("SHADPS4_HOST_MIRROR_RANGE")) {
            char* end = nullptr;
            host_mirror_lo = std::strtoull(range, &end, 0);
            if (end && *end == '-') {
                host_mirror_hi = std::strtoull(end + 1, nullptr, 0);
            }
            LOG_INFO(Render_Vulkan, "SHADPS4_HOST_MIRROR_RANGE: {:#x}-{:#x}", host_mirror_lo,
                     host_mirror_hi);
        }
        LOG_INFO(Render_Vulkan, "SHADPS4_HOST_MIRROR: {} (max binding {} bytes)",
                 host_mirror ? "on" : "unavailable (VK_EXT_external_memory_host)",
                 host_mirror_max);
    }
    Vulkan::SetObjectName(instance.GetDevice(), gds_buffer.Handle(), "GDS Buffer");
    Vulkan::SetObjectName(instance.GetDevice(), bda_pagetable_buffer.Handle(),
                          "BDA Page Table Buffer");

    memory_tracker = std::make_unique<MemoryTracker>(tracker);
    // Same family (READBACK_QUEUE=1) or a CONCURRENT one (3); never EXCLUSIVE across families.
    const bool lane_queue_ok =
        instance.HasReadbackQueue() &&
        (instance.GetReadbackQueueFamilyIndex() == instance.GetGraphicsQueueFamilyIndex() ||
         instance.BuffersShareReadbackFamily());
    if (FastLaneEnabled() && lane_queue_ok && PrebatchEnabled() && Offload::Enabled()) {
        readback_queue = std::make_unique<ReadbackQueue>(
            instance, instance.GetGraphicsQueue(), instance.GetGraphicsQueueFamilyIndex(),
            &Vulkan::Scheduler::submit_mutex);
        fast_lane = std::make_unique<ReadbackQueue>(instance);
        LOG_INFO(Render, "ReadbackFastLane: completed-producer copies on a second queue, the "
                         "rest prebatched");
    } else if (instance.HasReadbackQueue()) {
        readback_queue = std::make_unique<ReadbackQueue>(instance);
    } else if (PrebatchEnabled() && Offload::Enabled()) {
        readback_queue = std::make_unique<ReadbackQueue>(
            instance, instance.GetGraphicsQueue(), instance.GetGraphicsQueueFamilyIndex(),
            &Vulkan::Scheduler::submit_mutex);
        LOG_INFO(Render, "ReadbackPrebatch: copies of data no unsubmitted draw writes go "
                         "ahead of the batch being recorded");
    }

    if (Offload::LatencyEnabled()) {
        StartTickPoller(scheduler.GetMasterSemaphore());
    }
    if (Offload::Enabled() || Offload::VerifyEnabled() || Offload::ProfileEnabled()) {
        LOG_INFO(Render, "ReadbackOffload: offload={} verify={} profile={}", Offload::Enabled(),
                 Offload::VerifyEnabled(), Offload::ProfileEnabled());
    }

    std::memset(gds_buffer.mapped_data.data(), 0, DataShareBufferSize);

    // Set up garbage collection parameters
    if (!instance.CanReportMemoryUsage()) {
        trigger_gc_memory = DEFAULT_TRIGGER_GC_MEMORY;
        critical_gc_memory = DEFAULT_CRITICAL_GC_MEMORY;
        return;
    }

    const s64 device_local_memory = static_cast<s64>(instance.GetTotalMemoryBudget());
    const s64 min_spacing_expected = device_local_memory - 1_GB;
    const s64 min_spacing_critical = device_local_memory - 512_MB;
    const s64 mem_threshold = std::min<s64>(device_local_memory, TARGET_GC_THRESHOLD);
    const s64 min_vacancy_expected = (6 * mem_threshold) / 10;
    const s64 min_vacancy_critical = (2 * mem_threshold) / 10;
    trigger_gc_memory = static_cast<u64>(
        std::max<u64>(std::min(device_local_memory - min_vacancy_expected, min_spacing_expected),
                      DEFAULT_TRIGGER_GC_MEMORY));
    critical_gc_memory = static_cast<u64>(
        std::max<u64>(std::min(device_local_memory - min_vacancy_critical, min_spacing_critical),
                      DEFAULT_CRITICAL_GC_MEMORY));
}

BufferCache::~BufferCache() {
    scheduler.DrainRecording();
    // Not reached in the supported lifecycle. The presenter that owns this cache is a global
    // that is never reset (gnmdriver.cpp), and every exit once it exists is std::quick_exit
    // (emulator.cpp Run and Relaunch), which runs Emulator::Shutdown and no static destructor.
    // Guest threads therefore never outlive this object, and there is no teardown protocol
    // for them: an in-process teardown has to stop the guest and the command processor
    // first. One that did not must not continue into freed jobs with threads still waiting.
    const bool jobs_idle = std::ranges::all_of(
        fault_jobs, [](const FaultJob& job) { return job.state == FaultJobState::Free; });
    ASSERT_MSG(jobs_idle && fault_guest_waiters.load() == 0,
               "BufferCache destroyed with fault downloads in flight ({} guest waiters)",
               fault_guest_waiters.load());
}

void BufferCache::BeginGpuUnmap(VAddr device_addr, u64 size) {
    backing_span_generation.fetch_add(1, std::memory_order_release);
    // SHADPS4_BIND_MEMO_STATS: an unmap can change ClampRangeSize and the buffers found there.
    // Rasterizer::UnmapMemory bumps again once the unmap is complete.
    BumpStructGeneration();
    if (!Offload::Enabled() || size == 0) {
        return;
    }
    // Linearizes against job admission, not against CPU memory operations. Release the gate
    // before waiting for CP: finalization may need the memory manager's shared lock.
    fault_mapping_gate.Block(device_addr, size);
    if (HasOverlappingFaultJob(device_addr, size)) {
        liverpool->SendCommand<true>([this, device_addr, size] {
            SettleOverlappingJobs(device_addr, size);
            ASSERT_MSG(!HasOverlappingFaultJob(device_addr, size),
                       "GPU unmap reentered an active readback publication");
        });
    }
    // Keep the range blocked through physical unmap and until a later EndGpuMap. Late faults
    // retain the synchronous contract instead of admitting a job that could outlive unmap.
}

void BufferCache::EndGpuMap(VAddr device_addr, u64 size) {
    BumpStructGeneration(); // SHADPS4_BIND_MEMO_STATS: a map can change ClampRangeSize.
    if (Offload::Enabled() && size != 0) {
        // MapMemory can hold the memory manager's writer lock. This must not call CP or wait
        // for a job; TryAdmit's callback only publishes bookkeeping under the same gate.
        fault_mapping_gate.Unblock(device_addr, size);
    }
}

void BufferCache::InvalidateMemory(VAddr device_addr, u64 size, InvalidateOrigin origin) {
    if (!IsRegionRegistered(device_addr, size)) {
        return;
    }
    // SHADPS4_FAULT_WRITE_SKIP_SETTLE=1: a write fault (guest or CP thread) needs no
    // pre-settle. It cannot unmap, and a job neither copies nor publishes a GPU-clean page,
    // while a GPU-dirty page takes the flush callback below, which settles in
    // ResolveFaultSync. A GPU write landing later bumps the region sequence and vetoes the job.
    // Measured 2026-09-23: this settle, reached from CP-thread DMA fast-path faults, blocked
    // the CP on the GPU for 3.9% of its wall time.
    const bool skip_fault_settle = g_fault_write_skip_settle;
    if (Offload::Enabled() && !(skip_fault_settle && origin != InvalidateOrigin::Explicit) &&
        HasOverlappingFaultJob(device_addr, size)) {
        // InvalidateRegion marks CPU state on this thread wherever it finds no GPU bit, and an
        // unmap then removes the pages. A job over this range still has a publication to
        // make, so it is finalized first, while the range is mapped and untouched. The flush
        // callback below cannot do this: it never runs for the GPU-clean part of the range.
        Offload::Add(Offload::State().invalidate_settle_hops);
        liverpool->SendCommand<true>(
            [this, device_addr, size] { SettleOverlappingJobs(device_addr, size); });
    }
    const bool from_guest_fault = origin == InvalidateOrigin::GuestWriteFault;
    const bool offload_write = from_guest_fault && g_write_fault_offload;
    memory_tracker->InvalidateRegion(device_addr, size, from_guest_fault,
                                     [this, device_addr, size, offload_write] {
                                         ReadMemory(device_addr, size, true, offload_write);
                                     });
}

/// Span of the tracker pages a range touches. Publication and tracker clears act on whole
/// tracker pages, so this is the granularity jobs are compared at.
static std::pair<VAddr, VAddr> TrackerPageSpan(VAddr device_addr, u64 size) {
    return {Common::AlignDown(device_addr, TRACKER_BYTES_PER_PAGE),
            Common::AlignUp(device_addr + size, TRACKER_BYTES_PER_PAGE)};
}

bool BufferCache::HasOverlappingFaultJob(VAddr device_addr, u64 size) {
    // Fast path. The command processor raises the count, under the mutex and together with
    // the window, before it records the job's copy, and lowers it only once the job has been
    // published or vetoed. A zero read here therefore proves that every job admitted before
    // this load is finalized. One admitted after it is concurrent with this invalidation, as
    // a synchronous fault download would be. Where the range is GPU-dirty its bits are
    // intact, so the flush callback meets it in SettleOverlappingJobs; where the range is
    // GPU-clean the job copied nothing and writes nothing back there.
    if (pending_fault_jobs.load() == 0) {
        return false;
    }
    const auto [begin, end] = TrackerPageSpan(device_addr, size);
    std::scoped_lock lk{fault_window_mutex};
    for (const FaultWindow& window : fault_windows) {
        if (window.begin < end && begin < window.end) {
            return true;
        }
    }
    return false;
}

void BufferCache::CompleteEmptyWriteFault(VAddr device_addr, u64 size) {
    // The synchronous download marks CPU state only after a nonempty download. With jobs in
    // play an invalidation can legitimately arrive here with nothing left: InvalidateRegion
    // saw the GPU bits of a pending job, and that job was settled, publishing the same data,
    // before this callback collected anything. The caller is owed the marking
    // InvalidateRegion would have made itself had it found the range GPU-clean, which is the
    // state the range is now in. If GPU bits remain there is unpublished data the range set
    // does not describe; that is left exactly as the synchronous path leaves it.
    if (!memory_tracker->IsRegionGpuModified(device_addr, size)) {
        memory_tracker->MarkRegionAsCpuModified(device_addr, size);
        Offload::Add(Offload::State().write_marks);
    }
}

/// Superset of any window ResolveFaultWindow can pick for this fault, computable before the
/// buffer lookup so overlapping jobs are settled while no Buffer reference is held.
static bool EarlyFlushEnabled();

static std::pair<VAddr, u64> FaultSettleSpan(VAddr device_addr, u64 size, bool is_write) {
    if (is_write) {
        return {device_addr, size};
    }
    const u64 window = std::max(ReadFaultWidenSize(size), DefaultReadFaultWidenSize);
    const VAddr begin = Common::AlignDown(device_addr, window);
    const VAddr end = Common::AlignUp(device_addr + size, window);
    return {begin, end - begin};
}

/// SHADPS4_WRITE_FAULT_WIDEN=2: the window a write fault downloads, and settles before that.
static std::pair<VAddr, u64> WriteFaultWindow(VAddr device_addr, u64 size) {
    const VAddr begin = Common::AlignDown(device_addr, DefaultReadFaultWidenSize);
    const VAddr end = Common::AlignUp(device_addr + size, DefaultReadFaultWidenSize);
    return {begin, end - begin};
}

/// Calls func(first, count) for each run of equal consecutive entries of sources[0, size): the
/// copies of one source buffer, which a fault job lists together in address order.
template <typename T, typename Func>
static void ForEachSourceRun(const T* sources, size_t size, Func&& func) {
    for (size_t first = 0; first < size;) {
        size_t last = first + 1;
        while (last < size && sources[last] == sources[first]) {
            ++last;
        }
        func(first, last - first);
        first = last;
    }
}

void BufferCache::ReadMemory(VAddr device_addr, u64 size, bool is_write, bool offload_write) {
    const bool cp_self = liverpool->IsGpuThread();
    const auto origin = cp_self ? Offload::Origin::CpSelf : Offload::Origin::Guest;
    Offload::CountFault(origin);
    if (!Offload::Enabled() || cp_self || (is_write && !offload_write)) {
        // A command processor fault cannot leave its own thread: the wait would land there
        // anyway, and a second step queued to it could never run while it is blocked here.
        // Write invalidations also stay synchronous: an offloaded write job could mark CPU
        // state over clean pages after a concurrent unmap. Read jobs never make that mark.
        // SHADPS4_WRITE_FAULT_OFFLOAD admits guest write faults: a write job CPU-marks only in
        // its publication, which an unmap settles while the range is still mapped, and a vetoed
        // write job's fallback is skipped once an unmap has blocked the range (FinishFaultDownload).
        liverpool->SendCommand<true>([this, device_addr, size, is_write, origin] {
            ResolveFaultSync(device_addr, size, is_write, origin);
        });
        Offload::MaybeLog();
        return;
    }

    // Guest-origin fault. The command processor records an owned staging copy and flushes;
    // this thread waits for that tick on the master semaphore only (Scheduler::Wait may flush
    // and is not callable here); the command processor then votes and publishes. Tracker
    // bits, gpu_modified_ranges and page protection stay untouched until that last step.
    AmdGpu::Liverpool* const cp = liverpool;
    Vulkan::MasterSemaphore* const semaphore = scheduler.GetMasterSemaphore();
    fault_guest_waiters.fetch_add(1);
    FaultTicket ticket{};
    const bool latency = Offload::LatencyEnabled();
    const u64 t_fault = latency ? Offload::NowNs() : 0;
    u64 t_cp = 0;
    u64 t_wait = 0;
    u64 t_woke = 0;
    bool free_at_wait = false;
    u32 passes = 0;
    const u64 spin_ns = Offload::SpinWaitNs();
    for (;;) {
        cp->SendCommand<true>([&] {
            if (latency && passes == 0) {
                t_cp = Offload::NowNs();
            }
            ticket = PrepareFaultDownload(device_addr, size, is_write);
        });
        ++passes;
        if (ticket.id == 0) {
            fault_guest_waiters.fetch_sub(1);
            Offload::MaybeLog();
            return;
        }
        const u64 wait_start = Offload::ProfileNowNs();
        if (latency) {
            t_wait = Offload::NowNs();
            free_at_wait = !ticket.on_readback_queue && semaphore->IsFree(ticket.tick);
        }
        if (ticket.on_readback_queue) {
            JobSemaphore(ticket.on_fast_lane, ticket.on_asc).Wait(ticket.tick);
        } else if (spin_ns != 0) {
            semaphore->SpinWait(ticket.tick, spin_ns);
        } else {
            semaphore->Wait(ticket.tick);
        }
        if (latency) {
            t_woke = Offload::NowNs();
        }
        Offload::AddSince(Offload::State().guest_wait_ns, wait_start);
        if (!ticket.retry) {
            break;
        }
        // Another fault's job covered this window; publish it, then look again.
        cp->SendCommand<true>([&] { SettlePendingJob(ticket.id); });
    }
    // No teardown check: the cache outlives every guest thread (see ~BufferCache).
    cp->SendCommand<true>(
        [&] { FinishFaultDownload(ticket.id, device_addr, size, is_write); });
    fault_guest_waiters.fetch_sub(1);
    if (latency) {
        RecordFaultLatency(ticket, passes, t_fault, t_cp, t_wait, t_woke, free_at_wait,
                           device_addr);
    }
    Offload::MaybeLog();
}

void BufferCache::RecordFaultLatency(const FaultTicket& ticket, u32 passes, u64 t_fault,
                                     u64 t_cp, u64 t_wait, u64 t_woke, bool free_at_wait,
                                     VAddr device_addr) {
    auto& l = Offload::Latency();
    if (passes == 1 && ticket.mix_flush_ns != 0) {
        if (auto* timer = scheduler.BatchTimer()) {
            timer->RequestMix({.done_tick = ticket.done_tick,
                               .last_ahead = ticket.last_ahead,
                               .own_tick = ticket.own_tick,
                               .producer = ticket.producer,
                               .flush_ns = ticket.mix_flush_ns,
                               .cls = ClassifyReadbackAddr(device_addr),
                               .label_tick = ticket.label_tick});
        }
    }
    const u64 t_done = Offload::NowNs();
    const auto add = [](std::atomic<u64>& counter, u64 value) {
        counter.fetch_add(value, std::memory_order_relaxed);
    };
    if (passes == 1 && ticket.on_readback_queue) {
        add(l.side_jobs, 1);
        add(l.side_wait_ns, t_woke - t_wait);
        add(l.side_total_ns, t_done - t_fault);
        Offload::MaybeLogLatency();
        return;
    }
    if (passes != 1 || ticket.flush_end_ns == 0) {
        // Retried jobs have no single flush to time against.
        add(l.other_jobs, 1);
        Offload::MaybeLogLatency();
        return;
    }
    add(l.jobs, 1);
    add(l.hop_ns, t_cp - t_fault);
    add(l.prep_ns, ticket.prep_end_ns - t_cp);
    add(l.flush_ns, ticket.flush_end_ns - ticket.prep_end_ns);
    add(l.wait_ns, t_woke - t_wait);
    add(l.finish_ns, t_done - t_woke);
    add(l.total_ns, t_done - t_fault);
    add(l.batch_draws, ticket.batch_draws);
    add(l.inflight, ticket.inflight);
    add(l.free_at_wait, free_at_wait ? 1 : 0);
    if (g_tick_poller) {
        const u64 own_done = ticket.on_asc ? 0 : g_tick_poller->DoneNs(ticket.tick);
        const u64 prev_done = ticket.on_asc ? 0 : g_tick_poller->DoneNs(ticket.tick - 1);
        if (own_done != 0 && own_done <= t_woke + 1'000'000) {
            add(l.timed_jobs, 1);
            const u64 start = std::max(ticket.flush_end_ns, prev_done);
            add(l.ahead_ns, start - ticket.flush_end_ns);
            add(l.own_ns, own_done > start ? own_done - start : 0);
            add(l.wake_ns, t_woke > own_done ? t_woke - own_done : 0);
        }
    }
    Offload::MaybeLogLatency();
}

std::pair<VAddr, u64> BufferCache::ResolveFaultWindow(const Buffer& buffer, VAddr device_addr,
                                                      u64 size, bool is_write) {
    if (is_write) {
        // Invalidation path. The caller picked this exact range and is about to mark it CPU
        // modified, so it must not be widened.
        return {device_addr, size};
    }
    const u64 window = ReadFaultWidenSize(size);
    auto [wide_addr, wide_size] = WidenReadFault(buffer, device_addr, size, window);
    if (window != DefaultReadFaultWidenSize) {
        // Preflight the SAME page/byte intersections and 64-byte padding used
        // by DownloadBufferMemory, before it subtracts any GPU-dirty ranges.
        // Both passes run in this CP callback; GPU range additions are owned
        // by that thread. Never turn an oversized speculative transfer into
        // a dropped download or a null staging pointer.
        u64 padded_bytes = 0;
        bool fits = true;
        memory_tracker->ForEachDownloadRange<false>(
            wide_addr, wide_size, [&](u64 range_addr, u64 range_size) {
                gpu_modified_ranges.ForEachInRange(
                    range_addr, range_size, [&](VAddr begin, VAddr end) {
                        if (!fits) {
                            return;
                        }
                        const u64 bytes = end - begin;
                        if (bytes > DownloadBufferSize) {
                            fits = false;
                            return;
                        }
                        const u64 padded = (bytes + 63) & ~63ULL;
                        if (padded > DownloadBufferSize - padded_bytes) {
                            fits = false;
                        } else {
                            padded_bytes += padded;
                        }
                    });
            });
        if (!fits) {
            LOG_INFO(Render, "Read-fault window exceeded staging capacity; using 64 KiB");
            const auto fallback =
                WidenReadFault(buffer, device_addr, size, DefaultReadFaultWidenSize);
            wide_addr = fallback.first;
            wide_size = fallback.second;
        }
    }
    return {wide_addr, wide_size};
}

bool BufferCache::DownloadWindowMultiBuffer(VAddr window_addr, u64 window_size) {
    struct Part {
        Buffer* buffer;
        vk::BufferCopy copy;
    };
    boost::container::small_vector<Part, 32> parts;
    boost::container::small_vector<BufferId, 8> seen;
    boost::container::small_vector<std::pair<VAddr, VAddr>, 8> spans;
    u64 total_size_bytes = 0;
    ForEachBufferInRange(window_addr, window_size, [&](BufferId id, Buffer& buffer) {
        if (std::ranges::find(seen, id) != seen.end()) {
            return;
        }
        seen.push_back(id);
        const VAddr lo = std::max(window_addr, buffer.CpuAddr());
        const VAddr hi = std::min(window_addr + window_size, buffer.CpuAddr() + buffer.SizeBytes());
        if (hi <= lo) {
            return;
        }
        spans.emplace_back(lo, hi);
        memory_tracker->ForEachDownloadRange<false>(
            lo, hi - lo, [&](u64 range_addr, u64 range_size) {
                gpu_modified_ranges.ForEachInRange(
                    range_addr, range_size, [&](VAddr start, VAddr end) {
                        parts.push_back(Part{&buffer, vk::BufferCopy{
                                                          .srcOffset = start - buffer.CpuAddr(),
                                                          .dstOffset = total_size_bytes,
                                                          .size = end - start,
                                                      }});
                        total_size_bytes += (end - start + 63) & ~63ULL;
                    });
                gpu_modified_ranges.Subtract(range_addr, range_size);
                MemoryTracker::gpu_clear_epoch.fetch_add(1, std::memory_order_acq_rel);
                Common::FaultMonitor::CountRangeSubtract(
                    gpu_modified_ranges.m_ranges_set.iterative_size());
            });
    });
    if (total_size_bytes == 0) {
        return false;
    }
    // Write faults reach here with 64 KiB windows; a larger explicit invalidation that exceeds the
    // staging ring would have failed in DownloadBufferMemory as well.
    ASSERT(total_size_bytes <= DownloadBufferSize);
    Common::FaultMonitor::CountReadbackDownload(total_size_bytes);
    const auto [download, offset] = download_buffer.Map(total_size_bytes);
    download_buffer.Commit();
    scheduler.EndRendering();
    {
        auto command_access = scheduler.AcquireCommandBuffer();
        const auto cmdbuf = command_access.Handle();
        if (EnvIs1("SHADPS4_DOWNLOAD_BARRIER")) {
            // As in DownloadBufferMemory: make earlier GPU writes visible to the transfer read.
            const vk::MemoryBarrier2 pre_barrier = {
                .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
                .srcAccessMask =
                    vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
                .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
                .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
            };
            cmdbuf.pipelineBarrier2(vk::DependencyInfo{
                .memoryBarrierCount = 1,
                .pMemoryBarriers = &pre_barrier,
            });
        }
        for (const Part& part : parts) {
            vk::BufferCopy copy = part.copy;
            copy.dstOffset += offset;
            cmdbuf.copyBuffer(part.buffer->buffer, download_buffer.Handle(), copy);
        }
    }
    scheduler.Finish();
    auto* memory = Core::Memory::Instance();
    for (const Part& part : parts) {
        memory->TryWriteBacking(
            std::bit_cast<u8*>(part.buffer->CpuAddr() + part.copy.srcOffset),
            download + part.copy.dstOffset, part.copy.size);
    }
    // Only the spans inside the downloaded buffers: the rest of the window may be GPU-modified
    // image memory that was not downloaded here.
    for (const auto& [lo, hi] : spans) {
        memory_tracker->UnmarkRegionAsGpuModified(lo, hi - lo);
    }
    return true;
}

/// SHADPS4_WRITE_FAULT_OVERLAP_STATS=1 (diagnostic, command-processor thread): for each write
/// fault that reaches a download, does the faulting 8-byte write land on GPU-written bytes, or only
/// share its 4 KiB page (or 64 KiB window) with them? If writes rarely hit GPU bytes, the download
/// is avoidable in principle (the CPU and GPU halves of the page could be merged instead).
void BufferCache::NoteWriteFaultOverlap(VAddr device_addr, u64 size) {
    static const bool enabled = EnvIs1("SHADPS4_WRITE_FAULT_OVERLAP_STATS");
    if (!enabled) {
        return;
    }
    static u64 faults = 0, on_gpu = 0, page_gpu = 0, page_gpu_bytes = 0, window_gpu_bytes = 0;
    static u64 same_page_as_last = 0, last_page = 0;
    u64 hit = 0, page_bytes = 0, window_bytes = 0;
    gpu_modified_ranges.ForEachInRange(device_addr, size,
                                       [&](VAddr start, VAddr end) { hit += end - start; });
    const VAddr page = Common::AlignDown(device_addr, 4096);
    gpu_modified_ranges.ForEachInRange(page, 4096,
                                       [&](VAddr start, VAddr end) { page_bytes += end - start; });
    const VAddr window = Common::AlignDown(device_addr, DefaultReadFaultWidenSize);
    gpu_modified_ranges.ForEachInRange(
        window, DefaultReadFaultWidenSize,
        [&](VAddr start, VAddr end) { window_bytes += end - start; });
    ++faults;
    on_gpu += hit != 0;
    page_gpu += page_bytes != 0;
    page_gpu_bytes += page_bytes;
    window_gpu_bytes += window_bytes;
    same_page_as_last += page == last_page;
    last_page = page;
    if ((faults & 0x3ff) == 0) {
        LOG_INFO(Render, "WriteFaultOverlap: faults={} write_on_gpu_bytes={} ({:.1f}%) page_has_gpu={} "
                         "({:.1f}%) avg_gpu_bytes/page={:.0f} avg_gpu_bytes/64KiB={:.0f} "
                         "same_page_as_previous={:.1f}%",
                 faults, on_gpu, 100.0 * double(on_gpu) / double(faults), page_gpu,
                 100.0 * double(page_gpu) / double(faults),
                 double(page_gpu_bytes) / double(faults), double(window_gpu_bytes) / double(faults),
                 100.0 * double(same_page_as_last) / double(faults));
    }
}

void BufferCache::ResolveFaultSync(VAddr device_addr, u64 size, bool is_write,
                                   Offload::Origin origin) {
    // SHADPS4_ASC_STREAM: a CP-self fault inside an ASC dispatch runs inline; its download must be
    // ordered after main-stream producers, so record it on the main stream.
    const u32 prev_stream = scheduler.ActiveStream();
    scheduler.SwitchStream(0);
    SCOPE_EXIT {
        scheduler.SwitchStream(prev_stream);
    };
    if (is_write) {
        NoteWriteFaultOverlap(device_addr, size);
    } else {
        NoteDiagRead(device_addr);
    }
    const auto [settle_addr, settle_size] = FaultSettleSpan(device_addr, size, is_write);
    SettleOverlappingJobs(settle_addr, settle_size);
    Buffer& buffer = slot_buffers[FindBuffer(device_addr, size)];
    if (is_write && g_write_fault_widen == '2' && Offload::Enabled()) {
        const VAddr begin = Common::AlignDown(device_addr, DefaultReadFaultWidenSize);
        const VAddr end = Common::AlignUp(device_addr + size, DefaultReadFaultWidenSize);
        SettleOverlappingJobs(begin, end - begin);
        if (DownloadWindowMultiBuffer(begin, end - begin)) {
            Offload::CountSyncResolve(origin, Offload::SyncTiming{});
        }
    } else if (is_write && g_write_fault_widen == '1' && Offload::Enabled()) {
        // Download the read-fault window as a read (it contains the faulting range). The exact
        // write path below then finds no GPU data left and completes as an empty write fault,
        // marking only the faulting range CPU modified.
        const auto [wide_addr, wide_bytes] = ResolveFaultWindow(buffer, device_addr, size, false);
        if (wide_bytes > size) {
            SettleOverlappingJobs(wide_addr, wide_bytes);
            Offload::SyncTiming wide_timing{};
            const bool wide_downloaded = DownloadBufferMemory<false>(
                buffer, wide_addr, wide_bytes, false,
                Offload::ProfileEnabled() ? &wide_timing : nullptr);
            if (wide_downloaded) {
                Offload::CountSyncResolve(origin, wide_timing);
            }
        }
    }
    const auto [addr, bytes] = ResolveFaultWindow(buffer, device_addr, size, is_write);
    Offload::SyncTiming timing{};
    const bool downloaded = DownloadBufferMemory<false>(
        buffer, addr, bytes, is_write, Offload::ProfileEnabled() ? &timing : nullptr);
    Offload::CountSyncResolve(origin, timing);
    if (!downloaded && is_write && Offload::Enabled()) {
        // With the offload off nothing here differs from the path this was split out of.
        CompleteEmptyWriteFault(device_addr, size);
    }
}

u64 BufferCache::CollectFaultCopies(VAddr device_addr, u64 size, std::vector<FaultCopy>& copies,
                                    u64 total_size_bytes) {
    memory_tracker->ForEachDownloadRange<false>(
        device_addr, size, [&](u64 device_addr_out, u64 range_size) {
            gpu_modified_ranges.ForEachInRange(
                device_addr_out, range_size, [&](VAddr start, VAddr end) {
                    const u64 new_size = end - start;
                    copies.push_back(FaultCopy{
                        .addr = start,
                        .staging_offset = total_size_bytes,
                        .size = new_size,
                    });
                    // Same padding as DownloadBufferMemory.
                    total_size_bytes += (new_size + 63) & ~63ULL;
                });
        });
    return total_size_bytes;
}

std::unique_ptr<Buffer> BufferCache::AcquireFaultStaging() {
    if (!fault_staging_pool.empty()) {
        auto staging = std::move(fault_staging_pool.back());
        fault_staging_pool.pop_back();
        return staging;
    }
    // One per job plus one for the verifier. Never the shared download ring: it reclaims by
    // GPU tick alone and could hand these bytes out again before they are published.
    if (fault_staging_created >= MaxFaultJobs + 1) {
        return nullptr;
    }
    auto staging =
        std::make_unique<Buffer>(instance, scheduler, MemoryUsage::Download, 0,
                                 vk::BufferUsageFlagBits::eTransferDst, FaultStagingSize());
    if (staging->mapped_data.empty()) {
        LOG_ERROR(Render, "ReadbackOffload: staging buffer is not host mapped");
        return nullptr;
    }
    ++fault_staging_created;
    return staging;
}

void BufferCache::ReleaseFaultStaging(std::unique_ptr<Buffer> staging) {
    // Callers only release after the tick of the last copy into it has been reached.
    if (staging) {
        fault_staging_pool.push_back(std::move(staging));
    }
}

BufferCache::FaultTicket BufferCache::PrepareFaultDownload(VAddr device_addr, u64 size,
                                                           bool is_write) {
    if (!is_write) {
        NoteDiagRead(device_addr);
    }
    ASSERT_MSG(!is_write || g_write_fault_offload,
               "Write faults use the readback offload only with SHADPS4_WRITE_FAULT_OFFLOAD");
    if (is_write) {
        NoteWriteFaultOverlap(device_addr, size);
    }
    // A write job covers the whole WIDEN=2 window, so the window is what must be free of jobs.
    const auto [settle_addr, settle_size] = is_write ? WriteFaultWindow(device_addr, size)
                                                     : FaultSettleSpan(device_addr, size, is_write);
    // SHADPS4_FAULT_OVERLAP_RETRY=1: a read fault inside a pending job's window no longer makes
    // the command processor wait on the GPU for that job. The faulting guest thread waits for
    // the job's tick itself, has it settled (no GPU wait left by then) and retries from the
    // start; the job's own owner still frees it. No overlapping job is ever created, which is
    // what the settle guaranteed. Measured 2026-09-23 with deferred protection: this settle
    // blocked the CP on the GPU for 13.75% of its wall time.
    static const bool overlap_retry = [] {
        const char* value = std::getenv("SHADPS4_FAULT_OVERLAP_RETRY");
        return value && value[0] == '1';
    }();
    if (overlap_retry) {
        if (pending_fault_jobs.load(std::memory_order_relaxed) != 0) {
            const auto [begin, end] = TrackerPageSpan(settle_addr, settle_size);
            for (const FaultJob& job : fault_jobs) {
                if (job.state != FaultJobState::Pending) {
                    continue;
                }
                const auto [job_begin, job_end] = TrackerPageSpan(job.addr, job.size);
                if (job_begin < end && begin < job_end) {
                    FaultTicket retry_ticket{job.id, job.tick, true, job.on_readback_queue};
                    retry_ticket.on_fast_lane = job.on_fast_lane;
                    retry_ticket.on_asc = job.on_asc;
                    return retry_ticket;
                }
            }
        }
    } else {
        SettleOverlappingJobs(settle_addr, settle_size);
    }
    const BufferId buffer_id = FindBuffer(device_addr, size);
    Buffer& buffer = slot_buffers[buffer_id];
    auto [addr, bytes] = is_write ? std::pair<VAddr, u64>{settle_addr, settle_size}
                                  : ResolveFaultWindow(buffer, device_addr, size, is_write);

    std::vector<FaultCopy> copies;
    std::vector<std::pair<VAddr, VAddr>> spans;
    boost::container::small_vector<BufferId, 8> sources; // Source buffer of each copy.
    u64 total_size_bytes = 0;
    if (is_write) {
        // DownloadWindowMultiBuffer's walk, without subtracting anything: every buffer of the
        // window, in address order, clipped to it.
        boost::container::small_vector<BufferId, 8> seen;
        ForEachBufferInRange(addr, bytes, [&](BufferId id, Buffer& window_buffer) {
            if (std::ranges::find(seen, id) != seen.end()) {
                return;
            }
            seen.push_back(id);
            const VAddr lo = std::max(addr, window_buffer.CpuAddr());
            const VAddr hi =
                std::min(addr + bytes, window_buffer.CpuAddr() + window_buffer.SizeBytes());
            if (hi <= lo) {
                return;
            }
            spans.emplace_back(lo, hi);
            total_size_bytes = CollectFaultCopies(lo, hi - lo, copies, total_size_bytes);
            sources.resize(copies.size(), id);
        });
        // The job's window is the hull of what its publication touches: the buffer spans and
        // the faulting range. Snapshot, admission and overlap checks all use it.
        VAddr hull_begin = device_addr;
        VAddr hull_end = device_addr + size;
        for (const auto& [lo, hi] : spans) {
            hull_begin = std::min(hull_begin, lo);
            hull_end = std::max(hull_end, hi);
        }
        addr = hull_begin;
        bytes = hull_end - hull_begin;
    } else {
        spans.emplace_back(addr, addr + bytes);
        total_size_bytes = CollectFaultCopies(addr, bytes, copies);
        sources.resize(copies.size(), buffer_id);
    }
    if (total_size_bytes == 0) {
        // DownloadBufferMemory returns here too, without touching the tracker.
        Offload::Add(Offload::State().prepare_empty);
        if (is_write) {
            CompleteEmptyWriteFault(device_addr, size);
        }
        return {};
    }

    FaultJob* job = nullptr;
    for (FaultJob& candidate : fault_jobs) {
        if (candidate.state == FaultJobState::Free) {
            job = &candidate;
            break;
        }
    }
    std::unique_ptr<Buffer> staging;
    if (job && total_size_bytes <= FaultStagingSize()) {
        staging = AcquireFaultStaging();
    }
    if (!staging) {
        if (is_write) {
            // Nothing was mutated above and no job overlaps the window: the synchronous
            // WIDEN=2 download, exactly as a write fault without the offload takes it.
            Offload::Add(Offload::State().capacity_fallbacks);
            ResolveFaultSync(device_addr, size, is_write, Offload::Origin::Guest);
            return {};
        }
        // Out of job slots or staging, or too large for a job. Nothing was mutated above, and
        // the list just collected is not empty, so this download marks what is_write needs.
        Offload::SyncTiming timing{};
        DownloadBufferMemory<false>(buffer, addr, bytes, is_write,
                                    Offload::ProfileEnabled() ? &timing : nullptr);
        // A guest-origin synchronous download like any other, and additionally a capacity miss.
        Offload::CountSyncResolve(Offload::Origin::Guest, timing);
        Offload::Add(Offload::State().capacity_fallbacks);
        Offload::Add(Offload::State().capacity_wait_ns, timing.total_ns);
        return {};
    }

    boost::container::small_vector<vk::BufferCopy, 8> vk_copies;
    boost::container::small_vector<vk::Buffer, 8> vk_sources;
    for (size_t i = 0; i < copies.size(); ++i) {
        const Buffer& source = slot_buffers[sources[i]];
        vk_copies.push_back(vk::BufferCopy{
            .srcOffset = copies[i].addr - source.CpuAddr(),
            .dstOffset = copies[i].staging_offset,
            .size = copies[i].size,
        });
        vk_sources.push_back(source.Handle());
    }

    // Snapshot and copy list are taken in one uninterrupted step on this thread, which also
    // owns every GPU write marking, so the recorded copy matches them. The job is complete
    // before anything below can reenter.
    NoteReadbackProducer(addr, bytes);
    if (EarlyFlushEnabled()) {
        if (!readback_hot_ticks) {
            readback_hot_ticks = std::make_unique<u64[]>(1ULL << 20);
        }
        const u64 now = scheduler.CurrentTick();
        for (u64 slot = addr >> 20; slot <= ((addr + bytes - 1) >> 20) && slot < (1ULL << 20);
             ++slot) {
            readback_hot_ticks[slot] = now;
        }
    }
    const u64 id = next_fault_job_id++;
    const u64 tick = scheduler.CurrentTick();
    job->seqs.clear();
    memory_tracker->SnapshotGpuWriteSeq(addr, bytes, job->seqs);
    const bool admitted = fault_mapping_gate.TryAdmit(addr, bytes, [&] {
        // No GPU, memory-manager call or wait inside this callback. Unmap either blocks this
        // admission or observes the published window and settles it before removing pages.
        job->id = id;
        job->state = FaultJobState::Pending;
        job->abandoned = false;
        job->is_write = is_write;
        job->addr = addr;
        job->size = bytes;
        job->tick = tick;
        job->total_bytes = total_size_bytes;
        job->copies = std::move(copies);
        job->spans = std::move(spans);
        job->mark_addr = device_addr;
        job->mark_size = size;
        job->staging = std::move(staging);
        std::scoped_lock lk{fault_window_mutex};
        const auto [window_begin, window_end] = TrackerPageSpan(addr, bytes);
        fault_windows[job - fault_jobs.data()] = {window_begin, window_end};
        pending_fault_jobs.fetch_add(1);
    });
    if (!admitted) {
        job->seqs.clear();
        ReleaseFaultStaging(std::move(staging)); // Never submitted or host-consumed.
        Offload::Add(Offload::State().mapping_fallbacks);
        Offload::Add(Offload::State().sync_fallbacks);
        ResolveFaultSync(device_addr, size, is_write, Offload::Origin::Guest);
        return {};
    }
    const vk::Buffer staging_handle = job->staging->Handle();
    Offload::Add(Offload::State().jobs);
    if (is_write) {
        Offload::Add(Offload::State().write_jobs);
    }
    Common::FaultMonitor::CountReadbackDownload(total_size_bytes);

    job->on_readback_queue = false;
    job->on_fast_lane = false;
    job->on_asc = false;
    // SHADPS4_ASC_STREAM: the newest ASC-stream writer of the window and whether it completed.
    u64 asc_producer = 0;
    bool asc_done = true;
    if (scheduler.HasAscStream()) {
        auto* asc_sem = scheduler.GetAscSemaphore();
        asc_producer = NewestAscWriter(job->copies);
        if (asc_hist_evicted != 0 && !asc_sem->IsFree(asc_hist_evicted)) {
            // A history slot of an unfinished segment was overwritten: assume it wrote here.
            asc_producer = std::max(asc_producer, asc_hist_evicted);
        }
        if (asc_producer != 0) {
            // +1: a flush inside a dispatch's pass 2 reopens the segment, so the dispatch can land
            // one segment later than its history stamp (Opus review 2026-09-28).
            const u64 need = asc_producer + 1;
            asc_done = asc_sem->IsFree(need);
            if (!asc_done) {
                asc_sem->Refresh();
                asc_done = asc_sem->IsFree(need);
            }
        }
    }
    if (!asc_done && fast_lane) {
        // The producer is still on the ASC stream: copy right behind it there, so the guest waits
        // for the particle work only, not for the graphics queued on the main queue. No unfinished
        // main batch may write the window (it would not be ordered before this copy).
        auto* master = scheduler.GetMasterSemaphore();
        master->Refresh();
        const u64 completed = master->KnownGpuTick();
        if (completed != 0 && completed < tick && !BatchesWriteAny(job->copies, completed, tick)) {
            scheduler.SwitchStream(1);
            ForEachSourceRun(sources.data(), sources.size(), [&](size_t first, size_t count) {
                RecordFaultCopiesAsync(slot_buffers[sources[first]], staging_handle,
                                       {vk_copies.data() + first, count});
            });
            scheduler.SwitchStream(0);
            const u64 asc_value = scheduler.SubmitAscSegment();
            job->tick = asc_value;
            job->on_readback_queue = true;
            job->on_asc = true;
            FaultTicket asc_ticket{id, asc_value, false, true};
            asc_ticket.on_asc = true;
            static std::atomic<u64> asc_jobs{0};
            if (((asc_jobs.fetch_add(1) + 1) & 0x3ff) == 0) {
                LOG_INFO(Render_Vulkan, "AscStream: {} readback copies on the ASC stream",
                         asc_jobs.load());
            }
            return asc_ticket;
        }
    }
    if (fast_lane && asc_done) {
        // The newest batch that GPU-wrote any copied byte; batches before it completed first
        // (one queue), batches after it wrote none of these bytes. If it has completed, nothing
        // unfinished stands between the data and this copy.
        // Eligible when no batch from the newest completed one on wrote any copied byte: the
        // newest writer p then satisfies p + 1 <= completed (a write is stamped at bind time and
        // could be recorded into the next batch if a flush ran in between; Opus review
        // 2026-09-28). Only the unfinished batches are scanned (the CP back is the forest's
        // limiter; a 64-batch scan per fault measured -1.8% there, screen s64).
        auto* master = scheduler.GetMasterSemaphore();
        master->Refresh();
        const u64 completed = master->KnownGpuTick();
        const bool done = completed != 0 && completed < tick &&
                          !BatchesWriteAny(job->copies, completed, tick);
        const u64 producer = completed;
        if (done) {
            const u64 lane_tick = fast_lane->Submit(
                master->Handle(), producer, [&](vk::CommandBuffer cmd) {
                    ForEachSourceRun(vk_sources.data(), vk_sources.size(),
                                     [&](size_t first, size_t count) {
                                         RecordFaultCopiesOnReadbackQueue(
                                             cmd, vk_sources[first], staging_handle,
                                             {vk_copies.data() + first, count}, false);
                                     });
                });
            // No main-queue wait on the copy: a GPU write to the window marked before
            // publication changes its write sequence and vetoes the job (PublishFaultJob), and
            // one marked after it comes after the guest waited for this copy (Opus review).
            job->tick = lane_tick;
            job->on_readback_queue = true;
            job->on_fast_lane = true;
            FaultTicket lane_ticket{id, lane_tick, false, true};
            lane_ticket.on_fast_lane = true;
            if (Offload::LatencyEnabled()) {
                master->Refresh();
                lane_ticket.done_tick = master->KnownGpuTick();
                lane_ticket.last_ahead = tick - 1;
                lane_ticket.producer = producer;
                lane_ticket.mix_flush_ns = Vulkan::GpuBatchTimer::NowNs();
                lane_ticket.label_tick = Offload::g_last_gfx_label_tick.load(std::memory_order_relaxed);
            }
            if (PrebatchMode() == '2') {
                // Same batch cutting as a prebatched copy.
                scheduler.EndRendering();
                scheduler.Flush();
            }
            static std::atomic<u64> lane_jobs{0};
            if (((lane_jobs.fetch_add(1) + 1) & 0x3ff) == 0) {
                LOG_INFO(Render_Vulkan, "ReadbackFastLane: {} jobs on the fast lane",
                         lane_jobs.load());
            }
            return lane_ticket;
        }
    }
    if (readback_queue && asc_done) {
        // Every GPU write into this window was recorded no later than `producer`. If that batch
        // is already submitted, copy on the readback queue after it alone.
        // SHADPS4_READBACK_PREBATCH: the queue is the main queue, so submission order already
        // puts the copy after every submitted write; it only has to avoid bytes the batch still
        // being recorded writes, which BatchWritesAny checks exactly.
        const bool prebatch = readback_queue->InQueueOrder();
        const u64 producer = prebatch                   ? 0
                             : PreciseProducerEnabled() ? PreciseProducer(job->copies)
                                                        : ProducerTick(addr, bytes);
        // With SHADPS4_ASYNC_SUBMIT, `producer < tick` means the CP flushed that batch, not that
        // the recording worker has called vkQueueSubmit for it; only wait on signals already
        // submitted (the scheduler guards the same-queue case likewise). Review 2026-09-27.
        const VkSemaphore async_timeline = Vulkan::Scheduler::async_timeline.load();
        const bool producer_submitted =
            async_timeline == VK_NULL_HANDLE ||
            producer <= Vulkan::Scheduler::async_submitted_tick.load(std::memory_order_acquire);
        const bool side_ok = prebatch ? !BatchWritesAny(job->copies)
                                      : (producer != 0 && producer < tick && producer_submitted);
        if (side_ok) {
            static std::atomic<u64> side{0}, total{0};
            u64 side_tick = 0;
            if (prebatch && AsyncFaultCopies()) {
                // Asynchronous submission: batches recorded before this point may not be
                // submitted yet, so the recording worker submits the copy when it gets here.
                side_tick = readback_queue->ReserveTick();
                std::vector<vk::BufferCopy> owned(vk_copies.begin(), vk_copies.end());
                std::vector<vk::Buffer> owned_src(vk_sources.begin(), vk_sources.end());
                scheduler.Record([this, rq = readback_queue.get(), side_tick, staging_handle,
                                  owned = std::move(owned),
                                  owned_src = std::move(owned_src)](vk::CommandBuffer) {
                    rq->SubmitReserved(side_tick, [&](vk::CommandBuffer cmd) {
                        ForEachSourceRun(owned_src.data(), owned_src.size(),
                                         [&](size_t first, size_t count) {
                                             RecordFaultCopiesOnReadbackQueue(
                                                 cmd, owned_src[first], staging_handle,
                                                 {owned.data() + first, count}, true);
                                         });
                    });
                });
                scheduler.PublishRecording();
            } else {
                side_tick = readback_queue->Submit(
                    scheduler.GetMasterSemaphore()->Handle(), producer,
                    [&](vk::CommandBuffer cmd) {
                        ForEachSourceRun(vk_sources.data(), vk_sources.size(),
                                         [&](size_t first, size_t count) {
                                             RecordFaultCopiesOnReadbackQueue(
                                                 cmd, vk_sources[first], staging_handle,
                                                 {vk_copies.data() + first, count}, prebatch);
                                         });
                    });
            }
            job->tick = side_tick;
            job->on_readback_queue = true;
            FaultTicket side_ticket{id, side_tick, false, true};
            if (Offload::LatencyEnabled()) {
                auto* semaphore = scheduler.GetMasterSemaphore();
                semaphore->Refresh();
                side_ticket.done_tick = semaphore->KnownGpuTick();
                side_ticket.last_ahead = tick - 1;
                side_ticket.producer = PreciseProducer(job->copies);
                side_ticket.mix_flush_ns = Vulkan::GpuBatchTimer::NowNs();
                side_ticket.label_tick = Offload::g_last_gfx_label_tick.load(std::memory_order_relaxed);
            }
            if (prebatch && PrebatchMode() == '2') {
                // After the copy's submission, so the copy does not wait for this batch.
                scheduler.EndRendering();
                scheduler.Flush();
            }
            const u64 n = total.fetch_add(1) + 1;
            side.fetch_add(1);
            if ((n & 0x3ff) == 0) {
                LOG_INFO(Render_Vulkan, "ReadbackQueue: {} of {} jobs on the readback queue",
                         side.load(), n);
            }
            return side_ticket;
        }
        static std::atomic<u64> main_jobs{0};
        const u64 m = main_jobs.fetch_add(1) + 1;
        if (prebatch && (m & 0x3ff) == 0) {
            LOG_INFO(Render_Vulkan, "ReadbackPrebatch: {} jobs stayed in the recorded batch", m);
        }
    }

    scheduler.EndRendering();
    if (AsyncFaultCopies()) {
        ForEachSourceRun(sources.data(), sources.size(), [&](size_t first, size_t count) {
            RecordFaultCopiesAsync(slot_buffers[sources[first]], staging_handle,
                                   {vk_copies.data() + first, count});
        });
    } else {
        auto command_access = scheduler.AcquireCommandBuffer();
        ForEachSourceRun(sources.data(), sources.size(), [&](size_t first, size_t count) {
            RecordFaultCopies(command_access.Handle(), slot_buffers[sources[first]],
                              staging_handle, {vk_copies.data() + first, count});
        });
    }
    if (Offload::LatencyEnabled()) {
        FaultTicket ticket{.id = id, .tick = tick};
        ticket.producer = PreciseProducer(job->copies);
        scheduler.GetMasterSemaphore()->Refresh();
        ticket.done_tick = scheduler.GetMasterSemaphore()->KnownGpuTick();
        ticket.last_ahead = tick - 1;
        ticket.own_tick = tick;
        ticket.prep_end_ns = Offload::NowNs();
        ticket.mix_flush_ns = Vulkan::GpuBatchTimer::NowNs();
        ticket.label_tick = Offload::g_last_gfx_label_tick.load(std::memory_order_relaxed);
        scheduler.Flush();
        ticket.flush_end_ns = Offload::NowNs();
        ticket.batch_draws = scheduler.LastBatchDraws();
        const u64 known = scheduler.GetMasterSemaphore()->KnownGpuTick();
        ticket.inflight = tick > known + 1 ? static_cast<u32>(tick - known - 1) : 0;
        return ticket;
    }
    scheduler.Flush();
    return {id, tick};
}

void BufferCache::RecordFaultCopiesOnReadbackQueue(vk::CommandBuffer cmdbuf, vk::Buffer source,
                                                    vk::Buffer staging,
                                                    std::span<const vk::BufferCopy> copies,
                                                    bool in_queue_order) {
    // The producer's main-timeline value is waited on at the transfer stage, which makes its
    // writes visible here. Main-queue access tracking of `source` is left untouched.
    // In queue order (prebatch) there is no semaphore wait: a barrier's first scope covers
    // every earlier submission on the queue, and the trailing one keeps later submissions'
    // writes of `source` from overtaking this read.
    if (in_queue_order) {
        const vk::MemoryBarrier2 before{
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
        };
        cmdbuf.pipelineBarrier2(vk::DependencyInfo{
            .memoryBarrierCount = 1,
            .pMemoryBarriers = &before,
        });
    }
    cmdbuf.copyBuffer(source, staging, copies);
    if (in_queue_order) {
        const vk::MemoryBarrier2 after{
            .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .srcAccessMask = vk::AccessFlagBits2::eNone,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask = vk::AccessFlagBits2::eNone,
        };
        cmdbuf.pipelineBarrier2(vk::DependencyInfo{
            .memoryBarrierCount = 1,
            .pMemoryBarriers = &after,
        });
    }
    const vk::BufferMemoryBarrier2 host_barrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eHost,
        .dstAccessMask = vk::AccessFlagBits2::eHostRead,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = staging,
        .offset = 0,
        .size = VK_WHOLE_SIZE,
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &host_barrier,
    });
}

void BufferCache::RecordFaultCopies(vk::CommandBuffer cmdbuf, Buffer& source, vk::Buffer staging,
                                    std::span<const vk::BufferCopy> copies) {
    source.NoteMainAccess(scheduler.CurrentTick(), false);
    // A nested verification can run while a binding's barriers have been collected but not
    // recorded. The tracked access may already describe that future binding, so use an
    // explicit all-writes source dependency rather than trusting it for this readback.
    const vk::BufferMemoryBarrier2 source_barrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = source.Handle(),
        .offset = 0,
        .size = source.SizeBytes(),
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &source_barrier,
    });
    // Keep access tracking consistent for the closing dependency and subsequent users.
    (void)source.GetBarrier(vk::AccessFlagBits2::eTransferRead,
                            vk::PipelineStageFlagBits2::eTransfer);
    cmdbuf.copyBuffer(source.Handle(), staging, copies);
    // The timeline wait establishes completion; this dependency also makes transfer writes
    // available to the host domain before mapped bytes are read (including by the verifier).
    const vk::BufferMemoryBarrier2 host_barrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eHost,
        .dstAccessMask = vk::AccessFlagBits2::eHostRead,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = staging,
        .offset = 0,
        .size = VK_WHOLE_SIZE,
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &host_barrier,
    });
    if (const auto barrier = source.GetBarrier(
            vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
            vk::PipelineStageFlagBits2::eAllCommands)) {
        cmdbuf.pipelineBarrier2(vk::DependencyInfo{
            .dependencyFlags = vk::DependencyFlagBits::eByRegion,
            .bufferMemoryBarrierCount = 1,
            .pBufferMemoryBarriers = &*barrier,
        });
    }
}

/// SHADPS4_ASYNC_SUBMIT=1 also records a guest fault's copies through the recording worker
/// instead of draining it for a synchronous borrow (measured 2026-09-24: that drain was 2.8% of
/// the CP's wall time). Access tracking of the source is still updated here, in order.
bool BufferCache::AsyncFaultCopies() {
    return g_async_submit && scheduler.IsAsyncRecording();
}

void BufferCache::RecordFaultCopiesAsync(Buffer& source, vk::Buffer staging,
                                         std::span<const vk::BufferCopy> copies) {
    source.NoteMainAccess(scheduler.CurrentTick(), false);
    const vk::Buffer src = source.Handle();
    const u64 src_size = source.SizeBytes();
    (void)source.GetBarrier(vk::AccessFlagBits2::eTransferRead,
                            vk::PipelineStageFlagBits2::eTransfer);
    const auto closing = source.GetBarrier(
        vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
        vk::PipelineStageFlagBits2::eAllCommands);
    const bool has_closing = closing.has_value();
    const vk::BufferMemoryBarrier2 closing_barrier = closing.value_or(vk::BufferMemoryBarrier2{});
    scheduler.RecordArrays(
        [src, src_size, staging, has_closing,
         closing_barrier](vk::CommandBuffer cmdbuf, std::span<const vk::BufferCopy> list) {
            const vk::BufferMemoryBarrier2 source_barrier{
                .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
                .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
                .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
                .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .buffer = src,
                .offset = 0,
                .size = src_size,
            };
            cmdbuf.pipelineBarrier2(vk::DependencyInfo{
                .bufferMemoryBarrierCount = 1,
                .pBufferMemoryBarriers = &source_barrier,
            });
            cmdbuf.copyBuffer(src, staging, list);
            const vk::BufferMemoryBarrier2 host_barrier{
                .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
                .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
                .dstStageMask = vk::PipelineStageFlagBits2::eHost,
                .dstAccessMask = vk::AccessFlagBits2::eHostRead,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .buffer = staging,
                .offset = 0,
                .size = VK_WHOLE_SIZE,
            };
            cmdbuf.pipelineBarrier2(vk::DependencyInfo{
                .bufferMemoryBarrierCount = 1,
                .pBufferMemoryBarriers = &host_barrier,
            });
            if (has_closing) {
                cmdbuf.pipelineBarrier2(vk::DependencyInfo{
                    .dependencyFlags = vk::DependencyFlagBits::eByRegion,
                    .bufferMemoryBarrierCount = 1,
                    .pBufferMemoryBarriers = &closing_barrier,
                });
            }
        },
        copies);
}

static bool EarlyFlushEnabled() {
    return g_readback_early_flush;
}

static bool ProducerStatsEnabled() {
    return g_readback_producer_stats;
}

void BufferCache::NoteBatchWrite(VAddr addr, u64 size) {
    if (size == 0) {
        return;
    }
    if (g_record_stream == 1) {
        // SHADPS4_ASC_STREAM: keyed by the open ASC segment's value, not a main tick.
        const u64 value = scheduler.OpenAscValue();
        const u64 i = value % asc_write_hist.size();
        if (asc_write_hist_value[i] != value) {
            asc_hist_evicted = std::max(asc_hist_evicted, asc_write_hist_value[i]);
            asc_write_hist[i].clear();
            asc_write_hist_value[i] = value;
        }
        asc_write_hist[i].emplace_back(addr, addr + size);
        return;
    }
    if (PreciseProducerEnabled() || Offload::LatencyEnabled() || FastLaneEnabled()) {
        const u64 tick = scheduler.CurrentTick();
        const u64 i = tick % WriteHistory;
        if (write_hist_tick[i] != tick) {
            write_hist[i].clear();
            write_hist_tick[i] = tick;
        }
        write_hist[i].emplace_back(addr, addr + size);
    }
    if (!PrebatchEnabled()) {
        return;
    }
    const u64 tick = scheduler.CurrentTick();
    if (tick != batch_writes_tick) {
        batch_writes.clear();
        batch_writes_tick = tick;
    }
    batch_writes.emplace_back(addr, addr + size);
}

Vulkan::MasterSemaphore& BufferCache::JobSemaphore(bool on_fast_lane, bool on_asc) {
    return on_asc ? *scheduler.GetAscSemaphore() : LaneOf(on_fast_lane).Semaphore();
}

u64 BufferCache::NewestAscWriter(std::span<const FaultCopy> copies) const {
    u64 newest = 0;
    for (size_t i = 0; i < asc_write_hist.size(); ++i) {
        if (asc_write_hist_value[i] <= newest) {
            continue;
        }
        for (const auto& [begin, end] : asc_write_hist[i]) {
            bool hit = false;
            for (const FaultCopy& copy : copies) {
                if (begin < copy.addr + copy.size && copy.addr < end) {
                    hit = true;
                    break;
                }
            }
            if (hit) {
                newest = asc_write_hist_value[i];
                break;
            }
        }
    }
    return newest;
}

bool BufferCache::BatchesWriteAny(std::span<const FaultCopy> copies, u64 first,
                                  u64 last) const {
    // A batch missing from the history (older than WriteHistory or never recorded) counts as a
    // writer, so the answer is conservative.
    for (u64 t = first; t <= last; ++t) {
        const u64 i = t % WriteHistory;
        if (last - t >= WriteHistory) {
            return true;
        }
        if (write_hist_tick[i] != t) {
            continue; // No GPU write was noted in batch t.
        }
        for (const auto& [begin, end] : write_hist[i]) {
            for (const FaultCopy& copy : copies) {
                if (begin < copy.addr + copy.size && copy.addr < end) {
                    return true;
                }
            }
        }
    }
    return false;
}

u64 BufferCache::PreciseProducer(std::span<const FaultCopy> copies) const {
    const u64 current = scheduler.CurrentTick();
    for (u64 back = 0; back < WriteHistory && back < current; ++back) {
        const u64 t = current - back;
        const u64 i = t % WriteHistory;
        if (write_hist_tick[i] != t) {
            continue;
        }
        for (const auto& [begin, end] : write_hist[i]) {
            for (const FaultCopy& copy : copies) {
                if (begin < copy.addr + copy.size && copy.addr < end) {
                    return t;
                }
            }
        }
    }
    // Older than the history: complete long ago, or at worst ordered by an earlier wait.
    return current > WriteHistory ? current - WriteHistory : 0;
}

bool BufferCache::BatchWritesAny(std::span<const FaultCopy> copies) {
    if (batch_writes_tick != scheduler.CurrentTick()) {
        return false;
    }
    for (const FaultCopy& copy : copies) {
        const VAddr begin = copy.addr;
        const VAddr end = copy.addr + copy.size;
        for (const auto& [write_begin, write_end] : batch_writes) {
            if (write_begin < end && begin < write_end) {
                return true;
            }
        }
    }
    return false;
}

bool BufferCache::ReadProducerStatsEnabled() {
    static const bool enabled = EnvIs1("SHADPS4_READ_PRODUCER_STATS");
    return enabled;
}

void BufferCache::NoteDiagRead(VAddr addr) {
    if (!ReadProducerStatsEnabled()) {
        return;
    }
    const auto it = diag_window_writer.find(addr >> 16);
    const u64 producer = it == diag_window_writer.end() ? 0 : it->second;
    ++diag_read_hist[producer];
    // SHADPS4_READ_FAULT_RIP=1: one line per call for the offline join with the guest-side
    // "ReadFaultRip" lines (page_manager.cpp); a fault can reach here more than once.
    static const bool per_fault = EnvIs1("SHADPS4_READ_FAULT_RIP");
    if (per_fault && diag_reads < 400000) {
        // Was the last writable binding over this window already complete on the GPU? If so the
        // readback only waits for queue order, not for its data.
        const auto tick_it = diag_window_tick.find(addr >> 16);
        const u64 ptick = tick_it == diag_window_tick.end() ? 0 : tick_it->second;
        const u64 current = scheduler.CurrentTick();
        const bool done = ptick != 0 && scheduler.IsFree(ptick);
        if (producer == 0xc73c37c1ULL) { // the 2026-09-27 question; writers of its windows only
            ++diag_faulted_windows[addr >> 16];
        }
        LOG_INFO(Render,
                 "ReadProducerFault: k={} addr={:#x} producer={:#x} ptick={} cur={} gpu={} done={}",
                 diag_reads, addr, producer, ptick, current,
                 scheduler.GetMasterSemaphore()->KnownGpuTick(), done ? 1 : 0);
    }
    if ((++diag_reads & 0x3ff) != 0) {
        return;
    }
    std::vector<std::pair<u64, u64>> top(diag_read_hist.begin(), diag_read_hist.end());
    std::sort(top.begin(), top.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
    std::string line;
    for (size_t i = 0; i < std::min<size_t>(top.size(), 10); ++i) {
        line += fmt::format(" {:#x}={}", top[i].first, top[i].second);
    }
    LOG_INFO(Render, "ReadProducer: {} read faults, {} producers, windows tracked {}; top:{}", diag_reads,
             top.size(), diag_window_writer.size(), line);
}

void BufferCache::NoteGpuWrite(VAddr addr, u64 size) {
    NoteBatchWrite(addr, size);
    if (ReadProducerStatsEnabled() && size != 0) [[unlikely]] {
        const u64 first = addr >> 16;
        const u64 last = std::min((addr + size - 1) >> 16, first + 1023);
        for (u64 w = first; w <= last; ++w) {
            diag_window_writer[w] = diag_producer;
        }
        static const bool per_fault = EnvIs1("SHADPS4_READ_FAULT_RIP");
        if (per_fault) {
            const u64 tick = scheduler.CurrentTick();
            bool covers_faulted = false;
            for (u64 w = first; w <= last; ++w) {
                diag_window_tick[w] = tick;
                covers_faulted |= diag_faulted_windows.contains(w);
            }
            if (covers_faulted) {
                const u64 seen = diag_fault_window_writers[{diag_producer, addr, size}]++;
                if ((seen & (seen + 1)) == 0) { // seen = 0, 1, 3, 7, ...
                    LOG_INFO(Render,
                             "ReadFaultWindowWriter: producer={:#x} addr={:#x} size={:#x} "
                             "seen={} tick={}",
                             diag_producer, addr, size, seen, tick);
                }
            }
            static std::unordered_map<u64, u64> writes_per_producer; // CP back thread only
            const u64 count = writes_per_producer[diag_producer]++;
            if (count < 64 || (count & 0xfff) == 0) {
                LOG_INFO(Render, "ReadProducerWrite: producer={:#x} n={} addr={:#x} size={:#x}",
                         diag_producer, count, addr, size);
            }
        }
    }
    if ((!ProducerStatsEnabled() && !readback_queue && !EarlyFlushEnabled()) || size == 0 ||
        PreciseProducerEnabled()) {
        return;
    }
    if (!producer_ticks) {
        producer_ticks = std::make_unique<u64[]>(1ULL << 20);
    }
    const u64 tick = scheduler.CurrentTick();
    const u64 first = addr >> 20;
    const u64 last = (addr + size - 1) >> 20;
    if (EarlyFlushEnabled() && readback_hot_ticks && !readback_flush_requested) {
        // A slot read back within the last ~64 submissions is likely read back again; submit
        // this write soon so the readback finds it complete (and can use the readback queue).
        for (u64 slot = first; slot <= std::min(last, first + 64) && slot < (1ULL << 20);
             ++slot) {
            if (readback_hot_ticks[slot] != 0 && readback_hot_ticks[slot] + 64 > tick) {
                readback_flush_requested = true;
                break;
            }
        }
    }
    if (last - first > 64 || last >= (1ULL << 20)) {
        // Too wide to record per slot: every readback until the next submission sees it.
        producer_floor = tick;
        return;
    }
    for (u64 slot = first; slot <= last; ++slot) {
        producer_ticks[slot] = tick;
    }
}

u64 BufferCache::ProducerTick(VAddr addr, u64 size) const {
    u64 producer = producer_floor;
    if (!producer_ticks || size == 0) {
        return producer;
    }
    for (u64 slot = addr >> 20; slot <= ((addr + size - 1) >> 20) && slot < (1ULL << 20);
         ++slot) {
        producer = std::max(producer, producer_ticks[slot]);
    }
    return producer;
}

void BufferCache::WaitFaultJob(const FaultJob& job) {
    if (job.on_readback_queue) {
        JobSemaphore(job.on_fast_lane, job.on_asc).Wait(job.tick);
        return;
    }
    if (!scheduler.IsFree(job.tick)) {
        scheduler.Wait(job.tick);
    }
}

void BufferCache::NoteReadbackProducer(VAddr addr, u64 size) {
    if (!ProducerStatsEnabled() || !producer_ticks || size == 0) {
        return;
    }
    const u64 producer = ProducerTick(addr, size);
    static u64 jobs = 0, done = 0, in_current = 0, unknown = 0, age_sum = 0;
    const u64 current = scheduler.CurrentTick();
    ++jobs;
    if (producer == 0) {
        ++unknown;
    } else if (producer >= current) {
        ++in_current;
    } else {
        age_sum += current - producer;
        done += scheduler.IsFree(producer);
    }
    if ((jobs & 0x3ff) == 0) {
        LOG_INFO(Render_Vulkan,
                 "ReadbackProducer: jobs={} producer_done_at_fault={} ({:.1f}%) producer_in_"
                 "unsubmitted_batch={} unknown={} mean_age_ticks={:.1f}",
                 jobs, done, 100.0 * done / jobs, in_current, unknown,
                 jobs - in_current - unknown ? double(age_sum) / (jobs - in_current - unknown)
                                             : 0.0);
    }
}

void BufferCache::SettlePendingJob(u64 id) {
    for (FaultJob& job : fault_jobs) {
        if (job.state != FaultJobState::Pending || job.id != id) {
            continue;
        }
        job.state = FaultJobState::Settling;
        Offload::Add(Offload::State().cp_settles);
        const u64 wait_start = Offload::ProfileNowNs();
        WaitFaultJob(job);
        Offload::AddSince(Offload::State().cp_settle_wait_ns, wait_start);
        FinalizeFaultJob(job);
        return;
    }
}

void BufferCache::FinishFaultDownload(u64 id, VAddr device_addr, u64 size, bool is_write) {
    FaultJob* job = nullptr;
    for (FaultJob& candidate : fault_jobs) {
        if (candidate.state != FaultJobState::Free && candidate.id == id) {
            job = &candidate;
            break;
        }
    }
    bool vetoed = true;
    if (!job) {
        // Only teardown frees a job its owner has not consumed. Tracking is intact by
        // construction, so the synchronous path below is still correct.
        LOG_ERROR(Render, "ReadbackOffload: job {} missing at finish", id);
    } else {
        if (job->state == FaultJobState::Pending) {
            job->state = FaultJobState::Settling;
            // The owner already waited for this tick, so this is not expected to block.
            const u64 wait_start = Offload::ProfileNowNs();
            WaitFaultJob(*job);
            Offload::AddSince(Offload::State().cp_settle_wait_ns, wait_start);
            FinalizeFaultJob(*job);
        }
        if (job->state == FaultJobState::Settling) {
            // Only reachable nested inside a settle of this job, which the command queue
            // does not do. Let that settle free the slot and resolve synchronously here.
            LOG_ERROR(Render, "ReadbackOffload: job {} finished while settling", id);
            job->abandoned = true;
        } else {
            // A job the command processor settled earlier stays in Success or Vetoed until
            // this point, so removal after a veto is never mistaken for success.
            vetoed = job->state == FaultJobState::Vetoed;
            job->state = FaultJobState::Free;
        }
    }
    if (vetoed) {
        if (is_write && !fault_mapping_gate.TryAdmit(device_addr, size, [] {})) {
            // A write job was admitted with this range unblocked, so an unmap has begun over it
            // since. That unmap's own invalidation flushes and CPU-marks the whole range while
            // it is still mapped; a fallback here could CPU-mark it after the unmap. If the
            // guest's write finds the page still protected it faults again, and that fault is
            // not admitted as a job and resolves synchronously.
            Offload::Add(Offload::State().write_map_skips);
            return;
        }
        // Exactly one synchronous fallback per vetoed job, and no retry.
        Offload::Add(Offload::State().sync_fallbacks);
        ResolveFaultSync(device_addr, size, is_write, Offload::Origin::Guest);
    }
}

void BufferCache::SettleOverlappingJobs(VAddr device_addr, u64 size) {
    if (pending_fault_jobs.load(std::memory_order_relaxed) == 0) {
        return;
    }
    // Downloads and tracker clears act on whole tracker pages, so compare page spans.
    const auto [begin, end] = TrackerPageSpan(device_addr, size);
    for (;;) {
        FaultJob* found = nullptr;
        for (FaultJob& job : fault_jobs) {
            if (job.state != FaultJobState::Pending) {
                continue;
            }
            const auto [job_begin, job_end] = TrackerPageSpan(job.addr, job.size);
            if (job_begin < end && begin < job_end) {
                found = &job;
                break;
            }
        }
        if (!found) {
            return;
        }
        // Leave Pending before anything that can reenter, then rescan from the start. The
        // slot array never moves and a Settling slot is not reused, so the pointer holds.
        // This runs on the command processor and needs no callback from it, which is what a
        // command processor fault during a guest job requires.
        found->state = FaultJobState::Settling;
        Offload::Add(Offload::State().cp_settles);
        const u64 wait_start = Offload::ProfileNowNs();
        WaitFaultJob(*found);
        Offload::AddSince(Offload::State().cp_settle_wait_ns, wait_start);
        FinalizeFaultJob(*found);
    }
}

void BufferCache::FinalizeFaultJob(FaultJob& job) {
    // Precondition: state is Settling and the GPU has reached job.tick.
    bool ok = true;
    if (Offload::VerifyEnabled()) {
        // Only a completed comparison lets a job through. Anything else is a veto, which
        // leaves the dirty state for the synchronous fallback, so a verified run contains no
        // publication that was not checked.
        ok = VerifyFaultJob(job) == VerifyResult::Matched;
    }
    if (ok) {
        ok = PublishFaultJob(job);
    }
    ReleaseFaultStaging(std::move(job.staging));
    job.copies.clear();
    job.spans.clear();
    job.seqs.clear();
    {
        // After the publication or veto, never before: an invalidation that no longer sees
        // this window must find the tracker already in its final state for this job.
        std::scoped_lock lk{fault_window_mutex};
        fault_windows[&job - fault_jobs.data()] = {};
        pending_fault_jobs.fetch_sub(1);
    }
    if (ok) {
        Offload::Add(Offload::State().successes);
    }
    if (job.abandoned) {
        job.state = FaultJobState::Free;
    } else {
        job.state = ok ? FaultJobState::Success : FaultJobState::Vetoed;
    }
}

bool BufferCache::PublishFaultJob(FaultJob& job) {
    // One uninterrupted command processor step: vote, compare, then mutate. Nothing here can
    // fault or submit, so no GPU write marking lands between the vote and the clears.
    if (!memory_tracker->GpuWriteSeqMatches(job.addr, job.size, job.seqs)) {
        Offload::Add(Offload::State().seq_vetoes);
        return false;
    }
    // Compare BEFORE subtracting: on a mismatch the sync fallback must find the ranges.
    std::vector<FaultCopy> current;
    u64 current_bytes = 0;
    for (const auto& [lo, hi] : job.spans) {
        current_bytes = CollectFaultCopies(lo, hi - lo, current, current_bytes);
    }
    if (current_bytes != job.total_bytes || current != job.copies) {
        // A nested synchronous download took part of the window while this job settled.
        LOG_ERROR(Render, "ReadbackOffload: copy list changed for {:#x}:{:#x}; falling back",
                  job.addr, job.size);
        Offload::Add(Offload::State().list_vetoes);
        return false;
    }
    for (const auto& [lo, hi] : job.spans) {
        memory_tracker->ForEachDownloadRange<false>(
            lo, hi - lo, [&](u64 device_addr_out, u64 range_size) {
                gpu_modified_ranges.Subtract(device_addr_out, range_size); MemoryTracker::gpu_clear_epoch.fetch_add(1, std::memory_order_acq_rel);
                Common::FaultMonitor::CountRangeSubtract(
                    gpu_modified_ranges.m_ranges_set.iterative_size());
            });
    }
    if (!job.staging->is_coherent) {
        vmaInvalidateAllocation(instance.GetAllocator(), job.staging->buffer.allocation, 0,
                                job.total_bytes);
    }
    const u8* const download = job.staging->mapped_data.data();
    auto* memory = Core::Memory::Instance();
    for (const FaultCopy& copy : job.copies) {
        memory->TryWriteBacking(std::bit_cast<u8*>(copy.addr), download + copy.staging_offset,
                                copy.size);
    }
    // Only the buffer spans: for a write job the rest of the window may be GPU-modified image
    // memory that was not downloaded (as in DownloadWindowMultiBuffer).
    for (const auto& [lo, hi] : job.spans) {
        memory_tracker->UnmarkRegionAsGpuModified(lo, hi - lo);
    }
    if (job.is_write &&
        !memory_tracker->IsRegionGpuModified(job.mark_addr, job.mark_size)) {
        // What the synchronous WIDEN=2 path ends in: its exact download finds nothing left and
        // CompleteEmptyWriteFault CPU-marks the faulting range, never the window.
        memory_tracker->MarkRegionAsCpuModified(job.mark_addr, job.mark_size);
    }
    return true;
}

BufferCache::VerifyResult BufferCache::VerifyFaultJob(FaultJob& job) {
    // Oracle for GPU writes that bypass the region sequence. It does not prove the absence
    // of races; it only compares two readbacks the sequence claims are identical.
    if (!memory_tracker->GpuWriteSeqMatches(job.addr, job.size, job.seqs)) {
        // Differing bytes are expected, so nothing is compared. The veto PublishFaultJob
        // would have cast is counted here, since it is no longer reached.
        Offload::Add(Offload::State().seq_vetoes);
        return VerifyResult::Unverified;
    }
    // Look the buffers up without creating or merging anything. The source of each copy has to
    // be the cache's current buffer for those bytes, found independently of the job. A write
    // job's copies can come from several buffers.
    std::unique_ptr<Buffer> second;
    boost::container::small_vector<vk::BufferCopy, 8> vk_copies;
    boost::container::small_vector<BufferId, 8> sources;
    bool in_bounds = true;
    for (const FaultCopy& copy : job.copies) {
        const BufferId buffer_id = page_table[copy.addr >> CACHING_PAGEBITS].buffer_id;
        if (IsBufferInvalid(buffer_id) ||
            !slot_buffers[buffer_id].IsInBounds(copy.addr, copy.size)) {
            in_bounds = false;
            break;
        }
        sources.push_back(buffer_id);
        vk_copies.push_back(vk::BufferCopy{
            .srcOffset = copy.addr - slot_buffers[buffer_id].CpuAddr(),
            .dstOffset = copy.staging_offset,
            .size = copy.size,
        });
    }
    if (in_bounds && !job.copies.empty()) {
        second = AcquireFaultStaging();
    }
    if (!second) {
        // No source buffer spans the list any more, or no second staging is left. The job is
        // vetoed rather than published unchecked.
        LOG_WARNING(Render, "ReadbackOffload: no independent readback for {:#x}:{:#x}; vetoed",
                    job.addr, job.size);
        Offload::Add(Offload::State().verify_skips);
        return VerifyResult::Unverified;
    }
    // Same ordering rules as the first copy: end rendering, drain, record in stream order.
    // The reference is taken here and dropped before Finish, which can reenter the cache.
    scheduler.EndRendering();
    {
        auto command_access = scheduler.AcquireCommandBuffer();
        ForEachSourceRun(sources.data(), sources.size(), [&](size_t first, size_t count) {
            RecordFaultCopies(command_access.Handle(), slot_buffers[sources[first]],
                              second->Handle(), {vk_copies.data() + first, count});
        });
    }
    scheduler.Finish();
    VerifyResult result = VerifyResult::Unverified;
    // Finish can reenter. A marking since then makes differing bytes legitimate; the job is
    // then vetoed by the same vote in PublishFaultJob's place.
    if (!memory_tracker->GpuWriteSeqMatches(job.addr, job.size, job.seqs)) {
        Offload::Add(Offload::State().seq_vetoes);
    } else {
        for (const Buffer* staging : {second.get(), job.staging.get()}) {
            if (!staging->is_coherent) {
                vmaInvalidateAllocation(instance.GetAllocator(), staging->buffer.allocation, 0,
                                        job.total_bytes);
            }
        }
        Offload::Add(Offload::State().verify_checks);
        result = VerifyResult::Matched;
        const u8* const first_data = job.staging->mapped_data.data();
        const u8* const second_data = second->mapped_data.data();
        for (const FaultCopy& copy : job.copies) {
            if (std::memcmp(first_data + copy.staging_offset, second_data + copy.staging_offset,
                            copy.size) != 0) {
                LOG_CRITICAL(Render,
                             "ReadbackOffload: verify mismatch at {:#x}:{:#x} in job window "
                             "{:#x}:{:#x}; GPU write without a sequence bump",
                             copy.addr, copy.size, job.addr, job.size);
                Offload::Add(Offload::State().verify_mismatches);
                result = VerifyResult::Mismatch;
                break;
            }
        }
    }
    ReleaseFaultStaging(std::move(second));
    return result;
}

namespace {
// SHADPS4_RANGE_ADD_SKIP=1: skip gpu_modified_ranges.Add when the range is already recorded
// (one tree lookup instead of insert + join; the set stays identical). A direct-mapped memo of
// ranges added since the last Subtract (gpu_clear_epoch) answers repeats without the lookup.
bool RangeAddSkip() {
    return g_range_add_skip;
}
struct RangeAddMemo {
    VAddr addr;
    u64 size;
    u64 epoch;
};
std::array<RangeAddMemo, 256> range_add_memo{};
u64 range_add_calls = 0, range_add_skipped = 0;

template <typename Set>
void AddGpuModified(Set& set, VAddr addr, u64 size) {
    if (!RangeAddSkip()) {
        set.Add(addr, size);
        return;
    }
    const u64 epoch = MemoryTracker::gpu_clear_epoch.load(std::memory_order_acquire);
    auto& memo = range_add_memo[(addr >> 6) & (range_add_memo.size() - 1)];
    bool contained = memo.addr == addr && memo.size >= size && memo.epoch == epoch;
    if (!contained) {
        contained = set.Contains(addr, size);
        if (!contained) {
            set.Add(addr, size);
        }
        memo = {addr, size, epoch};
    }
    ++range_add_calls;
    range_add_skipped += contained;
    if ((range_add_calls & 0xfffff) == 0) {
        LOG_INFO(Render_Vulkan, "RangeAddSkip: adds={} contained={} ({:.1f}%)", range_add_calls,
                 range_add_skipped, 100.0 * range_add_skipped / range_add_calls);
    }
}
} // namespace

template <bool async>
bool BufferCache::DownloadBufferMemory(Buffer& buffer, VAddr device_addr, u64 size, bool is_write,
                                       Offload::SyncTiming* timing) {
    boost::container::small_vector<vk::BufferCopy, 1> copies;
    u64 total_size_bytes = 0;
    memory_tracker->ForEachDownloadRange<false>(
        device_addr, size, [&](u64 device_addr_out, u64 range_size) {
            const VAddr buffer_addr = buffer.CpuAddr();
            const auto add_download = [&](VAddr start, VAddr end) {
                const u64 new_offset = start - buffer_addr;
                const u64 new_size = end - start;
                copies.push_back(vk::BufferCopy{
                    .srcOffset = new_offset,
                    .dstOffset = total_size_bytes,
                    .size = new_size,
                });
                // Align up to avoid cache conflicts
                constexpr u64 align = 64ULL;
                constexpr u64 mask = ~(align - 1ULL);
                total_size_bytes += (new_size + align - 1) & mask;
            };
            gpu_modified_ranges.ForEachInRange(device_addr_out, range_size, add_download);
            gpu_modified_ranges.Subtract(device_addr_out, range_size); MemoryTracker::gpu_clear_epoch.fetch_add(1, std::memory_order_acq_rel);
            Common::FaultMonitor::CountRangeSubtract(
                gpu_modified_ranges.m_ranges_set.iterative_size());
        });
    if (total_size_bytes == 0) {
        return false;
    }
    Common::FaultMonitor::CountReadbackDownload(total_size_bytes);
    const auto [download, offset] = download_buffer.Map(total_size_bytes);
    for (auto& copy : copies) {
        // Modify copies to have the staging offset in mind
        copy.dstOffset += offset;
    }
    download_buffer.Commit();
    scheduler.EndRendering();
    auto command_access = scheduler.AcquireCommandBuffer();
    const auto cmdbuf = command_access.Handle();
    // SHADPS4_DOWNLOAD_BARRIER=1 (port of upstream c1d5b311 / #4889): make earlier GPU writes of
    // the buffer visible to this transfer read. Without it the copy may read data from before
    // the dispatch or draw that produced it (upstream: all-zero shadow readbacks).
    static const bool download_barrier = [] {
        const char* value = std::getenv("SHADPS4_DOWNLOAD_BARRIER");
        return value && value[0] == '1';
    }();
    if (download_barrier) {
        const vk::BufferMemoryBarrier2 pre_barrier = {
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
            .buffer = buffer.buffer,
            .offset = 0,
            .size = buffer.SizeBytes(),
        };
        cmdbuf.pipelineBarrier2(vk::DependencyInfo{
            .bufferMemoryBarrierCount = 1,
            .pBufferMemoryBarriers = &pre_barrier,
        });
    }
    cmdbuf.copyBuffer(buffer.buffer, download_buffer.Handle(), copies);
    const auto write_data = [&]() {
        auto* memory = Core::Memory::Instance();
        for (const auto& copy : copies) {
            const VAddr copy_device_addr = buffer.CpuAddr() + copy.srcOffset;
            const u64 dst_offset = copy.dstOffset - offset;
            memory->TryWriteBacking(std::bit_cast<u8*>(copy_device_addr), download + dst_offset,
                                    copy.size);
        }
        memory_tracker->UnmarkRegionAsGpuModified(device_addr, size);
        if (is_write) {
            memory_tracker->MarkRegionAsCpuModified(device_addr, size);
        }
    };
    if constexpr (async) {
        (void)timing;
        scheduler.DeferOperation(write_data);
    } else {
        if (timing) {
            // Scheduler::Finish, statement for statement (vk_scheduler.cpp): take the tick,
            // submit with an empty SubmitInfo, which is all Flush() does, then Wait on the
            // tick. The submit has advanced the current tick past it, so Wait does not flush
            // again and its time is the GPU wait alone.
            const u64 start = Offload::NowNs();
            const u64 presubmit_tick = scheduler.CurrentTick();
            scheduler.Flush();
            const u64 submitted = Offload::NowNs();
            scheduler.Wait(presubmit_tick);
            const u64 done = Offload::NowNs();
            timing->submit_ns += submitted - start;
            timing->gpu_wait_ns += done - submitted;
            timing->total_ns += done - start;
        } else {
            scheduler.Finish();
        }
        write_data();
    }
    return true;
}

void BufferCache::BindVertexBuffers(
    const Vulkan::GraphicsPipeline& pipeline, const Shader::BoundStage& vs,
    boost::container::small_vector<vk::BufferMemoryBarrier2, 16>& barriers) {
    const auto& regs = liverpool->BackRegs();
    Vulkan::VertexInputs<vk::VertexInputAttributeDescription2EXT> attributes;
    Vulkan::VertexInputs<vk::VertexInputBindingDescription2EXT> bindings;
    Vulkan::VertexInputs<vk::VertexInputBindingDivisorDescriptionEXT> divisors;
    Vulkan::VertexInputs<AmdGpu::Buffer> guest_buffers;
    pipeline.GetVertexInputs(attributes, bindings, divisors, guest_buffers,
                             regs.vgt_instance_step_rate_0, regs.vgt_instance_step_rate_1, vs);

    if (instance.IsVertexInputDynamicState()) {
        // Update current vertex inputs.
        scheduler.RecordArrays(
            [](vk::CommandBuffer cmdbuf, auto bindings, auto attributes) {
                cmdbuf.setVertexInputEXT(bindings, attributes);
            },
            std::span<const vk::VertexInputBindingDescription2EXT>{bindings},
            std::span<const vk::VertexInputAttributeDescription2EXT>{attributes});
    }

    if (bindings.empty()) {
        // If there are no bindings, there is nothing further to do.
        return;
    }

    struct BufferRange {
        VAddr base_address;
        VAddr end_address;
        vk::Buffer vk_buffer;
        u64 offset;

        [[nodiscard]] size_t GetSize() const {
            return end_address - base_address;
        }
    };

    // Build list of ranges covering the requested buffers
    Vulkan::VertexInputs<BufferRange> ranges{};
    for (const auto& buffer : guest_buffers) {
        if (buffer.base_address != 0 && buffer.GetSize() > 0) {
            ranges.emplace_back(buffer.base_address, buffer.base_address + buffer.GetSize());
        }
    }

    // Merge connecting ranges together
    Vulkan::VertexInputs<BufferRange> ranges_merged{};
    if (!ranges.empty()) {
        std::ranges::sort(ranges, [](const BufferRange& lhv, const BufferRange& rhv) {
            return lhv.base_address < rhv.base_address;
        });
        ranges_merged.emplace_back(ranges[0]);
        for (auto range : ranges) {
            auto& prev_range = ranges_merged.back();
            if (prev_range.end_address < range.base_address) {
                ranges_merged.emplace_back(range);
            } else {
                prev_range.end_address = std::max(prev_range.end_address, range.end_address);
            }
        }
    }

    // SHADPS4_VB_REPEAT_STATS=1 (diagnostic): how often a draw binds exactly the merged vertex
    // ranges of the previous draw, i.e. the ceiling for reusing its ObtainBuffer results.
    const bool repeat_stats = g_vb_repeat_stats;
    if (repeat_stats) {
        static std::vector<std::pair<VAddr, VAddr>> previous;
        static u64 draws = 0, repeats = 0, with_ranges = 0, range_total = 0;
        bool same = previous.size() == ranges_merged.size();
        for (size_t i = 0; same && i < ranges_merged.size(); ++i) {
            same = previous[i].first == ranges_merged[i].base_address &&
                   previous[i].second == ranges_merged[i].end_address;
        }
        previous.clear();
        for (const auto& range : ranges_merged) {
            previous.emplace_back(range.base_address, range.end_address);
        }
        ++draws;
        with_ranges += !ranges_merged.empty();
        repeats += same && !ranges_merged.empty();
        range_total += ranges_merged.size();
        if ((draws & ((1u << 18) - 1)) == 0) {
            LOG_INFO(Render_Vulkan,
                     "VB_REPEAT_STATS: draws {} with_ranges {} same_as_previous {} ({:.1f}%) "
                     "ranges/draw {:.2f}",
                     draws, with_ranges, repeats, 100.0 * repeats / std::max<u64>(with_ranges, 1),
                     double(range_total) / draws);
        }
    }

    // Map buffers for merged ranges
    for (auto& range : ranges_merged) {
        const u64 size = memory->ClampRangeSize(range.base_address, range.GetSize());
        const auto [buffer, offset] = ObtainBuffer(range.base_address, size, false);
        range.vk_buffer = buffer->buffer;
        range.offset = offset;
        if (IsRegionGpuModified(range.base_address, size)) {
            if (auto barrier =
                    buffer->GetBarrier(vk::AccessFlagBits2::eVertexAttributeRead,
                                       vk::PipelineStageFlagBits2::eVertexAttributeInput)) {
                barriers.emplace_back(*barrier);
            }
        }
    }

    // Bind vertex buffers
    Vulkan::VertexInputs<vk::Buffer> host_buffers;
    Vulkan::VertexInputs<vk::DeviceSize> host_offsets;
    Vulkan::VertexInputs<vk::DeviceSize> host_sizes;
    Vulkan::VertexInputs<vk::DeviceSize> host_strides;
    for (const auto& buffer : guest_buffers) {
        if (buffer.base_address != 0 && buffer.GetSize() > 0) {
            const auto host_buffer_info =
                std::ranges::find_if(ranges_merged, [&](const BufferRange& range) {
                    return buffer.base_address >= range.base_address &&
                           buffer.base_address < range.end_address;
                });
            ASSERT(host_buffer_info != ranges_merged.cend());
            host_buffers.emplace_back(host_buffer_info->vk_buffer);
            host_offsets.push_back(host_buffer_info->offset + buffer.base_address -
                                   host_buffer_info->base_address);
        } else {
            host_buffers.emplace_back(VK_NULL_HANDLE);
            host_offsets.push_back(0);
        }
        host_sizes.push_back(buffer.GetSize());
        host_strides.push_back(buffer.GetStride());
    }

    scheduler.RecordArrays(
        [dynamic = instance.IsVertexInputDynamicState()](vk::CommandBuffer cmdbuf, auto buffers,
                                                         auto offsets, auto sizes, auto strides) {
            if (dynamic) {
                cmdbuf.bindVertexBuffers(0, u32(buffers.size()), buffers.data(), offsets.data());
            } else {
                cmdbuf.bindVertexBuffers2(0, u32(buffers.size()), buffers.data(), offsets.data(),
                                          sizes.data(), strides.data());
            }
        },
        std::span<const vk::Buffer>{host_buffers}, std::span<const vk::DeviceSize>{host_offsets},
        std::span<const vk::DeviceSize>{host_sizes}, std::span<const vk::DeviceSize>{host_strides});
}

void BufferCache::BindIndexBuffer(
    u32 index_offset, boost::container::small_vector<vk::BufferMemoryBarrier2, 16>& barriers) {
    const auto& regs = liverpool->BackRegs();

    // Figure out index type and size.
    const bool is_index16 = regs.index_buffer_type.index_type == AmdGpu::IndexType::Index16;
    const vk::IndexType index_type = is_index16 ? vk::IndexType::eUint16 : vk::IndexType::eUint32;
    const u32 index_size = is_index16 ? sizeof(u16) : sizeof(u32);
    const VAddr index_address =
        regs.index_base_address.Address<VAddr>() + index_offset * index_size;

    // Bind index buffer.
    const u32 index_buffer_size = regs.num_indices * index_size;
    const auto [vk_buffer, offset] = ObtainBuffer(index_address, index_buffer_size, false);
    if (IsRegionGpuModified(index_address, index_buffer_size)) {
        if (auto barrier = vk_buffer->GetBarrier(vk::AccessFlagBits2::eIndexRead,
                                                 vk::PipelineStageFlagBits2::eIndexInput)) {
            barriers.emplace_back(*barrier);
        }
    }
    scheduler.Record([buffer = vk_buffer->Handle(), offset, index_type](vk::CommandBuffer cmdbuf) {
        cmdbuf.bindIndexBuffer(buffer, offset, index_type);
    });
}

void BufferCache::FillBuffer(VAddr address, u32 num_bytes, u32 value, bool is_gds) {
    ASSERT_MSG(address % 4 == 0, "GDS offset must be dword aligned");
    if (!is_gds) {
        texture_cache.ClearMeta(address);
        if (!IsRegionGpuModified(address, num_bytes)) {
            u32* buffer = std::bit_cast<u32*>(address);
            std::fill(buffer, buffer + num_bytes / sizeof(u32), value);
            memory_tracker->ClearRecent(address, num_bytes);
            return;
        }
    }
    Buffer* buffer = [&] {
        if (is_gds) {
            return &gds_buffer;
        }
        const auto [buffer, offset] = ObtainBuffer(address, num_bytes, true);
        return buffer;
    }();
    buffer->NoteMainAccess(scheduler.CurrentTick(), true);
    buffer->Fill(buffer->Offset(address), num_bytes, value);
}

void BufferCache::CopyBuffer(VAddr dst, VAddr src, u32 num_bytes, bool dst_gds, bool src_gds) {
    if (!dst_gds && !IsRegionGpuModified(dst, num_bytes)) {
        if (!src_gds && !IsRegionGpuModified(src, num_bytes) &&
            !texture_cache.FindImageFromRange(src, num_bytes)) {
            // Both buffers were not transferred to GPU yet. Can safely copy in host memory.
            memcpy(std::bit_cast<void*>(dst), std::bit_cast<void*>(src), num_bytes);
            memory_tracker->ClearRecent(dst, num_bytes);
            return;
        }
        // Without a readback there's nothing we can do with this
        // Fallback to creating dst buffer on GPU to at least have this data there
    }
    texture_cache.InvalidateMemoryFromGPU(dst, num_bytes);
    auto& src_buffer = [&] -> const Buffer& {
        if (src_gds) {
            return gds_buffer;
        }
        const auto buffer_id = FindBuffer(src, num_bytes);
        auto& buffer = slot_buffers[buffer_id];
        SynchronizeBuffer(buffer, src, num_bytes, false, true, nullptr);
        return buffer;
    }();
    auto& dst_buffer = [&] -> const Buffer& {
        if (dst_gds) {
            return gds_buffer;
        }
        const auto buffer_id = FindBuffer(dst, num_bytes);
        auto& buffer = slot_buffers[buffer_id];
        SynchronizeBuffer(buffer, dst, num_bytes, true, true, nullptr);
        NoteGpuWrite(dst, num_bytes);
        AddGpuModified(gpu_modified_ranges, dst, num_bytes);
        Common::FaultMonitor::CountRangeAdd(gpu_modified_ranges.m_ranges_set.iterative_size());
        return buffer;
    }();
    src_buffer.NoteMainAccess(scheduler.CurrentTick(), false);
    dst_buffer.NoteMainAccess(scheduler.CurrentTick(), true);
    const vk::BufferCopy region = {
        .srcOffset = src_buffer.Offset(src),
        .dstOffset = dst_buffer.Offset(dst),
        .size = num_bytes,
    };
    const vk::BufferMemoryBarrier2 buf_barriers_before[2] = {
        {
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eMemoryRead,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .buffer = dst_buffer.Handle(),
            .offset = dst_buffer.Offset(dst),
            .size = num_bytes,
        },
        {
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
            .buffer = src_buffer.Handle(),
            .offset = src_buffer.Offset(src),
            .size = num_bytes,
        },
    };
    scheduler.EndRendering();
    const vk::BufferMemoryBarrier2 buf_barriers_after[2] = {
        {
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask = vk::AccessFlagBits2::eMemoryRead,
            .buffer = dst_buffer.Handle(),
            .offset = dst_buffer.Offset(dst),
            .size = num_bytes,
        },
        {
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eTransferRead,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask = vk::AccessFlagBits2::eMemoryWrite,
            .buffer = src_buffer.Handle(),
            .offset = src_buffer.Offset(src),
            .size = num_bytes,
        },
    };
    scheduler.Record([source = src_buffer.Handle(), destination = dst_buffer.Handle(), region,
                      before = std::to_array(buf_barriers_before),
                      after = std::to_array(buf_barriers_after)](vk::CommandBuffer cmdbuf) {
        cmdbuf.pipelineBarrier2(vk::DependencyInfo{
            .dependencyFlags = vk::DependencyFlagBits::eByRegion,
            .bufferMemoryBarrierCount = 2,
            .pBufferMemoryBarriers = before.data(),
        });
        cmdbuf.copyBuffer(source, destination, region);
        cmdbuf.pipelineBarrier2(vk::DependencyInfo{
            .dependencyFlags = vk::DependencyFlagBits::eByRegion,
            .bufferMemoryBarrierCount = 2,
            .pBufferMemoryBarriers = after.data(),
        });
    });
}

std::pair<Buffer*, u32> BufferCache::ObtainBuffer(VAddr device_addr, u32 size, bool is_written,
                                                  bool is_texel_buffer, BufferId buffer_id,
                                                  Common::BufferSampleCaller sample_caller) {
    // This function is 4.93 us/draw, 61% of all per-draw CPU cost, and it has two halves that
    // share no code: the early exit below (a tracker query plus a guest-memory memcpy) and the
    // SynchronizeBuffer fall-through. Time them separately - the split decides which is worth
    // cutting, and it has never actually been measured.
    // SHADPS4_LEAN: no probe object at all. An inactive probe still costs a ~100-byte construction,
    // the sampling-flag guard and a destructor check per call (a19 profile: 9 of 1279 CP samples
    // under BufferSampleProbe frames).
    using ProbeClock = Common::BufferSampleProbe::Clock;
    std::optional<Common::BufferSampleProbe> sample_storage;
    if (!::Common::Lean()) {
        sample_storage.emplace(Common::BufferSampleProbe::ShouldSample(), is_written,
                               is_texel_buffer, size, sample_caller);
    }
    Common::BufferSampleProbe* const sample = sample_storage ? &*sample_storage : nullptr;
    const auto t_entry = Common::FaultMonitor::Now();
    u64 query_ns = 0;
    // For read-only buffers use device local stream buffer to reduce renderpass breaks.
    //
    // That trade is unmeasured. The fast path costs a guest->staging memcpy per binding per draw
    // (6.6 calls/draw, ~43 MB/frame, 2.3 us/draw measured) and cannot be memoised, because it
    // never clears the range's CPU-modified bits and so never write-protects the pages - there is
    // no invalidation signal to hang a cache on. The fall-through is the fully tracked path, so
    // lowering this threshold can only cost performance, never correctness. Overridable to A/B it
    // without a rebuild; 0 disables the fast path entirely.
    if (!is_written && size <= StreamCopyMaxSize()) {
        const bool gpu_modified = IsRegionGpuModified(device_addr, size);
        const auto t_query = Common::FaultMonitor::Now();
        query_ns = Common::FaultMonitor::DiffNs(t_entry, t_query);
        if (!gpu_modified) {
            const u64 offset =
                stream_buffer.Copy(device_addr, size, instance.UniformMinAlignment());
            if (sample) {
                sample->RecordStream();
            }
            Common::FaultMonitor::CountObtainStream(query_ns,
                                                    Common::FaultMonitor::SinceNs(t_query), size);
            return {&stream_buffer, offset};
        }
    }
    if (host_mirror && !is_written && (!is_texel_buffer || host_mirror_texel) &&
        sample_caller != Common::BufferSampleCaller::Other) {
        if (const auto mirrored = TryHostMirror(device_addr, size, is_texel_buffer);
            mirrored.first) {
            return mirrored;
        }
    }
    const bool buffer_id_fallback = IsBufferInvalid(buffer_id);
    if (sample) {
        sample->RecordBufferIdFallback(buffer_id_fallback);
    }
    if (buffer_id_fallback) {
        const auto t_find = sample ? sample->Now() : ProbeClock::time_point{};
        buffer_id = FindBuffer(device_addr, size);
        if (sample) {
            sample->AddFindNs(sample->SinceNs(t_find));
        }
    }
    Buffer& buffer = slot_buffers[buffer_id];
    buffer.NoteMainAccess(scheduler.CurrentTick(), is_written);
    // SHADPS4_READ_BIND_MEMO: a read-only range is synchronized when no page of it is CPU-dirty,
    // and g_dirty_epoch is bumped after every CPU-dirty publication (the invariant the tracker's
    // own clean memo, MemoSaysClean, already relies on). So if this exact (buffer, range) ran in
    // full at the current epoch and no buffer was created, joined, deleted, mapped or unmapped
    // since (StructGeneration), SynchronizeBuffer would upload nothing and change no state: skip
    // it and everything around it. GPU writes need no upload for a GPU read (the caller's
    // GetBarrier orders them). Texel reads also sync from images and are excluded; the stream
    // and host-mirror paths returned above. 58.9% of read bindings in fire repeat at an unchanged
    // epoch within a submission (DRAW_COST census, run diag-fire-a30-drawcost-20260928-t).
    ReadMemoEntry* read_memo = nullptr;
    u64 read_memo_epoch = 0;
    u64 read_memo_gen = 0;
    bool read_memo_verify = false;
    u64 read_memo_bytes_before = 0;
    if (g_read_bind_memo != 0 && !is_written && !is_texel_buffer) {
        read_memo_epoch = g_dirty_epoch.load(std::memory_order_acquire);
        const bool serial_guard = g_read_bind_memo >= 3;
        read_memo_gen = serial_guard ? buffer.memo_serial : StructGeneration();
        const u64 h = (device_addr * 0x9E3779B97F4A7C15ULL) ^ (u64(size) * 0xC2B2AE3D27D4EB4FULL) ^
                      u64(buffer_id.index);
        read_memo = &read_memo_table[(h ^ (h >> 31)) & (read_memo_table.size() - 1)];
        auto& st = read_memo_stats[sample_caller == Common::BufferSampleCaller::Other ? 1 : 0];
        ++st[0];
        const bool key = read_memo->addr == device_addr && read_memo->size == size &&
                         read_memo->buffer_index == buffer_id.index;
        const bool epoch_ok = read_memo->epoch == read_memo_epoch;
        const bool guard_ok = read_memo->gen == read_memo_gen && (!serial_guard || read_memo_gen);
        if (!key) {
            ++st[2];
        } else if (!epoch_ok) {
            ++st[3];
        } else if (!guard_ok) {
            ++st[4];
        } else {
            ++st[1];
            if (g_read_bind_memo == 1 || g_read_bind_memo == 3) {
                return {&buffer, buffer.Offset(device_addr)};
            }
            read_memo_verify = true;
            read_memo_bytes_before = sync_upload_bytes_total;
        }
        if ((++read_memo_calls & 0x3fffff) == 0) {
            const auto& b = read_memo_stats[0];
            const auto& o = read_memo_stats[1];
            LOG_INFO(Render,
                     "ReadBindMemo: bind calls={} hits={} miss key/epoch/guard={}/{}/{} | other "
                     "calls={} hits={} miss key/epoch/guard={}/{}/{} | verify_bad={}",
                     b[0], b[1], b[2], b[3], b[4], o[0], o[1], o[2], o[3], o[4],
                     read_memo_verify_bad);
        }
    }
    if (Offload::AscSpanStatsEnabled()) [[unlikely]] {
        auto& span = Offload::AscSpan();
        const u64 now = scheduler.CurrentTick();
        if (span.in_asc_dispatch) {
            span.w_access = std::max(span.w_access, buffer.diag_main_access_tick);
            span.w_write = std::max(span.w_write, buffer.diag_main_write_tick);
            ++span.binds;
            span.max_buffer_mb = std::max<u64>(span.max_buffer_mb, buffer.SizeBytes() >> 20);
        } else {
            buffer.diag_main_access_tick = now;
            if (is_written) {
                buffer.diag_main_write_tick = now;
            }
        }
    }
    // SHADPS4_WRITTEN_MEMO=1: a GPU-written binding of exactly this range was fully synchronized
    // and marked GPU-modified before, and since then no GPU-modified bit was cleared anywhere
    // (download/publication bump the epoch), no page in its regions became CPU-dirty, and no
    // readback job is pending (their write-sequence vote needs the marking). Then the walk
    // would upload nothing and change no bit: skip it, and the range add it would repeat.
    // Measured 2026-09-24: written bindings walk and lock every region with no fast path.
    const bool written_memo = g_written_memo;
    u64 memo_epoch = 0;
    WrittenMemoEntry* memo_entry = nullptr;
    if (written_memo && is_written) {
        static u64 calls = 0, hits = 0, miss_key = 0, miss_epoch = 0, miss_jobs = 0, miss_cpu = 0;
        ++calls;
        memo_epoch = MemoryTracker::gpu_clear_epoch.load(std::memory_order_acquire);
        const u64 h = (device_addr * 0x9E3779B97F4A7C15ULL) ^ (u64(size) * 0xC2B2AE3D27D4EB4FULL);
        memo_entry = &written_memo_table[(h ^ (h >> 29)) % written_memo_table.size()];
        bool hit = false;
        if (memo_entry->addr != device_addr || memo_entry->size != size) {
            ++miss_key;
        } else if (memo_entry->epoch != memo_epoch) {
            ++miss_epoch;
        } else if (pending_fault_jobs.load(std::memory_order_acquire) != 0) {
            ++miss_jobs;
        } else if (!memory_tracker->RangeCpuClean(device_addr, size)) {
            ++miss_cpu;
        } else {
            hit = true;
            ++hits;
        }
        if (!::Common::Lean() && (calls & 0xffff) == 0) {
            LOG_INFO(Render, "WrittenMemo: calls={} hits={} ({:.1f}%) miss_key={} miss_epoch={} "
                             "miss_jobs={} miss_cpu={}",
                     calls, hits, 100.0 * double(hits) / double(calls), miss_key, miss_epoch,
                     miss_jobs, miss_cpu);
        }
        if (hit) {
            // The write history (prebatch / fast lane) must still see this GPU write.
            NoteBatchWrite(device_addr, size);
            return {&buffer, buffer.Offset(device_addr)};
        }
    }
    const auto t_sync = Common::FaultMonitor::Now();
    const auto t_sample_sync = sample ? sample->Now() : ProbeClock::time_point{};
    const bool traffic = BufferTraffic::Enabled();
    const bool traffic_dirty = traffic && memory_tracker->IsRegionCpuModified(device_addr, size);
    last_sync_upload_bytes = 0;
    SynchronizeBuffer(buffer, device_addr, size, is_written, is_texel_buffer,
                      sample && sample->Active() ? sample : nullptr);
    if (traffic) {
        g_buffer_traffic.Record(device_addr, size, is_written, is_texel_buffer,
                                sample_caller == Common::BufferSampleCaller::Other, traffic_dirty,
                                last_sync_upload_bytes);
    }
    if (memo_entry) {
        // Epoch read before the walk: a clear during it makes the record miss next time.
        *memo_entry = {device_addr, size, memo_epoch};
    }
    if (read_memo) {
        // SHADPS4_READ_BIND_MEMO=2: a hit must have been a no-op.
        if (read_memo_verify && sync_upload_bytes_total != read_memo_bytes_before) {
            if (++read_memo_verify_bad <= 20) {
                LOG_ERROR(Render, "ReadBindMemo verify: hit uploaded {} bytes at {:#x}+{:#x}",
                          sync_upload_bytes_total - read_memo_bytes_before, device_addr, size);
            }
        }
        // Epoch and generation read before the walk: a change during it misses next time.
        *read_memo = {device_addr, size, read_memo_epoch, read_memo_gen, buffer_id.index};
    }
    if (sample) {
        sample->AddSyncNs(sample->SinceNs(t_sample_sync));
    }
    Common::FaultMonitor::CountObtainSync(query_ns, Common::FaultMonitor::SinceNs(t_sync));
    if (is_written) {
        const auto t_range_add = sample ? sample->Now() : ProbeClock::time_point{};
        NoteGpuWrite(device_addr, size);
        AddGpuModified(gpu_modified_ranges, device_addr, size);
        if (sample) {
            sample->AddRangeAddNs(sample->SinceNs(t_range_add));
        }
        Common::FaultMonitor::CountRangeAdd(gpu_modified_ranges.m_ranges_set.iterative_size());
    }
    return {&buffer, buffer.Offset(device_addr)};
}

std::pair<Buffer*, u32> BufferCache::TryHostMirror(VAddr device_addr, u32 size, bool is_texel) {
    ++hm_calls;
    std::pair<Buffer*, u32> result{};
    u64 backing = 0;
    const bool range_mode = host_mirror_hi > host_mirror_lo;
    const bool overlaps_range =
        range_mode && device_addr < host_mirror_hi && device_addr + size > host_mirror_lo;
    if (size > host_mirror_max) {
        ++hm_toobig;
        hm_max_seen = std::max<u64>(hm_max_seen, size);
        if (overlaps_range && hm_samples < 12) {
            ++hm_samples;
            LOG_INFO(Render_Vulkan, "HostMirror: too big {:#x}+{:#x}", device_addr, size);
        }
    } else if (range_mode && !overlaps_range) {
        // Range mode: any binding that overlaps the range (the per-frame heap) is mirrored.
        ++hm_outside;
    } else if (is_texel && texture_cache.FindImageFromRange(device_addr, size)) {
        // A cached image aliases this formatted buffer; its newest bytes may be in the image.
        ++hm_image;
    } else if (!range_mode && !memory_tracker->IsRegionCpuModified(device_addr, size)) {
        // Clean: the tracked copy is current and in VRAM; keep using it.
        ++hm_clean;
    } else if (IsRegionGpuModified(device_addr, size)) {
        // Newest bytes are in a cache buffer, not in guest memory.
        ++hm_gpu;
        if (is_texel && (hm_gpu & 0x3fff) == 1 && hm_samples < 16) {
            // Which GPU-written ranges keep this formatted binding off the mirror.
            ++hm_samples;
            u64 count = 0, bytes = 0;
            std::string first;
            gpu_modified_ranges.ForEachInRange(device_addr, size, [&](VAddr lo, VAddr hi) {
                ++count;
                bytes += hi - lo;
                if (count <= 4) {
                    first += fmt::format(" {:#x}+{:#x}", lo, hi - lo);
                }
            });
            LOG_INFO(Render_Vulkan, "HostMirror: texel {:#x}+{:#x} blocked by {} GPU ranges, {} KB:{}",
                     device_addr, size, count, bytes / 1024, first);
        }
    } else if (!MirrorBackingOffset(device_addr, size, backing)) {
        ++hm_nobacking;
    } else if ((backing >> HostMirrorChunkBits) != ((backing + size - 1) >> HostMirrorChunkBits)) {
        ++hm_straddle;
    } else if (Buffer* chunk = HostMirrorChunk(backing >> HostMirrorChunkBits)) {
        ++hm_hits;
        hm_bytes += size;
        result = {chunk, static_cast<u32>(backing & ((u64{1} << HostMirrorChunkBits) - 1))};
    }
    if ((hm_calls & 0xfffff) == 0) {
        LOG_INFO(Render_Vulkan,
                 "HostMirror: calls={} hits={} ({:.1f}%) clean={} gpu={} nobacking={} "
                 "straddle={} toobig={} (max {:#x}) outside={} image={} chunks={} "
                 "avg_hit_bytes={:.0f}",
                 hm_calls, hm_hits, 100.0 * double(hm_hits) / double(hm_calls), hm_clean, hm_gpu,
                 hm_nobacking, hm_straddle, hm_toobig, hm_max_seen, hm_outside, hm_image,
                 host_mirror_chunks.size(), hm_hits ? double(hm_bytes) / double(hm_hits) : 0.0);
    }
    return result;
}

bool BufferCache::MirrorBackingOffset(VAddr device_addr, u32 size, u64& backing) {
    // Spans are stable until an unmap (BeginGpuUnmap bumps the generation); four recent ones
    // cover the per-frame heap without taking the memory manager's lock per binding.
    const u64 generation = backing_span_generation.load(std::memory_order_acquire);
    for (auto& span : backing_spans) {
        if (span.generation == generation && device_addr >= span.start &&
            device_addr + size <= span.end) {
            backing = span.backing + (device_addr - span.start);
            return true;
        }
    }
    BackingSpanEntry entry{};
    if (!memory->BackingSpan(device_addr, entry.start, entry.end, entry.backing) ||
        device_addr + size > entry.end) {
        return false;
    }
    entry.generation = generation;
    backing_spans[backing_span_next++ & 3] = entry;
    backing = entry.backing + (device_addr - entry.start);
    return true;
}

Buffer* BufferCache::HostMirrorChunk(u64 index) {
    if (index >= host_mirror_chunks.size()) {
        host_mirror_chunks.resize(index + 1);
    }
    auto& slot = host_mirror_chunks[index];
    if (slot.buffer || slot.failed) {
        return slot.buffer.get();
    }
    slot.failed = true;
    auto& space = memory->GetAddressSpace();
    const u64 chunk_offset = index << HostMirrorChunkBits;
    const u64 chunk_size = std::min<u64>(u64{1} << HostMirrorChunkBits,
                                         space.BackingSizeBytes() - chunk_offset);
    void* pointer = space.BackingBase() + chunk_offset;
    const auto device = static_cast<VkDevice>(instance.GetDevice());
    const auto& d = VULKAN_HPP_DEFAULT_DISPATCHER;
    constexpr auto handle_type = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
    VkMemoryHostPointerPropertiesEXT props{VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
    VkResult r = d.vkGetMemoryHostPointerPropertiesEXT(device, handle_type, pointer, &props);
    if (r != VK_SUCCESS || props.memoryTypeBits == 0) {
        LOG_ERROR(Render_Vulkan, "HostMirror: chunk {} pointer properties failed ({}), bits={:#x}",
                  index, static_cast<int>(r), props.memoryTypeBits);
        return nullptr;
    }
    const auto& mem_props = instance.GetMemoryProperties();
    u32 type_index = ~0u;
    for (u32 i = 0; i < mem_props.memoryTypeCount; ++i) {
        if ((props.memoryTypeBits & (1u << i)) &&
            (mem_props.memoryTypes[i].propertyFlags & vk::MemoryPropertyFlagBits::eHostVisible)) {
            type_index = i;
            break;
        }
    }
    if (type_index == ~0u) {
        LOG_ERROR(Render_Vulkan, "HostMirror: no host-visible type in {:#x}", props.memoryTypeBits);
        return nullptr;
    }
    const VkExternalMemoryBufferCreateInfo external_ci{
        .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
        .handleTypes = handle_type,
    };
    const VkBufferCreateInfo buffer_ci{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = &external_ci,
        .size = chunk_size,
        .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    VkBuffer handle{};
    r = d.vkCreateBuffer(device, &buffer_ci, nullptr, &handle);
    if (r != VK_SUCCESS) {
        LOG_ERROR(Render_Vulkan, "HostMirror: vkCreateBuffer failed ({})", static_cast<int>(r));
        return nullptr;
    }
    const VkImportMemoryHostPointerInfoEXT import_info{
        .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT,
        .handleType = handle_type,
        .pHostPointer = pointer,
    };
    const VkMemoryAllocateInfo alloc_info{
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = &import_info,
        .allocationSize = chunk_size,
        .memoryTypeIndex = type_index,
    };
    VkDeviceMemory device_memory{};
    r = d.vkAllocateMemory(device, &alloc_info, nullptr, &device_memory);
    if (r == VK_SUCCESS) {
        r = d.vkBindBufferMemory(device, handle, device_memory, 0);
    }
    if (r != VK_SUCCESS) {
        LOG_ERROR(Render_Vulkan, "HostMirror: import of chunk {} failed ({})", index,
                  static_cast<int>(r));
        d.vkDestroyBuffer(device, handle, nullptr);
        return nullptr;
    }
    slot.buffer = std::make_unique<Buffer>(instance, scheduler, vk::Buffer{handle}, 0, chunk_size);
    slot.failed = false;
    LOG_INFO(Render_Vulkan, "HostMirror: imported backing chunk {} ({:#x}+{:#x}) as memory type {}",
             index, chunk_offset, chunk_size, type_index);
    return slot.buffer.get();
}

std::pair<Buffer*, u32> BufferCache::ObtainBufferForImage(VAddr gpu_addr, u32 size) {
    const u64 ibc_t0 = g_image_buffer_cost ? __rdtsc() : 0;
    const auto note = [&](const char* path, const Buffer* buf) {
        if (!g_image_buffer_cost) {
            return;
        }
        const u64 cyc = __rdtsc() - ibc_t0;
        if (cyc < 350000) {
            return;
        }
        static thread_local std::unordered_map<VAddr, u32> logged;
        if (logged[gpu_addr]++ < 8) {
            LOG_INFO(Render_Vulkan,
                     "ImageBufferCost: {:#x} size={} path={} total={} sync_walk={} sync_rest={} "
                     "cyc buffer={:#x} buffer_size={}",
                     gpu_addr, size, path, cyc, g_sync_walk_cyc, g_sync_rest_cyc,
                     buf ? buf->CpuAddr() : 0, buf ? buf->SizeBytes() : 0);
        }
    };
    g_sync_walk_cyc = 0;
    g_sync_rest_cyc = 0;
    // Check if any buffer contains the full requested range.
    const BufferId buffer_id = page_table[gpu_addr >> CACHING_PAGEBITS].buffer_id;
    if (buffer_id) {
        if (Buffer& buffer = slot_buffers[buffer_id]; buffer.IsInBounds(gpu_addr, size)) {
            SynchronizeBuffer(buffer, gpu_addr, size, false, false, nullptr);
            note("cached", &buffer);
            return {&buffer, buffer.Offset(gpu_addr)};
        }
    }
    // If some buffer within was GPU modified create a full buffer to avoid losing GPU data.
    if (IsRegionGpuModified(gpu_addr, size)) {
        auto result = ObtainBuffer(gpu_addr, size, false, false);
        note("gpu_modified", result.first);
        return result;
    }
    SCOPE_EXIT {
        note("staging", nullptr);
    };
    // In all other cases, just do a CPU copy to the staging buffer.
    const auto [data, offset] = staging_buffer.Map(size, 16);
    memory->CopySparseMemory(gpu_addr, data, size);
    staging_buffer.Commit();
    return {&staging_buffer, offset};
}

bool BufferCache::IsRegionRegistered(VAddr addr, size_t size) {
    // Check if we are missing some edge case here
    return buffer_ranges.Intersects(addr, size);
}

bool BufferCache::IsRegionCpuModified(VAddr addr, size_t size) {
    return memory_tracker->IsRegionCpuModified(addr, size);
}

bool BufferCache::IsRegionGpuModified(VAddr addr, size_t size) {
    return memory_tracker->IsRegionGpuModified(addr, size);
}

bool BufferCache::IsRegionCpuDirtyRelaxed(VAddr addr, u64 size) const {
    return memory_tracker->CpuDirtyRelaxed(addr, size);
}

BufferId BufferCache::FindBuffer(VAddr device_addr, u32 size) {
    ASSERT(device_addr != 0);
    const u64 page = device_addr >> CACHING_PAGEBITS;
    const BufferId buffer_id = page_table[page].buffer_id;
    if (!buffer_id) {
        return CreateBuffer(device_addr, size);
    }
    const Buffer& buffer = slot_buffers[buffer_id];
    if (buffer.IsInBounds(device_addr, size)) {
        return buffer_id;
    }
    return CreateBuffer(device_addr, size);
}

BufferCache::OverlapResult BufferCache::ResolveOverlaps(VAddr device_addr, u32 wanted_size) {
    static constexpr int STREAM_LEAP_THRESHOLD = 16;
    boost::container::small_vector<BufferId, 16> overlap_ids;
    VAddr begin = device_addr;
    VAddr end = device_addr + wanted_size;
    int stream_score = 0;
    bool has_stream_leap = false;
    const auto expand_begin = [&](VAddr add_value) {
        static constexpr VAddr min_page = CACHING_PAGESIZE + DEVICE_PAGESIZE;
        if (add_value > begin - min_page) {
            begin = min_page;
            device_addr = DEVICE_PAGESIZE;
            return;
        }
        begin -= add_value;
        device_addr = begin - CACHING_PAGESIZE;
    };
    const auto expand_end = [&](VAddr add_value) {
        static constexpr VAddr max_page = 1ULL << MemoryTracker::MAX_CPU_PAGE_BITS;
        if (add_value > max_page - end) {
            end = max_page;
            return;
        }
        end += add_value;
    };
    if (begin == 0) {
        return OverlapResult{
            .ids = std::move(overlap_ids),
            .begin = begin,
            .end = end,
            .has_stream_leap = has_stream_leap,
        };
    }
    for (; device_addr >> CACHING_PAGEBITS < Common::DivCeil(end, CACHING_PAGESIZE);
         device_addr += CACHING_PAGESIZE) {
        const BufferId overlap_id = page_table[device_addr >> CACHING_PAGEBITS].buffer_id;
        if (!overlap_id) {
            continue;
        }
        Buffer& overlap = slot_buffers[overlap_id];
        if (overlap.is_picked) {
            continue;
        }
        overlap_ids.push_back(overlap_id);
        overlap.is_picked = true;
        const VAddr overlap_device_addr = overlap.CpuAddr();
        const bool expands_left = overlap_device_addr < begin;
        if (expands_left) {
            begin = overlap_device_addr;
        }
        const VAddr overlap_end = overlap_device_addr + overlap.SizeBytes();
        const bool expands_right = overlap_end > end;
        if (overlap_end > end) {
            end = overlap_end;
        }
        stream_score += overlap.StreamScore();
        if (stream_score > STREAM_LEAP_THRESHOLD && !has_stream_leap) {
            // When this memory region has been joined a bunch of times, we assume it's being used
            // as a stream buffer. Increase the size to skip constantly recreating buffers.
            has_stream_leap = true;
            if (expands_right) {
                expand_end(CACHING_PAGESIZE * 128);
            }
            if (expands_left) {
                expand_begin(CACHING_PAGESIZE * 128);
            }
        }
    }
    return OverlapResult{
        .ids = std::move(overlap_ids),
        .begin = begin,
        .end = end,
        .has_stream_leap = has_stream_leap,
    };
}

void BufferCache::JoinOverlap(BufferId new_buffer_id, BufferId overlap_id,
                              bool accumulate_stream_score) {
    Buffer& new_buffer = slot_buffers[new_buffer_id];
    Buffer& overlap = slot_buffers[overlap_id];
    NoteBatchWrite(overlap.CpuAddr(), overlap.SizeBytes());
    new_buffer.NoteMainAccess(scheduler.CurrentTick(), true);
    overlap.NoteMainAccess(scheduler.CurrentTick(), false);
    if (accumulate_stream_score) {
        new_buffer.IncreaseStreamScore(overlap.StreamScore() + 1);
    }
    const size_t dst_base_offset = overlap.CpuAddr() - new_buffer.CpuAddr();
    const vk::BufferCopy copy = {
        .srcOffset = 0,
        .dstOffset = dst_base_offset,
        .size = overlap.SizeBytes(),
    };
    scheduler.EndRendering();
    auto command_access = scheduler.AcquireCommandBuffer();
    const auto cmdbuf = command_access.Handle();

    boost::container::static_vector<vk::BufferMemoryBarrier2, 2> pre_barriers{};
    if (auto src_barrier = overlap.GetBarrier(vk::AccessFlagBits2::eTransferRead,
                                              vk::PipelineStageFlagBits2::eTransfer)) {
        pre_barriers.push_back(*src_barrier);
    }
    if (auto dst_barrier =
            new_buffer.GetBarrier(vk::AccessFlagBits2::eTransferWrite,
                                  vk::PipelineStageFlagBits2::eTransfer, dst_base_offset)) {
        pre_barriers.push_back(*dst_barrier);
    }
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = static_cast<u32>(pre_barriers.size()),
        .pBufferMemoryBarriers = pre_barriers.data(),
    });

    cmdbuf.copyBuffer(overlap.Handle(), new_buffer.Handle(), copy);

    boost::container::static_vector<vk::BufferMemoryBarrier2, 2> post_barriers{};
    if (auto src_barrier =
            overlap.GetBarrier(vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
                               vk::PipelineStageFlagBits2::eAllCommands)) {
        post_barriers.push_back(*src_barrier);
    }
    if (auto dst_barrier = new_buffer.GetBarrier(
            vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
            vk::PipelineStageFlagBits2::eAllCommands, dst_base_offset)) {
        post_barriers.push_back(*dst_barrier);
    }
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = static_cast<u32>(post_barriers.size()),
        .pBufferMemoryBarriers = post_barriers.data(),
    });
    DeleteBuffer(overlap_id);
}

BufferId BufferCache::CreateBuffer(VAddr device_addr, u32 wanted_size) {
    const VAddr device_addr_end = Common::AlignUp(device_addr + wanted_size, CACHING_PAGESIZE);
    device_addr = Common::AlignDown(device_addr, CACHING_PAGESIZE);
    wanted_size = static_cast<u32>(device_addr_end - device_addr);
    const OverlapResult overlap = ResolveOverlaps(device_addr, wanted_size);
    const u32 size = static_cast<u32>(overlap.end - overlap.begin);
    const BufferId new_buffer_id =
        slot_buffers.insert(instance, scheduler, MemoryUsage::DeviceLocal, overlap.begin,
                            AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress, size);
    auto& new_buffer = slot_buffers[new_buffer_id];
    new_buffer.memo_serial = ++buffer_memo_serial;
    for (const BufferId overlap_id : overlap.ids) {
        JoinOverlap(new_buffer_id, overlap_id, !overlap.has_stream_leap);
    }
    Register(new_buffer_id);
    return new_buffer_id;
}

void BufferCache::ProcessFaultBuffer() {
    fault_manager.ProcessFaultBuffer();
}

void BufferCache::Register(BufferId buffer_id) {
    ChangeRegister<true>(buffer_id);
}

void BufferCache::Unregister(BufferId buffer_id) {
    ChangeRegister<false>(buffer_id);
}

template <bool insert>
void BufferCache::ChangeRegister(BufferId buffer_id) {
    // SHADPS4_BIND_MEMO_STATS: every CreateBuffer, JoinOverlap and DeleteBuffer passes here.
    BumpStructGeneration();
    Buffer& buffer = slot_buffers[buffer_id];
    const auto size = buffer.SizeBytes();
    const VAddr device_addr_begin = buffer.CpuAddr();
    const VAddr device_addr_end = device_addr_begin + size;
    const u64 page_begin = device_addr_begin / CACHING_PAGESIZE;
    const u64 page_end = Common::DivCeil(device_addr_end, CACHING_PAGESIZE);
    const u64 size_pages = page_end - page_begin;
    for (u64 page = page_begin; page != page_end; ++page) {
        if constexpr (insert) {
            page_table[page].buffer_id = buffer_id;
        } else {
            page_table[page].buffer_id = BufferId{};
        }
    }
    if constexpr (insert) {
        total_used_memory += Common::AlignUp(size, CACHING_PAGESIZE);
        buffer.SetLRUId(lru_cache.Insert(buffer_id, gc_tick));
        boost::container::small_vector<vk::DeviceAddress, 128> bda_addrs;
        bda_addrs.reserve(size_pages);
        for (u64 i = 0; i < size_pages; ++i) {
            vk::DeviceAddress addr = buffer.BufferDeviceAddress() + (i << CACHING_PAGEBITS);
            bda_addrs.push_back(addr);
        }
        WriteDataBuffer(bda_pagetable_buffer, page_begin * sizeof(vk::DeviceAddress),
                        bda_addrs.data(), bda_addrs.size() * sizeof(vk::DeviceAddress));
        buffer_ranges.Add(buffer.CpuAddr(), buffer.SizeBytes(), buffer_id);
    } else {
        total_used_memory -= Common::AlignUp(size, CACHING_PAGESIZE);
        lru_cache.Free(buffer.LRUId());
        const u64 offset = bda_pagetable_buffer.Offset(page_begin * sizeof(vk::DeviceAddress));
        bda_pagetable_buffer.Fill(offset, size_pages * sizeof(vk::DeviceAddress), 0);
        buffer_ranges.Subtract(buffer.CpuAddr(), buffer.SizeBytes());
    }
}

bool BufferCache::SynchronizeBuffer(Buffer& buffer, VAddr device_addr, u32 size, bool is_written,
                                    bool is_texel_buffer,
                                    Common::BufferSampleProbe* sample_probe) {
    boost::container::small_vector<vk::BufferCopy, 4> copies;
    size_t total_size_bytes = 0;
    VAddr buffer_start = buffer.CpuAddr();
    vk::Buffer src_buffer = VK_NULL_HANDLE;
    const auto t_upload = sample_probe ? sample_probe->Now()
                                       : Common::BufferSampleProbe::Clock::time_point{};
    // SHADPS4_DEFERRED_PROTECT=1: the re-protect syscalls and the guest-to-staging copy of this
    // upload move to the recording thread, in the same order (protect, then copy). The command
    // processor still clears the dirty bits and updates watcher counts now. A guest write that
    // lands before the deferred protect is applied is captured by the copy that follows it; one
    // after it faults as usual. The copy reads the physical backing, so pages read-protected in
    // the meantime (readbacks) cannot fault on the recording thread.
    // Measured 2026-09-23: VirtualProtect was 10.2% and this copy ~2% of CP wall time.
    const bool deferred_protect = g_deferred_protect;
    const bool defer = deferred_protect && scheduler.IsAsyncRecording();
    if (defer) {
        deferred_protects.clear();
        deferred_copies.clear();
        PageManager::SetDeferredProtectSink(&deferred_protects);
    }
    const u64 ibc_t0 = g_image_buffer_cost ? __rdtsc() : 0;
    SCOPE_EXIT {
        if (g_image_buffer_cost) {
            g_sync_rest_cyc = __rdtsc() - ibc_t0 - g_sync_walk_cyc;
        }
    };
    const auto upload_path = memory_tracker->ForEachUploadRange(
        device_addr, size, is_written,
        [&](u64 device_addr_out, u64 range_size) {
            copies.emplace_back(total_size_bytes, device_addr_out - buffer_start, range_size);
            total_size_bytes += range_size;
        },
        [&] {
            src_buffer = defer ? ReserveDeferredCopies(buffer, copies, total_size_bytes)
                               : UploadCopies(buffer, copies, total_size_bytes);
        });
    if (g_image_buffer_cost) {
        g_sync_walk_cyc = __rdtsc() - ibc_t0;
    }
    last_sync_upload_bytes = total_size_bytes;
    sync_upload_bytes_total += total_size_bytes;
    if (defer) {
        PageManager::SetDeferredProtectSink(nullptr);
        if (!src_buffer && !deferred_protects.empty()) {
            // Nothing to copy after all; apply the protection here.
            page_manager.ReconcileProtection(deferred_protects);
            deferred_protects.clear();
        }
    }
    if (sample_probe) {
        sample_probe->AddUploadNs(sample_probe->SinceNs(t_upload));
        sample_probe->RecordUpload(upload_path, total_size_bytes, copies.size());
    }

    if (src_buffer) {
        scheduler.EndRendering();
        const vk::BufferMemoryBarrier2 pre_barrier = {
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite |
                             vk::AccessFlagBits2::eTransferRead |
                             vk::AccessFlagBits2::eTransferWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .buffer = buffer.Handle(),
            .offset = 0,
            .size = buffer.SizeBytes(),
        };
        const vk::BufferMemoryBarrier2 post_barrier = {
            .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
            .buffer = buffer.Handle(),
            .offset = 0,
            .size = buffer.SizeBytes(),
        };
        if (defer) {
            scheduler.RecordArrays(
                [src_buffer, destination = buffer.Handle(), pre_barrier, post_barrier,
                 memory = memory, pm = &page_manager](vk::CommandBuffer cmdbuf, auto owned_copies,
                                                      auto cpu_copies, auto protects) {
                    // Protect first, then copy: see SynchronizeBuffer.
                    pm->ReconcileProtection(protects);
                    for (const DeferredCopy& copy : cpu_copies) {
                        if (!memory->TryReadBacking(copy.addr, copy.dest, copy.size)) {
                            memory->CopySparseMemory(copy.addr, copy.dest, copy.size);
                        }
                    }
                    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
                        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
                        .bufferMemoryBarrierCount = 1,
                        .pBufferMemoryBarriers = &pre_barrier,
                    });
                    cmdbuf.copyBuffer(src_buffer, destination, owned_copies);
                    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
                        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
                        .bufferMemoryBarrierCount = 1,
                        .pBufferMemoryBarriers = &post_barrier,
                    });
                },
                std::span<const vk::BufferCopy>{copies},
                std::span<const DeferredCopy>{deferred_copies},
                std::span<const PageManager::ProtectRange>{deferred_protects});
            TouchBuffer(buffer);
        } else {
        scheduler.RecordArrays(
            [src_buffer, destination = buffer.Handle(), pre_barrier,
             post_barrier](vk::CommandBuffer cmdbuf, auto owned_copies) {
                cmdbuf.pipelineBarrier2(vk::DependencyInfo{
                    .dependencyFlags = vk::DependencyFlagBits::eByRegion,
                    .bufferMemoryBarrierCount = 1,
                    .pBufferMemoryBarriers = &pre_barrier,
                });
                cmdbuf.copyBuffer(src_buffer, destination, owned_copies);
                cmdbuf.pipelineBarrier2(vk::DependencyInfo{
                    .dependencyFlags = vk::DependencyFlagBits::eByRegion,
                    .bufferMemoryBarrierCount = 1,
                    .pBufferMemoryBarriers = &post_barrier,
                });
            },
            std::span<const vk::BufferCopy>{copies});
        TouchBuffer(buffer);
        }
    }
    if (is_texel_buffer && !is_written) {
        const auto t_texel = sample_probe ? sample_probe->Now()
                                          : Common::BufferSampleProbe::Clock::time_point{};
        const bool result = SynchronizeBufferFromImage(buffer, device_addr, size);
        if (sample_probe) {
            sample_probe->AddTexelNs(sample_probe->SinceNs(t_texel));
        }
        return result;
    }
    return false;
}

vk::Buffer BufferCache::ReserveDeferredCopies(Buffer& buffer, std::span<vk::BufferCopy> copies,
                                              size_t total_size_bytes) {
    if (copies.empty()) {
        return VK_NULL_HANDLE;
    }
    // Commit flushes non-coherent memory immediately, before a deferred copy would have run.
    const auto [staging, offset] = staging_buffer.is_coherent
                                       ? staging_buffer.Map(total_size_bytes)
                                       : std::pair<u8*, u64>{nullptr, 0};
    if (!staging) {
        // Non-coherent staging or a large one-time transfer: synchronous, protection first.
        page_manager.ReconcileProtection(deferred_protects);
        deferred_protects.clear();
        return UploadCopies(buffer, copies, total_size_bytes);
    }
    for (auto& copy : copies) {
        deferred_copies.push_back(
            {buffer.CpuAddr() + copy.dstOffset, staging + copy.srcOffset, copy.size});
        copy.srcOffset += offset;
    }
    staging_buffer.Commit();
    return staging_buffer.Handle();
}

vk::Buffer BufferCache::UploadCopies(Buffer& buffer, std::span<vk::BufferCopy> copies,
                                     size_t total_size_bytes) {
    if (copies.empty()) {
        return VK_NULL_HANDLE;
    }
    const auto [staging, offset] = staging_buffer.Map(total_size_bytes);
    if (staging) {
        for (auto& copy : copies) {
            u8* const src_pointer = staging + copy.srcOffset;
            const VAddr device_addr = buffer.CpuAddr() + copy.dstOffset;
            memory->CopySparseMemory(device_addr, src_pointer, copy.size);
            // Apply the staging offset
            copy.srcOffset += offset;
        }
        staging_buffer.Commit();
        return staging_buffer.Handle();
    } else {
        // For large one time transfers use a temporary host buffer.
        auto temp_buffer =
            std::make_unique<Buffer>(instance, scheduler, MemoryUsage::Upload, 0,
                                     vk::BufferUsageFlagBits::eTransferSrc, total_size_bytes);
        const vk::Buffer src_buffer = temp_buffer->Handle();
        u8* const staging = temp_buffer->mapped_data.data();
        for (const auto& copy : copies) {
            u8* const src_pointer = staging + copy.srcOffset;
            const VAddr device_addr = buffer.CpuAddr() + copy.dstOffset;
            memory->CopySparseMemory(device_addr, src_pointer, copy.size);
        }
        scheduler.DeferOperation([buffer = std::move(temp_buffer)]() mutable { buffer.reset(); });
        return src_buffer;
    }
}

bool BufferCache::SynchronizeBufferFromImage(Buffer& buffer, VAddr device_addr, u32 size) {
    buffer.NoteMainAccess(scheduler.CurrentTick(), true);
    // TIER-2 SPLIT PROBE: the ob_sync gap (ob_sync - scan/copy/mark = ~7.7 ms/frame) is untimed and
    // was GUESSED to be lock/framing. This path (read-only texel buffers) is the prime suspect.
    // Split it: FindImageFromRange (texture-cache lookup under TextureCache::mutex) vs TileImage
    // (the real work, only when an image actually aliases). Strip before ship.
    static u64 sfi_calls = 0, sfi_found = 0, sfi_find_ns = 0, sfi_tile_ns = 0;
    const auto sfi_t0 = Common::FaultMonitor::Now();
    const ImageId image_id = texture_cache.FindImageFromRange(device_addr, size);
    sfi_find_ns += Common::FaultMonitor::SinceNs(sfi_t0);
    // SHADPS4_LEAN: sfi_calls stays 0, so the SfiSplit line is never formatted or written.
    const bool sfi_stats = !::Common::Lean();
    sfi_calls += sfi_stats;
    const auto sfi_report = [&] {
        if (!sfi_stats || sfi_calls < 300000) {
            return;
        }
        LOG_INFO(Render, "SfiSplit: calls={} found={:.1f}% find={:.3f}us/call tile={:.3f}us/found",
                 sfi_calls, 100.0 * static_cast<double>(sfi_found) / sfi_calls,
                 static_cast<double>(sfi_find_ns) / 1e3 / sfi_calls,
                 sfi_found ? static_cast<double>(sfi_tile_ns) / 1e3 / sfi_found : 0.0);
        sfi_calls = sfi_found = sfi_find_ns = sfi_tile_ns = 0;
    };
    if (!image_id) {
        sfi_report();
        return false;
    }
    ++sfi_found;
    Image& image = texture_cache.GetImage(image_id);
    ASSERT_MSG(device_addr == image.info.guest_address,
               "Texel buffer aliases image subresources {:x} : {:x}", device_addr,
               image.info.guest_address);
    const u32 buf_offset = buffer.Offset(image.info.guest_address);
    boost::container::small_vector<vk::BufferImageCopy, 8> buffer_copies;
    u32 copy_size = 0;
    for (u32 mip = 0; mip < image.info.resources.levels; mip++) {
        const auto& mip_info = image.info.mips_layout[mip];
        const u32 width = std::max(image.info.size.width >> mip, 1u);
        const u32 height = std::max(image.info.size.height >> mip, 1u);
        const u32 depth = std::max(image.info.size.depth >> mip, 1u);
        if (buf_offset + mip_info.offset + mip_info.size > buffer.SizeBytes()) {
            break;
        }
        buffer_copies.push_back(vk::BufferImageCopy{
            .bufferOffset = mip_info.offset,
            .bufferRowLength = mip_info.pitch,
            .bufferImageHeight = mip_info.height,
            .imageSubresource{
                .aspectMask = image.aspect_mask & ~vk::ImageAspectFlagBits::eStencil,
                .mipLevel = mip,
                .baseArrayLayer = 0,
                .layerCount = image.info.resources.layers,
            },
            .imageOffset = {0, 0, 0},
            .imageExtent = {width, height, depth},
        });
        copy_size += mip_info.size;
    }
    if (copy_size == 0) {
        sfi_report();
        return false;
    }
    if (PrebatchEnabled() || PreciseProducerEnabled()) {
        NoteBatchWrite(image.info.guest_address,
                       std::min<u64>(std::max<u64>(copy_size, image.info.guest_size),
                                     buffer.SizeBytes() - buf_offset));
    }
    if (pending_fault_jobs.load(std::memory_order_relaxed) != 0) {
        // TileImage writes this cache buffer with no tracker marking, so the region sequence
        // would not move and a pending fault job over these bytes would still pass its vote.
        // Bump it by hand, before the write is recorded: every publication from here on is
        // vetoed and resolved by a synchronous download recorded after the tiling. A settle
        // would also do, but it waits and can reenter the cache while the caller holds
        // `buffer`. A tiled image binds its whole guest size, so cover that too.
        const u64 written = std::min<u64>(std::max<u64>(copy_size, image.info.guest_size),
                                          buffer.SizeBytes() - buf_offset);
        memory_tracker->InvalidateGpuWriteSeq(image.info.guest_address, written);
        Offload::Add(Offload::State().alias_bumps);
    }
    auto& tile_manager = texture_cache.GetTileManager();
    const auto sfi_t1 = Common::FaultMonitor::Now();
    tile_manager.TileImage(image, buffer_copies, buffer.Handle(), buf_offset, copy_size);
    sfi_tile_ns += Common::FaultMonitor::SinceNs(sfi_t1);
    sfi_report();
    return true;
}

void BufferCache::SynchronizeBuffersInRange(VAddr device_addr, u64 size) {
    const VAddr device_addr_end = device_addr + size;
    ForEachBufferInRange(device_addr, size, [&](BufferId buffer_id, Buffer& buffer) {
        RENDERER_TRACE;
        VAddr start = std::max(buffer.CpuAddr(), device_addr);
        VAddr end = std::min(buffer.CpuAddr() + buffer.SizeBytes(), device_addr_end);
        u32 size = static_cast<u32>(end - start);
        SynchronizeBuffer(buffer, start, size, false, false, nullptr);
    });
}

void BufferCache::WriteDataBuffer(Buffer& buffer, VAddr address, const void* value, u32 num_bytes) {
    NoteBatchWrite(address, num_bytes);
    buffer.NoteMainAccess(scheduler.CurrentTick(), true);
    vk::BufferCopy copy = {
        .srcOffset = 0,
        .dstOffset = buffer.Offset(address),
        .size = num_bytes,
    };
    vk::Buffer src_buffer = staging_buffer.Handle();
    if (num_bytes < StagingBufferSize) {
        const auto [staging, offset] = staging_buffer.Map(num_bytes);
        std::memcpy(staging, value, num_bytes);
        copy.srcOffset = offset;
        staging_buffer.Commit();
    } else {
        // For large one time transfers use a temporary host buffer.
        // RenderDoc can lag quite a bit if the stream buffer is too large.
        Buffer temp_buffer{
            instance, scheduler, MemoryUsage::Upload, 0, vk::BufferUsageFlagBits::eTransferSrc,
            num_bytes};
        src_buffer = temp_buffer.Handle();
        u8* const staging = temp_buffer.mapped_data.data();
        std::memcpy(staging, value, num_bytes);
        scheduler.DeferOperation([buffer = std::move(temp_buffer)]() mutable {});
    }
    scheduler.EndRendering();
    auto command_access = scheduler.AcquireCommandBuffer();
    const auto cmdbuf = command_access.Handle();
    const vk::BufferMemoryBarrier2 pre_barrier = {
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryRead,
        .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .buffer = buffer.Handle(),
        .offset = buffer.Offset(address),
        .size = num_bytes,
    };
    const vk::BufferMemoryBarrier2 post_barrier = {
        .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
        .buffer = buffer.Handle(),
        .offset = buffer.Offset(address),
        .size = num_bytes,
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &pre_barrier,
    });
    cmdbuf.copyBuffer(src_buffer, buffer.Handle(), copy);
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &post_barrier,
    });
}

void BufferCache::RunGarbageCollector() {
    SCOPE_EXIT {
        ++gc_tick;
    };
    if (instance.CanReportMemoryUsage()) {
        total_used_memory = instance.GetDeviceMemoryUsage();
    }
    if (total_used_memory < trigger_gc_memory) {
        return;
    }
    const bool aggressive = total_used_memory >= critical_gc_memory;
    const u64 ticks_to_destroy = std::min<u64>(aggressive ? 80 : 160, gc_tick);
    int max_deletions = aggressive ? 64 : 32;
    const auto clean_up = [&](BufferId buffer_id) {
        if (max_deletions == 0) {
            return;
        }
        --max_deletions;
        // Settle before taking the reference: no in-flight fault job may span this download.
        SettleOverlappingJobs(slot_buffers[buffer_id].CpuAddr(),
                              slot_buffers[buffer_id].SizeBytes());
        Buffer& buffer = slot_buffers[buffer_id];
        // InvalidateMemory(buffer.CpuAddr(), buffer.SizeBytes());
        DownloadBufferMemory<true>(buffer, buffer.CpuAddr(), buffer.SizeBytes(), true);
        DeleteBuffer(buffer_id);
    };
}

void BufferCache::TouchBuffer(const Buffer& buffer) {
    lru_cache.Touch(buffer.LRUId(), gc_tick);
}

void BufferCache::DeleteBuffer(BufferId buffer_id) {
    Buffer& buffer = slot_buffers[buffer_id];
    Unregister(buffer_id);
    // SHADPS4_READBACK_QUEUE=1/2: a copy on the separate readback queue waits only for its
    // producer tick, not for this deletion tick, so it can still be queued when the main queue
    // passes the deletion and the buffer is freed (review 2026-09-27, use-after-free). Before
    // freeing, wait for every readback-queue copy submitted up to now; each waits on a producer
    // below the deletion tick, which has completed when this deferred operation runs.
    const u64 rq_tick = readback_queue && !readback_queue->InQueueOrder()
                            ? readback_queue->Semaphore().CurrentTick() - 1
                            : 0;
    // The fast lane likewise waits only for completed producers.
    const u64 lane_tick = fast_lane ? fast_lane->Semaphore().CurrentTick() - 1 : 0;
    scheduler.DeferOperation([this, buffer_id, rq_tick, lane_tick] {
        if (rq_tick != 0) {
            readback_queue->Semaphore().Wait(rq_tick);
        }
        if (lane_tick != 0) {
            fast_lane->Semaphore().Wait(lane_tick);
        }
        slot_buffers.erase(buffer_id);
    });
    buffer.is_deleted = true;
}

} // namespace VideoCore
