// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <chrono>
#include <cstddef>
#include <immintrin.h>
#include <unordered_set>
#include <boost/preprocessor/stringize.hpp>

#include "common/assert.h"
#include "common/debug.h"
#include "common/fault_monitor.h"
#include "common/fence_watch.h"
#include "video_core/buffer_cache/protect_queue.h"
#include "video_core/buffer_cache/submit_epoch.h"
#include "common/polyfill_thread.h"
#include "common/thread.h"
#include "core/debug_state.h"
#include "core/emulator_settings.h"
#include "core/libraries/kernel/process.h"
#include "core/libraries/videoout/driver.h"
#include "core/memory.h"
#include "core/platform.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/amdgpu/pm4_cmds.h"
#include "common/readback_offload.h"
#include "video_core/amdgpu/split_hazard.h"
#include "video_core/renderdoc.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

namespace Common::FenceWatch {

void NoteRelease(VAddr addr, u64 value, u32 size) {
    auto& st = Get();
    const u64 now = NowNs();
    Slot* slot = nullptr;
    Slot* oldest = &st.slots[0];
    if (addr >= State::PinnedBase && addr < State::PinnedBase + State::NumPinned * 8) {
        slot = &st.slots[State::NumSlots + (addr - State::PinnedBase) / 8];
    }
    for (size_t i = 0; slot == nullptr && i < State::NumSlots; ++i) {
        Slot& s = st.slots[i];
        const u64 a = s.addr.load(std::memory_order_relaxed);
        if (a == addr || a == 0) {
            slot = &s;
            break;
        }
        if (s.written_ns.load(std::memory_order_relaxed) <
            oldest->written_ns.load(std::memory_order_relaxed)) {
            oldest = &s;
        }
    }
    if (slot == nullptr) {
        slot = oldest;
        slot->releases.store(0, std::memory_order_relaxed);
    }
    slot->value.store(value, std::memory_order_relaxed);
    slot->size.store(size, std::memory_order_relaxed);
    slot->written_ns.store(now, std::memory_order_relaxed);
    slot->addr.store(addr, std::memory_order_release);
    slot->releases.fetch_add(1, std::memory_order_relaxed);
    st.last_release_ns.store(now, std::memory_order_release);
    static std::atomic<u64> total{0};
    const u64 n = total.fetch_add(1, std::memory_order_relaxed);
    if (n == 0 || (n & 0x3fff) == 0) {
        LOG_INFO(Render, "FenceWatch: armed; compute releases {} (latest {:#x} <- {:#x})", n + 1, addr, value);
    }
    static std::once_flag started;
    std::call_once(started, [] {
        std::thread([] {
            u64 reported_for = 0;
            u64 last_report_ns = 0;
            for (;;) {
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
                auto& w = Get();
                // The newest write among busy frame-fence slots; without one, any compute release.
                u64 last = 0;
                for (size_t i = State::NumSlots; i < w.slots.size(); ++i) {
                    if (w.slots[i].releases.load(std::memory_order_relaxed) > 100) {
                        last = std::max(last, w.slots[i].written_ns.load(std::memory_order_acquire));
                    }
                }
                if (last == 0) {
                    last = w.last_release_ns.load(std::memory_order_acquire);
                }
                const u64 now = NowNs();
                if (last == 0 || now - last < 2'000'000'000ULL) {
                    continue;
                }
                if (last == reported_for && now - last_report_ns < 10'000'000'000ULL) {
                    continue;
                }
                reported_for = last;
                last_report_ns = now;
                LOG_WARNING(Render, "FenceWatch: no frame-fence release for {:.1f} s (any compute release {:.1f} s ago); "
                            "covering download writes {}; gnm submission_lock {} waiters {} resets {} races {}",
                            double(now - last) / 1e9,
                            double(now - w.last_release_ns.load()) / 1e9, w.covering_writes.load(),
                            w.gnm_lock.load(), w.gnm_waiters.load(), w.gnm_resets.load(),
                            w.gnm_races.load());
                for (const Slot& s : w.slots) {
                    const VAddr a = s.addr.load(std::memory_order_acquire);
                    if (a == 0) {
                        continue;
                    }
                    const u32 sz = s.size.load(std::memory_order_relaxed);
                    const u64 v = s.value.load(std::memory_order_relaxed);
                    u64 live = 0;
                    const bool ok = Core::Memory::Instance()->TryReadBacking(
                        a, reinterpret_cast<u8*>(&live), sz);
                    const char* verdict = !ok          ? "unreadable"
                                          : live == v  ? "equal"
                                          : live < v   ? "LOWER (clobbered?)"
                                                       : "HIGHER";
                    LOG_WARNING(Render,
                                "FenceWatch:   {:#x} last signalled {:#x} ({} B, {:.1f} s ago, {} releases); "
                                "live {:#x}: {}",
                                a, v, sz, double(now - s.written_ns.load()) / 1e9, s.releases.load(),
                                live, verdict);
                }
            }
        }).detach();
    });
}

void NoteGuestWrite(VAddr addr, const void* data, u64 size, const char* who) {
    auto& st = Get();
    for (const Slot& s : st.slots) {
        const VAddr fence = s.addr.load(std::memory_order_acquire);
        const u32 sz = s.size.load(std::memory_order_relaxed);
        if (fence == 0 || fence < addr || fence + sz > addr + size) {
            continue;
        }
        const u64 n = st.covering_writes.fetch_add(1, std::memory_order_relaxed);
        u64 incoming = 0;
        std::memcpy(&incoming, static_cast<const u8*>(data) + (fence - addr), sz);
        const u64 last = s.value.load(std::memory_order_relaxed);
        if (incoming != last || n < 16) {
            LOG_WARNING(Render,
                        "FenceWatch: {} writes {:#x} over fence {:#x} (last signalled {:#x}, {:.3f} ms ago){}",
                        who, incoming, fence, last,
                        double(NowNs() - s.written_ns.load(std::memory_order_relaxed)) / 1e6,
                        incoming != last ? " CLOBBER" : "");
        }
    }
}

} // namespace Common::FenceWatch

namespace AmdGpu {

/// SHADPS4_DRAW_COST=1: cycles the CP back spent in WaitForWork (read and reset by NoteDrawCost).
thread_local u64 g_back_idle_cyc = 0;


namespace {
bool ViewportMemoEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("SHADPS4_VIEWPORT_MEMO");
        return value && value[0] >= '1';
    }();
    return enabled;
}

// SHADPS4_IB_TASK_FREE=1 (port of upstream PR #4996): an INDIRECT_BUFFER packet runs a nested
// coroutine whose final_suspend keeps its frame alive; nothing destroyed it, so every IB packet
// leaked one frame.
bool IbTaskFree() {
    static const bool enabled = [] {
        const char* value = std::getenv("SHADPS4_IB_TASK_FREE");
        return value && value[0] == '1';
    }();
    return enabled;
}

/// SHADPS4_ASC_YIELD=1: the graphics task yields to the round-robin at the next packet boundary
/// when the guest submitted async-compute work since it last looked. On GCN the ACEs parse and
/// run compute queues beside the graphics CP; the emulator's front otherwise parses a whole
/// graphics submission (thousands of draws, at back-thread speed) before an ASC submission gets a
/// turn, unless the graphics stream blocks on a wait. Second Son submits its particle compute and
/// reads the results in the same frame (REFERENCE 4.2 0x29c8d0), so that delay lands on Game:Main.
/// Only the interleaving changes, and the graphics task already yields mid-stream at unsatisfied
/// WAIT_REG_MEM / MEM_SEMAPHORE / REWIND; the round-robin resumes it with the task guards as there.
bool AscYield() {
    static const bool enabled = [] {
        const char* value = std::getenv("SHADPS4_ASC_YIELD");
        return value && value[0] == '1';
    }();
    return enabled;
}

/// SHADPS4_ASC_START_STATS=1 (diagnostic): time from SubmitAsc to the first run of that buffer's
/// task, and the number of ASC_YIELD yields; logged every 2,000 ASC submissions.
bool AscStartStats() {
    static const bool enabled = [] {
        const char* value = std::getenv("SHADPS4_ASC_START_STATS");
        return value && value[0] == '1';
    }();
    return enabled;
}

u64 AscNowNs() {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count());
}

struct AscStartCounters {
    u64 starts = 0, total_ns = 0, max_ns = 0, over_1ms = 0, yields = 0;
    std::array<u64, 6> hist{}; ///< <50 us, <200 us, <500 us, <1 ms, <3 ms, >=3 ms
    /// Split by whether the queue was busy (an earlier submission unfinished) at submit.
    std::array<u64, 2> split_n{}, split_ns{}, split_over_1ms{};
    /// Per mapped queue: starts, total ns, submits that found the queue busy.
    std::array<u64, 64> q_n{}, q_ns{}, q_busy{};
};
AscStartCounters g_asc_start; // front thread only

void NoteAscStart(u64 stamp, u32 queue) {
    auto& c = g_asc_start;
    const bool busy = (stamp & 1) != 0;
    const u64 submit_ns = stamp & ~u64{1};
    const u64 now = AscNowNs();
    const u64 ns = now > submit_ns ? now - submit_ns : 0;
    ++c.split_n[busy];
    c.split_ns[busy] += ns;
    c.split_over_1ms[busy] += ns >= 1'000'000;
    if (queue < c.q_n.size()) {
        ++c.q_n[queue];
        c.q_ns[queue] += ns;
        c.q_busy[queue] += busy;
    }
    ++c.starts;
    c.total_ns += ns;
    c.max_ns = std::max(c.max_ns, ns);
    const u64 us = ns / 1000;
    ++c.hist[us < 50 ? 0 : us < 200 ? 1 : us < 500 ? 2 : us < 1000 ? 3 : us < 3000 ? 4 : 5];
    if (c.starts % 2000 == 0) {
        LOG_INFO(Render, "AscStart: starts={} mean_us={:.1f} max_us={:.1f} gfx_yields={} "
                         "hist[<50,<200,<500,<1000,<3000,>=3000 us]={}/{}/{}/{}/{}/{}",
                 c.starts, double(c.total_ns) / double(c.starts) / 1000.0,
                 double(c.max_ns) / 1000.0, c.yields, c.hist[0], c.hist[1], c.hist[2],
                 c.hist[3], c.hist[4], c.hist[5]);
        std::string per_queue;
        for (u32 q = 0; q < c.q_n.size(); ++q) {
            if (c.q_n[q] != 0) {
                per_queue += fmt::format(" q{}={}/{:.0f}us/busy{}", q, c.q_n[q],
                                         double(c.q_ns[q]) / double(c.q_n[q]) / 1000.0,
                                         c.q_busy[q]);
            }
        }
        LOG_INFO(Render,
                 "AscStartSplit: idle_at_submit n={} mean_us={:.1f} over_1ms={} | busy_at_submit "
                 "n={} mean_us={:.1f} over_1ms={} | per queue (starts/mean/busy):{}",
                 c.split_n[0], c.split_n[0] ? double(c.split_ns[0]) / double(c.split_n[0]) / 1e3 : 0.0,
                 c.split_over_1ms[0], c.split_n[1],
                 c.split_n[1] ? double(c.split_ns[1]) / double(c.split_n[1]) / 1e3 : 0.0,
                 c.split_over_1ms[1], per_queue);
        c = AscStartCounters{};
    }
}

/// SHADPS4_ASC_RING_LATENCY=1 (diagnostic, CP split run-ahead only): where the time between a
/// guest ASC submission (sceGnmDingDong -> SubmitAsc) and the back reaching that submission's
/// RELEASE_MEM label goes. Second Son's Game:Main polls the particle ring's label right after
/// submitting the particle compute (0x1695e0), so this window is on the frame's critical path.
/// The back keeps a class clock (graphics stream / each ASC queue / idle) that changes only at
/// stream switches (Diag records the front emits when its round-robin changes queue) and around
/// WaitForWork; SubmitAsc snapshots it on the guest thread, the back snapshots it at the label.
/// Logged every ~10 s per ASC queue: per submission, and per group of submissions whose submit
/// times lie within AscRingGroupGapUs of each other (one particle update, inferred).
bool AscRingLatency() {
    static const bool enabled = [] {
        const char* value = std::getenv("SHADPS4_ASC_RING_LATENCY");
        return value && value[0] == '1';
    }();
    return enabled;
}

constexpr u32 AscRingQueues = 64;
constexpr u64 AscRingGroupGapUs = 3000;
enum AscRingDiagKind : u32 { AscRingSwitch = 1, AscRingBegin = 2, AscRingEnd = 3 };

/// The class clock at one instant (TSC ticks spent per class since start).
struct AscRingSnap {
    u64 tsc = 0, gfx = 0, asc = 0, own = 0, idle = 0;
};

/// Written by the back only. A seqlock lets SubmitAsc (guest threads) read a consistent snapshot.
struct AscRingClock {
    std::atomic<u32> seq{0};
    std::atomic<u64> since{0};
    std::atomic<u32> queue{0}; ///< Stream the back is executing: 0 = graphics, else ASC queue.
    std::atomic<bool> idle{false};
    std::atomic<u64> gfx{0}, asc{0}, idle_ticks{0};
    std::array<std::atomic<u64>, AscRingQueues> asc_q{};

    void Set(u32 new_queue, bool new_idle) { // back only
        const u64 now = __rdtsc();
        seq.fetch_add(1);
        const u64 start = since.load();
        if (start != 0) {
            const u64 d = now - start;
            if (idle.load()) {
                idle_ticks.fetch_add(d);
            } else if (const u32 q = queue.load(); q == 0) {
                gfx.fetch_add(d);
            } else {
                asc.fetch_add(d);
                asc_q[q % AscRingQueues].fetch_add(d);
            }
        }
        since.store(now);
        queue.store(new_queue);
        idle.store(new_idle);
        seq.fetch_add(1);
    }

    AscRingSnap Snap(u32 q) const {
        for (;;) {
            const u32 s = seq.load();
            if (s & 1) {
                _mm_pause();
                continue;
            }
            AscRingSnap snap;
            snap.gfx = gfx.load();
            snap.asc = asc.load();
            snap.own = asc_q[q % AscRingQueues].load();
            snap.idle = idle_ticks.load();
            const u64 start = since.load();
            const u32 cur = queue.load();
            const bool is_idle = idle.load();
            snap.tsc = __rdtsc();
            if (seq.load() != s) {
                continue;
            }
            if (start != 0 && snap.tsc > start) {
                const u64 d = snap.tsc - start;
                if (is_idle) {
                    snap.idle += d;
                } else if (cur == 0) {
                    snap.gfx += d;
                } else {
                    snap.asc += d;
                    if (cur == q) {
                        snap.own += d;
                    }
                }
            }
            return snap;
        }
    }
};
AscRingClock g_asc_ring_clock;

/// Sums of a class-clock difference (TSC ticks).
struct AscRingSpan {
    u64 wall = 0, gfx = 0, own = 0, other = 0, idle = 0;
    void Add(const AscRingSnap& from, const AscRingSnap& to) {
        wall += to.tsc - from.tsc;
        gfx += to.gfx - from.gfx;
        own += to.own - from.own;
        other += (to.asc - from.asc) - (to.own - from.own);
        idle += to.idle - from.idle;
    }
};

struct AscRingQueueState { // back only
    // The submission being executed (a queue's submissions are parsed one at a time, in order).
    bool open = false;
    AscRingSnap submit{};
    u64 front_tsc = 0, back_tsc = 0;
    u32 labels = 0;
    // The group (submissions close in submit time).
    bool group_open = false;
    AscRingSnap group_first{}, group_last_submit{}, group_label{}, group_label_submit{};
    u32 group_subs = 0, group_labels = 0;
    // The ~10 s window.
    u64 subs = 0, subs_nolabel = 0, label_count = 0, deferred_labels = 0, dispatches = 0;
    u64 to_front = 0, to_back = 0, to_label = 0;
    AscRingSpan sub_span{};
    u64 groups = 0, group_sub_sum = 0, group_submit_span = 0;
    AscRingSpan window{}, tail{};
    std::array<u64, 6> tail_hist{}; ///< <250 us, <500, <1000, <2000, <4000, >= 4000
    VAddr last_label_addr = 0;
};

/// What the front is doing (written by the front, sampled by the back when it runs out of work).
enum AscRingFrontState : u32 { FrontUnknown, FrontGfxTask, FrontAscTask, FrontGuestIdle };
std::atomic<u32> g_asc_ring_front_state{FrontUnknown};

struct AscRingState {
    std::array<AscRingQueueState, AscRingQueues> q{};
    u64 window_tsc = 0;
    u64 window_ns = 0;
    /// Back idle (WaitForWork) by the front's state when the back ran out of work.
    std::array<u64, 4> idle_n{}, idle_ticks{};
    std::array<u64, 5> idle_hist{}; ///< <5 us, <20, <100, <500, >= 500 (approx, 3.5 GHz)
    /// Back time in guest commands (SendCommand: read-fault downloads and the like); it is
    /// part of whatever stream class is current.
    u64 cmd_n = 0, cmd_ticks = 0;
    u64 prev_gfx = 0, prev_asc = 0, prev_idle = 0; ///< Class clock at the window start.
    /// Frame-fence discriminator: TSC of the newest gfx EOP label write (back), and at each
    /// SubmitGfx (guest thread) the time since it. Second Son waits for the previous frame's EOP
    /// label before it submits the next frame (0x1f1e30) with a 50 us poll: if Game:Main is
    /// waiting there, each submit follows a label write within about one poll.
    std::atomic<u64> last_eop_tsc{0};
    std::atomic<u64> sg_n{0}, sg_ticks{0};
    std::array<std::atomic<u64>, 5> sg_hist{}; ///< <100 us, <300, <1000, <3000, >= 3000 (approx)
    u32 front_queue = 0; ///< Front only: the stream of the last Switch marker.
    std::mutex submit_mutex; ///< Guards `submits` (SubmitAsc on guest threads, the front pops).
    std::array<std::deque<AscRingSnap>, AscRingQueues> submits{};
};
AscRingState g_asc_ring;

void EmitAscRingDiag(CpSplit::Channel& channel, u32 kind, u32 queue,
                     const AscRingSnap* snap = nullptr, u64 front_tsc = 0) {
    auto* diag = channel.Reserve<CpSplit::DiagPayload>(CpSplit::RecordType::Diag);
    diag->kind = kind;
    diag->queue = queue;
    if (snap) {
        diag->values[0] = snap->tsc;
        diag->values[1] = snap->gfx;
        diag->values[2] = snap->asc;
        diag->values[3] = snap->own;
        diag->values[4] = snap->idle;
    } else {
        diag->values[0] = diag->values[1] = diag->values[2] = diag->values[3] = 0;
        diag->values[4] = 0;
    }
    diag->values[5] = front_tsc;
    channel.Commit();
}

void AscRingCloseGroup(AscRingQueueState& s) {
    if (s.group_open && s.group_labels != 0) {
        ++s.groups;
        s.group_sub_sum += s.group_subs;
        s.group_submit_span += s.group_last_submit.tsc - s.group_first.tsc;
        s.window.Add(s.group_first, s.group_label);
        const u64 before = s.tail.wall;
        s.tail.Add(s.group_label_submit, s.group_label);
        // Bucketed in ticks here, converted by the logger's ratio only for the means; the
        // histogram uses a fixed 3 ticks/ns guess corrected below when logged (labelled approx).
        const u64 ticks = s.tail.wall - before;
        const u64 us = ticks / 3500; // ~3.5 GHz TSC (i5-13600K); the means use the measured ratio
        ++s.tail_hist[us < 250 ? 0 : us < 500 ? 1 : us < 1000 ? 2 : us < 2000 ? 3 : us < 4000 ? 4 : 5];
    }
    s.group_open = false;
    s.group_subs = s.group_labels = 0;
}

/// Returns true when it logged a window (every ~10 s).
bool AscRingLog() {
    auto& g = g_asc_ring;
    const u64 now_tsc = __rdtsc();
    const u64 now_ns = AscNowNs();
    if (g.window_tsc == 0) {
        g.window_tsc = now_tsc;
        g.window_ns = now_ns;
        return false;
    }
    if (now_ns - g.window_ns < 10'000'000'000ull) {
        return false;
    }
    const double ticks_per_us =
        double(now_tsc - g.window_tsc) / (double(now_ns - g.window_ns) / 1000.0);
    const auto us = [&](u64 ticks, u64 n) {
        return n ? double(ticks) / ticks_per_us / double(n) : 0.0;
    };
    for (u32 q = 1; q < AscRingQueues; ++q) {
        auto& s = g.q[q];
        if (s.subs == 0 && s.groups == 0) {
            continue;
        }
        const u64 labelled = s.subs - s.subs_nolabel;
        LOG_INFO(Render,
                 "AscRing q{}: window_s={:.1f} tsc_mhz={:.0f} subs={} no_label={} labels={} "
                 "deferred_labels={} dispatches={} label_addr={:#x} | per labelled sub (us): "
                 "submit->front={:.0f} submit->back={:.0f} submit->label={:.0f} [gfx={:.0f} "
                 "own={:.0f} other_asc={:.0f} idle={:.0f}]",
                 q, double(now_ns - g.window_ns) / 1e9, ticks_per_us, s.subs, s.subs_nolabel,
                 s.label_count, s.deferred_labels, s.dispatches, s.last_label_addr,
                 us(s.to_front, labelled), us(s.to_back, labelled), us(s.to_label, labelled),
                 us(s.sub_span.gfx, labelled), us(s.sub_span.own, labelled),
                 us(s.sub_span.other, labelled), us(s.sub_span.idle, labelled));
        LOG_INFO(Render,
                 "AscRingGroup q{}: groups={} subs_per_group={:.1f} dispatches_per_group={:.0f} | "
                 "mean us: submit_span={:.0f} | first_submit->last_label={:.0f} [gfx={:.0f} "
                 "own={:.0f} other_asc={:.0f} idle={:.0f}] | last_submit->last_label={:.0f} "
                 "[gfx={:.0f} own={:.0f} other_asc={:.0f} idle={:.0f}] "
                 "tail_hist_approx[<250,<500,<1000,<2000,<4000,>=4000 us]={}/{}/{}/{}/{}/{}",
                 q, s.groups, s.groups ? double(s.group_sub_sum) / double(s.groups) : 0.0,
                 s.groups ? double(s.dispatches) / double(s.groups) : 0.0,
                 us(s.group_submit_span, s.groups), us(s.window.wall, s.groups),
                 us(s.window.gfx, s.groups), us(s.window.own, s.groups),
                 us(s.window.other, s.groups), us(s.window.idle, s.groups),
                 us(s.tail.wall, s.groups), us(s.tail.gfx, s.groups), us(s.tail.own, s.groups),
                 us(s.tail.other, s.groups), us(s.tail.idle, s.groups), s.tail_hist[0],
                 s.tail_hist[1], s.tail_hist[2], s.tail_hist[3], s.tail_hist[4], s.tail_hist[5]);
        // Reset the window sums; keep the open submission and group.
        s.subs = s.subs_nolabel = s.label_count = s.deferred_labels = s.dispatches = 0;
        s.to_front = s.to_back = s.to_label = 0;
        s.sub_span = {};
        s.groups = s.group_sub_sum = s.group_submit_span = 0;
        s.window = {};
        s.tail = {};
        s.tail_hist = {};
    }
    LOG_INFO(Render,
             "AscRingBackIdle: window_s={:.1f} back idle ms (waits) by front state at idle start: "
             "front_in_gfx_task={:.1f} ({}) front_in_asc_task={:.1f} ({}) front_waits_for_guest="
             "{:.1f} ({}) unknown={:.1f} ({}) | idle_hist_approx[<5,<20,<100,<500,>=500 us]="
             "{}/{}/{}/{}/{}",
             double(now_ns - g.window_ns) / 1e9, us(g.idle_ticks[FrontGfxTask], 1) / 1e3,
             g.idle_n[FrontGfxTask], us(g.idle_ticks[FrontAscTask], 1) / 1e3,
             g.idle_n[FrontAscTask], us(g.idle_ticks[FrontGuestIdle], 1) / 1e3,
             g.idle_n[FrontGuestIdle], us(g.idle_ticks[FrontUnknown], 1) / 1e3,
             g.idle_n[FrontUnknown], g.idle_hist[0], g.idle_hist[1], g.idle_hist[2],
             g.idle_hist[3], g.idle_hist[4]);
    {
        // The class clock over the window (whole back wall time: every tick is in one class).
        const AscRingSnap snap = g_asc_ring_clock.Snap(0);
        LOG_INFO(Render,
                 "AscRingBackSplit: window_s={:.1f} back ms: gfx_stream={:.1f} asc_streams={:.1f} "
                 "idle={:.1f} | of which guest commands (SendCommand) ms={:.1f} n={}",
                 double(now_ns - g.window_ns) / 1e9, us(snap.gfx - g.prev_gfx, 1) / 1e3,
                 us(snap.asc - g.prev_asc, 1) / 1e3, us(snap.idle - g.prev_idle, 1) / 1e3,
                 us(g.cmd_ticks, 1) / 1e3, g.cmd_n);
        g.prev_gfx = snap.gfx;
        g.prev_asc = snap.asc;
        g.prev_idle = snap.idle;
        g.cmd_n = g.cmd_ticks = 0;
    }
    {
        const u64 n = g.sg_n.exchange(0, std::memory_order_relaxed);
        const u64 ticks = g.sg_ticks.exchange(0, std::memory_order_relaxed);
        std::array<u64, 5> h{};
        for (u32 i = 0; i < h.size(); ++i) {
            h[i] = g.sg_hist[i].exchange(0, std::memory_order_relaxed);
        }
        LOG_INFO(Render,
                 "AscRingFrameFence: gfx_submits={} mean_us_since_last_eop_label={:.0f} "
                 "hist_approx[<100,<300,<1000,<3000,>=3000 us]={}/{}/{}/{}/{}",
                 n, us(ticks, n), h[0], h[1], h[2], h[3], h[4]);
    }
    g.idle_n = {};
    g.idle_ticks = {};
    g.idle_hist = {};
    g.window_tsc = now_tsc;
    g.window_ns = now_ns;
    return true;
}

/// Back: around WaitForWork.
u32 AscRingIdleEnter() {
    g_asc_ring_clock.Set(g_asc_ring_clock.queue.load(), true);
    return g_asc_ring_front_state.load(std::memory_order_relaxed);
}
void AscRingIdleExit(u32 front_state) {
    const u64 start = g_asc_ring_clock.since.load();
    g_asc_ring_clock.Set(g_asc_ring_clock.queue.load(), false);
    const u64 ticks = g_asc_ring_clock.since.load() - start;
    auto& g = g_asc_ring;
    ++g.idle_n[front_state & 3];
    g.idle_ticks[front_state & 3] += ticks;
    const u64 us = ticks / 3500;
    ++g.idle_hist[us < 5 ? 0 : us < 20 ? 1 : us < 100 ? 2 : us < 500 ? 3 : 4];
}

/// Back: a Diag record of the ring-latency instrument, in stream order. True: a window was logged.
bool AscRingBack(const CpSplit::DiagPayload& diag) {
    const u32 q = diag.queue % AscRingQueues;
    auto& s = g_asc_ring.q[q];
    switch (diag.kind) {
    case AscRingSwitch:
        g_asc_ring_clock.Set(diag.queue, false);
        break;
    case AscRingBegin: {
        const AscRingSnap submit{diag.values[0], diag.values[1], diag.values[2], diag.values[3],
                                 diag.values[4]};
        const u64 gap_ticks = AscRingGroupGapUs * 3500; // ~3.5 GHz TSC, see AscRingCloseGroup
        if (s.group_open && submit.tsc - s.group_last_submit.tsc > gap_ticks) {
            AscRingCloseGroup(s);
        }
        if (!s.group_open) {
            s.group_open = true;
            s.group_first = submit;
        }
        s.group_last_submit = submit;
        ++s.group_subs;
        s.open = true;
        s.submit = submit;
        s.front_tsc = diag.values[5];
        s.back_tsc = __rdtsc();
        s.labels = 0;
        break;
    }
    case AscRingEnd:
        if (s.open) {
            ++s.subs;
            s.subs_nolabel += s.labels == 0;
            s.open = false;
        }
        return AscRingLog();
    default:
        break;
    }
    return false;
}

/// Back: an ASC RELEASE_MEM (the queue's completion label) is being executed.
void AscRingLabel(VAddr address, bool deferred_to_gpu) {
    const u32 q = g_asc_ring_clock.queue.load() % AscRingQueues;
    auto& s = g_asc_ring.q[q];
    ++s.label_count;
    s.deferred_labels += deferred_to_gpu;
    s.last_label_addr = address;
    if (!s.open) {
        return;
    }
    const AscRingSnap label = g_asc_ring_clock.Snap(q);
    if (s.labels++ == 0) {
        s.to_front += s.front_tsc - s.submit.tsc;
        s.to_back += s.back_tsc - s.submit.tsc;
        s.to_label += label.tsc - s.submit.tsc;
        s.sub_span.Add(s.submit, label);
    }
    if (s.group_open) {
        s.group_label = label;
        s.group_label_submit = s.submit;
        ++s.group_labels;
    }
}

void PrefetchProgram(const ShaderProgram& program) {
    const auto* code = program.Address<const char*>();
    if (!code) {
        return;
    }
    _mm_prefetch(code, _MM_HINT_T0);
    const void* info = g_bininfo_hints[BinaryInfoHintSlot(code)].load(std::memory_order_relaxed);
    if (info) {
        _mm_prefetch(static_cast<const char*>(info), _MM_HINT_T0);
        _mm_prefetch(static_cast<const char*>(info) + 63, _MM_HINT_T0);
    }
}

// Called after SET_SH_REG wrote words [offset, offset + count) of the SH register space.
void PrefetchProgramsOnWrite(const Regs& regs, u32 offset, u32 count) {
    constexpr u32 ps_lo = offsetof(Regs, ps_program) / sizeof(u32) - Regs::ShRegWordOffset;
    constexpr u32 vs_lo = offsetof(Regs, vs_program) / sizeof(u32) - Regs::ShRegWordOffset;
    const u32 end = offset + count;
    // The 40-bit address spans the LO and HI words.
    if (offset <= ps_lo + 1 && ps_lo < end) {
        PrefetchProgram(regs.ps_program);
    }
    if (offset <= vs_lo + 1 && vs_lo < end) {
        PrefetchProgram(regs.vs_program);
    }
}
} // namespace

/// SHADPS4_FENCE_AT_GPU=1: end-of-pipe/end-of-shader fences (EVENT_WRITE_EOP, EVENT_WRITE_EOS,
/// RELEASE_MEM) are written, and their interrupts raised, when the host GPU has actually
/// finished the preceding work, as on the real GPU - not when the CP parses the packet. Needed
/// once the GPU reads guest memory at execution time (SHADPS4_HOST_MIRROR): with CP-time fences
/// the game may reuse a buffer the host GPU has not read yet. GDS-sourced fences stay CP-time.
/// Mode 1 submits the batch at every fence; mode 2 defers without submitting (see
/// Rasterizer::DeferToGpu) and flushes before the CP waits on memory or goes idle; mode 3 is
/// mode 1 for EVENT_WRITE_EOP only (EOS and RELEASE_MEM stay CP-time).
static int FenceAtGpu() {
    static const int mode = [] {
        const char* value = std::getenv("SHADPS4_FENCE_AT_GPU");
        return value && value[0] >= '1' && value[0] <= '3' ? value[0] - '0' : 0;
    }();
    return mode;
}

static void WriteFenceValue(void* address, u64 data, u32 num_bytes) {
    auto* memory = Core::Memory::Instance();
    if (!memory->TryWriteBacking(address, &data, num_bytes)) {
        memcpy(address, &data, num_bytes);
    }
}

namespace {
/// SHADPS4_CP_WAIT_STATS=1: where the command processor's wall time goes. WaitRegMem waits are
/// split by whether the polled address is one the CP itself writes (EOS/EOP/RELEASE_MEM/
/// WRITE_DATA: a GPU-side dependency) or not (written by guest CPU code). Wait intervals
/// overlap with other queues' work done between yields; they are latencies, not CPU cost.
struct CpWaitStats {
    bool enabled = [] {
        const char* v = std::getenv("SHADPS4_CP_WAIT_STATS");
        return v && v[0] == '1';
    }();
    std::unordered_set<VAddr> cp_written;
    u64 idle_ns = 0;
    u64 gfx_wrm_cp_ns = 0, gfx_wrm_cpu_ns = 0, gfx_wrm_cp_n = 0, gfx_wrm_cpu_n = 0;
    u64 asc_wrm_cp_ns = 0, asc_wrm_cpu_ns = 0, asc_wrm_cp_n = 0, asc_wrm_cpu_n = 0;
    u64 gfx_wrm_instant = 0;
    u64 labels = 0;
    u64 window_start = 0;

    static u64 Now() {
        return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    std::chrono::steady_clock::now().time_since_epoch())
                                    .count());
    }
    void NoteWrite(const void* addr) {
        if (enabled) {
            cp_written.insert(reinterpret_cast<VAddr>(addr));
            ++labels;
        }
    }
    void CountWait(bool gfx, const void* addr, u64 ns) {
        const bool by_cp = cp_written.contains(reinterpret_cast<VAddr>(addr));
        if (gfx) {
            (by_cp ? gfx_wrm_cp_ns : gfx_wrm_cpu_ns) += ns;
            ++(by_cp ? gfx_wrm_cp_n : gfx_wrm_cpu_n);
        } else {
            (by_cp ? asc_wrm_cp_ns : asc_wrm_cpu_ns) += ns;
            ++(by_cp ? asc_wrm_cp_n : asc_wrm_cpu_n);
        }
        MaybeLog();
    }
    void MaybeLog() {
        const u64 now = Now();
        if (window_start == 0) {
            window_start = now;
            return;
        }
        const u64 span = now - window_start;
        if (span < 10'000'000'000ULL) {
            return;
        }
        const auto pct = [span](u64 ns) { return 100.0 * double(ns) / double(span); };
        LOG_INFO(Render, "CpWait: window_s={:.1f} idle_pct={:.1f} gfx_wrm_cp_pct={:.1f} "
                 "gfx_wrm_cpu_pct={:.1f} gfx_wrm_cp_n={} gfx_wrm_cpu_n={} gfx_wrm_instant={} "
                 "asc_wrm_cp_pct={:.1f} asc_wrm_cpu_pct={:.1f} asc_wrm_cp_n={} asc_wrm_cpu_n={} "
                 "labels={} label_addrs={}",
                 double(span) / 1e9, pct(idle_ns), pct(gfx_wrm_cp_ns), pct(gfx_wrm_cpu_ns),
                 gfx_wrm_cp_n, gfx_wrm_cpu_n, gfx_wrm_instant, pct(asc_wrm_cp_ns),
                 pct(asc_wrm_cpu_ns), asc_wrm_cp_n, asc_wrm_cpu_n, labels, cp_written.size());
        idle_ns = gfx_wrm_cp_ns = gfx_wrm_cpu_ns = gfx_wrm_cp_n = gfx_wrm_cpu_n = 0;
        asc_wrm_cp_ns = asc_wrm_cpu_ns = asc_wrm_cp_n = asc_wrm_cpu_n = 0;
        gfx_wrm_instant = labels = 0;
        window_start = now;
    }
};
CpWaitStats g_cp_wait;
} // namespace

namespace {
// ---- CP front/back split records (cp_split.h). Built on the front, read on the back. ----

/// A draw: the front's classification and lookup plus the per-stage data copied right after the
/// lookup. Followed by stage_count stage blocks (WriteStage).
struct SplitDrawRecord {
    Vulkan::Rasterizer::SplitDraw draw;
    u64 ctx_epoch;           ///< Liverpool::context_change_epoch at the draw (viewport memo).
    const char* marker;      ///< Host marker packet name (static string), never dereferenced
                             ///< on the front's behalf; null = no marker.
    const void* cmd_address; ///< For the host marker text only (formatted, never read).
    u32 stage_count;
    u32 verify;              ///< DrawVerifyFull / DrawVerifySnapshot (VERIFY only), else 0.
    u32 vi_bytes;            ///< FRONT_DERIVE: bytes of the VertexInputsPayload block after the
                             ///< stage blocks (0: none).
    u32 pad;
};
static_assert(sizeof(SplitDrawRecord) % 8 == 0);

/// SplitDrawRecord::verify. Full: the front is parked on this record (depth 0, or the periodic
/// synchronous run-ahead draw), so the back compares against the front's live state. Snapshot:
/// a posted draw; a register-window copy follows the stage blocks.
constexpr u32 DrawVerifyFull = 1u << 0;
constexpr u32 DrawVerifySnapshot = 1u << 1;

/// Run-ahead VERIFY: the register ranges the back's draw path reads (inventory 4.3), copied by
/// the front into each posted draw. Gfx SH (ps_program .. cs_program), all context registers,
/// and uconfig up to the VGT draw state (primitive type, num_indices, num_instances, TF base).
/// cs registers never go through regs (they live in mapped_queues[].cs_state).
struct RegWindow {
    u32 begin;
    u32 end;
};
constexpr u32 WordOf(size_t byte_offset) {
    return static_cast<u32>(byte_offset / sizeof(u32));
}
constexpr std::array<RegWindow, 3> VerifyWindows{{
    {WordOf(offsetof(Regs, ps_program)), WordOf(offsetof(Regs, cs_program))},
    {Regs::ContextRegWordOffset, Regs::ContextRegWordOffset + 0x400},
    {Regs::UconfigRegWordOffset,
     WordOf(offsetof(Regs, vgt_tf_memory_base) + sizeof(Regs::vgt_tf_memory_base))},
}};
constexpr u32 VerifyWindowDwords() {
    u32 dwords = 0;
    for (const auto& window : VerifyWindows) {
        dwords += window.end - window.begin;
    }
    return dwords;
}
/// Extents (8 CB + 1 DB) then the windows, padded to 8 bytes.
constexpr u32 VerifySnapshotBytes =
    ((NUM_COLOR_BUFFERS + 1 + VerifyWindowDwords()) * sizeof(u32) + 7) & ~u32{7};
static_assert(VerifyWindows[0].begin < VerifyWindows[0].end &&
              VerifyWindows[2].end <= Regs::NumRegs);

/// A dispatch (gfx queue, or ASC with POST_ASC). `cs` is the front's cs registers at the packet
/// (GetCsRegs() depends on the front's curr_qid; inventory section 0.3). Followed by at most one
/// stage block, then (VERIFY snapshot only) the register-window copy.
struct SplitDispatchRecord {
    ComputeProgram cs;
    const Vulkan::ComputePipeline* pipeline;
    const char* marker;
    const void* cmd_address;
    VAddr address;
    u32 offset;
    u32 size;
    u32 is_indirect;
    u32 stage_count;
    u32 queue;  ///< mapped_queues index: 0 = gfx, vqid + 1 = ASC.
    u32 verify; ///< DrawVerifyFull / DrawVerifySnapshot (VERIFY only), else 0.
    u64 run;    ///< SHADPS4_GUEST_DISPATCH_ORDER guest run id, else 0.
};
static_assert(sizeof(SplitDispatchRecord) % 8 == 0);

u32 StageBytes(const Shader::Info& info) {
    const size_t bytes = sizeof(CpSplit::StagePayload) +
                         (info.user_data.size() + info.flattened_ud_buf.size()) * sizeof(u32);
    return static_cast<u32>((bytes + 7) & ~size_t{7});
}

/// Copies what the back reads of a stage's per-draw data (Shader::BoundStage): user data,
/// flat buffer and pgm_base, as the lookup just left them in the shared Info.
u8* WriteStage(u8* out, const Shader::Info& info, u32 logical_stage) {
    auto* stage = reinterpret_cast<CpSplit::StagePayload*>(out);
    stage->info = &info;
    stage->pgm_base = info.pgm_base;
    stage->logical_stage = logical_stage;
    stage->ud_dw = static_cast<u32>(info.user_data.size());
    stage->flat_dw = static_cast<u32>(info.flattened_ud_buf.size());
    stage->pad = 0;
    auto* dwords = reinterpret_cast<u32*>(stage + 1);
    if (!info.user_data.empty()) {
        std::memcpy(dwords, info.user_data.data(), info.user_data.size_bytes());
    }
    if (!info.flattened_ud_buf.empty()) {
        std::memcpy(dwords + stage->ud_dw, info.flattened_ud_buf.data(),
                    info.flattened_ud_buf.size() * sizeof(u32));
    }
    return out + StageBytes(info);
}

/// The back's BoundStages, pointing into the record (valid until Channel::Done). Returns the
/// end of the last stage block.
const u8* ReadStages(const u8* in, u32 count,
                     std::array<Shader::BoundStage, Shader::MaxStageTypes>& out) {
    for (u32 i = 0; i < count; ++i) {
        const auto* stage = reinterpret_cast<const CpSplit::StagePayload*>(in);
        const auto* dwords = reinterpret_cast<const u32*>(stage + 1);
        ASSERT(stage->logical_stage < Shader::MaxStageTypes);
        auto& bound = out[stage->logical_stage];
        bound.info = static_cast<const Shader::Info*>(stage->info);
        bound.user_data = std::span<const u32>{dwords, stage->ud_dw};
        bound.flattened_ud_buf = std::span<const u32>{dwords + stage->ud_dw, stage->flat_dw};
        bound.pgm_base = stage->pgm_base;
        const size_t bytes = sizeof(CpSplit::StagePayload) +
                             (size_t{stage->ud_dw} + stage->flat_dw) * sizeof(u32);
        in += (bytes + 7) & ~size_t{7};
    }
    return in;
}

/// Run-ahead VERIFY, front: the extents and register windows the back must match (see
/// VerifyWindows). `out` is 8-aligned (stage blocks end 8-aligned).
void WriteVerifySnapshot(u8* out, const Regs& regs,
                         const std::array<CbDbExtent, NUM_COLOR_BUFFERS>& cb_extent,
                         const CbDbExtent& db_extent) {
    auto* words = reinterpret_cast<u32*>(out);
    for (u32 cb = 0; cb < NUM_COLOR_BUFFERS; ++cb) {
        *words++ = cb_extent[cb].raw;
    }
    *words++ = db_extent.raw;
    for (const auto& window : VerifyWindows) {
        const u32 count = window.end - window.begin;
        std::memcpy(words, &regs.reg_array[window.begin], size_t{count} * sizeof(u32));
        words += count;
    }
}

/// VERIFY: the Info still holds what the front's lookup left there. Only valid while the front
/// is parked on this very record (VerifyFull): at run-ahead the front's later lookups rewrite it.
void VerifyStages(const std::array<Shader::BoundStage, Shader::MaxStageTypes>& stages, u64 seq) {
    for (const auto& bound : stages) {
        if (!bound.info) {
            continue;
        }
        const auto& info = *bound.info;
        ASSERT_MSG(bound.pgm_base == info.pgm_base &&
                       std::ranges::equal(bound.user_data, info.user_data) &&
                       std::ranges::equal(bound.flattened_ud_buf, info.flattened_ud_buf),
                   "CpSplit VERIFY: stage {} data differs from its Info at record {}",
                   static_cast<u32>(info.l_stage), seq);
    }
}

/// VERIFY with POST_ASC, on a synchronous dispatch record (the front is parked on it, and the back
/// has executed everything published before it): re-runs the front's compute lookup on the back
/// and compares it with what the record carries. A difference means the front's lookup read
/// inputs (user-data / SRT tables or shader code in guest memory) that earlier posted work had not
/// written yet: the run-ahead read hazard. The guest CPU writing those tables in between also
/// shows up here, so the caller logs instead of asserting. Reads the front's pipeline cache and
/// cs_state (GetCsRegs), which is sound only while the front is parked.
bool DispatchRelookupMatches(Vulkan::Rasterizer& rasterizer,
                             const Vulkan::ComputePipeline* carried_pipeline,
                             const Shader::BoundStage& carried) {
    const Vulkan::ComputePipeline* pipeline = rasterizer.PrepareDispatchFront();
    if (pipeline != carried_pipeline) {
        return false;
    }
    if (!pipeline) {
        return true;
    }
    const Shader::Info* info = pipeline->GetStages()[0];
    if (!info) {
        return carried.info == nullptr;
    }
    return carried.info == info && carried.pgm_base == info->pgm_base &&
           std::ranges::equal(carried.user_data, info->user_data) &&
           std::ranges::equal(carried.flattened_ud_buf, info->flattened_ud_buf);
}

/// What DrawRelookup found. `what` is null when the re-lookup agrees with the record.
struct DrawRelookupDiff {
    const char* what{};
    u32 stage{};       ///< LogicalStage, for the per-stage differences.
    u32 index{};       ///< First differing dword (user data / flat buffer).
    u32 carried_dw{};  ///< Dword count the record carries.
    u32 relooked_dw{}; ///< Dword count the re-lookup produced.
};

u32 FirstDifferingDword(std::span<const u32> a, std::span<const u32> b) {
    const size_t n = std::min(a.size(), b.size());
    size_t i = 0;
    while (i < n && a[i] == b[i]) {
        ++i;
    }
    return static_cast<u32>(i);
}

/// The draw equivalent of DispatchRelookupMatches. VERIFY at run-ahead, on the periodic
/// synchronous draw record: the front is parked on this record and the back has executed every
/// record published before it, so guest memory now holds what that earlier posted work writes (a
/// GPU-side write is marked GPU-modified, and reading it here takes the ordinary back-thread
/// read fault and download, as the unsplit path would at this stream position). The front's
/// lookup for this draw ran BEFORE its Sync (EmitDraw), while that work could still be
/// outstanding; so a stale read of user-data / SRT tables, fetch V# tables or shader code there
/// shows up as a difference here. The guest CPU writing those tables between the two lookups
/// shows up too, so the caller logs instead of asserting.
///
/// `regs` is the FRONT's register file (what GetGraphicsPipeline reads); VerifyMirror has just
/// proven it equal to the back's mirror. Re-runs the pure classifier and the pipeline lookup only
/// (Rasterizer::RelookupDrawFront): not the hazard note or the empty-draw counters (the skip
/// decision is a pure function of the registers, so it cannot differ). Side effects, sound only
/// while the front is parked (as for DispatchRelookupMatches): it rewrites the pipeline cache's
/// scratch selection (key, infos, modules, fetch shader) and the stages' Info user data / flat
/// buffer / pgm_base, all of which the front's next lookup rewrites and which the back reads
/// only from records; it may bump prepared-pipeline LRU/stat counters or seed a prepared entry
/// (guarded memo: performance-only), and on a difference it may compile a pipeline (the cache
/// is keyed by content). The Info comparison must therefore run after VerifyStages.
DrawRelookupDiff DrawRelookup(Vulkan::Rasterizer& rasterizer, const Regs& regs,
                              const Vulkan::Rasterizer::SplitDraw& carried_draw,
                              const std::array<Shader::BoundStage, Shader::MaxStageTypes>& carried) {
    using Vulkan::Rasterizer;
    if (Rasterizer::ClassifyDraw(regs) != carried_draw.draw_class) {
        return {.what = "draw class"};
    }
    if (carried_draw.draw_class != Rasterizer::DrawClass::Draw || carried_draw.skip) {
        // The front ran no lookup (EmitDraw carried no stages).
        return {};
    }
    const Vulkan::GraphicsPipeline* pipeline = rasterizer.RelookupDrawFront();
    if (pipeline != carried_draw.pipeline) {
        return {.what = "pipeline"};
    }
    if (!pipeline) {
        return {};
    }
    const auto infos = pipeline->GetStages();
    for (u32 i = 0; i < infos.size() && i < carried.size(); ++i) {
        const Shader::Info* info = infos[i];
        const Shader::BoundStage& bound = carried[i];
        if (bound.info != info) {
            return {.what = "stage Info", .stage = i};
        }
        if (!info) {
            continue;
        }
        if (bound.pgm_base != info->pgm_base) {
            return {.what = "pgm_base", .stage = i};
        }
        const std::span<const u32> ud = info->user_data;
        if (!std::ranges::equal(bound.user_data, ud)) {
            return {.what = "user data",
                    .stage = i,
                    .index = FirstDifferingDword(bound.user_data, ud),
                    .carried_dw = static_cast<u32>(bound.user_data.size()),
                    .relooked_dw = static_cast<u32>(ud.size())};
        }
        const std::span<const u32> flat{info->flattened_ud_buf.data(),
                                        info->flattened_ud_buf.size()};
        if (!std::ranges::equal(bound.flattened_ud_buf, flat)) {
            return {.what = "flat buffer",
                    .stage = i,
                    .index = FirstDifferingDword(bound.flattened_ud_buf, flat),
                    .carried_dw = static_cast<u32>(bound.flattened_ud_buf.size()),
                    .relooked_dw = static_cast<u32>(flat.size())};
        }
    }
    return {};
}

/// A NOP whose payload has a back-side effect (flip IRQ, guest markers). Other NOPs (CB/DB
/// extent hints and the like) are front-only.
bool NopHasBackEffect(const PM4Header* header, bool guest_markers_enabled) {
    const auto* nop = reinterpret_cast<const PM4CmdNop*>(header);
    if (nop->header.count.Value() == 0) {
        return false;
    }
    switch (nop->data_block[0]) {
    case PM4CmdNop::PayloadType::PatchedFlip:
        return true;
    case PM4CmdNop::PayloadType::DebugMarkerPush:
    case PM4CmdNop::PayloadType::DebugColorMarkerPush:
    case PM4CmdNop::PayloadType::DebugMarkerPop:
        return guest_markers_enabled;
    default:
        return false;
    }
}
} // namespace

template <typename F>
void Liverpool::OnBack(F&& fn, CpSplit::DrainReason reason) {
    if (split_enabled) [[unlikely]] {
        // Synchronous at every depth: fn and what it captures live on the front's stack.
        split->CallSync(fn, reason);
    } else {
        fn();
    }
}

static const char* dcb_task_name{"DCB_TASK"};
static const char* ccb_task_name{"CCB_TASK"};

#define MAX_NAMES 56
static_assert(Liverpool::NumComputeRings <= MAX_NAMES);

#define NAME_NUM(z, n, name) BOOST_PP_STRINGIZE(name) BOOST_PP_STRINGIZE(n),
#define NAME_ARRAY(name, num) {BOOST_PP_REPEAT(num, NAME_NUM, name)}

static const char* acb_task_name[] = NAME_ARRAY(ACB_TASK, MAX_NAMES);

#define YIELD(name)                                                                                \
    FIBER_EXIT;                                                                                    \
    co_yield {};                                                                                   \
    FIBER_ENTER(name);

#define YIELD_CE() YIELD(ccb_task_name)
#define YIELD_GFX() YIELD(dcb_task_name)
#define YIELD_ASC(id) YIELD(acb_task_name[id])

#define RESUME(task, name)                                                                         \
    FIBER_EXIT;                                                                                    \
    task.handle.resume();                                                                          \
    FIBER_ENTER(name);

#define RESUME_CE(task) RESUME(task, ccb_task_name)
#define RESUME_GFX(task) RESUME(task, dcb_task_name)
#define RESUME_ASC(task, id) RESUME(task, acb_task_name[id])

std::array<u8, 48_KB> Liverpool::ConstantEngine::constants_heap;

/// Packets that neither write guest memory from the CP nor wait on it: register state, draws,
/// dispatches, cache-control events and nested IBs (which bump at their own start). Any other
/// packet closes the upload-deferral window (submit_epoch.h). Conservative on purpose: an extra
/// bump only uploads a deferred page earlier, a missing one could upload it too late.
static constexpr bool KeepsSubmitEpoch(PM4ItOpcode opcode) {
    switch (opcode) {
    case PM4ItOpcode::Nop:
    case PM4ItOpcode::SetContextReg:
    case PM4ItOpcode::SetShReg:
    case PM4ItOpcode::SetUconfigReg:
    case PM4ItOpcode::SetConfigReg:
    case PM4ItOpcode::SetQueueReg:
    case PM4ItOpcode::ContextControl:
    case PM4ItOpcode::ClearState:
    case PM4ItOpcode::SetPredication:
    case PM4ItOpcode::IndexType:
    case PM4ItOpcode::IndexBase:
    case PM4ItOpcode::IndexBufferSize:
    case PM4ItOpcode::NumInstances:
    case PM4ItOpcode::SetBase:
    case PM4ItOpcode::DrawIndex2:
    case PM4ItOpcode::DrawIndexOffset2:
    case PM4ItOpcode::DrawIndexAuto:
    case PM4ItOpcode::DrawIndirect:
    case PM4ItOpcode::DrawIndirectMulti:
    case PM4ItOpcode::DrawIndexIndirect:
    case PM4ItOpcode::DrawIndexIndirectMulti:
    case PM4ItOpcode::DrawIndexIndirectCountMulti:
    case PM4ItOpcode::DispatchDirect:
    case PM4ItOpcode::DispatchIndirect:
    // GPU copy/fill; its CPU fast paths clear deferral for the bytes they write themselves
    // (BufferCache::FillBuffer/CopyBuffer). Measured ~3,100 per frame in the forest.
    case PM4ItOpcode::DmaData:
    case PM4ItOpcode::EventWrite:
    case PM4ItOpcode::AcquireMem:
    case PM4ItOpcode::IncrementDeCounter:
    case PM4ItOpcode::PfpSyncMe:
    case PM4ItOpcode::IndirectBuffer:
        return true;
    default:
        return false;
    }
}

static std::span<const u32> NextPacket(std::span<const u32> span, size_t offset) {
    if (offset > span.size()) {
        LOG_ERROR(
            Lib_GnmDriver,
            ": packet length exceeds remaining submission size. Packet dword count={}, remaining "
            "submission dwords={}",
            offset, span.size());
        // Return empty subspan so check for next packet bails out
        return {};
    }

    return span.subspan(offset);
}

Liverpool::Liverpool() {
    num_counter_pairs = Libraries::Kernel::sceKernelIsNeoMode() ? 16 : 8;
    if (CpSplit::Enabled()) {
        split_enabled = true;
        split_verify = CpSplit::Verify();
        split_run_ahead = CpSplit::RunAhead();
        split_post_dma = split_run_ahead && CpSplit::PostDma();
        if (split_post_dma && SplitHazard::Enabled()) {
            // SplitHazard's page map is written by NoteWrite (DMA CPU paths, back) and read by
            // NoteDraw (front): only race-free while DMAs are synchronous.
            LOG_WARNING(Render, "CpSplit: POST_DMA off, SHADPS4_SPLIT_HAZARD_STATS is on");
            split_post_dma = false;
        }
        split_post_asc = split_run_ahead ? CpSplit::PostAscMask() : 0u;
        if (split_post_asc != 0 && SplitHazard::Enabled()) {
            // Same race as POST_DMA: posted ASC DMAs note writes on the back.
            LOG_WARNING(Render, "CpSplit: POST_ASC off, SHADPS4_SPLIT_HAZARD_STATS is on");
            split_post_asc = 0;
        }
        if ((split_post_asc & CpSplit::PostAscWriteData) && g_cp_wait.enabled) {
            // ExecuteGfxPacket's WRITE_DATA notes the address in g_cp_wait on the back, which the
            // front reads: only race-free while WRITE_DATA is synchronous.
            LOG_WARNING(Render, "CpSplit: POST_ASC WRITE_DATA off, SHADPS4_CP_WAIT_STATS is on");
            split_post_asc &= ~u32{CpSplit::PostAscWriteData};
        }
        split_post_task_done = split_run_ahead && CpSplit::PostTaskDone();
        split_task_guard = split_post_task_done ? CpSplit::TaskGuard() : 0u;
        if (CpSplit::TaskGuard() != 0 && !split_post_task_done) {
            LOG_WARNING(Render, "CpSplit: TASK_GUARD ignored without run-ahead POST_TASK_DONE");
        }
        split_post_fence = split_task_guard != 0 && CpSplit::PostFence();
        if (CpSplit::PostFence() && split_task_guard == 0) {
            LOG_WARNING(Render, "CpSplit: POST_FENCE ignored without TASK_GUARD");
        }
        if (split_post_fence && (SplitHazard::Enabled() || g_cp_wait.enabled)) {
            // The fence bodies note the label in SplitHazard / g_cp_wait on the back, which the
            // front reads: only race-free while fences are synchronous.
            LOG_WARNING(Render, "CpSplit: POST_FENCE off, SPLIT_HAZARD_STATS or CP_WAIT_STATS on");
            split_post_fence = false;
        }
        split_disp_guard = CpSplit::DispGuard();
        split_front_derive = CpSplit::FrontDerive();
        if (split_front_derive != 0) {
            front_vi = std::make_unique<CpSplit::VertexInputsDerived>();
        }
        split_verify_every = split_run_ahead ? CpSplit::VerifyEvery() : 1u;
        split = std::make_unique<CpSplit::Channel>();
        mirror = std::make_unique<CpSplit::BackMirror>();
        std::memcpy(&mirror->regs, &regs, sizeof(Regs));
        mirror->cb_extent = last_cb_extent;
        mirror->db_extent = last_db_extent;
        mirror->ctx_epoch = context_change_epoch;
        back_regs = &mirror->regs;
        back_cb_extent = mirror->cb_extent.data();
        back_db_extent = &mirror->db_extent;
        back_ctx_epoch = &mirror->ctx_epoch;
        if (split_run_ahead) {
            LOG_INFO(Render, "CpSplit: on (run-ahead, depth {}{}{}{}){}", CpSplit::Depth(),
                     split_post_dma ? ", POST_DMA" : "",
                     split_post_asc ? fmt::format(", POST_ASC mask {:#x}", split_post_asc)
                                    : std::string{},
                     split_post_task_done
                         ? fmt::format(", POST_TASK_DONE{}",
                                       split_task_guard
                                           ? fmt::format(" TASK_GUARD {}{}", split_task_guard,
                                                         split_post_fence ? ", POST_FENCE" : "")
                                           : std::string{})
                         : std::string{},
                     split_verify ? fmt::format(", VERIFY every {}", split_verify_every)
                                  : std::string{});
            if (!split_disp_guard) {
                LOG_INFO(Render, "CpSplit: DISP_GUARD 0 (no disp_after_pkt drain)");
            }
        } else {
            LOG_INFO(Render, "CpSplit: on (depth 0){}", split_verify ? ", VERIFY" : "");
        }
        if (split_front_derive != 0) {
            const char* detail = split_front_derive == 2 ? ", verify"
                                 : CpSplit::FrontDeriveVertexInputSkip()
                                     ? ", setVertexInputEXT skip"
                                     : "";
            LOG_INFO(Render, "CpSplit: FRONT_DERIVE {} (vertex inputs{})", split_front_derive,
                     detail);
        }
    }
    process_thread = std::jthread{std::bind_front(&Liverpool::Process, this)};
}

Liverpool::~Liverpool() {
    process_thread.request_stop();
    process_thread.join();
}

void Liverpool::ProcessCommands() {
    // CP split: commands need the caches and the scheduler, so only the back (gpu_id) runs them,
    // between records and while idle. The front's per-packet calls return here.
    if (split_enabled && !IsGpuThread()) [[unlikely]] {
        return;
    }
    // Process incoming commands with high priority
    while (num_commands) {
        Common::UniqueFunction<void> callback{};
        {
            std::scoped_lock lk{submit_mutex};
            callback = std::move(command_queue.front());
            command_queue.pop();
            --num_commands;
        }
        if (rasterizer) {
            rasterizer->GetScheduler().SwitchStream(0);
        }
        callback();
    }
}

void Liverpool::Process(std::stop_token stoken) {
    Common::SetCurrentThreadName("shadPS4:GpuCommandProcessor");
    gpu_id = std::this_thread::get_id();
    // Bill page faults taken on this thread separately: this thread's activity IS the frame.
    Common::FaultMonitor::MarkCpThread();
    Common::ApplyCommandProcessorPlacement();

    if (split_enabled) {
        // CP split: this thread keeps every CP identity (name, gpu_id, t_is_cp_thread, placement,
        // scheduler producer, guest command queue) and becomes the back. The PM4 parser moves to
        // a new front thread, started only now so gpu_id is set before it can send a command.
        front_thread = std::jthread{std::bind_front(&Liverpool::FrontProcess, this)};
        BackLoop(stoken);
        return;
    }
    ProcessLoop(stoken);
}

void Liverpool::FrontProcess(std::stop_token stoken) {
    Common::SetCurrentThreadName("shadPS4:GpuCommandFront");
    CpSplit::t_is_front = true;
    Common::ApplyHotThreadPlacement(Common::HotThread::Front);
    // SPLIT-TODO: placement. ApplyCommandProcessorPlacement pins the back; the front wants its
    // own P-core (with the recording worker and Game:Main, 4 hot threads on 6 P-cores).
    ProcessLoop(stoken);
    split->FrontExited();
}

void Liverpool::BackLoop(std::stop_token stoken) {
    std::stop_callback wake_on_stop{stoken, [this] { split->WakeBack(); }};
    bool stop_forwarded = false;
    for (;;) {
        if (const auto* record = split->Next()) {
            Common::SampleThreadPlacement("back", 8);
            ExecuteRecord(*record);
            split->Done(record);
            // Guest commands run at record boundaries, as they ran at packet boundaries before.
            if (num_commands.load(std::memory_order_relaxed) != 0 && AscRingLatency())
                [[unlikely]] {
                const u64 t0 = __rdtsc();
                ProcessCommands();
                g_asc_ring.cmd_ticks += __rdtsc() - t0;
                ++g_asc_ring.cmd_n;
                continue;
            }
            ProcessCommands();
            continue;
        }
        if (num_commands.load(std::memory_order_relaxed) != 0 && AscRingLatency()) [[unlikely]] {
            const u64 t0 = __rdtsc();
            ProcessCommands();
            g_asc_ring.cmd_ticks += __rdtsc() - t0;
            ++g_asc_ring.cmd_n;
        }
        ProcessCommands();
        if (split->FrontHasExited()) {
            break;
        }
        if (stoken.stop_requested() && !stop_forwarded) {
            // The front owns the submission loop and leaves it on its own stop token. Keep
            // executing its records until it has exited (it drains first).
            // SPLIT-TODO: a front stuck in a guest wait loop hangs shutdown, as the CP does today.
            stop_forwarded = true;
            front_thread.request_stop();
        }
        {
            // SHADPS4_DRAW_COST=1 (diagnostic): back idle time, reported by NoteDrawCost.
            static const bool draw_cost = [] {
                const char* value = std::getenv("SHADPS4_DRAW_COST");
                return value != nullptr && value[0] == '1';
            }();
            const u64 idle_t0 = draw_cost ? __rdtsc() : 0;
            const bool ring_latency = AscRingLatency();
            const u32 ring_front_state = ring_latency ? AscRingIdleEnter() : 0;
            split->WaitForWork(num_commands, stoken);
            if (ring_latency) [[unlikely]] {
                AscRingIdleExit(ring_front_state);
            }
            if (draw_cost) {
                g_back_idle_cyc += __rdtsc() - idle_t0;
            }
        }
    }
    front_thread.join();
}

void Liverpool::ExecuteRecord(const CpSplit::RecordHeader& record) {
    const auto* payload = reinterpret_cast<const u8*>(&record + 1);
    switch (record.type) {
    case CpSplit::RecordType::RegWrite: {
        const u8* end = reinterpret_cast<const u8*>(&record) + record.size;
        const u8* p = payload;
        while (p + sizeof(CpSplit::RegWritePayload) <= end) {
            const auto* write = reinterpret_cast<const CpSplit::RegWritePayload*>(p);
            if (write->count == 0) {
                break;
            }
            std::memcpy(&mirror->regs.reg_array[write->word_offset], write + 1,
                        size_t{write->count} * sizeof(u32));
            p += sizeof(CpSplit::RegWritePayload) + size_t{write->count} * sizeof(u32);
        }
        break;
    }
    case CpSplit::RecordType::ClearState:
        mirror->regs.SetDefaults();
        break;
    case CpSplit::RecordType::Extent: {
        const auto* extent = reinterpret_cast<const CpSplit::ExtentPayload*>(payload);
        if (extent->slot < NUM_COLOR_BUFFERS) {
            mirror->cb_extent[extent->slot].raw = extent->raw;
        } else {
            mirror->db_extent.raw = extent->raw;
        }
        break;
    }
    case CpSplit::RecordType::EpochBump: {
        const auto* epoch = reinterpret_cast<const CpSplit::EpochPayload*>(payload);
        VideoCore::BumpSubmitEpoch(static_cast<PM4ItOpcode>(epoch->opcode));
        break;
    }
    case CpSplit::RecordType::Draw:
        rasterizer->GetScheduler().SwitchStream(0);
        ExecuteDrawRecord(record);
        break;
    case CpSplit::RecordType::Dispatch:
        ExecuteDispatchRecord(record);
        break;
    case CpSplit::RecordType::Packet:
        // SHADPS4_ASC_STREAM: packets (DMA, WRITE_DATA, fences, flips) record on the main stream.
        if (rasterizer) {
            rasterizer->GetScheduler().SwitchStream(0);
        }
        if (record.flags & CpSplit::AscReleaseMem) [[unlikely]] {
            ExecuteAscReleaseMem(reinterpret_cast<const PM4CmdReleaseMem*>(payload),
                                 record.flags >> 8);
            break;
        }
        ExecuteGfxPacket(reinterpret_cast<const PM4Header*>(payload),
                         (record.flags & CpSplit::GuestMarkers) != 0);
        break;
    case CpSplit::RecordType::Call: {
        const auto* call = reinterpret_cast<const CpSplit::CallPayload*>(payload);
        if (rasterizer) {
            rasterizer->GetScheduler().SwitchStream(0);
        }
        back_in_call = true;
        call->invoke(call->context);
        back_in_call = false;
        break;
    }
    case CpSplit::RecordType::Retire: {
        {
            std::scoped_lock lock{submit_mutex};
            pending_retire.fetch_sub(1, std::memory_order_seq_cst);
        }
        submit_cv.notify_all();
        break;
    }
    case CpSplit::RecordType::Diag:
        if (AscRingBack(*reinterpret_cast<const CpSplit::DiagPayload*>(payload)) && rasterizer) {
            rasterizer->GetScheduler().LogRecordingWindow();
        }
        break;
    case CpSplit::RecordType::Wrap:
    default:
        UNREACHABLE_MSG("CpSplit: bad record type {}", static_cast<u32>(record.type));
    }
}

void Liverpool::ExecuteDrawRecord(const CpSplit::RecordHeader& record) {
    const auto& rec = *reinterpret_cast<const SplitDrawRecord*>(&record + 1);
    mirror->ctx_epoch = rec.ctx_epoch;
    std::array<Shader::BoundStage, Shader::MaxStageTypes> stages{};
    const u8* stages_end =
        ReadStages(reinterpret_cast<const u8*>(&rec + 1), rec.stage_count, stages);
    // FRONT_DERIVE block (read by BufferCache::BindVertexBuffers), between stages and snapshot.
    const CpSplit::VertexInputsPayload* vi_block = nullptr;
    if (rec.vi_bytes != 0) {
        vi_block = reinterpret_cast<const CpSplit::VertexInputsPayload*>(stages_end);
        stages_end += rec.vi_bytes;
    }
    if (rec.verify & DrawVerifyFull) {
        VerifyMirror("draw", record.seq, nullptr);
        VerifyStages(stages, record.seq);
        if (split_run_ahead) {
            // Posted work before this record has now executed: does the lookup still agree?
            // (Depth 0 posts nothing, so there it could only see guest CPU writes.)
            ++split_draw_relookup_checks;
            const DrawRelookupDiff diff = DrawRelookup(*rasterizer, regs, rec.draw, stages);
            if (diff.what && ++split_draw_relookup_diffs <= 16) {
                LOG_WARNING(Render,
                            "CpSplit VERIFY: draw record {}: re-lookup on the back differs from "
                            "the front's: {} (stage {}, first dword {}, carried {} dw, re-lookup "
                            "{} dw)",
                            record.seq, diff.what, diff.stage, diff.index, diff.carried_dw,
                            diff.relooked_dw);
            }
            if ((split_draw_relookup_checks & 0xfff) == 0) {
                LOG_INFO(Render,
                         "CpSplit VERIFY: {} synced draws re-looked-up on the back, {} differed",
                         split_draw_relookup_checks, split_draw_relookup_diffs);
            }
        }
    } else if (rec.verify & DrawVerifySnapshot) {
        VerifySnapshot(stages_end, record.seq);
    }
    if (rec.marker) {
        rasterizer->ScopeMarkerBegin(fmt::format("gfx:{}:{}", rec.cmd_address, rec.marker));
        rasterizer->ExecuteDraw(rec.draw, stages, vi_block);
        rasterizer->ScopeMarkerEnd();
    } else {
        rasterizer->ExecuteDraw(rec.draw, stages, vi_block);
    }
}

void Liverpool::ExecuteDispatchRecord(const CpSplit::RecordHeader& record) {
    const auto& rec = *reinterpret_cast<const SplitDispatchRecord*>(&record + 1);
    std::array<Shader::BoundStage, Shader::MaxStageTypes> stages{};
    const u8* stages_end =
        ReadStages(reinterpret_cast<const u8*>(&rec + 1), rec.stage_count, stages);
    if (rec.verify & DrawVerifyFull) {
        // The front is parked on this record: its cs_state for this queue, registers and the
        // Info its lookup just refreshed must equal the record (this is what the Call path read).
        VerifyMirror("dispatch", record.seq, &rec.cs, rec.queue);
        VerifyStages(stages, record.seq);
        if (split_post_asc != 0) {
            // Posted work before this record has now executed: does the lookup still agree?
            ++split_relookup_checks;
            const auto& carried = stages[static_cast<u32>(Shader::LogicalStage::Compute)];
            if (!DispatchRelookupMatches(*rasterizer, rec.pipeline, carried) &&
                ++split_relookup_diffs <= 16) {
                LOG_WARNING(Render,
                            "CpSplit VERIFY: dispatch record {} (queue {}): re-lookup on the back "
                            "differs from the front's (pipeline or user data / flat buffer)",
                            record.seq, rec.queue);
            }
            if ((split_relookup_checks & 0xfff) == 0) {
                LOG_INFO(Render,
                         "CpSplit VERIFY: {} synced dispatches re-looked-up on the back, {} "
                         "differed",
                         split_relookup_checks, split_relookup_diffs);
            }
        }
    } else if (rec.verify & DrawVerifySnapshot) {
        VerifySnapshot(stages_end, record.seq);
    }
    if (rec.queue != GfxQueueId && AscRingLatency()) [[unlikely]] {
        ++g_asc_ring.q[rec.queue % AscRingQueues].dispatches;
    }
    // BackCsRegs() returns the snapshot while this dispatch runs (HLE, IsCompute*, BindBuffers).
    back_cs = &rec.cs;
    rasterizer->SetDispatchRun(rec.run);
    rasterizer->SetDispatchQueue(rec.queue);
    Common::ReadbackOffload::AscSpan().in_asc_dispatch = rec.queue != GfxQueueId;
    if (rec.marker && rec.queue != GfxQueueId) {
        rasterizer->ScopeMarkerBegin(
            fmt::format("asc[{}]:{}:{}", rec.queue - 1, rec.cmd_address, rec.marker));
        rasterizer->ExecuteDispatch(rec.pipeline, stages, rec.is_indirect != 0, rec.address,
                                    rec.offset, rec.size);
        rasterizer->ScopeMarkerEnd();
    } else if (rec.marker) {
        rasterizer->ScopeMarkerBegin(fmt::format("gfx:{}:{}", rec.cmd_address, rec.marker));
        rasterizer->ExecuteDispatch(rec.pipeline, stages, rec.is_indirect != 0, rec.address,
                                    rec.offset, rec.size);
        rasterizer->ScopeMarkerEnd();
    } else {
        rasterizer->ExecuteDispatch(rec.pipeline, stages, rec.is_indirect != 0, rec.address,
                                    rec.offset, rec.size);
    }
    rasterizer->SetDispatchRun(0);
    rasterizer->SetDispatchQueue(0);
    Common::ReadbackOffload::AscSpan().in_asc_dispatch = false;
    back_cs = nullptr;
}

void Liverpool::VerifyMirror(const char* what, u64 seq, const ComputeProgram* cs,
                             u32 cs_queue) const {
    // Only for records the front is parked on (Sync): depth 0, dispatches, and the periodic
    // synchronous run-ahead draw. The front's state is then stable and published to this thread
    // (release store of write_pos, acquire load in Next).
    // A full memcmp rather than a hash: same pass over 213 KB, exact, and it names the word.
    const auto& back = mirror->regs.reg_array;
    const auto& front = regs.reg_array;
    if (std::memcmp(back.data(), front.data(), sizeof(back)) != 0) {
        u32 i = 0;
        while (back[i] == front[i]) {
            ++i;
        }
        UNREACHABLE_MSG("CpSplit VERIFY: register word {:#x} differs at {} record {}: back {:#x} "
                        "front {:#x}",
                        i, what, seq, back[i], front[i]);
    }
    for (u32 cb = 0; cb < NUM_COLOR_BUFFERS; ++cb) {
        ASSERT_MSG(mirror->cb_extent[cb].raw == last_cb_extent[cb].raw,
                   "CpSplit VERIFY: CB{} extent differs at {} record {}", cb, what, seq);
    }
    ASSERT_MSG(mirror->db_extent.raw == last_db_extent.raw,
               "CpSplit VERIFY: DB extent differs at {} record {}", what, seq);
    // The context epoch only feeds the draw path's viewport memo; dispatch records do not carry
    // it, so the mirror's copy is only current at draws.
    ASSERT_MSG(cs || mirror->ctx_epoch == context_change_epoch,
               "CpSplit VERIFY: context epoch differs at {} record {}", what, seq);
    if (cs) {
        // The Call path read mapped_queues[curr_qid] on the back; the record copied GetCsRegs()
        // on the front. Both must be this queue's state.
        ASSERT_MSG(cs_queue < NumTotalQueues && s32(cs_queue) == curr_qid &&
                       std::memcmp(cs, &mapped_queues[cs_queue].cs_state, sizeof(*cs)) == 0,
                   "CpSplit VERIFY: cs registers differ at {} record {} (queue {}, curr_qid {})",
                   what, seq, cs_queue, curr_qid);
    }
    static u64 checks = 0;
    if ((++checks & 0xffff) == 0) {
        LOG_INFO(Render, "CpSplit VERIFY: {} draw/dispatch records checked, 0 mismatches",
                 checks);
    }
}

void Liverpool::VerifySnapshot(const u8* snapshot, u64 seq) const {
    // Run-ahead: the front has moved on, so compare against the copy it made at this draw.
    // Registers outside the windows are covered by the periodic full check (VerifyMirror).
    const auto* words = reinterpret_cast<const u32*>(snapshot);
    for (u32 cb = 0; cb < NUM_COLOR_BUFFERS; ++cb) {
        ASSERT_MSG(mirror->cb_extent[cb].raw == words[cb],
                   "CpSplit VERIFY: CB{} extent differs at posted record {}", cb, seq);
    }
    ASSERT_MSG(mirror->db_extent.raw == words[NUM_COLOR_BUFFERS],
               "CpSplit VERIFY: DB extent differs at posted record {}", seq);
    words += NUM_COLOR_BUFFERS + 1;
    for (const auto& window : VerifyWindows) {
        const u32 count = window.end - window.begin;
        const u32* back = &mirror->regs.reg_array[window.begin];
        if (std::memcmp(back, words, size_t{count} * sizeof(u32)) != 0) {
            u32 i = 0;
            while (back[i] == words[i]) {
                ++i;
            }
            UNREACHABLE_MSG("CpSplit VERIFY: register word {:#x} differs at posted record "
                            "{}: back {:#x} front {:#x}",
                            window.begin + i, seq, back[i], words[i]);
        }
        words += count;
    }
    static u64 checks = 0;
    if ((++checks & 0xffff) == 0) {
        LOG_INFO(Render, "CpSplit VERIFY: {} posted draws/dispatches checked (register windows), 0 mismatches",
                 checks);
    }
}

void Liverpool::JournalRegs(u32 word_offset, u32 count) {
    if (split_enabled) [[unlikely]] {
        split->EmitRegWrite(word_offset, &regs.reg_array[word_offset], count);
    }
}

void Liverpool::JournalExtent(u32 reg_addr) {
    if (!split_enabled) [[likely]] {
        return;
    }
    switch (reg_addr) {
    case ContextRegs::CbColor0Base:
    case ContextRegs::CbColor1Base:
    case ContextRegs::CbColor2Base:
    case ContextRegs::CbColor3Base:
    case ContextRegs::CbColor4Base:
    case ContextRegs::CbColor5Base:
    case ContextRegs::CbColor6Base:
    case ContextRegs::CbColor7Base: {
        const u32 cb = (reg_addr - ContextRegs::CbColor0Base) /
                       (ContextRegs::CbColor1Base - ContextRegs::CbColor0Base);
        split->EmitExtent(cb, last_cb_extent[cb].raw);
        break;
    }
    case ContextRegs::CbColor0Cmask:
    case ContextRegs::CbColor1Cmask:
    case ContextRegs::CbColor2Cmask:
    case ContextRegs::CbColor3Cmask:
    case ContextRegs::CbColor4Cmask:
    case ContextRegs::CbColor5Cmask:
    case ContextRegs::CbColor6Cmask:
    case ContextRegs::CbColor7Cmask: {
        const u32 cb = (reg_addr - ContextRegs::CbColor0Cmask) /
                       (ContextRegs::CbColor1Cmask - ContextRegs::CbColor0Cmask);
        split->EmitExtent(cb, last_cb_extent[cb].raw);
        break;
    }
    case ContextRegs::DbZInfo:
        split->EmitExtent(NUM_COLOR_BUFFERS, last_db_extent.raw);
        break;
    default:
        break;
    }
}

void Liverpool::BumpEpoch(PM4ItOpcode opcode) {
    if (split_enabled) [[unlikely]] {
        // In stream order on the back. Delaying a bump only makes an upload happen earlier
        // (submit_epoch.h: an extra bump is safe, a missing one is not).
        if (VideoCore::g_epoch_defer_upload) {
            split->EmitEpochBump(static_cast<u32>(opcode));
        }
        return;
    }
    VideoCore::BumpSubmitEpoch(opcode);
}

void Liverpool::RunGfxPacket(const PM4Header* header, bool guest_markers_enabled,
                             CpSplit::DrainReason reason) {
    if (!split_enabled) [[likely]] {
        ExecuteGfxPacket(header, guest_markers_enabled);
        return;
    }
    // Verbatim copy: the guest may reuse the DCB once the front has moved on (run-ahead).
    // The copy reads guest memory between Reserve and Commit; a fault there is safe because a
    // fault drain never publishes the open record (Channel).
    const u32 dwords = header->type3.NumWords() + 1;
    void* copy = split->Reserve(CpSplit::RecordType::Packet, dwords * sizeof(u32),
                                guest_markers_enabled ? CpSplit::GuestMarkers : 0);
    std::memcpy(copy, header, dwords * sizeof(u32));
    split->Commit();
    split->Sync(reason);
}

void Liverpool::RunGfxDma(const PM4Header* header, bool guest_markers_enabled,
                          std::span<const u32> rest) {
    if (!split_post_dma) [[likely]] {
        RunGfxPacket(header, guest_markers_enabled, CpSplit::DrainReason::PacketDma);
        return;
    }
    // POST_DMA (run-ahead): nothing the front reads after this packet is produced by it, except
    //  - guest memory read by later pipeline lookups (shader code, SRT tables, fetch V# tables):
    //    measured 0 of ~4 M top-level checks against DMA CPU-path writes (SPLIT_HAZARD_STATS,
    //    2026-09-25); nested tables unmeasured (SPLIT-TODO);
    //  - the rest of this command buffer (a GPU-patched packet, as ASC does for dispatch dims):
    //    guarded here, synchronous if the destination overlaps it;
    //  - nested IBs: IndirectBuffer syncs while posted packets are outstanding;
    //  - WAIT_REG_MEM / COND_EXEC / semaphores / faults: those drain anyway.
    // SPLIT-TODO: only the innermost DCB level is guarded; a parent level's remainder is not.
    const auto* dma = reinterpret_cast<const PM4DmaData*>(header);
    VAddr write_dst = 0;
    u64 write_bytes = 0; ///< 0: a GDS destination, which no command buffer can be in.
    if (dma->dst_sel == DmaDataDst::Memory || dma->dst_sel == DmaDataDst::MemoryUsingL2) {
        const VAddr dst = dma->DstAddress<VAddr>();
        const VAddr dst_end = dst + dma->NumBytes();
        const auto lo = reinterpret_cast<VAddr>(rest.data());
        const VAddr hi = lo + rest.size_bytes();
        if (dst < hi && lo < dst_end) {
            RunGfxPacket(header, guest_markers_enabled, CpSplit::DrainReason::PacketDmaGuard);
            return;
        }
        write_dst = dst;
        write_bytes = dst_end - dst;
    }
    const u32 dwords = header->type3.NumWords() + 1;
    void* copy = split->Reserve(CpSplit::RecordType::Packet, dwords * sizeof(u32),
                                guest_markers_enabled ? CpSplit::GuestMarkers : 0);
    std::memcpy(copy, header, dwords * sizeof(u32));
    split->Commit();
    if (split_task_guard) [[unlikely]] {
        split->NotePostedWrite(write_dst, write_bytes);
    }
    split->PostRunAhead(CpSplit::PostKind::GfxPacket);
}

void Liverpool::RunAscPacket(const PM4Header* header, std::span<const u32> rest, VAddr dst,
                             u64 bytes, CpSplit::DrainReason guard_reason) {
    // POST_ASC: the same argument as RunGfxDma. What the front reads after this packet:
    //  - the rest of this ASC command buffer (the title patches dispatch dims this way; that
    //    case is the front's indirect_patches and never gets here): guarded, synchronous if
    //    [dst, dst + bytes) overlaps it;
    //  - nested ASC IBs: IndirectBuffer syncs while posted packets are outstanding;
    //  - guest memory read by later pipeline lookups: unmeasured for ASC (SPLIT-TODO); the
    //    VERIFY re-lookup on synced dispatches is the check;
    //  - WAIT_REG_MEM / semaphores / RELEASE_MEM / task done / faults: those drain anyway.
    // The packet is copied: the front advances the ring's read_addr right after parsing it, so
    // the guest may reuse the slot before the back runs it.
    const u32 dwords = header->type3.NumWords() + 1;
    void* copy = split->Reserve(CpSplit::RecordType::Packet, dwords * sizeof(u32), 0);
    std::memcpy(copy, header, dwords * sizeof(u32));
    split->Commit();
    const auto lo = reinterpret_cast<VAddr>(rest.data());
    const VAddr hi = lo + rest.size_bytes();
    if (bytes != 0 && dst < hi && lo < dst + bytes) {
        split->Sync(guard_reason);
        return;
    }
    if (split_task_guard) [[unlikely]] {
        split->NotePostedWrite(dst, bytes); // bytes == 0 (GDS destination): nothing to note.
    }
    split->PostRunAhead(CpSplit::PostKind::AscPacket);
    split->MaybeLog();
}

void Liverpool::RunFence(const PM4Header* header, std::span<const u32> rest, VAddr label,
                         u16 flags, CpSplit::DrainReason sync_reason) {
    // POST_FENCE: the label is written (and the interrupt raised) when the back reaches this
    // record, exactly when the synchronous path wrote it; the front just does not wait.
    // Front readers of the label drain first (WAIT_REG_MEM, COND_EXEC, semaphores); a label in
    // another buffer is caught by the TASK_GUARD checks; one in this buffer's rest syncs here.
    // The copy: the guest may reuse the command memory once the front has moved on.
    // Fences do not arm SyncIfPostedPackets, so the IB-end guards do not cover a label in a
    // parent level's rest: check every active level here (`rest` is the innermost one).
    constexpr u64 LabelBytes = 8; // 32- or 64-bit value; 8 covers both.
    const u32 dwords = header->type3.NumWords() + 1;
    void* copy = split->Reserve(CpSplit::RecordType::Packet, dwords * sizeof(u32), flags);
    std::memcpy(copy, header, dwords * sizeof(u32));
    split->Commit();
    const auto lo = reinterpret_cast<VAddr>(rest.data());
    const VAddr hi = lo + rest.size_bytes();
    if ((label < hi && lo < label + LabelBytes) ||
        SplitSpansOverlap(label, label + LabelBytes)) {
        split->Sync(sync_reason);
        return;
    }
    split->NotePostedWrite(label, LabelBytes);
    split->PostRunAhead(CpSplit::PostKind::Fence);
    split->MaybeLog();
}



static void NoteAscFence(const PM4CmdReleaseMem& packet) {
    if (!Common::FenceWatch::Enabled()) {
        return;
    }
    switch (packet.data_sel.Value()) {
    case DataSelect::Data32Low:
        Common::FenceWatch::NoteRelease(packet.Address<VAddr>(), packet.DataDWord(), 4);
        break;
    case DataSelect::Data64:
        Common::FenceWatch::NoteRelease(packet.Address<VAddr>(), packet.DataQWord(), 8);
        break;
    default:
        break;
    }
}

/// SHADPS4_ASC_RELEASE_FLUSH=1: cut and submit the batch being recorded when an ASC RELEASE_MEM
/// runs on the back. The label is written at CP time, so the guest's poll (Second Son's 0x1695e0
/// particle wait) passes at once; without a cut the producing dispatches stay in the unsubmitted
/// batch until the guest's first readback fault cuts it (46% of control-block jobs, run
/// diag-fire-fastlane3-20260928-i, Fable review 2026-09-28).
static bool AscReleaseFlush() {
    static const bool enabled = [] {
        const char* value = std::getenv("SHADPS4_ASC_RELEASE_FLUSH");
        return value && value[0] == '1';
    }();
    return enabled;
}

void Liverpool::ExecuteAscReleaseMem(const PM4CmdReleaseMem* release_mem, u32 pipe_id) {
    if (rasterizer) {
        rasterizer->ProcessDownloadImages();
    }
    if (AscReleaseFlush() && rasterizer) {
        auto& sched = rasterizer->GetScheduler();
        sched.EndRendering();
        sched.Flush();
    }
    if (rasterizer && rasterizer->GetScheduler().HasAscStream()) {
        // SHADPS4_ASC_STREAM: the guest's compute label. Hand the ASC work recorded so far to its
        // queue now (its own queue: this does not deepen the main one).
        rasterizer->GetScheduler().SubmitAscSegment();
    }
    if (Common::ReadbackOffload::AscSpanStatsEnabled() && rasterizer) [[unlikely]] {
        auto& sp = Common::ReadbackOffload::AscSpan();
        auto& sched = rasterizer->GetScheduler();
        auto* ms = sched.GetMasterSemaphore();
        ms->Refresh();
        const u64 cur = sched.CurrentTick();
        const u64 known = ms->KnownGpuTick();
        ++sp.spans;
        if (sp.binds == 0) {
            ++sp.empty;
        } else {
            const u64 floor_a = std::max(sp.w_access, known);
            const u64 floor_w = std::max(sp.w_write, known);
            // Unfinished batches newer than the newest one the span depends on (the open batch
            // included: an ASC submission could go ahead of it).
            const u64 over_a = cur > floor_a ? cur - floor_a : 0;
            const u64 over_w = cur > floor_w ? cur - floor_w : 0;
            sp.unfinished_sum += cur > known ? cur - known : 0;
            sp.over_access_sum += over_a;
            sp.over_write_sum += over_w;
            sp.access_is_current += sp.w_access == cur;
            sp.write_is_current += sp.w_write == cur;
            sp.big_buffer_spans += sp.max_buffer_mb >= 16;
            ++sp.over_access_hist[over_a == 0 ? 0 : over_a == 1 ? 1 : over_a < 4 ? 2 : over_a < 8 ? 3 : 4];
        }
        sp.w_access = sp.w_write = sp.binds = sp.max_buffer_mb = 0;
        if ((sp.spans & 0x7ff) == 0) {
            const double n = double(std::max<u64>(1, sp.spans - sp.empty));
            LOG_INFO(Render,
                     "AscSpan: spans={} empty={} unfinished_batches={:.2f} overtakeable_by_access="
                     "{:.2f} overtakeable_by_write={:.2f} access_newest_is_current={} "
                     "write_newest_is_current={} spans_binding_>=16MB={} "
                     "hist_overtake_access[0,1,2-3,4-7,8+]={}/{}/{}/{}/{}",
                     sp.spans, sp.empty, double(sp.unfinished_sum) / n,
                     double(sp.over_access_sum) / n, double(sp.over_write_sum) / n,
                     sp.access_is_current, sp.write_is_current, sp.big_buffer_spans,
                     sp.over_access_hist[0], sp.over_access_hist[1], sp.over_access_hist[2],
                     sp.over_access_hist[3], sp.over_access_hist[4]);
            const auto keep = sp.in_asc_dispatch;
            sp = {};
            sp.in_asc_dispatch = keep;
        }
    }
    g_cp_wait.NoteWrite(release_mem->Address<void*>());
    VideoCore::ProtectFence();
    if (AscRingLatency()) [[unlikely]] {
        // Just before the label write (or its deferral to the GPU, counted separately).
        AscRingLabel(release_mem->Address<VAddr>(),
                     (FenceAtGpu() == 1 || FenceAtGpu() == 2) && rasterizer &&
                         release_mem->data_sel != DataSelect::GdsMemStore);
    }
    if ((FenceAtGpu() == 1 || FenceAtGpu() == 2) && rasterizer &&
        release_mem->data_sel != DataSelect::GdsMemStore) {
        rasterizer->DeferToGpu(
            [packet = *release_mem, pipe_id] {
                packet.SignalFence(
                    [pipe_id] {
                        Platform::IrqC::Instance()->Signal(
                            static_cast<Platform::InterruptId>(pipe_id));
                    },
                    [](VAddr, u16, u16) {});
                NoteAscFence(packet);
            },
            FenceAtGpu() == 1);
        return;
    }
    release_mem->SignalFence(
        [pipe_id] {
            Platform::IrqC::Instance()->Signal(static_cast<Platform::InterruptId>(pipe_id));
        },
        [this](VAddr dst, u16 gds_index, u16 num_dwords) {
            rasterizer->CopyBuffer(dst, gds_index, num_dwords * sizeof(u32), false, true);
        });
    NoteAscFence(*release_mem);
}

Liverpool::SplitSpanScope::SplitSpanScope(Liverpool& liverpool, const std::span<const u32>* span,
                                          CpSplit::DrainReason entry_reason) {
    if (liverpool.split_task_guard == 0) [[likely]] {
        return;
    }
    lp = &liverpool;
    for (u32 i = 0; i < SplitSpanSlots; ++i) {
        if (!lp->split_spans[i]) {
            lp->split_spans[i] = span;
            slot = static_cast<s32>(i);
            lp->split_span_hwm = std::max(lp->split_span_hwm, i + 1);
            break;
        }
    }
    if (slot < 0) {
        ++lp->split_span_untracked;
    }
    // Nothing of this buffer has been parsed yet.
    lp->SplitGuardSpan(*span, entry_reason);
}

Liverpool::SplitSpanScope::~SplitSpanScope() {
    if (!lp) {
        return;
    }
    if (slot < 0) {
        --lp->split_span_untracked;
        return;
    }
    lp->split_spans[slot] = nullptr;
    while (lp->split_span_hwm > 0 && !lp->split_spans[lp->split_span_hwm - 1]) {
        --lp->split_span_hwm;
    }
}

void Liverpool::SplitGuardSpan(std::span<const u32> span, CpSplit::DrainReason reason) {
    if (split->BeginGuardCheck(reason)) {
        const auto lo = reinterpret_cast<VAddr>(span.data());
        split->GuardSpan(lo, lo + span.size_bytes(), reason);
    }
}

void Liverpool::SplitGuardActive() {
    if (!split->BeginGuardCheck(CpSplit::DrainReason::TaskGuardResume)) {
        return;
    }
    if (split_span_untracked != 0) {
        split->Sync(CpSplit::DrainReason::TaskGuardOverflow);
        return;
    }
    for (u32 i = 0; i < split_span_hwm; ++i) {
        if (const auto* span = split_spans[i]) {
            const auto lo = reinterpret_cast<VAddr>(span->data());
            if (split->GuardSpan(lo, lo + span->size_bytes(),
                                 CpSplit::DrainReason::TaskGuardResume)) {
                return;
            }
        }
    }
}

bool Liverpool::SplitSpansOverlap(VAddr lo, VAddr hi) const {
    if (split_span_untracked != 0) {
        return true;
    }
    for (u32 i = 0; i < split_span_hwm; ++i) {
        if (const auto* span = split_spans[i]) {
            const auto s_lo = reinterpret_cast<VAddr>(span->data());
            if (lo < s_lo + span->size_bytes() && s_lo < hi) {
                return true;
            }
        }
    }
    return false;
}

void Liverpool::EmitDraw(const PM4Header* header, const char* name, bool host_marker,
                         const CpSplit::DrawArgs& args) {
    Vulkan::Rasterizer::SplitDraw draw{.args = args};
    // Keep this lookup before the Sync below, also for the periodic VERIFY draw: the back's
    // re-lookup of that draw (DrawRelookup) is only meaningful because this one ran while
    // earlier posted records could still be outstanding. A drain before this line would blind it.
    rasterizer->PrepareDrawFront(draw);
    const bool has_stages = draw.pipeline && draw.draw_class ==
                                                 Vulkan::Rasterizer::DrawClass::Draw &&
                            !draw.skip;
    // Run-ahead: posted, except the periodic VERIFY draw, which syncs for the full check.
    // Depth 0: every draw syncs.
    bool sync = !split_run_ahead;
    u32 verify = 0;
    if (split_verify) {
        if (!split_run_ahead || (++split_verify_draws % split_verify_every) == 0) {
            sync = true;
            verify = DrawVerifyFull;
        } else {
            verify = DrawVerifySnapshot;
        }
    }
    u32 bytes = sizeof(SplitDrawRecord);
    if (has_stages) {
        for (const Shader::Info* info : draw.pipeline->GetStages()) {
            if (info) {
                bytes += StageBytes(*info);
            }
        }
    }
    // FRONT_DERIVE: derived here, before the Sync below and from the same Info the stage copy
    // reads, so the V# table reads happen at parse time (whether that differs from the back's
    // read at execute time is what mode 2 measures). Not for indirect draws: DrawIndirectTail
    // keeps the back-side derivation.
    u32 vi_bytes = 0;
    if (split_front_derive != 0 && has_stages && !args.is_indirect) {
        const auto stages = draw.pipeline->GetStages();
        const Shader::BoundStage vs{stages[static_cast<u32>(Shader::LogicalStage::Vertex)]};
        vi_bytes = draw.pipeline->DeriveVertexInputs(*front_vi, regs.vgt_instance_step_rate_0,
                                                     regs.vgt_instance_step_rate_1, vs);
        bytes += vi_bytes;
        // Number the description part: the back skips the setVertexInputEXT of a repeated id.
        auto& vi = *front_vi;
        bool same = vi.header.count == front_vi_prev_count;
        for (u32 i = 0; same && i < vi.header.count; ++i) {
            same = std::memcmp(&vi.attribs[i].desc, &front_vi_prev[i],
                               sizeof(CpSplit::VertexInputDesc)) == 0;
        }
        if (!same) {
            ++front_vi_desc_id;
            front_vi_prev_count = vi.header.count;
            for (u32 i = 0; i < vi.header.count; ++i) {
                front_vi_prev[i] = vi.attribs[i].desc;
            }
        }
        vi.header.desc_id = front_vi_desc_id;
    }
    if (verify == DrawVerifySnapshot) {
        bytes += VerifySnapshotBytes;
    }
    auto* rec = split->Reserve<SplitDrawRecord>(CpSplit::RecordType::Draw, bytes);
    rec->draw = draw;
    rec->ctx_epoch = context_change_epoch;
    rec->marker = host_marker ? name : nullptr;
    rec->cmd_address = header;
    rec->stage_count = 0;
    rec->verify = verify;
    rec->vi_bytes = vi_bytes;
    rec->pad = 0;
    // Copied now: the front's next lookup rewrites these Info fields (inventory 4.4). With
    // run-ahead this is what makes the record self-contained: the back reads user data, flat
    // buffer and pgm_base only from here, and the Info only for its immutable part.
    auto* out = reinterpret_cast<u8*>(rec + 1);
    if (has_stages) {
        const auto stages = draw.pipeline->GetStages();
        for (u32 i = 0; i < stages.size(); ++i) {
            if (stages[i]) {
                out = WriteStage(out, *stages[i], i);
                ++rec->stage_count;
            }
        }
    }
    if (vi_bytes != 0) {
        out = front_vi->Write(out);
    }
    if (verify == DrawVerifySnapshot) {
        WriteVerifySnapshot(out, regs, last_cb_extent, last_db_extent);
    }
    split->Commit();
    if (sync) {
        split->Sync(split_run_ahead ? CpSplit::DrainReason::VerifyFull
                                    : CpSplit::DrainReason::DepthZero);
    } else {
        split->PostRunAhead(CpSplit::PostKind::Draw);
    }
    split->MaybeLog();
}

namespace {
/// SHADPS4_DISPATCH_SYNC_STATS=1 (diagnostic, default off, front thread only): for every dispatch,
/// which guest sync-type packets the same queue issued since its previous dispatch. Decides whether
/// the game orders its back-to-back compute dispatches itself (CS_PARTIAL_FLUSH / ACQUIRE_MEM /
/// RELEASE_MEM / WAIT_REG_MEM) or relies on GCN overlapping them (2026-09-27, fire particles).
struct DispatchSyncStats {
    enum Bit : u32 {
        EvCsFlush = 1u << 0,
        EvCacheFlush = 1u << 1,
        EvOther = 1u << 2,
        Acquire = 1u << 3,
        Release = 1u << 4,
        WaitReg = 1u << 5,
        Dma = 1u << 6,
        WriteData = 1u << 7,
        EopEos = 1u << 8,
        Draw = 1u << 9,
        Count = 10
    };
    bool enabled = [] {
        const char* v = std::getenv("SHADPS4_DISPATCH_SYNC_STATS");
        return v && v[0] == '1';
    }();
    /// SHADPS4_GUEST_DISPATCH_ORDER=1: also number guest dispatch runs per queue (a run ends at
    /// every sync-type packet or draw of that queue); the back skips buffer barriers inside a run.
    bool order = [] {
        const char* v = std::getenv("SHADPS4_GUEST_DISPATCH_ORDER");
        return v && v[0] == '1';
    }();
    std::array<u64, 65> run{};
    u64 run_counter{};
    u64 Run(u32 key) {
        if (run[key] == 0) {
            run[key] = ++run_counter;
        }
        return run[key];
    }
    std::array<u32, 65> since{};          // per queue key: bits seen since its last dispatch
    std::array<bool, 65> had_dispatch{};  // queue key has dispatched before
    std::array<std::array<u64, Count>, 2> bit_hits{}; // [gfx=0/asc=1][bit]
    std::array<u64, 2> dispatches{}, back_to_back{}, first{};
    std::array<u64, 64> ev_types{};
    u64 total{};

    /// Packets that do not end a guest dispatch run: register writes, padding/markers and IB jumps.
    /// Everything else (events, fences, waits, DMA, draws, semaphores, ...) does.
    static bool RunNeutral(PM4ItOpcode op) {
        switch (op) {
        case PM4ItOpcode::Nop:
        case PM4ItOpcode::SetShReg:
        case PM4ItOpcode::SetContextReg:
        case PM4ItOpcode::SetUconfigReg:
        case PM4ItOpcode::SetConfigReg:
        case PM4ItOpcode::IndirectBuffer:
        case PM4ItOpcode::IndirectBufferConst:
        case PM4ItOpcode::DispatchDirect:
        case PM4ItOpcode::DispatchIndirect:
            return true;
        default:
            return false;
        }
    }

    void Note(u32 key, PM4ItOpcode op, const PM4Header* header) {
        if (!enabled && !order) {
            return;
        }
        if (order && !RunNeutral(op)) {
            run[key] = ++run_counter; // a new run starts after this packet
        }
        if (!enabled) {
            return;
        }
        u32 bit = 0;
        switch (op) {
        case PM4ItOpcode::EventWrite: {
            const auto type =
                static_cast<u32>(reinterpret_cast<const PM4CmdEventWrite*>(header)->event_type.Value());
            ++ev_types[type & 63];
            bit = type == static_cast<u32>(EventType::CsPartialFlush) ? EvCsFlush
                  : (type == static_cast<u32>(EventType::CacheFlush) ||
                     type == static_cast<u32>(EventType::CacheFlushTs))
                      ? EvCacheFlush
                      : EvOther;
            break;
        }
        case PM4ItOpcode::AcquireMem:
            bit = Acquire;
            break;
        case PM4ItOpcode::ReleaseMem:
            bit = Release;
            break;
        case PM4ItOpcode::WaitRegMem:
            bit = WaitReg;
            break;
        case PM4ItOpcode::DmaData:
            bit = Dma;
            break;
        case PM4ItOpcode::WriteData:
            bit = WriteData;
            break;
        case PM4ItOpcode::EventWriteEop:
        case PM4ItOpcode::EventWriteEos:
            bit = EopEos;
            break;
        case PM4ItOpcode::DrawIndex2:
        case PM4ItOpcode::DrawIndexAuto:
        case PM4ItOpcode::DrawIndexOffset2:
        case PM4ItOpcode::DrawIndexIndirect:
        case PM4ItOpcode::DrawIndirect:
        case PM4ItOpcode::DrawIndexIndirectMulti:
            bit = Draw;
            break;
        case PM4ItOpcode::DispatchDirect:
        case PM4ItOpcode::DispatchIndirect: {
            const u32 q = key == 64 ? 0 : 1;
            ++dispatches[q];
            if (!had_dispatch[key]) {
                had_dispatch[key] = true;
                ++first[q];
            } else if (since[key] == 0) {
                ++back_to_back[q];
            } else {
                for (u32 b = 0; b < Count; ++b) {
                    bit_hits[q][b] += (since[key] >> b) & 1;
                }
            }
            since[key] = 0;
            if ((++total & 0x7fff) == 0) {
                Log();
            }
            return;
        }
        default:
            return;
        }
        since[key] |= bit;
    }

    void Log() const {
        std::string ev;
        for (u32 t = 0; t < 64; ++t) {
            if (ev_types[t]) {
                ev += fmt::format(" {}:{}", t, ev_types[t]);
            }
        }
        for (u32 q = 0; q < 2; ++q) {
            const auto& h = bit_hits[q];
            LOG_INFO(Render,
                     "DispatchSync: queue={} dispatches={} back_to_back={} first={} with_sync: "
                     "cs_flush={} cache_flush={} ev_other={} acquire={} release={} wait_reg={} "
                     "dma={} write_data={} eop_eos={} draw={}",
                     q ? "asc" : "gfx", dispatches[q], back_to_back[q], first[q], h[0], h[1], h[2],
                     h[3], h[4], h[5], h[6], h[7], h[8], h[9]);
        }
        LOG_INFO(Render, "DispatchSync: event types (type:count){}", ev);
    }
};
DispatchSyncStats g_dispatch_sync;
} // namespace

void Liverpool::EmitDispatch(const PM4Header* header, const char* name, bool host_marker,
                             bool is_indirect, VAddr address, u32 offset, u32 size, u32 queue) {
    // POST_ASC (run-ahead): posted like draws. The record owns everything the back reads besides
    // the register mirror: the cs registers (copied here; the back's BackCsRegs() returns the
    // copy), the stage's user data / flat buffer / pgm_base, the pipeline pointer (stable) and
    // the indirect address. The ASC ring slot may be reused once the front moves on.
    const u32 post_bit =
        queue == GfxQueueId ? u32{CpSplit::PostGfxDispatch} : u32{CpSplit::PostAscDispatch};
    bool post = (split_post_asc & post_bit) != 0;
    u32 verify = 0;
    if (split_verify) {
        if (!post || (++split_verify_dispatches % split_verify_every) == 0) {
            verify = DrawVerifyFull;
        } else {
            verify = DrawVerifySnapshot;
        }
    }
    const bool verify_sync = post && verify == DrawVerifyFull;
    post = post && !verify_sync;
    if (post && split_disp_guard) {
        // The front lookup below reads guest memory (user data tables, flat buffer) that a posted
        // DMA_DATA / WRITE_DATA may not have written yet. DISP_GUARD=0 drops this, as for draws.
        split->SyncIfPostedPackets(CpSplit::DrainReason::DispatchAfterPkt);
    }
    const Vulkan::ComputePipeline* pipeline = rasterizer->PrepareDispatchFront();
    const Shader::Info* info = pipeline ? pipeline->GetStages()[0] : nullptr;
    const u32 stage_bytes = info ? StageBytes(*info) : 0u;
    u32 bytes = sizeof(SplitDispatchRecord) + stage_bytes;
    if (verify == DrawVerifySnapshot) {
        bytes += VerifySnapshotBytes;
    }
    auto* rec = split->Reserve<SplitDispatchRecord>(CpSplit::RecordType::Dispatch, bytes);
    std::memcpy(&rec->cs, &GetCsRegs(), sizeof(ComputeProgram));
    rec->queue = queue;
    rec->verify = verify;
    rec->pipeline = pipeline;
    rec->marker = host_marker ? name : nullptr;
    rec->cmd_address = header;
    rec->address = address;
    rec->offset = offset;
    rec->size = size;
    rec->is_indirect = is_indirect ? 1u : 0u;
    rec->stage_count = 0;
    rec->run = g_dispatch_sync.order
                   ? g_dispatch_sync.Run(queue == GfxQueueId ? 64u : ((queue - 1) & 63u))
                   : 0;
    auto* out = reinterpret_cast<u8*>(rec + 1);
    if (info) {
        out = WriteStage(out, *info, u32(Shader::LogicalStage::Compute));
        rec->stage_count = 1;
    }
    if (verify == DrawVerifySnapshot) {
        WriteVerifySnapshot(out, regs, last_cb_extent, last_db_extent);
    }
    split->Commit();
    if (post) {
        split->PostRunAhead(queue == GfxQueueId ? CpSplit::PostKind::GfxDispatch
                                                : CpSplit::PostKind::AscDispatch);
    } else {
        // Without POST_ASC (or for the periodic VERIFY dispatch) synchronous, as before.
        split->Sync(verify_sync ? CpSplit::DrainReason::VerifyDispatch
                                : CpSplit::DrainReason::Dispatch);
    }
    split->MaybeLog();
}

void Liverpool::ExecuteGfxPacket(const PM4Header* header, bool guest_markers_enabled) {
    // Bodies moved verbatim from ProcessGraphics (the inline path calls this directly).
    const PM4ItOpcode opcode = header->type3.opcode;
    switch (opcode) {
    case PM4ItOpcode::Nop: {
        const auto* nop = reinterpret_cast<const PM4CmdNop*>(header);
        if (nop->header.count.Value() == 0) {
            break;
        }

        switch (nop->data_block[0]) {
        case PM4CmdNop::PayloadType::PatchedFlip: {
            // There is no evidence that GPU CP drives flip events by parsing
            // special NOP packets. For convenience lets assume that it does.
            Platform::IrqC::Instance()->Signal(Platform::InterruptId::GfxFlip);
            break;
        }
        case PM4CmdNop::PayloadType::DebugMarkerPush: {
            if (guest_markers_enabled) {
                const auto marker_sz = nop->header.count.Value() * 2;
                const std::string_view label{reinterpret_cast<const char*>(&nop->data_block[1]),
                                             marker_sz};
                rasterizer->ScopeMarkerBegin(label, true);
            }
            break;
        }
        case PM4CmdNop::PayloadType::DebugColorMarkerPush: {
            if (guest_markers_enabled) {
                const auto marker_sz = nop->header.count.Value() * 2;
                const std::string_view label{reinterpret_cast<const char*>(&nop->data_block[1]),
                                             marker_sz};
                const u32 color = *reinterpret_cast<const u32*>(
                    reinterpret_cast<const u8*>(&nop->data_block[1]) + marker_sz);
                rasterizer->ScopedMarkerInsertColor(label, color, true);
            }
            break;
        }
        case PM4CmdNop::PayloadType::DebugMarkerPop: {
            if (guest_markers_enabled) {
                rasterizer->ScopeMarkerEnd(true);
            }
            break;
        }
        default:
            break;
        }
        break;
    }
    case PM4ItOpcode::EventWrite: {
        // The streamout-flush part is a register write and stays in ProcessGraphics.
        const auto* event = reinterpret_cast<const PM4CmdEventWrite*>(header);
        // SHADPS4_OCCLUSION_FAKE / SHADPS4_OCCLUSION_LOG (2026-10-02, probe; see
        // scripts/patch/occlusion-probe-2026-10-02.py): what the game's occlusion queries are answered
        // with, and a log of their pattern. Defaults keep the upstream fake (always "visible").
        static const int occlusion_fake = [] {
            const char* value = std::getenv("SHADPS4_OCCLUSION_FAKE");
            return value ? std::atoi(value) : 1;
        }();
        static const bool occlusion_log = [] {
            const char* value = std::getenv("SHADPS4_OCCLUSION_LOG");
            return value && value[0] == '1';
        }();
        if (occlusion_log && event->event_type.Value() != EventType::PixelPipeStatDump &&
            (event->event_index.Value() == EventIndex::ZpassDone ||
             event->event_type.Value() == EventType::PixelPipeStatControl ||
             event->event_type.Value() == EventType::PixelPipeStatReset)) {
            static u32 other_logged = 0;
            if (other_logged < 16) {
                ++other_logged;
                LOG_INFO(Render, "OcclusionQ: other event type={} index={} frame={}",
                         static_cast<u32>(event->event_type.Value()),
                         static_cast<u32>(event->event_index.Value()),
                         Common::FaultMonitor::presented_frames.load(std::memory_order_relaxed));
            }
        }
        if (event->event_index.Value() == EventIndex::ZpassDone) {
            if (event->event_type.Value() == EventType::PixelPipeStatDump) {
                static constexpr u64 OcclusionCounterValidMask = 0x8000000000000000ULL;
                static constexpr u64 OcclusionCounterStep = 0x2FFFFFFULL;
                u64* results = event->Address<u64*>();
                // SHADPS4_OCCLUSION_QUERIES=1: real queries (vk_occlusion.h) publish their counts
                // here; the probe and the fake below are then unused.
                if (rasterizer && rasterizer->OcclusionDump(results, num_counter_pairs)) {
                    break;
                }
                if (occlusion_log) [[unlikely]] {
                    static u64 dumps = 0, window_dumps = 0, window_slot8 = 0, window_frame0 = 0;
                    static std::unordered_set<u64> window_addrs;
                    const u64 frame =
                        Common::FaultMonitor::presented_frames.load(std::memory_order_relaxed);
                    const u64 addr = reinterpret_cast<u64>(results);
                    if (dumps < 48) {
                        LOG_INFO(Render,
                                 "OcclusionQ: dump #{} addr={:#x} slot={} had={:#x} next={:#x} "
                                 "counter={:#x} frame={}",
                                 dumps, addr, (addr & 8) ? 1 : 0, results[0], results[1],
                                 pixel_counter, frame);
                    }
                    if (window_dumps == 0) {
                        window_frame0 = frame;
                    }
                    ++dumps;
                    ++window_dumps;
                    window_slot8 += (addr & 8) ? 1 : 0;
                    window_addrs.insert(addr);
                    if (window_dumps == 2048) {
                        const u64 frames = frame - window_frame0;
                        LOG_INFO(Render,
                                 "OcclusionQ: dumps={} frames={} dumps/frame={:.1f} distinct_addrs={} "
                                 "slot1_share={:.1f}% fake_mode={} total={}",
                                 window_dumps, frames,
                                 frames ? static_cast<double>(window_dumps) / frames : 0.0,
                                 window_addrs.size(), 100.0 * window_slot8 / window_dumps,
                                 occlusion_fake, dumps);
                        window_dumps = 0;
                        window_slot8 = 0;
                        window_addrs.clear();
                    }
                }
                if (occlusion_fake != 2) {
                    for (s32 i = 0; i < num_counter_pairs; ++i, results += 2) {
                        *results = pixel_counter | OcclusionCounterValidMask;
                    }
                    if (occlusion_fake != 0) {
                        pixel_counter += OcclusionCounterStep;
                    }
                }
            }
        }
        break;
    }
    case PM4ItOpcode::EventWriteEos: {
        const auto* event_eos = reinterpret_cast<const PM4CmdEventWriteEos*>(header);
        if (rasterizer) {
            rasterizer->ProcessDownloadImages();
        }
        // g_cp_wait and SplitHazard are front-side diagnostics noted here on the back. Race-free
        // because fence/WRITE_DATA packets stay synchronous at every depth (front parked), and
        // POST_DMA is refused while SplitHazard is on. POST_ASC is refused with SplitHazard, and
        // its WRITE_DATA bit (posted ASC WRITE_DATA runs the WRITE_DATA case below) with
        // CP_WAIT_STATS. SPLIT-TODO(run-ahead): note these writes at parse time if these packets
        // are ever posted (inventory section 3).
        g_cp_wait.NoteWrite(event_eos->Address());
        SplitHazard::NoteWrite(reinterpret_cast<u64>(event_eos->Address()), 8);
        if (Common::ReadbackOffload::LatencyEnabled() && rasterizer) [[unlikely]] {
            Common::ReadbackOffload::g_last_gfx_label_tick.store(
                rasterizer->GetScheduler().CurrentTick(), std::memory_order_relaxed);
        }
        VideoCore::ProtectFence();
        if ((FenceAtGpu() == 1 || FenceAtGpu() == 2) && rasterizer &&
            event_eos->command != PM4CmdEventWriteEos::Command::GdsStore) {
            rasterizer->DeferToGpu([packet = *event_eos] { packet.SignalFence(WriteFenceValue); },
                                   FenceAtGpu() == 1);
        } else {
            event_eos->SignalFence(WriteFenceValue);
        }
        if (event_eos->command == PM4CmdEventWriteEos::Command::GdsStore) {
            ASSERT(event_eos->size == 1);
            if (rasterizer) {
                rasterizer->Finish();
                const u32 value = rasterizer->ReadDataFromGds(event_eos->gds_index);
                *event_eos->Address() = value;
            }
        }
        break;
    }
    case PM4ItOpcode::EventWriteEop: {
        const auto* event_eop = reinterpret_cast<const PM4CmdEventWriteEop*>(header);
        if (rasterizer) {
            rasterizer->ProcessDownloadImages();
        }
        g_cp_wait.NoteWrite(event_eop->Address<u32>());
        SplitHazard::NoteWrite(reinterpret_cast<u64>(event_eop->Address<u32>()), 8);
        if (Common::ReadbackOffload::LatencyEnabled() && rasterizer) [[unlikely]] {
            Common::ReadbackOffload::g_last_gfx_label_tick.store(
                rasterizer->GetScheduler().CurrentTick(), std::memory_order_relaxed);
        }
        VideoCore::ProtectFence();
        if (FenceAtGpu() != 0 && rasterizer) {
            rasterizer->DeferToGpu(
                [packet = *event_eop] {
                    packet.SignalFence(WriteFenceValue, [] {
                        Platform::IrqC::Instance()->Signal(Platform::InterruptId::GfxEop);
                    });
                },
                FenceAtGpu() != 2);
            break;
        }
        event_eop->SignalFence(
            [](void* address, u64 data, u32 num_bytes) {
                auto* memory = Core::Memory::Instance();
                if (!memory->TryWriteBacking(address, &data, num_bytes)) {
                    memcpy(address, &data, num_bytes);
                }
            },
            [] { Platform::IrqC::Instance()->Signal(Platform::InterruptId::GfxEop); });
        if (AscRingLatency()) [[unlikely]] {
            g_asc_ring.last_eop_tsc.store(__rdtsc(), std::memory_order_relaxed);
        }
        break;
    }
    case PM4ItOpcode::DmaData: {
        const auto* dma_data = reinterpret_cast<const PM4DmaData*>(header);
        if (dma_data->dst_addr_lo == 0x3022C || !rasterizer) {
            break;
        }
        if (dma_data->src_sel == DmaDataSrc::Data && dma_data->dst_sel == DmaDataDst::Gds) {
            rasterizer->FillBuffer(dma_data->dst_addr_lo, dma_data->NumBytes(), dma_data->data,
                                   true);
        } else if ((dma_data->src_sel == DmaDataSrc::Memory ||
                    dma_data->src_sel == DmaDataSrc::MemoryUsingL2) &&
                   dma_data->dst_sel == DmaDataDst::Gds) {
            rasterizer->CopyBuffer(dma_data->dst_addr_lo, dma_data->SrcAddress<VAddr>(),
                                   dma_data->NumBytes(), true, false);
        } else if (dma_data->src_sel == DmaDataSrc::Data &&
                   (dma_data->dst_sel == DmaDataDst::Memory ||
                    dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
            rasterizer->FillBuffer(dma_data->DstAddress<VAddr>(), dma_data->NumBytes(),
                                   dma_data->data, false);
        } else if (dma_data->src_sel == DmaDataSrc::Gds &&
                   (dma_data->dst_sel == DmaDataDst::Memory ||
                    dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
            rasterizer->CopyBuffer(dma_data->DstAddress<VAddr>(), dma_data->src_addr_lo,
                                   dma_data->NumBytes(), false, true);
        } else if ((dma_data->src_sel == DmaDataSrc::Memory ||
                    dma_data->src_sel == DmaDataSrc::MemoryUsingL2) &&
                   (dma_data->dst_sel == DmaDataDst::Memory ||
                    dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
            rasterizer->CopyBuffer(dma_data->DstAddress<VAddr>(), dma_data->SrcAddress<VAddr>(),
                                   dma_data->NumBytes(), false, false);
        } else {
            UNREACHABLE_MSG("WriteData src_sel = {}, dst_sel = {}", u32(dma_data->src_sel.Value()),
                            u32(dma_data->dst_sel.Value()));
        }
        break;
    }
    case PM4ItOpcode::WriteData: {
        const auto* write_data = reinterpret_cast<const PM4CmdWriteData*>(header);
        ASSERT(write_data->dst_sel.Value() == 2 || write_data->dst_sel.Value() == 5);
        const u32 data_size = (header->type3.count.Value() - 2) * 4;
        u64* address = write_data->Address<u64*>();
        g_cp_wait.NoteWrite(address);
        SplitHazard::NoteWrite(reinterpret_cast<u64>(address), data_size);
        if (Common::ReadbackOffload::LatencyEnabled() && rasterizer) [[unlikely]] {
            Common::ReadbackOffload::g_last_gfx_label_tick.store(
                rasterizer->GetScheduler().CurrentTick(), std::memory_order_relaxed);
        }
        VideoCore::ProtectFence();
        if (!write_data->wr_one_addr.Value()) {
            std::memcpy(address, write_data->data, data_size);
        } else {
            UNREACHABLE();
        }
        break;
    }
    case PM4ItOpcode::MemSemaphore: {
        // The wait half of a waiting semaphore stays in ProcessGraphics (it yields).
        const auto* mem_semaphore = reinterpret_cast<const PM4CmdMemSemaphore*>(header);
        if (mem_semaphore->IsSignaling()) {
            mem_semaphore->Signal();
        } else {
            mem_semaphore->Decrement();
        }
        break;
    }
    case PM4ItOpcode::PfpSyncMe: {
        if (rasterizer) {
            rasterizer->CpSync();
        }
        break;
    }
    default:
        UNREACHABLE_MSG("CpSplit: no back action for PM4 opcode {:#x}", static_cast<u32>(opcode));
    }
}

void Liverpool::ProcessLoop(std::stop_token stoken) {
    while (!stoken.stop_requested()) {
        Common::SampleThreadPlacement(CpSplit::t_is_front ? "front" : "cp", 2);
        {
            // Idle here is the CP thread waiting on the GUEST, not emulator cost. Splitting it out
            // is what says whether parallel command recording can pay off - see
            // notes\parallel-command-recording-plan.md.
            const auto t_wait = Common::FaultMonitor::Now();
            const u64 idle_start = g_cp_wait.enabled ? CpWaitStats::Now() : 0;
            g_asc_ring_front_state.store(FrontGuestIdle, std::memory_order_relaxed);
            std::unique_lock lk{submit_mutex};
            Common::CondvarWait(submit_cv, lk, stoken,
                                [this] { return num_commands || num_submits || submit_done; });
            Common::FaultMonitor::CountCpWait(Common::FaultMonitor::SinceNs(t_wait));
            if (g_cp_wait.enabled) {
                g_cp_wait.idle_ns += CpWaitStats::Now() - idle_start;
                g_cp_wait.MaybeLog();
            }
        }
        if (stoken.stop_requested()) {
            break;
        }

        // NOTE: StartCapture() used to be called here, when this thread woke with work. Combined
        // with ending at the first completed submit, that recorded an arbitrary slice of a frame.
        // Both capture boundaries now sit at present, in Presenter::Present.

        curr_qid = -1;

        const auto t_active = Common::FaultMonitor::Now();
        while (num_submits || num_commands) {
            ProcessCommands();

            curr_qid = (curr_qid + 1) % num_mapped_queues;

            auto& queue = mapped_queues[curr_qid];

            Task::Handle task{};
            {
                std::scoped_lock lock{queue.m_access};
                if (queue.submits.empty()) {
                    continue;
                }
                task = queue.submits.front();
            }
            if (split_task_guard) [[unlikely]] {
                // TASK_GUARD: a finished task no longer drains, so a DMA / WRITE_DATA it posted
                // may still be outstanding when the front resumes a task that is part-way through
                // its buffers. A task not started yet checks its own buffer at entry instead.
                SplitGuardActive();
            }
            const bool ring_latency = split_run_ahead && AscRingLatency();
            if (ring_latency) [[unlikely]] {
                g_asc_ring_front_state.store(curr_qid == s32(GfxQueueId) ? FrontGfxTask
                                                                         : FrontAscTask,
                                             std::memory_order_relaxed);
                if (g_asc_ring.front_queue != u32(curr_qid)) {
                    // Stream switch, in stream order: rides along with the task's next record.
                    g_asc_ring.front_queue = u32(curr_qid);
                    EmitAscRingDiag(*split, AscRingSwitch, u32(curr_qid));
                }
            }
            task.resume();

            if (task.done()) {
                if (ring_latency && curr_qid != s32(GfxQueueId)) [[unlikely]] {
                    EmitAscRingDiag(*split, AscRingEnd, u32(curr_qid));
                }
                // CP split: the guest reads num_submits == 0 as "GPU idle", so the back must
                // have finished the submission first. POST_TASK_DONE: instead of draining, count
                // it pending and post a Retire record; WaitGpuIdle/IsGpuIdle also wait for
                // pending_retire, which the back decrements when it reaches the record.
                if (split_post_task_done) {
                    if (!split_task_guard) {
                        // The old drain also made this task's posted DMA / WRITE_DATA land before
                        // the next top-level buffer is parsed (the nested-IB guards cover only
                        // IBs). TASK_GUARD checks each buffer against the posted writes instead,
                        // when the front starts it (SplitSpanScope) or resumes a task
                        // (SplitGuardActive above).
                        split->SyncIfPostedPackets(CpSplit::DrainReason::TaskDone);
                    }
                    pending_retire.fetch_add(1, std::memory_order_seq_cst);
                    split->PostRetire();
                } else {
                    FrontDrain(CpSplit::DrainReason::TaskDone);
                }
                task.destroy();

                std::scoped_lock lock{queue.m_access};
                queue.submits.pop();
                if (curr_qid != static_cast<s32>(GfxQueueId)) {
                    asc_outstanding.fetch_sub(1, std::memory_order_release);
                    ++asc_done_seq;
                }

                --num_submits;
                std::scoped_lock lock2{submit_mutex};
                submit_cv.notify_all();
            }
        }
        Common::FaultMonitor::CountCpActive(Common::FaultMonitor::SinceNs(t_active));
        if (split_enabled) {
            // CP split: the idle tail uses the scheduler, the caches and IrqC (back work). Same
            // order as below; submit_done is consumed here on the front.
            const bool flush_submit = submit_done;
            if (flush_submit) {
                submit_done = false;
            }
            OnBack(
                [&] {
                    if (FenceAtGpu() == 2 && rasterizer) {
                        rasterizer->FlushLazyFences();
                    }
                    if (flush_submit && rasterizer) {
                        rasterizer->OnSubmit();
                        rasterizer->Flush();
                    }
                    VideoCore::ProtectFence();
                    Platform::IrqC::Instance()->Signal(Platform::InterruptId::GpuIdle);
                },
                CpSplit::DrainReason::IdleTail);
            continue;
        }
        if (FenceAtGpu() == 2 && rasterizer) {
            rasterizer->FlushLazyFences();
        }

        if (submit_done) {
            // NOTE: EndCapture() used to live here, which ended the RenderDoc capture at the first
            // completed submit. A title splits one displayed frame across several submits, so that
            // recorded an arbitrary batch rather than a frame: captures had wildly varying content
            // and any batch that missed the final composite presented a black image. The capture
            // now ends after present, in Presenter::Present.
            if (rasterizer) {
                rasterizer->OnSubmit();
                rasterizer->Flush();
            }
            submit_done = false;
        }

        VideoCore::ProtectFence();
        Platform::IrqC::Instance()->Signal(Platform::InterruptId::GpuIdle);
    }
}

Liverpool::Task Liverpool::ProcessCeUpdate(std::span<const u32> ccb, bool is_indirect) {
    FIBER_ENTER(ccb_task_name);
    // TASK_GUARD: covers the top-level CCB and CE IBs (which had no posted-write guard before).
    SplitSpanScope split_span{*this, &ccb,
                              is_indirect ? CpSplit::DrainReason::TaskGuardIb
                                          : CpSplit::DrainReason::TaskGuard};

    while (!ccb.empty()) {
        ProcessCommands();

        const auto* header = reinterpret_cast<const PM4Header*>(ccb.data());
        const u32 type = header->type;
        if (type != 3) {
            // No other types of packets were spotted so far
            UNREACHABLE_MSG("Invalid PM4 type {}", type);
        }

        const PM4ItOpcode opcode = header->type3.opcode;
        const auto* it_body = reinterpret_cast<const u32*>(header) + 1;
        switch (opcode) {
        case PM4ItOpcode::Nop: {
            // const auto* nop = reinterpret_cast<const PM4CmdNop*>(header);
            break;
        }
        case PM4ItOpcode::WriteConstRam: {
            const auto* write_const = reinterpret_cast<const PM4WriteConstRam*>(header);
            memcpy(cblock.constants_heap.data() + write_const->Offset(), &write_const->data,
                   write_const->Size());
            break;
        }
        case PM4ItOpcode::DumpConstRam: {
            const auto* dump_const = reinterpret_cast<const PM4DumpConstRam*>(header);
            // CP split: a CP-side guest-memory write, so it runs on the back (fault identity).
            // A Call record: synchronous at every depth, so reading the front's heap is sound.
            // SPLIT-TODO(run-ahead): copy the heap slice into an owned record instead of reading
            // the front's constants_heap from the back.
            OnBack([&] {
                memcpy(dump_const->Address<void*>(),
                       cblock.constants_heap.data() + dump_const->Offset(), dump_const->Size());
            }, CpSplit::DrainReason::CallDumpConstRam);
            // A CP-side guest memory write: later draws may read it (submit_epoch.h).
            BumpEpoch(PM4ItOpcode::DumpConstRam);
            break;
        }
        case PM4ItOpcode::IncrementCeCounter: {
            ++cblock.ce_count;
            break;
        }
        case PM4ItOpcode::WaitOnDeCounterDiff: {
            const auto diff = it_body[0];
            while ((cblock.de_count - cblock.ce_count) >= diff) {
                YIELD_CE();
            }
            break;
        }
        case PM4ItOpcode::IndirectBufferConst: {
            const auto* indirect_buffer = reinterpret_cast<const PM4CmdIndirectBuffer*>(header);
            auto task = ProcessCeUpdate(
                {indirect_buffer->Address<const u32>(), indirect_buffer->ib_size}, true);
            RESUME_CE(task);

            while (!task.handle.done()) {
                YIELD_CE();
                RESUME_CE(task);
            }
            break;
        }
        default:
            const u32 count = header->type3.NumWords();
            UNREACHABLE_MSG("Unknown PM4 type 3 opcode {:#x} with count {}",
                            static_cast<u32>(opcode), count);
        }
        ccb = NextPacket(ccb, header->type3.NumWords() + 1);
    }

    FIBER_EXIT;
}


Liverpool::Task Liverpool::ProcessGraphics(std::span<const u32> dcb, std::span<const u32> ccb) {
    FIBER_ENTER(dcb_task_name);
    // Everything the guest wrote before this submission is now due (submit_epoch.h). An indirect
    // buffer is part of its parent submission, so it does not reopen the window.
    const bool is_indirect_gfx = std::exchange(starting_indirect_gfx, false);
    if (!is_indirect_gfx) {
        BumpEpoch(PM4ItOpcode::Nop);
    }
    SplitSpanScope split_span{*this, &dcb,
                              is_indirect_gfx ? CpSplit::DrainReason::TaskGuardIb
                                              : CpSplit::DrainReason::TaskGuard};

    cblock.Reset();

    // TODO: potentially, ASCs also can depend on CE and in this case the
    // CE task should be moved into more global scope
    Task ce_task{};

    if (!ccb.empty()) {
        // In case of CCB provided kick off CE asap to have the constant heap ready to use
        ce_task = ProcessCeUpdate(ccb);
        RESUME_GFX(ce_task);
    }
    const bool host_markers_enabled = rasterizer && EmulatorSettings.IsVkHostMarkersEnabled();
    const bool guest_markers_enabled = rasterizer && EmulatorSettings.IsVkGuestMarkersEnabled();

    const auto base_addr = reinterpret_cast<uintptr_t>(dcb.data());
    const bool asc_yield = AscYield();
    while (!dcb.empty()) {
        ProcessCommands();

        if (asc_yield) {
            // A new ASC submission, or one finished while more are queued (the round-robin
            // returns here after each finished submission). A blocked ASC task finishes nothing,
            // so this yields at most once per ASC submit and once per ASC completion.
            const u64 seq = asc_submit_seq.load(std::memory_order_acquire);
            const bool next_queued = asc_done_seq != gfx_seen_asc_done &&
                                     asc_outstanding.load(std::memory_order_acquire) != 0;
            if (seq != gfx_seen_asc_seq || next_queued) {
                gfx_seen_asc_seq = seq;
                gfx_seen_asc_done = asc_done_seq;
                ++g_asc_start.yields;
                YIELD_GFX();
            }
        }

        const auto* header = reinterpret_cast<const PM4Header*>(dcb.data());
        const u32 type = header->type;

        switch (type) {
        default:
            UNREACHABLE_MSG("Wrong PM4 type {}", type);
            break;
        case 0:
            UNREACHABLE_MSG("Unimplemented PM4 type 0, base reg: {}, size: {}",
                            header->type0.base.Value(), header->type0.NumWords());
            break;
        case 2:
            // Type-2 packet are used for padding purposes
            dcb = NextPacket(dcb, 1);
            continue;
        case 3:
            const u32 count = header->type3.NumWords();
            const PM4ItOpcode opcode = header->type3.opcode;
            g_dispatch_sync.Note(64, opcode, header);
            switch (opcode) {
            case PM4ItOpcode::Nop: {
                // Flip IRQ and guest markers are back work; the body is in ExecuteGfxPacket.
                if (split_enabled) {
                    if (NopHasBackEffect(header, guest_markers_enabled)) {
                        RunGfxPacket(header, guest_markers_enabled,
                                     CpSplit::DrainReason::PacketOther);
                    }
                    break;
                }
                ExecuteGfxPacket(header, guest_markers_enabled);
                break;
            }
            case PM4ItOpcode::ContextControl: {
                break;
            }
            case PM4ItOpcode::ClearState: {
                regs.SetDefaults();
                ++context_change_epoch;
                if constexpr (Common::FaultMonitor::DetailCountersEnabled) {
                    ++regs_generation;
                    ++regs_generation_ctx;
                }
                if (split_enabled) [[unlikely]] {
                    split->EmitClearState();
                }
                break;
            }
            case PM4ItOpcode::SetConfigReg: {
                const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
                const auto reg_addr = Regs::ConfigRegWordOffset + set_data->reg_offset;
                const auto* payload = reinterpret_cast<const u32*>(header + 2);
                const auto sz = (count - 1) * sizeof(u32);
                if (Common::FaultMonitor::DetailCountersEnabled &&
                    std::memcmp(&regs.reg_array[reg_addr], payload, sz) != 0) {
                    ++regs_generation_ctx; // value-aware: only a real change dirties the key
                }
                std::memcpy(&regs.reg_array[reg_addr], payload, sz);
                JournalRegs(reg_addr, count - 1);
                if constexpr (Common::FaultMonitor::DetailCountersEnabled) { ++regs_generation; }
                break;
            }
            case PM4ItOpcode::SetContextReg: {
                const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
                const auto reg_addr = Regs::ContextRegWordOffset + set_data->reg_offset;
                const auto* payload = reinterpret_cast<const u32*>(header + 2);

                const auto ctx_sz = (count - 1) * sizeof(u32);
                if (Common::FaultMonitor::DetailCountersEnabled &&
                    std::memcmp(&regs.reg_array[reg_addr], payload, ctx_sz) != 0) {
                    ++regs_generation_ctx; // value-aware: only a real change dirties the key
                }
                if (ViewportMemoEnabled() &&
                    std::memcmp(&regs.reg_array[reg_addr], payload, ctx_sz) != 0) {
                    ++context_change_epoch;
                }
                std::memcpy(&regs.reg_array[reg_addr], payload, ctx_sz);
                JournalRegs(reg_addr, count - 1);
                if constexpr (Common::FaultMonitor::DetailCountersEnabled) { ++regs_generation; }

                // In the case of HW, render target memory has alignment as color block operates on
                // tiles. There is no information of actual resource extents stored in CB context
                // regs, so any deduction of it from slices/pitch will lead to a larger surface
                // created. The same applies to the depth targets. Fortunately, the guest always
                // sends a trailing NOP packet right after the context regs setup, so we can use the
                // heuristic below and extract the hint to determine actual resource dims.

                switch (reg_addr) {
                case ContextRegs::CbColor0Base:
                case ContextRegs::CbColor1Base:
                case ContextRegs::CbColor2Base:
                case ContextRegs::CbColor3Base:
                case ContextRegs::CbColor4Base:
                case ContextRegs::CbColor5Base:
                case ContextRegs::CbColor6Base:
                case ContextRegs::CbColor7Base: {
                    const auto col_buf_id = (reg_addr - ContextRegs::CbColor0Base) /
                                            (ContextRegs::CbColor1Base - ContextRegs::CbColor0Base);
                    ASSERT(col_buf_id < NUM_COLOR_BUFFERS);

                    const auto nop_offset = header->type3.count;
                    if (nop_offset == 0x0e || nop_offset == 0x0d || nop_offset == 0x0b) {
                        ASSERT_MSG(payload[nop_offset] == 0xc0001000,
                                   "NOP hint is missing in CB setup sequence");
                        last_cb_extent[col_buf_id].raw = payload[nop_offset + 1];
                    } else {
                        last_cb_extent[col_buf_id].raw = 0;
                    }
                    break;
                }
                case ContextRegs::CbColor0Cmask:
                case ContextRegs::CbColor1Cmask:
                case ContextRegs::CbColor2Cmask:
                case ContextRegs::CbColor3Cmask:
                case ContextRegs::CbColor4Cmask:
                case ContextRegs::CbColor5Cmask:
                case ContextRegs::CbColor6Cmask:
                case ContextRegs::CbColor7Cmask: {
                    const auto col_buf_id =
                        (reg_addr - ContextRegs::CbColor0Cmask) /
                        (ContextRegs::CbColor1Cmask - ContextRegs::CbColor0Cmask);
                    ASSERT(col_buf_id < NUM_COLOR_BUFFERS);

                    const auto nop_offset = header->type3.count;
                    if (nop_offset == 0x04) {
                        ASSERT_MSG(payload[nop_offset] == 0xc0001000,
                                   "NOP hint is missing in CB setup sequence");
                        last_cb_extent[col_buf_id].raw = payload[nop_offset + 1];
                    }
                    break;
                }
                case ContextRegs::DbZInfo: {
                    if (header->type3.count == 8) {
                        ASSERT_MSG(payload[20] == 0xc0001000,
                                   "NOP hint is missing in DB setup sequence");
                        last_db_extent.raw = payload[21];
                    } else {
                        last_db_extent.raw = 0;
                    }
                    break;
                }
                default:
                    break;
                }
                JournalExtent(reg_addr);
                break;
            }
            case PM4ItOpcode::SetShReg: {
                const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
                const auto set_size = (count - 1) * sizeof(u32);

                if (set_data->reg_offset >= 0x200 &&
                    set_data->reg_offset <= (0x200 + sizeof(ComputeProgram) / 4)) {
                    ASSERT(set_size <= sizeof(ComputeProgram));
                    auto* addr = reinterpret_cast<u32*>(&mapped_queues[GfxQueueId].cs_state) +
                                 (set_data->reg_offset - 0x200);
                    std::memcpy(addr, header + 2, set_size);
                } else {
                    std::memcpy(&regs.reg_array[Regs::ShRegWordOffset + set_data->reg_offset],
                                header + 2, set_size);
                    JournalRegs(Regs::ShRegWordOffset + set_data->reg_offset, count - 1);
                    if constexpr (Common::FaultMonitor::DetailCountersEnabled) { ++regs_generation; }
                    if (PgmPrefetchEnabled()) {
                        PrefetchProgramsOnWrite(regs, set_data->reg_offset, count - 1);
                    }
                }
                break;
            }
            case PM4ItOpcode::SetUconfigReg: {
                const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
                const auto uc_addr = Regs::UconfigRegWordOffset + set_data->reg_offset;
                const auto uc_sz = (count - 1) * sizeof(u32);
                if (Common::FaultMonitor::DetailCountersEnabled &&
                    std::memcmp(&regs.reg_array[uc_addr], header + 2, uc_sz) != 0) {
                    ++regs_generation_ctx; // value-aware: only a real change dirties the key
                }
                std::memcpy(&regs.reg_array[uc_addr], header + 2, uc_sz);
                JournalRegs(uc_addr, count - 1);
                if constexpr (Common::FaultMonitor::DetailCountersEnabled) { ++regs_generation; }
                break;
            }
            case PM4ItOpcode::SetPredication: {
                LOG_WARNING(Render, "Unimplemented IT_SET_PREDICATION");
                break;
            }
            case PM4ItOpcode::IndexType: {
                const auto* index_type = reinterpret_cast<const PM4CmdDrawIndexType*>(header);
                regs.index_buffer_type.raw = index_type->raw;
                JournalField(regs.index_buffer_type);
                break;
            }
            case PM4ItOpcode::DrawIndex2: {
                const auto* draw_index = reinterpret_cast<const PM4CmdDrawIndex2*>(header);
                regs.max_index_size = draw_index->max_size;
                regs.index_base_address.base_addr_lo = draw_index->index_base_lo;
                regs.index_base_address.base_addr_hi = draw_index->index_base_hi;
                regs.num_indices = draw_index->index_count;
                regs.draw_initiator = draw_index->draw_initiator;
                JournalField(regs.max_index_size);
                JournalField(regs.index_base_address);
                JournalField(regs.num_indices);
                JournalField(regs.draw_initiator);
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (split_enabled && rasterizer) {
                    EmitDraw(header, "DrawIndex2", host_markers_enabled, {.is_indexed = true});
                    break;
                }
                if (rasterizer) {
                    const auto cmd_address = reinterpret_cast<const void*>(header);
                    if (host_markers_enabled) {
                        rasterizer->ScopeMarkerBegin(fmt::format("gfx:{}:DrawIndex2", cmd_address));
                        rasterizer->Draw(true);
                        rasterizer->ScopeMarkerEnd();
                    } else {
                        rasterizer->Draw(true);
                    }
                }
                break;
            }
            case PM4ItOpcode::DrawIndexOffset2: {
                const auto* draw_index_off =
                    reinterpret_cast<const PM4CmdDrawIndexOffset2*>(header);
                regs.max_index_size = draw_index_off->max_size;
                regs.num_indices = draw_index_off->index_count;
                regs.draw_initiator = draw_index_off->draw_initiator;
                JournalField(regs.max_index_size);
                JournalField(regs.num_indices);
                JournalField(regs.draw_initiator);
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (split_enabled && rasterizer) {
                    EmitDraw(header, "DrawIndexOffset2", host_markers_enabled,
                             {.is_indexed = true, .index_offset = draw_index_off->index_offset});
                    break;
                }
                if (rasterizer) {
                    const auto cmd_address = reinterpret_cast<const void*>(header);
                    if (host_markers_enabled) {
                        rasterizer->ScopeMarkerBegin(
                            fmt::format("gfx:{}:DrawIndexOffset2", cmd_address));
                        rasterizer->Draw(true, draw_index_off->index_offset);
                        rasterizer->ScopeMarkerEnd();
                    } else {
                        rasterizer->Draw(true, draw_index_off->index_offset);
                    }
                }
                break;
            }
            case PM4ItOpcode::DrawIndexAuto: {
                const auto* draw_index = reinterpret_cast<const PM4CmdDrawIndexAuto*>(header);
                regs.num_indices = draw_index->index_count;
                regs.draw_initiator = draw_index->draw_initiator;
                JournalField(regs.num_indices);
                JournalField(regs.draw_initiator);
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (split_enabled && rasterizer) {
                    EmitDraw(header, "DrawIndexAuto", host_markers_enabled, {});
                    break;
                }
                if (rasterizer) {
                    const auto cmd_address = reinterpret_cast<const void*>(header);
                    if (host_markers_enabled) {
                        rasterizer->ScopeMarkerBegin(
                            fmt::format("gfx:{}:DrawIndexAuto", cmd_address));
                        rasterizer->Draw(false);
                        rasterizer->ScopeMarkerEnd();
                    } else {
                        rasterizer->Draw(false);
                    }
                }
                break;
            }
            case PM4ItOpcode::DrawIndirect: {
                const auto* draw_indirect = reinterpret_cast<const PM4CmdDrawIndirect*>(header);
                const auto offset = draw_indirect->data_offset;
                const auto stride = sizeof(DrawIndirectArgs);
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (split_enabled && rasterizer) {
                    EmitDraw(header, "DrawIndirect", host_markers_enabled,
                             {.is_indirect = true,
                              .arg_address = indirect_args_addr,
                              .offset = offset,
                              .stride = u32(stride),
                              .max_count = 1});
                    break;
                }
                if (rasterizer) {
                    const auto cmd_address = reinterpret_cast<const void*>(header);
                    if (host_markers_enabled) {
                        rasterizer->ScopeMarkerBegin(
                            fmt::format("gfx:{}:DrawIndirect", cmd_address));
                        rasterizer->DrawIndirect(false, indirect_args_addr, offset, stride, 1, 0);
                        rasterizer->ScopeMarkerEnd();
                    } else {
                        rasterizer->DrawIndirect(false, indirect_args_addr, offset, stride, 1, 0);
                    }
                }
                break;
            }
            case PM4ItOpcode::DrawIndirectMulti: {
                const auto* draw_indirect =
                    reinterpret_cast<const PM4CmdDrawIndirectMulti*>(header);
                const auto offset = draw_indirect->data_offset;
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (split_enabled && rasterizer) {
                    EmitDraw(header, "DrawIndirectMulti", host_markers_enabled,
                             {.is_indirect = true,
                              .arg_address = indirect_args_addr,
                              .offset = offset,
                              .stride = draw_indirect->stride,
                              .max_count = draw_indirect->count});
                    break;
                }
                if (rasterizer) {
                    const auto cmd_address = reinterpret_cast<const void*>(header);
                    if (host_markers_enabled) {
                        rasterizer->ScopeMarkerBegin(
                            fmt::format("gfx:{}:DrawIndirectMulti", cmd_address));
                        rasterizer->DrawIndirect(false, indirect_args_addr, offset,
                                                 draw_indirect->stride, draw_indirect->count, 0);
                        rasterizer->ScopeMarkerEnd();
                    } else {
                        rasterizer->DrawIndirect(false, indirect_args_addr, offset,
                                                 draw_indirect->stride, draw_indirect->count, 0);
                    }
                }
                break;
            }
            case PM4ItOpcode::DrawIndexIndirect: {
                const auto* draw_index_indirect =
                    reinterpret_cast<const PM4CmdDrawIndexIndirect*>(header);
                const auto offset = draw_index_indirect->data_offset;
                const auto stride = sizeof(DrawIndexedIndirectArgs);
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (split_enabled && rasterizer) {
                    EmitDraw(header, "DrawIndexIndirect", host_markers_enabled,
                             {.is_indexed = true,
                              .is_indirect = true,
                              .arg_address = indirect_args_addr,
                              .offset = offset,
                              .stride = u32(stride),
                              .max_count = 1});
                    break;
                }
                if (rasterizer) {
                    const auto cmd_address = reinterpret_cast<const void*>(header);
                    if (host_markers_enabled) {
                        rasterizer->ScopeMarkerBegin(
                            fmt::format("gfx:{}:DrawIndexIndirect", cmd_address));
                        rasterizer->DrawIndirect(true, indirect_args_addr, offset, stride, 1, 0);
                        rasterizer->ScopeMarkerEnd();
                    } else {
                        rasterizer->DrawIndirect(true, indirect_args_addr, offset, stride, 1, 0);
                    }
                }
                break;
            }
            case PM4ItOpcode::DrawIndexIndirectMulti: {
                const auto* draw_index_indirect =
                    reinterpret_cast<const PM4CmdDrawIndexIndirectMulti*>(header);
                const auto offset = draw_index_indirect->data_offset;
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (split_enabled && rasterizer) {
                    EmitDraw(header, "DrawIndexIndirectMulti", host_markers_enabled,
                             {.is_indexed = true,
                              .is_indirect = true,
                              .arg_address = indirect_args_addr,
                              .offset = offset,
                              .stride = draw_index_indirect->stride,
                              .max_count = draw_index_indirect->count});
                    break;
                }
                if (rasterizer) {
                    const auto cmd_address = reinterpret_cast<const void*>(header);
                    if (host_markers_enabled) {
                        rasterizer->ScopeMarkerBegin(
                            fmt::format("gfx:{}:DrawIndexIndirectMulti", cmd_address));
                        rasterizer->DrawIndirect(true, indirect_args_addr, offset,
                                                 draw_index_indirect->stride,
                                                 draw_index_indirect->count, 0);
                        rasterizer->ScopeMarkerEnd();
                    } else {
                        rasterizer->DrawIndirect(true, indirect_args_addr, offset,
                                                 draw_index_indirect->stride,
                                                 draw_index_indirect->count, 0);
                    }
                }
                break;
            }
            case PM4ItOpcode::DrawIndexIndirectCountMulti: {
                const auto* draw_index_indirect =
                    reinterpret_cast<const PM4CmdDrawIndexIndirectCountMulti*>(header);
                const auto offset = draw_index_indirect->data_offset;
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDump(base_addr, reinterpret_cast<uintptr_t>(header), regs);
                }
                if (split_enabled && rasterizer) {
                    EmitDraw(header, "DrawIndexIndirectCountMulti", host_markers_enabled,
                             {.is_indexed = true,
                              .is_indirect = true,
                              .arg_address = indirect_args_addr,
                              .offset = offset,
                              .stride = draw_index_indirect->stride,
                              .max_count = draw_index_indirect->count,
                              .count_address = draw_index_indirect->count_indirect_enable.Value()
                                                   ? VAddr(draw_index_indirect->count_addr)
                                                   : VAddr{0}});
                    break;
                }
                if (rasterizer) {
                    const auto cmd_address = reinterpret_cast<const void*>(header);
                    if (host_markers_enabled) {
                        rasterizer->ScopeMarkerBegin(
                            fmt::format("gfx:{}:DrawIndexIndirectCountMulti", cmd_address));
                        rasterizer->DrawIndirect(true, indirect_args_addr, offset,
                                                 draw_index_indirect->stride,
                                                 draw_index_indirect->count,
                                                 draw_index_indirect->count_indirect_enable.Value()
                                                     ? draw_index_indirect->count_addr
                                                     : 0);
                        rasterizer->ScopeMarkerEnd();
                    } else {
                        rasterizer->DrawIndirect(true, indirect_args_addr, offset,
                                                 draw_index_indirect->stride,
                                                 draw_index_indirect->count,
                                                 draw_index_indirect->count_indirect_enable.Value()
                                                     ? draw_index_indirect->count_addr
                                                     : 0);
                    }
                }
                break;
            }
            case PM4ItOpcode::DispatchDirect: {
                const auto* dispatch_direct = reinterpret_cast<const PM4CmdDispatchDirect*>(header);
                auto& cs_program = GetCsRegs();
                cs_program.dim_x = dispatch_direct->dim_x;
                cs_program.dim_y = dispatch_direct->dim_y;
                cs_program.dim_z = dispatch_direct->dim_z;
                cs_program.dispatch_initiator = dispatch_direct->dispatch_initiator;
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDumpCompute(base_addr, reinterpret_cast<uintptr_t>(header),
                                                   cs_program);
                }
                if (split_enabled) {
                    if (rasterizer && (cs_program.dispatch_initiator & 1)) {
                        EmitDispatch(header, "DispatchDirect", host_markers_enabled, false, 0, 0,
                                     0, GfxQueueId);
                    }
                    break;
                }
                if (rasterizer && (cs_program.dispatch_initiator & 1)) {
                    const auto cmd_address = reinterpret_cast<const void*>(header);
                    if (host_markers_enabled) {
                        rasterizer->ScopeMarkerBegin(
                            fmt::format("gfx:{}:DispatchDirect", cmd_address));
                        rasterizer->DispatchDirect();
                        rasterizer->ScopeMarkerEnd();
                    } else {
                        rasterizer->DispatchDirect();
                    }
                }
                break;
            }
            case PM4ItOpcode::DispatchIndirect: {
                const auto* dispatch_indirect =
                    reinterpret_cast<const PM4CmdDispatchIndirect*>(header);
                auto& cs_program = GetCsRegs();
                const auto offset = dispatch_indirect->data_offset;
                const auto size = sizeof(PM4CmdDispatchIndirect::GroupDimensions);
                if (DebugState.DumpingCurrentReg()) {
                    DebugState.PushRegsDumpCompute(base_addr, reinterpret_cast<uintptr_t>(header),
                                                   cs_program);
                }
                if (split_enabled) {
                    if (rasterizer && (cs_program.dispatch_initiator & 1)) {
                        EmitDispatch(header, "DispatchIndirect", host_markers_enabled, true,
                                     indirect_args_addr, offset, size, GfxQueueId);
                    }
                    break;
                }
                if (rasterizer && (cs_program.dispatch_initiator & 1)) {
                    const auto cmd_address = reinterpret_cast<const void*>(header);
                    if (host_markers_enabled) {
                        rasterizer->ScopeMarkerBegin(
                            fmt::format("gfx:{}:DispatchIndirect", cmd_address));
                        rasterizer->DispatchIndirect(indirect_args_addr, offset, size);
                        rasterizer->ScopeMarkerEnd();
                    } else {
                        rasterizer->DispatchIndirect(indirect_args_addr, offset, size);
                    }
                }
                break;
            }
            case PM4ItOpcode::NumInstances: {
                const auto* num_instances = reinterpret_cast<const PM4CmdDrawNumInstances*>(header);
                regs.num_instances.num_instances = num_instances->num_instances;
                JournalField(regs.num_instances);
                break;
            }
            case PM4ItOpcode::IndexBase: {
                const auto* index_base = reinterpret_cast<const PM4CmdDrawIndexBase*>(header);
                regs.index_base_address.base_addr_lo = index_base->addr_lo;
                regs.index_base_address.base_addr_hi = index_base->addr_hi;
                JournalField(regs.index_base_address);
                break;
            }
            case PM4ItOpcode::IndexBufferSize: {
                const auto* index_size = reinterpret_cast<const PM4CmdDrawIndexBufferSize*>(header);
                regs.num_indices = index_size->num_indices;
                JournalField(regs.num_indices);
                break;
            }
            case PM4ItOpcode::SetBase: {
                const auto* set_base = reinterpret_cast<const PM4CmdSetBase*>(header);
                ASSERT(set_base->base_index == PM4CmdSetBase::BaseIndex::DrawIndexIndirPatchTable);
                indirect_args_addr = set_base->Address<u64>();
                break;
            }
            case PM4ItOpcode::EventWrite: {
                const auto* event = reinterpret_cast<const PM4CmdEventWrite*>(header);
                LOG_DEBUG(Render, "Encountered EventWrite: event_type = {}, event_index = {}",
                          magic_enum::enum_name(event->event_type.Value()),
                          magic_enum::enum_name(event->event_index.Value()));
                if (event->event_type.Value() == EventType::SoVgtStreamoutFlush) {
                    // TODO: handle proper synchronization, for now signal that update is done
                    // immediately
                    regs.cp_strmout_cntl.offset_update_done = 1;
                    JournalField(regs.cp_strmout_cntl);
                } else if (!split_enabled) {
                    ExecuteGfxPacket(header, guest_markers_enabled);
                } else if (event->event_index.Value() == EventIndex::ZpassDone &&
                           event->event_type.Value() == EventType::PixelPipeStatDump) {
                    // Occlusion counter write into guest memory: back work.
                    RunGfxPacket(header, guest_markers_enabled,
                                 CpSplit::DrainReason::PacketOther);
                }
                break;
            }
            // Fences, DMA, CP memory writes and the CP sync barrier: back work, bodies in
            // ExecuteGfxPacket. Synchronous at every depth, except DMA_DATA under POST_DMA.
            case PM4ItOpcode::EventWriteEos:
            case PM4ItOpcode::EventWriteEop: {
                if (split_post_fence) [[unlikely]] {
                    VAddr label = 0;
                    if (opcode == PM4ItOpcode::EventWriteEop) {
                        label = reinterpret_cast<VAddr>(
                            reinterpret_cast<const PM4CmdEventWriteEop*>(header)->Address<u32>());
                    } else if (const auto* eos =
                                   reinterpret_cast<const PM4CmdEventWriteEos*>(header);
                               eos->command != PM4CmdEventWriteEos::Command::GdsStore) {
                        label = eos->Address<VAddr>();
                    }
                    if (label != 0) {
                        RunFence(header, dcb, label,
                                 guest_markers_enabled ? u16{CpSplit::GuestMarkers} : u16{0},
                                 CpSplit::DrainReason::PacketFence);
                        break;
                    }
                }
                RunGfxPacket(header, guest_markers_enabled, CpSplit::DrainReason::PacketFence);
                break;
            }
            case PM4ItOpcode::DmaData: {
                RunGfxDma(header, guest_markers_enabled, dcb);
                break;
            }
            case PM4ItOpcode::WriteData: {
                RunGfxPacket(header, guest_markers_enabled,
                             CpSplit::DrainReason::PacketWriteData);
                break;
            }
            case PM4ItOpcode::PfpSyncMe: {
                RunGfxPacket(header, guest_markers_enabled, CpSplit::DrainReason::PacketOther);
                break;
            }
            case PM4ItOpcode::CopyData: {
                const auto* copy_data = reinterpret_cast<const PM4CmdCopyData*>(header);
                LOG_WARNING(Render,
                            "unhandled IT_COPY_DATA src_sel = {}, dst_sel = {}, "
                            "count_sel = {}, wr_confirm = {}, engine_sel = {}",
                            u32(copy_data->src_sel.Value()), u32(copy_data->dst_sel.Value()),
                            copy_data->count_sel.Value(), copy_data->wr_confirm.Value(),
                            u32(copy_data->engine_sel.Value()));
                break;
            }
            case PM4ItOpcode::MemSemaphore: {
                const auto* mem_semaphore = reinterpret_cast<const PM4CmdMemSemaphore*>(header);
                if (!mem_semaphore->IsSignaling()) {
                    // The front reads guest memory the back may still write: drain first.
                    FrontDrain(CpSplit::DrainReason::MemSemaphore);
                    while (!mem_semaphore->Signaled()) {
                        YIELD_GFX();
                    }
                }
                // Signal() or Decrement(), in ExecuteGfxPacket.
                RunGfxPacket(header, guest_markers_enabled, CpSplit::DrainReason::MemSemaphore);
                break;
            }
            case PM4ItOpcode::AcquireMem: {
                // const auto* acquire_mem = reinterpret_cast<PM4CmdAcquireMem*>(header);
                break;
            }
            case PM4ItOpcode::Rewind: {
                if (!rasterizer) {
                    break;
                }
                const PM4CmdRewind* rewind = reinterpret_cast<const PM4CmdRewind*>(header);
                // The flag is written by the guest CPU, not the back: no drain needed.
                while (!rewind->Valid()) {
                    YIELD_GFX();
                }
                break;
            }
            case PM4ItOpcode::WaitRegMem: {
                const auto* wait_reg_mem = reinterpret_cast<const PM4CmdWaitRegMem*>(header);
                // ASSERT(wait_reg_mem->engine.Value() == PM4CmdWaitRegMem::Engine::Me);
                // Optimization: VO label waits are special because the emulator
                // will write to the label when presentation is finished. So if
                // there are no other submits to yield to we can sleep the thread
                // instead and allow other tasks to run.
                const u64* wait_addr = wait_reg_mem->Address<u64*>();
                // CP split: the back may still owe writes this wait reads (inventory 0.6), so
                // drain before testing. At depth 0 this only flushes register records.
                FrontDrain(CpSplit::DrainReason::WaitRegMem);
                if (vo_port->IsVoLabel(wait_addr) &&
                    num_submits == mapped_queues[GfxQueueId].submits.size()) {
                    const auto vo_start = Common::FaultMonitor::Now();
                    vo_port->WaitVoLabel([&] { return wait_reg_mem->Test(regs.reg_array); });
                    Common::FaultMonitor::CountVoLabelWait(
                        Common::FaultMonitor::SinceNs(vo_start));
                    break;
                }
                const auto wrm_start = Common::FaultMonitor::Now();
                const u64 wrm_ns0 = g_cp_wait.enabled ? CpWaitStats::Now() : 0;
                bool wrm_waited = false;
                if (FenceAtGpu() == 2 && rasterizer && !wait_reg_mem->Test(regs.reg_array)) {
                    OnBack([&] { rasterizer->FlushLazyFences(); },
                           CpSplit::DrainReason::WaitRegMem);
                }
                while (!wait_reg_mem->Test(regs.reg_array)) {
                    wrm_waited = true;
                    YIELD_GFX();
                }
                if (g_cp_wait.enabled) {
                    if (wrm_waited) {
                        g_cp_wait.CountWait(true, wait_reg_mem->Address<u64*>(),
                                            CpWaitStats::Now() - wrm_ns0);
                    } else {
                        ++g_cp_wait.gfx_wrm_instant;
                    }
                }
                Common::FaultMonitor::CountWaitRegMem(Common::FaultMonitor::SinceNs(wrm_start));
                break;
            }
            case PM4ItOpcode::IndirectBuffer: {
                const auto* indirect_buffer = reinterpret_cast<const PM4CmdIndirectBuffer*>(header);
                if (split_task_guard < 2 &&
                    (split_post_dma || (split_post_asc & CpSplit::PostAscDma) ||
                     (split_post_asc & CpSplit::PostAscWriteData))) [[unlikely]] {
                    // A posted DMA may still be writing the IB about to be parsed. TASK_GUARD >= 2:
                    // the IB's own entry check (SplitSpanScope) does this for its span only.
                    split->SyncIfPostedPackets(CpSplit::DrainReason::IbStart);
                }
                auto task = ProcessGraphics(
                    {indirect_buffer->Address<const u32>(), indirect_buffer->ib_size}, {});
                starting_indirect_gfx = true;
                RESUME_GFX(task);

                while (!task.handle.done()) {
                    YIELD_GFX();
                    RESUME_GFX(task);
                }
                if (split_post_dma || (split_post_asc & CpSplit::PostAscWriteData)) [[unlikely]] {
                    // A posted DMA inside the IB may patch the rest of the parent level.
                    // TASK_GUARD 3: only a write into that rest drains (`dcb` still starts at
                    // this packet); higher levels check theirs at their own IB end.
                    if (split_task_guard >= 3) {
                        SplitGuardSpan(dcb, CpSplit::DrainReason::IbEnd);
                    } else {
                        split->SyncIfPostedPackets(CpSplit::DrainReason::IbEnd);
                    }
                }
                if (IbTaskFree()) {
                    task.handle.destroy();
                }
                break;
            }
            case PM4ItOpcode::IncrementDeCounter: {
                ++cblock.de_count;
                break;
            }
            case PM4ItOpcode::WaitOnCeCounter: {
                while (cblock.ce_count <= cblock.de_count && !ce_task.handle.done()) {
                    if (split_task_guard) [[unlikely]] {
                        // The DE resumes the CE directly, not through ProcessLoop: check the
                        // CE's unparsed rest against this task's posted writes.
                        SplitGuardActive();
                    }
                    RESUME_GFX(ce_task);
                }
                break;
            }
            case PM4ItOpcode::StrmoutBufferUpdate: {
                const auto* strmout = reinterpret_cast<const PM4CmdStrmoutBufferUpdate*>(header);
                LOG_WARNING(Render_Vulkan,
                            "Unimplemented IT_STRMOUT_BUFFER_UPDATE, update_memory = {}, "
                            "source_select = {}, buffer_select = {}",
                            strmout->update_memory.Value(),
                            magic_enum::enum_name(strmout->source_select.Value()),
                            strmout->buffer_select.Value());
                break;
            }
            case PM4ItOpcode::GetLodStats: {
                LOG_WARNING(Render_Vulkan, "Unimplemented IT_GET_LOD_STATS");
                break;
            }
            case PM4ItOpcode::CondExec: {
                const auto* cond_exec = reinterpret_cast<const PM4CmdCondExec*>(header);
                if (cond_exec->command.Value() != 0) {
                    LOG_WARNING(Render, "IT_COND_EXEC used a reserved command");
                }
                // Reads guest memory the back may write.
                FrontDrain(CpSplit::DrainReason::CondExec);
                const auto skip = *cond_exec->Address() == false;
                if (skip) {
                    dcb = NextPacket(dcb,
                                     header->type3.NumWords() + 1 + cond_exec->exec_count.Value());
                    continue;
                }
                break;
            }
            default:
                UNREACHABLE_MSG("Unknown PM4 type 3 opcode {:#x} with count {}",
                                static_cast<u32>(opcode), count);
            }
            if (!KeepsSubmitEpoch(opcode)) {
                BumpEpoch(opcode);
            }
            dcb = NextPacket(dcb, header->type3.NumWords() + 1);
            break;
        }
    }

    if (ce_task.handle) {
        while (!ce_task.handle.done()) {
            if (split_task_guard) [[unlikely]] {
                SplitGuardActive(); // As at WAIT_ON_CE_COUNTER.
            }
            RESUME_GFX(ce_task);
        }
        ce_task.handle.destroy();
    }

    FIBER_EXIT;
}

template <bool is_indirect>
Liverpool::Task Liverpool::ProcessCompute(std::span<const u32> acb, u32 vqid) {
    FIBER_ENTER(acb_task_name[vqid]);
    // CP split: ASC packets are parsed on the front like the DCB. Their back work runs through
    // Call records (OnBack), including the pipeline lookup inside DispatchDirect/Indirect. A
    // Call is a full drain at every depth (the front stays parked until it returns), so this
    // stays sound with run-ahead; each one costs a round trip. With POST_ASC (run-ahead),
    // dispatches are owned Dispatch records with the lookup on the front (EmitDispatch), and
    // DMA_DATA (and optionally WRITE_DATA) owned Packet records (RunAscPacket), all posted.
    // RELEASE_MEM, MEM_SEMAPHORE and WAIT_REG_MEM stay synchronous.
    if constexpr (!is_indirect) {
        BumpEpoch(PM4ItOpcode::Nop);
        if (AscStartStats()) [[unlikely]] {
            u64 submit_ns = 0;
            {
                auto& mq = mapped_queues[vqid + 1];
                std::scoped_lock lock{mq.m_access};
                auto& times = asc_submit_ns[vqid + 1];
                if (!times.empty()) {
                    submit_ns = times.front();
                    times.pop_front();
                }
            }
            if (submit_ns != 0) {
                NoteAscStart(submit_ns, vqid + 1);
            }
        }
        if (split_run_ahead && AscRingLatency()) [[unlikely]] {
            AscRingSnap submit{};
            bool have = false;
            {
                std::scoped_lock lock{g_asc_ring.submit_mutex};
                auto& stamps = g_asc_ring.submits[(vqid + 1) % AscRingQueues];
                if (!stamps.empty()) {
                    submit = stamps.front();
                    stamps.pop_front();
                    have = true;
                }
            }
            if (have) {
                EmitAscRingDiag(*split, AscRingBegin, vqid + 1, &submit, __rdtsc());
            }
        }
    }
    SplitSpanScope split_span{*this, &acb,
                              is_indirect ? CpSplit::DrainReason::TaskGuardIb
                                          : CpSplit::DrainReason::TaskGuard};
    auto& queue = asc_queues[{vqid}];
    const bool host_markers_enabled = rasterizer && EmulatorSettings.IsVkHostMarkersEnabled();

    struct IndirectPatch {
        const PM4Header* header;
        VAddr indirect_addr;
    };
    boost::container::small_vector<IndirectPatch, 4> indirect_patches;

    auto base_addr = reinterpret_cast<VAddr>(acb.data());
    size_t acb_size = acb.size_bytes();
    while (!acb.empty()) {
        ProcessCommands();

        auto* header = reinterpret_cast<const PM4Header*>(acb.data());
        u32 next_dw_off = header->type3.NumWords() + 1;

        // If we have a buffered packet, use it.
        if (queue.tmp_dwords > 0) [[unlikely]] {
            header = reinterpret_cast<const PM4Header*>(queue.tmp_packet.data());
            next_dw_off = header->type3.NumWords() + 1 - queue.tmp_dwords;
            std::memcpy(queue.tmp_packet.data() + queue.tmp_dwords, acb.data(),
                        next_dw_off * sizeof(u32));
            queue.tmp_dwords = 0;
        }

        // If the packet is split across ring boundary, buffer until next submission
        if (next_dw_off > acb.size()) [[unlikely]] {
            std::memcpy(queue.tmp_packet.data(), acb.data(), acb.size_bytes());
            queue.tmp_dwords = acb.size();
            if constexpr (!is_indirect) {
                *queue.read_addr += acb.size();
                *queue.read_addr %= queue.ring_size_dw;
            }
            break;
        }

        if (header->type == 2) {
            // Type-2 packet are used for padding purposes
            next_dw_off = 1;
            acb = NextPacket(acb, next_dw_off);
            if constexpr (!is_indirect) {
                *queue.read_addr += next_dw_off;
                *queue.read_addr %= queue.ring_size_dw;
            }
            continue;
        }

        if (header->type != 3) {
            // No other types of packets were spotted so far
            UNREACHABLE_MSG("Invalid PM4 type {}", header->type.Value());
        }

        const PM4ItOpcode opcode = header->type3.opcode;
        g_dispatch_sync.Note(vqid & 63, opcode, header);

        const auto* it_body = reinterpret_cast<const u32*>(header) + 1;
        switch (opcode) {
        case PM4ItOpcode::Nop: {
            const auto* nop = reinterpret_cast<const PM4CmdNop*>(header);
            break;
        }
        case PM4ItOpcode::IndirectBuffer: {
            const auto* indirect_buffer = reinterpret_cast<const PM4CmdIndirectBuffer*>(header);
            if (split_task_guard < 2 &&
                (split_post_dma || (split_post_asc & CpSplit::PostAscDma) ||
                 (split_post_asc & CpSplit::PostAscWriteData))) [[unlikely]] {
                // A posted DMA / WRITE_DATA may still be writing the IB about to be parsed.
                // TASK_GUARD >= 2: the IB's own entry check (SplitSpanScope) does this.
                split->SyncIfPostedPackets(CpSplit::DrainReason::AscIbStart);
            }
            auto task = ProcessCompute<true>(
                {indirect_buffer->Address<const u32>(), indirect_buffer->ib_size}, vqid);
            RESUME_ASC(task, vqid);

            while (!task.handle.done()) {
                YIELD_ASC(vqid);
                RESUME_ASC(task, vqid);
            }
            if ((split_post_asc & CpSplit::PostAscDma) ||
                (split_post_asc & CpSplit::PostAscWriteData)) [[unlikely]] {
                // A posted DMA / WRITE_DATA inside the IB may patch the rest of the parent level.
                // TASK_GUARD 3: only a write into that rest drains.
                if (split_task_guard >= 3) {
                    SplitGuardSpan(acb, CpSplit::DrainReason::IbEnd);
                } else {
                    split->SyncIfPostedPackets(CpSplit::DrainReason::IbEnd);
                }
            }
            if (IbTaskFree()) {
                task.handle.destroy();
            }
            break;
        }
        case PM4ItOpcode::DmaData: {
            const auto* dma_data = reinterpret_cast<const PM4DmaData*>(header);
            if (dma_data->dst_addr_lo == 0x3022C || !rasterizer) {
                break;
            }
            if (split_post_asc & CpSplit::PostAscDma) [[unlikely]] {
                // POST_ASC: the back runs ExecuteGfxPacket's DMA_DATA body, which is this one
                // minus the indirect-patch detection below (front-only bookkeeping, kept here).
                const bool src_mem = dma_data->src_sel == DmaDataSrc::Memory ||
                                     dma_data->src_sel == DmaDataSrc::MemoryUsingL2;
                const bool dst_mem = dma_data->dst_sel == DmaDataDst::Memory ||
                                     dma_data->dst_sel == DmaDataDst::MemoryUsingL2;
                if (src_mem && dst_mem) {
                    const VAddr dst_addr = dma_data->DstAddress<VAddr>();
                    const auto* patched =
                        reinterpret_cast<const PM4Header*>(dst_addr - sizeof(PM4Header));
                    if (dst_addr >= base_addr && dst_addr < base_addr + acb_size &&
                        dma_data->NumBytes() == sizeof(PM4CmdDispatchIndirect::GroupDimensions) &&
                        patched->type == 3 &&
                        patched->type3.opcode == PM4ItOpcode::DispatchDirect) {
                        indirect_patches.emplace_back(patched, dma_data->SrcAddress<VAddr>());
                        break;
                    }
                }
                RunAscPacket(header, acb, dst_mem ? dma_data->DstAddress<VAddr>() : VAddr{0},
                             dst_mem ? dma_data->NumBytes() : 0u,
                             CpSplit::DrainReason::AscDmaGuard);
                break;
            }
            if (dma_data->src_sel == DmaDataSrc::Data && dma_data->dst_sel == DmaDataDst::Gds) {
                OnBack(
                    [&] {
                        rasterizer->FillBuffer(dma_data->dst_addr_lo, dma_data->NumBytes(),
                                               dma_data->data, true);
                    },
                    CpSplit::DrainReason::CallAscDma);
            } else if ((dma_data->src_sel == DmaDataSrc::Memory ||
                        dma_data->src_sel == DmaDataSrc::MemoryUsingL2) &&
                       dma_data->dst_sel == DmaDataDst::Gds) {
                OnBack(
                    [&] {
                        rasterizer->CopyBuffer(dma_data->dst_addr_lo,
                                               dma_data->SrcAddress<VAddr>(),
                                               dma_data->NumBytes(), true, false);
                    },
                    CpSplit::DrainReason::CallAscDma);
            } else if (dma_data->src_sel == DmaDataSrc::Data &&
                       (dma_data->dst_sel == DmaDataDst::Memory ||
                        dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                OnBack(
                    [&] {
                        rasterizer->FillBuffer(dma_data->DstAddress<VAddr>(),
                                               dma_data->NumBytes(), dma_data->data, false);
                    },
                    CpSplit::DrainReason::CallAscDma);
            } else if (dma_data->src_sel == DmaDataSrc::Gds &&
                       (dma_data->dst_sel == DmaDataDst::Memory ||
                        dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                OnBack(
                    [&] {
                        rasterizer->CopyBuffer(dma_data->DstAddress<VAddr>(),
                                               dma_data->src_addr_lo, dma_data->NumBytes(),
                                               false, true);
                    },
                    CpSplit::DrainReason::CallAscDma);
            } else if ((dma_data->src_sel == DmaDataSrc::Memory ||
                        dma_data->src_sel == DmaDataSrc::MemoryUsingL2) &&
                       (dma_data->dst_sel == DmaDataDst::Memory ||
                        dma_data->dst_sel == DmaDataDst::MemoryUsingL2)) {
                const u32 num_bytes = dma_data->NumBytes();
                const VAddr src_addr = dma_data->SrcAddress<VAddr>();
                const VAddr dst_addr = dma_data->DstAddress<VAddr>();
                const PM4Header* header =
                    reinterpret_cast<const PM4Header*>(dst_addr - sizeof(PM4Header));
                if (dst_addr >= base_addr && dst_addr < base_addr + acb_size &&
                    num_bytes == sizeof(PM4CmdDispatchIndirect::GroupDimensions) &&
                    header->type == 3 && header->type3.opcode == PM4ItOpcode::DispatchDirect) {
                    indirect_patches.emplace_back(header, src_addr);
                } else {
                    OnBack(
                        [&] {
                            rasterizer->CopyBuffer(dst_addr, src_addr, num_bytes, false, false);
                        },
                        CpSplit::DrainReason::CallAscDma);
                }
            } else {
                UNREACHABLE_MSG("WriteData src_sel = {}, dst_sel = {}",
                                u32(dma_data->src_sel.Value()), u32(dma_data->dst_sel.Value()));
            }
            break;
        }
        case PM4ItOpcode::AcquireMem: {
            break;
        }
        case PM4ItOpcode::Rewind: {
            if (!rasterizer) {
                break;
            }
            const PM4CmdRewind* rewind = reinterpret_cast<const PM4CmdRewind*>(header);
            while (!rewind->Valid()) {
                YIELD_ASC(vqid);
            }
            break;
        }
        case PM4ItOpcode::SetShReg: {
            const auto* set_data = reinterpret_cast<const PM4CmdSetData*>(header);
            const auto set_size = (header->type3.NumWords() - 1) * sizeof(u32);

            if (set_data->reg_offset >= 0x200 &&
                set_data->reg_offset <= (0x200 + sizeof(ComputeProgram) / 4)) {
                ASSERT(set_size <= sizeof(ComputeProgram));
                auto* addr = reinterpret_cast<u32*>(&mapped_queues[vqid + 1].cs_state) +
                             (set_data->reg_offset - 0x200);
                std::memcpy(addr, header + 2, set_size);
            } else {
                std::memcpy(&regs.reg_array[Regs::ShRegWordOffset + set_data->reg_offset],
                            header + 2, set_size);
                JournalRegs(Regs::ShRegWordOffset + set_data->reg_offset,
                            u32(set_size / sizeof(u32)));
            }
            break;
        }
        case PM4ItOpcode::SetQueueReg: {
            const auto* set_data = reinterpret_cast<const PM4CmdSetQueueReg*>(header);
            LOG_WARNING(Render, "Encountered compute SetQueueReg: vqid = {}, reg_offset = {:#x}",
                        set_data->vqid.Value(), set_data->reg_offset.Value());
            break;
        }
        case PM4ItOpcode::DispatchDirect: {
            const auto* dispatch_direct = reinterpret_cast<const PM4CmdDispatchDirect*>(header);
            if (auto it = std::ranges::find(indirect_patches, header, &IndirectPatch::header);
                it != indirect_patches.end()) {
                const auto size = sizeof(PM4CmdDispatchIndirect::GroupDimensions);
                if ((split_post_asc & CpSplit::PostAscDispatch) && rasterizer) [[unlikely]] {
                    // As the Call below: the queue's cs_state as it is (dims not updated), no
                    // host marker. The back reads the patched dims in stream order.
                    EmitDispatch(header, "DispatchIndirect", false, true, it->indirect_addr, 0,
                                 u32(size), vqid + 1);
                    break;
                }
                OnBack([&] { rasterizer->DispatchIndirect(it->indirect_addr, 0, size); },
                       CpSplit::DrainReason::CallAscDispatchPatched);
                break;
            }
            auto& cs_program = GetCsRegs();
            cs_program.dim_x = dispatch_direct->dim_x;
            cs_program.dim_y = dispatch_direct->dim_y;
            cs_program.dim_z = dispatch_direct->dim_z;
            cs_program.dispatch_initiator = dispatch_direct->dispatch_initiator;
            if (DebugState.DumpingCurrentReg()) {
                DebugState.PushRegsDumpCompute(base_addr, reinterpret_cast<uintptr_t>(header),
                                               cs_program);
            }
            if (rasterizer && (cs_program.dispatch_initiator & 1)) {
                if (split_post_asc & CpSplit::PostAscDispatch) [[unlikely]] {
                    EmitDispatch(header, "DispatchDirect", host_markers_enabled, false, 0, 0, 0,
                                 vqid + 1);
                    break;
                }
                OnBack(
                    [&] {
                        const auto cmd_address = reinterpret_cast<const void*>(header);
                        if (host_markers_enabled) {
                            rasterizer->ScopeMarkerBegin(
                                fmt::format("asc[{}]:{}:DispatchDirect", vqid, cmd_address));
                            rasterizer->DispatchDirect();
                            rasterizer->ScopeMarkerEnd();
                        } else {
                            rasterizer->DispatchDirect();
                        }
                    },
                    CpSplit::DrainReason::CallAscDispatch);
            }
            break;
        }
        case PM4ItOpcode::DispatchIndirect: {
            const auto* dispatch_indirect =
                reinterpret_cast<const PM4CmdDispatchIndirectMec*>(header);
            auto& cs_program = GetCsRegs();
            const auto ib_address = dispatch_indirect->Address<VAddr>();
            const auto size = sizeof(PM4CmdDispatchIndirect::GroupDimensions);
            if (DebugState.DumpingCurrentReg()) {
                DebugState.PushRegsDumpCompute(base_addr, reinterpret_cast<uintptr_t>(header),
                                               cs_program);
            }
            if (rasterizer && (cs_program.dispatch_initiator & 1)) {
                if (split_post_asc & CpSplit::PostAscDispatch) [[unlikely]] {
                    EmitDispatch(header, "DispatchIndirect", host_markers_enabled, true,
                                 ib_address, 0, u32(size), vqid + 1);
                    break;
                }
                OnBack(
                    [&] {
                        const auto cmd_address = reinterpret_cast<const void*>(header);
                        if (host_markers_enabled) {
                            rasterizer->ScopeMarkerBegin(
                                fmt::format("asc[{}]:{}:DispatchIndirect", vqid, cmd_address));
                            rasterizer->DispatchIndirect(ib_address, 0, size);
                            rasterizer->ScopeMarkerEnd();
                        } else {
                            rasterizer->DispatchIndirect(ib_address, 0, size);
                        }
                    },
                    CpSplit::DrainReason::CallAscDispatchIndirect);
            }
            break;
        }
        case PM4ItOpcode::WriteData: {
            const auto* write_data = reinterpret_cast<const PM4CmdWriteData*>(header);
            ASSERT(write_data->dst_sel.Value() == 2 || write_data->dst_sel.Value() == 5);
            const u32 data_size = (header->type3.count.Value() - 2) * 4;
            if (split_post_asc & CpSplit::PostAscWriteData) [[unlikely]] {
                // POST_ASC: ExecuteGfxPacket's WRITE_DATA body, which adds SplitHazard::
                // NoteWrite (off: POST_ASC is refused with it) and g_cp_wait.NoteWrite (this bit
                // is refused with CP_WAIT_STATS). Readers of the label (WAIT_REG_MEM, semaphore
                // waits, task done) drain first; the guest CPU sees it later, never earlier.
                RunAscPacket(header, acb,
                             reinterpret_cast<VAddr>(write_data->Address<const void*>()),
                             data_size, CpSplit::DrainReason::AscWriteDataGuard);
                break;
            }
            OnBack(
                [&] {
                    g_cp_wait.NoteWrite(write_data->Address<void*>());
                    VideoCore::ProtectFence();
                    if (!write_data->wr_one_addr.Value()) {
                        std::memcpy(write_data->Address<void*>(), write_data->data, data_size);
                    } else {
                        UNREACHABLE();
                    }
                },
                CpSplit::DrainReason::CallAscWriteData);
            break;
        }
        case PM4ItOpcode::MemSemaphore: {
            // Stays synchronous under POST_ASC (a Call waits for everything posted). Posting the
            // signal would keep stream order, and its readers drain first, but it is not
            // measured to matter yet (call_asc_sem says). SPLIT-TODO(run-ahead).
            const auto* mem_semaphore = reinterpret_cast<const PM4CmdMemSemaphore*>(header);
            if (mem_semaphore->IsSignaling()) {
                OnBack([&] { mem_semaphore->Signal(); }, CpSplit::DrainReason::CallAscSemaphore);
            } else {
                FrontDrain(CpSplit::DrainReason::MemSemaphore);
                while (!mem_semaphore->Signaled()) {
                    YIELD_ASC(vqid);
                }
                OnBack([&] { mem_semaphore->Decrement(); },
                       CpSplit::DrainReason::CallAscSemaphore);
            }
            break;
        }
        case PM4ItOpcode::WaitRegMem: {
            const auto* wait_reg_mem = reinterpret_cast<const PM4CmdWaitRegMem*>(header);
            ASSERT(wait_reg_mem->engine.Value() == PM4CmdWaitRegMem::Engine::Me);
            const auto wrm_start = Common::FaultMonitor::Now();
            const u64 wrm_ns0 = g_cp_wait.enabled ? CpWaitStats::Now() : 0;
            bool wrm_waited = false;
            FrontDrain(CpSplit::DrainReason::WaitRegMem);
            if (FenceAtGpu() == 2 && rasterizer && !wait_reg_mem->Test(regs.reg_array)) {
                OnBack([&] { rasterizer->FlushLazyFences(); }, CpSplit::DrainReason::WaitRegMem);
            }
            while (!wait_reg_mem->Test(regs.reg_array)) {
                wrm_waited = true;
                YIELD_ASC(vqid);
            }
            if (g_cp_wait.enabled && wrm_waited) {
                g_cp_wait.CountWait(false, wait_reg_mem->Address<u64*>(),
                                    CpWaitStats::Now() - wrm_ns0);
            }
            Common::FaultMonitor::CountWaitRegMem(Common::FaultMonitor::SinceNs(wrm_start));
            break;
        }
        case PM4ItOpcode::ReleaseMem: {
            const auto* release_mem = reinterpret_cast<const PM4CmdReleaseMem*>(header);
            if (split_post_fence && release_mem->data_sel != DataSelect::GdsMemStore)
                [[unlikely]] {
                ASSERT(queue.pipe_id <= 0xff);
                RunFence(header, acb, release_mem->Address<VAddr>(),
                         static_cast<u16>(CpSplit::AscReleaseMem | (queue.pipe_id << 8)),
                         CpSplit::DrainReason::CallAscRelease);
                break;
            }
            // Synchronous under POST_ASC alone: a fence + IRQ (and g_cp_wait's label set).
            OnBack([&] { ExecuteAscReleaseMem(release_mem, queue.pipe_id); },
                   CpSplit::DrainReason::CallAscRelease);
            break;
        }
        case PM4ItOpcode::EventWrite: {
            // const auto* event = reinterpret_cast<const PM4CmdEventWrite*>(header);
            break;
        }
        default:
            UNREACHABLE_MSG("Unknown PM4 type 3 opcode {:#x} with count {}",
                            static_cast<u32>(opcode), header->type3.NumWords());
        }

        if (!KeepsSubmitEpoch(opcode)) {
            BumpEpoch(opcode);
        }
        acb = NextPacket(acb, next_dw_off);

        if constexpr (!is_indirect) {
            *queue.read_addr += next_dw_off;
            *queue.read_addr %= queue.ring_size_dw;
        }
    }

    FIBER_EXIT;
}

Liverpool::CmdBuffer Liverpool::CopyCmdBuffers(std::span<const u32> dcb, std::span<const u32> ccb) {
    auto& queue = mapped_queues[GfxQueueId];
    ASSERT_MSG(queue.dcb_buffer.capacity() >= queue.dcb_buffer_offset + dcb.size(),
               "dcb copy buffer out of reserved space");
    ASSERT_MSG(queue.ccb_buffer.capacity() >= queue.ccb_buffer_offset + ccb.size(),
               "ccb copy buffer out of reserved space");

    queue.dcb_buffer.resize(
        std::max(queue.dcb_buffer.size(), queue.dcb_buffer_offset + dcb.size()));
    queue.ccb_buffer.resize(
        std::max(queue.ccb_buffer.size(), queue.ccb_buffer_offset + ccb.size()));

    const u32 prev_dcb_buffer_offset = queue.dcb_buffer_offset;
    const u32 prev_ccb_buffer_offset = queue.ccb_buffer_offset;
    if (!dcb.empty()) {
        std::memcpy(queue.dcb_buffer.data() + queue.dcb_buffer_offset, dcb.data(),
                    dcb.size_bytes());
        queue.dcb_buffer_offset += dcb.size();
        dcb = std::span<const u32>{queue.dcb_buffer.begin() + prev_dcb_buffer_offset,
                                   queue.dcb_buffer.begin() + queue.dcb_buffer_offset};
    }

    if (!ccb.empty()) {
        std::memcpy(queue.ccb_buffer.data() + queue.ccb_buffer_offset, ccb.data(),
                    ccb.size_bytes());
        queue.ccb_buffer_offset += ccb.size();
        ccb = std::span<const u32>{queue.ccb_buffer.begin() + prev_ccb_buffer_offset,
                                   queue.ccb_buffer.begin() + queue.ccb_buffer_offset};
    }

    return std::make_pair(dcb, ccb);
}

void Liverpool::SubmitGfx(std::span<const u32> dcb, std::span<const u32> ccb) {
    auto& queue = mapped_queues[GfxQueueId];

    if (EmulatorSettings.IsCopyGpuBuffers()) {
        std::tie(dcb, ccb) = CopyCmdBuffers(dcb, ccb);
    }

    Common::FaultMonitor::CountGuestSubmit(true);
    Common::SampleThreadPlacement("guest-submit-gfx", 4);
    if (AscRingLatency()) [[unlikely]] {
        auto& g = g_asc_ring;
        const u64 eop = g.last_eop_tsc.load(std::memory_order_relaxed);
        const u64 now = __rdtsc();
        if (eop != 0 && now > eop) {
            const u64 ticks = now - eop;
            const u64 us = ticks / 3500; // ~3.5 GHz TSC; the mean uses the measured ratio
            g.sg_n.fetch_add(1, std::memory_order_relaxed);
            g.sg_ticks.fetch_add(ticks, std::memory_order_relaxed);
            g.sg_hist[us < 100 ? 0 : us < 300 ? 1 : us < 1000 ? 2 : us < 3000 ? 3 : 4].fetch_add(
                1, std::memory_order_relaxed);
        }
    }
    auto task = ProcessGraphics(dcb, ccb);
    {
        std::scoped_lock lock{queue.m_access};
        queue.submits.emplace(task.handle);
    }

    std::scoped_lock lk{submit_mutex};
    ++num_submits;
    submit_cv.notify_one();
}

void Liverpool::SubmitAsc(u32 gnm_vqid, std::span<const u32> acb) {
    ASSERT_MSG(gnm_vqid > 0 && gnm_vqid < NumTotalQueues, "Invalid virtual ASC queue index");
    auto& queue = mapped_queues[gnm_vqid];

    const auto vqid = gnm_vqid - 1;
    Common::FaultMonitor::CountGuestSubmit(false);
    const auto& task = ProcessCompute(acb, vqid);
    {
        std::scoped_lock lock{queue.m_access};
        if (split_run_ahead && AscRingLatency()) [[unlikely]] {
            // Under m_access and before the push below, so stamps keep the queue's task order.
            std::scoped_lock ring_lock{g_asc_ring.submit_mutex};
            g_asc_ring.submits[gnm_vqid % AscRingQueues].push_back(
                g_asc_ring_clock.Snap(gnm_vqid));
        }
        // Before the push: the front may run and pop this task before this function returns.
        asc_outstanding.fetch_add(1, std::memory_order_relaxed);
        queue.submits.emplace(task.handle);
        if (AscStartStats()) [[unlikely]] {
            // Low bit: an earlier submission of this queue is still queued or running.
            asc_submit_ns[gnm_vqid].push_back((AscNowNs() & ~u64{1}) |
                                              (queue.submits.size() > 1 ? 1 : 0));
        }
    }

    std::scoped_lock lk{submit_mutex};
    num_mapped_queues = std::max(num_mapped_queues, gnm_vqid + 1);
    ++num_submits;
    asc_submit_seq.fetch_add(1, std::memory_order_release);
    submit_cv.notify_one();
}

} // namespace AmdGpu
