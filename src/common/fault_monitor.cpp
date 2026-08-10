// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <ctime>
#include <fstream>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include <fmt/format.h>

#include "common/fault_monitor.h"
#include "common/logging/log.h"
#include "common/path_util.h"
#include "common/thread.h"

namespace Common::FaultMonitor {

std::atomic<u64> guest_read_faults{0};
std::atomic<u64> guest_read_unhandled{0};
std::atomic<u64> guest_read_ns{0};
std::atomic<u64> guest_write_faults{0};
std::atomic<u64> guest_write_unhandled{0};
std::atomic<u64> guest_write_ns{0};
std::atomic<u64> jit_patch_faults{0};
std::atomic<u64> jit_patch_ns{0};
std::atomic<u64> presented_frames{0};
std::atomic<u64> range_fragments{0};
std::atomic<u64> range_adds{0};
std::atomic<u64> range_subtracts{0};
std::atomic<u64> gpu_submits{0};
std::atomic<u64> gpu_waits{0};
std::atomic<u64> gpu_wait_ns{0};
std::atomic<u64> download_calls{0};
std::atomic<u64> download_bytes{0};
std::atomic<u64> draw_count{0};
std::atomic<u64> draw_pipeline_ns{0};
std::atomic<u64> draw_bindres_ns{0};
std::atomic<u64> draw_beginrender_ns{0};
std::atomic<u64> draw_finddepth_ns{0};
std::atomic<u64> draw_bindbuf_ns{0};
std::atomic<u64> draw_bindtex_ns{0};
std::atomic<u64> bindbuf_find_ns{0};
std::atomic<u64> bindbuf_obtain_ns{0};
std::atomic<u64> obtain_query_ns{0};
std::atomic<u64> obtain_stream_ns{0};
std::atomic<u64> obtain_stream_calls{0};
std::atomic<u64> obtain_stream_bytes{0};
std::atomic<u64> obtain_sync_ns{0};
std::atomic<u64> obtain_sync_calls{0};
std::atomic<u64> upload_scan_ns{0};
std::atomic<u64> upload_copy_ns{0};
std::atomic<u64> upload_mark_ns{0};
std::atomic<u64> upload_written_calls{0};
std::atomic<u64> protect_write_calls{0};
std::atomic<u64> protect_write_ns{0};
std::atomic<u64> protect_read_calls{0};
std::atomic<u64> protect_read_ns{0};
std::atomic<u64> bindtex_findimage_ns{0};
std::atomic<u64> bindtex_view_ns{0};
std::atomic<u64> bindtex_sampler_ns{0};
std::atomic<u64> cp_wait_ns{0};
std::atomic<u64> cp_active_ns{0};
std::atomic<u64> dispatch_count{0};
std::atomic<u64> dispatch_ns{0};
std::atomic<u64> draw_pre_ns{0};
std::atomic<u64> draw_vtxbuf_ns{0};
std::atomic<u64> draw_descwrite_ns{0};
std::atomic<u64> draw_dynstate_ns{0};
std::atomic<u64> draw_schedbegin_ns{0};
std::atomic<u64> draw_emit_ns{0};
std::atomic<u64> stream_map_ns{0};
std::atomic<u64> stream_valid_ns{0};
std::atomic<u64> stream_copy_ns{0};
std::atomic<u64> stream_commit_ns{0};
std::atomic<u64> waitregmem_count{0};
std::atomic<u64> waitregmem_ns{0};
std::atomic<u64> volabel_count{0};
std::atomic<u64> volabel_ns{0};
std::atomic<u64> present_count{0};
std::atomic<u64> present_ns{0};
std::atomic<u64> flip_pending_total{0};
std::atomic<u64> present_acquire_ns{0};
std::atomic<u64> present_flush_ns{0};
std::atomic<u64> present_lock_ns{0};
std::atomic<u64> present_khr_ns{0};
std::atomic<u64> renderframe_queue_ns{0};
std::atomic<u64> renderframe_fence_ns{0};
std::atomic<u64> renderframe_count{0};

namespace {

// Short enough to see when the decay starts, long enough that a multi-hour run stays small:
// 240 lines per hour.
constexpr auto REPORT_INTERVAL = std::chrono::seconds(15);

std::mutex page_mutex;
std::unordered_map<u64, u64> read_fault_pages;

} // namespace

void RecordReadFaultPage(u64 addr) {
    std::scoped_lock lk(page_mutex);
    ++read_fault_pages[addr >> 12];
}

namespace {

/// Top read-faulting pages, plus how concentrated the distribution is. If a handful of pages
/// account for most faults, a targeted fix is possible; if it is spread thin, it is not.
std::string TopReadFaultPages(size_t top_n) {
    std::vector<std::pair<u64, u64>> pages;
    u64 total = 0;
    {
        std::scoped_lock lk(page_mutex);
        pages.assign(read_fault_pages.begin(), read_fault_pages.end());
    }
    for (const auto& [page, count] : pages) {
        total += count;
    }
    if (pages.empty() || total == 0) {
        return "no read faults";
    }
    const size_t n = std::min(top_n, pages.size());
    std::partial_sort(pages.begin(), pages.begin() + n, pages.end(),
                      [](const auto& a, const auto& b) { return a.second > b.second; });

    std::string out = fmt::format("{} distinct pages, ", pages.size());
    u64 top_sum = 0;
    for (size_t i = 0; i < n; i++) {
        top_sum += pages[i].second;
    }
    out += fmt::format("top {} = {:.1f}% |", n, 100.0 * top_sum / total);
    for (size_t i = 0; i < n; i++) {
        out += fmt::format(" {:#x}={:.1f}%", pages[i].first << 12,
                           100.0 * pages[i].second / static_cast<double>(total));
    }
    return out;
}

std::mutex stop_mutex;
std::condition_variable stop_cv;
bool stop_requested = false;

struct Sample {
    u64 read;
    u64 read_unhandled;
    u64 read_ns;
    u64 write;
    u64 write_unhandled;
    u64 write_ns;
    u64 jit;
    u64 jit_ns;
    u64 frames;
    u64 fragments;
    u64 adds;
    u64 subtracts;
    u64 submits;
    u64 waits;
    u64 wait_ns;
    u64 wrm_count;
    u64 wrm_ns;
    u64 vo_count;
    u64 vo_ns;
    u64 pres_count;
    u64 pres_ns;
    u64 pending;
    u64 acq_ns;
    u64 flush_ns;
    u64 lock_ns;
    u64 khr_ns;
    u64 rf_count;
    u64 rf_queue_ns;
    u64 rf_fence_ns;
    u64 dl_calls;
    u64 dl_bytes;
    u64 draws;
    u64 d_pipeline_ns;
    u64 d_bindres_ns;
    u64 d_begin_ns;
    u64 d_finddepth_ns;
    u64 d_bindbuf_ns;
    u64 d_bindtex_ns;
    u64 bb_find_ns;
    u64 bb_obtain_ns;
    u64 ob_query_ns;
    u64 ob_stream_ns;
    u64 ob_stream_calls;
    u64 ob_stream_bytes;
    u64 ob_sync_ns;
    u64 ob_sync_calls;
    u64 up_scan_ns;
    u64 up_copy_ns;
    u64 up_mark_ns;
    u64 up_written_calls;
    u64 prot_w_calls;
    u64 prot_w_ns;
    u64 prot_r_calls;
    u64 prot_r_ns;
    u64 bt_findimage_ns;
    u64 bt_view_ns;
    u64 bt_sampler_ns;
    u64 cp_wait_ns_v;
    u64 cp_active_ns_v;
    u64 disp_count;
    u64 disp_ns;
    u64 d_pre_ns;
    u64 d_vtxbuf_ns;
    u64 d_descwrite_ns;
    u64 d_dynstate_ns;
    u64 d_schedbegin_ns;
    u64 d_emit_ns;
    u64 sc_map_ns;
    u64 sc_valid_ns;
    u64 sc_copy_ns;
    u64 sc_commit_ns;
};

Sample Read() {
    return Sample{
        guest_read_faults.load(std::memory_order_relaxed),
        guest_read_unhandled.load(std::memory_order_relaxed),
        guest_read_ns.load(std::memory_order_relaxed),
        guest_write_faults.load(std::memory_order_relaxed),
        guest_write_unhandled.load(std::memory_order_relaxed),
        guest_write_ns.load(std::memory_order_relaxed),
        jit_patch_faults.load(std::memory_order_relaxed),
        jit_patch_ns.load(std::memory_order_relaxed),
        presented_frames.load(std::memory_order_relaxed),
        range_fragments.load(std::memory_order_relaxed),
        range_adds.load(std::memory_order_relaxed),
        range_subtracts.load(std::memory_order_relaxed),
        gpu_submits.load(std::memory_order_relaxed),
        gpu_waits.load(std::memory_order_relaxed),
        gpu_wait_ns.load(std::memory_order_relaxed),
        waitregmem_count.load(std::memory_order_relaxed),
        waitregmem_ns.load(std::memory_order_relaxed),
        volabel_count.load(std::memory_order_relaxed),
        volabel_ns.load(std::memory_order_relaxed),
        present_count.load(std::memory_order_relaxed),
        present_ns.load(std::memory_order_relaxed),
        flip_pending_total.load(std::memory_order_relaxed),
        present_acquire_ns.load(std::memory_order_relaxed),
        present_flush_ns.load(std::memory_order_relaxed),
        present_lock_ns.load(std::memory_order_relaxed),
        present_khr_ns.load(std::memory_order_relaxed),
        renderframe_count.load(std::memory_order_relaxed),
        renderframe_queue_ns.load(std::memory_order_relaxed),
        renderframe_fence_ns.load(std::memory_order_relaxed),
        download_calls.load(std::memory_order_relaxed),
        download_bytes.load(std::memory_order_relaxed),
        draw_count.load(std::memory_order_relaxed),
        draw_pipeline_ns.load(std::memory_order_relaxed),
        draw_bindres_ns.load(std::memory_order_relaxed),
        draw_beginrender_ns.load(std::memory_order_relaxed),
        draw_finddepth_ns.load(std::memory_order_relaxed),
        draw_bindbuf_ns.load(std::memory_order_relaxed),
        draw_bindtex_ns.load(std::memory_order_relaxed),
        bindbuf_find_ns.load(std::memory_order_relaxed),
        bindbuf_obtain_ns.load(std::memory_order_relaxed),
        obtain_query_ns.load(std::memory_order_relaxed),
        obtain_stream_ns.load(std::memory_order_relaxed),
        obtain_stream_calls.load(std::memory_order_relaxed),
        obtain_stream_bytes.load(std::memory_order_relaxed),
        obtain_sync_ns.load(std::memory_order_relaxed),
        obtain_sync_calls.load(std::memory_order_relaxed),
        upload_scan_ns.load(std::memory_order_relaxed),
        upload_copy_ns.load(std::memory_order_relaxed),
        upload_mark_ns.load(std::memory_order_relaxed),
        upload_written_calls.load(std::memory_order_relaxed),
        protect_write_calls.load(std::memory_order_relaxed),
        protect_write_ns.load(std::memory_order_relaxed),
        protect_read_calls.load(std::memory_order_relaxed),
        protect_read_ns.load(std::memory_order_relaxed),
        bindtex_findimage_ns.load(std::memory_order_relaxed),
        bindtex_view_ns.load(std::memory_order_relaxed),
        bindtex_sampler_ns.load(std::memory_order_relaxed),
        cp_wait_ns.load(std::memory_order_relaxed),
        cp_active_ns.load(std::memory_order_relaxed),
        dispatch_count.load(std::memory_order_relaxed),
        dispatch_ns.load(std::memory_order_relaxed),
        draw_pre_ns.load(std::memory_order_relaxed),
        draw_vtxbuf_ns.load(std::memory_order_relaxed),
        draw_descwrite_ns.load(std::memory_order_relaxed),
        draw_dynstate_ns.load(std::memory_order_relaxed),
        draw_schedbegin_ns.load(std::memory_order_relaxed),
        draw_emit_ns.load(std::memory_order_relaxed),
        stream_map_ns.load(std::memory_order_relaxed),
        stream_valid_ns.load(std::memory_order_relaxed),
        stream_copy_ns.load(std::memory_order_relaxed),
        stream_commit_ns.load(std::memory_order_relaxed),
    };
}

std::ofstream OpenCsv() {
    const auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &now);
#else
    localtime_r(&now, &tm);
#endif
    char name[64];
    std::strftime(name, sizeof(name), "fault_rate_%Y%m%d_%H%M%S.csv", &tm);

    const auto path = FS::GetUserPath(FS::PathType::LogDir) / name;
    std::ofstream csv(path);
    if (csv) {
        csv << "elapsed_s,fps,read_per_s,write_per_s,jit_per_s,"
               "faults_per_frame,read_us_each,write_us_each,handler_pct_of_wall,"
               "read_total,read_unhandled,write_total,write_unhandled,jit_total,frames_total,"
               "range_fragments,range_adds_per_s,range_subs_per_s,"
               "submits_per_frame,waits_per_frame,gpu_wait_ms_per_frame,"
               "waitregmem_per_frame,waitregmem_ms_per_frame,volabel_per_frame,"
               "volabel_ms_per_frame,present_ms_each,present_ms_per_frame,avg_flip_pending,"
               "acquire_ms,flush_ms,submitlock_ms,presentkhr_ms,"
               "rf_queue_ms,rf_fence_ms,download_per_frame,download_kb_per_frame,"
               "draws_per_frame,pipeline_us_each,bindres_us_each,beginrender_us_each,"
               "finddepth_us_each,draw_ms_per_frame,bindbuf_us_each,bindtex_us_each,"
               "bb_find_us_each,bb_obtain_us_each,"
               "ob_query_us_each,ob_stream_us_each,ob_sync_us_each,"
               "ob_stream_per_draw,ob_sync_per_draw,ob_stream_kb_per_draw,"
               "up_scan_us_each,up_copy_us_each,up_mark_us_each,up_written_per_draw,"
               "protw_per_draw,protw_us_each,protr_per_draw,protr_us_each,"
               "sc_map_us_each,sc_valid_us_each,sc_copy_us_each,sc_commit_us_each,"
               "bt_findimage_us_each,bt_view_us_each,bt_sampler_us_each,"
               "d_pre_us_each,d_vtxbuf_us_each,d_descwrite_us_each,d_dynstate_us_each,"
               "d_schedbegin_us_each,d_emit_us_each,"
               "cp_wait_ms_per_frame,cp_active_ms_per_frame,disp_per_frame,disp_us_each\n";
        csv.flush();
        LOG_INFO(Render, "FaultMonitor: rate CSV at {}", FS::PathToUTF8String(path));
    } else {
        LOG_WARNING(Render, "FaultMonitor: could not open {}", FS::PathToUTF8String(path));
    }
    return csv;
}

void ReportLoop() {
    Common::SetCurrentThreadName("shadPS4:FaultMonitor");

    std::ofstream csv = OpenCsv();
    const auto started = std::chrono::steady_clock::now();
    auto prev_time = started;
    Sample prev = Read();

    while (true) {
        std::unique_lock lk(stop_mutex);
        if (stop_cv.wait_for(lk, REPORT_INTERVAL, [] { return stop_requested; })) {
            break;
        }
        lk.unlock();

        const auto now = std::chrono::steady_clock::now();
        const Sample cur = Read();

        const double dt = std::chrono::duration<double>(now - prev_time).count();
        const double elapsed = std::chrono::duration<double>(now - started).count();
        if (dt <= 0.0) {
            continue;
        }

        const u64 d_read = cur.read - prev.read;
        const u64 d_write = cur.write - prev.write;
        const u64 d_jit = cur.jit - prev.jit;
        const u64 d_frames = cur.frames - prev.frames;
        const u64 d_read_ns = cur.read_ns - prev.read_ns;
        const u64 d_write_ns = cur.write_ns - prev.write_ns;
        const u64 d_jit_ns = cur.jit_ns - prev.jit_ns;
        const u64 d_adds = cur.adds - prev.adds;
        const u64 d_subs = cur.subtracts - prev.subtracts;
        const u64 d_submits = cur.submits - prev.submits;
        const u64 d_waits = cur.waits - prev.waits;
        const u64 d_wait_ns = cur.wait_ns - prev.wait_ns;

        // The decisive split: if submits/frame stays flat while gpu_wait_ms/frame grows, the GPU
        // side is getting slower rather than being asked to do more.
        const double submits_per_frame =
            d_frames ? static_cast<double>(d_submits) / static_cast<double>(d_frames) : 0.0;
        const double waits_per_frame =
            d_frames ? static_cast<double>(d_waits) / static_cast<double>(d_frames) : 0.0;
        const double wait_ms_per_frame =
            d_frames ? static_cast<double>(d_wait_ns) / static_cast<double>(d_frames) / 1e6 : 0.0;

        const u64 d_wrm = cur.wrm_count - prev.wrm_count;
        const u64 d_wrm_ns = cur.wrm_ns - prev.wrm_ns;
        const u64 d_vo = cur.vo_count - prev.vo_count;
        const u64 d_vo_ns = cur.vo_ns - prev.vo_ns;
        const double wrm_per_frame =
            d_frames ? static_cast<double>(d_wrm) / static_cast<double>(d_frames) : 0.0;
        const double wrm_ms_per_frame =
            d_frames ? static_cast<double>(d_wrm_ns) / static_cast<double>(d_frames) / 1e6 : 0.0;
        const double vo_per_frame =
            d_frames ? static_cast<double>(d_vo) / static_cast<double>(d_frames) : 0.0;
        const double vo_ms_per_frame =
            d_frames ? static_cast<double>(d_vo_ns) / static_cast<double>(d_frames) / 1e6 : 0.0;

        const u64 d_pres = cur.pres_count - prev.pres_count;
        const u64 d_pres_ns = cur.pres_ns - prev.pres_ns;
        const u64 d_pending = cur.pending - prev.pending;
        const double present_ms_each = d_pres ? static_cast<double>(d_pres_ns) / d_pres / 1e6 : 0.0;
        const double present_ms_per_frame =
            d_frames ? static_cast<double>(d_pres_ns) / static_cast<double>(d_frames) / 1e6 : 0.0;
        const double avg_pending =
            d_pres ? static_cast<double>(d_pending) / static_cast<double>(d_pres) : 0.0;

        const auto per_present = [&](u64 now, u64 before) {
            return d_pres ? static_cast<double>(now - before) / d_pres / 1e6 : 0.0;
        };
        const double acq_ms = per_present(cur.acq_ns, prev.acq_ns);
        const double flush_ms = per_present(cur.flush_ns, prev.flush_ns);
        const double lock_ms = per_present(cur.lock_ns, prev.lock_ns);
        const double khr_ms = per_present(cur.khr_ns, prev.khr_ns);

        const u64 d_rf = cur.rf_count - prev.rf_count;
        const auto per_rf = [&](u64 now, u64 before) {
            return d_rf ? static_cast<double>(now - before) / d_rf / 1e6 : 0.0;
        };
        const double rf_queue_ms = per_rf(cur.rf_queue_ns, prev.rf_queue_ns);
        const double rf_fence_ms = per_rf(cur.rf_fence_ns, prev.rf_fence_ns);

        // Readback downloads: one submit-and-fence-wait each, so this is the bulk of
        // submits_per_frame. The kB column is the cost side of widening them.
        const u64 d_dl = cur.dl_calls - prev.dl_calls;
        const u64 d_dl_bytes = cur.dl_bytes - prev.dl_bytes;
        const double dl_per_frame =
            d_frames ? static_cast<double>(d_dl) / static_cast<double>(d_frames) : 0.0;
        const double dl_kb_per_frame =
            d_frames ? static_cast<double>(d_dl_bytes) / static_cast<double>(d_frames) / 1024.0
                     : 0.0;

        // Per-draw CPU, measured rather than inferred from a stack scan. The *_us_each columns are
        // microseconds per draw call; draw_ms_per_frame is what all three cost per rendered frame.
        const u64 d_draws = cur.draws - prev.draws;
        const u64 d_pipe = cur.d_pipeline_ns - prev.d_pipeline_ns;
        const u64 d_bind = cur.d_bindres_ns - prev.d_bindres_ns;
        const u64 d_begin = cur.d_begin_ns - prev.d_begin_ns;
        const u64 d_fdepth = cur.d_finddepth_ns - prev.d_finddepth_ns;
        const auto per_draw_us = [&](u64 ns) {
            return d_draws ? static_cast<double>(ns) / static_cast<double>(d_draws) / 1000.0 : 0.0;
        };
        const double draws_per_frame =
            d_frames ? static_cast<double>(d_draws) / static_cast<double>(d_frames) : 0.0;
        const double pipeline_us = per_draw_us(d_pipe);
        const double bindres_us = per_draw_us(d_bind);
        const double beginrender_us = per_draw_us(d_begin);
        const double finddepth_us = per_draw_us(d_fdepth);
        const double bindbuf_us = per_draw_us(cur.d_bindbuf_ns - prev.d_bindbuf_ns);
        const double bindtex_us = per_draw_us(cur.d_bindtex_ns - prev.d_bindtex_ns);
        const double bb_find_us = per_draw_us(cur.bb_find_ns - prev.bb_find_ns);
        const double bb_obtain_us = per_draw_us(cur.bb_obtain_ns - prev.bb_obtain_ns);
        // All six are normalised per DRAW, not per call, so they add up against bb_obtain_us_each
        // directly. ob_stream_per_draw / ob_sync_per_draw say how the bindings split between the
        // two halves, which is what decides where a cut is worth making.
        const double ob_query_us = per_draw_us(cur.ob_query_ns - prev.ob_query_ns);
        const double ob_stream_us = per_draw_us(cur.ob_stream_ns - prev.ob_stream_ns);
        const double ob_sync_us = per_draw_us(cur.ob_sync_ns - prev.ob_sync_ns);
        const double ob_stream_per_draw =
            d_draws ? static_cast<double>(cur.ob_stream_calls - prev.ob_stream_calls) /
                          static_cast<double>(d_draws)
                    : 0.0;
        const double ob_sync_per_draw =
            d_draws ? static_cast<double>(cur.ob_sync_calls - prev.ob_sync_calls) /
                          static_cast<double>(d_draws)
                    : 0.0;
        const double ob_stream_kb_per_draw =
            d_draws ? static_cast<double>(cur.ob_stream_bytes - prev.ob_stream_bytes) /
                          static_cast<double>(d_draws) / 1024.0
                    : 0.0;
        // Phases of ForEachUploadRange, per draw, so they add against ob_sync_us_each.
        const double up_scan_us = per_draw_us(cur.up_scan_ns - prev.up_scan_ns);
        const double up_copy_us = per_draw_us(cur.up_copy_ns - prev.up_copy_ns);
        const double up_mark_us = per_draw_us(cur.up_mark_ns - prev.up_mark_ns);
        const double up_written_per_draw =
            d_draws ? static_cast<double>(cur.up_written_calls - prev.up_written_calls) /
                          static_cast<double>(d_draws)
                    : 0.0;
        // Protection changes are per draw for rate, but per CALL for cost - a VirtualProtect is
        // a syscall and its individual cost is the thing worth knowing.
        const u64 d_protw = cur.prot_w_calls - prev.prot_w_calls;
        const u64 d_protr = cur.prot_r_calls - prev.prot_r_calls;
        const double protw_per_draw =
            d_draws ? static_cast<double>(d_protw) / static_cast<double>(d_draws) : 0.0;
        const double protr_per_draw =
            d_draws ? static_cast<double>(d_protr) / static_cast<double>(d_draws) : 0.0;
        const double protw_us =
            d_protw ? static_cast<double>(cur.prot_w_ns - prev.prot_w_ns) / d_protw / 1000.0 : 0.0;
        const double protr_us =
            d_protr ? static_cast<double>(cur.prot_r_ns - prev.prot_r_ns) / d_protr / 1000.0 : 0.0;
        const double bt_findimage_us = per_draw_us(cur.bt_findimage_ns - prev.bt_findimage_ns);
        const double bt_view_us = per_draw_us(cur.bt_view_ns - prev.bt_view_ns);
        const double bt_sampler_us = per_draw_us(cur.bt_sampler_ns - prev.bt_sampler_ns);
        const double d_pre_us = per_draw_us(cur.d_pre_ns - prev.d_pre_ns);
        const double d_vtxbuf_us = per_draw_us(cur.d_vtxbuf_ns - prev.d_vtxbuf_ns);
        const double d_descwrite_us = per_draw_us(cur.d_descwrite_ns - prev.d_descwrite_ns);
        const double d_dynstate_us = per_draw_us(cur.d_dynstate_ns - prev.d_dynstate_ns);
        const double d_schedbegin_us = per_draw_us(cur.d_schedbegin_ns - prev.d_schedbegin_ns);
        const double d_emit_us = per_draw_us(cur.d_emit_ns - prev.d_emit_ns);
        // Per FRAME, not per draw: this is frame-level work that a per-draw column would hide.
        const auto per_frame_ms = [&](u64 ns) {
            return d_frames ? static_cast<double>(ns) / static_cast<double>(d_frames) / 1e6 : 0.0;
        };
        const double cp_wait_ms = per_frame_ms(cur.cp_wait_ns_v - prev.cp_wait_ns_v);
        const double cp_active_ms = per_frame_ms(cur.cp_active_ns_v - prev.cp_active_ns_v);
        const u64 d_disp = cur.disp_count - prev.disp_count;
        const double disp_per_frame =
            d_frames ? static_cast<double>(d_disp) / static_cast<double>(d_frames) : 0.0;
        const double disp_us =
            d_disp ? static_cast<double>(cur.disp_ns - prev.disp_ns) / d_disp / 1000.0 : 0.0;
        const double sc_map_us = per_draw_us(cur.sc_map_ns - prev.sc_map_ns);
        const double sc_valid_us = per_draw_us(cur.sc_valid_ns - prev.sc_valid_ns);
        const double sc_copy_us = per_draw_us(cur.sc_copy_ns - prev.sc_copy_ns);
        const double sc_commit_us = per_draw_us(cur.sc_commit_ns - prev.sc_commit_ns);
        const double draw_ms_per_frame =
            d_frames ? static_cast<double>(d_pipe + d_bind + d_begin) /
                           static_cast<double>(d_frames) / 1e6
                     : 0.0;

        const double read_rate = static_cast<double>(d_read) / dt;
        const double write_rate = static_cast<double>(d_write) / dt;
        const double jit_rate = static_cast<double>(d_jit) / dt;
        const double fps = static_cast<double>(d_frames) / dt;

        // Does the framerate fall because there are MORE faults per frame, or because each
        // fault costs MORE? These two columns separate them.
        const double per_frame =
            d_frames ? static_cast<double>(d_read + d_write) / static_cast<double>(d_frames) : 0.0;
        const double read_us = d_read ? static_cast<double>(d_read_ns) / d_read / 1000.0 : 0.0;
        const double write_us = d_write ? static_cast<double>(d_write_ns) / d_write / 1000.0 : 0.0;
        // Handler time as a share of one thread's wall time over the interval.
        const double handler_pct =
            static_cast<double>(d_read_ns + d_write_ns + d_jit_ns) / (dt * 1e9) * 100.0;

        LOG_INFO(Render,
                 "FaultRate: t={:.0f}s fps={:.1f} read={:.0f}/s write={:.0f}/s jit={:.0f}/s "
                 "| per_frame={:.0f} read={:.2f}us write={:.2f}us handler={:.1f}% of wall "
                 "| ranges frags={} add={:.0f}/s sub={:.0f}/s "
                 "| gpu submits/frame={:.1f} waits/frame={:.1f} wait={:.1f}ms/frame "
                 "| waitregmem {:.1f}/frame {:.1f}ms/frame volabel {:.1f}/frame {:.1f}ms/frame "
                 "| present {:.2f}ms each {:.1f}ms/frame pending={:.2f} "
                 "| acquire={:.2f}ms flush={:.2f}ms submitlock={:.2f}ms presentKHR={:.2f}ms "
                 "| renderframe queue={:.2f}ms fence={:.2f}ms "
                 "| downloads {:.1f}/frame {:.1f}kB/frame "
                 "| draws {:.0f}/frame pipeline={:.2f}us bindres={:.2f}us begin={:.2f}us "
                 "(finddepth={:.2f}us bindbuf={:.2f}us [find={:.2f} obtain={:.2f}] bindtex={:.2f}us) = {:.1f}ms/frame "
                 "| obtain query={:.2f}us stream={:.2f}us sync={:.2f}us "
                 "({:.2f} stream + {:.2f} sync per draw, {:.1f}kB/draw copied) "
                 "| upload scan={:.2f}us copy={:.2f}us mark={:.2f}us ({:.2f} written/draw) "
                 "| protect w={:.3f}/draw {:.2f}us each r={:.3f}/draw {:.2f}us each "
                 "| streamcopy map={:.2f}us valid={:.2f}us copy={:.2f}us commit={:.2f}us "
                 "| bindtex findimage={:.2f}us view={:.2f}us sampler={:.2f}us "
                 "| drawtail pre={:.2f} vtxbuf={:.2f} descwrite={:.2f} dynstate={:.2f} "
                 "schedbegin={:.2f} emit={:.2f}us "
                 "| CP wait={:.1f}ms/frame active={:.1f}ms/frame dispatch {:.0f}/frame {:.2f}us each "
                 "| totals read={} write={} jit={} frames={} unhandled(r/w)={}/{}",
                 elapsed, fps, read_rate, write_rate, jit_rate, per_frame, read_us, write_us,
                 handler_pct, cur.fragments, static_cast<double>(d_adds) / dt,
                 static_cast<double>(d_subs) / dt, submits_per_frame, waits_per_frame,
                 wait_ms_per_frame, wrm_per_frame, wrm_ms_per_frame, vo_per_frame, vo_ms_per_frame,
                 present_ms_each, present_ms_per_frame, avg_pending, acq_ms, flush_ms, lock_ms,
                 khr_ms, rf_queue_ms, rf_fence_ms, dl_per_frame, dl_kb_per_frame, draws_per_frame,
                 pipeline_us, bindres_us, beginrender_us, finddepth_us, bindbuf_us, bb_find_us,
                 bb_obtain_us, bindtex_us,
                 draw_ms_per_frame, ob_query_us, ob_stream_us, ob_sync_us, ob_stream_per_draw,
                 ob_sync_per_draw, ob_stream_kb_per_draw, up_scan_us, up_copy_us, up_mark_us,
                 up_written_per_draw, protw_per_draw, protw_us, protr_per_draw, protr_us, sc_map_us,
                 sc_valid_us, sc_copy_us, sc_commit_us, bt_findimage_us, bt_view_us, bt_sampler_us,
                 d_pre_us, d_vtxbuf_us, d_descwrite_us, d_dynstate_us, d_schedbegin_us, d_emit_us,
                 cp_wait_ms, cp_active_ms, disp_per_frame, disp_us,
                 cur.read,
                 cur.write, cur.jit, cur.frames, cur.read_unhandled, cur.write_unhandled);

        if (d_read > 0) {
            LOG_INFO(Render, "FaultPages: {}", TopReadFaultPages(6));
        }

        if (csv) {
            csv << fmt::format(
                "{:.1f},{:.2f},{:.1f},{:.1f},{:.1f},{:.1f},{:.4f},{:.4f},{:.2f},{},{},{},{},{},{},"
                "{},{:.1f},{:.1f},{:.2f},{:.2f},{:.3f},{:.2f},{:.3f},{:.2f},{:.3f},"
                "{:.3f},{:.3f},{:.2f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},"
                "{:.2f},{:.1f},{:.1f},{:.3f},{:.3f},{:.3f},{:.3f},{:.2f},{:.3f},{:.3f},{:.3f},{:.3f},"
                "{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},"
                "{:.3f},{:.3f},{:.3f},{:.3f},{:.4f},{:.3f},{:.4f},{:.3f},"
                "{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},"
                "{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},"
                "{:.2f},{:.2f},{:.1f},{:.3f}\n",
                elapsed, fps, read_rate, write_rate, jit_rate, per_frame, read_us, write_us,
                handler_pct, cur.read, cur.read_unhandled, cur.write, cur.write_unhandled, cur.jit,
                cur.frames, cur.fragments, static_cast<double>(d_adds) / dt,
                static_cast<double>(d_subs) / dt, submits_per_frame, waits_per_frame,
                wait_ms_per_frame, wrm_per_frame, wrm_ms_per_frame, vo_per_frame, vo_ms_per_frame,
                present_ms_each, present_ms_per_frame, avg_pending, acq_ms, flush_ms, lock_ms,
                khr_ms, rf_queue_ms, rf_fence_ms, dl_per_frame, dl_kb_per_frame, draws_per_frame,
                pipeline_us, bindres_us, beginrender_us, finddepth_us, draw_ms_per_frame, bindbuf_us,
                bindtex_us, bb_find_us, bb_obtain_us, ob_query_us, ob_stream_us, ob_sync_us,
                ob_stream_per_draw, ob_sync_per_draw, ob_stream_kb_per_draw, up_scan_us, up_copy_us,
                up_mark_us, up_written_per_draw, protw_per_draw, protw_us, protr_per_draw, protr_us,
                sc_map_us, sc_valid_us, sc_copy_us, sc_commit_us, bt_findimage_us, bt_view_us,
                bt_sampler_us, d_pre_us, d_vtxbuf_us, d_descwrite_us, d_dynstate_us,
                d_schedbegin_us, d_emit_us, cp_wait_ms, cp_active_ms, disp_per_frame, disp_us);
            csv.flush(); // a killed process must not lose the curve
        }

        prev = cur;
        prev_time = now;
    }
}

// Joined at static destruction, which requests the stop and wakes the thread immediately.
struct Reporter {
    Reporter() : thread(ReportLoop) {}
    ~Reporter() {
        {
            std::scoped_lock lk(stop_mutex);
            stop_requested = true;
        }
        stop_cv.notify_all();
        if (thread.joinable()) {
            thread.join();
        }
    }
    std::thread thread;
};

} // namespace

void Start() {
    static Reporter reporter;
    (void)reporter;
}

} // namespace Common::FaultMonitor
