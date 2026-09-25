// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <thread>
#include <immintrin.h>

#include "common/logging/log.h"
#include "video_core/amdgpu/cp_split.h"

namespace AmdGpu::CpSplit {

namespace {
using Clock = std::chrono::steady_clock;

u64 NowNs() {
    return static_cast<u64>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch())
            .count());
}

/// How long either side spins before parking. At depth 0 a hand-over happens every ~1-2 us
/// (derived: ~1.75 us of CP time per draw/dispatch), so parking on every one (5-50 us per wake)
/// would be unusable. SHADPS4_CP_SPLIT_SPIN_US overrides. SPLIT-TODO: tune; measure P-core cost.
u64 SpinNs() {
    static const u64 ns = [] {
        const char* value = std::getenv("SHADPS4_CP_SPLIT_SPIN_US");
        return (value ? std::strtoull(value, nullptr, 10) : 50ull) * 1000ull;
    }();
    return ns;
}

constexpr u32 AlignUp(u32 value, u32 align) {
    return (value + align - 1) & ~(align - 1);
}
} // namespace

int Mode() {
    static const int mode = [] {
        const char* value = std::getenv("SHADPS4_CP_SPLIT");
        const int v = value ? std::atoi(value) : 0;
        // SPLIT-TODO(depth>0): n >= 2 = run-ahead depth n-1. Until then every n >= 1 is depth 0.
        return v > 0 ? 1 : 0;
    }();
    return mode;
}

bool Verify() {
    static const bool verify = [] {
        const char* value = std::getenv("SHADPS4_CP_SPLIT_VERIFY");
        return value && value[0] == '1';
    }();
    return verify;
}

Channel::Channel(u32 ring_bytes) {
    ASSERT_MSG(ring_bytes >= (64u << 10) && (ring_bytes & (ring_bytes - 1)) == 0,
               "CP split ring size must be a power of two >= 64 KB");
    capacity = ring_bytes;
    mask = ring_bytes - 1;
    // Over-allocate so the ring starts on a cache line.
    storage_owner = std::make_unique<u8[]>(std::size_t{ring_bytes} + 64);
    const auto base = reinterpret_cast<std::uintptr_t>(storage_owner.get());
    storage = reinterpret_cast<u8*>((base + 63) & ~std::uintptr_t{63});
    log_window_start_ns = NowNs();
}

Channel::~Channel() = default;

void* Channel::Reserve(RecordType type, u32 payload_bytes, u16 flags) {
    const u32 bytes = AlignUp(u32(sizeof(RecordHeader)) + payload_bytes, RecordAlign);
    ASSERT_MSG(bytes <= capacity / 2, "CP split record of {} bytes does not fit the ring", bytes);
    u64 pos = reserve_pos;
    const u32 offset = u32(pos & mask);
    if (offset + bytes > capacity) {
        // Records never wrap: fill the tail with a Wrap record and start again at offset 0.
        const u32 filler = capacity - offset;
        WaitForSpace(pos + filler);
        auto* wrap = reinterpret_cast<RecordHeader*>(storage + offset);
        *wrap = RecordHeader{RecordType::Wrap, 0, filler, 0};
        pos += filler;
        reserve_pos = pos;
    }
    WaitForSpace(pos + bytes);
    auto* header = reinterpret_cast<RecordHeader*>(storage + (pos & mask));
    *header = RecordHeader{type, flags, bytes, ++emitted};
    reserve_pos = pos + bytes;
    ++counters.records;
    counters.bytes += bytes;
    return header + 1;
}

void Channel::Commit(bool sync) {
    if (!sync) {
        return;
    }
    ++counters.syncs;
    Publish();
    WaitDone(emitted);
}

void Channel::Drain() {
    Publish();
    WaitDone(emitted);
}

void Channel::EmitRegWrite(u32 word_offset, const u32* data, u32 count) {
    ASSERT(u64{word_offset} + count <= Regs::NumRegs);
    auto* payload =
        Reserve<RegWritePayload>(RecordType::RegWrite, u32(sizeof(RegWritePayload)) + count * 4);
    payload->word_offset = word_offset;
    payload->count = count;
    std::memcpy(payload + 1, data, std::size_t{count} * 4);
    Commit(false);
}

void Channel::EmitClearState() {
    Reserve(RecordType::ClearState, 0);
    Commit(false);
}

void Channel::EmitExtent(u32 slot, u32 raw) {
    auto* payload = Reserve<ExtentPayload>(RecordType::Extent);
    payload->slot = slot;
    payload->raw = raw;
    Commit(false);
}

void Channel::EmitEpochBump(u32 opcode) {
    auto* payload = Reserve<EpochPayload>(RecordType::EpochBump);
    payload->opcode = opcode;
    payload->pad = 0;
    Commit(false);
}

void Channel::FrontExited() {
    Drain();
    front_exited.store(true, std::memory_order_release);
    WakeBack();
}

void Channel::Publish() {
    if (write_pos.load(std::memory_order_relaxed) == reserve_pos) {
        return;
    }
    // seq_cst store, then seq_cst load of back_parked: with the back's store of back_parked and
    // load of write_pos in WaitForWork, one of the two always sees the other (no lost wake-up).
    write_pos.store(reserve_pos, std::memory_order_seq_cst);
    WakeBackIfParked();
}

void Channel::WaitForSpace(u64 end) {
    if (end - cached_read <= capacity) {
        return;
    }
    ++counters.full_waits;
    for (;;) {
        cached_read = read_pos.load(std::memory_order_acquire);
        if (end - cached_read <= capacity) {
            return;
        }
        // The back frees space only by consuming what has been published.
        Publish();
        _mm_pause();
        std::this_thread::yield();
    }
}

void Channel::WaitDone(u64 seq) {
    if (done_seq.load(std::memory_order_acquire) >= seq) {
        return;
    }
    const u64 deadline = NowNs() + SpinNs();
    for (u32 spins = 1;; ++spins) {
        if (done_seq.load(std::memory_order_acquire) >= seq) {
            return;
        }
        if ((spins & 63) == 0 && NowNs() > deadline) {
            break;
        }
        _mm_pause();
    }
    ++counters.front_parks;
    for (;;) {
        front_parked.store(true, std::memory_order_seq_cst);
        const u64 current = done_seq.load(std::memory_order_seq_cst);
        if (current >= seq) {
            break;
        }
        done_seq.wait(current, std::memory_order_acquire);
    }
    front_parked.store(false, std::memory_order_relaxed);
}

const RecordHeader* Channel::Next() {
    for (;;) {
        if (read == cached_write) {
            cached_write = write_pos.load(std::memory_order_acquire);
            if (read == cached_write) {
                return nullptr;
            }
        }
        const auto* header = reinterpret_cast<const RecordHeader*>(storage + (read & mask));
        if (header->type != RecordType::Wrap) {
            return header;
        }
        read += header->size;
        read_pos.store(read, std::memory_order_release);
    }
}

void Channel::Done(const RecordHeader* record) {
    const u64 seq = record->seq;
    read += record->size;
    read_pos.store(read, std::memory_order_release);
    // seq_cst: pairs with the front's store of front_parked / load of done_seq in WaitDone.
    done_seq.store(seq, std::memory_order_seq_cst);
    if (front_parked.load(std::memory_order_seq_cst)) {
        done_seq.notify_one();
    }
}

bool Channel::BackHasWork(const std::atomic<u32>& num_commands) const {
    return write_pos.load(std::memory_order_seq_cst) != read ||
           num_commands.load(std::memory_order_seq_cst) != 0;
}

void Channel::WaitForWork(const std::atomic<u32>& num_commands, const std::stop_token& stoken) {
    const auto done = [&] {
        return BackHasWork(num_commands) || stoken.stop_requested() || FrontHasExited();
    };
    const u64 deadline = NowNs() + SpinNs();
    for (u32 spins = 1;; ++spins) {
        if (done()) {
            return;
        }
        if ((spins & 63) == 0 && NowNs() > deadline) {
            break;
        }
        _mm_pause();
    }
    const u32 ticket = doorbell.load(std::memory_order_acquire);
    back_parked.store(true, std::memory_order_seq_cst);
    if (!done()) {
        doorbell.wait(ticket, std::memory_order_acquire);
    }
    back_parked.store(false, std::memory_order_relaxed);
}

void Channel::WakeBackIfParked() {
    if (back_parked.load(std::memory_order_seq_cst)) {
        WakeBack();
    }
}

void Channel::WakeBack() {
    doorbell.fetch_add(1, std::memory_order_seq_cst);
    doorbell.notify_one();
}

void Channel::MaybeLog() {
    // SPLIT-TODO: per-record-type counts, RegWrite dwords/draw, flat_dw histogram, drains by
    // cause (inventory section 4.5, "counters every window").
    if ((counters.records & 0xffff) != 0) {
        return;
    }
    const u64 now = NowNs();
    if (now - log_window_start_ns < 10'000'000'000ull) {
        return;
    }
    LOG_INFO(Render, "CpSplit: window_s={:.1f} records={} syncs={} bytes={} full_waits={} "
             "front_parks={}",
             double(now - log_window_start_ns) / 1e9, counters.records, counters.syncs,
             counters.bytes, counters.full_waits, counters.front_parks);
    counters = {};
    log_window_start_ns = now;
}

} // namespace AmdGpu::CpSplit
