// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// q8 (NATIVE-RENDERER sec. 11 step 1, v1): the worker-readable T# / S# -> bindless slot table.
// The CP back inserts every texture it resolved (BindlessResolve via A2ResolveMaterial) whose image
// is sampled-only; lean workers look whole materials up themselves; the back compares one table
// generation per stretch instead of validating every image. Anything that could make an entry
// wrong bumps the generation (a 'kill'): a change to a table image (layout / state serial, CPU or GPU
// dirty, overlap epoch, unregister, depth association), bindless set rotation, sampler GC, meta
// clear. Entries of an older generation are misses. Writers: the CP back only (inserts); kills from
// any thread (atomics). Readers: per-entry seqlock.
// SHADPS4_TEX_TABLE=1 on, =2 verify (the back resolves anyway and compares).
#pragma once

#include <array>
#include <atomic>
#include <cstdlib>
#include "common/types.h"

namespace VideoCore::TexTable {

inline int Mode() {
    static const int m = [] {
        const char* e = std::getenv("SHADPS4_TEX_TABLE");
        return e ? std::atoi(e) : 0;
    }();
    return m;
}

inline std::atomic<u64> g_gen{1};
inline std::atomic<u64> g_kills{0};
/// 0 worker materials looked up, 1 all hit, 2 published pre-resolved, 3 publish rejected (generation
/// moved), 4 back inserts (images), 5 verify compares, 6 verify BAD, 7 ineligible images at insert.
inline std::array<std::atomic<u64>, 8> g_c{};

inline u64 Gen() noexcept {
    return g_gen.load(std::memory_order_acquire);
}
inline void Kill() noexcept {
    g_gen.fetch_add(1, std::memory_order_acq_rel);
    g_kills.fetch_add(1, std::memory_order_relaxed);
}
/// A change to an image: a kill if the image is in the current table.
inline void NoteImage(const u64& image_tab_gen) noexcept {
    if (Mode() == 0) {
        return;
    }
    const u64 g = std::atomic_ref<u64>{const_cast<u64&>(image_tab_gen)}.load(
        std::memory_order_relaxed);
    if (g != 0 && g == g_gen.load(std::memory_order_relaxed)) {
        Kill();
    }
}

struct ImgEntry {
    std::atomic<u64> seq{0};
    u64 gen{};
    u32 dkey{};
    u32 slot{};
    u32 id{}; ///< VideoCore::ImageId index (bulk touch)
    std::array<u32, 8> words{};
};
struct SmpEntry {
    std::atomic<u64> seq{0};
    u64 gen{};
    u32 slot{};
    u64 lru{};
    std::array<u32, 4> words{};
};
constexpr u32 ImgSize = 4096, SmpSize = 1024;
inline std::array<ImgEntry, ImgSize> g_img{};
inline std::array<SmpEntry, SmpSize> g_smp{};

inline u32 ImgIndex(u32 dkey, const std::array<u32, 8>& w) noexcept {
    u64 h = (u64{dkey} | 1) * 0x9E3779B97F4A7C15ull;
    for (const u32 x : w) {
        h = (h ^ x) * 0x100000001B3ull;
    }
    return static_cast<u32>((h ^ (h >> 29)) & (ImgSize - 1));
}
inline u32 SmpIndex(const std::array<u32, 4>& w) noexcept {
    u64 h = 0xC2B2AE3D27D4EB4Full;
    for (const u32 x : w) {
        h = (h ^ x) * 0x100000001B3ull;
    }
    return static_cast<u32>((h ^ (h >> 29)) & (SmpSize - 1));
}

/// The CP back only.
inline void InsertImg(u32 dkey, const std::array<u32, 8>& w, u32 slot, u32 id, u64 gen) noexcept {
    auto& e = g_img[ImgIndex(dkey, w)];
    const u64 s = e.seq.load(std::memory_order_relaxed);
    e.seq.store(s + 1, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);
    e.gen = gen;
    e.dkey = dkey;
    e.slot = slot;
    e.id = id;
    e.words = w;
    e.seq.store(s + 2, std::memory_order_release);
}
inline void InsertSmp(const std::array<u32, 4>& w, u32 slot, u64 lru, u64 gen) noexcept {
    auto& e = g_smp[SmpIndex(w)];
    const u64 s = e.seq.load(std::memory_order_relaxed);
    e.seq.store(s + 1, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);
    e.gen = gen;
    e.slot = slot;
    e.lru = lru;
    e.words = w;
    e.seq.store(s + 2, std::memory_order_release);
}
/// Any thread. True + the slot on a hit of generation `gen`.
inline bool LookupImg(u32 dkey, const std::array<u32, 8>& w, u64 gen, u32& slot) noexcept {
    const auto& e = g_img[ImgIndex(dkey, w)];
    const u64 s1 = e.seq.load(std::memory_order_acquire);
    if (s1 & 1) {
        return false;
    }
    const u64 eg = e.gen;
    const u32 ek = e.dkey, es = e.slot;
    const std::array<u32, 8> ew = e.words;
    std::atomic_thread_fence(std::memory_order_acquire);
    if (e.seq.load(std::memory_order_relaxed) != s1 || eg != gen || ek != dkey || ew != w) {
        return false;
    }
    slot = es;
    return true;
}
inline bool LookupSmp(const std::array<u32, 4>& w, u64 gen, u32& slot) noexcept {
    const auto& e = g_smp[SmpIndex(w)];
    const u64 s1 = e.seq.load(std::memory_order_acquire);
    if (s1 & 1) {
        return false;
    }
    const u64 eg = e.gen;
    const u32 es = e.slot;
    const std::array<u32, 4> ew = e.words;
    std::atomic_thread_fence(std::memory_order_acquire);
    if (e.seq.load(std::memory_order_relaxed) != s1 || eg != gen || ew != w) {
        return false;
    }
    slot = es;
    return true;
}

} // namespace VideoCore::TexTable
