// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/assert.h"
#include <atomic>
#include <chrono>
#include "common/logging/log.h"
#include "common/readback_offload.h"
#include "core/libraries/error_codes.h"
#include "core/libraries/libs.h"
#include "libc_internal_memory.h"

namespace Libraries::LibcInternal {

void* PS4_SYSV_ABI internal_memset(void* s, int c, size_t n) {
    if (Common::ReadbackOffload::LatencyEnabled() && n >= 4096) [[unlikely]] {
        // Diagnostic: time of large guest memsets (their write faults included) and a thread
        // flag that tags the readback jobs they raise (batch mix class "memset").
        static std::atomic<u64> calls{0}, bytes{0}, ns{0}, max_ns{0};
        const auto t0 = std::chrono::steady_clock::now();
        Common::ReadbackOffload::t_in_hle_memset = true;
        void* result = std::memset(s, c, n);
        Common::ReadbackOffload::t_in_hle_memset = false;
        const u64 d = static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                           std::chrono::steady_clock::now() - t0)
                                           .count());
        bytes.fetch_add(n, std::memory_order_relaxed);
        ns.fetch_add(d, std::memory_order_relaxed);
        u64 m = max_ns.load(std::memory_order_relaxed);
        while (d > m && !max_ns.compare_exchange_weak(m, d, std::memory_order_relaxed)) {
        }
        const u64 k = calls.fetch_add(1, std::memory_order_relaxed) + 1;
        if ((k & 0x3fff) == 0) {
            LOG_INFO(Lib_LibcInternal,
                     "HleMemsetLarge: calls={} mean_bytes={} total_ms={:.1f} mean_us={:.2f} "
                     "max_us={:.1f} (cumulative; n >= 4096)",
                     k, bytes.load() / k, double(ns.load()) / 1e6, double(ns.load()) / 1e3 / k,
                     double(max_ns.load()) / 1e3);
        }
        return result;
    }
    return std::memset(s, c, n);
}

void* PS4_SYSV_ABI internal_memcpy(void* dest, const void* src, size_t n) {
    return std::memcpy(dest, src, n);
}

s32 PS4_SYSV_ABI internal_memcpy_s(void* dest, size_t destsz, const void* src, size_t count) {
#ifdef _WIN64
    return memcpy_s(dest, destsz, src, count);
#else
    std::memcpy(dest, src, count);
    return 0; // ALL OK
#endif
}

s32 PS4_SYSV_ABI internal_memcmp(const void* s1, const void* s2, size_t n) {
    return std::memcmp(s1, s2, n);
}

void RegisterlibSceLibcInternalMemory(Core::Loader::SymbolsResolver* sym) {

    LIB_FUNCTION("NFLs+dRJGNg", "libSceLibcInternal", 1, "libSceLibcInternal", internal_memcpy_s);
    LIB_FUNCTION("Q3VBxCXhUHs", "libSceLibcInternal", 1, "libSceLibcInternal", internal_memcpy);
    LIB_FUNCTION("8zTFvBIAIN8", "libSceLibcInternal", 1, "libSceLibcInternal", internal_memset);
    LIB_FUNCTION("DfivPArhucg", "libSceLibcInternal", 1, "libSceLibcInternal", internal_memcmp);
}

} // namespace Libraries::LibcInternal
