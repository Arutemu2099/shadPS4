// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <span>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>

namespace Common {

/**
 * A bounded, single-producer command queue with one recording worker.
 *
 * The queue owns a fixed pool of chunks. Small callable objects are placement-stored in those
 * chunks, so recording them does not allocate. A chunk is published when it is full or when Drain
 * is called. The worker executes published chunks in FIFO order and recycles a chunk only after all
 * of its callables have been invoked (or discarded) and destroyed. The sole producer owns the
 * unpublished current chunk; when that chunk has room, recording takes the producer-only fast path
 * without acquiring the queue mutex. Publishing, acquiring a replacement chunk, and draining
 * still use the mutex.
 *
 * Only one producer may call Record, RecordWithPayload, or Drain. Factories and callbacks must not
 * call back into this object. A factory runs after reservation but before publication, without the
 * queue mutex held; the producer-owned current chunk keeps its reserved payload private until the
 * returned callable has been constructed.
 *
 * RecordWithPayload reserves the callable and the requested extra storage in one unpublished
 * chunk. The factory receives an aligned byte span in that storage and must return an invocable
 * callable. The callable is responsible for any non-trivial objects it constructs in the byte
 * span; the queue treats that span as raw storage and keeps it alive until the callable is
 * destroyed.
 */
template <std::size_t ChunkCount = 10, std::size_t ChunkBytes = 64 * 1024>
class CommandRecordingWorker {
    static_assert(ChunkCount >= 2, "The command worker needs at least two chunks");
    static_assert(ChunkBytes > 0, "The command worker needs non-empty chunks");

    using InvokeFunction = void (*)(void*);
    using DestroyFunction = void (*)(void*) noexcept;

    struct CommandHeader {
        InvokeFunction invoke{};
        DestroyFunction destroy{};
        void* object{};
        std::size_t next_offset{};
    };

    static_assert(alignof(CommandHeader) <= alignof(std::max_align_t));
    static_assert(ChunkBytes >= sizeof(CommandHeader));

    enum class ChunkState : std::uint8_t {
        Free,
        Building,
        Queued,
        Executing,
        Discarding,
    };

    struct Chunk {
        alignas(std::max_align_t) std::array<std::byte, ChunkBytes> storage{};
        std::size_t used{};
        std::size_t first_offset{};
        std::size_t command_count{};
        ChunkState state{ChunkState::Free};
    };

    struct Layout {
        std::size_t header_offset{};
        std::size_t object_offset{};
        std::size_t payload_offset{};
        std::size_t end_offset{};
    };

    struct Reservation {
        Chunk* chunk{};
        std::size_t old_used{};
        std::size_t old_first_offset{};
        std::size_t old_command_count{};
        Layout layout{};
        CommandHeader* header{};
        void* object{};
        std::span<std::byte> payload{};
    };

public:
    static constexpr std::size_t DefaultChunkCount = ChunkCount;
    static constexpr std::size_t DefaultChunkBytes = ChunkBytes;

    struct Counters {
        std::uint64_t commands{};
        std::uint64_t chunks{};
        std::uint64_t bytes{};
        std::uint64_t worker_ns{};
        std::uint64_t drain_ns{};
        std::uint64_t space_wait_ns{};
    };

    explicit CommandRecordingWorker(const bool measure_timing = false)
        : measure_timing_{measure_timing}, worker_{&CommandRecordingWorker::WorkerLoop, this} {}

    ~CommandRecordingWorker() noexcept {
        try {
            Drain();
        } catch (...) {
            // Destruction must not throw. Drain has already discarded all queued captures before
            // reporting a callback exception.
        }
        {
            std::lock_guard lock{mutex_};
            stop_requested_ = true;
        }
        work_cv_.notify_all();
        if (worker_.joinable()) {
            worker_.join();
        }
    }

    CommandRecordingWorker(const CommandRecordingWorker&) = delete;
    CommandRecordingWorker& operator=(const CommandRecordingWorker&) = delete;
    CommandRecordingWorker(CommandRecordingWorker&&) = delete;
    CommandRecordingWorker& operator=(CommandRecordingWorker&&) = delete;

    template <typename F>
        requires std::is_object_v<std::remove_cvref_t<F>>
    void Record(F&& function) {
        using Callable = std::remove_cvref_t<F>;
        static_assert(alignof(Callable) <= alignof(std::max_align_t),
                      "Command callable alignment exceeds max_align_t");
        static_assert(std::is_invocable_v<Callable&>, "Command callable must be invocable");
        RecordImpl<Callable>(
            [&](void* object) {
                std::construct_at(reinterpret_cast<Callable*>(object), std::forward<F>(function));
            },
            sizeof(Callable), alignof(Callable), 0);
    }

    template <typename Factory>
        requires std::is_object_v<std::remove_reference_t<Factory>>
    void RecordWithPayload(const std::size_t payload_bytes, Factory&& factory) {
        using FactoryType = std::remove_reference_t<Factory>;
        using Result = std::invoke_result_t<FactoryType&, std::span<std::byte>>;
        using Callable = std::remove_cvref_t<Result>;
        static_assert(!std::is_void_v<Result>, "Payload factory must return a callable");
        static_assert(std::is_object_v<Callable>, "Payload factory must return an object");
        static_assert(alignof(Callable) <= alignof(std::max_align_t),
                      "Command callable alignment exceeds max_align_t");
        static_assert(std::is_invocable_v<Callable&>, "Factory result must be invocable");

        RecordImpl<Callable>(
            [&](void* object, std::span<std::byte> payload) {
                // Keep the result's lifetime through placement construction. This also supports
                // factories returning an lvalue callable, although returning by value is preferred.
                decltype(auto) result = std::invoke(factory, payload);
                if constexpr (std::is_lvalue_reference_v<decltype(result)>) {
                    std::construct_at(reinterpret_cast<Callable*>(object), result);
                } else {
                    std::construct_at(reinterpret_cast<Callable*>(object), std::move(result));
                }
            },
            sizeof(Callable), alignof(Callable), payload_bytes);
    }

    /// Publishes the producer's partial chunk and waits for every prior command to finish.
    /// Rethrows the first callback exception after all commands through this drain point have been
    /// invoked or discarded and destroyed.
    void Drain() {
        const auto drain_start = measure_timing_ ? Clock::now() : Clock::time_point{};
        const auto finish_drain_timing = [this, drain_start] {
            if (measure_timing_) {
                drain_ns_ += ToNanoseconds(Clock::now() - drain_start);
            }
        };
        Chunk* discard_current = nullptr;
        std::uint64_t target{};
        {
            std::unique_lock lock{mutex_};
            if (current_ != nullptr) {
                if (failure_) {
                    discard_current = current_;
                    current_ = nullptr;
                    discard_current->state = ChunkState::Discarding;
                } else if (current_->command_count != 0) {
                    PublishCurrentLocked();
                } else {
                    ReleaseEmptyCurrentLocked();
                }
            }
            target = accepted_commands_;
        }

        if (discard_current != nullptr) {
            DestroyChunkContents(*discard_current);
            std::lock_guard lock{mutex_};
            ReleaseDiscardedChunkLocked(*discard_current);
        }

        std::exception_ptr error;
        {
            std::unique_lock lock{mutex_};
            done_cv_.wait(lock, [this, target] { return completed_commands_ >= target; });
            error = failure_;
        }
        if (error) {
            finish_drain_timing();
            std::rethrow_exception(error);
        }
        finish_drain_timing();
    }

    /// Returns cumulative work counters. Only the sole producer may call this, after Drain.
    /// Timing counters written by the worker are synchronized by the queue mutex.
    [[nodiscard]] Counters GetCounters() const noexcept {
        std::lock_guard lock{mutex_};
        return Counters{accepted_commands_, published_chunks_, reserved_bytes_,
                        worker_ns_,         drain_ns_,         space_wait_ns_};
    }

private:
    using Clock = std::chrono::steady_clock;

    static std::uint64_t ToNanoseconds(const Clock::duration duration) noexcept {
        return static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count());
    }

    template <typename Callable, typename Construct>
    void RecordImpl(Construct&& construct, const std::size_t object_bytes,
                    const std::size_t object_alignment, const std::size_t payload_bytes) {
        if (TryRecordFast<Callable>(construct, object_bytes, object_alignment, payload_bytes)) {
            return;
        }

        std::unique_lock lock{mutex_};
        CheckAcceptingLocked();
        Reservation reservation =
            ReserveLocked(lock, object_bytes, object_alignment, payload_bytes);
        lock.unlock();
        try {
            if constexpr (std::is_invocable_v<Construct&, void*, std::span<std::byte>>) {
                std::invoke(construct, reservation.object, reservation.payload);
            } else {
                std::invoke(construct, reservation.object);
            }
        } catch (...) {
            RollbackReservation(reservation);
            throw;
        }

        std::construct_at(reservation.header,
                          CommandHeader{&Invoke<Callable>, &Destroy<Callable>, reservation.object,
                                        reservation.layout.end_offset});
        CommitReservation(reservation);
        if (reservation.chunk->used == ChunkBytes) {
            std::lock_guard publish_lock{mutex_};
            // A callback may fail while the producer is constructing this command. If that
            // happened before this lock was acquired, leave the raced command in the producer-owned
            // chunk so Drain can destroy it after the worker's failure is observed.
            if (!failure_) {
                assert(current_ == reservation.chunk);
                PublishCurrentLocked();
            }
        }
    }

    template <typename Callable, typename Construct>
    [[nodiscard]] bool TryRecordFast(Construct& construct, const std::size_t object_bytes,
                                     const std::size_t object_alignment,
                                     const std::size_t payload_bytes) {
        // The failure flag is published by the worker while holding mutex_. A false result does
        // not reserve any state that the worker can observe; a raced command remains in current_
        // for Drain to discard if failure is published before its full-chunk handoff.
        if (failure_flag_.load(std::memory_order_acquire)) {
            return false;
        }

        // Check the empty layout first so an oversized command never publishes older work before
        // throwing. This also means the fast path has the same error behavior as ReserveLocked.
        if (!TryLayout(0, object_bytes, object_alignment, payload_bytes)) {
            throw std::length_error("command recording worker command exceeds chunk size");
        }

        Chunk* const chunk = current_;
        if (chunk == nullptr) {
            return false;
        }
        const auto layout = TryLayout(chunk->used, object_bytes, object_alignment, payload_bytes);
        if (!layout) {
            return false;
        }

        const std::size_t old_used = chunk->used;
        const std::size_t old_first_offset = chunk->first_offset;
        const std::size_t old_command_count = chunk->command_count;
        chunk->used = layout->end_offset;
        Reservation reservation{
            .chunk = chunk,
            .old_used = old_used,
            .old_first_offset = old_first_offset,
            .old_command_count = old_command_count,
            .layout = *layout,
            .header =
                reinterpret_cast<CommandHeader*>(chunk->storage.data() + layout->header_offset),
            .object = chunk->storage.data() + layout->object_offset,
            .payload =
                std::span<std::byte>{chunk->storage.data() + layout->payload_offset, payload_bytes},
        };
        try {
            if constexpr (std::is_invocable_v<Construct&, void*, std::span<std::byte>>) {
                std::invoke(construct, reservation.object, reservation.payload);
            } else {
                std::invoke(construct, reservation.object);
            }
        } catch (...) {
            RollbackReservation(reservation);
            throw;
        }

        std::construct_at(reservation.header,
                          CommandHeader{&Invoke<Callable>, &Destroy<Callable>, reservation.object,
                                        reservation.layout.end_offset});
        CommitReservation(reservation);
        if (reservation.chunk->used == ChunkBytes) {
            std::lock_guard publish_lock{mutex_};
            if (!failure_) {
                assert(current_ == reservation.chunk);
                PublishCurrentLocked();
            }
        }
        return true;
    }

    void CommitReservation(const Reservation& reservation) {
        reservation.chunk->command_count = reservation.old_command_count + 1;
        if (reservation.old_command_count == 0) {
            reservation.chunk->first_offset = reservation.layout.header_offset;
        }
        ++accepted_commands_;
        reserved_bytes_ += reservation.layout.end_offset - reservation.old_used;
    }

    template <typename Callable>
    static void Invoke(void* object) {
        std::invoke(*std::launder(reinterpret_cast<Callable*>(object)));
    }

    template <typename Callable>
    static void Destroy(void* object) noexcept {
        std::destroy_at(std::launder(reinterpret_cast<Callable*>(object)));
    }

    static constexpr std::size_t AlignUp(const std::size_t value, const std::size_t alignment) {
        const std::size_t mask = alignment - 1;
        if (value > (static_cast<std::size_t>(-1) - mask)) {
            throw std::length_error("command recording worker layout overflow");
        }
        return (value + mask) & ~mask;
    }

    [[nodiscard]] static std::optional<Layout> TryLayout(const std::size_t begin,
                                                         const std::size_t object_bytes,
                                                         const std::size_t object_alignment,
                                                         const std::size_t payload_bytes) {
        const std::size_t header_offset = AlignUp(begin, alignof(CommandHeader));
        if (header_offset > (static_cast<std::size_t>(-1) - sizeof(CommandHeader))) {
            throw std::length_error("command recording worker header overflow");
        }
        const std::size_t object_offset =
            AlignUp(header_offset + sizeof(CommandHeader), object_alignment);
        if (object_offset > (static_cast<std::size_t>(-1) - object_bytes)) {
            throw std::length_error("command recording worker callable overflow");
        }
        const std::size_t payload_offset =
            AlignUp(object_offset + object_bytes, alignof(std::max_align_t));
        if (payload_offset > (static_cast<std::size_t>(-1) - payload_bytes)) {
            throw std::length_error("command recording worker payload overflow");
        }
        const std::size_t raw_end_offset = payload_offset + payload_bytes;
        if (raw_end_offset > (static_cast<std::size_t>(-1) - alignof(CommandHeader) + 1)) {
            throw std::length_error("command recording worker end overflow");
        }
        // Traversal starts at the next command header, so include the alignment padding in the
        // reservation. Without this, a non-aligned payload leaves the next header in the gap.
        const std::size_t end_offset = AlignUp(raw_end_offset, alignof(CommandHeader));
        if (end_offset > ChunkBytes) {
            return std::nullopt;
        }
        return Layout{header_offset, object_offset, payload_offset, end_offset};
    }

    void CheckAcceptingLocked() {
        if (failure_) {
            std::rethrow_exception(failure_);
        }
        if (stop_requested_) {
            throw std::runtime_error("command recording worker is stopped");
        }
    }

    [[nodiscard]] bool HasFreeChunkLocked() const {
        for (const auto& chunk : chunks_) {
            if (chunk.state == ChunkState::Free) {
                return true;
            }
        }
        return false;
    }

    Chunk* AcquireCurrentLocked(std::unique_lock<std::mutex>& lock) {
        while (current_ == nullptr) {
            const auto can_acquire = [this] {
                return stop_requested_ || failure_ || HasFreeChunkLocked();
            };
            if (measure_timing_ && !stop_requested_ && !failure_ && !HasFreeChunkLocked()) {
                const auto wait_start = Clock::now();
                space_cv_.wait(lock, can_acquire);
                space_wait_ns_ += ToNanoseconds(Clock::now() - wait_start);
            } else {
                space_cv_.wait(lock, can_acquire);
            }
            CheckAcceptingLocked();
            for (auto& chunk : chunks_) {
                if (chunk.state == ChunkState::Free) {
                    chunk.state = ChunkState::Building;
                    current_ = &chunk;
                    break;
                }
            }
        }
        return current_;
    }

    [[nodiscard]] Reservation ReserveLocked(std::unique_lock<std::mutex>& lock,
                                            const std::size_t object_bytes,
                                            const std::size_t object_alignment,
                                            const std::size_t payload_bytes) {
        // Validate against an empty chunk before publishing the current one. An oversized command
        // therefore cannot unexpectedly publish older work before throwing.
        const auto empty_layout = TryLayout(0, object_bytes, object_alignment, payload_bytes);
        if (!empty_layout) {
            throw std::length_error("command recording worker command exceeds chunk size");
        }

        Chunk* chunk = AcquireCurrentLocked(lock);
        auto layout = TryLayout(chunk->used, object_bytes, object_alignment, payload_bytes);
        if (!layout) {
            assert(chunk->command_count != 0);
            PublishCurrentLocked();
            chunk = AcquireCurrentLocked(lock);
            layout = TryLayout(chunk->used, object_bytes, object_alignment, payload_bytes);
            assert(layout.has_value());
        }

        const std::size_t old_used = chunk->used;
        const std::size_t old_first_offset = chunk->first_offset;
        const std::size_t old_command_count = chunk->command_count;
        chunk->used = layout->end_offset;
        return Reservation{
            .chunk = chunk,
            .old_used = old_used,
            .old_first_offset = old_first_offset,
            .old_command_count = old_command_count,
            .layout = *layout,
            .header =
                reinterpret_cast<CommandHeader*>(chunk->storage.data() + layout->header_offset),
            .object = chunk->storage.data() + layout->object_offset,
            .payload =
                std::span<std::byte>{chunk->storage.data() + layout->payload_offset, payload_bytes},
        };
    }

    void RollbackReservation(const Reservation& reservation) {
        assert(current_ == reservation.chunk);
        assert(reservation.chunk->state == ChunkState::Building);
        reservation.chunk->used = reservation.old_used;
        reservation.chunk->first_offset = reservation.old_first_offset;
        reservation.chunk->command_count = reservation.old_command_count;
    }

    void PublishCurrentLocked() {
        assert(current_ != nullptr);
        assert(current_->command_count != 0);
        assert(published_size_ < ChunkCount);
        current_->state = ChunkState::Queued;
        published_queue_[(published_head_ + published_size_) % ChunkCount] = current_;
        ++published_size_;
        ++published_chunks_;
        current_ = nullptr;
        work_cv_.notify_one();
    }

    void ReleaseEmptyCurrentLocked() {
        assert(current_ != nullptr);
        assert(current_->command_count == 0);
        current_->used = 0;
        current_->first_offset = 0;
        current_->state = ChunkState::Free;
        current_ = nullptr;
        space_cv_.notify_one();
    }

    [[nodiscard]] Chunk* PopPublishedLocked() {
        assert(published_size_ != 0);
        Chunk* chunk = published_queue_[published_head_];
        published_head_ = (published_head_ + 1) % ChunkCount;
        --published_size_;
        chunk->state = ChunkState::Executing;
        return chunk;
    }

    static void DestroyChunkContents(Chunk& chunk) noexcept {
        std::size_t offset = chunk.first_offset;
        for (std::size_t index = 0; index < chunk.command_count; ++index) {
            auto* const header =
                std::launder(reinterpret_cast<CommandHeader*>(chunk.storage.data() + offset));
            const std::size_t next_offset = header->next_offset;
            header->destroy(header->object);
            offset = next_offset;
        }
        assert(offset == chunk.used);
    }

    void ReleaseCompletedChunkLocked(Chunk& chunk) {
        assert(chunk.state == ChunkState::Executing);
        completed_commands_ += chunk.command_count;
        chunk.used = 0;
        chunk.first_offset = 0;
        chunk.command_count = 0;
        chunk.state = ChunkState::Free;
        done_cv_.notify_all();
        space_cv_.notify_one();
    }

    void ReleaseDiscardedChunkLocked(Chunk& chunk) {
        assert(chunk.state == ChunkState::Discarding);
        completed_commands_ += chunk.command_count;
        chunk.used = 0;
        chunk.first_offset = 0;
        chunk.command_count = 0;
        chunk.state = ChunkState::Free;
        done_cv_.notify_all();
        space_cv_.notify_one();
    }

    void WorkerLoop() {
        for (;;) {
            Chunk* chunk{};
            {
                std::unique_lock lock{mutex_};
                work_cv_.wait(lock, [this] { return stop_requested_ || published_size_ != 0; });
                if (published_size_ == 0) {
                    if (stop_requested_) {
                        return;
                    }
                    continue;
                }
                chunk = PopPublishedLocked();
            }

            const auto worker_start = measure_timing_ ? Clock::now() : Clock::time_point{};
            std::exception_ptr callback_error;
            std::size_t offset = chunk->first_offset;
            for (std::size_t index = 0; index < chunk->command_count; ++index) {
                auto* const header =
                    std::launder(reinterpret_cast<CommandHeader*>(chunk->storage.data() + offset));
                const std::size_t next_offset = header->next_offset;
                if (!callback_error) {
                    try {
                        header->invoke(header->object);
                    } catch (...) {
                        callback_error = std::current_exception();
                    }
                }
                header->destroy(header->object);
                offset = next_offset;
            }
            assert(offset == chunk->used);
            const std::uint64_t chunk_ns =
                measure_timing_ ? ToNanoseconds(Clock::now() - worker_start) : 0;

            if (!callback_error) {
                std::lock_guard lock{mutex_};
                worker_ns_ += chunk_ns;
                ReleaseCompletedChunkLocked(*chunk);
                continue;
            }

            std::array<Chunk*, ChunkCount> discarded{};
            std::size_t discarded_count = 0;
            {
                std::lock_guard lock{mutex_};
                failure_ = callback_error;
                failure_flag_.store(true, std::memory_order_release);
                worker_ns_ += chunk_ns;
                chunk->state = ChunkState::Discarding;
                completed_commands_ += chunk->command_count;
                chunk->used = 0;
                chunk->first_offset = 0;
                chunk->command_count = 0;
                chunk->state = ChunkState::Free;
                while (published_size_ != 0) {
                    auto* const queued = PopPublishedLocked();
                    queued->state = ChunkState::Discarding;
                    discarded[discarded_count++] = queued;
                }
                done_cv_.notify_all();
                space_cv_.notify_all();
            }

            // Destructors can execute arbitrary user code; keep them outside the queue mutex just
            // like callbacks.
            for (std::size_t index = 0; index < discarded_count; ++index) {
                Chunk& queued = *discarded[index];
                const auto discard_start = measure_timing_ ? Clock::now() : Clock::time_point{};
                DestroyChunkContents(queued);
                const std::uint64_t discard_ns =
                    measure_timing_ ? ToNanoseconds(Clock::now() - discard_start) : 0;
                std::lock_guard lock{mutex_};
                worker_ns_ += discard_ns;
                ReleaseDiscardedChunkLocked(queued);
            }
            done_cv_.notify_all();
            return;
        }
    }

    mutable std::mutex mutex_;
    std::condition_variable work_cv_;
    std::condition_variable done_cv_;
    std::condition_variable space_cv_;
    std::array<Chunk, ChunkCount> chunks_{};
    std::array<Chunk*, ChunkCount> published_queue_{};
    Chunk* current_{};
    std::size_t published_head_{};
    std::size_t published_size_{};
    std::uint64_t accepted_commands_{};
    std::uint64_t completed_commands_{};
    std::uint64_t published_chunks_{};
    std::uint64_t reserved_bytes_{};
    std::exception_ptr failure_{};
    std::atomic<bool> failure_flag_{};
    bool stop_requested_{};
    const bool measure_timing_;
    std::uint64_t worker_ns_{};
    std::uint64_t drain_ns_{};
    std::uint64_t space_wait_ns_{};
    std::thread worker_;
};

} // namespace Common
