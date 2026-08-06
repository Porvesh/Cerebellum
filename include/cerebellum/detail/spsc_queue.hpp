// Bounded wait-free SPSC ring — adapted from the queue you pasted, which came
// from a different codebase, so it lives under detail/ rather than being vendored
// as somebody else's dependency.
//
// Two changes from the original: the cache-line grouping pairs each side's
// atomic with the cache it actually touches, and size() exists because the
// refresh trigger needs a count and not just empty() (§4.4).
//
// In Cerebellum this carries slot indices, never actions — chunk ownership moves
// through here, chunk contents do not. See chunk_queue.hpp for why.

#pragma once

#include <atomic>
#include <cstddef>
#include <memory>
#include <utility>

namespace cerebellum::detail {

// std::hardware_destructive_interference_size needs libstdc++ 12; this machine
// links against 11 (.clangd).
inline constexpr std::size_t kCacheLineSize = 64;

inline constexpr std::size_t next_pow2(std::size_t n) {
    std::size_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

// Exactly one thread may call push(), exactly one other may call pop(). Each
// side keeps a cached copy of the other's index so the common case touches only
// its own cache line. Capacity is rounded up to a power of two so wrapping is a
// mask — check capacity() if the exact number matters to a caller.
template <typename T>
class SpscQueue {
public:
    explicit SpscQueue(std::size_t capacity)
        : cap_(next_pow2(capacity)), mask_(cap_ - 1),
          buf_(std::make_unique<T[]>(cap_)) {}

    SpscQueue(const SpscQueue&) = delete;
    SpscQueue& operator=(const SpscQueue&) = delete;

    // Producer side. False if full — never blocks, because a producer that
    // blocks here has coupled the GPU to the control thread (§7).
    bool push(const T& value) { return emplace(value); }
    bool push(T&& value) { return emplace(std::move(value)); }

    // Consumer side. False if empty.
    bool pop(T& out) {
        const std::size_t head = head_.load(std::memory_order_relaxed);
        if (head == tail_cache_) {
            tail_cache_ = tail_.load(std::memory_order_acquire);
            if (head == tail_cache_) return false;  // genuinely empty
        }
        out = std::move(buf_[head & mask_]);
        head_.store(head + 1, std::memory_order_release);
        return true;
    }

    // Approximate from either side: the other index can move between the two
    // loads. Good enough for the refresh trigger, which exists precisely to hold
    // slack — do not build a correctness argument on it.
    std::size_t size() const {
        const std::size_t tail = tail_.load(std::memory_order_acquire);
        const std::size_t head = head_.load(std::memory_order_acquire);
        return tail - head;
    }

    bool empty() const { return size() == 0; }
    std::size_t capacity() const { return cap_; }

private:
    template <typename U>
    bool emplace(U&& value) {
        const std::size_t tail = tail_.load(std::memory_order_relaxed);
        if (tail - head_cache_ == cap_) {
            head_cache_ = head_.load(std::memory_order_acquire);
            if (tail - head_cache_ == cap_) return false;  // genuinely full
        }
        buf_[tail & mask_] = std::forward<U>(value);
        tail_.store(tail + 1, std::memory_order_release);
        return true;
    }

    const std::size_t cap_;
    const std::size_t mask_;
    std::unique_ptr<T[]> buf_;

    // Grouped by which thread touches them, not by what they mean: the consumer
    // owns head_ and tail_cache_, the producer owns tail_ and head_cache_. Two
    // hot lines. Giving each of the four its own line costs 128 bytes and buys
    // no additional isolation.
    alignas(kCacheLineSize) std::atomic<std::size_t> head_{0};
    std::size_t tail_cache_{0};

    alignas(kCacheLineSize) std::atomic<std::size_t> tail_{0};
    std::size_t head_cache_{0};
};

}  // namespace cerebellum::detail
