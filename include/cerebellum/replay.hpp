// Offline playback and action capture using the same ObservationSource and
// ActionSink contracts as a live robot. Dataset decoding stays outside the
// real-time core: load frames first, then start playback on a monotonic clock.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <utility>
#include <vector>

#include "cerebellum/loop.hpp"
#include "cerebellum/observation.hpp"

namespace cerebellum {

struct ReplayObservationFrame {
    Nanos offset{};
    ObservationSnapshot observation{};
};

// Newest-wins playback. Only the inference worker calls latest()/latest_at();
// start() is called before RuntimeLoop starts, so the cursor needs no lock.
class ReplayObservationSource final : public ObservationSource {
   public:
    explicit ReplayObservationSource(std::vector<ReplayObservationFrame> frames) {
        if (frames.empty()) throw std::invalid_argument("replay has no observation frames");
        frames_.reserve(frames.size());
        for (std::size_t i = 0; i < frames.size(); ++i) {
            ReplayObservationFrame &frame = frames[i];
            if (frame.offset < Nanos::zero()) {
                throw std::invalid_argument("replay observation offset is negative");
            }
            if (i == 0 && frame.offset != Nanos::zero()) {
                throw std::invalid_argument("replay must begin at offset zero");
            }
            if (i > 0) {
                if (frame.offset <= frames[i - 1].offset) {
                    throw std::invalid_argument("replay observation offsets must increase");
                }
                if (frame.observation.sequence <= frames[i - 1].observation.sequence) {
                    throw std::invalid_argument("replay observation sequences must increase");
                }
            }
            // Recorded steady-clock epochs are meaningless in a new process.
            // Supply a temporary nonzero value for structural validation; start()
            // rebases every capture stamp onto this playback's clock.
            frame.observation.capture_time = TimePoint{Nanos{1}};
            frame.observation.validate();
            frames_.push_back(Frame{frame.offset,
                                    std::make_shared<ObservationSnapshot>(
                                        std::move(frame.observation))});
        }
    }

    void start(TimePoint origin = now()) noexcept {
        origin_ = origin;
        cursor_ = 0;
        for (Frame &frame : frames_) frame.observation->capture_time = origin_ + frame.offset;
        started_ = true;
    }

    std::shared_ptr<const ObservationSnapshot> latest() noexcept override {
        return latest_at(now());
    }

    // Public for deterministic replay tests and tools that own their clock.
    std::shared_ptr<const ObservationSnapshot> latest_at(TimePoint timestamp) noexcept {
        if (!started_) return {};
        const Nanos elapsed = timestamp > origin_ ? timestamp - origin_ : Nanos::zero();
        while (cursor_ + 1 < frames_.size() && frames_[cursor_ + 1].offset <= elapsed) {
            ++cursor_;
        }
        return frames_[cursor_].observation;
    }

    bool started() const noexcept { return started_; }
    bool finished(TimePoint timestamp = now()) const noexcept {
        return started_ && timestamp >= origin_ + frames_.back().offset;
    }
    std::size_t frame_count() const noexcept { return frames_.size(); }
    std::size_t current_index() const noexcept { return cursor_; }

   private:
    struct Frame {
        Nanos offset{};
        std::shared_ptr<ObservationSnapshot> observation;
    };

    std::vector<Frame> frames_;
    TimePoint origin_{};
    std::size_t cursor_ = 0;
    bool started_ = false;
};

struct ReplayActionRecord {
    ActionEmission emission{};
    TimePoint emitted_at{};
};

// Preallocated so emit() remains allocation-free on the control thread. A
// caller that undersizes capacity gets an explicit dropped-record count rather
// than a vector growth or a blocked controller.
class ReplayActionSink final : public ActionSink {
   public:
    explicit ReplayActionSink(std::size_t capacity) { records_.reserve(capacity); }

    void emit(const ActionEmission &emission) noexcept override {
        if (records_.size() == records_.capacity()) {
            ++dropped_records_;
            return;
        }
        records_.push_back(ReplayActionRecord{emission, now()});
    }

    const std::vector<ReplayActionRecord> &records() const noexcept { return records_; }
    std::uint64_t dropped_records() const noexcept { return dropped_records_; }

   private:
    std::vector<ReplayActionRecord> records_;
    std::uint64_t dropped_records_ = 0;
};

// Serialization runs after control stops, so stream I/O never enters the
// real-time path. Timestamps are relative to the first scheduled deadline.
inline void write_replay_actions_csv(std::ostream &out,
                                     const std::vector<ReplayActionRecord> &records,
                                     int action_dim) {
    if (action_dim <= 0 || action_dim > kPaddedActionDim) {
        throw std::invalid_argument("replay CSV action_dim is invalid");
    }
    out << "step,chunk_id,chunk_index,chunk_boundary,deadline_offset_ns,emit_offset_ns,fallback,"
           "observation_age_ns,safety_flags,safety_rejected";
    for (int d = 0; d < action_dim; ++d) out << ",action_" << d;
    out << '\n';
    if (records.empty()) return;

    const TimePoint origin = records.front().emission.deadline;
    std::uint64_t previous_chunk_id = 0;
    for (const ReplayActionRecord &record : records) {
        const std::uint64_t chunk_id = record.emission.chunk_id;
        const bool chunk_boundary = chunk_id != 0 && chunk_id != previous_chunk_id;
        out << record.emission.step << ',' << chunk_id << ',' << record.emission.chunk_index << ','
            << (chunk_boundary ? 1 : 0) << ','
            << (record.emission.deadline - origin).count() << ','
            << (record.emitted_at - origin).count() << ','
            << (record.emission.fallback ? 1 : 0) << ','
            << record.emission.observation_age.count() << ',' << record.emission.safety_flags << ','
            << (record.emission.safety_rejected ? 1 : 0);
        for (int d = 0; d < action_dim; ++d) {
            out << ',' << record.emission.action[static_cast<std::size_t>(d)];
        }
        out << '\n';
        if (chunk_id != 0) previous_chunk_id = chunk_id;
    }
}

// Discrete finite differences of the commands that actually crossed the
// safety boundary. They are kept in millionths of action units so the common
// integer percentile recorder can preserve their full distributions.
struct ActionSmoothnessSummary {
    static constexpr std::int64_t kScale = 1'000'000;
    Summary first_difference_linf{};
    Summary second_difference_linf{};
    Summary third_difference_linf{};
};

inline ActionSmoothnessSummary summarize_action_smoothness(
    const std::vector<ReplayActionRecord> &records, int action_dim) {
    if (action_dim <= 0 || action_dim > kPaddedActionDim) {
        throw std::invalid_argument("action smoothness action_dim is invalid");
    }
    PercentileRecorder first(records.size());
    PercentileRecorder second(records.size());
    PercentileRecorder third(records.size());
    const auto valid_window = [&](std::size_t end, std::size_t length) {
        for (std::size_t offset = 0; offset < length; ++offset) {
            const ActionEmission &emission = records[end - offset].emission;
            if (emission.fallback || emission.safety_rejected) return false;
            if (offset > 0 &&
                records[end - offset + 1].emission.step != emission.step + 1) {
                return false;
            }
        }
        return true;
    };
    const auto record_difference = [&](PercentileRecorder &recorder, std::size_t end,
                                       int order) {
        double linf = 0.0;
        for (int d = 0; d < action_dim; ++d) {
            const std::size_t dim = static_cast<std::size_t>(d);
            double value = 0.0;
            for (int k = 0; k <= order; ++k) {
                // (-1)^k C(order, k), for the three orders supported here.
                static constexpr int coefficients[4][4] = {
                    {1, 0, 0, 0}, {1, -1, 0, 0}, {1, -2, 1, 0}, {1, -3, 3, -1}};
                value += coefficients[order][k] *
                         records[end - static_cast<std::size_t>(k)].emission.action[dim];
            }
            linf = std::max(linf, std::abs(value));
        }
        recorder.record(static_cast<std::int64_t>(linf * ActionSmoothnessSummary::kScale + 0.5));
    };
    for (std::size_t i = 1; i < records.size(); ++i) {
        if (valid_window(i, 2)) record_difference(first, i, 1);
        if (i >= 2 && valid_window(i, 3)) record_difference(second, i, 2);
        if (i >= 3 && valid_window(i, 4)) record_difference(third, i, 3);
    }
    return ActionSmoothnessSummary{first.summarize(), second.summarize(), third.summarize()};
}

}  // namespace cerebellum
