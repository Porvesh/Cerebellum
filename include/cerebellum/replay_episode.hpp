// Portable, versioned offline episode storage and post-run quality metrics.

#pragma once

#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

#include "cerebellum/replay.hpp"

namespace cerebellum {

struct ReplayEpisode {
    std::string task;
    int state_dim = 0;
    int action_dim = 0;  // zero means the episode has no reference actions
    std::vector<ReplayObservationFrame> frames;
    std::vector<Action> reference_actions;

    void validate() const;
};

// The CBR file is deliberately simple: fixed little-endian numeric fields,
// one episode-wide camera schema/task, then fixed-shape frames. No JSON parser,
// codec, or dataset SDK enters the C++ runtime. Images remain camera-native HWC
// bytes, exactly as ObservationSnapshot transports them to Python.
ReplayEpisode read_replay_episode(std::istream &input);
void write_replay_episode(std::ostream &output, const ReplayEpisode &episode);

struct ReplayQualitySummary {
    std::uint64_t compared_actions = 0;
    std::uint64_t fallback_actions = 0;
    std::uint64_t missing_reference_actions = 0;
    double mean_absolute_error = 0.0;
    double root_mean_square_error = 0.0;
    double max_linf_error = 0.0;
};

// Aligns emitted absolute control steps with reference_actions[step]. Fallbacks
// are counted but excluded from numeric error: scoring a held safety command as
// a model prediction would hide an availability failure inside an average.
ReplayQualitySummary evaluate_replay_actions(
    const std::vector<ReplayActionRecord> &records, const std::vector<Action> &reference_actions,
    int action_dim);

}  // namespace cerebellum
