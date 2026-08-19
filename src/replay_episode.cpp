#include "cerebellum/replay_episode.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <istream>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cerebellum {
namespace {

constexpr std::array<char, 8> kMagic{'C', 'B', 'R', 'P', 'L', 'Y', '1', '\0'};
constexpr std::uint32_t kMaxFrames = 1'000'000;
constexpr std::uint32_t kMaxTaskBytes = 1U << 20;
constexpr std::uint16_t kMaxImages = 64;
constexpr std::size_t kMaxImageBytes = 1U << 30;

[[noreturn]] void malformed(const std::string &message) {
    throw std::invalid_argument("invalid replay episode: " + message);
}

void read_exact(std::istream &input, void *destination, std::size_t size, const char *field) {
    input.read(static_cast<char *>(destination), static_cast<std::streamsize>(size));
    if (!input) malformed(std::string("truncated ") + field);
}

void write_exact(std::ostream &output, const void *source, std::size_t size, const char *field) {
    output.write(static_cast<const char *>(source), static_cast<std::streamsize>(size));
    if (!output) throw std::runtime_error(std::string("failed to write replay ") + field);
}

std::uint16_t read_u16(std::istream &input, const char *field) {
    std::array<std::uint8_t, 2> bytes{};
    read_exact(input, bytes.data(), bytes.size(), field);
    return static_cast<std::uint16_t>(bytes[0]) |
           static_cast<std::uint16_t>(static_cast<std::uint16_t>(bytes[1]) << 8U);
}

std::uint32_t read_u32(std::istream &input, const char *field) {
    std::array<std::uint8_t, 4> bytes{};
    read_exact(input, bytes.data(), bytes.size(), field);
    std::uint32_t value = 0;
    for (unsigned i = 0; i < bytes.size(); ++i) value |= static_cast<std::uint32_t>(bytes[i]) << (8U * i);
    return value;
}

std::uint64_t read_u64(std::istream &input, const char *field) {
    std::array<std::uint8_t, 8> bytes{};
    read_exact(input, bytes.data(), bytes.size(), field);
    std::uint64_t value = 0;
    for (unsigned i = 0; i < bytes.size(); ++i) value |= static_cast<std::uint64_t>(bytes[i]) << (8U * i);
    return value;
}

float read_f32(std::istream &input, const char *field) {
    return std::bit_cast<float>(read_u32(input, field));
}

void write_u16(std::ostream &output, std::uint16_t value, const char *field) {
    const std::array<std::uint8_t, 2> bytes{
        static_cast<std::uint8_t>(value), static_cast<std::uint8_t>(value >> 8U)};
    write_exact(output, bytes.data(), bytes.size(), field);
}

void write_u32(std::ostream &output, std::uint32_t value, const char *field) {
    std::array<std::uint8_t, 4> bytes{};
    for (unsigned i = 0; i < bytes.size(); ++i) bytes[i] = static_cast<std::uint8_t>(value >> (8U * i));
    write_exact(output, bytes.data(), bytes.size(), field);
}

void write_u64(std::ostream &output, std::uint64_t value, const char *field) {
    std::array<std::uint8_t, 8> bytes{};
    for (unsigned i = 0; i < bytes.size(); ++i) bytes[i] = static_cast<std::uint8_t>(value >> (8U * i));
    write_exact(output, bytes.data(), bytes.size(), field);
}

void write_f32(std::ostream &output, float value, const char *field) {
    write_u32(output, std::bit_cast<std::uint32_t>(value), field);
}

std::size_t image_bytes(std::uint16_t channels, std::uint16_t height, std::uint16_t width) {
    const std::size_t count = static_cast<std::size_t>(channels) * height * width;
    if (count == 0 || count > kMaxImageBytes) malformed("camera byte count is invalid");
    return count;
}

void require_u16_size(std::size_t size, const char *field) {
    if (size > std::numeric_limits<std::uint16_t>::max()) {
        throw std::invalid_argument(std::string(field) + " exceeds uint16 storage");
    }
}

void validate_recorded_observation(const ObservationSnapshot &observation) {
    if (observation.state.empty()) throw std::invalid_argument("replay robot state is empty");
    for (float value : observation.state) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("replay robot state is non-finite");
        }
    }
    if (observation.images.empty()) {
        throw std::invalid_argument("replay observation has no camera images");
    }
    for (std::size_t i = 0; i < observation.images.size(); ++i) {
        observation.images[i].validate();
        for (std::size_t earlier = 0; earlier < i; ++earlier) {
            if (observation.images[earlier].feature_name ==
                observation.images[i].feature_name) {
                throw std::invalid_argument("replay camera feature names must be unique");
            }
        }
    }
    if (observation.task.empty()) throw std::invalid_argument("replay task is empty");
}

}  // namespace

void ReplayEpisode::validate() const {
    if (task.empty()) throw std::invalid_argument("replay task is empty");
    if (task.size() > kMaxTaskBytes) throw std::invalid_argument("replay task is too large");
    if (state_dim <= 0 || state_dim > std::numeric_limits<std::uint16_t>::max()) {
        throw std::invalid_argument("replay state_dim is invalid");
    }
    if (action_dim < 0 || action_dim > kPaddedActionDim) {
        throw std::invalid_argument("replay action_dim is invalid");
    }
    if (frames.empty() || frames.size() > kMaxFrames) {
        throw std::invalid_argument("replay frame count is invalid");
    }
    if (action_dim == 0) {
        if (!reference_actions.empty()) {
            throw std::invalid_argument("replay has references but action_dim is zero");
        }
    } else if (reference_actions.size() != frames.size()) {
        throw std::invalid_argument("replay reference action count differs from frame count");
    }

    const ObservationSnapshot &first = frames.front().observation;
    if (first.images.size() > kMaxImages) throw std::invalid_argument("replay has too many cameras");
    for (std::size_t i = 0; i < frames.size(); ++i) {
        const ReplayObservationFrame &frame = frames[i];
        if (frame.offset < Nanos::zero() || (i == 0 && frame.offset != Nanos::zero()) ||
            (i > 0 && frame.offset <= frames[i - 1].offset)) {
            throw std::invalid_argument("replay frame offsets must start at zero and increase");
        }
        if (i > 0 && frame.observation.sequence <= frames[i - 1].observation.sequence) {
            throw std::invalid_argument("replay frame sequences must increase");
        }
        const ObservationSnapshot &checked = frame.observation;
        validate_recorded_observation(checked);
        if (checked.task != task || checked.state.size() != static_cast<std::size_t>(state_dim)) {
            throw std::invalid_argument("replay observation task/state schema changed");
        }
        if (checked.images.size() != first.images.size()) {
            throw std::invalid_argument("replay camera count changed");
        }
        for (std::size_t camera = 0; camera < checked.images.size(); ++camera) {
            const CameraImage &actual = checked.images[camera];
            const CameraImage &expected = first.images[camera];
            if (actual.feature_name != expected.feature_name || actual.channels != expected.channels ||
                actual.height != expected.height || actual.width != expected.width) {
                throw std::invalid_argument("replay camera schema changed");
            }
        }
        if (action_dim > 0) {
            for (int d = 0; d < action_dim; ++d) {
                if (!std::isfinite(reference_actions[i][static_cast<std::size_t>(d)])) {
                    throw std::invalid_argument("replay reference action is non-finite");
                }
            }
        }
    }
}

ReplayEpisode read_replay_episode(std::istream &input) {
    std::array<char, kMagic.size()> magic{};
    read_exact(input, magic.data(), magic.size(), "magic");
    if (magic != kMagic) malformed("bad magic or unsupported version");

    const std::uint32_t frame_count = read_u32(input, "frame count");
    const std::uint16_t state_dim = read_u16(input, "state dimension");
    const std::uint16_t camera_count = read_u16(input, "camera count");
    const std::uint16_t action_dim = read_u16(input, "action dimension");
    const std::uint16_t reserved = read_u16(input, "reserved field");
    const std::uint32_t task_size = read_u32(input, "task size");
    if (frame_count == 0 || frame_count > kMaxFrames) malformed("frame count is invalid");
    if (state_dim == 0) malformed("state dimension is zero");
    if (camera_count == 0 || camera_count > kMaxImages) malformed("camera count is invalid");
    if (action_dim > kPaddedActionDim) malformed("action dimension is invalid");
    if (reserved != 0) malformed("reserved field is nonzero");
    if (task_size == 0 || task_size > kMaxTaskBytes) malformed("task size is invalid");

    ReplayEpisode episode;
    episode.state_dim = state_dim;
    episode.action_dim = action_dim;
    episode.task.resize(task_size);
    read_exact(input, episode.task.data(), task_size, "task");

    struct ImageSpec {
        std::string name;
        std::uint16_t channels = 0;
        std::uint16_t height = 0;
        std::uint16_t width = 0;
        std::size_t bytes = 0;
    };
    std::vector<ImageSpec> images;
    images.reserve(camera_count);
    for (std::uint16_t i = 0; i < camera_count; ++i) {
        const std::uint16_t name_size = read_u16(input, "camera name size");
        if (name_size == 0) malformed("camera name is empty");
        ImageSpec spec;
        spec.name.resize(name_size);
        read_exact(input, spec.name.data(), name_size, "camera name");
        spec.channels = read_u16(input, "camera channels");
        spec.height = read_u16(input, "camera height");
        spec.width = read_u16(input, "camera width");
        spec.bytes = image_bytes(spec.channels, spec.height, spec.width);
        images.push_back(std::move(spec));
    }

    episode.frames.reserve(frame_count);
    if (action_dim > 0) episode.reference_actions.reserve(frame_count);
    for (std::uint32_t i = 0; i < frame_count; ++i) {
        ReplayObservationFrame frame;
        frame.observation.sequence = read_u64(input, "frame sequence");
        const std::uint64_t offset_ns = read_u64(input, "frame offset");
        if (offset_ns > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
            malformed("frame offset exceeds monotonic duration range");
        }
        frame.offset = Nanos{static_cast<std::int64_t>(offset_ns)};
        frame.observation.capture_time = TimePoint{Nanos{1}};
        frame.observation.task = episode.task;
        frame.observation.state.resize(state_dim);
        for (float &value : frame.observation.state) value = read_f32(input, "robot state");
        frame.observation.images.reserve(images.size());
        for (const ImageSpec &spec : images) {
            CameraImage image;
            image.feature_name = spec.name;
            image.channels = spec.channels;
            image.height = spec.height;
            image.width = spec.width;
            image.pixels.resize(spec.bytes);
            read_exact(input, image.pixels.data(), image.pixels.size(), "camera pixels");
            frame.observation.images.push_back(std::move(image));
        }
        episode.frames.push_back(std::move(frame));

        if (action_dim > 0) {
            Action action{};
            for (std::uint16_t d = 0; d < action_dim; ++d) {
                action[d] = read_f32(input, "reference action");
            }
            episode.reference_actions.push_back(action);
        }
    }
    if (input.peek() != std::char_traits<char>::eof()) malformed("trailing bytes");
    episode.validate();
    return episode;
}

void write_replay_episode(std::ostream &output, const ReplayEpisode &episode) {
    episode.validate();
    const ObservationSnapshot &first = episode.frames.front().observation;
    require_u16_size(first.images.size(), "camera count");
    write_exact(output, kMagic.data(), kMagic.size(), "magic");
    write_u32(output, static_cast<std::uint32_t>(episode.frames.size()), "frame count");
    write_u16(output, static_cast<std::uint16_t>(episode.state_dim), "state dimension");
    write_u16(output, static_cast<std::uint16_t>(first.images.size()), "camera count");
    write_u16(output, static_cast<std::uint16_t>(episode.action_dim), "action dimension");
    write_u16(output, 0, "reserved field");
    write_u32(output, static_cast<std::uint32_t>(episode.task.size()), "task size");
    write_exact(output, episode.task.data(), episode.task.size(), "task");
    for (const CameraImage &image : first.images) {
        require_u16_size(image.feature_name.size(), "camera name");
        write_u16(output, static_cast<std::uint16_t>(image.feature_name.size()), "camera name size");
        write_exact(output, image.feature_name.data(), image.feature_name.size(), "camera name");
        write_u16(output, image.channels, "camera channels");
        write_u16(output, image.height, "camera height");
        write_u16(output, image.width, "camera width");
    }
    for (std::size_t i = 0; i < episode.frames.size(); ++i) {
        const ReplayObservationFrame &frame = episode.frames[i];
        write_u64(output, frame.observation.sequence, "frame sequence");
        write_u64(output, static_cast<std::uint64_t>(frame.offset.count()), "frame offset");
        for (float value : frame.observation.state) write_f32(output, value, "robot state");
        for (const CameraImage &image : frame.observation.images) {
            write_exact(output, image.pixels.data(), image.pixels.size(), "camera pixels");
        }
        if (episode.action_dim > 0) {
            for (int d = 0; d < episode.action_dim; ++d) {
                write_f32(output, episode.reference_actions[i][static_cast<std::size_t>(d)],
                          "reference action");
            }
        }
    }
}

ReplayQualitySummary evaluate_replay_actions(
    const std::vector<ReplayActionRecord> &records, const std::vector<Action> &reference_actions,
    int action_dim) {
    if (action_dim <= 0 || action_dim > kPaddedActionDim) {
        throw std::invalid_argument("quality action_dim is invalid");
    }
    ReplayQualitySummary result;
    long double absolute_sum = 0.0;
    long double square_sum = 0.0;
    for (const ReplayActionRecord &record : records) {
        if (record.emission.fallback) {
            ++result.fallback_actions;
            continue;
        }
        if (record.emission.step < 0 ||
            static_cast<std::uint64_t>(record.emission.step) >= reference_actions.size()) {
            ++result.missing_reference_actions;
            continue;
        }
        const Action &reference = reference_actions[static_cast<std::size_t>(record.emission.step)];
        double linf = 0.0;
        for (int d = 0; d < action_dim; ++d) {
            const std::size_t index = static_cast<std::size_t>(d);
            const double error = static_cast<double>(record.emission.action[index]) - reference[index];
            const double magnitude = std::abs(error);
            absolute_sum += magnitude;
            square_sum += error * error;
            linf = std::max(linf, magnitude);
        }
        result.max_linf_error = std::max(result.max_linf_error, linf);
        ++result.compared_actions;
    }
    if (result.compared_actions > 0) {
        const long double values =
            static_cast<long double>(result.compared_actions) * action_dim;
        result.mean_absolute_error = static_cast<double>(absolute_sum / values);
        result.root_mean_square_error = static_cast<double>(std::sqrt(square_sum / values));
    }
    return result;
}

}  // namespace cerebellum
