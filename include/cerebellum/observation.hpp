#pragma once

#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "cerebellum/timing.hpp"

namespace cerebellum {

// Camera-native HWC bytes. Keeping pixels as uint8_t makes a three-camera
// observation four times smaller than expanding it to float before transport;
// Python converts to the CHW float representation expected by SmolVLA.
struct CameraImage {
    std::string feature_name;
    std::uint16_t channels = 0;
    std::uint16_t height = 0;
    std::uint16_t width = 0;
    std::vector<std::uint8_t> pixels;

    void validate() const {
        if (feature_name.empty()) throw std::invalid_argument("camera feature name is empty");
        if (channels == 0 || height == 0 || width == 0) {
            throw std::invalid_argument("camera dimensions must be positive");
        }
        const std::size_t expected = static_cast<std::size_t>(channels) * height * width;
        if (pixels.size() != expected) {
            throw std::invalid_argument("camera pixel count does not match its dimensions");
        }
    }
};

struct ObservationSnapshot {
    std::uint64_t sequence = 0;
    TimePoint capture_time{};
    std::vector<float> state;
    std::vector<CameraImage> images;
    std::string task;

    void validate() const {
        if (capture_time == TimePoint{}) throw std::invalid_argument("capture time is missing");
        if (state.empty()) throw std::invalid_argument("robot state is empty");
        for (float value : state) {
            if (!std::isfinite(value)) throw std::invalid_argument("robot state is non-finite");
        }
        if (images.empty()) throw std::invalid_argument("observation has no camera images");
        for (std::size_t i = 0; i < images.size(); ++i) {
            images[i].validate();
            for (std::size_t earlier = 0; earlier < i; ++earlier) {
                if (images[earlier].feature_name == images[i].feature_name) {
                    throw std::invalid_argument("camera feature names must be unique");
                }
            }
        }
        if (task.empty()) throw std::invalid_argument("task is empty");
    }
};

// The inference worker asks for one immutable newest-wins snapshot. Returning a
// shared owner lets Retina/Axon replace the current observation without copying
// image buffers or invalidating an inference already encoding the previous one.
class ObservationSource {
public:
    virtual ~ObservationSource() = default;
    virtual std::shared_ptr<const ObservationSnapshot> latest() noexcept = 0;
};

}  // namespace cerebellum
