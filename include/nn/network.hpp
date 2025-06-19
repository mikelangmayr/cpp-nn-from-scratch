#pragma once
#include "layer.hpp"
#include <vector>

namespace nn {

class Network {
public:
    std::vector<Layer> layers;

    Network(const std::vector<size_t>& sizes) {
        for (size_t i = 0; i + 1 < sizes.size(); ++i)
            layers.emplace_back(sizes[i], sizes[i + 1]);
    }

    Tensor forward(Tensor x) {
        for (auto& layer : layers)
            x = layer.forward(x);
        return x;
    }
};

} // namespace nn
