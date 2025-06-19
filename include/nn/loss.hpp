#pragma once
#include "tensor.hpp"
#include <cmath>

namespace nn {

inline double mse_loss(const Tensor& y, const Tensor& t) {
    double sum = 0.0;
    for (size_t i = 0; i < y.data.size(); ++i)
        sum += std::pow(y.data[i] - t.data[i], 2);
    return sum / y.data.size();
}

} // namespace nn
