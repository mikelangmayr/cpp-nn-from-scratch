#pragma once
#include <cmath>

namespace nn {

inline double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

inline double sigmoid_deriv(double x) {
    double s = sigmoid(x);
    return s * (1 - s);
}

} // namespace nn
