#pragma once
#include <vector>
#include <cassert>

namespace nn {

class Tensor {
public:
    std::vector<double> data;
    size_t rows = 0, cols = 0;

    Tensor() = default;

    Tensor(size_t r, size_t c, double val = 0.0)
        : data(r * c, val), rows(r), cols(c) {}

    double& operator()(size_t i, size_t j) {
        assert(i < rows && j < cols);
        return data[i * cols + j];
    }

    double operator()(size_t i, size_t j) const {
        assert(i < rows && j < cols);
        return data[i * cols + j];
    }
};

} // namespace nn
