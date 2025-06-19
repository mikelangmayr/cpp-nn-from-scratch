#pragma once
#include "tensor.hpp"
#include "activation.hpp"

namespace nn {

class Layer {
public:
    Tensor weights;
    Tensor biases;
    Tensor input, output;

    Layer(size_t in, size_t out)
        : weights(out, in), biases(out, 1) {}

    Tensor forward(const Tensor& x) {
        input = x;
        output = Tensor(weights.rows, 1);
        for (size_t i = 0; i < weights.rows; ++i) {
            double sum = 0.0;
            for (size_t j = 0; j < weights.cols; ++j)
                sum += weights(i, j) * x(j, 0);
            output(i, 0) = sigmoid(sum + biases(i, 0));
        }
        return output;
    }
};

} // namespace nn
