#include "nn/network.hpp"
#include <iostream>

int main() {
    nn::Network net({2, 2, 1});
    nn::Tensor x(2, 1);
    x(0, 0) = 1.0;
    x(1, 0) = 0.0;
    auto y = net.forward(x);
    std::cout << "Output: " << y(0, 0) << std::endl;
    return 0;
}
