#include "nn/network.hpp"
#include <catch2/catch_test_macros.hpp>

TEST_CASE("XOR forward pass") {
    nn::Network net({2, 2, 1});
    nn::Tensor x(2, 1);
    x(0, 0) = 1.0;
    x(1, 0) = 0.0;
    auto y = net.forward(x);
    REQUIRE(y.rows == 1);
    REQUIRE(y.cols == 1);
}
