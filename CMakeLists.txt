cmake_minimum_required(VERSION 3.12)
project(cpp_nn_from_scratch LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.5.2
)
FetchContent_MakeAvailable(Catch2)

enable_testing()

add_library(nn
  include/nn/tensor.hpp
  include/nn/layer.hpp
  include/nn/network.hpp
  include/nn/activation.hpp
  include/nn/loss.hpp
)
target_include_directories(nn PUBLIC include)

add_executable(main src/main.cpp)
target_link_libraries(main PRIVATE nn)

add_executable(tests tests/test_network.cpp)
target_link_libraries(tests PRIVATE nn Catch2::Catch2WithMain)
include(CTest)
include(Catch)
catch_discover_tests(tests)
