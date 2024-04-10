cmake_minimum_required(VERSION 3.7)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -pthread")

project(run)
add_executable(run src/nn_blocks.cpp src/transformer.cpp src/sampler.cpp src/tokenizer.cpp src/main.cpp)