cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(torch_test LANGUAGES CXX CUDA)

set(CMAKE_C_COMPILER /usr/bin/gcc-11)
set(CMAKE_CXX_COMPILER /usr/bin/g++-11)

# point directly to your libtorch folder
set(Torch_DIR "${CMAKE_SOURCE_DIR}/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)

# standard C++17 + CUDA separable compilation
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# your headers and sources
include_directories(${CMAKE_SOURCE_DIR}/include)
file(GLOB_RECURSE CPP_SOURCES  ${CMAKE_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE CUDA_SOURCES ${CMAKE_SOURCE_DIR}/src/*.cu)

add_executable(main main.cpp ${CPP_SOURCES} ${CUDA_SOURCES})
target_link_libraries(main PRIVATE "${TORCH_LIBRARIES}" CUDA::cudart)

message(STATUS "TorchConfig found in: ${Torch_DIR}")
