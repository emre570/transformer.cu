cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(torch_test LANGUAGES CXX CUDA)

find_package(Torch REQUIRED)

include_directories(include)

file(GLOB KERNEL_SOURCES src/*.cu)
file(GLOB CPP_SOURCES src/*.cpp)

add_executable(main main.cpp ${CPP_SOURCES} ${KERNEL_SOURCES})
target_link_libraries(main "${TORCH_LIBRARIES}")
set_property(TARGET main PROPERTY CXX_STANDARD 17)
set_property(TARGET main PROPERTY CUDA_SEPARABLE_COMPILATION ON)