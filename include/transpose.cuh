#pragma once

#include <cuda_runtime.h>

__global__ void tiled_transpose_kernel(const float* input, float* output, int width, int height);

// Host fonksiyon
void launch_transpose(const float* d_input, float* d_output, int width, int height);
