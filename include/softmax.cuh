#pragma once

#include <cuda_runtime.h>

__global__ void softmaxKernel(float* input, float* output, int num_rows, int num_cols);

// Eğer istersen wrapper da ekleyebiliriz (launch_softmax)
void launch_softmax(float* input, float* output, int num_rows, int num_cols);
