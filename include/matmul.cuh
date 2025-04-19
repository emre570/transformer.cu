#pragma once

#include <cuda_runtime.h>

// Naive tiled matmul
__global__ void tiled_matmul_kernel(const float* A, const float* B, float* C,
                                    int M, int N, int K);

// Host tarafı çağırma fonksiyonu
void launch_matmul(const float* d_A, const float* d_B, float* d_C,
                   int M, int N, int K);
