#include "matmul.cuh"
#include <iostream>

#define TILE_WIDTH 32
#define BLOCK_X TILE_WIDTH
#define BLOCK_Y TILE_WIDTH

__global__ void tiled_matmul_kernel(const float* A, const float* B, float* C,
                                    int M, int N, int K) {
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x, ty = threadIdx.y;
    int Row = blockIdx.y * TILE_WIDTH + ty;
    int Col = blockIdx.x * TILE_WIDTH + tx;

    float Cvalue = 0.0f;

    for (int ph = 0; ph < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++ph) {
        if (Row < M && (ph * TILE_WIDTH + tx) < K)
            As[ty][tx] = A[Row * K + ph * TILE_WIDTH + tx];
        else
            As[ty][tx] = 0.0f;

        if (Col < N && (ph * TILE_WIDTH + ty) < K)
            Bs[ty][tx] = B[(ph * TILE_WIDTH + ty) * N + Col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            Cvalue += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (Row < M && Col < N)
        C[Row * N + Col] = Cvalue;
}

void launch_matmul(const float* d_A, const float* d_B, float* d_C,
                   int M, int N, int K) {
    dim3 blockDim(BLOCK_X, BLOCK_Y);
    dim3 gridDim((N + TILE_WIDTH - 1) / TILE_WIDTH,
                 (M + TILE_WIDTH - 1) / TILE_WIDTH);

    tiled_matmul_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
}
