#include "transpose.cuh"
#include <iostream>

#define TILE_WIDTH 32

__global__ void tiled_transpose_kernel(const float* input, float* output, int width, int height) {
    __shared__ float tile[TILE_WIDTH][TILE_WIDTH + 1];

    int x = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int y = blockIdx.y * TILE_WIDTH + threadIdx.y;

    if (x < width && y < height)
        tile[threadIdx.y][threadIdx.x] = input[y * width + x];

    __syncthreads();

    x = blockIdx.y * TILE_WIDTH + threadIdx.x;
    y = blockIdx.x * TILE_WIDTH + threadIdx.y;

    if (x < height && y < width)
        output[y * height + x] = tile[threadIdx.x][threadIdx.y];
}

void launch_transpose(const float* d_input, float* d_output, int width, int height) {
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim((width + TILE_WIDTH - 1) / TILE_WIDTH,
                 (height + TILE_WIDTH - 1) / TILE_WIDTH);

    tiled_transpose_kernel<<<gridDim, blockDim>>>(d_input, d_output, width, height);
}
