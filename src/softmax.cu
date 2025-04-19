#include "softmax.cuh"
#include <curand.h>
#include <cuda_runtime.h>
#include <iostream>

#define ELEMENTS_PER_THREAD 16
#define BLOCK_SIZE 512

__global__ void softmaxKernel(float* input, float* output, int num_rows, int num_cols) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int base_idx = tid * ELEMENTS_PER_THREAD;

    float tile[ELEMENTS_PER_THREAD];
    
    float* input_row = input + row * num_cols;
    float* output_row = output + row * num_cols;
    
    float local_max = -INFINITY;
    float local_norm = 0.0f;

    #pragma unroll
    //Fill tile with ELEMENTS_PER_THREAD of elements
    for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
        int idx = base_idx + i;
        if (idx < num_cols) {
            tile[i] = input_row[idx];
        }
    }

    #pragma unroll
    //Local max from tile
    for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
        if (tile[i] > local_max)
            local_max = tile[i];
    }

    #pragma unroll
    //Local norm from tile
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        float x = tile[i];

        tile[i] = __expf(fminf(x - local_max, 80.0f));
        local_norm += tile[i];
    }

    //Warp-level reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
        local_norm += __shfl_down_sync(0xffffffff, local_norm, offset);
    }

    //Write SMEM per warp
    __shared__ float smem_max[32]; 
    __shared__ float smem_norm[32];

    if (tid % 32 == 0) {
        smem_max[tid / 32] = local_max;
        smem_norm[tid / 32] = local_norm;
    }
    __syncthreads();

    //Block-wise warp reduction
    if (blockDim.x > 32) {
        if (tid < 32) {
            local_max = smem_max[tid];
            local_norm = smem_norm[tid];

            for (int offset = 16; offset > 0; offset /= 2) {
                local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
                local_norm += __shfl_down_sync(0xffffffff, local_norm, offset);
            }

            if (tid == 0) {
                smem_max[0] = local_max;
                smem_norm[0] = local_norm;
            }
        }
    }
    __syncthreads();

    //Final Softmax calculation
    //float row_max = smem_max[0];
    float row_norm = smem_norm[0];
    if (row_norm < 1e-6f) row_norm = 1e-6f;
    __syncthreads();

    if (threadIdx.x == 0 && row == 0) {
        printf("row_norm = %f\n", row_norm);
        for (int i = 0; i < ELEMENTS_PER_THREAD; ++i)
            printf("tile[%d] = %f\n", i, tile[i]);
    }    

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
        int idx = base_idx + i;
        if (idx < num_cols) {
            output_row[idx] = tile[i] / row_norm;
        }
    }
}

void launch_softmax(float* input, float* output, int num_rows, int num_cols) {
    dim3 grid_dim(num_rows); // her row için 1 block

    int threads_per_row = (num_cols + ELEMENTS_PER_THREAD - 1) / ELEMENTS_PER_THREAD;
    threads_per_row = std::min(threads_per_row, BLOCK_SIZE);  // safety cap
    dim3 block_dim(threads_per_row);

    softmaxKernel<<<grid_dim, block_dim>>>(input, output, num_rows, num_cols);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Softmax Kernel Error: " << cudaGetErrorString(err) << std::endl;
    }
}
