//#include <cudaruntime.h>
#include <torch/torch.h>
#include "matmul.cuh"
#include "transpose.cuh"
#include "softmax.cuh"

void calc_matmul(torch::Tensor A, torch::Tensor B, torch::Tensor C, int M, int N, int K){
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Kernel çağrısı
    launch_matmul(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    float seconds = milliseconds / 1000.0;
    double flop = 2.0 * M * N * K;
    double gflops = flop / (seconds * 1e9);

    std::cout << "Matmul Runtime: " << milliseconds << " ms" << std::endl;
    std::cout << "Matmul Performance: " << gflops << " GFLOPS" << std::endl;

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
}

void calc_transpose(torch::Tensor input, int rows, int cols){
    auto output = torch::empty({cols, rows}, torch::device(torch::kCUDA).dtype(torch::kFloat32));

    // zamanlayıcı başlat
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // kernel çağrısı
    launch_transpose(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        cols, rows
    );

    // zamanlayıcı bitiş
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    float seconds = milliseconds / 1000.0;
    double flop = 2.0 * rows * cols;
    double gflops = flop / (seconds * 1e9);

    double bytes = 2.0 * rows * cols * sizeof(float);  // input + output
    double bandwidthGBs = bytes / (milliseconds / 1000.0) / 1e9;

    std::cout << "Transpose Runtime: " << milliseconds << " ms" << std::endl;
    std::cout << "Transpose Performance: " << gflops << " GFLOPS" << std::endl;
    std::cout << "Transpose Effective Bandwidth: " << bandwidthGBs << " GB/s" << std::endl;
}

void calc_softmax(torch::Tensor logits){
    auto output = torch::empty_like(logits);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    launch_softmax(
        logits.data_ptr<float>(),
        output.data_ptr<float>(),
        logits.size(0), logits.size(1)
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    float seconds = milliseconds / 1000.0;
    double flop = 5.0 * logits.size(0) * logits.size(1);
    double gflops = flop / (seconds * 1e9);

    std::cout << "Softmax Runtime: " << milliseconds << " ms" << std::endl;
    std::cout << "Softmax Performance: " << gflops << " GFLOPS" << std::endl;
}

int main() {
    int M=8192, N=8192, K=8192;

    torch::Tensor A = torch::randn({M, N}, torch::kCUDA);
    torch::Tensor B = torch::randn({N, K}, torch::kCUDA);
    torch::Tensor C = torch::zeros({M, K}, torch::kCUDA);

    calc_matmul(A,B,C,M,N,K);
    calc_transpose(A, M, N);
    calc_softmax(A);
    
    return 0;
}
