#include <iostream>
#include "embedding.h"
#include "positional_encoding.h"
#include "layer_norm.h"
#include "residual_connection.h"
#include "feed_forward.h"
#include "softmax.cuh"

void init_input_embedding(){
    InputEmbedding embed(30522, 512);

    auto token_ids = torch::tensor({3, 5, 10}, torch::kInt64).to(torch::kCUDA);
    auto embedded = embed.forward(token_ids);

    std::cout << embedded << std::endl;
}

void init_pos_enc(){
    auto pos_enc = PositionalEncoding(512, 0.1, 2048);
    pos_enc->to(torch::kCUDA);  // ✅ BU GERÇEKTEN TAŞIR

    auto dummy_input = torch::randn({1, 128, 512}).to(torch::kCUDA);
    auto out = pos_enc->forward(dummy_input);

    std::cout << out << std::endl;
}

void init_layernorm(){
    torch::manual_seed(0);

    auto x = torch::randn({2, 4, 8}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    LayerNorm ln(/*eps=*/1e-6);
    ln->to(torch::kCUDA);  // Move parameters to GPU
    auto out = ln->forward(x);

    std::cout << "Input shape: " << x.sizes() << std::endl;
    std::cout << "Output shape: " << out.sizes() << std::endl;

    auto mean = out.mean(-1);
    auto stddev = out.std(-1, false);

    std::cout << "Output mean (per vector): " << mean << std::endl;
    std::cout << "Output stddev (per vector): " << stddev << std::endl;

}

void init_residual(){
    torch::manual_seed(0);

    // Dummy input tensor
    auto x = torch::randn({2, 4, 8}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    // Dummy sublayer: a simple linear projection
    auto linear = torch::nn::Linear(8, 8);  // input and output dim must match for residual
    linear->to(torch::kCUDA);

    // Create ResidualConnection module
    ResidualConnection residual(0.1f);  // dropout = 10%
    residual->to(torch::kCUDA);

    // Apply residual(x, sublayer)
    auto out = residual->forward(x, [&](torch::Tensor normed) {
        return linear->forward(normed);
    });

    // Print shapes
    std::cout << "Input shape: " << x.sizes() << std::endl;
    std::cout << "Output shape: " << out.sizes() << std::endl;

    // Optional: check that values changed (just a quick check)
    std::cout << "First input row:\n" << x[0][0] << std::endl;
    std::cout << "First output row:\n" << out[0][0] << std::endl;
}

void init_ffn(){
    torch::manual_seed(0);

    const int batch = 2;
    const int seq_len = 4;
    const int d_model = 512;
    const int d_ff = 2048;
    const float dropout_p = 0.1;

    // Dummy input tensor
    auto x = torch::randn({batch, seq_len, d_model}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    // Create FeedForward module
    FeedForward ff(d_model, d_ff, dropout_p);
    ff->to(torch::kCUDA);

    // Run forward
    auto out = ff->forward(x);

    // Print results
    std::cout << "Input shape: " << x.sizes() << "\n";
    std::cout << "Output shape: " << out.sizes() << "\n";

    std::cout << "First input row:\n" << x[0][0] << "\n";
    std::cout << "First output row:\n" << out[0][0] << "\n";
}

torch::Tensor softmax_cuda(torch::Tensor input) {
    TORCH_CHECK(input.dim() == 2, "softmax_cuda only supports 2D tensors");
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");

    int num_rows = input.size(0);
    int num_cols = input.size(1);

    auto output = torch::empty_like(input);

    float* input_ptr = input.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    // Call your launcher
    launch_softmax(input_ptr, output_ptr, num_rows, num_cols);

    // Optional: Wait for kernel to finish and check for errors
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA softmax kernel failed: ") + cudaGetErrorString(err));
    }

    return output;
}

int main() {
    auto input = torch::randn({128, 256}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    //init_input_embedding();
    //init_pos_enc();
    //tensor_ops();
    //init_layernorm();
    //init_residual();
    //init_ffn();
    auto result = softmax_cuda(input);
    std::cout << result << "\n";
}
