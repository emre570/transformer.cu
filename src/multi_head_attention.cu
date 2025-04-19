#include "multi_head_attention.cuh"
#include "matmul.cuh"
#include "transpose.cuh"
#include "softmax.cuh"

#include <cuda_runtime.h>
#include <iostream>

torch::Tensor multi_head_attention_cuda(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim
) {
    int B = batch_size;
    int H = num_heads;
    int S = seq_len;
    int D = head_dim;

    int total_heads = B * H;

    // Boyut kontrolü
    TORCH_CHECK(Q.dim() == 3 && K.dim() == 3 && V.dim() == 3, "Q, K, V must be 3D tensors");
    TORCH_CHECK(Q.size(0) == B && Q.size(1) == S && Q.size(2) == H * D, "Q has incorrect shape");

    // View + contiguous
    auto Q_reshaped = Q.view({B * H, S, D}).contiguous();
    auto K_reshaped = K.view({B * H, S, D}).contiguous();
    auto V_reshaped = V.view({B * H, S, D}).contiguous();

    float* q_ptr = Q_reshaped.data_ptr<float>();
    float* k_ptr = K_reshaped.data_ptr<float>();
    float* v_ptr = V_reshaped.data_ptr<float>();

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

    auto K_T_tensor = torch::empty({total_heads, D, S}, options);
    float* k_t_ptr = K_T_tensor.data_ptr<float>();

    auto attn_scores_tensor = torch::empty({total_heads, S, S}, options);
    float* attn_ptr = attn_scores_tensor.data_ptr<float>();

    auto softmax_tensor = torch::empty({total_heads, S, S}, options);
    float* softmax_tensor_ptr = softmax_tensor.data_ptr<float>();

    auto output_tensor = torch::empty({total_heads, S, D}, options);
    float* out_ptr = output_tensor.data_ptr<float>();

    for (int i = 0; i < total_heads; i++) {
        launch_transpose(
            k_ptr + i * S * D,       // input: [S, D]
            k_t_ptr + i * D * S,     // output: [D, S]
            D, S                     // rows, cols
        );
    }

    for (int i = 0; i < total_heads; i++) {
        launch_matmul(
            q_ptr + i * S * D,         // A: [S × D]
            k_t_ptr + i * D * S,       // B: [D × S]
            attn_ptr + i * S * S,      // C: [S × S]
            S, S, D                    // M, N, K
        );
    }

    attn_scores_tensor.mul_(1.0f / std::sqrt((float)D));

    for (int i = 0; i < total_heads; i++) {
        launch_softmax(
            attn_ptr + i * S * S,
            softmax_tensor_ptr + i * S * S,
            S, S
        );
    }

    for (int i = 0; i < total_heads; i++) {
        launch_matmul(
            softmax_tensor_ptr + i * S * S,  // [S × S]
            v_ptr + i * S * D,               // [S × D]
            out_ptr + i * S * D,             // [S × D]
            S, D, S                          // M, N, K
        );
    }

    auto output_tensor_reshaped = output_tensor.view({B, H, S, D})          // [B, H, S, D]
                  .permute({0, 2, 1, 3})       // [B, S, H, D]
                  .contiguous()
                  .view({B, S, H * D});       // [B, S, H*D]

    return output_tensor_reshaped;
}