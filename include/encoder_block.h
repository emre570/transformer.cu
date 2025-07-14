// // encoder_block.h
// #pragma once
// #include <torch/torch.h>
// #include "multi_head_attention.cuh"
// #include "feed_forward.h"
// #include "residual_connection.h"

// struct EncoderBlockImpl : torch::nn::Module {
//     multi_head_attention_cuda self_attention;
//     FeedForward feed_forward;
//     ResidualConnection residual1;
//     ResidualConnection residual2;

//     EncoderBlockImpl(int d_model, int n_heads, int d_ff, float dropout);

//     torch::Tensor forward(torch::Tensor x, torch::Tensor mask);
// };
// TORCH_MODULE(EncoderBlock);
