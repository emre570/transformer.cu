// encoder.h
#pragma once
#include <torch/torch.h>
#include <vector>
#include "encoder_block.h"
#include "layer_norm.h"

struct EncoderImpl : torch::nn::Module {
    std::vector<EncoderBlock> layers;
    LayerNorm norm;

    EncoderImpl(int num_layers, int d_model, int n_heads, int d_ff, float dropout);

    torch::Tensor forward(torch::Tensor x, torch::Tensor mask);
};
TORCH_MODULE(Encoder);
