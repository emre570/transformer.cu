#pragma once
#include <torch/torch.h>
#include "layer_norm.h"

struct ResidualConnectionImpl : torch::nn::Module{
    LayerNorm norm;
    torch::nn::Dropout dropout;

    ResidualConnectionImpl(int d_model, float dropout);

    torch::Tensor forward(torch::Tensor x, std::function<torch::Tensor(torch::Tensor)> sublayer);
};

TORCH_MODULE(ResidualConnection);
