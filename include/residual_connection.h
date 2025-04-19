#pragma once
#include <torch/torch.h>
#include "layer_norm.h"

class ResidualConnectionImpl : public torch::nn::Module {
public:
    ResidualConnectionImpl(float dropout_p);
    torch::Tensor forward(torch::Tensor x, std::function<torch::Tensor(torch::Tensor)> sublayer);

private:
    LayerNorm norm;
    torch::nn::Dropout dropout;
};

TORCH_MODULE(ResidualConnection);
