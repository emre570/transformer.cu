// encoder.h

#pragma once
#include <torch/torch.h>

struct EncoderImpl : torch::nn::Module{
    torch::nn::ModuleList layers{nullptr};
    torch::nn::LayerNorm norm{nullptr};

    EncoderImpl(int num_layers, int embed_dim);
    torch::Tensor forward(torch::Tensor x, torch::Tensor mask);
};

TORCH_MODULE(Encoder);