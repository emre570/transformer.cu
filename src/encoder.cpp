// encoder.cpp
#include "encoder.h"

EncoderImpl::EncoderImpl(int num_layers, int d_model, int n_heads, int d_ff, float dropout)
    : norm(1e-6f)
{
    for (int i = 0; i < num_layers; ++i) {
        auto block = EncoderBlock(d_model, n_heads, d_ff, dropout);
        layers.push_back(block);
        register_module("encoder_block_" + std::to_string(i), block);
    }

    register_module("norm", norm);
}

torch::Tensor EncoderImpl::forward(torch::Tensor x, torch::Tensor mask) {
    for (auto& layer : layers) {
        x = layer->forward(x, mask);
    }

    return norm->forward(x);
}
