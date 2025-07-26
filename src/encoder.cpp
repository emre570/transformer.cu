// encoder.cpp

#include "encoder.h"
#include "encoder_block.h"

EncoderImpl::EncoderImpl(int num_layers, int embed_dim){
    //Fill layers inside ModuleList
    layers = register_module("layers", torch::nn::ModuleList());

    for (int i = 0; i < num_layers; ++i){
        layers->push_back(register_module("layer_" + std::to_string(i), EncoderBlock(embed_dim)));
    }
    
    norm = register_module("norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({embed_dim})));
}

torch::Tensor EncoderImpl::forward(torch::Tensor x, torch::Tensor mask){
    for (auto& layer : *layers) {
        auto encoder_block = std::dynamic_pointer_cast<EncoderBlockImpl>(layer);
        x = encoder_block->forward(x, mask);
    }
    return norm(x);
}
