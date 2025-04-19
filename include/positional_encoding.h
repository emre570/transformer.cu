#pragma once
#include <torch/torch.h>

class PositionalEncodingImpl : public torch::nn::Module {
public:
    PositionalEncodingImpl(int d_model, float dropout, int seq_len);
    torch::Tensor forward(torch::Tensor x);

private:
    int d_model;
    int seq_len;
    torch::nn::Dropout dropout;
    torch::Tensor pe;
};

TORCH_MODULE(PositionalEncoding);  // Bu satır C++ tarafında typedef gibi