#pragma once
#include <torch/torch.h>

class FeedForwardImpl : public torch::nn::Module {
public:
    FeedForwardImpl(int d_model, int d_ff, float dropout_p);
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Linear linear_1{nullptr}, linear_2{nullptr};
    torch::nn::Dropout dropout{nullptr};
};

TORCH_MODULE(FeedForward);
