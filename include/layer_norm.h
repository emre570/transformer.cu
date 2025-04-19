#pragma once
#include <torch/torch.h>

// Normal class değil, Module tanımı: LayerNormImpl + TORCH_MODULE makrosu
class LayerNormImpl : public torch::nn::Module {
public:
    LayerNormImpl(float eps = 1e-6);
    torch::Tensor forward(torch::Tensor x);

private:
    float eps;
    torch::Tensor alpha;  // learnable scale
    torch::Tensor bias;   // learnable bias
};

TORCH_MODULE(LayerNorm);  // LayerNorm -> shared_ptr<LayerNormImpl>
