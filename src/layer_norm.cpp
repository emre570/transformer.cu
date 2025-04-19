#include "layer_norm.h"
#include <cmath>

LayerNormImpl::LayerNormImpl(float eps_) : eps(eps_) {
    auto alpha = torch::ones({1}, torch::requires_grad(true).dtype(torch::kFloat32));

    auto bias = torch::zeros({1}, torch::requires_grad(true).dtype(torch::kFloat32));

    this->register_parameter("alpha", alpha);
    this->alpha = alpha;

    this->register_parameter("bias", bias);
    this->bias = bias;
}

torch::Tensor LayerNormImpl::forward(torch::Tensor x) {
    auto mean = x.mean(-1, true);
    auto var = x.var(-1, false, true);
    auto normalized = (x - mean) / sqrt(var + eps);
    auto y = this->alpha * normalized + this->bias;

    return y;
}
