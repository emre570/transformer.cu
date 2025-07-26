#include <torch/torch.h>
#include "layer_norm.h"
#include "residual_connection.h"

ResidualConnectionImpl::ResidualConnectionImpl(int d_model, float dropout_p)
    : norm(d_model),
      dropout(torch::nn::DropoutOptions(dropout_p))
{
    register_module("norm", norm);
    register_module("dropout", dropout);
}

torch::Tensor ResidualConnectionImpl::forward(torch::Tensor x, std::function<torch::Tensor(torch::Tensor)> sublayer) {
    auto normed = norm->forward(x);
    auto out = sublayer(normed);
    out = dropout(out);
    return x + out;
}
