#include "residual_connection.h"

ResidualConnectionImpl::ResidualConnectionImpl(float dropout_p)
    : norm(/*eps=*/1e-6), dropout(torch::nn::DropoutOptions(dropout_p)) {

    register_module("norm", norm);
    register_module("dropout", dropout);
}

torch::Tensor ResidualConnectionImpl::forward(torch::Tensor x, std::function<torch::Tensor(torch::Tensor)> sublayer) {
    auto norm_x = norm->forward(x);
    auto sublayer_out = sublayer(norm_x);
    auto dropped = dropout(sublayer_out);

    return x + dropped;
}
