#include "feed_forward.h"

FeedForwardImpl::FeedForwardImpl(int d_model, int d_ff, float dropout_p)
    : linear_1(torch::nn::Linear(d_model, d_ff)),
      linear_2(torch::nn::Linear(d_ff, d_model)),
      dropout(torch::nn::Dropout(dropout_p)) {

    register_module("linear_1", linear_1);
    register_module("linear_2", linear_2);
    register_module("dropout", dropout);
}

torch::Tensor FeedForwardImpl::forward(torch::Tensor x) {
    // TODO: Apply first linear
    auto out = linear_1->forward(x);
    // TODO: Apply relu
    out = torch::relu(out);
    // TODO: Apply dropout
    out = dropout->forward(out);
    // TODO: Apply second linear
    out = linear_2->forward(out);

    return out;
}
