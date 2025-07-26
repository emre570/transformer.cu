#include <torch/torch.h>
#include "encoder_block.h"

EncoderBlockImpl::EncoderBlockImpl(int d_model, int n_heads, int d_ff, float dropout)
    : num_heads(n_heads),
      head_dim(d_model / n_heads),
      residual1(dropout),
      residual2(dropout),
      feed_forward(d_model, d_ff, dropout)
{
    register_module("residual1", residual1);
    register_module("residual2", residual2);
    register_module("feed_forward", feed_forward);
}

torch::Tensor EncoderBlockImpl::forward(torch::Tensor x, torch::Tensor mask) {
    int batch_size = x.size(0);
    int seq_len = x.size(1);

    // Residual 1: Self Attention
    x = residual1->forward(x, [&](torch::Tensor normed) {
        return multi_head_attention_cuda(
            normed, normed, normed,
            batch_size, num_heads, seq_len, head_dim
        );
    });

    // Residual 2: FeedForward
    x = residual2->forward(x, [&](torch::Tensor normed) {
        return feed_forward->forward(normed);
    });

    return x;
}