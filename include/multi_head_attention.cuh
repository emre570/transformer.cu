#pragma once
#include <torch/torch.h>

/**
 * @param Q [batch, seq_len, num_heads * head_dim]
 * @param K same shape as Q
 * @param V same shape as Q
 * @param batch_size örn: 3
 * @param num_heads örn: 8
 * @param seq_len örn: 4096
 * @param head_dim örn: 32
 *
 * @return [batch, seq_len, num_heads * head_dim]
 */
torch::Tensor multi_head_attention_cuda(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim
);
