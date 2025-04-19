// embedding.h

#pragma once
#include <torch/torch.h>

class InputEmbedding {
public:
    InputEmbedding(int vocab_size, int embedding_dim);
    torch::Tensor forward(torch::Tensor token_ids);

private:
    torch::Tensor embedding_weights;
};
