// embedding.cpp

#include "embedding.h"

InputEmbedding::InputEmbedding(int vocab_size, int embedding_dim) {
    embedding_weights = torch::randn({vocab_size, embedding_dim}, torch::kCUDA);
    embedding_weights.set_requires_grad(true);
}

torch::Tensor InputEmbedding::forward(torch::Tensor token_ids) {
    auto out = torch::embedding(embedding_weights, token_ids);
    return out * std::sqrt((float)embedding_weights.size(1));
}
