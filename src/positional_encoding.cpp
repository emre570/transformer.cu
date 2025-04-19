#include "positional_encoding.h"
#include <cmath>  // std::log, std::exp, std::sin, std::cos

PositionalEncodingImpl::PositionalEncodingImpl(int d_model, float dropout_p, int seq_len)
    : d_model(d_model), seq_len(seq_len), dropout(torch::nn::DropoutOptions().p(dropout_p)) {
    
    auto pe = torch::zeros({seq_len, d_model}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

    // Başlangıç ve bitiş değerleri ile aralık oluşturma
    auto position = torch::arange(0, seq_len, options).unsqueeze(1);

    // Belirli bir adım değeri ile aralık oluşturma
    auto div_term = torch::arange(0, d_model, 2, options);
    
    pe.index_put_({torch::indexing::Slice(), torch::indexing::Slice(0, torch::indexing::None, 2)},
    torch::sin(position * div_term));
    pe.index_put_({torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None, 2)},
    torch::cos(position * div_term));
    
    pe = pe.unsqueeze(0);  // [1, seq_len, d_model]
    this->register_buffer("pe", pe);
    this->pe = pe;
}

torch::Tensor PositionalEncodingImpl::forward(torch::Tensor x) {
    pe = pe.to(x.device());
    x = x + pe.index({torch::indexing::Slice(), torch::indexing::Slice(0, x.size(1)), torch::indexing::Ellipsis}).detach(); 

    return dropout(x);
}
