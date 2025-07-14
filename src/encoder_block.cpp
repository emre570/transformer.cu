// // encoder_block.cpp
// #include "encoder_block.h"

// EncoderBlockImpl::EncoderBlockImpl(int d_model, int n_heads, int d_ff, float dropout)
//     : self_attention(d_model, n_heads, dropout),
//       feed_forward(d_model, d_ff, dropout),
//       residual1(dropout),
//       residual2(dropout) {

//     register_module("self_attention", self_attention);
//     register_module("feed_forward", feed_forward);
//     register_module("residual1", residual1);
//     register_module("residual2", residual2);
// }

// torch::Tensor EncoderBlockImpl::forward(torch::Tensor x, torch::Tensor mask) {
//     x = residual1->forward(x, [&](torch::Tensor normed) {
//         return self_attention->forward(normed, normed, normed, mask);
//     });

//     x = residual2->forward(x, [&](torch::Tensor normed) {
//         return feed_forward->forward(normed);
//     });

//     return x;
// }
