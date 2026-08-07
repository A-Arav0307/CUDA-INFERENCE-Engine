#include "mlp.h"
#include "cpu_math.h"
#include <iostream> 
#include <cassert> 

//showing how github works

std::vector<double> linear_layer(
    const std::vector<double>& input,
    const std::vector<double>& W, size_t n_out, size_t n_in,
    const std::vector<double>& b
){
    assert(b.size() == n_out);
    assert(W.size() == n_out * n_in);
    assert(input.size() == n_in);

    std::vector<double> logits = matvec(W, n_out, n_in, input);
    return vector_add(logits, b);
}


size_t predict(
    const std::vector<double>& input,
    const std::vector<double>& W1, size_t n_hidden, size_t n_in,
    const std::vector<double>& b1,
    const std::vector<double>& W2, size_t n_out,
    const std::vector<double>& b2
){
    assert(W1.size() == n_hidden * n_in);
    assert(b1.size() == n_hidden);
    assert(W2.size() == n_out * n_hidden);
    assert(b2.size() == n_out);
    assert(input.size() == n_in);

    auto h1 = linear_layer(input, W1, n_hidden, n_in, b1);
    auto h1_relu = relu(h1);
    auto logits = linear_layer(h1_relu, W2, n_out, n_hidden, b2);
    auto probs = softmax(logits);
    return argmax(probs);
}