#pragma once
#include <vector> 


std::vector<double> linear_layer(
    const std::vector<double>& input,
    const std::vector<double>& W, size_t n_out, size_t n_in,
    const std::vector<double>& b
);

size_t predict(
    const std::vector<double>& input,
    const std::vector<double>& W1, size_t n_hidden, size_t n_in,
    const std::vector<double>& b1,
    const std::vector<double>& W2, size_t n_out,
    const std::vector<double>& b2
); 