#pragma once
#include <vector> 

double sum_vector(const std::vector<double>& v);
double dot_product(const std::vector<double>& v); 
std::vector<double> vector_add(const std::vector<double>& a, const std::vector<double>& b); 
int argmax(const std::vector<double>& v);
std::vector<double> matvec(const std::vector<double>& M, size_t rows, size_t cols, const std::vector<double>& v); 
std::vector<double> matmul(const std::vector<double>& A, size_t A_rows, size_t A_cols, const std::vector<double>& B, size_t B_rows, size_t B_cols); 
std::vector<double> relu(const std::vector<double>& v); 
std::vector<double> softmax(const std::vector<double>& v); 