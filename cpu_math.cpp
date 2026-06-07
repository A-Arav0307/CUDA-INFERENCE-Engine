#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>


double sum_vector(const std::vector<double>& v) {
    double sum = 0.0;
    for (double x : v){
        sum += x;
    }
    return sum;
}


double dot_product(const std::vector<double>& a, const std::vector<double>& b) {
    assert(a.size() == b.size());
    double result = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        result += a[i] * b[i];
    }
    return result;
}


std::vector<double> vector_add(const std::vector<double>& a, const std::vector<double>& b) {
    assert(a.size() == b.size());
    std::vector<double> result;
    for (size_t i = 0; i < a.size(); ++i) {
        result.push_back(a[i] + b[i]);
    }
    return result;
}


int argmax(const std::vector<double>& v) {
    if (v.size() == 0){
        return 0;
    }
    double maximum = v[0];
    int final = 0;

    for (size_t i = 0; i < v.size(); ++i){
        if (v[i] > maximum){
            maximum = v[i];
            final = i;
        }

    }

    return final;

}


std::vector<double> matvec(const std::vector<double>& M, size_t rows, size_t cols, const std::vector<double>& v){
    assert(M.size() == rows * cols and v.size() == cols);
    std::vector<double> multiplied_matrix(rows, 0.0);
    for (size_t i = 0; i < rows; ++i){
        for (size_t j = 0; j < cols; ++j){
            multiplied_matrix[i] += M[i*cols + j] * v[j];
        }
    }
    return multiplied_matrix;
}

std::vector<double> matmul(const std::vector<double>& A, size_t A_rows, size_t A_cols, const std::vector<double>& B, size_t B_rows, size_t B_cols){
    assert(A_cols == B_rows);
    std::vector<double> multiplied_matrix(A_rows*B_cols, 0.0);
    for (size_t i = 0; i < B_cols; ++i){
        for (size_t j = 0; j < A_rows; ++j){
            for (size_t k = 0; k < A_cols; ++k){
                multiplied_matrix[i * B_cols + j] += A[i * A_cols + k] * B[k * B_cols + j];
            }

        }
    }

    return multiplied_matrix;
}


int main() {
    return 0;
}
