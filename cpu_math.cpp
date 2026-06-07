#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <algorithm>


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

std::vector<double> relu(const std::vector<double>& v){
    int size = v.size();
    std::vector<double> new_list(size, 0.0);

    for (size_t i = 0; i < v.size(); ++i){
        new_list[i] = std::max(v[i], 0.0);
    }
    return new_list;
}

//returns the vector v with all values adding up to 1
std::vector<double> softmax(const std::vector<double>& v){
    if (v.empty()){
        std::vector<double>empty = {};
        return empty;
    }

    std::vector<double> final_list(v.size(), 0.0);
    double max = v[0];
    for (double val : v){
        max = std::max(max, val);
    }
    double sum = 0;
    for (size_t i = 0; i < v.size(); ++i){
        final_list[i] = std::exp(v[i] - max);
        sum += std::exp(v[i] - max);
    }

    for (size_t i = 0; i < final_list.size(); ++i){
        final_list[i] = final_list[i] / sum;
    }

    return final_list;
}


int main() {
    // sum_vector
    assert(std::fabs(sum_vector({1.0, 2.0, 3.0}) - 6.0) < 1e-9);
    assert(std::fabs(sum_vector({}) - 0.0) < 1e-9);
    assert(std::fabs(sum_vector({-1.0, 1.0}) - 0.0) < 1e-9);
    assert(std::fabs(sum_vector({0.5, 0.25, 0.125}) - 0.875) < 1e-9);

    // dot_product
    assert(std::fabs(dot_product({1, 2, 3}, {4, 5, 6}) - 32.0) < 1e-9);
    assert(std::fabs(dot_product({}, {}) - 0.0) < 1e-9);
    assert(std::fabs(dot_product({1.5, -2.0}, {2.0, 0.5}) - 2.0) < 1e-9);

    // vector_add
    {
        std::vector<double> c = vector_add({1, 2, 3}, {10, 20, 30});
        assert(c.size() == 3);
        assert(std::fabs(c[0] - 11.0) < 1e-9);
        assert(std::fabs(c[1] - 22.0) < 1e-9);
        assert(std::fabs(c[2] - 33.0) < 1e-9);
    }
    {
        std::vector<double> c = vector_add({}, {});
        assert(c.empty());
    }

    // argmax
    assert(argmax({0.1, 0.9, 0.0}) == 1);
    assert(argmax({3, 3, 3}) == 0);
    assert(argmax({-5, -1, -10}) == 1);
    assert(argmax({7}) == 0);
    assert(argmax({-1, -2, -3, -0.5}) == 3);

    // ===== matvec =====
    {
        std::vector<double> M = {1, 2, 3, 4, 5, 6};  // 2×3
        auto r = matvec(M, 2, 3, {1, 1, 1});
        assert(r.size() == 2);
        assert(std::fabs(r[0] -  6.0) < 1e-9);
        assert(std::fabs(r[1] - 15.0) < 1e-9);
    }
    {
        std::vector<double> M = {1, 2, 3, 4, 5, 6};
        auto r = matvec(M, 2, 3, {1, 0, 0});
        assert(std::fabs(r[0] - 1.0) < 1e-9);
        assert(std::fabs(r[1] - 4.0) < 1e-9);
    }
    {
        std::vector<double> M = {1, 2, 3, 4, 5, 6};
        auto r = matvec(M, 2, 3, {0, 0, 1});
        assert(std::fabs(r[0] - 3.0) < 1e-9);
        assert(std::fabs(r[1] - 6.0) < 1e-9);
    }
    {
        std::vector<double> M = {1, 2, 3, 4, 5, 6};
        auto r = matvec(M, 2, 3, {1, 2, 3});
        assert(std::fabs(r[0] - 14.0) < 1e-9);
        assert(std::fabs(r[1] - 32.0) < 1e-9);
    }
    {
        // 2×2 identity matrix
        std::vector<double> I = {1, 0, 0, 1};
        auto r = matvec(I, 2, 2, {5, 7});
        assert(std::fabs(r[0] - 5.0) < 1e-9);
        assert(std::fabs(r[1] - 7.0) < 1e-9);
    }
    std::cout << "matvec passed!" << std::endl;

    // ===== matmul =====
    {
        // 2×3 times 3×2  →  2×2
        std::vector<double> A = {1, 2, 3, 4, 5, 6};
        std::vector<double> B = {1, 2, 3, 4, 5, 6};
        auto r = matmul(A, 2, 3, B, 3, 2);
        assert(r.size() == 4);
        assert(std::fabs(r[0] - 22.0) < 1e-9);
        assert(std::fabs(r[1] - 28.0) < 1e-9);
        assert(std::fabs(r[2] - 49.0) < 1e-9);
        assert(std::fabs(r[3] - 64.0) < 1e-9);
    }
    {
        // identity * B = B
        std::vector<double> I = {1, 0, 0, 1};
        std::vector<double> B = {1, 2, 3, 4};
        auto r = matmul(I, 2, 2, B, 2, 2);
        assert(r.size() == 4);
        assert(std::fabs(r[0] - 1.0) < 1e-9);
        assert(std::fabs(r[1] - 2.0) < 1e-9);
        assert(std::fabs(r[2] - 3.0) < 1e-9);
        assert(std::fabs(r[3] - 4.0) < 1e-9);
    }
    {
        // 1×1 times 1×1
        std::vector<double> A = {2};
        std::vector<double> B = {3};
        auto r = matmul(A, 1, 1, B, 1, 1);
        assert(r.size() == 1);
        assert(std::fabs(r[0] - 6.0) < 1e-9);
    }
    std::cout << "matmul passed!" << std::endl;

    // ===== relu =====
    {
        auto r = relu({-1, 0, 1, -2.5, 3});
        assert(r.size() == 5);
        assert(std::fabs(r[0] - 0.0) < 1e-9);
        assert(std::fabs(r[1] - 0.0) < 1e-9);
        assert(std::fabs(r[2] - 1.0) < 1e-9);
        assert(std::fabs(r[3] - 0.0) < 1e-9);
        assert(std::fabs(r[4] - 3.0) < 1e-9);
    }
    {
        auto r = relu({});
        assert(r.empty());
    }
    {
        auto r = relu({-1, -2, -3});
        assert(r.size() == 3);
        assert(std::fabs(r[0] - 0.0) < 1e-9);
        assert(std::fabs(r[1] - 0.0) < 1e-9);
        assert(std::fabs(r[2] - 0.0) < 1e-9);
    }
    {
        auto r = relu({1, 2, 3});
        assert(std::fabs(r[0] - 1.0) < 1e-9);
        assert(std::fabs(r[1] - 2.0) < 1e-9);
        assert(std::fabs(r[2] - 3.0) < 1e-9);
    }
    std::cout << "relu passed!" << std::endl;

    // ===== softmax =====
    {
        // uniform input → uniform output
        auto r = softmax({0, 0, 0});
        assert(r.size() == 3);
        assert(std::fabs(r[0] - 1.0/3.0) < 1e-9);
        assert(std::fabs(r[1] - 1.0/3.0) < 1e-9);
        assert(std::fabs(r[2] - 1.0/3.0) < 1e-9);
        double s = 0; for (double x : r) s += x;
        assert(std::fabs(s - 1.0) < 1e-9);
    }
    {
        //checking if overflow occurs.
        auto r = softmax({1000, 1000, 1000});
        assert(r.size() == 3);
        assert(std::fabs(r[0] - 1.0/3.0) < 1e-9);
        assert(std::fabs(r[1] - 1.0/3.0) < 1e-9);
        assert(std::fabs(r[2] - 1.0/3.0) < 1e-9);
    }
    {
        auto r = softmax({1, 2, 3});
        assert(r.size() == 3);
        assert(std::fabs(r[0] - 0.09003057) < 1e-6);
        assert(std::fabs(r[1] - 0.24472847) < 1e-6);
        assert(std::fabs(r[2] - 0.66524096) < 1e-6);
        double s = 0; for (double x : r) s += x;
        assert(std::fabs(s - 1.0) < 1e-9);
    }
    {
        // single element softmax → [1.0]
        auto r = softmax({7});
        assert(r.size() == 1);
        assert(std::fabs(r[0] - 1.0) < 1e-9);
    }
    {
        auto r = softmax({});
        assert(r.empty());
    }
    std::cout << "softmax passed!" << std::endl;

    std::cout << "All tests passed!" << std::endl;
    return 0;
}
