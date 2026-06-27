#include "cpu_math.h"
#include "mlp.h"
#include "loader.h"
#include <iostream> 
#include <cassert> 
#include <cmath>

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

    // ===== predict =====
    {
        std::vector<double> input = {1, 2, 3};
        std::vector<double> W1 = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        std::vector<double> b1 = {0, 0, 0, 0};
        std::vector<double> W2 = {0, 0, 0, 0, 0, 0, 0, 0};
        std::vector<double> b2 = {0, 0};
        assert(predict(input, W1, 4, 3, b1, W2, 2, b2) == 0);
    }
    {
        std::vector<double> input = {1, 2, 3};
        std::vector<double> W1 = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        std::vector<double> b1 = {0, 0, 0, 0};
        std::vector<double> W2 = {0, 0, 0, 0, 0, 0, 0, 0};
        std::vector<double> b2 = {0.1, 0.9};
        assert(predict(input, W1, 4, 3, b1, W2, 2, b2) == 1);
    }
    {
        std::vector<double> input = {1, 2, 3};
        std::vector<double> W1 = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        std::vector<double> b1 = {0, 0, 0, 0};
        std::vector<double> W2 = {0, 0, 0, 0, 0, 0, 0, 0};
        std::vector<double> b2 = {2.0, -1.0};
        assert(predict(input, W1, 4, 3, b1, W2, 2, b2) == 0);
    }
    {
        std::vector<double> input = {1, 1};
        std::vector<double> W1 = {1, 0, 0, 1};
        std::vector<double> b1 = {0, 0};
        std::vector<double> W2 = {1, 0, 1, 1};
        std::vector<double> b2 = {0, 0};
        assert(predict(input, W1, 2, 2, b1, W2, 2, b2) == 1);
    }
    {
        // relu clips the negative hidden unit
        std::vector<double> input = {1};
        std::vector<double> W1 = {1, -1};
        std::vector<double> b1 = {0, 0};
        std::vector<double> W2 = {1, 2, 0, 0};
        std::vector<double> b2 = {0, 0};
        assert(predict(input, W1, 2, 1, b1, W2, 2, b2) == 0);
    }
    {
        // b1 shifts the pre-activations
        std::vector<double> input = {0};
        std::vector<double> W1 = {5, 5};
        std::vector<double> b1 = {1, -5};
        std::vector<double> W2 = {0, 1, 1, 0};
        std::vector<double> b2 = {0, 0};
        assert(predict(input, W1, 2, 1, b1, W2, 2, b2) == 1);

        
    }

    {
        //test for cpp loader
        
    }


    std::cout << "predict passed!" << std::endl;

    std::cout << "All tests passed!" << std::endl;
    return 0;
}
