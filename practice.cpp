#include <iostream>
#include <vector>

double dot_product(const std::vector<double>& a, const std::vector<double>& b) {
    double result = 0.0;

    for (int i = 0; i < a.size(); i++) {
        result += a[i] * b[i];
    }

    return result;
}

int main() {
    std::vector<double> x = {1.0, 2.0, 3.0};
    std::vector<double> y = {4.0, 5.0, 6.0};

    std::cout << "Dot product: " << dot_product(x, y) << std::endl;

    return 0;
}