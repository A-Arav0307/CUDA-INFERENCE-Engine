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


int main() {
    return 0;
}
