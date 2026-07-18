#include <iostream>
#include <vector>
#include <cstdint>
#include <fstream>
#include "loader.h"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while (0)

int predict_cuda(
    const float* h_input, int n_in,
    const float* h_W1, const float* h_b1, int n_hidden,
    const float* h_W2, const float* h_b2, int n_out
);

//convers double values to floats
std::vector<float> to_float(const std::vector<double>& vec) {
    return std::vector<float>(vec.begin(), vec.end());
}

int main() {
    auto W1_double = load_weights("scripts/weights/W1.bin", 128 * 784);
    auto b1_double = load_weights("scripts/weights/b1.bin", 128);
    auto W2_double = load_weights("scripts/weights/W2.bin", 10 * 128);
    auto b2_double = load_weights("scripts/weights/b2.bin", 10);
    auto images_double = load_weights("scripts/weights/test_images.bin", 100 * 784);

    std::ifstream labels_file("scripts/weights/test_labels.bin", std::ios::binary);
if (!labels_file.is_open()) {
    std::cerr << "Error: Could not open labels file!" << std::endl;
    return 1;
}
std::vector<int64_t> labels(100);
labels_file.read(reinterpret_cast<char*>(labels.data()), 100 * sizeof(int64_t));

std::ifstream pytorch_file("scripts/weights/test_preds_pytorch.bin", std::ios::binary);
if (!pytorch_file.is_open()) {
    std::cerr << "Error: Could not open PyTorch predictions file!" << std::endl;
    return 1;
}
std::vector<int64_t> cpu_preds(100);
pytorch_file.read(reinterpret_cast<char*>(cpu_preds.data()), 100 * sizeof(int64_t));

    //to be used by the gpu
    auto W1 = to_float(W1_double);
    auto b1 = to_float(b1_double);
    auto W2 = to_float(W2_double);
    auto b2 = to_float(b2_double);
    auto images = to_float(images_double);

    int cuda_correct = 0;
    int cpu_match = 0;

    for (int i = 0; i < 100; ++i) {
        const float* current_img = &images[i * 784]; //finds where image starts because matrix is 1-dimensional
        int target_label = labels[i];
        int expected_cpu_pred = cpu_preds[i];
        int cuda_pred = predict_cuda(current_img, 784, W1.data(), b1.data(), 128, W2.data(), b2.data(), 10);


        if (cuda_pred == target_label) cuda_correct++;
        if (cuda_pred == expected_cpu_pred) cpu_match++;
    }

    std::cout << cuda_correct << "/100 correct (CUDA)" << std::endl;
    std::cout << cpu_match << "/100 match CPU predictions" << std::endl;
}
