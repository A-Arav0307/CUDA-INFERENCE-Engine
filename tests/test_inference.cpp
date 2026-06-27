#include <cmath>
#include <cstdio>
#include "loader.h"
#include "mlp.h"
#include <vector>
#include <cassert>
#include <fstream>
#include <iostream>
#include <span>

int main() {
    // test for cpp loader
    auto b1 = load_weights("scripts/weights/b1.bin", 128);
    assert(b1.size() == 128);
    assert(std::fabs(b1[0] - (-0.020934427)) < 1e-5);

    auto b2 = load_weights("scripts/weights/b2.bin", 10);
    assert(b2.size() == 10);
    assert(std::fabs(b2[0] - (-0.1982307)) < 1e-5);

    auto W1 = load_weights("scripts/weights/W1.bin", 128 * 784);
    assert(W1.size() == 128 * 784);
    assert(std::fabs(W1[0] - 0.004421954) < 1e-5);

    auto W2 = load_weights("scripts/weights/W2.bin", 10 * 128);
    assert(W2.size() == 10 * 128);
    assert(std::fabs(W2[0] - (-0.037437834)) < 1e-5);

    auto images = load_weights("scripts/weights/test_images.bin", 100*784);
    
    std::ifstream file("scripts/weights/test_labels.bin", std::ios::binary);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open labels file!" << std::endl;
        return 1; 
    }
    std::vector<int64_t> image_labels(100);
    file.read(reinterpret_cast<char*>(image_labels.data()), 100 * sizeof(int64_t));
    size_t correct = 0;
    size_t pytorch_matches = 0;



    std::ifstream pytorch_file("scripts/weights/test_preds_pytorch.bin", std::ios::binary);

    if (!pytorch_file.is_open()) {
        std::cerr << "Error: Could not open labels file!" << std::endl;
        return 1; 
    }
    std::vector<int64_t> pytorch_data(100);
    pytorch_file.read(reinterpret_cast<char*>(pytorch_data.data()), 100 * sizeof(int64_t));




   for (size_t i = 0; i < 100; ++i) {
        size_t start_idx = i * 784;
        size_t end_idx = (i + 1) * 784;
        std::vector<double> chunk(images.begin() + start_idx, images.begin() + end_idx);

        size_t predicted_value = predict(chunk, W1, 128, 784, b1, W2, 10, b2); 
        if (predicted_value == (size_t)image_labels[i]) correct += 1;
        if (predicted_value == (size_t)pytorch_data[i]) pytorch_matches += 1;
    }

    std::cout << correct << "/100 correct" << std::endl;
    std::cout << pytorch_matches << "/100 match PyTorch predictions" << std::endl;

}

        