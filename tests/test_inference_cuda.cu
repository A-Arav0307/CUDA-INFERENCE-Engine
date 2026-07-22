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

__global__ void tiled_matmul_kernel(const float* inputs, const float* W, float* outputs, int batch_size, int rows, int cols);
__global__ void batched_bias_add_kernel(float* v, const float* bias, int total_n, int rows);
__global__ void relu_kernel(float* v, int n);

int predict_cuda(
    const float* h_input, int n_in,
    const float* h_W1, const float* h_b1, int n_hidden,
    const float* h_W2, const float* h_b2, int n_out
);

struct DeviceWeights {
    float* d_W1; float* d_b1;
    float* d_W2; float* d_b2;
    int n_in, n_hidden, n_out;
};

DeviceWeights upload_weights(const float* h_W1, const float* h_b1, int n_hidden, int n_in, const float* h_W2, const float* h_b2, int n_out);

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
    DeviceWeights weights = upload_weights(W1.data(), b1.data(), 128, 784, W2.data(), b2.data(), 10);

    //bias_add_kernel tests
    float* d_all_inputs = nullptr;
    float* d_hidden_batched = nullptr;
    float* d_out_batched = nullptr;

    cudaMalloc(&d_all_inputs, 100 * 784 * sizeof(float));
    cudaMalloc(&d_hidden_batched, 100 * 128 * sizeof(float));
    cudaMalloc(&d_out_batched, 100 * 10 * sizeof(float));

    cudaMemcpy(d_all_inputs, images.data(), 100 * 784 * sizeof(float), cudaMemcpyHostToDevice);


    dim3 blockDim(16, 16);
    dim3 gridDim_layer1((100 + 15) / 16, (128 + 15) / 16);
    dim3 gridDim_layer2((100 + 15) / 16, (10 + 15) / 16);

    int threadsPerBlock = 256;
    int blocks_hidden = (100 * 128 + threadsPerBlock - 1) / threadsPerBlock;  // for hidden-layer-sized calls
    int blocks_out = (100 * 10 + threadsPerBlock - 1) / threadsPerBlock;

    //building 2 layer mlp from cuda kernels
    tiled_matmul_kernel<<<gridDim_layer1, blockDim>>>(d_all_inputs, weights.d_W1, d_hidden_batched, 100, 128, 784);
    batched_bias_add_kernel<<<blocks_hidden, threadsPerBlock>>>(d_hidden_batched, weights.d_b1, 100*128, 128);
    relu_kernel<<<blocks_hidden, threadsPerBlock>>>(d_hidden_batched, 100*128);
    tiled_matmul_kernel<<<gridDim_layer2, blockDim>>>(d_hidden_batched, weights.d_W2, d_out_batched, 100, 10, 128);
    batched_bias_add_kernel<<<blocks_out, threadsPerBlock>>>(d_out_batched, weights.d_b2, 100*10, 10);

    std::vector<float> h_out_batched(100*10, 0.0f);
    cudaMemcpy(h_out_batched.data(), d_out_batched, 100*10*sizeof(float), cudaMemcpyDeviceToHost);
    int max_idx = 0;
    float max_val = h_out_batched[0];

    for (int i = 0; i < 100; ++i) {
    int max_idx = 0;
    float max_val = h_out_batched[i * 10];
        for (int j = 1; j < 10; ++j) {
            if (h_out_batched[i * 10 + j] > max_val) {
                max_val = h_out_batched[i * 10 + j];
                max_idx = j;
            }
        }
    if (max_idx == labels[i]) cuda_correct++;
    if (max_idx == cpu_preds[i]) cpu_match++;
}

    std::cout << cuda_correct << "/100 correct (CUDA)" << std::endl;
    std::cout << cpu_match << "/100 match CPU predictions" << std::endl;
}
