#include <iostream>
#include <vector>
#include <cstdint>
#include <fstream>
#include <chrono>
#include "loader.h"
#include "mlp.h"

int predict_cuda(
    const float* d_input, int n_in,
    const float* d_W1, const float* d_b1, int n_hidden,
    const float* d_W2, const float* d_b2, int n_out
);

struct DeviceWeights {
    float* d_W1; float* d_b1;
    float* d_W2; float* d_b2;
    int n_in, n_hidden, n_out;
};

std::vector<double> load_weights(const std::string& path, size_t n);

DeviceWeights upload_weights(const float* h_W1, const float* h_b1, int n_hidden, int n_in, const float* h_W2, const float* h_b2, int n_out);
__global__ void tiled_matmul_kernel(const float* inputs, const float* W, float* outputs, int batch_size, int rows, int cols);
__global__ void batched_bias_add_kernel(float* v, const float* bias, int total_n, int rows);
__global__ void relu_kernel(float* v, int n);

std::vector<float> to_float(const std::vector<double>& vec) {
    return std::vector<float>(vec.begin(), vec.end());
}

int main() {

    //change how many images at a time should be used
    int N = 100;


    auto W1_double = load_weights("scripts/weights/W1.bin", 128 * 784);
    auto b1_double = load_weights("scripts/weights/b1.bin", 128);
    auto W2_double = load_weights("scripts/weights/W2.bin", 10 * 128);
    auto b2_double = load_weights("scripts/weights/b2.bin", 10);
    auto images_double = load_weights("scripts/weights/test_images.bin", 100 * 784);
    std::vector<float> h_out_batched(N * 10, 0.0f);

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

    //to be used by the gpu and other important imports
    auto W1 = to_float(W1_double);
    auto b1 = to_float(b1_double);
    auto W2 = to_float(W2_double);
    auto b2 = to_float(b2_double);
    auto images = to_float(images_double);

    std::vector<float> images_N(N * 784);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < 784; ++j) {
            images_N[i * 784 + j] = images[(i % 100) * 784 + j];
        }
    }

    int cuda_correct = 0;
    int cpu_match = 0;
    DeviceWeights weights = upload_weights(W1.data(), b1.data(), 128, 784, W2.data(), b2.data(), 10);

    //bias_add_kernel tests
    float* d_all_inputs = nullptr;
    float* d_hidden_batched = nullptr;
    float* d_out_batched = nullptr;

    cudaMalloc(&d_all_inputs, N * 784 * sizeof(float));
    cudaMalloc(&d_hidden_batched, N * 128 * sizeof(float));
    cudaMalloc(&d_out_batched, N * 10 * sizeof(float));


    dim3 blockDim(16, 16);
    dim3 gridDim_layer1((N + 15) / 16, (128 + 15) / 16);
    dim3 gridDim_layer2((N + 15) / 16, (10 + 15) / 16);

    int threadsPerBlock = 256;
    int blocks_hidden = (N * 128 + threadsPerBlock - 1) / threadsPerBlock;  // for hidden-layer-sized calls
    int blocks_out = (N * 10 + threadsPerBlock - 1) / threadsPerBlock;


    //timing tests for cpu:

    auto t0 = std::chrono::high_resolution_clock::now();
    int correct = 0;
    int pytorch_matches = 0;

   for (size_t i = 0; i < N; ++i) {
        size_t start_idx = (i % 100) * 784;
        size_t end_idx = start_idx + 784;
        std::vector<double> chunk(images_double.begin() + start_idx, images_double.begin() + end_idx);

        size_t predicted_value = predict(chunk, W1_double, 128, 784, b1_double, W2_double, 10, b2_double);
        if (predicted_value == (size_t) labels[i % 100]) correct += 1;
        if (predicted_value == (size_t) cpu_preds[i % 100]) pytorch_matches += 1;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "Time for CPU run: " << cpu_ms << "ms\n" << std::endl;

    //timing tests for naive CUDA:
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    //throwaway benchmark for one image to absorb cuda one time context-init cost
    float* d_input = nullptr;
    float* test_img = &images[784];
    cudaMalloc(&d_input, 784 * sizeof(float));
    cudaMemcpy(d_input, test_img, 784 * sizeof(float), cudaMemcpyHostToDevice);
    cudaFree(d_input);
    cudaEventRecord(start);

    for (int i = 0; i < N; ++i) {
       const float* current_img = &images_N[i * 784];
       float* d_input = nullptr;
       cudaMalloc(&d_input, 784 * sizeof(float));
       cudaMemcpy(d_input, current_img, 784 * sizeof(float), cudaMemcpyHostToDevice);

       int pred = predict_cuda(d_input, 784, weights.d_W1, weights.d_b1, 128, weights.d_W2, weights.d_b2, 10);
       cudaFree(d_input);


    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float naive_ms = 0;
    cudaEventElapsedTime(&naive_ms, start, stop);
    std::cout << "Time for naive CUDA run: " << naive_ms << "ms" << std::endl;

    //tiled matmul kernels (tile_size = 16) below

        //copying data from cpu to gpu
    cudaEvent_t start_h2d, stop_h2d;
    cudaEventCreate(&start_h2d);
    cudaEventCreate(&stop_h2d);

    cudaEventRecord(start_h2d);
    cudaMemcpy(d_all_inputs, images_N.data(), N * 784 * sizeof(float), cudaMemcpyHostToDevice);
    cudaEventRecord(stop_h2d);
    cudaEventSynchronize(stop_h2d);

    float h2d_ms = 0;
    cudaEventElapsedTime(&h2d_ms, start_h2d, stop_h2d);
    std::cout << "time for H2D transfer: " << h2d_ms << "ms" << std::endl;

    //tiled matmul kernel benchmark
    cudaEvent_t start_compute, stop_compute;
    cudaEventCreate(&start_compute);
    cudaEventCreate(&stop_compute);
    cudaEventRecord(start_compute);
       //2-layer mlp from test_inference_cuda.cu
    tiled_matmul_kernel<<<gridDim_layer1, blockDim>>>(d_all_inputs, weights.d_W1, d_hidden_batched, N, 128, 784);
    batched_bias_add_kernel<<<blocks_hidden, threadsPerBlock>>>(d_hidden_batched, weights.d_b1, N*128, 128);
    relu_kernel<<<blocks_hidden, threadsPerBlock>>>(d_hidden_batched, N*128);
    tiled_matmul_kernel<<<gridDim_layer2, blockDim>>>(d_hidden_batched, weights.d_W2, d_out_batched, N, 10, 128);
    batched_bias_add_kernel<<<blocks_out, threadsPerBlock>>>(d_out_batched, weights.d_b2, N*10, 10);

    cudaEventRecord(stop_compute);
    cudaEventSynchronize(stop_compute);

    float compute_ms = 0;
    cudaEventElapsedTime(&compute_ms, start_compute, stop_compute);
    std::cout << "Time for compute: " << compute_ms << "ms" << std::endl;

    //copy back result predictions into host

    cudaEvent_t start_outputs, end_outputs;
    cudaEventCreate(&start_outputs);
    cudaEventCreate(&end_outputs);

    cudaEventRecord(start_outputs);
    cudaMemcpy(h_out_batched.data(), d_out_batched, 10 * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(end_outputs);
    cudaEventSynchronize(end_outputs);

    float d2h_ms = 0;
    cudaEventElapsedTime(&d2h_ms, start_outputs, end_outputs);
    std::cout << "time for D2H transfer: " << d2h_ms << "ms" << std::endl;
}
