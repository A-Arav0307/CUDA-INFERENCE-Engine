#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while (0)

__global__ void matvec_kernel(const float* M, const float* v, float* out, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows) return;
    float sum = 0.0f;
    for (int j = 0; j < cols; ++j) {
        sum += M[i * cols + j] * v[j];
    }
    out[i] = sum;
}

__global__ void relu_kernel(float* v, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= n) return;

    if (v[index] < 0.0f) {
        v[index] = 0.0f;
    }
}

__global__ void bias_add_kernel(float* v, const float* bias, int n) {
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    if (index >= n) return;
    v[index] += bias[index];
}

__global__ void batched_matmul_kernel(
    const float* inputs, const float* W, float* outputs,
    int batch_size, int rows, int cols
) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    if (row >= rows) return;
    if (batch_idx >= batch_size) return;

    for (int j = 0; j < cols; ++j) {
        sum += W[row * cols + j] * inputs[batch_idx * cols + j];

    }
    outputs[batch_idx * rows + row] = sum;


}
// created because image dimensions are constant, so no need to reallocate and Memcpy memory from host to device and vice versa each time,
//results in 100x speedup

//w1 is the weight matrix that connects input layer to hidden layer and maps 784 inputs to 128 hidden neurons (128 x 784) matrix

struct DeviceWeights {
    float* d_W1; float* d_b1;
    float* d_W2; float* d_b2;
    int n_in, n_hidden, n_out;
};

DeviceWeights upload_weights(const float* h_W1, const float* h_b1,
    int n_hidden, int n_in, const float* h_W2, const float* h_b2, int n_out) {
        DeviceWeights w;
        w.n_in = n_in; w.n_hidden = n_hidden; w.n_out = n_out;
        cudaMalloc(&w.d_W1, n_hidden * n_in * sizeof(float));
        cudaMalloc(&w.d_b1, n_hidden * sizeof(float));
        cudaMalloc(&w.d_W2, n_out * n_hidden * sizeof(float));
        cudaMalloc(&w.d_b2, n_out * sizeof(float));
        cudaMemcpy(w.d_W1, h_W1, n_hidden * n_in * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(w.d_b1, h_b1, n_hidden * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(w.d_W2, h_W2, n_out * n_hidden * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(w.d_b2, h_b2, n_out * sizeof(float), cudaMemcpyHostToDevice);
        return w;
    }


int predict_cuda(
    const float* d_input, int n_in,
    const float* d_W1, const float* d_b1, int n_hidden,
    const float* d_W2, const float* d_b2, int n_out
) {
    // int index = threadIdx.x + blockDim.x * blockIdx.x;
    float* d_hidden = nullptr;
    float* d_out = nullptr;

    cudaMalloc(&d_hidden, n_hidden * sizeof(float));
    cudaMalloc(&d_out, n_out * sizeof(float));

    //matvec layer 1
        //calculate threads and blocks
    int threadsPerBlock = 256;
    int blocks_hidden = (128 + threadsPerBlock - 1) / threadsPerBlock; //ceiling division formula for blocks

    matvec_kernel<<<blocks_hidden, threadsPerBlock>>>(d_W1, d_input, d_hidden, 128, 784);
    bias_add_kernel<<<blocks_hidden, threadsPerBlock>>>(d_hidden, d_b1, n_hidden);
    relu_kernel<<<blocks_hidden, threadsPerBlock>>>(d_hidden, n_hidden);
    int blocks_out = (n_out + threadsPerBlock - 1) / threadsPerBlock;
    matvec_kernel<<<blocks_out, threadsPerBlock>>>(d_W2, d_hidden, d_out, n_out, n_hidden);
    bias_add_kernel<<<blocks_out, threadsPerBlock>>>(d_out, d_b2, n_out);

    //array of 10 ints
    float h_out[10];
    //free memory
    cudaMemcpy(h_out, d_out, n_out * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_hidden);
    cudaFree(d_out);

    //h_out is length 10, representing numbers 0 - 9
    int max_idx = 0;
    float max_val = h_out[0];
    for (int j = 1; j < n_out; ++j) {
        if (h_out[j] > max_val) {
            max_val = h_out[j];
            max_idx = j;
        }
    }
    //returns highest probability number
    return max_idx;

}

// int main() {
//     float h_W[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
//     float h_inputs[] = {1, 0, 1, 0, 0, 1, 0, 1};
//     float h_outputs[6];

//     float* d_W = nullptr;
//     float* d_inputs = nullptr;
//     float* d_outputs = nullptr;

//     cudaMalloc(&d_W, 12 * sizeof(float));
//     cudaMalloc(&d_inputs, 8 * sizeof(float));
//     cudaMalloc(&d_outputs, 6 * sizeof(float));

//     cudaMemcpy(d_W, h_W, 12 * sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_inputs, h_inputs, 8 * sizeof(float), cudaMemcpyHostToDevice);

//     dim3 blockDim(2, 3);
//     dim3 gridDim(1, 1);

//     batched_matmul_kernel<<<gridDim, blockDim>>>(d_inputs, d_W, d_outputs, 2, 3, 4);
//     cudaMemcpy(h_outputs, d_outputs, 6 * sizeof(float), cudaMemcpyDeviceToHost);


// for (int i = 0; i < 6; ++i) {
//     printf("h_outputs[%d] = %f\n", i, h_outputs[i]);

// }
// }
