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

int predict_cuda(
    const float* h_input, int n_in,
    const float* h_W1, const float* h_b1, int n_hidden,
    const float* h_W2, const float* h_b2, int n_out
) {
    // int index = threadIdx.x + blockDim.x * blockIdx.x;

    float* d_in = nullptr;
    float* d_W1 = nullptr;
    float* d_W2 = nullptr;
    float* d_b1 = nullptr;
    float* d_b2 = nullptr;
    float* d_hidden = nullptr;
    float* d_out = nullptr;

    cudaMalloc(&d_in, n_in * sizeof(float));
    cudaMalloc(&d_W1, n_hidden * n_in * sizeof(float));
    cudaMalloc(&d_b1, n_hidden * sizeof(float));
    cudaMalloc(&d_hidden, n_hidden * sizeof(float));
    cudaMalloc(&d_W2, n_out * n_hidden * sizeof(float));
    cudaMalloc(&d_b2, n_out * sizeof(float));
    cudaMalloc(&d_out, n_out * sizeof(float));

    //writing data from cpu to gpu
    cudaMemcpy(d_in, h_input, n_in * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W1, h_W1, n_hidden * n_in * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b1, h_b1, n_hidden * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W2, h_W2, n_out * n_hidden * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b2, h_b2, n_out * sizeof(float), cudaMemcpyHostToDevice);

    //matvec layer 1
        //calculate threads and blocks
    int threadsPerBlock = 256;
    int blocks_hidden = (128 + threadsPerBlock - 1); //ceiling division formula for blocks

    matvec_kernel<<<blocks_hidden, threadsPerBlock>>>(d_W1, d_in, d_hidden, 128, 784);
    bias_add_kernel<<<blocks_hidden, threadsPerBlock>>>(d_hidden, d_b1, n_hidden);
    relu_kernel<<<blocks_hidden, threadsPerBlock>>>(d_hidden, n_hidden);
    int blocks_out = (n_out + threadsPerBlock - 1) / threadsPerBlock;
    matvec_kernel<<<blocks_out, threadsPerBlock>>>(d_W2, d_hidden, d_out, n_out, n_hidden);
    bias_add_kernel<<<blocks_out, threadsPerBlock>>>(d_out, d_b2, n_out);

    //array of 10 ints
    float h_out[10];
    //free memory
    cudaMemcpy(h_out, d_out, n_out * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_W1);
    cudaFree(d_b1);
    cudaFree(d_hidden);
    cudaFree(d_W2);
    cudaFree(d_b2);
    cudaFree(d_out);

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
//     // MATVEC KERNEL /////////////////
//     float h_M[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
//     float h_v[] = {1.0f, 1.0f, 1.0f};
//     float h_out[] = {0.0f, 0.0f};

//     float* d_M = nullptr;
//     float* d_v = nullptr;
//     float* d_out = nullptr;

//     cudaMalloc(&d_M, 2 * 3 * sizeof(float));
//     cudaMalloc(&d_v, 3 * sizeof(float));
//     cudaMalloc(&d_out, 2 * sizeof(float));

//     cudaMemcpy(d_M, h_M, 2 * 3 * sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_v, h_v, 3 * sizeof(float), cudaMemcpyHostToDevice);

//     matvec_kernel<<<1, 256>>>(d_M, d_v, d_out, 2, 3);
//     cudaDeviceSynchronize();

//     cudaMemcpy(h_out, d_out, 2 * sizeof(float), cudaMemcpyDeviceToHost);
//     printf("--- GPU Results ---\n");
//     for (int i = 0; i < 2; ++i) {
//         printf("h_out[%d] = %f\n", i, h_out[i]);
//     }
//     ///////////////////////////////

//     //CUDA RELU AND BIAS ADD KERNELS////////////
//     float h_relu[] = {1.2f, -3.4f, 2.0f, -0.1f, 5.1f};
//     float* d_relu = nullptr;
//     cudaMalloc(&d_relu, 5 * sizeof(float));
//     cudaMemcpy(d_relu, h_relu, 5 * sizeof(float), cudaMemcpyHostToDevice);
//     relu_kernel<<<1, 256>>>(d_relu, 5);
//     cudaDeviceSynchronize();
//     //brings computed answer on gpu back to the cpu
//     cudaMemcpy(h_relu, d_relu, 5 * sizeof(float), cudaMemcpyDeviceToHost);


//     //PREDICT CUDA FN/////
//         //now need to load 100 image dataset and feed into cuda fn//



// }
