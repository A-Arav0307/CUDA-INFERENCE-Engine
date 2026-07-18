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

__global__ void myKernel() {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    printf("Hello from block %d, thread %d (Global ID: %d)\n", blockIdx.x, threadIdx.x, index);
}

int main(){
    myKernel<<<2, 4>>>();
     CUDA_CHECK(cudaGetLastError());
     CUDA_CHECK(cudaDeviceSynchronize());
}