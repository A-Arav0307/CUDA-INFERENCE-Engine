# Phase 5 Exercises — CUDA Inference

Work through these in order on your Windows machine with the RTX 3060.

---

## Exercise 1 — Set up the CUDA environment

On your Windows machine:

1. Install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (12.x recommended)
2. Verify with `nvcc --version` and `nvidia-smi` in a terminal
3. Make sure your RTX 3060 shows up in `nvidia-smi` output

Create `src/hello_cuda.cu` with a kernel that prints "Hello from thread X" for each thread. Launch it with 1 block of 8 threads.

Before writing: what does `printf` inside a kernel print to? The GPU has no stdout — CUDA has a special mechanism for this. Look it up.

---

## Exercise 2 — Write a CUDA matvec kernel

Create `src/inference_cuda.cu`. Write a CUDA kernel:

```cpp
__global__ void matvec_kernel(const float* M, const float* v, float* out, int rows, int cols);
```

Each thread computes one element of the output vector. Thread `i` computes the dot product of row `i` of M with v.

Questions to answer before writing:
- How do you compute the global thread index inside the kernel?
- What's the guard condition you need to avoid going out of bounds?
- For a (128 × 784) matrix and a 784-element vector, how many threads do you need total?
- If you use 256 threads per block, how many blocks do you need? (Think about ceiling division.)

Write a small test in `main()` that:
1. Creates a known 2×3 matrix and 3-element vector on the CPU
2. Copies them to the GPU
3. Launches the kernel
4. Copies the result back
5. Asserts the output matches what you'd compute by hand

---

## Exercise 3 — Write CUDA ReLU and bias-add kernels

Add two more kernels to `inference_cuda.cu`:

```cpp
__global__ void relu_kernel(float* v, int n);
__global__ void bias_add_kernel(float* v, const float* bias, int n);
```

`relu_kernel` modifies `v` in-place: replace any negative value with 0.  
`bias_add_kernel` adds `bias[i]` to `v[i]` for each element, also in-place.

Why in-place? Think about what that saves compared to writing to a separate output buffer.

Test both with small known inputs before moving on.

---

## Exercise 4 — Write predict_cuda()

Declare and implement:

```cpp
int predict_cuda(
    const float* h_input, int n_in,
    const float* h_W1, const float* h_b1, int n_hidden,
    const float* h_W2, const float* h_b2, int n_out
);
```

`h_` means host (CPU) pointer. This function should:

1. Allocate device memory for input, W1, b1, hidden layer output, W2, b2, output layer
2. Copy all inputs to device
3. Run: matvec(W1, input) → bias_add(b1) → relu → matvec(W2) → bias_add(b2)
4. Copy the 10-element output back to host
5. Return argmax of the output (on the CPU — no need for a kernel just for this)
6. Free all device memory

Note: you don't need a softmax kernel here. For inference, argmax of pre-softmax logits gives the same answer as argmax of softmax outputs. Why?

---

## Exercise 5 — Run the 100-image test on GPU

Create `tests/test_inference_cuda.cu`. It should:

1. Load weights and test data the same way as `test_inference.cpp` (reuse your loader — but convert `double` vectors to `float` before passing to `predict_cuda`)
2. Run `predict_cuda()` on all 100 images
3. Print:
   ```
   X/100 correct (CUDA)
   X/100 match CPU predictions
   ```

The CUDA result should exactly match your Phase 4 CPU result. If it doesn't, the bug is almost certainly in your matvec kernel — check the thread index arithmetic and the bias add ordering.

Add `test_inference_cuda` as an executable in `CMakeLists.txt` using `enable_language(CUDA)`.

---

## Exercise 6 — Wire it all together

From the project root on Windows, you should be able to run:

```
cmake -S . -B build
cmake --build build
./build/test_inference_cuda
```

And see:
```
99/100 correct (CUDA)
99/100 match CPU predictions
```

If your counts disagree between CUDA and CPU, add a debug print inside the loop showing the image index, CPU prediction, and CUDA prediction for any mismatch. The first mismatch will tell you a lot about what's wrong.

---

## Stretch goal

Time your GPU inference vs CPU inference using `cudaEvent_t` for GPU timing and `std::chrono` for CPU timing. For 100 images the GPU will likely be *slower* due to memory transfer overhead — that's expected and worth understanding. How many images would you need to process before the GPU wins?
