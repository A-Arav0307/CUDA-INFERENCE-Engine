# CUDA Inference Engine

a small c++/cuda inference engine i'm building from scratch to learn how ml inference actually works under the hood instead of treating `model(x)` as a black box. plan is to train an mlp on mnist in pytorch, export the weights, then run inference in c++ first and then in cuda.

## status

cpu math and a 2-layer mlp forward pass are working on hardcoded weights. next step is splitting into headers with a cmake build.

- [x] cpu math: sum_vector, dot_product, vector_add, argmax, matvec, matmul, relu, softmax
- [x] 2-layer mlp forward pass on hardcoded weights
- [ ] split into headers + cmake build
- [ ] pytorch training + weight export, c++ load, verify predictions match
- [ ] cuda kernels for vectoradd, relu, matmul
- [ ] tiled matmul w/ shared memory, batched inference
- [ ] benchmarks vs pytorch, cpu, naive cuda, optimized cuda

## build & run

```
g++ -std=c++17 -Wall -Wextra -O2 cpu_math.cpp -o cpu_math
./cpu_math
```

if everything works:

```
matvec passed!
matmul passed!
relu passed!
softmax passed!
predict passed!
All tests passed!
```

## notes

matrices are flat 1d `std::vector<double>`, row-major, with `rows`/`cols` passed alongside. picked over `vector<vector<double>>` so the cuda port later doesn't need a memory layout rewrite.

softmax subtracts the max before `exp()` so `softmax({1000, 1000, 1000})` doesn't overflow.
