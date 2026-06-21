# CUDA Inference Engine

a small c++/cuda inference engine i'm building from scratch to learn how ml inference actually works under the hood instead of treating `model(x)` as a black box. plan is to train an mlp on mnist in pytorch, export the weights, then run inference in c++ first and then in cuda.

## status

cpu math and a 2-layer mlp forward pass are working. project is now split into a proper multi-file layout with separate headers, source, tests, and a cmake build. next step is loading real weights exported from pytorch.

- [x] cpu math: sum_vector, dot_product, vector_add, argmax, matvec, matmul, relu, softmax
- [x] 2-layer mlp forward pass on hardcoded weights
- [x] split into headers + cmake build (`include/`, `src/`, `tests/`, CMakeLists.txt)
- [ ] pytorch training + weight export, c++ load, verify predictions match
- [ ] cuda kernels for vectoradd, relu, matmul
- [ ] tiled matmul w/ shared memory, batched inference
- [ ] benchmarks vs pytorch, cpu, naive cuda, optimized cuda

## build & run

```
cmake -S . -B build
cmake --build build
./build/final_test
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

gradient descent works by treating the loss as a hilly landscape and rolling downhill. the gradient tells you the slope at your current weights — how much each weight contributed to the loss. you subtract it: `weight = weight - lr × gradient`. do this thousands of times and the weights converge to values that minimize loss. backpropagation (`loss.backward()`) computes those gradients efficiently by working backwards through the network using the chain rule. `optimizer.step()` then applies the update.

learning rate controls step size. too small (e.g. sgd at `lr=0.001`) and the weights barely move — slow convergence. too large and you overshoot the minimum. `lr=0.1` is a typical starting point for sgd on mnist. adam adapts the lr per parameter automatically, making it less sensitive to the initial choice.
