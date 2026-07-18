# CUDA Inference Engine

a small c++/cuda inference engine i'm building from scratch to learn how ml inference actually works under the hood instead of treating `model(x)` as a black box. plan is to train an mlp on mnist in pytorch, export the weights, then run inference in c++ first and then in cuda.

## status

cpu and gpu inference pipelines are both complete and matching. trained an mlp on mnist in pytorch, exported weights as raw float32 binaries, built a c++ loader, ran inference on 100 test images on the cpu, then reimplemented the same forward pass as cuda kernels and confirmed identical predictions on the gpu. next step is optimizing and benchmarking the cuda implementation.

- [x] cpu math: sum_vector, dot_product, vector_add, argmax, matvec, matmul, relu, softmax
- [x] 2-layer mlp forward pass on hardcoded weights
- [x] split into headers + cmake build (`include/`, `src/`, `tests/`, CMakeLists.txt)
- [x] pytorch training + weight export (`scripts/train.py`, `scripts/export_weights.py`)
- [x] c++ weight loader reading raw float32 binaries (`src/loader.cpp`)
- [x] full c++ inference on 100 mnist test images — 99/100 correct, 100/100 match pytorch
- [x] cuda kernels for matvec, relu, bias add, wired into a full `predict_cuda` pipeline
- [x] gpu inference on 100 mnist test images, matching cpu predictions
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

to run the full inference test (requires exported weights in `scripts/weights/`):

```
./build/inference_test
```

output:

```
99/100 correct
100/100 match PyTorch predictions
```

to run the gpu inference test (requires a cuda-capable gpu):

```
./build/test_inference_cuda
```

output:

```
99/100 correct (CUDA)
99/100 match CPU predictions
```

## notes

matrices are flat 1d `std::vector<double>`, row-major, with `rows`/`cols` passed alongside. picked over `vector<vector<double>>` so the cuda port later doesn't need a memory layout rewrite.

softmax subtracts the max before `exp()` so `softmax({1000, 1000, 1000})` doesn't overflow.

gradient descent works by treating the loss as a hilly landscape and rolling downhill. the gradient tells you the slope at your current weights — how much each weight contributed to the loss. you subtract it: `weight = weight - lr × gradient`. do this thousands of times and the weights converge to values that minimize loss. backpropagation (`loss.backward()`) computes those gradients efficiently by working backwards through the network using the chain rule. `optimizer.step()` then applies the update.

learning rate controls step size. too small (e.g. sgd at `lr=0.001`) and the weights barely move — slow convergence. too large and you overshoot the minimum. `lr=0.1` is a typical starting point for sgd on mnist. adam adapts the lr per parameter automatically, making it less sensitive to the initial choice.
