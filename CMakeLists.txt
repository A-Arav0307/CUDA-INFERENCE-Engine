cmake_minimum_required(VERSION 3.18)
project(cuda_net LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

add_library(inference_core STATIC src/cpu_math.cpp src/mlp.cpp src/loader.cpp)

add_executable(final_test tests/test_main.cpp)
add_executable(inference_test tests/test_inference.cpp)
add_executable(test_inference_cuda tests/test_inference_cuda.cu src/cuda_inference.cu)

target_link_libraries(final_test PRIVATE inference_core)
target_link_libraries(inference_test PRIVATE inference_core)
target_link_libraries(test_inference_cuda PRIVATE inference_core)

target_include_directories(inference_core PUBLIC include)
target_compile_options(inference_core PRIVATE -Wall -Wextra -O2)

set_target_properties(test_inference_cuda PROPERTIES
    CUDA_STANDARD 17
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

