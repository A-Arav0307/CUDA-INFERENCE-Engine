cmake_minimum_required(VERSION 3.15)
project(cuda_net LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

add_library(inference_core STATIC src/cpu_math.cpp src/mlp.cpp src/loader.cpp) 
add_executable(final_test tests/test_main.cpp)
add_executable(inference_test tests/test_inference.cpp)
target_link_libraries(final_test PRIVATE inference_core)
target_link_libraries(inference_test PRIVATE inference_core)

target_include_directories(inference_core PUBLIC include)
target_compile_options(inference_core PRIVATE -Wall -Wextra -O2)

