#include "loader.h"
#include <iostream>
#include <vector> 
#include <fstream>
#include <stdexcept> 

std::vector<double> load_weights(const std::string& path, size_t n) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open weights file: " + path); 
    }
    std::vector<double> final_list(n, 0.0);
    std::vector<float> buf(n, 0.0f); 
    size_t total_bytes = n * sizeof(float);
    

    if (!file.read(reinterpret_cast<char*>(buf.data()), total_bytes)) {
        throw std::runtime_error("Failed to read the requested number of bytes.");
    }

    for (size_t i=0; i < n; ++i){
        final_list[i] = buf[i]; 
    }


    return final_list; 
}   