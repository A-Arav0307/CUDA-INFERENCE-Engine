//#include "loader.h"
#include <iostream>
#include <fstream>
#include <stdexcept> 

int main() {
    //std::count << load_weights()
}

std::vector<double> load_weights(const std::string& path, size_t n) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open weights file: " + path); 
    }
    std::vector<double> final_list(n, 0.0);
    size_t total_bytes = n * sizeof(double);


    if (!file.read(reinterpret_cast<char*>(final_list.data()), total_bytes)) {
        throw std::runtime_error("Failed to read the requested number of bytes.");
    }

    return final_list; 
}   