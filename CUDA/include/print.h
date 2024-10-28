#pragma once

#include <vector>
#include <iostream>
#include <stdio.h>

template<typename T>
void print_array(std::vector<T>& input, unsigned elements_per_row, unsigned limit) {
    unsigned count = 0;
    for (size_t i = 0; i < input.size(); i += elements_per_row){
        printf("%u:", i);
        for (size_t j = 0; j < elements_per_row; j++){
            if (i + j < input.size()) std::cout << "\t" << input[ i + j ];
            count++;
        }
        std::cout << std::endl;
        if (count >= limit) break;
    }
    printf("\n");
}

template<typename T>
void print_comparison(std::vector<T>& CPU, std::vector<T>& GPU, unsigned elements_per_row, unsigned limit){
    assert(CPU.size() == GPU.size());

    unsigned count = 0;
    for (size_t i = 0; i < CPU.size(); i += elements_per_row){
        printf("CPU %u:", i);
        for (size_t j = 0; j < elements_per_row; j++){
            if (i + j < CPU.size()) std::cout << "\t" << CPU[ i + j ];
            count++;
        }
        printf("\n");
        printf("GPU %u:", i);
        for (size_t j = 0; j < elements_per_row; j++){
            if (i + j < GPU.size()) std::cout << "\t" << GPU[ i + j ];
        }
        printf("\n\n");
        if (count >= limit) break;
    }
    printf("\n");
}

template<typename T>
void print_if_mismatch(std::vector<T>& CPU, std::vector<T>& GPU, unsigned limit){
    assert(CPU.size() == GPU.size());

    unsigned count = 0;
    for (size_t i = 0; i < CPU.size(); i++){
        if (CPU[i] != GPU[i]){
            printf("Values mismatch! Index: %lu, CPU: %u, GPU: %u, Previous CPU: %u, Previous GPU: %u\n", i, CPU[i], GPU[i], CPU[i-1], GPU[i-1]);
            count++;
        }
        if (count >= limit) break;
    }
}
