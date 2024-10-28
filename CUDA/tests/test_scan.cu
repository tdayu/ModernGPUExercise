#include <random>
#include <vector>
#include <algorithm>

#include "scan.cuh"
#include "print.h"

int main(int argc, char** argv) {
    std::mt19937 gen(1234);
    std::uniform_int_distribution<> prefix_sum_distribution(0, 32);

    constexpr unsigned NT = 512;
    constexpr unsigned VT = 3;

    const size_t number_of_elements = (argc > 1) ? std::stol(argv[1]) : 1024 * 1024 * 16;
    std::vector<unsigned> host_input(number_of_elements);
    std::vector<unsigned> host_output(number_of_elements + 1);
    std::vector<unsigned> host_cpu_output(number_of_elements);

    for (size_t i = 0; i < number_of_elements; i++){
        host_input[i] = prefix_sum_distribution(gen);
    }

    Scan::scan<unsigned, NT, VT>(host_input, number_of_elements, host_output);
    std::exclusive_scan(host_input.begin(), host_input.end(), host_cpu_output.begin(), 0);

    print_comparison(host_cpu_output, host_output, 10, 200);
    print_if_mismatch(host_cpu_output, host_output, 100);
}
