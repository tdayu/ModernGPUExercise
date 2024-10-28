#include <scan.cuh>
#include <benchmark.cuh>
#include <random>
#include <array>


// NT = number of threads
// VT = values per thread
// NV = number of values per block
template <typename T, unsigned NT, unsigned VT>
void benchmark_scan(
    const std::vector<T> host_input,
    const unsigned number_of_elements,
    std::vector<T>& host_output)
{
    constexpr unsigned NV = NT * VT;
    std::vector <unsigned> number_of_blocks;

    unsigned launch_blocks = ( number_of_elements + NV - 1 ) / NV;
    number_of_blocks.push_back(launch_blocks);
    while (launch_blocks > 1) {
        launch_blocks = ( launch_blocks + NV - 1 ) / NV;
        number_of_blocks.push_back(launch_blocks);
    }
    const unsigned number_of_sweeps = number_of_blocks.size() - 1;

    T * dev_input;
    T * dev_output;
    std::vector <T*> dev_spines(number_of_sweeps);

    auto preprocess = [number_of_sweeps=number_of_sweeps, number_of_elements=number_of_elements, &dev_input, &dev_output, &dev_spines, &number_of_blocks, &host_input](){
        CUDA_CALL(cudaSetDevice(0));
        CUDA_CALL(cudaMalloc(&dev_input, number_of_elements * sizeof(T)));
        CUDA_CALL(cudaMalloc(&dev_output, (number_of_elements + 1) * sizeof(T)));
        for (size_t i = 0; i < number_of_sweeps; i++){
            CUDA_CALL(cudaMalloc(&(dev_spines[i]), number_of_blocks[i] * sizeof(T)));
        }
        CUDA_CALL(cudaMemcpy(dev_input, host_input.data(), number_of_elements * sizeof(T), cudaMemcpyHostToDevice));
    };

    auto process = [number_of_sweeps=number_of_sweeps, number_of_elements=number_of_elements, &dev_input, &dev_spines, &dev_output, &number_of_blocks](){
        Scan::launch_kernels<T, NT, VT>(
          dev_input,
          number_of_elements,
          number_of_sweeps,
          number_of_blocks.data(),
          dev_spines.data(),
          dev_output);
    };

    auto postprocess = [number_of_elements=number_of_elements, &dev_input, &dev_output, &dev_spines, &host_output] (){
        CUDA_CALL(cudaMemcpy(host_output.data(), dev_output, (number_of_elements + 1) * sizeof(T), cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaFree(dev_input));
        CUDA_CALL(cudaFree(dev_output));
        for (unsigned i = 0; i < dev_spines.size(); i++){
            CUDA_CALL(cudaFree(dev_spines[i]));
        }
    };

    constexpr unsigned repetitions = 1000;
    float timing = Benchmark::benchmark(preprocess, process, postprocess, repetitions);
    float seconds = timing / 1000;
    float elements_per_second = 1e-9 * static_cast<float>(repetitions) * number_of_elements / seconds;
    float bandwidth = elements_per_second * sizeof(T);

    printf("Number of elements: %u, Timing (ms): %.3f, Billions of elements per second: %.2f, Bandwidth (GB/s): %.0f\n", number_of_elements, timing, elements_per_second, bandwidth);
}

int main(int argc, char** argv) {
    std::mt19937 gen(1234);
    std::uniform_int_distribution<> prefix_sum_distribution(0, 32);

    constexpr unsigned NT = 256;
    constexpr unsigned VT = 4;

    const std::array<unsigned, 15> number_of_inputs{
        1000,
        2000,
        5000,
        10000,
        20000,
        50000,
        100000,
        200000,
        500000,
        1000000,
        2000000,
        5000000,
        10000000,
        20000000,
        50000000
    };

    constexpr unsigned max_size = 50000000;
    std::vector<unsigned> host_input(max_size);
    std::vector<unsigned> host_output(max_size + 1);
    std::vector<unsigned> host_cpu_output(max_size);

    for (size_t i = 0; i < max_size; i++){
        host_input[i] = prefix_sum_distribution(gen);
    }

    for (auto number_of_elements : number_of_inputs){
        benchmark_scan<unsigned, NT, VT>(host_input, number_of_elements, host_output);
    }

    // // for (size_t i = 0; i < number_of_elements; i += NT * VT){
    // //     auto begin = host_input.begin() + i;
    // //     auto end   = host_input.begin() + min(host_input.end() - host_input.begin(), i + NT * VT);
    // //     auto output_iter = host_cpu_output.begin() + i;
    // //     std::exclusive_scan(begin, end, output_iter, 0);
    // // }
    // std::exclusive_scan(host_input.begin(), host_input.end(), host_cpu_output.begin(), 0);

    // unsigned nprint = 0;
    // for (size_t i = 0; i < number_of_elements; i++){
    //     if (host_output[i] != host_cpu_output[i]) {
    //         printf(
    //         "Wrong values! Index: %u, GPU: %u, CPU: %u, Previous GPU: %u, Previous CPU: %u\n",
    //         i, host_output[i], host_cpu_output[i], host_output[i - 1], host_cpu_output[i - 1]);
    //         nprint++;
    //     }
    //     if (nprint >= 512) break;
    // }

    // printf("Total: GPU: %u, CPU: %u\n", host_output[number_of_elements], std::reduce(host_input.begin(), host_input.end()));
}
