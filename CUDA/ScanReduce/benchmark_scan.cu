#include <random>
#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <array>
#include <assert.h>

#define CUDA_CALL(x) {\
    const cudaError_t a=(x);\
    if(a != cudaSuccess) {\
        printf(\
            "\nerror in line:%d CUDAError:%s(err_num=%d)\n",\
            __LINE__,\
            cudaGetErrorString(a),\
            a);\
        cudaDeviceReset();\
        assert(0);\
    }\
}

#define FULL_MASK 0xFFFFFFFF
#define WARP_SIZE 32

enum ScanType {
    Exclusive,
    Inclusive
};

__device__ unsigned int laneid()
{
    unsigned int laneid;
    asm ("mov.u32 %0, %%laneid;" : "=r"(laneid));
    return laneid;
}

template <typename P1, typename P2, typename P3>
float benchmark(
    P1 preprocess,
    P2 process,
    P3 postprocess,
    unsigned repetitions)
{
    preprocess();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (unsigned i = 0; i < repetitions; i++){
        process();
    }
    cudaEventRecord(stop);

    postprocess();

    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    return milliseconds;
}

template<typename T>
__device__ T warp_inclusive_scan(const T x, const unsigned mask = FULL_MASK, const int width = 32){
    T scan(x);

    for (int delta = 1; delta < width; delta *= 2){
        T shuffled = __shfl_up_sync(mask, scan, delta);
        if (laneid() >= delta) scan += shuffled;
    }

    return scan;
}

template<typename T, unsigned NT, unsigned VT>
__device__ void global_to_shared(const unsigned total, const T * global, T * shared){
    constexpr unsigned NV = NT * VT;
    #pragma unroll
    for (int i = 0; i < VT; i++){
        const unsigned global_index = NV * blockIdx.x + NT * i + threadIdx.x;
        const unsigned shared_index = NT * i + threadIdx.x;
        shared[shared_index] = (global_index < total) ? global[global_index] : 0;
    }
    __syncthreads();
}

template<typename T, unsigned VT>
__device__ void shared_to_registers(const T * shared, T * registers){
    #pragma unroll
    for (int i = 0; i < VT; i++){
        registers[i] = shared[VT * threadIdx.x + i];
    }
}

template<typename T, unsigned NT, unsigned VT>
__device__ void shared_to_global(const unsigned total, T * global, T * shared){
    constexpr unsigned NV = NT * VT;
    #pragma unroll
    for (int i = 0; i < VT; i++){
        const unsigned global_index = NV * blockIdx.x + NT * i + threadIdx.x;
        const unsigned shared_index = NT * i + threadIdx.x;
        if (global_index < total) global[global_index] = shared[shared_index];
    }
}

template<typename T, unsigned VT>
__device__ void registers_to_shared(T * shared, const T * registers){
    #pragma unroll
    for (int i = 0; i < VT; i++){
        shared[VT * threadIdx.x + i] = registers[i];
    }
    __syncthreads();
}

template<typename T, unsigned VT>
__device__ void linear_scan(T * values){
    #pragma unroll
    for (int i = 1; i < VT; i++){
        values[i] += values[i-1];
    }    
}

template<typename T, unsigned NT>
__device__ void cta_upsweep(const T x, T * shared_values, T * spine){
    T warp_scanned = warp_inclusive_scan(x);

    // Scan the warp sums
    __syncthreads();
    if (laneid() == WARP_SIZE - 1) shared_values[threadIdx.x / 32] = warp_scanned;
    __syncthreads();

    constexpr unsigned warp_spine_size = ( NT + WARP_SIZE - 1 ) / WARP_SIZE;
    static_assert( warp_spine_size <= 32 );

    const bool predicate = threadIdx.x < warp_spine_size;
    const unsigned mask = __ballot_sync(FULL_MASK, predicate);
    if (predicate){
        T spine_scanned = shared_values[threadIdx.x];
        spine_scanned = warp_inclusive_scan(spine_scanned, mask, warp_spine_size);
        shared_values[threadIdx.x + 1] = spine_scanned;
        if (threadIdx.x == 0) shared_values[0] = 0;
        if (spine != nullptr && threadIdx.x == (warp_spine_size - 1)) spine[blockIdx.x] = spine_scanned;
    }
    __syncthreads();

    T upsweep = warp_scanned + shared_values[threadIdx.x / 32];
    __syncthreads();

    shared_values[threadIdx.x + 1] = upsweep;
    __syncthreads();
}

template<typename T, unsigned NT>
__device__ void cta_shared_upsweep(const T x, T * shared_values, T * spine){
    T warp_scanned = warp_inclusive_scan(x);
    shared_values[threadIdx.x] = warp_scanned;

    T value;
    for (int delta = WARP_SIZE; delta < NT; delta++){
        if (threadIdx.x >= delta) value = shared_values[threadIdx.x - delta];
        __syncthreads();
        if (threadIdx.x >= delta) shared_values[threadIdx.x] += value;
        __syncthreads();
    }

    const bool last_thread = (threadIdx.x == NT - 1);
    if (last_thread) spine[blockIdx.x] = shared_values[threadIdx.x];
}

template<typename T, unsigned VT, ScanType type>
__device__ void cta_downsweep(T * shared_values, T * values){
    if constexpr (type == ScanType::Exclusive) {
        #pragma unroll
        for (int i = VT - 1; i > 0; i--){
            values[i] = values[i - 1];
        }
        values[0] = 0;
    }

    #pragma unroll
    for (int i = 0; i < VT; i++){
        values[i] += shared_values[threadIdx.x];
    }
    __syncthreads();
}

// NT = number of threads
// VT = values per thread
// NV = number of values per block
template <typename T, unsigned NT, unsigned VT, ScanType type>
__global__ void block_scan(
    const unsigned total,
    const T * input,
    T* output,
    T* spine = nullptr)
{
    constexpr unsigned NV = NT * VT;
    constexpr unsigned shared_size = (VT > 1) ? NV : NT + 1;
    __shared__ T shared_values[shared_size]; // shared storage
    T values[VT]; // registers storage

    global_to_shared<T, NT, VT>(total, input, shared_values);
    shared_to_registers<T, VT>(shared_values, values);
    linear_scan<T, VT>(values);
    cta_upsweep<T, NT>(values[VT - 1], shared_values, spine);
    // cta_shared_upsweep<T, NT>(values[VT - 1], shared_values, spine);
    cta_downsweep<T, VT, type>(shared_values, values);
    registers_to_shared<T, VT>(shared_values, values);
    shared_to_global<T, NT, VT>(total, output, shared_values);
}

// NT = number of threads
// VT = values per thread
// NV = number of values per block
template <typename T, unsigned NT, unsigned VT>
__global__ void block_downsweep(
    const unsigned total,
    const T * spine,
    T * values)
{
    constexpr unsigned NV = NT * VT;

    // skip the first block since it will be 0
    #pragma unroll
    for (int i = 0; i < VT; i++){
        const unsigned value_index = ( blockIdx.x + 1 ) * NV + NT * i + threadIdx.x;
        const unsigned spine_index = blockIdx.x;
        const T spine_value = spine[spine_index];
        if (value_index < total) values[value_index] += spine_value;
    }
}

// NT = number of threads
// VT = values per thread
// NV = number of values per block
template <typename T, unsigned NT, unsigned VT>
void scan(
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
        if (number_of_sweeps > 0) {
            block_scan<T, NT, VT, ScanType::Exclusive><<<number_of_blocks[0], NT>>>(
                number_of_elements,
                dev_input,
                dev_output,
                dev_spines[0]
            );
            for (size_t i = 0; i < number_of_sweeps - 1; i++){
                block_scan<T, NT, VT, ScanType::Inclusive><<<number_of_blocks[i+1], NT>>>(
                    number_of_blocks[i],
                    dev_spines[i],
                    dev_spines[i],
                    dev_spines[i+1]
                );
            }
            block_scan<T, NT, VT, ScanType::Inclusive><<<1, NT>>>(
                number_of_blocks[number_of_sweeps - 1],
                dev_spines[number_of_sweeps - 1],
                dev_spines[number_of_sweeps - 1],
                dev_output + number_of_elements
            );
            for (size_t i = 0; i < number_of_sweeps - 1; i++){
                const unsigned source_spine = number_of_sweeps - i - 1;
                const unsigned target_spine = source_spine - 1;
                const unsigned downsweep_blocks = number_of_blocks[source_spine] - 1;
                const unsigned number_of_spines = number_of_blocks[target_spine];

                block_downsweep<unsigned, NT, VT><<<downsweep_blocks, NT>>>(
                    number_of_spines,
                    dev_spines[number_of_sweeps - i - 1],
                    dev_spines[number_of_sweeps - i - 2]
                );
            }
            // Final spine to output downsweep
            block_downsweep<unsigned, NT, VT><<<number_of_blocks[0] - 1, NT>>>(
                number_of_elements,
                dev_spines[0],
                dev_output
            );
        }
        else {
            block_scan<T, NT, VT, ScanType::Exclusive><<<1, NT>>>(
                number_of_elements,
                dev_input,
                dev_output,
                dev_output + number_of_elements
            );
        }
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
    float timing = benchmark(preprocess, process, postprocess, repetitions);
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

    // constexpr std::array<unsigned, 4> NT_values = { 128, 256, 512, 1024 };
    // constexpr std::array<unsigned, 8> VT_values = { 1, 2, 3, 4, 5, 6, 7, 8 };

    const std::array<unsigned, 17> number_of_inputs{
        1024,
        2048,
        4096,
        8192,
        16384,
        32768,
        65536,
        131072,
        262144,
        524288,
        1048576,
        2097152,
        4194304,
        8388608,
        16777216,
        33554432,
        67108864
    };

    constexpr unsigned max_size = 67108864;
    std::vector<unsigned> host_input(max_size);
    std::vector<unsigned> host_output(max_size + 1);
    std::vector<unsigned> host_cpu_output(max_size);

    for (size_t i = 0; i < max_size; i++){
        host_input[i] = prefix_sum_distribution(gen);
    }

    for (auto number_of_elements : number_of_inputs){
        scan<unsigned, NT, VT>(host_input, number_of_elements, host_output);
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
