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

__device__ unsigned int laneid()
{
    unsigned int laneid;
    asm ("mov.u32 %0, %%laneid;" : "=r"(laneid));
    return laneid;
}

template<typename T>
__device__ T warp_inclusive_scan(const T x){
    T scan(x);

    for (int delta = 1; delta < WARP_SIZE; delta *= 2){
        T shuffled = __shfl_up_sync(FULL_MASK, scan, delta);
        if (laneid() >= delta) scan += shuffled;
    }

    return scan;
}

template<typename T, unsigned NT, unsigned VT>
__device__ void global_to_shared(const T * global, T * shared){
    #pragma unroll
    for (i = 0; i < VT; i++){
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
__device__ void shared_to_global(const T * global, T * shared){
    #pragma unroll
    for (int i = 0; i < VT; i++){
        const unsigned global_index = NV * blockIdx.x + NT * i + threadIdx.x;
        const unsigned shared_index = NT * i + threadIdx.x;
        if (global_index < total) output[global_index] = shared_values[shared_index];
    }
}

template<typename T, unsigned VT>
__device__ void registers_to_shared(T * shared, const T * registers){
    #pragma unroll
    for (int i = 0; i < VT; i++){
        shared[VT * threadIdx.x + i] = registers[i];
    }
}

template<typename T, unsigned VT>
__device__ void linear_scan(T * values){
    #pragma unroll
    for (int i = 1; i < VT; i++){
        values[i] += values[i-1];
    }    
}

template<typename T, NT>
__device__ void cta_upsweep(const T x, T * shared_values, T * spine){
    T warp_scanned = warp_inclusive_scan(x);

    shared_values[threadIdx.x] = warp_scanned;
    __syncthreads();
    for (int delta = WARP_SIZE; delta < NT; delta *= 2){
        if (threadIdx.x >= delta) shared_values[threadIdx.x] += shared_values[threadIdx.x - delta];
        __syncthreads();
    }

    const bool last_thread = (threadIdx.x == NT - 1);
    if (last_thread) spine[NT] = shared_values[threadIdx.x];
}

template<typename T, VT>
__device__ void cta_downsweep(T * shared_values, T * values){
    const T downsweep = (threadIdx.x > 0) ? shared_values[threadIdx.x - 1] : 0;
    #pragma unroll
    for (int i = 0; i < VT; i++){
        values[i] += downsweep;
    }
    __syncthreads();
}

// NT = number of threads
// VT = values per thread
// NV = number of values per block
template <typename T, unsigned NT, unsigned VT>
__global__ void block_scan(
    const unsigned total,
    const T * input,
    T* output,
    T* spine)
{
    constexpr unsigned NV = NT * VT;
    __shared__ T shared_values[NV]; // shared storage
    T values[VT]; // registers storage

    global_to_shared<T, NT, VT>(input, shared_values);
    shared_to_registers<T, VT>(shared_values, values);
    linear_scan<T, VT>(values);
    cta_upsweep<T, NT>(values[VT - 1], shared_values, spine);
    cta_downsweep<T, VT>(shared_values, values);
    registers_to_shared<T, VT>(shared_values, values);
    shared_to_global<T, NT, VT>(output, shared_values);
}

int main() {
    std::mt19937 gen(1234);
    std::uniform_int_distribution<> prefix_sum_distribution(0, 128);

    // constexpr std::array<unsigned, 4> NT_values = { 128, 256, 512, 1024 };
    // constexpr std::array<unsigned, 8> VT_values = { 1, 2, 3, 4, 5, 6, 7, 8 };

    constexpr size_t number_of_elements = 1024 * 1024 * 32;
    std::vector<unsigned> host_input(number_of_elements);
    std::vector<unsigned> host_output(number_of_elements);

    for (size_t i = 0; i < number_of_elements; i++){
        host_input[i] = prefix_sum_distribution(gen);
    }

    constexpr unsigned NT = 512;
    constexpr unsigned VT = 5;
    constexpr unsigned NV = NT * VT;

    unsigned * dev_input;
    unsigned * dev_output;
    unsigned * dev_spine;

    constexpr unsigned number_of_blocks = ( number_of_elements + NV - 1 ) / NV;

    CUDA_CALL(cudaSetDevice(0));
    CUDA_CALL(cudaMalloc(&dev_input, number_of_elements * sizeof(unsigned)));
    CUDA_CALL(cudaMalloc(&dev_output, number_of_elements * sizeof(unsigned)));
    CUDA_CALL(cudaMalloc(&dev_spine, number_of_blocks * sizeof(unsigned)));

    CUDA_CALL(cudaMemcpy(dev_input, host_input.data(), number_of_elements * sizeof(unsigned), cudaMemcpyHostToDevice));
    block_scan<unsigned, NT, VT><<<number_of_blocks, NT>>>(
        number_of_elements,
        dev_input,
        dev_output,
        dev_spine
    );
    CUDA_CALL(cudaMemcpy(host_output.data(), dev_output, number_of_elements * sizeof(unsigned), cudaMemcpyDeviceToHost));

    return 0;
}
