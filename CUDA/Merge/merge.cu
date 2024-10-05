#include <random>
#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <array>
#include <assert.h>
#include <algorithm>

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
void print_comparison(std::vector<T>& A, std::vector<T>& B, unsigned elements_per_row, unsigned limit){
    assert(A.size() == B.size());

    unsigned count = 0;
    for (size_t i = 0; i < A.size(); i += elements_per_row){
        printf("A %u:", i);
        for (size_t j = 0; j < elements_per_row; j++){
            if (i + j < A.size()) std::cout << "\t" << A[ i + j ];
            count++;
        }
        printf("\n");
        printf("B %u:", i);
        for (size_t j = 0; j < elements_per_row; j++){
            if (i + j < B.size()) std::cout << "\t" << B[ i + j ];
        }
        printf("\n\n");
        if (count >= limit) break;
    }
    printf("\n");
}

template<typename T>
void print_if_mismatch(std::vector<T>& A, std::vector<T>& B, unsigned limit){
    assert(A.size() == B.size());

    unsigned count = 0;
    for (size_t i = 0; i < A.size(); i++){
        if (A[i] != B[i]){
            printf("Values mismatch! Index: %lu, A: %u, B: %u, Previous A: %u, Previous B: %u\n", i, A[i], B[i], A[i-1], B[i-1]);
            count++;
        }
        if (count >= limit) break;
    }
}

__device__ unsigned int laneid()
{
    unsigned int laneid;
    asm ("mov.u32 %0, %%laneid;" : "=r"(laneid));
    return laneid;
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

// NT = number of threads
// VT = values per thread
// NV = number of values per block
template <typename T, typename COMP>
__device__ int merge_path(
    const T * listA,
    const T * listB,
    const int countA,
    const int countB,
    const int diagonal,
    COMP comp){

    int a_begin = max(0, diagonal - countB);
    int a_end = min(diagonal, countA);

    while (a_begin < a_end){
        int a_mid = (a_begin + a_end) / 2;
        T aKey = listA[a_mid];
        T bKey = listB[diagonal - a_mid - 1];

        bool predicate = comp(aKey, bKey);

        if (predicate) a_begin = a_mid + 1;
        else a_end = a_mid;
    }

    return a_begin;
}

// NT = number of threads
// VT = values per thread
// NV = number of values per block
template <typename T, typename COMP, unsigned NT>
__global__ void merge_path_partition(
    const T * listA,
    const T * listB,
    const int countA,
    const int countB,
    const int partition_size,
    const int number_of_partitions,
    COMP comp,
    int * partitions){

    const int total = countA + countB;
    const int diagonal = min((blockIdx.x * NT + threadIdx.x) * partition_size, total);

    if ((blockIdx.x * NT + threadIdx.x) <= number_of_partitions){
        const int a_begin = merge_path(listA, listB, countA, countB, diagonal, comp);
        partitions[blockIdx.x * NT + threadIdx.x] = a_begin;
    }
}

// NT = number of threads
// VT = values per thread
// NV = number of values per block
template <unsigned VT, typename T, typename COMP>
__device__ void serial_merge(
    const T * keysA,
    const T * keysB,
    const int beginA,
    const int countA,
    const int countB,
    const int diagonal,
    COMP comp,
    int * indices){

    const int beginB = diagonal - beginA;
    const bool in_rangeA = beginA < countA;
    const bool in_rangeB = beginB < countB;

    T keyA, keyB;
    if (in_rangeA) keyA = keysA[beginA];
    if (in_rangeB) keyB = keysB[beginB];

    int indexA = beginA;

    if (in_rangeA || in_rangeB){
        #pragma unroll
        for (unsigned i = 0; i < VT; i++){
            const int indexB = diagonal + i - indexA;
            const bool predicate = (!(indexB < countB)) || (indexA < countA) && comp(keyA, keyB);
            if (predicate){
                indices[i] = indexA;
                indexA++;
                if (indexA < countA) keyA = keysA[indexA];
            }
            else{
                indices[i] = countA + diagonal + i - indexA;
                int nextB = diagonal + i - indexA + 1;
                if (nextB < countB) keyB = keysB[nextB];
            }
        }
    }
}

// NT = number of threads
// VT = values per thread
// NV = number of values per block
template <typename T, typename COMP, unsigned NT, unsigned VT>
__global__ void block_merge(
    const T * listA,
    const T * listB,
    const int countA,
    const int countB,
    const int * partitions,
    COMP comp,
    T * output){

    constexpr unsigned NV = NT * VT;
    union Shared {
        T keys[NV];
        int indices[NV];
    };
    __shared__ Shared shared;

    const int total = countA + countB;
    const int block_total = min(NV, total - blockIdx.x * NV);
    const int block_countA = partitions[blockIdx.x + 1] - partitions[blockIdx.x];
    const int block_countB = block_total - block_countA;

    const int offsetA = partitions[blockIdx.x];
    const int offsetB = blockIdx.x * NV - offsetA;

    #pragma unroll
    for (unsigned i = 0; i < VT; i++){
        const unsigned index = NT * i + threadIdx.x;
        if (index < block_total) shared.keys[index] = (index < block_countA) ? listA[offsetA + index] : listB[offsetB + index - block_countA];
    }
    __syncthreads();

    int indices[VT];

    const int diagonal = min(threadIdx.x * VT, block_total);
    const int beginA = merge_path(shared.keys, shared.keys + block_countA, block_countA, block_countB, diagonal, comp);

    serial_merge<VT>(
        shared.keys,
        shared.keys + block_countA,
        beginA,
        block_countA,
        block_countB,
        diagonal,
        comp,
        indices);

    T keys[VT];
    #pragma unroll
    for (unsigned i = 0; i < VT; i++){
        if (threadIdx.x * VT + i < block_total) keys[i] = shared.keys[indices[i]];
    }
    __syncthreads();
    #pragma unroll
    for (unsigned i = 0; i < VT; i++){
        shared.keys[VT * threadIdx.x + i] = keys[i];
    }
    __syncthreads();
    #pragma unroll
    for (unsigned i = 0; i < VT; i++){
        if (blockIdx.x * NV + i * NT + threadIdx.x < total) output[blockIdx.x * NV + i * NT + threadIdx.x] = shared.keys[i * NT + threadIdx.x];
    }
}

template<typename T>
class leq{
    public:
        __host__ __device__
        bool operator()(const T a, const T b){
            return a <= b;
        };
};

template<typename T>
class lt{
    public:
        __host__ __device__
        bool operator()(const T a, const T b){
            return a < b;
        };
};

// NT = number of threads
// VT = values per thread
// NV = number of values per block
template <unsigned NT, unsigned VT, unsigned PartitionNT, typename T, typename COMP>
void gpu_merge(
    const std::vector<T>& host_listA,
    const std::vector<T>& host_listB,
    const int countA,
    const int countB,
    std::vector<T>& host_output,
    COMP comp)
{
    constexpr unsigned NV = NT * VT;
    const int total = countA + countB;
    const int number_of_partitions = ( total + NV - 1 ) / NV;
    const int partition_number_of_blocks = ( number_of_partitions / PartitionNT ) + 1;

    T * dev_listA;
    T * dev_listB;
    T * dev_output;
    int * dev_partitions;

    CUDA_CALL(cudaSetDevice(0));
    CUDA_CALL(cudaMalloc(&dev_listA, countA * sizeof(T)));
    CUDA_CALL(cudaMalloc(&dev_listB, countB * sizeof(T)));
    CUDA_CALL(cudaMalloc(&dev_output, total * sizeof(T)));
    CUDA_CALL(cudaMalloc(&dev_partitions, (number_of_partitions + 1) * sizeof(int)));

    CUDA_CALL(cudaMemcpy(dev_listA, host_listA.data(), countA * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(dev_listB, host_listB.data(), countB * sizeof(T), cudaMemcpyHostToDevice));

    merge_path_partition<T, COMP, PartitionNT><<<partition_number_of_blocks, PartitionNT>>>(
        dev_listA,
        dev_listB,
        countA,
        countB,
        NV,
        number_of_partitions,
        comp,
        dev_partitions);

    std::vector<int> host_partitions(number_of_partitions + 1);
    CUDA_CALL(cudaMemcpy(host_partitions.data(), dev_partitions, (number_of_partitions + 1) * sizeof(int), cudaMemcpyDeviceToHost));

    block_merge<T, COMP, NT, VT><<<number_of_partitions, NT>>>(
        dev_listA,
        dev_listB,
        countA,
        countB,
        dev_partitions,
        comp,
        dev_output);

    CUDA_CALL(cudaMemcpy(host_output.data(), dev_output, total * sizeof(T), cudaMemcpyDeviceToHost));
}

int main(int argc, char** argv) {
    std::mt19937 gen(1234);
    std::uniform_int_distribution<> prefix_sum_distribution(0, 4);

    constexpr unsigned NT = 512;
    constexpr unsigned VT = 5;
    constexpr unsigned PartitionNT = 128;

    // constexpr std::array<unsigned, 4> NT_values = { 128, 256, 512, 1024 };
    // constexpr std::array<unsigned, 8> VT_values = { 1, 2, 3, 4, 5, 6, 7, 8 };

    const size_t countA = (argc == 3) ? std::stol(argv[1]) : 1024 * 1024;
    const size_t countB = (argc == 3) ? std::stol(argv[2]) : 1024 * 1024;
    const size_t total = countA + countB;
    std::vector<unsigned> host_listA(countA);
    std::vector<unsigned> host_listB(countB);
    std::vector<unsigned> host_cpu_merged(total);
    std::vector<unsigned> host_output(total);

    for (size_t i = 0; i < countA; i++){
        host_listA[i] = prefix_sum_distribution(gen);
    }
    for (size_t i = 0; i < countB; i++){
        host_listB[i] = prefix_sum_distribution(gen);
    }

    std::inclusive_scan(host_listA.begin(), host_listA.end(), host_listA.begin());
    std::inclusive_scan(host_listB.begin(), host_listB.end(), host_listB.begin());

    std::merge(host_listA.begin(), host_listA.end(), host_listB.begin(), host_listB.end(), host_cpu_merged.begin());

    printf("ListA:\n");
    print_array(host_listA, 10, 200);
    printf("ListB:\n");
    print_array(host_listB, 10, 200);

    constexpr lt<unsigned> comp;

    gpu_merge<NT, VT, PartitionNT>(
        host_listA,
        host_listB,
        int(countA),
        int(countB),
        host_output,
        comp
    );

    print_comparison(host_cpu_merged, host_output, 10, 400);
    print_if_mismatch(host_cpu_merged, host_output, 400);
}
