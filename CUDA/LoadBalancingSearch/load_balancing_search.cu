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

enum Bound {
    Lower,
    Upper
};


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
template <Bound bound, typename T, typename COMP>
__device__ int merge_path(
    const T * listA,
    const T * keysB,
    const int countA,
    const int countB,
    const int diagonal,
    COMP comp){

    int a_begin = max(0, diagonal - countB);
    int a_end = min(diagonal, countA);

    while (a_begin < a_end){
        int a_mid = (a_begin + a_end) / 2;
        T aKey = listA[a_mid];
        T bKey = keysB[diagonal - a_mid - 1];

        bool predicate = (bound == Bound::Lower) ? !comp(bKey, aKey) : comp(aKey, bKey);

        if (predicate) a_begin = a_mid + 1;
        else a_end = a_mid;
    }

    return a_begin;
}

__device__ unsigned load_balancing_merge_path(
    const unsigned * keysB,
    const unsigned countB,
    const unsigned countA,
    const unsigned diagonal,
    const unsigned offset = 0){

    unsigned begin = max(diagonal, countB) - countB;
    unsigned end = min(diagonal, countA);

    while (begin < end){
        const unsigned mid = (begin + end) / 2;
        const unsigned key = keysB[diagonal - mid - 1];

        bool predicate = (mid + offset) < key;

        if (predicate) begin = mid + 1;
        else end = mid;
    }

    return begin;
}

// NT = number of threads
// VT = values per thread
// NV = number of values per block
template <unsigned NT>
__global__ void load_balancing_partition(
    const unsigned * keysB,
    const unsigned countB,
    const unsigned countA,
    const unsigned partition_size,
    const unsigned number_of_partitions,
    unsigned * partitions){

    const unsigned total = countA + countB;
    const unsigned diagonal = min((blockIdx.x * NT + threadIdx.x) * partition_size, total);

    if ((blockIdx.x * NT + threadIdx.x) <= number_of_partitions){
        const unsigned begin = load_balancing_merge_path(keysB, countB, countA, diagonal);
        partitions[blockIdx.x * NT + threadIdx.x] = begin;
    }
}

// VT = values per thread
// Do a load balancing search of A (work indices) into B (exclusive scan of work items)
template <unsigned VT>
__device__ void serial_load_balancing_search(
    const unsigned * keysB,
    const unsigned countB,
    const unsigned countA,
    const unsigned beginA,
    const unsigned diagonal,
    const unsigned offsetA,
    unsigned * indicesA){

    unsigned indexA = beginA;
    unsigned indexB = diagonal - indexA;

    #pragma unroll
    for (unsigned i = 0; i < VT; i++){
        if ( indexA < countA || indexB < countB ){
            bool predicate;
            if (indexB >= countB) predicate = true;
            else if (indexA >= countA) predicate = false;
            else predicate = indexA + offsetA < keysB[indexB];

            if (predicate){
                indicesA[indexA++] = indexB;
            }
            else{
                indexB++;
            }
        }
    }
}

// NT = number of threads
// VT = values per thread
// NV = number of values per block
// Load balancing search of A into B
template<unsigned NT, unsigned VT>
__global__ void block_load_balancing_search(
    const unsigned * keysB,
    const unsigned * partitions,
    const unsigned countB,
    const unsigned countA,
    unsigned * indices,
    unsigned * ranks){

    constexpr unsigned NV = NT * VT;
    __shared__ unsigned shared_storage[NV + 1];

    const unsigned total = countA + countB;
    const unsigned block_total = min(NV, total - blockIdx.x * NV);
    const unsigned block_countA = partitions[blockIdx.x + 1] - partitions[blockIdx.x];
    const unsigned block_countB = block_total - block_countA;

    const unsigned offsetA = partitions[blockIdx.x];
    const unsigned offsetB = blockIdx.x * NV - offsetA;

    #pragma unroll
    for (unsigned i = 0; i < VT; i++){
        const unsigned index = NT * i + threadIdx.x;
        // if (index < block_countB) shared_storage[index] = keysB[offsetB + index];
        if (index < block_countB) shared_storage[index + 1] = keysB[offsetB + index];
    }
    if (threadIdx.x == 0 && offsetB > 0) shared_storage[0] = keysB[offsetB - 1];
    __syncthreads();

    const unsigned diagonal = min(threadIdx.x * VT, block_total);
    const unsigned beginA = load_balancing_merge_path(shared_storage + 1, block_countB, block_countA, diagonal, offsetA);

    serial_load_balancing_search<VT>(
        // shared_storage,
        shared_storage + 1,
        block_countB,
        block_countA,
        beginA,
        diagonal,
        offsetA,
        // shared_storage + block_countB);
        shared_storage + block_countB + 1);

    __syncthreads();

    #pragma unroll
    for (unsigned i = 0; i < VT; i++){
        if (i * NT + threadIdx.x < block_countA) {
            const unsigned block_indexB = shared_storage[block_countB + 1 + i * NT + threadIdx.x];
            // const unsigned block_indexB = shared_storage[block_countB + i * NT + threadIdx.x];
            const unsigned indexA = offsetA + i * NT + threadIdx.x;
            const unsigned indexB = block_indexB + offsetB - 1;
            const unsigned rank = indexA - shared_storage[block_indexB];

            indices[indexA] = indexB;
            ranks[indexA] = rank;
        }
    }
}

// NT = number of threads
// VT = values per thread
// NV = number of values per block
template <unsigned NT, unsigned VT, unsigned PartitionNT>
void gpu_load_balancing_search(
    const std::vector<unsigned>& host_exclusive_scan,
    const unsigned count,
    const unsigned total_work,
    std::vector<unsigned>& host_indices,
    std::vector<unsigned>& host_ranks)
{
    constexpr unsigned NV = NT * VT;
    const int number_of_searches = count + total_work;
    const int number_of_partitions = ( number_of_searches + NV - 1 ) / NV;
    const int partition_number_of_blocks = ( number_of_partitions / PartitionNT ) + 1;

    unsigned * dev_exclusive_scan;
    unsigned * dev_indices;
    unsigned * dev_ranks;
    unsigned * dev_partitions;

    CUDA_CALL(cudaSetDevice(0));
    CUDA_CALL(cudaMalloc(&dev_exclusive_scan, count * sizeof(unsigned)));
    CUDA_CALL(cudaMalloc(&dev_indices, total_work * sizeof(unsigned)));
    CUDA_CALL(cudaMalloc(&dev_ranks, total_work * sizeof(unsigned)));
    CUDA_CALL(cudaMalloc(&dev_partitions, (number_of_partitions + 1) * sizeof(int)));

    CUDA_CALL(cudaMemcpy(dev_exclusive_scan, host_exclusive_scan.data(), count * sizeof(unsigned), cudaMemcpyHostToDevice));

    load_balancing_partition<PartitionNT><<<partition_number_of_blocks, PartitionNT>>>(
        dev_exclusive_scan,
        count,
        total_work,
        NV,
        number_of_partitions,
        dev_partitions);

    block_load_balancing_search<NT, VT><<<number_of_partitions, NT>>>(
        dev_exclusive_scan,
        dev_partitions,
        count,
        total_work,
        dev_indices,
        dev_ranks
    );

    CUDA_CALL(cudaMemcpy(host_indices.data(), dev_indices, total_work * sizeof(unsigned), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(host_ranks.data(), dev_ranks, total_work * sizeof(unsigned), cudaMemcpyDeviceToHost));

    CUDA_CALL(cudaFree(dev_exclusive_scan));
    CUDA_CALL(cudaFree(dev_indices));
    CUDA_CALL(cudaFree(dev_ranks));
    CUDA_CALL(cudaFree(dev_partitions));
}

int main(int argc, char** argv) {
    std::mt19937 gen(1234);
    std::uniform_int_distribution<> prefix_sum_distribution(0, 8);

    constexpr unsigned NT = 256;
    constexpr unsigned VT = 7;
    constexpr unsigned PartitionNT = 128;

    // constexpr std::array<unsigned, 4> NT_values = { 128, 256, 512, 1024 };
    // constexpr std::array<unsigned, 8> VT_values = { 1, 2, 3, 4, 5, 6, 7, 8 };

    const unsigned count = (argc == 2) ? std::stol(argv[1]) : 1024 * 256;
    std::vector<unsigned> host_list(count);

    for (unsigned i = 0; i < count; i++){
        host_list[i] = prefix_sum_distribution(gen);
    }

    // CPU Implementation of load balancing search
    std::vector<unsigned> host_exclusive_scan(count);
    std::exclusive_scan(host_list.begin(), host_list.end(), host_exclusive_scan.begin(), 0);
    const unsigned total_work = std::accumulate(host_list.begin(), host_list.end(), 0);

    std::vector<unsigned> host_cpu_indices(total_work);
    std::vector<unsigned> host_cpu_ranks(total_work);
    std::vector<unsigned> host_indices(total_work);
    std::vector<unsigned> host_ranks(total_work);

    unsigned upper_bound = 0;
    for (unsigned i = 0; i < total_work; i++){
        while( !(i < host_exclusive_scan[upper_bound]) && upper_bound < count ){
            upper_bound++;
        }
        const auto index = upper_bound - 1;
        const auto rank = i - host_exclusive_scan[index];
        host_cpu_indices[i] = index;
        host_cpu_ranks[i] = rank;
    }

    gpu_load_balancing_search<NT, VT, PartitionNT>(  
        host_exclusive_scan,
        count,
        total_work,
        host_indices,
        host_ranks
    );

    printf("Work count:\n");
    print_array(host_list, 10, 100);
    printf("Exclusive scan:\n");
    print_array(host_exclusive_scan, 10, 100);
    printf("CPU indices:\n");
    print_array(host_cpu_indices, 10, 400);
    printf("CPU ranks:\n");
    print_array(host_cpu_ranks, 10, 400);
    printf("GPU indices:\n");
    print_array(host_indices, 10, 400);
    printf("GPU ranks:\n");
    print_array(host_ranks, 10, 400);

    printf("CPU vs GPU indices:\n");
    print_comparison(host_cpu_indices, host_indices, 10, 400);

    printf("CPU vs GPU ranks:\n");
    print_comparison(host_cpu_ranks, host_ranks, 10, 400);

    printf("CPU-GPU indices mismatches:\n");
    print_if_mismatch(host_cpu_indices, host_indices, 100);

    printf("CPU-GPU ranks mismatches:\n");
    print_if_mismatch(host_cpu_ranks, host_ranks, 100);

    return 0;
}
