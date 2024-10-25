#pragma once

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
