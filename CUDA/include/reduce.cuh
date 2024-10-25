#pragma once

#include "warp.cuh"

template<typename T>
__device__ T warp_reduce(const T x, const unsigned mask = Warp::full_mask, const int width = 32){
    T sum(x);

__shfl_xor_sync()

    for (int delta = 1; delta < width; delta *= 2){
        sum += __shfl_down_sync(mask, sum, delta);
    }

    if constexpr (type == ScanType::Exclusive) scan -= x;
    return scan;
}

template<typename T, unsigned VT>
__device__ void linear_reduce(T * values){
    #pragma unroll
    for (int i = 1; i < VT; i++){
        values[i] += values[i-1];
    }    
}
