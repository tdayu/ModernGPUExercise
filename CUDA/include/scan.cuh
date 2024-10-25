#pragma once

#include "warp.cuh"

enum ScanType {
    Exclusive,
    Inclusive
};

template<typename T, ScanType type>
__device__ T warp_scan(const T x, const unsigned mask = Warp::full_mask, const int width = 32){
    T scan(x);

    for (int delta = 1; delta < width; delta *= 2){
        T shuffled = __shfl_up_sync(mask, scan, delta);
        if (laneid() >= delta) scan += shuffled;
    }

    if constexpr (type == ScanType::Exclusive) scan -= x;
    return scan;
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
    T warp_scanned = warp_scan<T, ScanType::Inclusive>(x);

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
        spine_scanned = warp_scan<T, ScanType::Inclusive>(spine_scanned, mask, warp_spine_size);
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
