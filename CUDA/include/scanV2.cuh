#pragma once

#include "warp.cuh"
#include "memory.cuh"
#include "common.cuh"

#include <vector>

namespace ScanV2 {
  enum ScanType {
    Exclusive,
    Inclusive
  };

  namespace Device {
    template<typename T, ScanType type>
    __device__ T warp_scan(const T x, const unsigned mask = Warp::full_mask, const int width = 32)
    {
      T scan(x);

      for (int delta = 1; delta < width; delta *= 2){
        T shuffled = __shfl_up_sync(mask, scan, delta);
        if (Warp::laneid() >= delta) scan += shuffled;
      }

      if constexpr (type == ScanType::Exclusive) scan -= x;
      return scan;
    }

    template<typename T>
    __device__ T warp_reduce(const T x, const unsigned mask = Warp::full_mask, const int width = 32)
    {
      T scan = warp_scan<T, ScanType::Inclusive>(x, mask, width);
      T sum = __shfl_sync(mask, scan, width - 1);
      return sum;
    }

    template<typename T, unsigned VT>
    __device__ void linear_scan(T * values){
      #pragma unroll
      for (int i = 1; i < VT; i++){
        values[i] += values[i-1];
      }
    }

    template<typename T, unsigned VT>
    __device__ T linear_reduce(T * values){
      #pragma unroll
      for (int i = 1; i < VT; i++){
        values[i] += values[i-1];
      }
      return values[VT - 1];
    }

    template<typename T, unsigned NT>
    __device__ void cta_reduce(const T x, T * shared_values, T * output)
    {
      T warp_sum = warp_reduce(x);

      // Scan the warp sums
      __syncthreads();
      if (Warp::laneid() == Warp::warp_size - 1) shared_values[threadIdx.x / 32] = warp_sum;
      __syncthreads();

      constexpr unsigned warp_spine_size = ( NT + Warp::warp_size - 1 ) / Warp::warp_size;
      static_assert( warp_spine_size <= 32 );

      const bool predicate = threadIdx.x < warp_spine_size;
      const unsigned mask = __ballot_sync(Warp::full_mask, predicate);
      if (predicate){
        T spine_sum = shared_values[threadIdx.x];
        spine_sum = warp_reduce(spine_sum, mask, warp_spine_size);
        if (threadIdx.x == 0) output[blockIdx.x] = spine_sum;
      }
    }

    template <typename T, unsigned NT, unsigned VT>
    __global__ void block_reduce(
      const unsigned total,
      const T * input,
      T* spine)
    {
      constexpr unsigned NV = NT * VT;
      constexpr unsigned shared_size = (VT > 1) ? NV : NT + 1;
      __shared__ T shared_values[shared_size]; // shared storage
      T values[VT]; // registers storage

      global_to_shared<T, NT, VT>(total, input, shared_values);
      shared_to_registers<T, VT>(shared_values, values);
      const T linear_sum = linear_reduce<T, VT>(values);
      cta_reduce<T, NT>(linear_sum, shared_values, spine);
    }

    template<typename T, unsigned NT>
    __device__ void cta_upsweep(const T x, T * shared_values, T * spine)
    {
      T warp_scanned = warp_scan<T, ScanType::Inclusive>(x);

      // Scan the warp sums
      __syncthreads();
      if (Warp::laneid() == Warp::warp_size - 1) shared_values[threadIdx.x / 32] = warp_scanned;
      __syncthreads();

      constexpr unsigned warp_spine_size = ( NT + Warp::warp_size - 1 ) / Warp::warp_size;
      static_assert( warp_spine_size <= 32 );

      const bool predicate = threadIdx.x < warp_spine_size;
      const unsigned mask = __ballot_sync(Warp::full_mask, predicate);
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
    __device__ void cta_downsweep(T * shared_values, T * values)
    {
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
      cta_downsweep<T, VT, type>(shared_values, values);
      registers_to_shared<T, VT>(shared_values, values);
      shared_to_global<T, NT, VT>(total, output, shared_values);
    }

    template <typename T, unsigned VT>
    __device__ void cta_add_spine(
      const T * spines,
      T * values)
    {
      #pragma unroll
      for (int i = 0; i < VT; i++){
        const unsigned spine_index = blockIdx.x;
        const T spine_value = spine[spine_index];
        values[i] += spine_value;
      }
    }

    template <typename T, unsigned NT, unsigned VT>
    __global__ void block_scan_and_downsweep(
      const unsigned total,
      const T * input,
      const T * spines,
      T * output)
    {
      constexpr unsigned NV = NT * VT;
      constexpr unsigned shared_size = (VT > 1) ? NV : NT + 1;
      __shared__ T shared_values[shared_size]; // shared storage
      T values[VT]; // registers storage

      global_to_shared<T, NT, VT>(total, input, shared_values);
      shared_to_registers<T, VT>(shared_values, values);
      linear_scan<T, VT>(values);
      cta_upsweep<T, NT>(values[VT - 1], shared_values, spine);
      cta_downsweep<T, VT, type>(shared_values, values);
      cta_add_spine<T, VT>(spines, values);
      registers_to_shared<T, VT>(shared_values, values);
      shared_to_global<T, NT, VT>(total, output, shared_values);
    }
  }

  template <typename T, unsigned NT, unsigned VT>
  void launch_kernels(
    const T* dev_input,
    const unsigned number_of_elements,
    const unsigned number_of_sweeps,
    const unsigned * number_of_blocks,
    T** dev_spines,
    T* dev_output)
  {
    if (number_of_sweeps > 0) {
        Device::block_reduce<T, NT, VT, ScanType::Exclusive><<<number_of_blocks[0], NT>>>(
            number_of_elements,
            dev_input,
            dev_spines[0]
        );
        for (size_t i = 0; i < number_of_sweeps - 1; i++){
            Device::block_reduce<T, NT, VT, ScanType::Inclusive><<<number_of_blocks[i+1], NT>>>(
                number_of_blocks[i],
                dev_spines[i],
                dev_spines[i+1]
            );
        }
        Device::block_scan<T, NT, VT, ScanType::Inclusive><<<1, NT>>>(
            number_of_blocks[number_of_sweeps - 1],
            dev_spines[number_of_sweeps - 1],
            dev_spines[number_of_sweeps - 1],
            dev_output + number_of_elements
        );

        for (size_t i = 0; i < number_of_sweeps - 1; i++){
            const unsigned source_spine = number_of_sweeps - i - 1;
            const unsigned target_spine = source_spine - 1;
            const unsigned downsweep_blocks = number_of_blocks[source_spine];
            const unsigned number_of_spines = number_of_blocks[target_spine];

            Device::block_scan_and_downsweep<unsigned, NT, VT><<<downsweep_blocks, NT>>>(
                number_of_spines,
                dev_spines[target_spine],
                dev_spines[source_spine],
                dev_spines[target_spine]
            );
        }

        Device::block_scan_and_downsweep<unsigned, NT, VT><<<number_of_blocks[0], NT>>>(
            number_of_elements,
            dev_output,
            dev_spines[0],
            dev_output
        );
    }
    else {
        Device::block_scan<T, NT, VT, ScanType::Exclusive><<<1, NT>>>(
            number_of_elements,
            dev_output,
            dev_output,
            dev_output + number_of_elements
        );
    }
  }

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

    T * dev_output;
    std::vector <T*> dev_spines(number_of_sweeps);

    CUDA_CALL(cudaSetDevice(0));
    CUDA_CALL(cudaMalloc(&dev_output, (number_of_elements + 1) * sizeof(T)));
    for (size_t i = 0; i < number_of_sweeps; i++){
        CUDA_CALL(cudaMalloc(&(dev_spines[i]), number_of_blocks[i] * sizeof(T)));
    }

    CUDA_CALL(cudaMemcpy(dev_output, host_input.data(), number_of_elements * sizeof(T), cudaMemcpyHostToDevice));

    launch_kernels<T, NT, VT>(
      dev_output,
      number_of_elements,
      number_of_sweeps,
      number_of_blocks.data(),
      dev_spines.data(),
      dev_output);

    CUDA_CALL(cudaMemcpy(host_output.data(), dev_output, (number_of_elements + 1) * sizeof(T), cudaMemcpyDeviceToHost));
  }
}
