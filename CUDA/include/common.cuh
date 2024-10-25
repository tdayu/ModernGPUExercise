#pragma once

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
