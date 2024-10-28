#pragma once

namespace Warp {
    constexpr unsigned full_mask = 0xFFFFFFFF;
    constexpr unsigned warp_size = 32;

    __device__ unsigned int laneid()
    {
        unsigned int laneid;
        asm ("mov.u32 %0, %%laneid;" : "=r"(laneid));
        return laneid;
    }
};
