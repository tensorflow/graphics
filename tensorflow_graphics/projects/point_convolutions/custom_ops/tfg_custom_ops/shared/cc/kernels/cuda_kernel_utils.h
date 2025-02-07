/////////////////////////////////////////////////////////////////////////////
/// Copyright 2020 Google LLC
///
/// Licensed under the Apache License, Version 2.0 (the "License");
/// you may not use this file except in compliance with the License.
/// You may obtain a copy of the License at
///
///    https://www.apache.org/licenses/LICENSE-2.0
///
/// Unless required by applicable law or agreed to in writing, software
/// distributed under the License is distributed on an "AS IS" BASIS,
/// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
/// See the License for the specific language governing permissions and
/// limitations under the License.
/////////////////////////////////////////////////////////////////////////////
/// \brief Utilities for the cuda implementations of the tensor operations.
/////////////////////////////////////////////////////////////////////////////

#ifndef CUDA_KERNEL_UTILS_H_
#define CUDA_KERNEL_UTILS_H_

#include <stdio.h>
#include "cuda_runtime.h"
#include "cuda_fp16.h"

namespace mccnn{

    ///////////////////////// DEVICE FUNCTIONS

    /**
     *  Function to compute the global index of the current thread.
     *  @return   Current thread index.
     */
    __device__ __forceinline__ unsigned long long int compute_global_index_gpu_funct()
    {
        return threadIdx.x + blockDim.x*blockIdx.x;
    }

    /**
     *  Function to compute the total number of threads in execution..
     *  @return   Total number of threads.
     */
    __device__ __forceinline__ unsigned long long int compute_total_threads_gpu_funct()
    {
        return gridDim.x*blockDim.x;
    }

    /**
     *  Function to do an atomic max operation on floats.
     *  @param  pAddress    Address in which we want to perform the atomic operation.
     *  @param  pVal        Value we want to input.
     *  @return Stored value.
     */
    __device__ static float atomicMax(float* pAddress, const float pVal)
    {
        int* address_as_i = (int*) pAddress;
        int old = *address_as_i, assumed;
        do {
            assumed = old;
            old = ::atomicCAS(address_as_i, assumed,
                __float_as_int(::fmaxf(pVal, __int_as_float(assumed))));
        } while (assumed != old);
        return __int_as_float(old);
    }
}

#endif