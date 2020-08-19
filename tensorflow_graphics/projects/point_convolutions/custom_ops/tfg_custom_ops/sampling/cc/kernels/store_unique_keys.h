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
/// \brief Declaraion of the CUDA operations to store the first point index
///     of each unique key.
/////////////////////////////////////////////////////////////////////////////

#ifndef STORE_UNIQUE_KEYS_CUH_
#define STORE_UNIQUE_KEYS_CUH_

#include "tfg_custom_ops/shared/cc/kernels/defines.hpp"
#include "tfg_custom_ops/shared/cc/kernels/gpu_device.hpp"
#include <memory>

namespace mccnn{
        
    /**
     *  Method to store the first point index for each unique keys on the gpu.
     *  @param  pDevice                 Device.
     *  @param  pNumPts                 Number of points.
     *  @param  pInKeysGPUPtr           Input pointer to the vector of keys  
     *      sorted from bigger to smaller residing on GPU memory.
     *  @param  pFIndexKeys             Output pointer to the array with 
     *      the first point index for each unique key.
     */
    void store_unique_keys_gpu(
        std::unique_ptr<IGPUDevice>& pDevice,
        const unsigned int pNumPts,
        const mccnn::int64_m* pInKeysGPUPtr,
        int* pFIndexKeys);
}

#endif