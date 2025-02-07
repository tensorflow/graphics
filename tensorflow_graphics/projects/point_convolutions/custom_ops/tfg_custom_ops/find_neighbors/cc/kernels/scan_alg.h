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
/// \brief Declaraion of the CUDA operations to execute the parallel scan
///         algorithm in an int array.
/////////////////////////////////////////////////////////////////////////////

#ifndef SCAN_ALG_CUH_
#define SCAN_ALG_CUH_

#include "tfg_custom_ops/shared/cc/kernels/gpu_device.hpp"
#include <memory>

namespace mccnn{

    /**
     *  Method to execute the parallel scan algorithm in an int array.
     *  @param  pDevice                 GPU device.
     *  @param  pNumElems               Number of elements in the array.
     *      The number of elements should be multiple of T*2.
     *  @param  pInGPUPtrElems          Input pointer to the array on 
     *      the GPU.
     *  @return The total accumulation of elements at the end of the array.
     */
    unsigned int scan_alg(
        std::unique_ptr<IGPUDevice>& pDevice,
        const unsigned int pNumElems,
        int* pInGPUPtrElems);
        
}

#endif