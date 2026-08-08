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
/// \brief Declaraion of the CUDA operations to build the data structure to 
///     access the sparse regular grid. 
/////////////////////////////////////////////////////////////////////////////

#ifndef BUILD_GRID_DS_CUH_
#define BUILD_GRID_DS_CUH_

#include "tfg_custom_ops/shared/cc/kernels/defines.hpp"
#include "tfg_custom_ops/shared/cc/kernels/gpu_device.hpp"
#include <memory>

namespace mccnn{
        
    /**
     *  Method to compute the data structure to access a sparse regular grid.
     *  @param  pDevice                 Device.
     *  @param  pDSSize                 Size of the data structure.
     *  @param  pNumPts                 Number of points.
     *  @param  pInGPUPtrKeys           Input pointer to the vector of keys 
     *      on the GPU.
     *  @param  pInGPUPtrNumCells       Input pointer to the vector of number  
     *      of cells on the GPU.
     *  @param  pInGPUpOutGPUPtrKeys    Output pointer to the data structure  
     *      on the GPU.
     *  @paramT D                       Number of dimensions.
     */
    template<int D>
    void build_grid_ds_gpu(
        std::unique_ptr<IGPUDevice>& pDevice,
        const unsigned int pDSSize, 
        const unsigned int pNumPts,
        const mccnn::int64_m* pInGPUPtrKeys,
        const int* pInGPUPtrNumCells,
        int* pOutGPUPtrDS);
}

#endif