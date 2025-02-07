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
/// \brief Declaraion of the CUDA operations to compute the keys indices 
///     of a point cloud into a regular grid. 
/////////////////////////////////////////////////////////////////////////////

#ifndef COMPUTE_KEYS_CUH_
#define COMPUTE_KEYS_CUH_

#include "tfg_custom_ops/shared/cc/kernels/defines.hpp"
#include "tfg_custom_ops/shared/cc/kernels/gpu_device.hpp"
#include <memory>

namespace mccnn{

    /**
     *  Method to compute the keys on the gpu.
     *  @param  pDevice                 Device.
     *  @param  pNumPts                 Number of points.
     *  @param  pInGPUPtrPts            Input pointer to the vector of points 
     *      on the GPU.
     *  @param  pInGPUPtrBatchIds       Input pointer to the vector of batch 
     *      ids on the GPU.
     *  @param  pInGPUPtrSAABBMin       Input pointer to the vector of minimum 
     *      points of the bounding boxes on the GPU scaled by the inverse
     *      cell size.
     *  @param  pInGPUPtrNumCells       Input pointer to the vector of number  
     *      of cells on the GPU.
     *  @param  pInGPUPtrInvCellSizes      Input pointer to the vector with the 
     *      inverse sizes of each cell.
     *  @param  pInGPUpOutGPUPtrKeys    Output pointer to the vector of keys  
     *      on the GPU.
     *  @paramT D                       Number of dimensions.
     */
    template<int D>
    void compute_keys_gpu(
        std::unique_ptr<IGPUDevice>& pDevice,
        const unsigned int pNumPts,
        const float* pInGPUPtrPts,
        const int* pInGPUPtrBatchIds,
        const float* pInGPUPtrSAABBMin,
        const int* pInGPUPtrNumCells,
        const float* pInGPUPtrInvCellSizes,
        mccnn::int64_m* pOutGPUPtrKeys);
}

#endif