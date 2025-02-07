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
/// \brief Declaraion of the CUDA operations to store in memory the pooled
///     points.
/////////////////////////////////////////////////////////////////////////////

#ifndef STORED_POOLED_PTS_CUH_
#define STORED_POOLED_PTS_CUH_

#include "tfg_custom_ops/shared/cc/kernels/defines.hpp"
#include "tfg_custom_ops/shared/cc/kernels/gpu_device.hpp"
#include <memory>

namespace mccnn{
        
    /**
     *  Method to pool a set of points from a point cloud.
     *  @param  pDevice                 Device.
     *  @param  pNumPts                 Number of input points.
     *  @param  pNumPooledPts           Number of pooled points.
     *  @param  pPtsGPUPtr              Input pointer to the gpu array with
     *      the point coordinates.
     *  @param  pBatchIdsGPUPtr         Input pointer to the gpu array with
     *      the batch ids.
     *  @param  pSelectedGPUPtr         Input pointer to the gpu array with
     *      the selected points.
     *  @param  pOutPtsGPUPtr           Output pointer to the gpu array with
     *      the selected point coordinates.
     *  @param  pOutBatchIdsGPUPtr      Output pointer to the gpu array with
     *      the selected batch ids.
     *  @param  pOutIndicesGPUPtr       Output pointer to the gpu array with
     *      the selected indices.
     *  @paramt D                       Number of dimensions.
     */
    template<int D>
    void store_pooled_pts_gpu(
        std::unique_ptr<IGPUDevice>& pDevice,
        const unsigned int pNumPts,
        const unsigned int pNumPooledPts,
        const float* pPtsGPUPtr,
        const int* pBatchIdsGPUPtr,
        const int* pSelectedGPUPtr,
        float* pOutPtsGPUPtr,
        int* pOutBatchIdsGPUPtr,
        int* pOutIndicesGPUPtr);
}

#endif