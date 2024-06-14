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
/// \brief Declaraion of the CUDA operations to count the neighbors for each
///         point.
/////////////////////////////////////////////////////////////////////////////

#ifndef COUNT_NEIGHBORS_CUH_
#define COUNT_NEIGHBORS_CUH_

#include "tfg_custom_ops/shared/cc/kernels/defines.hpp"
#include "tfg_custom_ops/shared/cc/kernels/gpu_device.hpp"
#include <memory>

namespace mccnn{

    /**
     *  Method to count the number of neighbors.
     *  @param  pDevice             GPU device.
     *  @param  pNumSamples         Number of samples.
     *  @param  pNumRanges          Number of ranges per point.
     *  @param  pInGPUPtrSamples    Input pointer to the vector of samples 
     *      coordinates on the GPU.
     *  @param  pInGPUPtrPts        Input pointer to the vector of point 
     *      coordinates on the GPU.
     *  @param  pInGPUPtrRanges     Input pointer to the vector of sample 
     *      ranges on the GPU.
     *  @param  pInvRadii           Inverse of the radius used on the 
     *      search of neighbors in each dimension.
     *  @param  pOutGPUPtrNumNeighs Output pointer to the vector with the 
     *      number of neighbors for each sample on the GPU. The memory
     *      should be initialized to 0 outside this function.
     *  @tparam D                   Number of dimensions.
     */
    template<int D>
    void count_neighbors(
        std::unique_ptr<IGPUDevice>& pDevice,
        const unsigned int pNumSamples,
        const unsigned int pNumRanges,
        const float* pInGPUPtrSamples,
        const float* pInGPUPtrPts,
        const int* pInGPUPtrRanges,
        const float* pInGPUPtrInvRadii,
        int* pOutGPUPtrNumNeighs);
        
}

#endif