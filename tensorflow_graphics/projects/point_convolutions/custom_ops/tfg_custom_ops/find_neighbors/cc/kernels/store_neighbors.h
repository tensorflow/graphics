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
/// \brief Declaraion of the CUDA operations to store the neighbors for each
///         point.
/////////////////////////////////////////////////////////////////////////////

#ifndef STORE_NEIGHBORS_CUH_
#define STORE_NEIGHBORS_CUH_

#include "tfg_custom_ops/shared/cc/kernels/gpu_device.hpp"
#include <memory>

namespace mccnn{

    /**
     *  Method to store the number of neighbors.
     *  @param  pDevice                 GPU device.
     *  @param  pMaxNeighbors           Maximum number of neighbors. If zero or less, 
     *      there is not limit.
     *  @param  pNumSamples             Number of samples.
     *  @param  pNumRanges              Number of ranges per point.
     *  @param  pInGPUPtrSamples        Input pointer to the vector of samples 
     *      coordinates on the GPU.
     *  @param  pInGPUPtrPts            Input pointer to the vector of point 
     *      coordinates on the GPU.
     *  @param  pInGPUPtrRanges         Input pointer to the vector of sample 
     *      ranges on the GPU.
     *  @param  pInGPUPtrInvRadii       Inverse of the radius used on the 
     *      search of neighbors in each dimension.
     *  @param  pOutGPUPtrNumNeighsU    Input/Output pointer to the vector  
     *      with the number of neighbors for each sample without the limit of
     *      pMaxNeighbors.
     *  @param  pOutGPUPtrNumNeighs     Input/Output pointer to the vector  
     *      with the number of neighbors for each sample on the GPU.
     *  @param  pOutGPUPtrNeighs        Output pointer to the vector with the 
     *      number of neighbors for each sample on the GPU.
     *  @tparam D                       Number of dimensions.
     */
    template<int D>
    void store_neighbors(
        std::unique_ptr<IGPUDevice>& pDevice,
        const int pMaxNeighbors,
        const unsigned int pNumSamples,
        const unsigned int pNumRanges,
        const float* pInGPUPtrSamples,
        const float* pInGPUPtrPts,
        const int* pInGPUPtrRanges,
        const float* pInGPUPtrInvRadii,
        int* pOutGPUPtrNumNeighsU,
        int* pOutGPUPtrNumNeighs,
        int* pOutGPUPtrNeighs);
        
}

#endif