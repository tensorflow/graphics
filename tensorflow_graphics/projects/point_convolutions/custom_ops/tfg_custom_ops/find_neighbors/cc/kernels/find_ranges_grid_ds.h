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
/// \brief Declaraion of the CUDA operations to find the ranges in the list
///     of points for a grid cell and its 26 neighbors.
/////////////////////////////////////////////////////////////////////////////

#ifndef BUILD_GRID_DS_CUH_
#define BUILD_GRID_DS_CUH_

#include "tfg_custom_ops/shared/cc/kernels/defines.hpp"
#include "tfg_custom_ops/shared/cc/kernels/gpu_device.hpp"
#include <memory>
#include <vector>

namespace mccnn{
        
    /**
     *  Method to find the ranges in the list of points for a grid cell 
     *  and its 26 neighbors.
     *  @param  pDevice                 Device.
     *  @param  pNumSamples             Number of samples.
     *  @param  pNumPts                 Number of points.
     *  @param  pLastDOffsets           Number of displacement in the last
     *      dimension in the positive and negative axis.
     *  @param  pNumOffsets             Number of offsets applied to  the 
     *      keys.
     *  @param  pInGPUPtrOffsets        List of offsets to apply.
     *  @param  pInGPUPtrSampleKeys     Input pointer to the vector of keys 
     *      of each sample on the GPU.
     *  @param  pInGPUPtrPtsKeys        Input pointer to the vector of keys 
     *      of each point on the GPU.
     *  @param  pInGPUPtrGridDS         Input grid acceleration data 
     *      structure.
     *  @param  pInGPUPtrNumCells       Input pointer to the vector of number  
     *      of cells on the GPU.
     *  @param  pOutGPUPtrRanges        Output pointer to the array containing
     *      the search ranges for each sample. 
     *  @paramT D                       Number of dimensions.
     */
    template<int D>
    void find_ranges_grid_ds_gpu(
        std::unique_ptr<IGPUDevice>& pDevice,
        const unsigned int pNumSamples, 
        const unsigned int pNumPts,
        const unsigned int pLastDOffsets,
        const unsigned int pNumOffsets,
        const int* pInGPUPtrOffsets,
        const mccnn::int64_m* pInGPUPtrSampleKeys,
        const mccnn::int64_m* pInGPUPtrPtsKeys,
        const int* pInGPUPtrGridDS,
        const int* pInGPUPtrNumCells,
        int* pOutGPUPtrRanges);

    /**
     *  Method to compute the total number of offsets
     *  to apply for each range search.
     *  @param  pNumDimensions  Number of dimensions.
     *  @param  pAxisOffset     Offset apply to each axis.
     *  @param  pOutVector      Output parameter with the 
     *      displacements applied to each axis.
     */
    unsigned int computeTotalNumOffsets(
        const unsigned int pNumDimensions,
        const unsigned int pAxisOffset,
        std::vector<int>& pOutVector);
}

#endif