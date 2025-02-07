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
/// \brief Implementation of the CUDA operations to build the data structure to 
///     access the sparse regular grid. 
/////////////////////////////////////////////////////////////////////////////

#include "tfg_custom_ops/shared/cc/kernels/defines.hpp"
#include "tfg_custom_ops/shared/cc/kernels/math_helper.h"
#include "tfg_custom_ops/shared/cc/kernels/cuda_kernel_utils.h"
#include "tfg_custom_ops/shared/cc/kernels/grid_utils.h"

#include "tfg_custom_ops/build_grid_ds/cc/kernels/build_grid_ds.h"

///////////////////////// GPU

/**
 *  GPU kernel to compute the grid data structure.
 *  @param  pNumPts         Number of points.
 *  @param  pKeys           Array of keys.
 *  @param  pNumCells       Number of cells.
 *  @param  pOutDS          Output array with the data structure.
 *  @paramT D                       Number of dimensions.
 */
 template<int D>
 __global__ void build_grid_gpu_kernel(
    const unsigned int pNumPts,
    const mccnn::int64_m* __restrict__ pKeys,
    const mccnn::ipoint<D>* __restrict__ pNumCells,
    int2* __restrict__ pOutDS)
{
    int initPtIndex = mccnn::compute_global_index_gpu_funct();
    int totalThreads = mccnn::compute_total_threads_gpu_funct();

    for(int curPtIndex = initPtIndex; curPtIndex < pNumPts; curPtIndex += totalThreads)
    {
        //Get the key and compute the index into the ds.
        mccnn::int64_m curKey = pKeys[curPtIndex];
        int dsIndex = mccnn::compute_ds_index_from_key_gpu_funct(curKey, pNumCells[0]);

        //Check if it is the first point in the ds cell.
        int prevPtIndex = curPtIndex-1;
        if(prevPtIndex >= 0){
            if(dsIndex != 
                mccnn::compute_ds_index_from_key_gpu_funct(pKeys[prevPtIndex], pNumCells[0])){
                    pOutDS[dsIndex].x = curPtIndex;
            }
        }

        //Check if it is the last point in the ds cell.
        int nextPtIndex = curPtIndex+1;
        if(nextPtIndex == pNumPts){
            pOutDS[dsIndex].y = pNumPts;
        }else if(dsIndex != 
            mccnn::compute_ds_index_from_key_gpu_funct(pKeys[nextPtIndex], pNumCells[0])){
            pOutDS[dsIndex].y = nextPtIndex;
        }
    }
}

///////////////////////// CPU

template<int D>
void mccnn::build_grid_ds_gpu(
    std::unique_ptr<IGPUDevice>& pDevice,
    const unsigned int pDSSize, 
    const unsigned int pNumPts,
    const mccnn::int64_m* pInGPUPtrKeys,
    const int* pInGPUPtrNumCells,
    int* pOutGPUPtrDS)
{
    //Get the cuda stream.
    auto cudaStream = pDevice->getCUDAStream();

    //Get the device properties.
    const GpuDeviceProperties& gpuProps = pDevice->get_device_properties();

    //Calculate the ideal number of blocks for the selected block size.
    unsigned int numMP = gpuProps.numMPs_;
    unsigned int blockSize = gpuProps.warpSize_*2;
    unsigned int numBlocks = pDevice->get_max_active_block_x_sm(
        blockSize,(const void*)build_grid_gpu_kernel<D>, 0);
    pDevice->check_error(__FILE__, __LINE__);

    //Calculate the total number of blocks to execute.
    unsigned int execBlocks = pNumPts/blockSize;
    execBlocks += (pNumPts%blockSize != 0)?1:0;
    unsigned int totalNumBlocks = numMP*numBlocks;
    totalNumBlocks = (totalNumBlocks > execBlocks)?execBlocks:totalNumBlocks;

    //Initialize to zero the output array.
    pDevice->memset(pOutGPUPtrDS, 0, sizeof(int)*pDSSize);
    pDevice->check_error(__FILE__, __LINE__);

    //Execute the cuda kernel.
    build_grid_gpu_kernel<D><<<totalNumBlocks, blockSize, 0, cudaStream>>>(
        pNumPts, 
        pInGPUPtrKeys,
        (const mccnn::ipoint<D>*)pInGPUPtrNumCells,
        (int2*)pOutGPUPtrDS);
    pDevice->check_error(__FILE__, __LINE__);
}

///////////////////////// CPU Template declaration

#define BUILD_GRID_DS_TEMP_DECL(Dims)               \
    template void mccnn::build_grid_ds_gpu<Dims>(   \
        std::unique_ptr<IGPUDevice>& pDevice,       \
        const unsigned int pDSSize,                 \
        const unsigned int pNumPts,                 \
        const mccnn::int64_m* pInGPUPtrKeys,        \
        const int* pInGPUPtrNumCells,               \
        int* pOutGPUPtrDS);

DECLARE_TEMPLATE_DIMS(BUILD_GRID_DS_TEMP_DECL)