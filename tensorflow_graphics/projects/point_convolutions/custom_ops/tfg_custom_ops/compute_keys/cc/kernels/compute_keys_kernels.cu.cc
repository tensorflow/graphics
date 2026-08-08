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
/// \brief Implementation of the CUDA operations to compute the keys indices 
///     of a point cloud into a regular grid. 
/////////////////////////////////////////////////////////////////////////////

#include "tfg_custom_ops/shared/cc/kernels/defines.hpp" 
#include "tfg_custom_ops/shared/cc/kernels/math_helper.h"
#include "tfg_custom_ops/shared/cc/kernels/cuda_kernel_utils.h"
#include "tfg_custom_ops/shared/cc/kernels/grid_utils.h"

#include "compute_keys.h"

///////////////////////// GPU

/**
 *  GPU kernel to compute the keys of each point.
 *  @param  pNumPts         Number of points.
 *  @param  pPts            Array of points.
 *  @param  pBatchIds       Array of batch ids.
 *  @param  pSAABBMin       Array of scaled minimum point of bounding
 *      boxes.
 *  @param  pNumCells       Number of cells.
 *  @param  pInvCellSize    Inverse cell size.
 *  @param  pOutKeys        Output array with the point keys.
 */
template<int D>
__global__ void compute_keys_gpu_kernel(
    const unsigned int pNumPts,
    const mccnn::fpoint<D>* __restrict__ pPts,
    const int* __restrict__ pBatchIds,
    const mccnn::fpoint<D>* __restrict__ pSAABBMin,
    const mccnn::ipoint<D>* __restrict__ pNumCells,
    const mccnn::fpoint<D>* __restrict__ pInvCellSize,
    mccnn::int64_m* __restrict__ pOutKeys)
{
    int initPtIndex = mccnn::compute_global_index_gpu_funct();
    int totalThreads = mccnn::compute_total_threads_gpu_funct();

    for(int curPtIndex = initPtIndex; curPtIndex < pNumPts; curPtIndex += totalThreads)
    {
        //Get the values for the point.
        int curBatchId = pBatchIds[curPtIndex];
        mccnn::fpoint<D> curPt = pPts[curPtIndex];
        mccnn::fpoint<D> curSAABBMin = pSAABBMin[curBatchId];

        //Compute the current cell indices.
        mccnn::ipoint<D> cell = mccnn::compute_cell_gpu_funct(
            curPt, curSAABBMin, pNumCells[0], pInvCellSize[0]);

        //Compute the key index of the cell.
        mccnn::int64_m keyIndex = mccnn::compute_key_gpu_funct(
            cell, pNumCells[0], curBatchId);

        //Save the key index.
        pOutKeys[curPtIndex] = keyIndex;
    }
}

///////////////////////// CPU

template<int D>
void mccnn::compute_keys_gpu(
    std::unique_ptr<IGPUDevice>& pDevice,
    const unsigned int pNumPts,
    const float* pInGPUPtrPts,
    const int* pInGPUPtrBatchIds,
    const float* pInGPUPtrSAABBMin,
    const int* pInGPUPtrNumCells,
    const float* pInGPUPtrInvCellSizes,
    mccnn::int64_m* pOutGPUPtrKeys)
{
    //Get the cuda stream.
    auto cudaStream = pDevice->getCUDAStream();

    //Get the device properties.
    const GpuDeviceProperties& gpuProps = pDevice->get_device_properties();

    //Calculate the ideal number of blocks for the selected block size.
    unsigned int numMP = gpuProps.numMPs_;
    unsigned int blockSize = gpuProps.warpSize_*2;
    unsigned int numBlocks = pDevice->get_max_active_block_x_sm(
        blockSize,(const void*)compute_keys_gpu_kernel<D>, 0);
    pDevice->check_error(__FILE__, __LINE__);

    //Calculate the total number of blocks to execute.
    unsigned int execBlocks = pNumPts/blockSize;
    execBlocks += (pNumPts%blockSize != 0)?1:0;
    unsigned int totalNumBlocks = numMP*numBlocks;
    totalNumBlocks = (totalNumBlocks > execBlocks)?execBlocks:totalNumBlocks;

    //Execute the cuda kernel.
    compute_keys_gpu_kernel<D><<<totalNumBlocks, blockSize, 0, cudaStream>>>(
        pNumPts, 
        (const mccnn::fpoint<D>*)pInGPUPtrPts, 
        pInGPUPtrBatchIds,
        (const mccnn::fpoint<D>*)pInGPUPtrSAABBMin, 
        (const mccnn::ipoint<D>*)pInGPUPtrNumCells,
        (const mccnn::fpoint<D>*)pInGPUPtrInvCellSizes,
        pOutGPUPtrKeys);
    pDevice->check_error(__FILE__, __LINE__);
}

///////////////////////// CPU Template declaration

#define COMPUTE_KEYS_TEMP_DECL(Dims)                \
    template void mccnn::compute_keys_gpu<Dims>(    \
            std::unique_ptr<IGPUDevice>& pDevice,   \
            const unsigned int pNumPts,             \
            const float* pInGPUPtrPts,              \
            const int* pInGPUPtrBatchIds,           \
            const float* pInGPUPtrSAABBMin,         \
            const int* pInGPUPtrNumCells,           \
            const float* pInGPUPtrInvCellSizes,     \
            mccnn::int64_m* pOutGPUPtrKeys);

DECLARE_TEMPLATE_DIMS(COMPUTE_KEYS_TEMP_DECL)