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
/// \brief Implementation of the CUDA operations to store in memory the sampled
///     points.
/////////////////////////////////////////////////////////////////////////////

#include "tfg_custom_ops/shared/cc/kernels/defines.hpp"
#include "tfg_custom_ops/shared/cc/kernels/math_helper.h"
#include "tfg_custom_ops/shared/cc/kernels/cuda_kernel_utils.h"
#include "tfg_custom_ops/shared/cc/kernels/grid_utils.h"

#include "tfg_custom_ops/sampling/cc/kernels/store_sampled_pts.h"

///////////////////////// GPU

/**
 *  GPU kernel to count the unique keys in a list.
 *  @param  pNumPts             Number of input points.
 *  @param  pPts                Input pointer to the array with the points.
 *  @param  pBatchIds           Input pointer to the array with the batch ids.
 *  @param  pSelected           Input pointer to the array with the selected pts.
 *  @param  pNumSampledPoints    Input pointer to the counter of already saved pts.
 *  @param  pOutPts             Output pointer to the array with the output pts.
 *  @param  pBatchIds           Output pointer to the array with the output batchids.
 *  @param  pOutIndices         Output pointer to the array with the output indices.
 *  @paramt D                   Number of dimensions.
 */
 template <int D>
 __global__ void store_sampled_pts_gpu_kernel(
    const unsigned int pNumPts,
    const mccnn::fpoint<D>* __restrict__ pPts,
    const int* __restrict__ pBatchIds,
    const int* __restrict__ pSelected,
    int* __restrict__ pNumSampledPoints,
    mccnn::fpoint<D>* __restrict__ pOutPts,
    int* __restrict__ pOutBatchIds,
    int* __restrict__ pOutIndices)
{
    //Get the global thread index.
    int initPtIndex = mccnn::compute_global_index_gpu_funct();
    int totalThreads = mccnn::compute_total_threads_gpu_funct();
 
    for(unsigned int curPtIndex = initPtIndex; 
        curPtIndex < pNumPts; 
        curPtIndex += totalThreads)
    {
        //Check if the point was selected.
        if(pSelected[curPtIndex] == 1){
            //Get the index where to save the current point.
            int saveIndex = atomicAdd(pNumSampledPoints, 1);
            pOutPts[saveIndex] = pPts[curPtIndex];
            pOutBatchIds[saveIndex] = pBatchIds[curPtIndex];
            pOutIndices[saveIndex] = curPtIndex;
        }
    }
}

///////////////////////// CPU
template<int D>
void mccnn::store_sampled_pts_gpu(
    std::unique_ptr<IGPUDevice>& pDevice,
    const unsigned int pNumPts,
    const unsigned int pNumSampledPts,
    const float* pPtsGPUPtr,
    const int* pBatchIdsGPUPtr,
    const int* pSelectedGPUPtr,
    float* pOutPtsGPUPtr,
    int* pOutBatchIdsGPUPtr,
    int* pOutIndicesGPUPtr)
{
    //Get the cuda stream.
    auto cudaStream = pDevice->getCUDAStream();

    //Initialize memory.
    int* tmpCounter = pDevice->getIntTmpGPUBuffer(1);
    pDevice->memset(tmpCounter, 0, sizeof(int));
    pDevice->check_error(__FILE__, __LINE__);

    //Get the device properties.
    const GpuDeviceProperties& gpuProps = pDevice->get_device_properties();

    //Calculate the ideal number of blocks for the selected block size.
    unsigned int numMP = gpuProps.numMPs_;
    unsigned int blockSize = gpuProps.warpSize_*4;
    
    //Calculate the total number of blocks to execute.
    unsigned int execBlocks = pNumPts/blockSize;
    execBlocks += (pNumPts%blockSize != 0)?1:0;
    
    //Execute the cuda kernel.
    unsigned int numBlocks = 0;

    //Calculate the total number of blocks to execute in parallel.
    numBlocks = pDevice->get_max_active_block_x_sm(
        blockSize,(const void*)store_sampled_pts_gpu_kernel<D>, 0);
    pDevice->check_error(__FILE__, __LINE__);
    unsigned int totalNumBlocks = numMP*numBlocks;
    totalNumBlocks = (totalNumBlocks > execBlocks)?execBlocks:totalNumBlocks;

    //Call kernel.
    store_sampled_pts_gpu_kernel<D>
        <<<totalNumBlocks, blockSize, 0, cudaStream>>>(
        pNumPts, (const mccnn::fpoint<D>*)pPtsGPUPtr,
        pBatchIdsGPUPtr, pSelectedGPUPtr, tmpCounter,
        (mccnn::fpoint<D>*)pOutPtsGPUPtr, pOutBatchIdsGPUPtr,
        pOutIndicesGPUPtr);
    pDevice->check_error(__FILE__, __LINE__);
}

#define STORE_SAMPLED_PTS_TEMP_DECL(Dims)                \
    template void mccnn::store_sampled_pts_gpu<Dims>(    \
        std::unique_ptr<IGPUDevice>& pDevice,           \
        const unsigned int pNumPts,                     \
        const unsigned int pNumSampledPts,               \
        const float* pPtsGPUPtr,                        \
        const int* pBatchIdsGPUPtr,                     \
        const int* pSelectedGPUPtr,                     \
        float* pOutPtsGPUPtr,                           \
        int* pOutBatchIdsGPUPtr,                        \
        int* pOutIndicesGPUPtr);

DECLARE_TEMPLATE_DIMS(STORE_SAMPLED_PTS_TEMP_DECL)