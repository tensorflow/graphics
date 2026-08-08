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
/// \brief Implementation of the CUDA operations to store the first point index
///     of each unique key.
/////////////////////////////////////////////////////////////////////////////

#include "tfg_custom_ops/shared/cc/kernels/defines.hpp"
#include "tfg_custom_ops/shared/cc/kernels/math_helper.h"
#include "tfg_custom_ops/shared/cc/kernels/cuda_kernel_utils.h"

#include "tfg_custom_ops/sampling/cc/kernels/store_unique_keys.h"

///////////////////////// GPU

/**
 *  GPU kernel to store in the index of the first point of each cell.
 *  @param  pNumPts         Number of points.
 *  @param  pKeys           Pointer to the array of keys.
 *  @param  pGlobalCounter  Pointer to the global counter.
 *  @param  pOutFIndices    Output pointer with the starting 
 *      index for each unique key.
 */
 __global__ void store_unique_keys_gpu_kernel(
    const unsigned int pNumPts,
    const mccnn::int64_m* __restrict__ pKeys,
    int* __restrict__ pGlobalCounter,
    int* __restrict__ pOutFIndices)
{
    //Get the global thread index.
    int initPtIndex = mccnn::compute_global_index_gpu_funct();
    int totalThreads = mccnn::compute_total_threads_gpu_funct();
 
    for(unsigned int curPtIndex = initPtIndex; 
        curPtIndex < pNumPts; 
        curPtIndex += totalThreads)
    {
        //Check if it is the first element of the cell.
        if(curPtIndex == 0 || pKeys[curPtIndex] != pKeys[curPtIndex-1]){
            int curIndex = atomicAdd(pGlobalCounter, 1);
            pOutFIndices[curIndex] = curPtIndex;
        }
    }
}

///////////////////////// CPU

void mccnn::store_unique_keys_gpu(
    std::unique_ptr<IGPUDevice>& pDevice,
    const unsigned int pNumPts,
    const mccnn::int64_m* pInKeysGPUPtr,
    int* pFIndexKeys)
{
    //Get the cuda stream.
    auto cudaStream = pDevice->getCUDAStream();

    //Get the gpu counter.
    int* tmpCounter = pDevice->getIntTmpGPUBuffer(sizeof(int));
    pDevice->memset(tmpCounter, 0, sizeof(int));
    pDevice->check_error(__FILE__, __LINE__);

    //Get the device properties.
    const GpuDeviceProperties& gpuProps = pDevice->get_device_properties();

    //Calculate the ideal number of blocks for the selected block size.
    unsigned int numMP = gpuProps.numMPs_;
    unsigned int blockSize = gpuProps.warpSize_*4;
    unsigned int numBlocks = pDevice->get_max_active_block_x_sm(
        blockSize,(const void*)store_unique_keys_gpu_kernel, 0);
    pDevice->check_error(__FILE__, __LINE__);

    //Calculate the total number of blocks to execute.
    unsigned int execBlocks = pNumPts/blockSize;
    execBlocks += (pNumPts%blockSize != 0)?1:0;
    unsigned int totalNumBlocks = numMP*numBlocks;
    totalNumBlocks = (totalNumBlocks > execBlocks)?execBlocks:totalNumBlocks;

    //Execute the cuda kernel.
    store_unique_keys_gpu_kernel<<<totalNumBlocks, blockSize, 0, cudaStream>>>(
        pNumPts, pInKeysGPUPtr, tmpCounter, pFIndexKeys);
    pDevice->check_error(__FILE__, __LINE__);
}
