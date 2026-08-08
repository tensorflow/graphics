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
/// \brief Implementation of the CUDA operations to count the unique keys  
///     in a sorted array of keys.
/////////////////////////////////////////////////////////////////////////////

#include "tfg_custom_ops/shared/cc/kernels/defines.hpp"
#include "tfg_custom_ops/shared/cc/kernels/math_helper.h"
#include "tfg_custom_ops/shared/cc/kernels/cuda_kernel_utils.h"

#include "tfg_custom_ops/sampling/cc/kernels/count_unique_keys.h"

///////////////////////// GPU

/**
 *  GPU kernel to count the unique keys in a list.
 *  @param  pNumPts         Number of points.
 *  @param  pKeys           Pointer to the array of keys.
 *  @param  pGlobalCounter  Pointer to the global counter.
 */
 __global__ void count_unique_keys_gpu_kernel(
    const unsigned int pNumPts,
    const mccnn::int64_m* __restrict__ pKeys,
    int* __restrict__ pGlobalCounter)
{
    __shared__ int localCounter;

    //Initialize the local counter.
    if(threadIdx.x == 0){
        localCounter = 0;
    }

    __syncthreads();

    //Get the global thread index.
    int initPtIndex = mccnn::compute_global_index_gpu_funct();
    int totalThreads = mccnn::compute_total_threads_gpu_funct();
 
    for(unsigned int curPtIndex = initPtIndex; 
        curPtIndex < pNumPts-1; 
        curPtIndex += totalThreads)
    {
        //Check if it is the last element of the cell.
        if(pKeys[curPtIndex] != pKeys[curPtIndex+1]){
            atomicAdd(&localCounter, 1);
        }
    }

    __syncthreads();

    //Add the local counter into the glocal counter.
    if(threadIdx.x == 0){
        atomicAdd(pGlobalCounter, localCounter);
    }
}

///////////////////////// CPU

unsigned int mccnn::count_unique_keys_gpu(
    std::unique_ptr<IGPUDevice>& pDevice,
    const unsigned int pNumPts,
    const mccnn::int64_m* pInKeysGPUPtr)
{
    //Get the cuda stream.
    auto cudaStream = pDevice->getCUDAStream();

    //Get the gpu counter.
    int* tmpCounter = pDevice->getIntTmpGPUBuffer(1);
    pDevice->memset(tmpCounter, 0, sizeof(int));
    pDevice->check_error(__FILE__, __LINE__);

    //Get the device properties.
    const GpuDeviceProperties& gpuProps = pDevice->get_device_properties();

    //Calculate the ideal number of blocks for the selected block size.
    unsigned int numMP = gpuProps.numMPs_;
    unsigned int blockSize = gpuProps.warpSize_*4;
    unsigned int numBlocks = pDevice->get_max_active_block_x_sm(
        blockSize,(const void*)count_unique_keys_gpu_kernel, sizeof(int));
    pDevice->check_error(__FILE__, __LINE__);

    //Calculate the total number of blocks to execute.
    unsigned int execBlocks = pNumPts/blockSize;
    execBlocks += (pNumPts%blockSize != 0)?1:0;
    unsigned int totalNumBlocks = numMP*numBlocks;
    totalNumBlocks = (totalNumBlocks > execBlocks)?execBlocks:totalNumBlocks;

    //Execute the cuda kernel.
    count_unique_keys_gpu_kernel<<<totalNumBlocks, blockSize, 0, cudaStream>>>(
        pNumPts, pInKeysGPUPtr, tmpCounter);
    pDevice->check_error(__FILE__, __LINE__);

    //Get the total number of keys.
    int* numUniqueKeys = pDevice->getIntTmpGPUBuffer(1, true);;
    pDevice->memcpy_device_to_host(
        (void*)numUniqueKeys,
        (void*)tmpCounter,
        sizeof(int));

    //Wait for the result.
    cudaEvent_t resEvent;
    cudaEventCreate(&resEvent);
    cudaEventRecord(resEvent, cudaStream);
    cudaEventSynchronize(resEvent);

    return numUniqueKeys[0]+1;
}
