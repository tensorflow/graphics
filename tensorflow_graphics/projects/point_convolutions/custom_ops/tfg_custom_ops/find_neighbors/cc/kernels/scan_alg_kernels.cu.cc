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
/// \brief Implementation of the CUDA operations to execute the parallel scan
///         algorithm in an int array.
/////////////////////////////////////////////////////////////////////////////

#include <vector>

#include "tfg_custom_ops/shared/cc/kernels/cuda_kernel_utils.h"

#include "tfg_custom_ops/find_neighbors/cc/kernels/scan_alg.h"

#define NUM_THREADS 256

///////////////////////// GPU

/**
 *  GPU kernel to execute the parallel scan algorithm in an int array.
 *  Code based on the paper: Parallel Prefix Sum (Scan) with CUDA
 *  https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch39.html
 *  @param  pNumElems               Number of elements in the array.
 *  @param  pNumProcBlocks          Number of blocks to process.
 *  @param  pElems                  Input pointer to the array.
 *  @param  pOutAuxBlockCounter     Output pointer to the array containing
 *      the counter of each block. If null only the individual blocks 
 *      are executed (for arrays of length smaller than T).
 */
 __global__ void scan_alg_gpu_kernel(
    const unsigned int pNumElems,
    const unsigned int pNumProcBlocks,
    int* __restrict__ pElems,
    int* __restrict__ pOutAuxBlockCounter)
{
    //Declare shared memory.
    __shared__ int temp[NUM_THREADS*2];

    //Get the local and global thread index.
    int localThread = threadIdx.x;
    int initElemIndex = mccnn::compute_global_index_gpu_funct();
    int totalThreads = mccnn::compute_total_threads_gpu_funct();
    
    for(int curElemIndex = initElemIndex; 
        curElemIndex < pNumProcBlocks*NUM_THREADS; 
        curElemIndex+=totalThreads)
    {
        //Initialize the offset.
        int offset = 1;

        //Load the global memory into shared memory.
        if((curElemIndex*2) < pNumElems){
            temp[2*localThread] = pElems[2*curElemIndex]; 
        }else{
            temp[2*localThread] = 0; 
        }
        if((curElemIndex*2+1) < pNumElems){
            temp[2*localThread+1] = pElems[2*curElemIndex+1];
        }else{
            temp[2*localThread+1] = 0;
        }

        //Build sum in place up the tree
        for (int d = NUM_THREADS; d > 0; d >>= 1)
        {
            __syncthreads();

            if (localThread < d)
            {
                int ai = offset*(2*localThread+1)-1;
                int bi = offset*(2*localThread+2)-1;
                temp[bi] += temp[ai];
            }

            offset *= 2;
        }

        //Clear the last element
        if (localThread == 0){ 
            temp[(NUM_THREADS*2) - 1] = 0; 
        } 
        
        //Traverse down tree & build scan
        for (int d = 1; d < (NUM_THREADS*2); d *= 2) 
        {
            offset >>= 1;

            __syncthreads();

            if (localThread < d)
            {
                int ai = offset*(2*localThread+1)-1;
                int bi = offset*(2*localThread+2)-1;
                int t = temp[ai];
                temp[ai] = temp[bi];
                temp[bi] += t;
            }
        }

        __syncthreads();

        //Update the block counters.
        if(localThread == 0){
            int blockId = curElemIndex/NUM_THREADS;
            int elemIndex = min((2*NUM_THREADS-1), pNumElems-(curElemIndex*2)-1);
            pOutAuxBlockCounter[blockId] = temp[elemIndex] + pElems[2*curElemIndex+elemIndex];
        }

        __syncthreads();

        //Write the results to device memory
        if((curElemIndex*2) < pNumElems)
            pElems[2*curElemIndex] = temp[2*localThread]; 
        if((curElemIndex*2+1) < pNumElems)
            pElems[2*curElemIndex+1] = temp[2*localThread+1]; 

        __syncthreads();
    }
}

/**
 *  GPU kernel to execute the parallel scan algorithm in an int array.
 *  NOTE: The number of threads per block have to be a divisor of 
 *  pCBlockSize in order to work.
 *  @param  pNumElems               Number of elements in the array.
 *  @param  pCBlockSize             Block size used for the counters.
 *  @param  pBlockCounter           Counter values for each block.
 *  @param  pElems                  Input pointer to the array.
 */
 __global__ void propagate_down_counters_gpu_kernel(
    const unsigned int pNumElems,
    const unsigned int pCBlockSize,
    const int* __restrict__ pBlockCounter,
    int* __restrict__ pElems)
{
    //Declare the shared memory.
    __shared__ int blockOffset;

    //Get the local and global thread index.
    int localThread = threadIdx.x;
    int initElemIndex = mccnn::compute_global_index_gpu_funct();
    int totalThreads = mccnn::compute_total_threads_gpu_funct();
    
    for(int curElemIndex = initElemIndex+pCBlockSize; 
        curElemIndex < pNumElems; 
        curElemIndex+=totalThreads)
    {
        //The the offset of the block.
        if(localThread == 0){
            blockOffset = pBlockCounter[curElemIndex/pCBlockSize];
        }

        __syncthreads();

        //Update the counters.
        pElems[curElemIndex] += blockOffset;
    }
}

///////////////////////// CPU

unsigned int mccnn::scan_alg(
    std::unique_ptr<IGPUDevice>& pDevice,
    const unsigned int pNumElems,
    int* pInGPUPtrElems)
{
    //Get the cuda stream.
    auto cudaStream = pDevice->getCUDAStream();

    //Get the device properties.
    const GpuDeviceProperties& gpuProps = pDevice->get_device_properties();

    //Calculate the ideal number of blocks for the selected block size.
    unsigned int numMP = gpuProps.numMPs_;
    unsigned int blockSize = NUM_THREADS;
    unsigned int numBlocks = pDevice->get_max_active_block_x_sm(
        blockSize,
        (const void*)scan_alg_gpu_kernel, 
        blockSize*2*sizeof(int)); 
    pDevice->check_error(__FILE__, __LINE__);

    //Compute the hierarchical scan.
    unsigned int numIterations = 0;
    int curNumElems = pNumElems;
    int* curGPUPtr = pInGPUPtrElems;
    std::vector<std::pair<int*, int> > counters;
    counters.push_back(std::make_pair(curGPUPtr, curNumElems));
    while(curNumElems > 1 || numIterations == 0)
    {
        //Calculate the total number of blocks to execute.
        unsigned int execBlocks = curNumElems/(blockSize*2);
        execBlocks += (curNumElems%(blockSize*2) != 0)?1:0;
        unsigned int totalNumBlocks = numMP*numBlocks;
        totalNumBlocks = (totalNumBlocks > execBlocks)?execBlocks:totalNumBlocks;

        //Create the auxiliar counter array.
        int* auxiliarCoutner = pDevice->getIntTmpGPUBuffer(execBlocks);
        pDevice->memset(auxiliarCoutner, 0, sizeof(int)*execBlocks);

        //Store the auxiliar counter in the vector.
        counters.push_back(std::make_pair(auxiliarCoutner, execBlocks));

        //Execute the cuda kernel.
        scan_alg_gpu_kernel<<<totalNumBlocks, blockSize, 0, cudaStream>>>(
            curNumElems,
            execBlocks,
            curGPUPtr,
            auxiliarCoutner);

        pDevice->check_error(__FILE__, __LINE__);

        //Set the variables for the next pass.
        curNumElems = execBlocks;
        curGPUPtr = auxiliarCoutner;

        //Increment the iteration counter.
        numIterations++;
    }

    //Propagate down the counters.
    unsigned int propBlockSize = gpuProps.warpSize_*2;
    for(int i = counters.size()-1; i > 0; i--)
    {
        curNumElems = counters[i-1].second-(blockSize*2);
        if(curNumElems > 0){
            //Calculate the total number of blocks to execute.
            unsigned int execBlocks = curNumElems/propBlockSize;
            execBlocks += (curNumElems%propBlockSize != 0)?1:0;
            unsigned int totalNumBlocks = numMP*numBlocks;
            totalNumBlocks = (totalNumBlocks > execBlocks)?execBlocks:totalNumBlocks;

            //Propagate the offsets.
            propagate_down_counters_gpu_kernel<<<totalNumBlocks, propBlockSize, 0, cudaStream>>>(
                counters[i-1].second,
                blockSize*2,
                counters[i].first,
                counters[i-1].first);
        }
    }

    //Get the total accumulated value.
    int* accumScan = pDevice->getIntTmpGPUBuffer(1, true);
    pDevice->memcpy_device_to_host(
        (void*)accumScan, 
        (void*)counters[counters.size()-1].first, 
        sizeof(int));

    //Wait for the result.
    cudaEvent_t resEvent;
    cudaEventCreate(&resEvent);
    cudaEventRecord(resEvent, cudaStream);
    cudaEventSynchronize(resEvent);

    return accumScan[0];
}