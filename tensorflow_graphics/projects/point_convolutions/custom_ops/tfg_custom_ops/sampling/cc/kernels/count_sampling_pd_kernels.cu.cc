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
/// \brief Implementation of the CUDA operations to sample a set of points 
///     from a point cloud. 
/////////////////////////////////////////////////////////////////////////////

#include "tfg_custom_ops/shared/cc/kernels/defines.hpp"
#include "tfg_custom_ops/shared/cc/kernels/math_helper.h"
#include "tfg_custom_ops/shared/cc/kernels/cuda_kernel_utils.h"
#include "tfg_custom_ops/shared/cc/kernels/grid_utils.h"

#include "tfg_custom_ops/sampling/cc/kernels/count_sampling_pd.h"

///////////////////////// GPU

/**
 *  GPU kernel to count the unique keys in a list.
 *  @param  pNumPts             Number of points.
 *  @param  pNumSampledPts       Number of sampled points.
 *  @param  pKeys               Pointer to the array of keys.
 *  @param  pPts                Input points.
 *  @param  pNumCells           Number of cells.
 *  @param  pUniqueKeyIndices   Input index of the first point of 
 *      each cell.
 *  @param  pOutPts             Output sampled points.
 *  @param  pOutBatchIds        Output batch ids.
 */
 template <int D>
 __global__ void count_sampled_pd_gpu_kernel(
    const unsigned int pNumPts,
    const unsigned int pNumUniqueKeys,
    const mccnn::ipoint<D> pCurrentCellBlock,
    const mccnn::int64_m* __restrict__ pKeys,
    const mccnn::fpoint<D>* __restrict__ pPts,
    const mccnn::ipoint<D>* __restrict__ pNumCells,
    const int2* __restrict__ pNeighbors,
    const int* __restrict__ pNeighPtIndices,
    const int* __restrict__ pUniqueKeyIndices,
    int* __restrict__ pNumSampledPoints,
    int* __restrict__ pUsed)
{
    //Get the global thread index.
    int initPtIndex = mccnn::compute_global_index_gpu_funct();
    int totalThreads = mccnn::compute_total_threads_gpu_funct();
 
    for(unsigned int curPtIndex = initPtIndex; 
        curPtIndex < pNumUniqueKeys; 
        curPtIndex += totalThreads)
    {
        //Find the start index.
        int curPtCellIter = pUniqueKeyIndices[curPtIndex];

        //Get the key and batch id.
        mccnn::int64_m curKey = pKeys[curPtCellIter];
        mccnn::ipoint<D+1> cellIndex = mccnn::compute_cell_from_key_gpu_funct(
            curKey, pNumCells[0]);

        //Check that it is a valid cell.
        bool validCell = true;
#pragma unroll
        for(int dimIter = 0; dimIter < D; ++dimIter)
            validCell = validCell && ((cellIndex[dimIter+1]%2)==pCurrentCellBlock[dimIter]);

        if(validCell){

            //Iterate until there are no more points in the cell.
            while(curPtCellIter < pNumPts && pKeys[curPtCellIter] == curKey)
            {
                //Check the range of neighbors.
                int startNeighIndex = (curPtCellIter > 0)? 
                    pNeighPtIndices[curPtCellIter-1]: 0;
                int endNeighIndex = pNeighPtIndices[curPtCellIter];

                //Check if some of the neighbors have been previously selected.
                bool valid = true;
                for(int neighIter = startNeighIndex; 
                    neighIter < endNeighIndex;
                    ++neighIter)
                {
                    int neighIndex = pNeighbors[neighIter].x;
                    valid = valid && (pUsed[neighIndex] == 0);
                }
                
                //If this is a valid point, we select it.
                if(valid){
                    atomicAdd(pNumSampledPoints, 1);
                    pUsed[curPtCellIter] = 1;
                }

                //Increment counter.
                curPtCellIter++;
            }
        }
    }
}

///////////////////////// CPU

template <int D>
void mccnn::count_sampling_pd_gpu(
    std::unique_ptr<IGPUDevice>& pDevice,
    const unsigned int pNumPts,
    const unsigned int pNumUniqueKeys,
    const int* pUniqueKeyIndexs,
    const mccnn::int64_m* pInKeysGPUPtr,
    const float* pPtsGPUPtr,
    const int* pNeighbors,
    const int* pNeighStartIndex,
    const int* pNumCellsGPUPtr,
    int& pOutNumSampledPts,
    int* pSelectedGPUPtr)
{
    //Get the cuda stream.
    auto cudaStream = pDevice->getCUDAStream();

    //Initialize memory.
    int* tmpCounter = pDevice->getIntTmpGPUBuffer(1);
    pDevice->memset(tmpCounter, 0, sizeof(int));
    pDevice->memset(pSelectedGPUPtr, 0, sizeof(int)*pNumPts);
    pDevice->check_error(__FILE__, __LINE__);

    //Get the device properties.
    const GpuDeviceProperties& gpuProps = pDevice->get_device_properties();

    //Calculate the ideal number of blocks for the selected block size.
    unsigned int numMP = gpuProps.numMPs_;
    unsigned int blockSize = gpuProps.warpSize_*4;
    
    //Calculate the total number of blocks to execute.
    unsigned int execBlocks = pNumUniqueKeys/blockSize;
    execBlocks += (pNumUniqueKeys%blockSize != 0)?1:0;
    
    //Execute the cuda kernel.
    unsigned int numBlocks = 0;

    //Calculate the total number of blocks to execute in parallel.
    numBlocks = pDevice->get_max_active_block_x_sm(
        blockSize,(const void*)count_sampled_pd_gpu_kernel<D>, 0);
    pDevice->check_error(__FILE__, __LINE__);
    unsigned int totalNumBlocks = numMP*numBlocks;
    totalNumBlocks = (totalNumBlocks > execBlocks)?execBlocks:totalNumBlocks;

    //Compute the number of calls we need to do.
    unsigned int numCalls = 1;
    for(int i = 0; i < D; ++i)
        numCalls *= 2;

    //Iterate over the different calls required.
    for(int callIter = 0; callIter < numCalls; ++callIter)
    {
        //Compute valid cells mod.
        int auxInt = callIter;
        mccnn::ipoint<D> validCode;
        for(int i = 0; i < D; ++i){
            validCode[i] = auxInt%2;
            auxInt = auxInt/2;
        }

        //Call kernel.
        count_sampled_pd_gpu_kernel<D>
            <<<totalNumBlocks, blockSize, 0, cudaStream>>>(
            pNumPts, pNumUniqueKeys, validCode,
            pInKeysGPUPtr, (const mccnn::fpoint<D>*)pPtsGPUPtr,
            (const mccnn::ipoint<D>*)pNumCellsGPUPtr,
            (const int2*)pNeighbors, pNeighStartIndex,
            pUniqueKeyIndexs, tmpCounter, pSelectedGPUPtr);
    }
    pDevice->check_error(__FILE__, __LINE__);

    //Copy the number of points into cpu memory.
    int* auxNumSampledPts = pDevice->getIntTmpGPUBuffer(1, true);
    pDevice->memcpy_device_to_host(
        (void*)auxNumSampledPts,
        (void*)tmpCounter,
        sizeof(int));

    //Wait for the result.
    cudaEvent_t resEvent;
    cudaEventCreate(&resEvent);
    cudaEventRecord(resEvent, cudaStream);
    cudaEventSynchronize(resEvent);

    //Collect the result.
    pOutNumSampledPts = auxNumSampledPts[0];
}

#define COUNT_SAMPLING_PD_TEMP_DECL(Dims)                \
    template void mccnn::count_sampling_pd_gpu<Dims>(    \
        std::unique_ptr<IGPUDevice>& pDevice,           \
        const unsigned int pNumPts,                     \
        const unsigned int pNumUniqueKeys,              \
        const int* pUniqueKeyIndexs,                    \
        const mccnn::int64_m* pInKeysGPUPtr,            \
        const float* pPtsGPUPtr,                        \
        const int* pNeighbors,                          \
        const int* pNeighStartIndex,                    \
        const int* pNumCellsGPUPtr,                     \
        int& pOutNumSampledPts,                          \
        int* pSelectedGPUPtr);

DECLARE_TEMPLATE_DIMS(COUNT_SAMPLING_PD_TEMP_DECL)