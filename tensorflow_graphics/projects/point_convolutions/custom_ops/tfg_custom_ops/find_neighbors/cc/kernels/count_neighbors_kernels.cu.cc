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
/// \brief Implementation of the CUDA operations to count the neighbors for 
///         each point.
/////////////////////////////////////////////////////////////////////////////

#include "tfg_custom_ops/shared/cc/kernels/defines.hpp"
#include "tfg_custom_ops/shared/cc/kernels/math_helper.h"
#include "tfg_custom_ops/shared/cc/kernels/cuda_kernel_utils.h"

#include "tfg_custom_ops/find_neighbors/cc/kernels/count_neighbors.h"

#define NUM_THREADS_X_RANGE 16

///////////////////////// GPU

/**
 *  GPU kernel to count the number of neighbors for each sample.
 *  @param  pNumSamples     Number of samples.
 *  @param  pNumRanges      Number of ranges per sample.
 *  @param  pSamples        3D coordinates of each sample.
 *  @param  pPts            3D coordinates of each point.
 *  @param  pRanges         Search ranges for each sample.
 *  @param  pInvRadii       Inverse of the radius used on the 
 *      search of neighbors in each dimension.
 *  @param  pOutNumNeighs   Number of neighbors for each sample.
 *  @tparam D               Number of dimensions
 */
 template<int D>
 __global__ void count_neighbors_gpu_kernel(
    const unsigned int pNumSamples,
    const unsigned int pNumRanges,
    const mccnn::fpoint<D>* __restrict__ pSamples,
    const mccnn::fpoint<D>* __restrict__ pPts,
    const int2* __restrict__ pRanges,
    const mccnn::fpoint<D>* __restrict__ pInvRadii,
    int* __restrict__ pOutNumNeighs)
{
    //Declare shared memory.
    extern __shared__ int localCounter[];

    //Get the global thread index.
    int initSampleIndex = mccnn::compute_global_index_gpu_funct();
    int totalThreads = mccnn::compute_total_threads_gpu_funct();

    for(long long curIter = initSampleIndex; 
        curIter < pNumSamples*pNumRanges*NUM_THREADS_X_RANGE; curIter += totalThreads)
    {        
        //Get the point id.
        int sampleIndex = curIter/(pNumRanges*NUM_THREADS_X_RANGE);
        
        //Initialize the shared memory
        localCounter[threadIdx.x] = sampleIndex;
        localCounter[blockDim.x + threadIdx.x] = 0;

        __syncthreads();

        //Get the offset and index of the local counter.
        int localIndex = curIter%NUM_THREADS_X_RANGE;
        int offsetCounter = sampleIndex-localCounter[0];

        //Get the current sample coordinates and the search range.
        mccnn::fpoint<D> curSampleCoords = pSamples[sampleIndex];
        int2 curRange = pRanges[curIter/NUM_THREADS_X_RANGE];

        //Iterate over the points.
        for(int curPtIter = curRange.x+localIndex; 
            curPtIter < curRange.y; curPtIter+=NUM_THREADS_X_RANGE)
        {
            //Check if the point is closer than the selected radius.
            mccnn::fpoint<D> curPtCoords = pPts[curPtIter];
            if(length((curSampleCoords - curPtCoords)*pInvRadii[0]) < 1.0f){
                //Increment the shared counters.
                atomicAdd(&localCounter[blockDim.x+offsetCounter], 1);
            }
        }

        __syncthreads();

        //Update the global counters.
        if(threadIdx.x == 0){
            atomicAdd(&pOutNumNeighs[sampleIndex], localCounter[blockDim.x]);
        }else if(sampleIndex != localCounter[threadIdx.x-1]){
            atomicAdd(&pOutNumNeighs[sampleIndex], localCounter[blockDim.x+offsetCounter]);
        }

        __syncthreads();
    }
}

///////////////////////// CPU

template<int D>
void mccnn::count_neighbors(
    std::unique_ptr<IGPUDevice>& pDevice,
    const unsigned int pNumSamples,
    const unsigned int pNumRanges,
    const float* pInGPUPtrSamples,
    const float* pInGPUPtrPts,
    const int* pInGPUPtrRanges,
    const float* pInGPUPtrInvRadii,
    int* pOutGPUPtrNumNeighs)
{
    //Get the cuda stream.
    auto cudaStream = pDevice->getCUDAStream();

    //Get the device properties.
    const GpuDeviceProperties& gpuProps = pDevice->get_device_properties();

    //Calculate the ideal number of blocks for the selected block size.
    unsigned int numMP = gpuProps.numMPs_;
    unsigned int blockSize = gpuProps.warpSize_*2;
    unsigned int numBlocks = pDevice->get_max_active_block_x_sm(
        blockSize,
        (const void*)count_neighbors_gpu_kernel<D>, 
        blockSize*2*sizeof(int));
    pDevice->check_error(__FILE__, __LINE__);

    //Calculate the total number of blocks to execute.
    unsigned int execBlocks = (pNumSamples*pNumRanges*NUM_THREADS_X_RANGE)/blockSize;
    execBlocks += ((pNumSamples*pNumRanges*NUM_THREADS_X_RANGE)%blockSize != 0)?1:0;
    unsigned int totalNumBlocks = numMP*numBlocks;
    totalNumBlocks = (totalNumBlocks > execBlocks)?execBlocks:totalNumBlocks;

    //Execute the cuda kernel.
    count_neighbors_gpu_kernel<D><<<totalNumBlocks, blockSize, blockSize*2*sizeof(int), cudaStream>>>(
        pNumSamples,
        pNumRanges,
        (const mccnn::fpoint<D>*)pInGPUPtrSamples, 
        (const mccnn::fpoint<D>*)pInGPUPtrPts, 
        (const int2*)pInGPUPtrRanges,
        (const mccnn::fpoint<D>*)pInGPUPtrInvRadii,
        pOutGPUPtrNumNeighs);
    pDevice->check_error(__FILE__, __LINE__);
}

///////////////////////// CPU Template declaration

#define COUNT_NEIGHS_TEMP_DECL(Dims)            \
    template void mccnn::count_neighbors<Dims>( \
        std::unique_ptr<IGPUDevice>& pDevice,   \
        const unsigned int pNumSamples,         \
        const unsigned int pNumRanges,          \
        const float* pInGPUPtrSamples,          \
        const float* pInGPUPtrPts,              \
        const int* pInGPUPtrRanges,             \
        const float* pInvRadii,                 \
        int* pOutGPUPtrNumNeighs);

DECLARE_TEMPLATE_DIMS(COUNT_NEIGHS_TEMP_DECL)