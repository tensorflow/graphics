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
/// \brief Implementation of the CUDA operations to store the neighbors for 
///         each point.
/////////////////////////////////////////////////////////////////////////////

#include <time.h>
#include "tfg_custom_ops/shared/cc/kernels/defines.hpp"
#include "tfg_custom_ops/shared/cc/kernels/math_helper.h"
#include "tfg_custom_ops/shared/cc/kernels/cuda_kernel_utils.h"
#include "tfg_custom_ops/shared/cc/kernels/rnd_utils.h"

#include "tfg_custom_ops/find_neighbors/cc/kernels/store_neighbors.h"

#define NUM_THREADS_X_RANGE 16

///////////////////////// GPU

/**
 *  GPU kernel to store the number of neighbors for each sample.
 *  @param  pSeed           Seed used to initialize the random state.
 *  @param  pMaxNeighbors   Maximum neighbors allowed per sample.
 *  @param  pNumSamples     Number of samples.
 *  @param  pNumRanges      Number of ranges per sample.
 *  @param  pSamples        3D coordinates of each sample.
 *  @param  pPts            3D coordinates of each point.
 *  @param  pRanges         Search ranges for each sample.
 *  @param  pInvRadii       Inverse of the radius used on the 
 *      search of neighbors in each dimension.
 *  @param  pOutNumNeighsU  Number of neighbors for each sample
 *      without the limit imposed by pMaxNeighbors.
 *  @param  pAuxCounter     Auxiliar counter.
 *  @param  pOutNumNeighs   Number of neighbors for each sample.
 *  @param  pOutNeighs      Final beighbors.
 *  @tparam D               Number of dimensions.
 */
 template<int D>
 __global__ void store_neighbors_limited_gpu_kernel(
    const int pSeed,
    const int pMaxNeighbors,
    const unsigned int pNumSamples,
    const unsigned int pNumRanges,
    const mccnn::fpoint<D>* __restrict__ pSamples,
    const mccnn::fpoint<D>* __restrict__ pPts,
    const int2* __restrict__ pRanges,
    const mccnn::fpoint<D>* __restrict__ pInvRadii,
    const int* __restrict__ pOutNumNeighsU,
    int* __restrict__ pAuxCounter,
    int* __restrict__ pOutNumNeighs,
    int2* __restrict__ pOutNeighs)
{
    //Get the global thread index.
    int initSampleIndex = mccnn::compute_global_index_gpu_funct();
    int totalThreads = mccnn::compute_total_threads_gpu_funct();

    //Initialize the random seed generator.
    int curSeed = mccnn::wang_hash(pSeed+initSampleIndex);

    for(long long curIter = initSampleIndex; 
        curIter < pNumSamples*pNumRanges*NUM_THREADS_X_RANGE; 
        curIter += totalThreads)
    {        
        //Get the point id.
        int sampleIndex = curIter/(pNumRanges*NUM_THREADS_X_RANGE);

        //Get the offset and index of the local counter.
        int localIndex = curIter%NUM_THREADS_X_RANGE;

        //Get the current sample coordinates and the search range.
        mccnn::fpoint<D> curSampleCoords = pSamples[sampleIndex];
        int2 curRange = pRanges[curIter/NUM_THREADS_X_RANGE];

        //Iterate over the points.
        for(int curPtIter = curRange.x+localIndex; 
            curPtIter < curRange.y; curPtIter+=NUM_THREADS_X_RANGE)
        {
            //Check if the point is closer than the selected radius.
            mccnn::fpoint<D> curPtCoors = pPts[curPtIter];
            if(length((curSampleCoords - curPtCoors)*pInvRadii[0]) < 1.0f){
                //Increment the shared counters.
                int neighIndex = atomicAdd(&pAuxCounter[sampleIndex], 1);
                if(neighIndex < pMaxNeighbors){
                    neighIndex = atomicAdd(&pOutNumNeighs[sampleIndex], 1);
                    pOutNeighs[neighIndex] = make_int2(curPtIter, sampleIndex);
                }else{
                    float ratio = (float)pMaxNeighbors/(float)pOutNumNeighsU[sampleIndex];
                    curSeed = mccnn::rand_xorshift(curSeed);
                    float uniformVal = mccnn::seed_to_float(curSeed);
                    if(uniformVal < ratio){
                        curSeed = mccnn::rand_xorshift(curSeed);
                        uniformVal = mccnn::seed_to_float(curSeed);
                        neighIndex = (int)floor(uniformVal*((float)pMaxNeighbors))+1;
                        pOutNeighs[pOutNumNeighs[sampleIndex] - neighIndex] = make_int2(curPtIter, sampleIndex);
                    }
                }
            }
        }
    }
}

/**
 *  GPU kernel to store the number of neighbors for each sample.
 *  @param  pNumSamples     Number of samples.
 *  @param  pNumRanges      Number of ranges per sample.
 *  @param  pSamples        3D coordinates of each sample.
 *  @param  pPts            3D coordinates of each point.
 *  @param  pRanges         Search ranges for each sample.
 *  @param  pInvRadii       Inverse of the radius used on the 
 *      search of neighbors in each dimension.
 *  @param  pOutNumNeighs   Number of neighbors for each sample.
 *  @param  pOutNeighs      Final beighbors.
 *  @tparam D               Number of dimensions.
 */
 template<int D>
 __global__ void store_neighbors_gpu_kernel(
    const unsigned int pNumSamples,
    const unsigned int pNumRanges,
    const mccnn::fpoint<D>* __restrict__ pSamples,
    const mccnn::fpoint<D>* __restrict__ pPts,
    const int2* __restrict__ pRanges,
    const mccnn::fpoint<D>* __restrict__ pInvRadii,
    int* __restrict__ pOutNumNeighs,
    int2* __restrict__ pOutNeighs)
{
    //Get the global thread index.
    int initSampleIndex = mccnn::compute_global_index_gpu_funct();
    int totalThreads = mccnn::compute_total_threads_gpu_funct();

    for(long long curIter = initSampleIndex; 
        curIter < pNumSamples*pNumRanges*NUM_THREADS_X_RANGE; 
        curIter += totalThreads)
    {        
        //Get the point id.
        int sampleIndex = curIter/(pNumRanges*NUM_THREADS_X_RANGE);

        //Get the offset and index of the local counter.
        int localIndex = curIter%NUM_THREADS_X_RANGE;

        //Get the current sample coordinates and the search range.
        mccnn::fpoint<D> curSampleCoords = pSamples[sampleIndex];
        int2 curRange = pRanges[curIter/NUM_THREADS_X_RANGE];

        //Iterate over the points.
        for(int curPtIter = curRange.x+localIndex; 
            curPtIter < curRange.y; curPtIter+=NUM_THREADS_X_RANGE)
        {
            //Check if the point is closer than the selected radius.
            mccnn::fpoint<D> curPtCoors = pPts[curPtIter];
            if(length((curSampleCoords - curPtCoors)*pInvRadii[0]) < 1.0f){
                //Increment the shared counters.
                int neighIndex = atomicAdd(&pOutNumNeighs[sampleIndex], 1);
                pOutNeighs[neighIndex] = make_int2(curPtIter, sampleIndex);
            }
        }
    }
}

///////////////////////// CPU

template<int D>
void mccnn::store_neighbors(
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
    int* pOutGPUPtrNeighs)
{
    //Get the cuda stream.
    auto cudaStream = pDevice->getCUDAStream();

    //Get the device properties.
    const GpuDeviceProperties& gpuProps = pDevice->get_device_properties();

    unsigned int numBlocks;
    unsigned int blockSize;
    if(pMaxNeighbors > 0){
        //Create a temporal tensor.
        int* tmpVect = pDevice->getIntTmpGPUBuffer(pNumSamples);
        pDevice->memset(tmpVect, 0, sizeof(int)*pNumSamples);
        pDevice->check_error(__FILE__, __LINE__);

        //Calculate the ideal number of blocks for the selected block size.
        unsigned int numMP = gpuProps.numMPs_;
        blockSize = gpuProps.warpSize_*2;
        numBlocks = pDevice->get_max_active_block_x_sm(
            blockSize, (const void*)store_neighbors_limited_gpu_kernel<D>, 0);
        pDevice->check_error(__FILE__, __LINE__);

        //Calculate the total number of blocks to execute.
        unsigned int execBlocks = (pNumSamples*pNumRanges*NUM_THREADS_X_RANGE)/blockSize;
        execBlocks += ((pNumSamples*pNumRanges*NUM_THREADS_X_RANGE)%blockSize != 0)?1:0;
        unsigned int totalNumBlocks = numMP*numBlocks;
        totalNumBlocks = (totalNumBlocks > execBlocks)?execBlocks:totalNumBlocks;

        //Execute the cuda kernel.
        store_neighbors_limited_gpu_kernel<D><<<totalNumBlocks, blockSize, 0, cudaStream>>>(
            time(NULL),
            pMaxNeighbors,
            pNumSamples,
            pNumRanges,
            (const mccnn::fpoint<D>*)pInGPUPtrSamples, 
            (const mccnn::fpoint<D>*)pInGPUPtrPts, 
            (const int2*)pInGPUPtrRanges,
            (const mccnn::fpoint<D>*)pInGPUPtrInvRadii,
            pOutGPUPtrNumNeighsU,
            tmpVect,
            pOutGPUPtrNumNeighs,
            (int2*)pOutGPUPtrNeighs);
        pDevice->check_error(__FILE__, __LINE__);
    }else{
        //Calculate the ideal number of blocks for the selected block size.
        unsigned int numMP = gpuProps.numMPs_;
        blockSize = gpuProps.warpSize_*2;
        numBlocks = pDevice->get_max_active_block_x_sm(
            blockSize, (const void*)store_neighbors_gpu_kernel<D>, 0);
        pDevice->check_error(__FILE__, __LINE__);

        //Calculate the total number of blocks to execute.
        unsigned int execBlocks = (pNumSamples*pNumRanges*NUM_THREADS_X_RANGE)/blockSize;
        execBlocks += ((pNumSamples*pNumRanges*NUM_THREADS_X_RANGE)%blockSize != 0)?1:0;
        unsigned int totalNumBlocks = numMP*numBlocks;
        totalNumBlocks = (totalNumBlocks > execBlocks)?execBlocks:totalNumBlocks;

        //Execute the cuda kernel.
        store_neighbors_gpu_kernel<D><<<totalNumBlocks, blockSize, 0, cudaStream>>>(
            pNumSamples,
            pNumRanges,
            (const mccnn::fpoint<D>*)pInGPUPtrSamples, 
            (const mccnn::fpoint<D>*)pInGPUPtrPts, 
            (const int2*)pInGPUPtrRanges,
            (const mccnn::fpoint<D>*)pInGPUPtrInvRadii,
            pOutGPUPtrNumNeighs,
            (int2*)pOutGPUPtrNeighs);
        pDevice->check_error(__FILE__, __LINE__);
    }

#ifdef DEBUG_INFO
    cudaEventRecord(stop, cudaStream);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    float gpuOccupancy = (float)(numBlocks*blockSize)/(float)gpuProps.maxThreadsXMP_;

    fprintf(stderr, "### STORE NEIGHBORS ###\n");
    fprintf(stderr, "Num samples: %d\n", pNumSamples);
    fprintf(stderr, "Occupancy: %f\n", gpuOccupancy);
    fprintf(stderr, "Execution time: %f\n", milliseconds);
    fprintf(stderr, "\n");
#endif
}

///////////////////////// CPU Template declaration

#define STORE_NEIGHS_TEMP_DECL(Dims)            \
    template void mccnn::store_neighbors<Dims>( \
        std::unique_ptr<IGPUDevice>& pDevice,   \
        const int pMaxNeighbors,                \
        const unsigned int pNumSamples,         \
        const unsigned int pNumRanges,          \
        const float* pInGPUPtrSamples,          \
        const float* pInGPUPtrPts,              \
        const int* pInGPUPtrRanges,             \
        const float* pInGPUPtrInvRadii,         \
        int* pOutGPUPtrNumNeighsU,              \
        int* pOutGPUPtrNumNeighs,               \
        int* pOutGPUPtrNeighs);

DECLARE_TEMPLATE_DIMS(STORE_NEIGHS_TEMP_DECL)