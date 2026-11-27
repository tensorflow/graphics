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

#include "tfg_custom_ops/sampling/cc/kernels/sampling_avg.h"

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
 __global__ void sampling_avg_gpu_kernel(
    const unsigned int pNumPts,
    const unsigned int pNumSampledPts,
    const mccnn::int64_m* __restrict__ pKeys,
    const mccnn::fpoint<D>* __restrict__ pPts,
    const mccnn::ipoint<D>* __restrict__ pNumCells,
    const int* __restrict__ pUniqueKeyIndices,
    mccnn::fpoint<D>* __restrict__ pOutPts,
    int* __restrict__ pOutBatchIds)
{
    //Get the global thread index.
    int initPtIndex = mccnn::compute_global_index_gpu_funct();
    int totalThreads = mccnn::compute_total_threads_gpu_funct();
 
    for(unsigned int curPtIndex = initPtIndex; 
        curPtIndex < pNumSampledPts; 
        curPtIndex += totalThreads)
    {
        //Find the start index.
        int curPtCellIter = pUniqueKeyIndices[curPtIndex];

        //Get the key and batch id.
        mccnn::int64_m curKey = pKeys[curPtCellIter];
        mccnn::int64_m totalNumCells = mccnn::compute_total_num_cells_gpu_funct(pNumCells[0]);
        int batchId = curKey/totalNumCells;
        
        //Declare the initial point coordinate and the counter of points.
        mccnn::fpoint<D> curOutPt(0.0f);
        int curNumPts = 0;

        //Iterate until there are no more points in the cell.
        while(curPtCellIter < pNumPts && pKeys[curPtCellIter] == curKey)
        {
            curOutPt += pPts[curPtCellIter];
            curNumPts++;
            curPtCellIter++;
        }

        //Finish the selection process.
        curOutPt = curOutPt/((float)curNumPts);

        //Store the result.
        pOutPts[curPtIndex] = curOutPt;
        pOutBatchIds[curPtIndex] = batchId;
    }
}

///////////////////////// CPU

template <int D>
void mccnn::sampling_avg_gpu(
    std::unique_ptr<IGPUDevice>& pDevice,
    const unsigned int pNumPts,
    const unsigned int pNumSampledPts,
    const mccnn::int64_m* pInKeysGPUPtr,
    const float* pPtsGPUPtr,
    const int* pNumCellsGPUPtr,
    const int* pUniqueKeyIndexs,
    float* pOutPtsGPUPtr,
    int* pBatchIdsGPUPtr)
{
    //Get the cuda stream.
    auto cudaStream = pDevice->getCUDAStream();

    //Get the device properties.
    const GpuDeviceProperties& gpuProps = pDevice->get_device_properties();

    //Calculate the ideal number of blocks for the selected block size.
    unsigned int numMP = gpuProps.numMPs_;
    unsigned int blockSize = gpuProps.warpSize_*4;
    
    //Calculate the total number of blocks to execute.
    unsigned int execBlocks = pNumSampledPts/blockSize;
    execBlocks += (pNumSampledPts%blockSize != 0)?1:0;
    
    //Execute the cuda kernel.
    unsigned int numBlocks = 0;

    //Calculate the total number of blocks to execute in parallel.
    numBlocks = pDevice->get_max_active_block_x_sm(
        blockSize,(const void*)sampling_avg_gpu_kernel<D>, 0);
    pDevice->check_error(__FILE__, __LINE__);
    unsigned int totalNumBlocks = numMP*numBlocks;
    totalNumBlocks = (totalNumBlocks > execBlocks)?execBlocks:totalNumBlocks;

    //Execute the kernel.
    sampling_avg_gpu_kernel<D><<<totalNumBlocks, blockSize, 0, cudaStream>>>(
        pNumPts, pNumSampledPts, pInKeysGPUPtr,
        (const mccnn::fpoint<D>*)pPtsGPUPtr, 
        (const mccnn::ipoint<D>*)pNumCellsGPUPtr,
        pUniqueKeyIndexs, 
        (mccnn::fpoint<D>*)pOutPtsGPUPtr, 
        pBatchIdsGPUPtr);

    pDevice->check_error(__FILE__, __LINE__);

#ifdef DEBUG_INFO
    cudaEventRecord(stop, cudaStream);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    float gpuOccupancy = (float)(numBlocks*blockSize)/(float)gpuProps.maxThreadsXMP_;

    fprintf(stderr, "### SAMPLING AVG ###\n");
    fprintf(stderr, "Num points: %d\n", pNumPts);
    fprintf(stderr, "Num sampled pts: %d\n", pNumSampledPts);
    fprintf(stderr, "Occupancy: %f\n", gpuOccupancy);
    fprintf(stderr, "Execution time: %f\n", milliseconds);
    fprintf(stderr, "\n");
#endif
}

#define SAMPLING_AVG_TEMP_DECL(Dims)             \
    template void mccnn::sampling_avg_gpu<Dims>( \
        std::unique_ptr<IGPUDevice>& pDevice,   \
        const unsigned int pNumPts,             \
        const unsigned int pNumSampledPts,       \
        const mccnn::int64_m* pInKeysGPUPtr,    \
        const float* pPtsGPUPtr,                \
        const int* pNumCellsGPUPtr,             \
        const int* pUniqueKeyIndexs,            \
        float* pOutPtsGPUPtr,                   \
        int* pBatchIdsGPUPtr);

DECLARE_TEMPLATE_DIMS(SAMPLING_AVG_TEMP_DECL)