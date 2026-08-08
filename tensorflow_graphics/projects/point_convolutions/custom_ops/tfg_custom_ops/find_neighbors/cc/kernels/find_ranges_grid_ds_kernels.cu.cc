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

#include "tfg_custom_ops/find_neighbors/cc/kernels/find_ranges_grid_ds.h"

#include <vector>

///////////////////////// GPU

/**
 *  GPU kernel to find the ranges in the list of points for a grid cell 
 *  and its neighbors.
 *  @param  pNumSamples     Number of samples.
 *  @param  pNumPts         Number of points.
 *  @param  pLastDOffsets   Number of displacement in the last
 *      dimension in the positive and negative axis.
 *  @param  pNumOffsets     Number of offsets applied to  the 
 *      keys.
 *  @param  pOffsets        List of offsets to apply.
 *  @param  pSampleKeys     Array of keys for each sample.
 *  @param  pPtKeys         Array of keys for each point.
 *  @param  pGridDs         Grid data structure.
 *  @param  pNumCells       Number of cells.
 *  @param  pOutDS          Output array with the ranges.
 *  @paramT D               Number of dimensions.
 */
template<int D>
__global__ void find_ranges_grid_ds_gpu_kernel(
    const unsigned int pNumSamples,
    const unsigned int pNumPts,
    const unsigned int pLastDOffsets,
    const unsigned int pNumOffsets,
    const mccnn::ipoint<D>* __restrict__ pOffsets,
    const mccnn::int64_m* __restrict__ pSampleKeys,
    const mccnn::int64_m* __restrict__ pPtKeys,
    const int2* __restrict__ pGridDs,
    const mccnn::ipoint<D>* __restrict__ pNumCells,
    int2* __restrict__ pOutRanges)
{
    int initPtIndex = mccnn::compute_global_index_gpu_funct();
    int totalThreads = mccnn::compute_total_threads_gpu_funct();

    //Calculate the total number of cells.
    mccnn::int64_m totalCells = mccnn::compute_total_num_cells_gpu_funct(pNumCells[0]);

    for(int curIter = initPtIndex; curIter < pNumSamples*pNumOffsets; curIter += totalThreads)
    {
        //Calculate the point and offset index.
        int curPtIndex = curIter/pNumOffsets;
        int curOffset = curIter%pNumOffsets;

        //Get the current offset.
        mccnn::ipoint<D> cellOffset = pOffsets[curOffset];

        //Get the key of the current point.
        mccnn::int64_m curKey = pSampleKeys[curPtIndex];

        //Get the new cell with the offset.
        mccnn::ipoint<D+1> cellIndex = 
            mccnn::compute_cell_from_key_gpu_funct(curKey, pNumCells[0]);
#pragma unroll
        for(int i=0; i < D; ++i)
            cellIndex[i+1] += cellOffset[i];

        //Check if we are out of the bounding box.
        bool inside = true;
#pragma unroll
        for(int i=0; i < D-1; ++i)
            inside = inside && cellIndex[i+1] >= 0 && cellIndex[i+1] < pNumCells[0][i];
        if(inside)
        {
            //Get the range of pts to check in the data structure.
            int curDsIndex = mccnn::compute_ds_index_from_cell_gpu_funct(cellIndex, pNumCells[0]);
            int2 dsRange = pGridDs[curDsIndex];
            int rangeSize = dsRange.y-dsRange.x-1;

            //Tube has at least 1 element.
            if(rangeSize >= 0){

                //Compute max key of the range.
                mccnn::ipoint<D> auxCell(&cellIndex[1]);
                mccnn::int64_m auxKey = mccnn::compute_key_gpu_funct(auxCell, pNumCells[0], cellIndex[0]);
                mccnn::int64_m maxKey = auxKey+pLastDOffsets;
                mccnn::int64_m minKey = auxKey-pLastDOffsets;
                maxKey = (auxKey/totalCells == maxKey/totalCells)?maxKey:((auxKey/totalCells)*totalCells) + totalCells - 1;
                minKey = (auxKey/totalCells == minKey/totalCells)?minKey:((auxKey/totalCells)*totalCells);

                //Declare iterators and auxiliar variables.
                int2 curMinRange = make_int2(0, rangeSize);
                int2 curMaxRange = make_int2(0, rangeSize);

                //Search for the range.
                bool stopMinRange = rangeSize <= 1;
                bool stopMaxRange = stopMinRange;
                while(!stopMinRange || !stopMaxRange){
                    
                    //Compute the pivots.
                    int minPivot = (curMinRange.y + curMinRange.x)/2;
                    int maxPivot = (curMaxRange.y + curMaxRange.x)/2;

                    //Check the minimum range.
                    if(!stopMinRange){
                        mccnn::int64_m curMinKey = pPtKeys[minPivot+dsRange.x];
                        if(curMinKey > maxKey) curMinRange.x = minPivot;
                        else curMinRange.y = minPivot;
                    }

                    //Check the maximum range.
                    if(!stopMaxRange){
                        mccnn::int64_m curMaxKey = pPtKeys[maxPivot+dsRange.x];
                        if(curMaxKey >= minKey) curMaxRange.x = maxPivot;
                        else curMaxRange.y = maxPivot;
                    }

                    //Check the stopping condition.
                    stopMinRange = (curMinRange.y - curMinRange.x) <= 1;
                    stopMaxRange = (curMaxRange.y - curMaxRange.x) <= 1;
                }

                int2 resultingRange = make_int2(0, 0);

                //Get the values of the keys.
                mccnn::int64_m lastMinKey1 = pPtKeys[curMinRange.x+dsRange.x];
                mccnn::int64_m lastMinKey2 = pPtKeys[curMinRange.y+dsRange.x];

                //Test for the init of the range.
                if(lastMinKey1 >= minKey && lastMinKey1 <= maxKey){
                    resultingRange.x = curMinRange.x+dsRange.x;
                }else if(lastMinKey2 >= minKey && lastMinKey2 <= maxKey){
                    resultingRange.x = curMinRange.y+dsRange.x;
                }

                mccnn::int64_m lastMaxKey1 = pPtKeys[curMaxRange.x+dsRange.x];
                mccnn::int64_m lastMaxKey2 = pPtKeys[curMaxRange.y+dsRange.x];

                //Test for the end of the range.
                if(lastMaxKey2 >= minKey && lastMaxKey2 <= maxKey){
                    resultingRange.y = curMaxRange.y+dsRange.x+1;
                }else if(lastMaxKey1 >= minKey && lastMaxKey1 <= maxKey){
                    resultingRange.y = curMaxRange.x+dsRange.x+1;
                }

                //Store in memory the resulting range.
                pOutRanges[curIter] = resultingRange;
            }
        }
    }
}

/**
 *  GPU kernel for the 2D case.
 *  @param  pNumSamples     Number of samples.
 *  @param  pNumPts         Number of points.
 *  @param  pNumOffsets     Number of offsets applied to  the 
 *      keys.
 *  @param  pOffsets        List of offsets to apply.
 *  @param  pSampleKeys     Array of keys for each sample.
 *  @param  pGridDs         Grid data structure.
 *  @param  pNumCells       Number of cells.
 *  @param  pOutDS          Output array with the ranges.
 *  @paramT D               Number of dimensions.
 */
template<int D>
__global__ void find_ranges_grid_ds_2d_gpu_kernel(
    const unsigned int pNumSamples,
    const unsigned int pNumPts,
    const unsigned int pNumOffsets,
    const mccnn::ipoint<D>* __restrict__ pOffsets,
    const mccnn::int64_m* __restrict__ pSampleKeys,
    const int2* __restrict__ pGridDs,
    const mccnn::ipoint<D>* __restrict__ pNumCells,
    int2* __restrict__ pOutRanges)
{
    int initPtIndex = mccnn::compute_global_index_gpu_funct();
    int totalThreads = mccnn::compute_total_threads_gpu_funct();

    //Calculate the total number of cells.
    mccnn::int64_m totalCells = mccnn::compute_total_num_cells_gpu_funct(pNumCells[0]);

    for(int curIter = initPtIndex; curIter < pNumSamples*pNumOffsets; curIter += totalThreads)
    {
        //Calculate the point and offset index.
        int curPtIndex = curIter/pNumOffsets;
        int curOffset = curIter%pNumOffsets;

        //Get the current offset.
        mccnn::ipoint<D> cellOffset = pOffsets[curOffset];

        //Get the key of the current point.
        mccnn::int64_m curKey = pSampleKeys[curPtIndex];

        //Get the new cell with the offset.
        mccnn::ipoint<D+1> cellIndex = 
            mccnn::compute_cell_from_key_gpu_funct(curKey, pNumCells[0]);
#pragma unroll
        for(int i=0; i < D; ++i)
            cellIndex[i+1] += cellOffset[i];

        //Check if we are out of the bounding box.
        bool inside = true;
#pragma unroll
        for(int i=0; i < D; ++i)
            inside = inside && cellIndex[i+1] >= 0 && cellIndex[i+1] < pNumCells[0][i];
        if(inside)
        {
            //Get the range of pts to check in the data structure.
            int curDsIndex = mccnn::compute_ds_index_from_cell_gpu_funct(cellIndex, pNumCells[0]);
            int2 dsRange = pGridDs[curDsIndex];

            //Store in memory the resulting range.
            pOutRanges[curIter] = dsRange;
        }
    }
}

///////////////////////// CPU

template<int D>
void mccnn::find_ranges_grid_ds_gpu(
    std::unique_ptr<IGPUDevice>& pDevice,
    const unsigned int pNumSamples, 
    const unsigned int pNumPts,
    const unsigned int pLastDOffsets,
    const unsigned int pNumOffsets,
    const int* pInGPUPtrOffsets,
    const mccnn::int64_m* pInGPUPtrSampleKeys,
    const mccnn::int64_m* pInGPUPtrPtsKeys,
    const int* pInGPUPtrGridDS,
    const int* pInGPUPtrNumCells,
    int* pOutGPUPtrRanges)
{
    //Get the cuda stream.
    auto cudaStream = pDevice->getCUDAStream();

    //Initialize to zero the output array.
    pDevice->memset(pOutGPUPtrRanges, 0, sizeof(int)*pNumSamples*pNumOffsets*2);
    pDevice->check_error(__FILE__, __LINE__);

    //Get the device properties.
    const GpuDeviceProperties& gpuProps = pDevice->get_device_properties();

    //Calculate the ideal number of blocks for the selected block size.
    unsigned int numMP = gpuProps.numMPs_;
    unsigned int blockSize = gpuProps.warpSize_*2;
    unsigned int numBlocks = pDevice->get_max_active_block_x_sm(
        blockSize,(const void*)find_ranges_grid_ds_gpu_kernel<D>, 0);
    pDevice->check_error(__FILE__, __LINE__);

    //Calculate the total number of blocks to execute.
    unsigned int execBlocks = (pNumSamples*pNumOffsets)/blockSize;
    execBlocks += ((pNumSamples*pNumOffsets)%blockSize != 0)?1:0;
    unsigned int totalNumBlocks = numMP*numBlocks;
    totalNumBlocks = (totalNumBlocks > execBlocks)?execBlocks:totalNumBlocks;

    //Execute the cuda kernel.
    if(D > 2){
        find_ranges_grid_ds_gpu_kernel<D><<<totalNumBlocks, blockSize, 0, cudaStream>>>(
            pNumSamples,
            pNumPts, 
            pLastDOffsets,
            pNumOffsets,
            (const mccnn::ipoint<D>*)pInGPUPtrOffsets,
            pInGPUPtrSampleKeys, 
            pInGPUPtrPtsKeys,
            (const int2*)pInGPUPtrGridDS, 
            (const mccnn::ipoint<D>*)pInGPUPtrNumCells,
            (int2*)pOutGPUPtrRanges);
    }else{
        find_ranges_grid_ds_2d_gpu_kernel<D><<<totalNumBlocks, blockSize, 0, cudaStream>>>(
            pNumSamples,
            pNumPts, 
            pNumOffsets,
            (const mccnn::ipoint<D>*)pInGPUPtrOffsets,
            pInGPUPtrSampleKeys,
            (const int2*)pInGPUPtrGridDS, 
            (const mccnn::ipoint<D>*)pInGPUPtrNumCells,
            (int2*)pOutGPUPtrRanges);
    }
    pDevice->check_error(__FILE__, __LINE__);
}

unsigned int mccnn::computeTotalNumOffsets(
    const unsigned int pNumDimensions,
    const unsigned int pAxisOffset,
    std::vector<int>& pOutVector)
{
    //Calculate the total number of offsets.
    unsigned int cellsXAxis = pAxisOffset*2 + 1;
    unsigned int numOffsets = cellsXAxis;
    for(int i = 0 ; i < std::max((int)pNumDimensions-2, 1); ++i)
        numOffsets *= cellsXAxis;

    //Calculate each offset.
    pOutVector.clear();
    std::vector<int> curOffset(pNumDimensions, 0);
    for(int i = 0; i < numOffsets; ++i)
    {
        int auxInt = i;
        
        for(int j = std::max((int)pNumDimensions-2, 1); j >= 0 ; --j) 
        {
            int auxInt2 = auxInt%cellsXAxis;
            auxInt2 = auxInt2-pAxisOffset;
            curOffset[j] = auxInt2;
            auxInt = auxInt/cellsXAxis;
        }  
        if(pNumDimensions != 2) curOffset[pNumDimensions-1] = 0;

        for(int j = 0; j < pNumDimensions; ++j)
            pOutVector.push_back(curOffset[j]);
    }

    return numOffsets;
}

///////////////////////// CPU Template declaration

#define FIND_RANGES_GRID_DS_TEMP_DECL(Dims)             \
    template void mccnn::find_ranges_grid_ds_gpu<Dims>( \
        std::unique_ptr<IGPUDevice>& pDevice,           \
        const unsigned int pNumSamples,                 \
        const unsigned int pNumPts,                     \
        const unsigned int pLastDOffsets,               \
        const unsigned int pNumOffsets,                 \
        const int* pInGPUPtrOffsets,                    \
        const mccnn::int64_m* pInGPUPtrSampleKeys,      \
        const mccnn::int64_m* pInGPUPtrPtsKeys,         \
        const int* pInGPUPtrGridDS,                     \
        const int* pInGPUPtrNumCells,                   \
        int* pOutGPUPtrRanges);

DECLARE_TEMPLATE_DIMS(FIND_RANGES_GRID_DS_TEMP_DECL)