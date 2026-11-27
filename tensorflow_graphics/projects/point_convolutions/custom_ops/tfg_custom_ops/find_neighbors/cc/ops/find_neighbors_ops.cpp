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
/// \brief C++ operations definition to build the data structure to access
///     the sparse regular grid.
/////////////////////////////////////////////////////////////////////////////

#include "tfg_custom_ops/shared/cc/kernels/tf_utils.hpp"
#include "tfg_custom_ops/shared/cc/kernels/tf_gpu_device.hpp"

#include "tfg_custom_ops/compute_keys/cc/kernels/compute_keys.h"

#include "tfg_custom_ops/find_neighbors/cc/kernels/find_ranges_grid_ds.h"
#include "tfg_custom_ops/find_neighbors/cc/kernels/count_neighbors.h"
#include "tfg_custom_ops/find_neighbors/cc/kernels/elem_wise_min.h"
#include "tfg_custom_ops/find_neighbors/cc/kernels/scan_alg.h"
#include "tfg_custom_ops/find_neighbors/cc/kernels/store_neighbors.h"

//This value determines how many cuda kernel call are made to search in 
//the neighboring cells:
//  - 3 Dimensions 9 offsets per block -> 1 cuda kernel executions
//  - 3 Dimensions 3 offsets per block -> 3 cuda kernel executions
//  - 6 Dimensions 9 offsets per block -> 27 cuda kernel executions
//  - 6 Dimensions 3 offsets per block -> 81 cuda kernel executions
#define NUM_OFFSETS_X_COMPUTE_BLOCK 9

/**
 *  Declaration of the tensorflow operation.
 */
REGISTER_OP("FindNeighbors")
    .Input("samples: float32")
    .Input("samples_batch_ids: int32")
    .Input("points: float32")
    .Input("point_keys: int64")
    .Input("grid_acc_ds: int32")
    .Input("num_cells: int32")
    .Input("scaled_aabb_min: float32")
    .Input("inv_cell_sizes: float32")
    .Input("inv_radii: float32")
    .Output("samples_neigh_indices: int32")
    .Output("neigh_indices: int32")
    .Attr("max_neighs:int")
    .SetShapeFn([](shape_inference::InferenceContext* pIC) {
        shape_inference::ShapeHandle outputDims1 = pIC->MakeShape({pIC->Dim(pIC->input(0), 0)});
        shape_inference::ShapeHandle outputDims2 = pIC->MakeShape({-1, 2});
        pIC->set_output(0, outputDims1);
        pIC->set_output(1, outputDims2);
        return Status::OK();
    });

namespace mccnn{
        
    /**
     *  Operation to find the neighboring points to a set of samples.
     */
    class FindNeighborsOp: public OpKernel{
        
        public:
        
            /**
             *  Constructor.
             *  @param  pContext    Constructor context of the operation.
             */
            explicit FindNeighborsOp(
                OpKernelConstruction* pContext)
                :OpKernel(pContext){
                OP_REQUIRES_OK(pContext, pContext->GetAttr("max_neighs", &maxNeighs_));
            }
        
            /**
             *  Method to compute the operation.
             *  @param  pContext    Context of the operation.
             */
            void Compute(OpKernelContext * pContext) override{

                //Get the input tensors.
                const Tensor& inSamples = pContext->input(0); 
                const Tensor& inBatchIds = pContext->input(1); 
                const Tensor& inPts = pContext->input(2); 
                const Tensor& inPtKeys = pContext->input(3); 
                const Tensor& inGridDs = pContext->input(4); 
                const Tensor& inNumCells = pContext->input(5); 
                const Tensor& inSAABBMin = pContext->input(6); 
                const Tensor& inInvCellSizes = pContext->input(7); 
                const Tensor& inInvRadii = pContext->input(8); 

                //Get variables from tensors.
                unsigned int numSamples = inSamples.shape().dim_size(0);
                unsigned int numPts = inPts.shape().dim_size(0);
                unsigned int numDimensions = inSamples.shape().dim_size(1);

                //Get the pointers to GPU/CPU data from the tensors.
                const float* samplesGPUPtr = mccnn::tensorflow_utils::
                    get_const_tensor_pointer<float>(inSamples);
                const int* batchIdsGPUPtr = mccnn::tensorflow_utils::
                    get_const_tensor_pointer<int>(inBatchIds);
                const float* ptsGPUPtr = mccnn::tensorflow_utils::
                    get_const_tensor_pointer<float>(inPts);
                const mccnn::int64_m* ptKeysGPUPtr = mccnn::tensorflow_utils::
                    get_const_tensor_pointer<mccnn::int64_m>(inPtKeys);
                const int* gridDsGPUPtr = mccnn::tensorflow_utils::
                    get_const_tensor_pointer<int>(inGridDs);
                const int* numCellsGPUPtr = mccnn::tensorflow_utils::
                    get_const_tensor_pointer<int>(inNumCells);
                const float* sAABBMinGPUPtr = mccnn::tensorflow_utils::
                    get_const_tensor_pointer<float>(inSAABBMin);
                const float* invCellSizeGPUPtr = mccnn::tensorflow_utils::
                    get_const_tensor_pointer<float>(inInvCellSizes);
                const float* invRadiiGPUPtr = mccnn::tensorflow_utils::
                    get_const_tensor_pointer<float>(inInvRadii);

                //Check for the correctness of the input.
                OP_REQUIRES(pContext, numDimensions >= MIN_DIMENSIONS 
                    && numDimensions <= MAX_DIMENSIONS, 
                    errors::InvalidArgument("FindNeighborsOp expects a valid number of dimension"));
                OP_REQUIRES(pContext, inNumCells.shape().dim_size(0) == numDimensions, 
                    errors::InvalidArgument("FindNeighborsOp expects a number of dimensions in"
                    " inNumCells equal to the number of dimensions in the input points"));
                OP_REQUIRES(pContext, inPts.shape().dim_size(1) == numDimensions, 
                    errors::InvalidArgument("FindNeighborsOp expects a number of dimensions in"
                    " inPts equal to the number of dimensions in the input points"));
                OP_REQUIRES(pContext, inSAABBMin.shape().dim_size(1) == numDimensions, 
                    errors::InvalidArgument("FindNeighborsOp expects a number of dimensions in"
                    " inSAABBMin equal to the number of dimensions in the input points"));
                OP_REQUIRES(pContext, inInvCellSizes.shape().dim_size(0) == numDimensions, 
                    errors::InvalidArgument("FindNeighborsOp expects a number of dimensions in"
                    " inInvCellSizes equal to the number of dimensions in the input points"));
                OP_REQUIRES(pContext, inInvRadii.shape().dim_size(0) == numDimensions, 
                    errors::InvalidArgument("FindNeighborsOp expects a number of dimensions in"
                    " inInvRadii equal to the number of dimensions in the input points"));
                OP_REQUIRES(pContext, inBatchIds.shape().dim_size(0) == numSamples, 
                    errors::InvalidArgument("FindNeighborsOp expects the same number of batch"
                    " ids as samples."));
                OP_REQUIRES(pContext, inPtKeys.shape().dim_size(0) == numPts, 
                    errors::InvalidArgument("FindNeighborsOp expects the same number of keys"
                    " as points."));
                OP_REQUIRES(pContext, inGridDs.dims() == 4 &&
                    inGridDs.shape().dim_size(3) == 2, 
                    errors::InvalidArgument("FindNeighborsOp expects a grid acceleration data"
                    " structure with the right format (B, X, Y, 2)."));

                //Get the gpu device.
                std::unique_ptr<mccnn::IGPUDevice> gpuDevice = make_unique<mccnn::TFGPUDevice>(pContext);

                //Create the first output.
                int* outputGPUPtr1 = nullptr;
                TensorShape outShape1 = TensorShape{numSamples};
                OP_REQUIRES_OK(pContext, mccnn::tensorflow_utils::allocate_output_tensor<int>
                    (0, pContext, outShape1, &outputGPUPtr1));
                gpuDevice->memset(outputGPUPtr1, 0, sizeof(int)*numSamples);
                gpuDevice->check_error(__FILE__, __LINE__);

                //Create the temporal tensors.
                mccnn::int64_m* tmpGPUPtr1 = gpuDevice->getInt64TmpGPUBuffer(numSamples);
                int* tmpGPUPtr2 = gpuDevice->getIntTmpGPUBuffer(
                    numSamples*NUM_OFFSETS_X_COMPUTE_BLOCK*2);
                int* tmpGPUPtr3 = gpuDevice->getIntTmpGPUBuffer(
                    NUM_OFFSETS_X_COMPUTE_BLOCK*numDimensions);

                //Compute the keys of the samples.
                DIMENSION_SWITCH_CALL(numDimensions, mccnn::compute_keys_gpu,
                    gpuDevice, numSamples, samplesGPUPtr, batchIdsGPUPtr, 
                    sAABBMinGPUPtr, numCellsGPUPtr, invCellSizeGPUPtr, tmpGPUPtr1);

                //Compute the number of offsets to used in the search.
                std::vector<int> combOffsets;
                unsigned int numOffsets = mccnn::computeTotalNumOffsets(
                    numDimensions, 1, combOffsets);
                unsigned int numOfComputations = numOffsets/NUM_OFFSETS_X_COMPUTE_BLOCK;

                //Created the pinned cpu memory to store the offsets.
                int bufferSize = (int)combOffsets.size();
                int* combOffsetsPinnedMem = gpuDevice->getIntTmpGPUBuffer(bufferSize, true);
                memcpy((void*)combOffsetsPinnedMem, (void*)&combOffsets[0], sizeof(int)*bufferSize);

                for(int i = 0; i < numOfComputations; ++i)
                {
                    //Copy to device the offsets.
                    gpuDevice->memcpy_host_to_device(tmpGPUPtr3, 
                        &combOffsetsPinnedMem[numDimensions*NUM_OFFSETS_X_COMPUTE_BLOCK*i], 
                        sizeof(int)*numDimensions*NUM_OFFSETS_X_COMPUTE_BLOCK);
                    gpuDevice->check_error(__FILE__, __LINE__);

                    //Find ranges of points to check in the data structure.
                    DIMENSION_SWITCH_CALL(numDimensions, mccnn::find_ranges_grid_ds_gpu,
                        gpuDevice, numSamples, numPts, 1, NUM_OFFSETS_X_COMPUTE_BLOCK, 
                        tmpGPUPtr3, tmpGPUPtr1, ptKeysGPUPtr,
                        gridDsGPUPtr, numCellsGPUPtr, tmpGPUPtr2);

                    //Count neighbors.
                    DIMENSION_SWITCH_CALL(numDimensions, mccnn::count_neighbors,
                        gpuDevice, numSamples, NUM_OFFSETS_X_COMPUTE_BLOCK, samplesGPUPtr, 
                        ptsGPUPtr, tmpGPUPtr2, invRadiiGPUPtr, outputGPUPtr1);
                }

                //Clamp the number of neighbors if required.
                int* tmpGPUPtr4 = nullptr;
                if(maxNeighs_ > 0){
                    tmpGPUPtr4 = gpuDevice->getIntTmpGPUBuffer(numSamples);
                    gpuDevice->memcpy_device_to_device(tmpGPUPtr4, outputGPUPtr1, sizeof(int)*numSamples);
                    mccnn::elem_wise_min_value<int>(gpuDevice, numSamples, maxNeighs_, outputGPUPtr1);
                }

                //Scan algorithm of the number of neighbors per sample.
                unsigned int numNeighbors =  mccnn::scan_alg(
                    gpuDevice, numSamples, outputGPUPtr1);

                //Create the second output tensor.
                int* outputGPUPtr2 = nullptr;
                TensorShape outShape2 = TensorShape{numNeighbors, 2};
                OP_REQUIRES_OK(pContext, mccnn::tensorflow_utils::allocate_output_tensor<int>
                    (1, pContext, outShape2, &outputGPUPtr2));
                gpuDevice->memset(outputGPUPtr2, 0, sizeof(int)*numNeighbors*2);
                gpuDevice->check_error(__FILE__, __LINE__);

                for(int i = 0; i < numOfComputations; ++i)
                {
                    if(numOfComputations > 1){
                        //Copy to device the offsets.
                        gpuDevice->memcpy_host_to_device(tmpGPUPtr3, 
                            &combOffsetsPinnedMem[numDimensions*NUM_OFFSETS_X_COMPUTE_BLOCK*i], 
                            sizeof(int)*numDimensions*NUM_OFFSETS_X_COMPUTE_BLOCK);
                        gpuDevice->check_error(__FILE__, __LINE__);

                        //Find ranges of points to check in the data structure.
                        DIMENSION_SWITCH_CALL(numDimensions, mccnn::find_ranges_grid_ds_gpu,
                            gpuDevice, numSamples, numPts, 1, NUM_OFFSETS_X_COMPUTE_BLOCK, 
                            tmpGPUPtr3, tmpGPUPtr1, ptKeysGPUPtr,
                            gridDsGPUPtr, numCellsGPUPtr, tmpGPUPtr2);
                    }

                    //Store neighbors.
                    DIMENSION_SWITCH_CALL(numDimensions, mccnn::store_neighbors,
                        gpuDevice, maxNeighs_, numSamples, NUM_OFFSETS_X_COMPUTE_BLOCK,  
                        samplesGPUPtr, ptsGPUPtr, tmpGPUPtr2, invRadiiGPUPtr, 
                        tmpGPUPtr4, outputGPUPtr1, outputGPUPtr2);
                }
            }

        private:

            /**Maximum number of neighbors to store per sample.*/
            int maxNeighs_;
    };
}
            
REGISTER_KERNEL_BUILDER(Name("FindNeighbors").Device(DEVICE_GPU), mccnn::FindNeighborsOp);