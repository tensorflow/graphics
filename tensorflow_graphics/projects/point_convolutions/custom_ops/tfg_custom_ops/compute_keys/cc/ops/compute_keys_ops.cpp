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
/// \brief C++ operations definition to compute the keys indices of a point 
///     cloud into a regular grid. 
/////////////////////////////////////////////////////////////////////////////

#include "tfg_custom_ops/shared/cc/kernels/defines.hpp"
#include "tfg_custom_ops/shared/cc/kernels/tf_utils.hpp"
#include "tfg_custom_ops/shared/cc/kernels/tf_gpu_device.hpp"

#include "tfg_custom_ops/compute_keys/cc/kernels/compute_keys.h"

/**
 *  Declaration of the tensorflow operation.
 */
REGISTER_OP("ComputeKeys")
    .Input("points: float32")
    .Input("batch_ids: int32")
    .Input("scaled_aabb_min: float32")
    .Input("num_cells: int32")
    .Input("inv_cell_sizes: float32")
    .Output("keys: int64")
    .SetShapeFn([](shape_inference::InferenceContext* pIC) {
        shape_inference::ShapeHandle outputDims = 
            pIC->MakeShape({pIC->Dim(pIC->input(0), 0)});
        pIC->set_output(0, outputDims);
        return Status::OK();
    });

namespace mccnn{

    /**
     *  Operation to compute the keys of each point in a regular grid.
     */
    class ComputeKeysOp: public OpKernel{
        
        public:
        
            /**
             *  Constructor.
             *  @param  pContext    Constructor context of the operation.
             */
            explicit ComputeKeysOp(
                OpKernelConstruction* pContext)
                :OpKernel(pContext){
            }
        
            /**
             *  Method to compute the operation.
             *  @param  pContext    Context of the operation.
             */
            void Compute(OpKernelContext * pContext) override{

                //Get the input tensors.
                const Tensor& inPts = pContext->input(0); 
                const Tensor& inBatchIds = pContext->input(1); 
                const Tensor& inSAABBMin = pContext->input(2); 
                const Tensor& inNumCells = pContext->input(3); 
                const Tensor& inInvCellSizes = pContext->input(4); 

                //Get variables from tensors.
                unsigned int numPts = inPts.shape().dim_size(0);
                unsigned int numDimensions = inPts.shape().dim_size(1);
                unsigned int batchSize = inSAABBMin.shape().dim_size(0);

                //Get the pointers to GPU data from the tensors.
                const float* ptsGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<float>(inPts);
                const int* batchIdsGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<int>(inBatchIds);
                const float* aabbMinGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<float>(inSAABBMin);
                const int* numCellsGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<int>(inNumCells);
                const float* invCellSizesGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<float>(inInvCellSizes);

                //Check for the correctness of the input.
                OP_REQUIRES(pContext, numDimensions >= MIN_DIMENSIONS 
                    && numDimensions <= MAX_DIMENSIONS, 
                    errors::InvalidArgument("ComputeKeysOp expects a valid number of dimension"));
                OP_REQUIRES(pContext, inBatchIds.shape().dim_size(0) == numPts, 
                    errors::InvalidArgument("ComputeKeysOp expects the same number of batch"
                    " ids as number of points."));
                OP_REQUIRES(pContext, inSAABBMin.shape().dim_size(1) == numDimensions, 
                    errors::InvalidArgument("ComputeKeysOp expects aabb with the same "
                    "number of dimensions as the points."));
                OP_REQUIRES(pContext, inNumCells.shape().dim_size(0) == numDimensions, 
                    errors::InvalidArgument("ComputeKeysOp expects number of cells with the same "
                    "number of dimensions as the points."));
                OP_REQUIRES(pContext, inInvCellSizes.shape().dim_size(0) == numDimensions, 
                    errors::InvalidArgument("ComputeKeysOp expects cell sizes with the same "
                    "number of dimensions as the points."));

                //Create the output tensor.
                mccnn::int64_m* outputGPUPtr = nullptr;
                TensorShape outShape = TensorShape{numPts};
                OP_REQUIRES_OK(pContext, mccnn::tensorflow_utils::allocate_output_tensor<mccnn::int64_m>
                    (0, pContext, outShape, &outputGPUPtr));

                //Get the gpu device.
                std::unique_ptr<mccnn::IGPUDevice> gpuDevice = make_unique<mccnn::TFGPUDevice>(pContext);

                //Compute the keys.
                DIMENSION_SWITCH_CALL(numDimensions, mccnn::compute_keys_gpu,
                    gpuDevice, numPts, ptsGPUPtr, batchIdsGPUPtr, 
                    aabbMinGPUPtr, numCellsGPUPtr, invCellSizesGPUPtr, 
                    outputGPUPtr);                    
            }
    };
}
            
REGISTER_KERNEL_BUILDER(Name("ComputeKeys").Device(DEVICE_GPU), mccnn::ComputeKeysOp);