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
/////////////////////////////////////////////////////////////////////////////

#include "tfg_custom_ops/shared/cc/kernels/defines.hpp"
#include "tfg_custom_ops/shared/cc/kernels/tf_utils.hpp"
#include "tfg_custom_ops/shared/cc/kernels/tf_gpu_device.hpp"
#include "tfg_custom_ops/build_grid_ds/cc/kernels/build_grid_ds.h"

/**
 *  Declaration of the tensorflow operation.
 */
REGISTER_OP("BuildGridDs")
    .Input("keys: int64")
    .Input("num_cells_gpu: int32")
    .Input("num_cells_cpu: int32")
    .Output("acc_ds: int32")
    .Attr("batch_size: int")
    .SetShapeFn([](shape_inference::InferenceContext* pIC) {
        int batchSize;
        TF_RETURN_IF_ERROR(pIC->GetAttr("batch_size", &batchSize));
        shape_inference::ShapeHandle outputDims = 
            pIC->MakeShape({batchSize, -1, -1, 2});
        pIC->set_output(0, outputDims);
        return Status::OK();
    });

namespace mccnn{
        
    /**
     *  Operation to build the data structure to access a regular grid.
     */
    class BuildGridDsOp: public OpKernel{
        
        public:
        
            /**
             *  Constructor.
             *  @param  pContext    Constructor context of the operation.
             */
            explicit BuildGridDsOp(
                OpKernelConstruction* pContext)
                :OpKernel(pContext){
                OP_REQUIRES_OK(pContext, pContext->GetAttr("batch_size", &batchSize_));
                OP_REQUIRES(pContext, batchSize_ > 0, 
                    errors::InvalidArgument("BuildGridDsOp requires a positive batch size"));
            }
        
            /**
             *  Method to compute the operation.
             *  @param  pContext    Context of the operation.
             */
            void Compute(OpKernelContext * pContext) override{

                //Get the input tensors.
                const Tensor& inKeys = pContext->input(0); 
                const Tensor& inNumCellsGPU = pContext->input(1); 
                const Tensor& inNumCellsCPU = pContext->input(2); 

                //Get variables from tensors.
                unsigned int numPts = inKeys.shape().dim_size(0);
                unsigned int numDimensions = inNumCellsGPU.shape().dim_size(0);

                //Get the pointers to GPU/CPU data from the tensors.
                const mccnn::int64_m* keysGPUPtr = mccnn::tensorflow_utils::
                    get_const_tensor_pointer<mccnn::int64_m>(inKeys);
                const int* numCellsGPUPtr = mccnn::tensorflow_utils::
                    get_const_tensor_pointer<int>(inNumCellsGPU);
                auto numCellsCPU = inNumCellsCPU.flat<int>();

                //Check for the correctness of the input.
                OP_REQUIRES(pContext, numDimensions >= MIN_DIMENSIONS 
                    && numDimensions <= MAX_DIMENSIONS, 
                    errors::InvalidArgument("BuildGridDsOp expects a valid number of dimension"));
                OP_REQUIRES(pContext, inNumCellsCPU.shape().dim_size(0) == numDimensions, 
                    errors::InvalidArgument("BuildGridDsOp expects number of cells with the"
                    " same dimensions in both parameters"));

                //Create the output tensor.
                int* outputGPUPtr = nullptr;
                TensorShape outShape = TensorShape{batchSize_, numCellsCPU(0), numCellsCPU(1), 2};
                OP_REQUIRES_OK(pContext, mccnn::tensorflow_utils::allocate_output_tensor<int>
                    (0, pContext, outShape, &outputGPUPtr));

                //Get the gpu device.
                std::unique_ptr<mccnn::IGPUDevice> gpuDevice = make_unique<mccnn::TFGPUDevice>(pContext);

                //Compute the data structure.
                DIMENSION_SWITCH_CALL(numDimensions, mccnn::build_grid_ds_gpu,
                    gpuDevice, batchSize_*numCellsCPU(0)*numCellsCPU(1)*2, 
                    numPts, keysGPUPtr, numCellsGPUPtr, outputGPUPtr);
            }

        private:

            /**Batch size.*/
            int   batchSize_;
    };
}
            
REGISTER_KERNEL_BUILDER(Name("BuildGridDs").Device(DEVICE_GPU).HostMemory("num_cells_cpu"), mccnn::BuildGridDsOp);