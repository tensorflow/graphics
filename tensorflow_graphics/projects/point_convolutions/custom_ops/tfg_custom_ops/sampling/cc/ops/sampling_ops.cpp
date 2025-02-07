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
/// \brief C++ operations definition to perform a sample operation in a point
///     cloud. 
/////////////////////////////////////////////////////////////////////////////

#include "tfg_custom_ops/shared/cc/kernels/defines.hpp"
#include "tfg_custom_ops/shared/cc/kernels/tf_utils.hpp"
#include "tfg_custom_ops/shared/cc/kernels/tf_gpu_device.hpp"

#include "tfg_custom_ops/sampling/cc/kernels/count_unique_keys.h"
#include "tfg_custom_ops/sampling/cc/kernels/store_unique_keys.h"
#include "tfg_custom_ops/sampling/cc/kernels/sampling_avg.h"
#include "tfg_custom_ops/sampling/cc/kernels/count_sampling_pd.h"
#include "tfg_custom_ops/sampling/cc/kernels/store_sampled_pts.h"

/**
 *  Declaration of the tensorflow operation.
 */
REGISTER_OP("Sampling")
    .Input("points: float32")
    .Input("batch_ids: int32")
    .Input("point_keys: int64")
    .Input("num_cells: int32")
    .Input("neighbors: int32")
    .Input("sample_neigh_indices: int32")
    .Output("sample_pts: float32")
    .Output("sample_batch_ids: int32")
    .Output("sample_indices: int32")
    .Attr("mode: int")
    .SetShapeFn([](shape_inference::InferenceContext* pIC) {
        shape_inference::ShapeHandle outputDims1 = 
            pIC->MakeShape({-1, pIC->Dim(pIC->input(0), 1)});
        shape_inference::ShapeHandle outputDims2 = 
            pIC->MakeShape({-1});
        pIC->set_output(0, outputDims1);
        pIC->set_output(1, outputDims2);
        pIC->set_output(2, outputDims2);
        return Status::OK();
    });

namespace mccnn{

    /**
     *  Operation to perform a sampling operation on a regular grid.
     */
    class SamplingOp: public OpKernel{
        
        public:
        
            /**
             *  Constructor.
             *  @param  pContext    Constructor context of the operation.
             */
            explicit SamplingOp(
                OpKernelConstruction* pContext)
                :OpKernel(pContext){

                OP_REQUIRES_OK(pContext, pContext->GetAttr("mode", &mode_));
                OP_REQUIRES(pContext, mode_ >= 0 && mode_ < 2, 
                    errors::InvalidArgument("SamplingOp requires a valid sampling mode."));
            }
        
            /**
             *  Method to compute the operation.
             *  @param  pContext    Context of the operation.
             */
            void Compute(OpKernelContext * pContext) override{

                //Get the input tensors.
                const Tensor& inPts = pContext->input(0); 
                const Tensor& inBatchIds = pContext->input(1); 
                const Tensor& inPtKeys = pContext->input(2); 
                const Tensor& inNumCells = pContext->input(3);
                const Tensor& inNeighbors = pContext->input(4); 
                const Tensor& inStartNeighIds = pContext->input(5); 

                //Get variables from tensors.
                unsigned int numPts = inPts.shape().dim_size(0);
                unsigned int numDimensions = inPts.shape().dim_size(1);
                 unsigned int numNeighbors = inNeighbors.shape().dim_size(0);

                //Get the pointers to GPU data from the tensors.
                const float* ptsGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<float>(inPts);
                const int* batchIdsGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<int>(inBatchIds);
                const mccnn::int64_m* ptKeysGPUPtr = mccnn::tensorflow_utils::
                    get_const_tensor_pointer<mccnn::int64_m>(inPtKeys);
                const int* numCellsGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<int>(inNumCells);
                const int* inNeighborsGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<int>(inNeighbors);
                const int* inStartNeighIdsGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<int>(inStartNeighIds);
                
                
                //Check for the correctness of the input.
                OP_REQUIRES(pContext, numDimensions >= MIN_DIMENSIONS 
                    && numDimensions <= MAX_DIMENSIONS, 
                    errors::InvalidArgument("SamplingOp expects a valid number of dimension"));
                OP_REQUIRES(pContext, inBatchIds.shape().dim_size(0) == numPts, 
                    errors::InvalidArgument("SamplingOp expects the same number of keys"
                    " as number of points."));
                OP_REQUIRES(pContext, inPtKeys.shape().dim_size(0) == numPts, 
                    errors::InvalidArgument("SamplingOp expects the same number of keys"
                    " as number of points."));
                OP_REQUIRES(pContext, inNumCells.shape().dim_size(0) == numDimensions, 
                    errors::InvalidArgument("SamplingOp expects a number of dimensions in"
                    " inNumCells equal to the number of dimensions in the input points"));
                OP_REQUIRES(pContext, inNeighbors.dims() == 2 && 
                    inNeighbors.shape().dim_size(1) == 2, 
                    errors::InvalidArgument("SamplingOp expects a neighbor tensor with 2 dimensions "
                    "and 2 indices per neighbor."));
                OP_REQUIRES(pContext, inStartNeighIds.shape().dim_size(0) == numPts, 
                    errors::InvalidArgument("SamplingOp expects the same number of points "
                    "in inStartNeighIds as in the points tensor."));
                

                //Get the gpu device.
                std::unique_ptr<mccnn::IGPUDevice> gpuDevice = make_unique<mccnn::TFGPUDevice>(pContext);

                //Count the number of unique keys.
                unsigned int numKeys = mccnn::count_unique_keys_gpu(gpuDevice, numPts, ptKeysGPUPtr);
                
                //Declare the temporal buffers.
                int* tmpGPUPtr1 = gpuDevice->getIntTmpGPUBuffer(numKeys);

                //Store the first indices of each key.
                mccnn::store_unique_keys_gpu(gpuDevice, numPts, ptKeysGPUPtr, tmpGPUPtr1);

                //Poisson disk sampling.
                if(mode_ == 0){
                    
                    //Declare temporal buffers.
                    int* tmpGPUPtr2 = gpuDevice->getIntTmpGPUBuffer(numPts);

                    //Count the number of sampleed points.
                    int numSampledPts;
                    DIMENSION_SWITCH_CALL(numDimensions, mccnn::count_sampling_pd_gpu, 
                        gpuDevice, numPts, numKeys, tmpGPUPtr1, ptKeysGPUPtr,
                        ptsGPUPtr, inNeighborsGPUPtr, inStartNeighIdsGPUPtr,
                        numCellsGPUPtr, numSampledPts, tmpGPUPtr2);

                    //Create the output tensors.
                    float* outPtsGPUPtr = nullptr;
                    int* outBatchIdsGPUPtr = nullptr;
                    int* outIndicesGPUPtr = nullptr;
                    TensorShape outShape1 = TensorShape{numSampledPts, numDimensions};
                    TensorShape outShape2 = TensorShape{numSampledPts};
                    TensorShape outShape3 = TensorShape{numSampledPts};
                    OP_REQUIRES_OK(pContext, mccnn::tensorflow_utils::allocate_output_tensor<float>
                        (0, pContext, outShape1, &outPtsGPUPtr));
                    OP_REQUIRES_OK(pContext, mccnn::tensorflow_utils::allocate_output_tensor<int>
                        (1, pContext, outShape2, &outBatchIdsGPUPtr));
                    OP_REQUIRES_OK(pContext, mccnn::tensorflow_utils::allocate_output_tensor<int>
                        (2, pContext, outShape3, &outIndicesGPUPtr));

                    //Store the sampled points.
                    DIMENSION_SWITCH_CALL(numDimensions, mccnn::store_sampled_pts_gpu,
                        gpuDevice, numPts, numSampledPts, ptsGPUPtr,
                        batchIdsGPUPtr, tmpGPUPtr2, outPtsGPUPtr, 
                        outBatchIdsGPUPtr, outIndicesGPUPtr);

                //Cell average.
                }else{
                    //Create the output tensors.
                    float* outPtsGPUPtr = nullptr;
                    int* outBatchIdsGPUPtr = nullptr;
                    int* outIndicesGPUPtr = nullptr;
                    TensorShape outShape1 = TensorShape{numKeys, numDimensions};
                    TensorShape outShape2 = TensorShape{numKeys};
                    TensorShape outShape3 = TensorShape{1};
                    OP_REQUIRES_OK(pContext, mccnn::tensorflow_utils::allocate_output_tensor<float>
                        (0, pContext, outShape1, &outPtsGPUPtr));
                    OP_REQUIRES_OK(pContext, mccnn::tensorflow_utils::allocate_output_tensor<int>
                        (1, pContext, outShape2, &outBatchIdsGPUPtr));
                    OP_REQUIRES_OK(pContext, mccnn::tensorflow_utils::allocate_output_tensor<int>
                        (2, pContext, outShape3, &outIndicesGPUPtr));
                    

                    //Select the new point.
                    DIMENSION_SWITCH_CALL(numDimensions, mccnn::sampling_avg_gpu,
                        gpuDevice, numPts, numKeys, ptKeysGPUPtr, ptsGPUPtr, 
                        numCellsGPUPtr, tmpGPUPtr1, outPtsGPUPtr, outBatchIdsGPUPtr);
                }
            }

        private:

            /**Mode used to sample points.*/
            int mode_;
    };
}
            
REGISTER_KERNEL_BUILDER(Name("Sampling").Device(DEVICE_GPU), mccnn::SamplingOp);