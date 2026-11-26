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
/// \brief C++ operations definition to compute the pdf of each neighbor.
/////////////////////////////////////////////////////////////////////////////

#include "tfg_custom_ops/shared/cc/kernels/defines.hpp"
#include "tfg_custom_ops/shared/cc/kernels/tf_utils.hpp"
#include "tfg_custom_ops/shared/cc/kernels/tf_gpu_device.hpp"

#include "tfg_custom_ops/compute_pdf/cc/kernels/compute_pdf.h"

/**
 *  Declaration of the tensorflow operations.
 */
REGISTER_OP("ComputePdf")
    .Input("points: float32")
    .Input("neighbors: int32")
    .Input("sample_neigh_indices: int32")
    .Input("inv_badnwidth: float32")
    .Input("inv_radii: float32")
    .Output("pdfs: float32")
    .Attr("mode: int")
    .SetShapeFn([](shape_inference::InferenceContext* pIC) {
        shape_inference::ShapeHandle outputDims = 
            pIC->MakeShape({pIC->Dim(pIC->input(2), 0)});
        pIC->set_output(0, outputDims);
        return Status::OK();
    });

REGISTER_OP("ComputePdfWithPtGrads")
    .Input("points: float32")
    .Input("neighbors: int32")
    .Input("sample_neigh_indices: int32")
    .Input("inv_badnwidth: float32")
    .Input("inv_radii: float32")
    .Output("pdfs: float32")
    .Attr("mode: int")
    .SetShapeFn([](shape_inference::InferenceContext* pIC) {
        shape_inference::ShapeHandle outputDims = 
            pIC->MakeShape({pIC->Dim(pIC->input(2), 0)});
        pIC->set_output(0, outputDims);
        return Status::OK();
    });

REGISTER_OP("ComputePdfPtGrads")
    .Input("points: float32")
    .Input("neighbors: int32")
    .Input("sample_neigh_indices: int32")
    .Input("inv_badnwidth: float32")
    .Input("inv_radii: float32")
    .Input("pdf_grads: float32")
    .Output("pt_grads: float32")
    .Attr("mode: int")
    .SetShapeFn([](shape_inference::InferenceContext* pIC) {
        shape_inference::ShapeHandle outputDims = 
            pIC->MakeShape({
            pIC->Dim(pIC->input(0), 0),
            pIC->Dim(pIC->input(0), 1)});
        pIC->set_output(0, outputDims);
        return Status::OK();
    });

namespace mccnn{

    /**
     *  Operation to compute the pdf of each neighbor.
     */
    class ComputePDFOp: public OpKernel{
        
        public:
        
            /**
             *  Constructor.
             *  @param  pContext    Constructor context of the operation.
             */
            explicit ComputePDFOp(
                OpKernelConstruction* pContext)
                :OpKernel(pContext){
                OP_REQUIRES_OK(pContext, pContext->GetAttr("mode", &mode_));
                OP_REQUIRES(pContext, mode_ >= 0 && mode_ < 2, 
                    errors::InvalidArgument("ComputePDFOp requires a mode between 0 and 1"));
            }
        
            /**
             *  Method to compute the operation.
             *  @param  pContext    Context of the operation.
             */
            void Compute(OpKernelContext * pContext) override{

                //Get the input tensors.
                const Tensor& inPts = pContext->input(0); 
                const Tensor& inNeighbors = pContext->input(1); 
                const Tensor& inSampleNeighIndices = pContext->input(2);
                const Tensor& inInvBandwidth = pContext->input(3);
                const Tensor& inInvRadii = pContext->input(4);

                //Get variables from tensors.
                unsigned int numPts = inPts.shape().dim_size(0);
                unsigned int numSamples = inSampleNeighIndices.shape().dim_size(0);
                unsigned int numNeighbors = inNeighbors.shape().dim_size(0);
                unsigned int numDimensions = inPts.shape().dim_size(1);

                //Get the pointers to GPU data from the tensors.
                const float* ptsGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<float>(inPts);
                const int* neighborsGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<int>(inNeighbors);
                const int* sampleNeighIGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<int>(inSampleNeighIndices);
                const float* invBandwidthGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<float>(inInvBandwidth);
                const float* inInvRadiiGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<float>(inInvRadii);

                //Check for the correctness of the input.
                OP_REQUIRES(pContext, numSamples == numPts, 
                    errors::InvalidArgument("ComputePDFOp expects the same points as samples."));
                OP_REQUIRES(pContext, numDimensions >= MIN_DIMENSIONS 
                    && numDimensions <= MAX_DIMENSIONS, 
                    errors::InvalidArgument("ComputePDFOp expects a valid number of dimension"));
                OP_REQUIRES(pContext, inInvBandwidth.shape().dim_size(0) == numDimensions, 
                    errors::InvalidArgument("ComputePDFOp expects a number of dimensions in"
                    " inInvBandwidth equal to the number of dimensions in the input points"));
                OP_REQUIRES(pContext, inInvRadii.shape().dim_size(0) == numDimensions, 
                    errors::InvalidArgument("ComputePDFOp expects a number of dimensions in"
                    " inInvRadii equal to the number of dimensions in the input points"));
                OP_REQUIRES(pContext, inNeighbors.dims() == 2 && 
                    inNeighbors.shape().dim_size(1) == 2, 
                    errors::InvalidArgument("ComputePDFOp expects a neighbor tensor with 2 dimensions "
                    "and 2 indices per neighbor."));

                //Get the gpu device.
                std::unique_ptr<mccnn::IGPUDevice> gpuDevice = make_unique<mccnn::TFGPUDevice>(pContext);

                //Create the output tensor.
                float* outputGPUPtr = nullptr;
                TensorShape outShape = TensorShape{numSamples};
                OP_REQUIRES_OK(pContext, mccnn::tensorflow_utils::allocate_output_tensor<float>
                    (0, pContext, outShape, &outputGPUPtr));

                //Compute the PDF  
                DIMENSION_SWITCH_CALL(numDimensions, mccnn::compute_pdf_gpu,
                    gpuDevice, mode_, numSamples, numNeighbors, inInvRadiiGPUPtr, 
                    invBandwidthGPUPtr, ptsGPUPtr, neighborsGPUPtr, sampleNeighIGPUPtr, 
                    outputGPUPtr);   
            }

        private:

            /**Mode used to compute the sigma value:
             *      - 0: Constant.
             *      - 1: Based on the number of points in the receptive field. 1/sqrt(N)*/
            int     mode_;
    };

    /**
     *  Operation to compute the gradients of each point wrt the pdf values.
     */
    class ComputePDFPtGradsOp: public OpKernel{
        
        public:
        
            /**
             *  Constructor.
             *  @param  pContext    Constructor context of the operation.
             */
            explicit ComputePDFPtGradsOp(
                OpKernelConstruction* pContext)
                :OpKernel(pContext){
                OP_REQUIRES_OK(pContext, pContext->GetAttr("mode", &mode_));
                OP_REQUIRES(pContext, mode_ >= 0 && mode_ < 2, 
                    errors::InvalidArgument("ComputePDFPtGradsOp requires a mode"
                    " between 0 and 1"));
            }
        
            /**
             *  Method to compute the operation.
             *  @param  pContext    Context of the operation.
             */
            void Compute(OpKernelContext * pContext) override{

                //Get the input tensors.
                const Tensor& inPts = pContext->input(0); 
                const Tensor& inNeighbors = pContext->input(1); 
                const Tensor& inSampleNeighIndices = pContext->input(2);
                const Tensor& inInvBandwidth = pContext->input(3);
                const Tensor& inInvRadii = pContext->input(4);
                const Tensor& inPDFGrads = pContext->input(5);

                //Get variables from tensors.
                unsigned int numSamples = inSampleNeighIndices.shape().dim_size(0);
                unsigned int numPts = inPts.shape().dim_size(0);
                unsigned int numNeighbors = inNeighbors.shape().dim_size(0);
                unsigned int numDimensions = inPts.shape().dim_size(1);

                //Get the pointers to GPU data from the tensors.
                const float* ptsGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<float>(inPts);
                const int* neighborsGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<int>(inNeighbors);
                const int* sampleNeighIGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<int>(inSampleNeighIndices);
                const float* invBandwidthGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<float>(inInvBandwidth);
                const float* inInvRadiiGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<float>(inInvRadii);
                const float* inPDFGradsGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<float>(inPDFGrads);

                //Check for the correctness of the input.
                OP_REQUIRES(pContext, numSamples == numPts, 
                    errors::InvalidArgument("ComputePDFPtGradsOp expects the same points as samples."));
                OP_REQUIRES(pContext, numDimensions >= MIN_DIMENSIONS 
                    && numDimensions <= MAX_DIMENSIONS, 
                    errors::InvalidArgument("ComputePDFPtGradsOp expects a valid number of dimension"));
                OP_REQUIRES(pContext, inInvBandwidth.shape().dim_size(0) == numDimensions, 
                    errors::InvalidArgument("ComputePDFPtGradsOp expects a number of dimensions in"
                    " inInvBandwidth equal to the number of dimensions in the input points"));
                OP_REQUIRES(pContext, inInvRadii.shape().dim_size(0) == numDimensions, 
                    errors::InvalidArgument("ComputePDFPtGradsOp expects a number of dimensions in"
                    " inInvRadii equal to the number of dimensions in the input points"));
                OP_REQUIRES(pContext, inNeighbors.dims() == 2 && 
                    inNeighbors.shape().dim_size(1) == 2, 
                    errors::InvalidArgument("ComputePDFPtGradsOp expects a neighbor tensor with 2 "
                    "dimensions and 2 indices per neighbor."));
                OP_REQUIRES(pContext, inPDFGrads.shape().dim_size(0) == numSamples, 
                    errors::InvalidArgument("ComputePDFPtGradsOp expects a number of pdf "
                    " gradients equal to the number of neighbors."));

                //Get the gpu device.
                std::unique_ptr<mccnn::IGPUDevice> gpuDevice = make_unique<mccnn::TFGPUDevice>(pContext);

                //Create the output tensor.
                float* outputGPUPtr = nullptr;
                TensorShape outShape = TensorShape{numPts, numDimensions};
                OP_REQUIRES_OK(pContext, mccnn::tensorflow_utils::allocate_output_tensor<float>
                    (0, pContext, outShape, &outputGPUPtr));

                //Compute the PDF  
                DIMENSION_SWITCH_CALL(numDimensions, mccnn::compute_pdf_grads_gpu,
                    gpuDevice, mode_, numPts, numSamples, numNeighbors, inInvRadiiGPUPtr,
                    invBandwidthGPUPtr, ptsGPUPtr, neighborsGPUPtr, sampleNeighIGPUPtr, 
                    inPDFGradsGPUPtr, outputGPUPtr);   
            }

        private:

            /**Mode used to compute the sigma value:
             *      - 0: Constant.
             *      - 1: Based on the number of points in the receptive field. 1/sqrt(N)*/
            int     mode_;
    };
}
            
REGISTER_KERNEL_BUILDER(Name("ComputePdf").Device(DEVICE_GPU), mccnn::ComputePDFOp);
REGISTER_KERNEL_BUILDER(Name("ComputePdfWithPtGrads").Device(DEVICE_GPU), mccnn::ComputePDFOp);
REGISTER_KERNEL_BUILDER(Name("ComputePdfPtGrads").Device(DEVICE_GPU), mccnn::ComputePDFPtGradsOp);