/////////////////////////////////////////////////////////////////////////////
/// \file basis_proj.cpp
///
/// \brief C++ operations definition to project input features to a set of 
///     basis functions.
///
/// \copyright Copyright (c) 2019 Visual Computing group of Ulm University,  
///            Germany. See the LICENSE file at the top-level directory of 
///            this distribution.
///
/// \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
/////////////////////////////////////////////////////////////////////////////

#include "tfg_custom_ops/shared/cc/kernels/defines.hpp"
#include "tfg_custom_ops/shared/cc/kernels/tf_utils.hpp"
#include "tfg_custom_ops/shared/cc/kernels/tf_gpu_device.hpp"

#include "tfg_custom_ops/basis_proj/cc/kernels/basis_utils.h"
#include "tfg_custom_ops/basis_proj/cc/kernels/basis_proj.h"
#include "tfg_custom_ops/basis_proj/cc/kernels/basis_proj_grads.h"

/**
 *  Declaration of the tensorflow operations.
 */
REGISTER_OP("BasisProj")
    .Input("basis_neighbors: float32")
    .Input("pt_features: float32")
    .Input("neighbors: int32")
    .Input("sample_neigh_indices: int32")
    .Output("features: float32")
    .SetShapeFn([](shape_inference::InferenceContext* pIC) {
        shape_inference::ShapeHandle outputDims = 
            pIC->MakeShape({
                pIC->Dim(pIC->input(3), 0),
                pIC->Dim(pIC->input(1), 1),
                pIC->Dim(pIC->input(0), 1)});
        pIC->set_output(0, outputDims);
        return Status::OK();
    });

REGISTER_OP("BasisProjGrads")
    .Input("basis_neighbors: float32")
    .Input("pt_features: float32")
    .Input("neighbors: int32")
    .Input("sample_neigh_indices: int32")
    .Input("in_gradients: float32")
    .Output("basis_gradients: float32")
    .Output("feat_gradients: float32")
    .SetShapeFn([](shape_inference::InferenceContext* pIC) {
        pIC->set_output(0, pIC->input(0));
        pIC->set_output(1, pIC->input(1));
        return Status::OK();
    });

namespace mccnn{

    /**
     *  Operation to project input features into a set of basis functions.
     */
    class BasisProjOp: public OpKernel{
        
        public:
        
            /**
             *  Constructor.
             *  @param  pContext    Constructor context of the operation.
             */
            explicit BasisProjOp(
                OpKernelConstruction* pContext)
                :OpKernel(pContext){
            }
        
            /**
             *  Method to compute the operation.
             *  @param  pContext    Context of the operation.
             */
            void Compute(OpKernelContext * pContext) override{

                //Get the input tensors.
                const Tensor& inBasisIns = pContext->input(0);
                const Tensor& inPtFeatures = pContext->input(1); 
                const Tensor& inNeighbors = pContext->input(2); 
                const Tensor& inSampleNeighIndices = pContext->input(3);

                //Get variables from tensors.
                unsigned int numPts = inPtFeatures.shape().dim_size(0);
                unsigned int numSamples = inSampleNeighIndices.shape().dim_size(0);
                unsigned int numNeighbors = inNeighbors.shape().dim_size(0);
                unsigned int numBasis = inBasisIns.shape().dim_size(1);
                unsigned int numInFeatures = inPtFeatures.shape().dim_size(1);

                //Get the pointers to GPU data from the tensors.
                const float* inBasisGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<float>(inBasisIns);
                const float* inPtFeaturesGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<float>(inPtFeatures);
                const int* inNeighborsGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<int>(inNeighbors);
                const int* inSampleNeighIGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<int>(inSampleNeighIndices);

                //Check for the correctness of the input.
                bool correct = false;
                for(int i = MIN_BASIS; i <= MAX_BASIS; i*=2)
                    correct = correct || (numBasis == i);
                OP_REQUIRES(pContext, correct, 
                    errors::InvalidArgument("BasisProjOp expects a valid number of basis functions."));
                OP_REQUIRES(pContext, numInFeatures > 0 && (numInFeatures < 8 
                    || numInFeatures%MULTIPLE_IN_FEATURES==0), 
                    errors::InvalidArgument("BasisProjOp expects a valid number of input features, "
                    "between 1 and "+std::to_string(MULTIPLE_IN_FEATURES)+" or multiple of "
                    +std::to_string(MULTIPLE_IN_FEATURES)));
                OP_REQUIRES(pContext, inBasisIns.shape().dim_size(0) == numNeighbors, 
                    errors::InvalidArgument("BasisProjOp expects the same number of basis as the number of neighbors."));
                OP_REQUIRES(pContext, inNeighbors.dims() == 2 && inNeighbors.shape().dim_size(1) == 2, 
                    errors::InvalidArgument("BasisProjOp expects a neighbor tensor with 2 dimensions "
                    "and 2 indices per neighbor."));

                //Get the gpu device.
                std::unique_ptr<mccnn::IGPUDevice> gpuDevice = make_unique<mccnn::TFGPUDevice>(pContext);

                //Create temporal tensor.
                //float* tmpBuffer = gpuDevice->getFloatTmpGPUBuffer(numSamples*numInFeatures*numBasis);
                float* outputGPUPtr = nullptr;
                TensorShape outShape = TensorShape{numSamples, numInFeatures, numBasis};
                OP_REQUIRES_OK(pContext, mccnn::tensorflow_utils::allocate_output_tensor<float>
                    (0, pContext, outShape, &outputGPUPtr));

                //Compute the projected points.
                BASIS_SWITCH_CALL(numBasis, 
                    mccnn::basis_proj_gpu,
                    gpuDevice, 
                    numSamples, numInFeatures, inPtFeaturesGPUPtr, 
                    inBasisGPUPtr, inNeighborsGPUPtr, 
                    inSampleNeighIGPUPtr, outputGPUPtr)
            }

    };

    /**
     *  Operation to compute a monte carlo convolution.
     */
    class BasisProjGradsOp: public OpKernel{
        
        public:
        
            /**
             *  Constructor.
             *  @param  pContext    Constructor context of the operation.
             */
            explicit BasisProjGradsOp(
                OpKernelConstruction* pContext)
                :OpKernel(pContext){
            }
        
            /**
             *  Method to compute the operation.
             *  @param  pContext    Context of the operation.
             */
            void Compute(OpKernelContext * pContext) override{

                //Get the input tensors.
                const Tensor& inBasisIns = pContext->input(0);
                const Tensor& inPtFeatures = pContext->input(1); 
                const Tensor& inNeighbors = pContext->input(2); 
                const Tensor& inSampleNeighIndices = pContext->input(3);
                const Tensor& inGradients = pContext->input(4);

                //Get variables from tensors.
                unsigned int numPts = inPtFeatures.shape().dim_size(0);
                unsigned int numSamples = inSampleNeighIndices.shape().dim_size(0);
                unsigned int numNeighbors = inNeighbors.shape().dim_size(0);
                unsigned int numBasis = inBasisIns.shape().dim_size(1);
                unsigned int numInFeatures = inPtFeatures.shape().dim_size(1);

                //Get the pointers to GPU data from the tensors.
                const float* inBasisGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<float>(inBasisIns);
                const float* inPtFeaturesGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<float>(inPtFeatures);
                const int* inNeighborsGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<int>(inNeighbors);
                const int* inSampleNeighIGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<int>(inSampleNeighIndices);
                const float* inGradientsGPUPtr = mccnn::tensorflow_utils::get_const_tensor_pointer<float>(inGradients);

                //Check for the correctness of the input.
                bool correct = false;
                for(int i = MIN_BASIS; i <= MAX_BASIS; i*=2)
                    correct = correct || (numBasis == i);
                OP_REQUIRES(pContext, correct, 
                    errors::InvalidArgument("BasisProjOpGrads expects a valid number of basis functions."));
                OP_REQUIRES(pContext, numInFeatures > 0 && (numInFeatures < 8 
                    || numInFeatures%MULTIPLE_IN_FEATURES==0), 
                    errors::InvalidArgument("BasisProjOpGrads expects a valid number of input features, "
                    "between 1 and "+std::to_string(MULTIPLE_IN_FEATURES)+" or multiple of "
                    +std::to_string(MULTIPLE_IN_FEATURES)));
                OP_REQUIRES(pContext, inBasisIns.shape().dim_size(0) == numNeighbors, 
                    errors::InvalidArgument("BasisProjOpGrads expects the same number of basis as the number of neighbors."));
                OP_REQUIRES(pContext, inNeighbors.dims() == 2 && inNeighbors.shape().dim_size(1) == 2, 
                    errors::InvalidArgument("BasisProjOpGrads expects a neighbor tensor with 2 dimensions "
                    "and 2 indices per neighbor."));
                OP_REQUIRES(pContext, inGradients.dims() == 3 && inGradients.shape().dim_size(0) == numSamples &&
                    inGradients.shape().dim_size(1) == numInFeatures && inGradients.shape().dim_size(2) == numBasis, 
                    errors::InvalidArgument("BasisProjOpGrads expects a neighbor tensor with 2 dimensions "
                    "and 2 indices per neighbor."));

                //Get the gpu device.
                std::unique_ptr<mccnn::IGPUDevice> gpuDevice = make_unique<mccnn::TFGPUDevice>(pContext);

                //Create temporal tensor.
                float* output1GPUPtr = nullptr;
                float* output2GPUPtr = nullptr;
                TensorShape out1Shape = TensorShape{numNeighbors, numBasis};
                TensorShape out2Shape = TensorShape{numPts, numInFeatures};
                OP_REQUIRES_OK(pContext, mccnn::tensorflow_utils::allocate_output_tensor<float>
                    (0, pContext, out1Shape, &output1GPUPtr));
                OP_REQUIRES_OK(pContext, mccnn::tensorflow_utils::allocate_output_tensor<float>
                    (1, pContext, out2Shape, &output2GPUPtr));

                //Compute the projected points.
                BASIS_SWITCH_CALL(numBasis, 
                    mccnn::basis_proj_grads_gpu,
                    gpuDevice, 
                    numPts, numSamples, numNeighbors, 
                    numInFeatures, inPtFeaturesGPUPtr, 
                    inBasisGPUPtr, inNeighborsGPUPtr, 
                    inSampleNeighIGPUPtr, inGradientsGPUPtr,
                    output2GPUPtr, output1GPUPtr)
            }
    };
}

REGISTER_KERNEL_BUILDER(Name("BasisProj").Device(DEVICE_GPU), mccnn::BasisProjOp);
REGISTER_KERNEL_BUILDER(Name("BasisProjGrads").Device(DEVICE_GPU), mccnn::BasisProjGradsOp);