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
/// \brief Utilities for the tensorflow interface.
/////////////////////////////////////////////////////////////////////////////
#ifndef TF_UTILS_H_
#define TF_UTILS_H_

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"

using namespace tensorflow;

namespace mccnn{
        
    namespace tensorflow_utils
    {
        /**
         *  Template to get the pointer of a tensor.
         *  @param  pTensor Tensor from which the pointer will be obtained.
         *  @return Pointer of the GPU array containing the tensor.
         */
        template<class T>
        inline T* get_tensor_pointer(const Tensor& pTensor)
        {
            auto tensorFlat = pTensor.flat<T>();
            return &(tensorFlat(0));
        }

        /**
         *  Template to get the constant pointer of a tensor.
         *  @param  pTensor Tensor from which the pointer will be obtained.
         *  @return Pointer of the GPU array containing the tensor.
         */
        template<class T>
        inline const T* get_const_tensor_pointer(const Tensor& pTensor)
        {
            auto tensorFlat = pTensor.flat<T>();
            return &(tensorFlat(0));
        }

        /**
         *  Template to allocate an output for a tensorflow operation.
         *  @param  pIndex      Index of the output.
         *  @param  pContext    Context of the operation.
         *  @param  pShape      Shape of the output tensor.
         *  @param  pOutPtr     Output parameter with the GPU array pointer.
         *  @return Status of the allocation operation.
         */
        template<class T>
        inline Status allocate_output_tensor(
            const unsigned int pIndex, 
            OpKernelContext* pContext, 
            TensorShape& pShape,
            T** pOutPtr)
        {
            Tensor* outTensor = nullptr;
            Status retStatus = pContext->allocate_output(
                pIndex, pShape, &outTensor);
            auto outTensorFlat = outTensor->flat<T>();
            *pOutPtr = &(outTensorFlat(0));
            return retStatus;
        }
    }
}

#endif