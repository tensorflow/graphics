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
/// \brief Neural network utils.
/////////////////////////////////////////////////////////////////////////////

#ifndef NN_UTILS_H_
#define NN_UTILS_H_

//Leaky relu weight.
#define LEAKY_RELU_WEIGHT 0.2f

//ELU weight
#define ELU_WEIGHT 1.0f

namespace mccnn{

    /**
     *  Template to define activation functions.
     */
    template<int A> 
    struct activation_function_struct{
        __forceinline__ __device__ float forward(
            float pValue){return 0.0f;}

        __forceinline__ __device__ float backward(
            float pValue){return 0.0f;}
    };

    // A = 0 : RELU
    template<> 
    struct activation_function_struct<0>{
        __forceinline__ __device__ float forward(
            float pValue){
                return max(pValue, 0.0f);
        }

        __forceinline__ __device__ float backward(
            float pValue){
                return (pValue > 0.0f)? 1.0f: 0.0f;
        }
    };

    // A = 1 : Leaky RELU
    template<> 
    struct activation_function_struct<1>{
        __forceinline__ __device__ float forward(
            float pValue){
                return (pValue >= 0.0f)?pValue:pValue*LEAKY_RELU_WEIGHT;
        }

        __forceinline__ __device__ float backward(
            float pValue){
                return (pValue > 0.0f)?1.0f:LEAKY_RELU_WEIGHT;
        }
    };

    // A = 2 : ELU
    template<> 
    struct activation_function_struct<2>{
        __forceinline__ __device__ float forward(
            const float pValue){
                return (pValue <= 0.0f)?ELU_WEIGHT*(exp(pValue) - 1.0f):pValue;
        }

        __forceinline__ __device__ float backward(
            const float pValue){
                return (pValue <= 0.0f)?ELU_WEIGHT + pValue:1.0f;
        }
    };

    // A = 2 : Exp
    template<> 
    struct activation_function_struct<3>{
        __forceinline__ __device__ float forward(
            const float pValue){
                return exp(pValue);
        }

        __forceinline__ __device__ float backward(
            const float pValue){
                return pValue;
        }
    };
}

#endif