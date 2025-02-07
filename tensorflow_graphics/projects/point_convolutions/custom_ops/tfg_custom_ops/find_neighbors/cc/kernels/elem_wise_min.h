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
/// \brief Declaraion of the CUDA operations to perform a min operation 
///     element wise.
/////////////////////////////////////////////////////////////////////////////

#ifndef ELEM_WISE_MIN_CUH_
#define ELEM_WISE_MIN_CUH_

#include "tfg_custom_ops/shared/cc/kernels/defines.hpp"
#include "tfg_custom_ops/shared/cc/kernels/gpu_device.hpp"
#include <memory>

namespace mccnn{
        
    /**
     *  Method to perform element wise minimum operation with a given
     *  minimum value.
     *  @param      pDevice                 Device.
     *  @param      pNumElements            Number of elements in the array.
     *  @param      pMinValue               Minimum value.
     *  @param      pValuesGPUPtr           Input/Output pointer to the vector of 
     *      values in GPU memory
     *  @rparamt    T                       Type of the elements.
     */
    template<class T>
    void elem_wise_min_value(
        std::unique_ptr<IGPUDevice>& pDevice,
        const unsigned int pNumElements,
        const T pMinValue,
        T* pValuesGPUPtr);
}

#endif