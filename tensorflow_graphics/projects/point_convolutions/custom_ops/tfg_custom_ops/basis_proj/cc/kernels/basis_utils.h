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
/// \brief Basis utils file.
/////////////////////////////////////////////////////////////////////////////

#ifndef BASIS_UTILS_CUH_
#define BASIS_UTILS_CUH_

#include "tfg_custom_ops/shared/cc/kernels/defines.hpp"
#include "tfg_custom_ops/shared/cc/kernels/math_helper.h"

//Definition of the minimum and maximum kernel points.
#define MIN_BASIS 8
#define MAX_BASIS 64

//Definition of the number of which the number of features should be 
// multiple of.
#define MULTIPLE_IN_FEATURES 8

//Macros to declare and call a template function with a variable
//number of dimensions and variable basis functions.
#define DECLARE_TEMPLATE_DIMS_BASIS(Func)  \
    Func(8 )                         \
    Func(16)                         \
    Func(32)                         \
    Func(64)

#define BASIS_CASE_SWITCH(K, Func, ...)                     \
    case K:                                                 \
        Func<K>(__VA_ARGS__);                               \
        break;

#define BASIS_SWITCH_CALL(Var, Func, ...)               \
    switch(Var){                                        \
        BASIS_CASE_SWITCH(8, Func, __VA_ARGS__)         \
        BASIS_CASE_SWITCH(16, Func, __VA_ARGS__)        \
        BASIS_CASE_SWITCH(32, Func, __VA_ARGS__)        \
        BASIS_CASE_SWITCH(64, Func, __VA_ARGS__)        \
    };


#endif