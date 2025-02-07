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
/// \brief Definitions.
/////////////////////////////////////////////////////////////////////////////

#ifndef DEFINES_H_
#define DEFINES_H_

#include <memory>

//Definition of the minimum and maximum number of dimensions.
#define MIN_DIMENSIONS 2
#define MAX_DIMENSIONS 6

//Macros to declare and call a template function with a variable
//number of dimensions.
#define DECLARE_TEMPLATE_DIMS(Func) \
    Func(2)                         \
    Func(3)                         \
    Func(4)                         \
    Func(5)                         \
    Func(6)                                              

#define DIMENSION_CASE_SWITCH(Dim, Func, ...)   \
    case Dim:                                   \
        Func<Dim>(__VA_ARGS__);                 \
        break;

#define DIMENSION_SWITCH_CALL(Var, Func, ...)       \
    switch(Var){                                    \
        DIMENSION_CASE_SWITCH(2, Func, __VA_ARGS__) \
        DIMENSION_CASE_SWITCH(3, Func, __VA_ARGS__) \
        DIMENSION_CASE_SWITCH(4, Func, __VA_ARGS__) \
        DIMENSION_CASE_SWITCH(5, Func, __VA_ARGS__) \
        DIMENSION_CASE_SWITCH(6, Func, __VA_ARGS__) \
    }  


//Definition of the min and max operation for cuda code.
#define MCCNN_MAX(a, b) (a < b) ? b : a;
#define MCCNN_MIN(a, b) (a > b) ? b : a;

namespace mccnn{
    //Definition of the int 64 bit.
    typedef long long int64_m;
}

//Definition of make unique for C++11 (Only available in C++14)
template<typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args)
{
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

#endif