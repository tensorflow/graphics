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
/// \brief Implementation of the CUDA operations to create pseudo-random
///     numbers
/////////////////////////////////////////////////////////////////////////////

#ifndef RND_UTILS_CUH_
#define RND_UTILS_CUH_

namespace mccnn{

    /**
     *  Pseudo random number generator: http://www.reedbeta.com/blog/quick-and-easy-gpu-random-numbers-in-d3d11/
     */
    __device__ __forceinline__ unsigned int wang_hash(unsigned int pSeed)
    {
        pSeed = (pSeed ^ 61) ^ (pSeed >> 16);
        pSeed *= 9;
        pSeed = pSeed ^ (pSeed >> 4);
        pSeed *= 0x27d4eb2d;
        pSeed = pSeed ^ (pSeed >> 15);
        return pSeed;
    }

    __device__ __forceinline__ unsigned int rand_xorshift(unsigned int pSeed)
    {
        pSeed ^= (pSeed << 13);
        pSeed ^= (pSeed >> 17);
        pSeed ^= (pSeed << 5);
        return pSeed;
    }

    __device__ __forceinline__ unsigned int seed_to_float(unsigned int pSeed)
    {
        return float(pSeed) * (1.0 / 4294967296.0);
    }
}

#endif