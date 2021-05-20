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
/// \brief Implementation of the CUDA operations to compute a monte carlo 
///     convolution.
/////////////////////////////////////////////////////////////////////////////

#include "tfg_custom_ops/shared/cc/kernels/defines.hpp"
#include "tfg_custom_ops/shared/cc/kernels/math_helper.h"
#include "tfg_custom_ops/shared/cc/kernels/cuda_kernel_utils.h"
#include "tfg_custom_ops/basis_proj/cc/kernels/basis_utils.h"
#include "tfg_custom_ops/basis_proj/cc/kernels/basis_proj.h"

///////////////////////// GPU

//WARNING - Group features should be equal or smaller than K.
template<int K>
__global__ void compute_weighted_in_features(
    const unsigned int pGroupFeatures,
    const unsigned int pNumSamples,       
    const unsigned int pNumInFeatures,
    const float* __restrict__ pInPtProjBasisGPUPtr,
    const int2* __restrict__ pInNeighborsGPUPtr,
    const int* __restrict__ pSampleNeighIdsGPUPtr,
    const float* __restrict__ pInFeaturesGPUPts,
    float* __restrict__ pOutProjFeatGPUPtr)
{
    extern __shared__ float sharedMemory[];

    //Get the pointers to shared memory.
    float* accumWeightFeatures = sharedMemory;
    float* features = &sharedMemory[blockDim.x*pGroupFeatures];

    //Compute the total number of blocks executed and other
    //useful indices.
    unsigned int numGroupsXBlock = blockDim.x/K;
    unsigned int numFeatureBlocks = pNumInFeatures/pGroupFeatures;
    unsigned int localId = threadIdx.x%K;
    unsigned int groupId = threadIdx.x/K;
    unsigned int totalBlocks = pNumSamples*numFeatureBlocks;

    for(int curIter = blockIdx.x; 
        curIter < totalBlocks; 
        curIter += gridDim.x)
    {
        //Get the sample id and the feature offset.
        int sampleId = curIter/numFeatureBlocks;
        int featureOffset = (curIter%numFeatureBlocks)*pGroupFeatures;

        //Get the range of points for this receptive field.
        int2 rangePts;
        rangePts.x = (sampleId > 0)?pSampleNeighIdsGPUPtr[sampleId-1]:0;
        rangePts.y = pSampleNeighIdsGPUPtr[sampleId];
        int numNeighbors = rangePts.y - rangePts.x;
        numNeighbors += numGroupsXBlock-(numNeighbors%numGroupsXBlock);

        //Initialize shared memory.
#pragma unroll(8)
        for(int featIter = 0; featIter < pGroupFeatures; ++featIter)
            accumWeightFeatures[featIter*blockDim.x + threadIdx.x] = 0.0f;

        //Iterate over the neighbors.
        for(int curNeighIter = groupId; 
            curNeighIter < numNeighbors; 
            curNeighIter += numGroupsXBlock)
        {
            int neighIndex = curNeighIter+rangePts.x;
            float curWeight = 0.0;

            if(neighIndex < rangePts.y){
                //Get the neighbor index.
                int2 neighAndSampleIndices = pInNeighborsGPUPtr[neighIndex];

                //Save the weights in shared memory.
                curWeight = pInPtProjBasisGPUPtr[neighIndex*K + localId];

                //Save the features in shared memory.
                if(localId < pGroupFeatures)
                    features[groupId*pGroupFeatures + localId] = pInFeaturesGPUPts[
                        neighAndSampleIndices.x*pNumInFeatures 
                        + featureOffset + localId];
            }else if(localId < pGroupFeatures){
                features[groupId*pGroupFeatures + localId] = 0.0f;
            }

            __syncthreads();

            //Iterate over the features.
            //TODO - Kahan summation. However performance drops by half.
#pragma unroll(8)
            for(int featIter = 0; featIter < pGroupFeatures; ++featIter)
                accumWeightFeatures[featIter*blockDim.x + threadIdx.x] += 
                    features[groupId*pGroupFeatures + featIter]*curWeight;

            __syncthreads();
        }

        //Save the result.
        if(threadIdx.x < K){
            for(int featIter = 0; featIter < pGroupFeatures; ++featIter){
                float accumContribs = 0.0f;
#pragma unroll(4)
                for(int groupIter = 0; groupIter < numGroupsXBlock; ++groupIter){
                    accumContribs += accumWeightFeatures[featIter*blockDim.x + 
                        localId + groupIter*K];
                }

                pOutProjFeatGPUPtr[sampleId*pNumInFeatures*K + (featureOffset + featIter)*K 
                    + localId] = accumContribs;
            }
        }

        __syncthreads();
    }
}

///////////////////////// CPU
          
template<int K>
void mccnn::basis_proj_gpu(
    std::unique_ptr<IGPUDevice>& pDevice,
    const unsigned int pNumSamples,
    const unsigned int pNumInFeatures,
    const float* pInPtFeaturesGPUPtr,
    const float* pInBasisGPUPtr,
    const int* pInNeighborsGPUPtr,
    const int* pInSampleNeighIGPUPtr,
    float*  pOutFeaturesGPUPtr)
{
    //Get the cuda stream.
    auto cudaStream = pDevice->getCUDAStream();

    //Initialize to zero the output array.
    pDevice->memset(pOutFeaturesGPUPtr, 0, sizeof(float)*pNumSamples*pNumInFeatures*K);
    pDevice->check_error(__FILE__, __LINE__);

    //Get the device properties.
    const GpuDeviceProperties& gpuProps = pDevice->get_device_properties();

    //Get information of the Device.
    unsigned int numMP = gpuProps.numMPs_;

    //Define the block size.
    unsigned int  blockSize = 64;

    //Determine the group of features.
    unsigned int groupFeatSize = min(MULTIPLE_IN_FEATURES, pNumInFeatures);

    //Calculate the shared memory needed.
    unsigned int sharedMemSize = (blockSize*(groupFeatSize+1))*sizeof(float);

    //Compute the number of blocks
    unsigned int numBlocks = pDevice->get_max_active_block_x_sm(
        blockSize, (const void*)compute_weighted_in_features<K>, sharedMemSize);
    pDevice->check_error(__FILE__, __LINE__);

    //Calculate the total number of blocks to execute.
    unsigned int numFeatureBlocks = pNumInFeatures/groupFeatSize;
    unsigned int execBlocks = pNumSamples*numFeatureBlocks;
    unsigned int totalNumBlocks = numMP*numBlocks;
    totalNumBlocks = (totalNumBlocks > execBlocks)?execBlocks:totalNumBlocks;

    //Compute the accumulation of weighted input features.
    compute_weighted_in_features<K>
        <<<totalNumBlocks, blockSize, sharedMemSize, cudaStream>>>(
        groupFeatSize, pNumSamples, pNumInFeatures, pInBasisGPUPtr, 
        (const int2*)pInNeighborsGPUPtr, pInSampleNeighIGPUPtr,
        pInPtFeaturesGPUPtr, pOutFeaturesGPUPtr);
    pDevice->check_error(__FILE__, __LINE__);
}

///////////////////////// CPU Template declaration

#define COMPUTE_BASIS_PROJ_KS_TEMP_DECL(K)              \
    template void mccnn::basis_proj_gpu<K>(             \
        std::unique_ptr<IGPUDevice>& pDevice,           \
        const unsigned int pNumSamples,                 \
        const unsigned int pNumInFeatures,              \
        const float* pInPtFeaturesGPUPtr,               \
        const float* pInBasisGPUPtr,                    \
        const int* pInNeighborsGPUPtr,                  \
        const int* pInSampleNeighIGPUPtr,               \
        float*  pOutFeaturesGPUPtr);

DECLARE_TEMPLATE_DIMS_BASIS(COMPUTE_BASIS_PROJ_KS_TEMP_DECL)