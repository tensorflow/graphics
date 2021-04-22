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
/// \brief Implementation of the CUDA operations to compute the gradients
///     of a basis projection operation.
/////////////////////////////////////////////////////////////////////////////

#include "tfg_custom_ops/shared/cc/kernels/defines.hpp"
#include "tfg_custom_ops/shared/cc/kernels/math_helper.h"
#include "tfg_custom_ops/shared/cc/kernels/cuda_kernel_utils.h"

#include "tfg_custom_ops/basis_proj/cc/kernels/basis_utils.h"
#include "tfg_custom_ops/basis_proj/cc/kernels/basis_proj_grads.h"

///////////////////////// GPU

template<int K>
__global__ void compute_grads_in_features(
    const unsigned int pGroupFeatures,
    const unsigned int pNumSamples,       
    const unsigned int pNumInFeatures,
    const float* __restrict__ pInFeaturesGPUPtr,
    const float* __restrict__ pInPtProjBasisGPUPtr,
    const int2* __restrict__ pInNeighborsGPUPtr,
    const int* __restrict__ pSampleNeighIdsGPUPtr,
    const float* __restrict__ pInFeatGradsGPUPts,
    float* __restrict__ pOutFeatGradsGPUPtr,
    float* __restrict__ pOutPrProjBasisGradsGPUPtr)
{
    extern __shared__ float sharedMemory[];

    //Compute the total number of blocks executed and other
    //useful indices.
    unsigned int numGroupsXBlock = blockDim.x/K;
    unsigned int numFeatureBlocks = pNumInFeatures/pGroupFeatures;
    unsigned int localId = threadIdx.x%K;
    unsigned int groupId = threadIdx.x/K;
    unsigned int totalBlocks = pNumSamples*numFeatureBlocks;

    //Get the pointers to shared memory.
    float* accumFeatGrads = sharedMemory;
    float* features = &sharedMemory[blockDim.x*pGroupFeatures];
    float* inFeatGrads = &sharedMemory[blockDim.x*pGroupFeatures 
        + numGroupsXBlock*pGroupFeatures];

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

        //Initialize shared memory with the gradients of the point.
        for(int auxIter = threadIdx.x; 
            auxIter < K*pGroupFeatures; 
            auxIter += blockDim.x)
            inFeatGrads[auxIter] = pInFeatGradsGPUPts[
                sampleId*pNumInFeatures*K + featureOffset*K + auxIter];

        __syncthreads();

        //Iterate over the neighbors.
        for(int curNeighIter = groupId; 
            curNeighIter < numNeighbors; 
            curNeighIter += numGroupsXBlock)
        {
            int neighIndex = curNeighIter+rangePts.x;
            float curWeight = 0.0f; 
            float curWeightGrad = 0.0f;
            float curWeightAccumError = 0.0f;
            int2 neighAndSampleIndices;

            if(neighIndex < rangePts.y){
                //Get the neighbor index.
                neighAndSampleIndices = pInNeighborsGPUPtr[neighIndex];

                //Save the weights in shared memory.
                curWeight = pInPtProjBasisGPUPtr[neighIndex*K + localId];                

                //Save the features in shared memory.
                //THIS REQUIRES K >= pGroupFeatures
                if(localId < pGroupFeatures)
                    features[groupId*pGroupFeatures + localId] = pInFeaturesGPUPtr[
                        neighAndSampleIndices.x*pNumInFeatures 
                        + featureOffset + localId];
            }

            __syncthreads();

            //Iterate over the feature gradients.
            for(int featIter = 0; featIter < pGroupFeatures; ++featIter)
            {
                accumFeatGrads[featIter*blockDim.x + threadIdx.x] = 
                    inFeatGrads[featIter*K + localId]*curWeight;

                //(Kahan summation algorithm for numerical stabitility).
                float auxVar1 = inFeatGrads[featIter*K + localId]*
                    features[groupId*pGroupFeatures + featIter] - 
                    curWeightAccumError;
                float auxVar2 = curWeightGrad + auxVar1;
                curWeightAccumError = (auxVar2 - curWeightGrad) - auxVar1;
                curWeightGrad = auxVar2;
            }

            __syncthreads();

            //Accumulate the contribution of each K and store in memory.
            if(neighIndex < rangePts.y){
                atomicAdd(&pOutPrProjBasisGradsGPUPtr[neighIndex*K + localId], curWeightGrad);

                if(localId < pGroupFeatures){
                    //(Kahan summation algorithm for numerical stabitility).
                    float accum = 0.0f;
                    float accumError = 0.0f;
#pragma unroll
                    for(int kIter = 0; kIter < K; ++kIter){
                        float auxVar1 = accumFeatGrads[localId*blockDim.x + groupId*K + kIter] - accumError;
                        float auxVar2 = accum + auxVar1;
                        accumError = (auxVar2 - accum) - auxVar1;
                        accum = auxVar2;
                    }

                    atomicAdd(&pOutFeatGradsGPUPtr[neighAndSampleIndices.x*pNumInFeatures +
                        featureOffset + localId], accum);
                }
            }

            __syncthreads();
        }
    }
}

///////////////////////// CPU
          

template<int K>
void mccnn::basis_proj_grads_gpu(
    std::unique_ptr<IGPUDevice>& pDevice,
    const unsigned int pNumPts,
    const unsigned int pNumSamples,
    const unsigned int pNumNeighbors,
    const unsigned int pNumInFeatures,
    const float* pInPtFeaturesGPUPtr,
    const float* pInBasisGPUPtr,
    const int* pInNeighborsGPUPtr,
    const int* pInSampleNeighIGPUPtr,
    const float* pInGradientsGPUPtr,
    float* pOutFeatGradsGPUPtr,
    float* pOutBasisGradsGPUPtr)
{
    //Get the cuda stream.
    auto cudaStream = pDevice->getCUDAStream();

    //Initialize to zero the output array.
    pDevice->memset(pOutFeatGradsGPUPtr, 0, sizeof(float)*pNumPts*pNumInFeatures);
    pDevice->memset(pOutBasisGradsGPUPtr, 0, sizeof(float)*K*pNumNeighbors);
    pDevice->check_error(__FILE__, __LINE__);

    //Get the device properties.
    const GpuDeviceProperties& gpuProps = pDevice->get_device_properties();

    //Get information of the Device.
    unsigned int numMP = gpuProps.numMPs_;
    unsigned int blockSize = 64;

    //Determine the group of features.
    unsigned int groupFeatSize = min(MULTIPLE_IN_FEATURES, pNumInFeatures);

    //Calculate the shared memory needed.
    unsigned int sharedMemSize = groupFeatSize*(blockSize + blockSize/K + K)*sizeof(float);

    //Compute the number of blocks
    unsigned int numBlocks = pDevice->get_max_active_block_x_sm(
        blockSize, (const void*)compute_grads_in_features<K>, sharedMemSize);
    pDevice->check_error(__FILE__, __LINE__);

    //Calculate the total number of blocks to execute.
    unsigned int numFeatureBlocks = pNumInFeatures/groupFeatSize;
    unsigned int execBlocks = pNumSamples*numFeatureBlocks;
    unsigned int totalNumBlocks = numMP*numBlocks;
    totalNumBlocks = (totalNumBlocks > execBlocks)?execBlocks:totalNumBlocks;

    //Compute the accumulation of weighted input features.
    compute_grads_in_features<K>
        <<<totalNumBlocks, blockSize, sharedMemSize, cudaStream>>>(
        groupFeatSize, pNumSamples, pNumInFeatures, pInPtFeaturesGPUPtr,
        pInBasisGPUPtr, (const int2*)pInNeighborsGPUPtr, pInSampleNeighIGPUPtr,
        pInGradientsGPUPtr, pOutFeatGradsGPUPtr, pOutBasisGradsGPUPtr);
    pDevice->check_error(__FILE__, __LINE__);

}

///////////////////////// CPU Template declaration

#define COMPUTE_BASIS_PROJ_GRADS_TEMP_DECL(K)           \
    template void mccnn::basis_proj_grads_gpu<K>(       \
        std::unique_ptr<IGPUDevice>& pDevice,           \
        const unsigned int pNumPts,                     \
        const unsigned int pNumSamples,                 \
        const unsigned int pNumNeighbors,               \
        const unsigned int pNumInFeatures,              \
        const float* pInPtFeaturesGPUPtr,               \
        const float* pInBasisGPUPtr,                    \
        const int* pInNeighborsGPUPtr,                  \
        const int* pInSampleNeighIGPUPtr,               \
        const float* pInGradientsGPUPtr,                \
        float* pOutFeatGradsGPUPtr,                     \
        float* pOutBasisGradsGPUPtr);   

DECLARE_TEMPLATE_DIMS_BASIS(COMPUTE_BASIS_PROJ_GRADS_TEMP_DECL)