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
/// \brief Implementation of the CUDA operations to compute the pdf of each 
///     neighboring point. 
/////////////////////////////////////////////////////////////////////////////

#include "tfg_custom_ops/shared/cc/kernels/defines.hpp"
#include "tfg_custom_ops/shared/cc/kernels/math_helper.h"
#include "tfg_custom_ops/shared/cc/kernels/cuda_kernel_utils.h"

#include "tfg_custom_ops/compute_pdf/cc/kernels/compute_pdf.h"

///////////////////////// GPU

/**
 *  Template method to get the bandwidth.
 */
template<int D, int M> 
struct bandwidth_struct{
    __forceinline__ __device__ mccnn::fpoint<D> operator()(
        const mccnn::fpoint<D>& pConstBandwidth,
        const int pNumNeighbors){return mccnn::fpoint<D>(0.0f);}
};

//Mode 0: Constant bandwidth.
template<int D> 
struct bandwidth_struct<D, 0>{ 
    __forceinline__ __device__ mccnn::fpoint<D> operator()(
        const mccnn::fpoint<D>& pConstBandwidth,
        const int pNumNeighbors){
        return pConstBandwidth;
    }
};

//Mode 1: Based on the number of points.
template<int D>  
struct bandwidth_struct<D, 1>{ 
    __forceinline__ __device__ mccnn::fpoint<D> operator()(
        const mccnn::fpoint<D>& pConstBandwidth,
        const int pNumNeighbors){
        return mccnn::fpoint<D>(sqrt((float)pNumNeighbors));
    }
};

/**
 *  GPU kernel to compute the pdfs on the gpu.
 *  @param  pNumSamples         Number of samples.
 *  @param  pNumNeighbors       Number of neighbors.
 *  @param  pInvRadii           Inverse radii used to select the 
 *      neighbors.
 *  @param  pInvBandwidth       Constant inverse banwidth.
 *  @param  pPts                Array of points.
 *  @param  pNeighbors          Array of neighbors.
 *  @param  pNeighIndexXSample  Indices of neighbors x sample.
 *  @param  pOutPDF             Output array with the pdfs.
 *  @paramt D                   Number of dimensions. 
 *  @paramt M                   Mode used to compute the bandwidth. 
 */
template<int D, int M>
__global__ void compute_pdf_gpu_kernel(
    const unsigned int pNumSamples,
    const unsigned int pNumNeighbors,
    const mccnn::fpoint<D>* __restrict__ pInvRadii,
    const mccnn::fpoint<D>* __restrict__ pInvBandwidth,
    const mccnn::fpoint<D>* __restrict__ pPts,
    const int2* __restrict__ pNeighbors,
    const int* __restrict__ pNeighIndexXSample,
    float* __restrict__ pOutPDF)
{
    //Get the global thread index.
    int iniPtIndex = mccnn::compute_global_index_gpu_funct();
    int totalThreads = mccnn::compute_total_threads_gpu_funct();

    for(unsigned int curIter = iniPtIndex; 
        curIter < pNumSamples; 
        curIter += totalThreads)
    {
        //Get the current point coordinates.
        mccnn::fpoint<D> curPt = pPts[curIter];

        //Get the range of points for this receptive field.
        int2 rangePts;
        rangePts.x = (curIter > 0)?pNeighIndexXSample[curIter-1]:0;
        rangePts.y = pNeighIndexXSample[curIter];
        int numPts = rangePts.y-rangePts.x;

        //Get the proper badnwidth.
        bandwidth_struct<D, M> getBwStruct;
        mccnn::fpoint<D> curBandwidth = getBwStruct(pInvBandwidth[0], pNumSamples);

        //Iterate over the points in the receptive field and compute the PDF.
        float accumPdf = 0.0f;
        float accumError = 0.0f; //Kahan summation algorithm for numerical stability.
        for(int i = 0; i < numPts; ++i)
        {
            //Get the neighbor coordinate.
            int2 neighIndex = pNeighbors[rangePts.x+i];

            //Comopute the contribution to the KDE.
            mccnn::fpoint<D> auxVec = (pPts[neighIndex.x] - curPt)*curBandwidth*pInvRadii[0];
            auxVec = curBandwidth*mccnn::expf(auxVec*auxVec*(-0.5f))*(0.39894228f);
            float localPDF = 1.0f;
#pragma unroll
            for(int d = 0; d < D; ++d)
                localPDF *= auxVec[d];

            //Accumulate the contribution (Kahan summation algorithm for numerical stabitility).
            float auxVar1 = localPDF - accumError;
            float auxVar2 = accumPdf + auxVar1;
            accumError = (auxVar2 - accumPdf) - auxVar1;
            accumPdf = auxVar2;
        }

        //Save the PDF.
        pOutPDF[curIter] = accumPdf;
    }
}

/**
 *  GPU kernel to compute gradients of the point wrt the pdf values
 *  on the gpu.
 *  @param  pNumSamples         Number of samples.
 *  @param  pNumNeighbors       Number of neighbors.
 *  @param  pInvRadii           Inverse radii used to select the 
 *      neighbors.
 *  @param  pInvBandwidth       Constant inverse banwidth.
 *  @param  pPts                Array of points.
 *  @param  pNeighbors          Array of neighbors.
 *  @param  pNeighIndexXSample  Indices of neighbors x sample.
 *  @param  pPDFGrads           Input pdf gradients.
 *  @param  pOutPtGrads         Output array with the point gradients.
 *  @paramt D                   Number of dimensions. 
 *  @paramt M                   Mode used to compute the bandwidth. 
 */
 template<int D, int M>
 __global__ void compute_pdf_grads_gpu_kernel(
    const unsigned int pNumSamples,
    const unsigned int pNumNeighbors,
    const mccnn::fpoint<D>* __restrict__ pInvRadii,
    const mccnn::fpoint<D>* __restrict__ pInvBandwidth,
    const mccnn::fpoint<D>* __restrict__ pPts,
    const int2* __restrict__ pNeighbors,
    const int* __restrict__ pNeighIndexXSample,
    const float* __restrict__ pPDFGrads,
    mccnn::fpoint<D>* __restrict__ pOutPtGrads)
 {
    //Get the global thread index.
    int iniPtIndex = mccnn::compute_global_index_gpu_funct();
    int totalThreads = mccnn::compute_total_threads_gpu_funct();
 
    for(unsigned int curIter = iniPtIndex; 
        curIter < pNumSamples; 
        curIter += totalThreads)
    { 
        //Get the current point coordinates.
        mccnn::fpoint<D> curPt = pPts[curIter];
 
        //Get the range of points for this receptive field.
        int2 rangePts;
        rangePts.x = (curIter > 0)?pNeighIndexXSample[curIter-1]:0;
        rangePts.y = pNeighIndexXSample[curIter];
        int numPts = rangePts.y-rangePts.x;
 
        //Get the proper badnwidth.
        bandwidth_struct<D, M> getBwStruct;
        mccnn::fpoint<D> curBandwidth = getBwStruct(pInvBandwidth[0], pNumSamples);

        //Get the current pdf gradient.
        float curPDFGrad = pPDFGrads[curIter];

        //Iterate over the points in the receptive field and compute the PDF.
        mccnn::fpoint<D> accumGradient(0.0f);
        mccnn::fpoint<D> accumError(0.0f); //Kahan summation algorithm for numerical stability.
        for(int i = 0; i < numPts; ++i)
        {
            //Get the neighbor coordinate.
            int2 neighIndex = pNeighbors[rangePts.x+i];
 
            //Comopute the contribution to the gradient.
            mccnn::fpoint<D> diffVec = (pPts[neighIndex.x] - curPt)*curBandwidth*pInvRadii[0];
            mccnn::fpoint<D> auxVec = curBandwidth*mccnn::expf(diffVec*diffVec*(-0.5f))*(0.39894228f);
            float localPDF = curPDFGrad;
#pragma unroll
            for(int d = 0; d < D; ++d)
                localPDF *= auxVec[d];
            diffVec = diffVec*curBandwidth*pInvRadii[0]*localPDF;
            
            //Accumulate the contribution (Kahan summation algorithm for numerical stabitility).
            mccnn::fpoint<D> auxVar1 = diffVec - accumError;
            mccnn::fpoint<D> auxVar2 = accumGradient + auxVar1;
            accumError = (auxVar2 - accumGradient) - auxVar1;
            accumGradient = auxVar2;

#pragma unroll
            for(int d = 0; d < D; ++d)
                atomicAdd(&pOutPtGrads[neighIndex.x][d], -diffVec[d]);
        }
#pragma unroll
        for(int d = 0; d < D; ++d)
            atomicAdd(&pOutPtGrads[curIter][d], accumGradient[d]);
    }
 }

///////////////////////// CPU

template<int D>
void mccnn::compute_pdf_gpu(
    std::unique_ptr<IGPUDevice>& pDevice,
    const unsigned int pMode,
    const unsigned int pNumSamples,
    const unsigned int pNumNeighbors,
    const float* pInGPUPtrInvRadii,
    const float* pInGPUPtrInvBandwidth,
    const float* pInGPUPtrPts,
    const int* pInGPUPtrNeighbors,
    const int* pInGPUPtrSampleNeighI,
    float* pOutGPUPtrPDFs)
{
    //Get the cuda stream.
    auto cudaStream = pDevice->getCUDAStream();

    //Get the device properties.
    const GpuDeviceProperties& gpuProps = pDevice->get_device_properties();

    //Calculate the ideal number of blocks for the selected block size.
    unsigned int numMP = gpuProps.numMPs_;
    unsigned int blockSize = gpuProps.warpSize_*2;
    unsigned int numBlocks = pDevice->get_max_active_block_x_sm(
        blockSize,(const void*)compute_pdf_gpu_kernel<D, 1>, 0);
    pDevice->check_error(__FILE__, __LINE__);

    //Calculate the total number of blocks to execute.
    unsigned int execBlocks = pNumSamples/blockSize;
    execBlocks += (pNumSamples%blockSize != 0)?1:0;
    unsigned int totalNumBlocks = numMP*numBlocks;
    totalNumBlocks = (totalNumBlocks > execBlocks)?execBlocks:totalNumBlocks;

    //Execute the appropriate cuda kernel based on the selected mode.
    if(pMode == 0){
        compute_pdf_gpu_kernel<D, 0><<<totalNumBlocks, blockSize, 0, cudaStream>>>(
            pNumSamples, pNumNeighbors, 
            (const mccnn::fpoint<D>*)pInGPUPtrInvRadii,
            (const mccnn::fpoint<D>*)pInGPUPtrInvBandwidth, 
            (const mccnn::fpoint<D>*)pInGPUPtrPts,
            (const int2*)pInGPUPtrNeighbors,
            pInGPUPtrSampleNeighI, 
            pOutGPUPtrPDFs);
    }else if(pMode == 1){
        compute_pdf_gpu_kernel<D, 1><<<totalNumBlocks, blockSize, 0, cudaStream>>>(
            pNumSamples, pNumNeighbors, 
            (const mccnn::fpoint<D>*)pInGPUPtrInvRadii,
            (const mccnn::fpoint<D>*)pInGPUPtrInvBandwidth, 
            (const mccnn::fpoint<D>*)pInGPUPtrPts,
            (const int2*)pInGPUPtrNeighbors,
            pInGPUPtrSampleNeighI, 
            pOutGPUPtrPDFs);
    }
    pDevice->check_error(__FILE__, __LINE__);
}

template<int D>
void mccnn::compute_pdf_grads_gpu(
    std::unique_ptr<IGPUDevice>& pDevice,
    const unsigned int pMode,
    const unsigned int pNumPts,
    const unsigned int pNumSamples,
    const unsigned int pNumNeighbors,
    const float* pInGPUPtrInvRadii,
    const float* pInGPUPtrInvBandwidth,
    const float* pInGPUPtrPts,
    const int* pInGPUPtrNeighbors,
    const int* pInGPUPtrSampleNeighI,
    const float* pInGPUPtrPDFGrad,
    float* pOutGPUPtrPtGrads)
{
    //Get the cuda stream.
    auto cudaStream = pDevice->getCUDAStream();

    //Initialize to zero the output array.
    pDevice->memset(pOutGPUPtrPtGrads, 0, sizeof(float)*pNumPts*D);
    pDevice->check_error(__FILE__, __LINE__);

    //Get the device properties.
    const GpuDeviceProperties& gpuProps = pDevice->get_device_properties();

    //Calculate the ideal number of blocks for the selected block size.
    unsigned int numMP = gpuProps.numMPs_;
    unsigned int blockSize = gpuProps.warpSize_*2;
    unsigned int numBlocks = pDevice->get_max_active_block_x_sm(
        blockSize,(const void*)compute_pdf_grads_gpu_kernel<D, 1>, 0);
    pDevice->check_error(__FILE__, __LINE__);

    //Calculate the total number of blocks to execute.
    unsigned int execBlocks = pNumSamples/blockSize;
    execBlocks += (pNumSamples%blockSize != 0)?1:0;
    unsigned int totalNumBlocks = numMP*numBlocks;
    totalNumBlocks = (totalNumBlocks > execBlocks)?execBlocks:totalNumBlocks;

    //Execute the appropriate cuda kernel based on the selected mode.
    if(pMode == 0){
        compute_pdf_grads_gpu_kernel<D, 0><<<totalNumBlocks, blockSize, 0, cudaStream>>>(
            pNumSamples, pNumNeighbors, 
            (const mccnn::fpoint<D>*)pInGPUPtrInvRadii,
            (const mccnn::fpoint<D>*)pInGPUPtrInvBandwidth, 
            (const mccnn::fpoint<D>*)pInGPUPtrPts,
            (const int2*)pInGPUPtrNeighbors,
            pInGPUPtrSampleNeighI, 
            pInGPUPtrPDFGrad,
            (mccnn::fpoint<D>*)pOutGPUPtrPtGrads);
    }else if(pMode == 1){
        compute_pdf_grads_gpu_kernel<D, 1><<<totalNumBlocks, blockSize, 0, cudaStream>>>(
            pNumSamples, pNumNeighbors, 
            (const mccnn::fpoint<D>*)pInGPUPtrInvRadii,
            (const mccnn::fpoint<D>*)pInGPUPtrInvBandwidth, 
            (const mccnn::fpoint<D>*)pInGPUPtrPts,
            (const int2*)pInGPUPtrNeighbors,
            pInGPUPtrSampleNeighI,
            pInGPUPtrPDFGrad, 
            (mccnn::fpoint<D>*)pOutGPUPtrPtGrads);
    }
    pDevice->check_error(__FILE__, __LINE__);
}

///////////////////////// CPU Template declaration

#define COMPUTE_PDF_TEMP_DECL(Dims)                 \
    template void mccnn::compute_pdf_gpu<Dims>(     \
        std::unique_ptr<IGPUDevice>& pDevice,       \
        const unsigned int pMode,                   \
        const unsigned int pNumSamples,             \
        const unsigned int pNumNeighbors,           \
        const float* pInGPUPtrInvRadii,             \
        const float* pInGPUPtrInvBandwidth,         \
        const float* pInGPUPtrPts,                  \
        const int* pInGPUPtrNeighbors,              \
        const int* pInGPUPtrSampleNeighI,           \
        float* pOutGPUPtrPDFs);

#define COMPUTE_PDF_GRADS_TEMP_DECL(Dims)               \
    template void mccnn::compute_pdf_grads_gpu<Dims>(   \
            std::unique_ptr<IGPUDevice>& pDevice,       \
            const unsigned int pMode,                   \
            const unsigned int pNumPts,                 \
            const unsigned int pNumSamples,             \
            const unsigned int pNumNeighbors,           \
            const float* pInGPUPtrInvRadii,             \
            const float* pInGPUPtrInvBandwidth,         \
            const float* pInGPUPtrPts,                  \
            const int* pInGPUPtrNeighbors,              \
            const int* pInGPUPtrSampleNeighI,           \
            const float* pInGPUPtrPDFGrad,              \
            float* pOutGPUPtrPtGrads);

DECLARE_TEMPLATE_DIMS(COMPUTE_PDF_TEMP_DECL)
DECLARE_TEMPLATE_DIMS(COMPUTE_PDF_GRADS_TEMP_DECL)