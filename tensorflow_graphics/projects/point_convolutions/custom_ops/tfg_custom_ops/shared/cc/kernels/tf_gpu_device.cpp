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
/// \brief Implementation of the tensorflow gpu device.
/////////////////////////////////////////////////////////////////////////////

#include "cuda_runtime.h"

#include "tf_gpu_device.hpp"
#include "tf_utils.hpp"

#include <stdio.h>
#include <stdlib.h>

namespace mccnn{

    TFGPUDevice::TFGPUDevice(tensorflow::OpKernelContext* pContext):
        IGPUDevice(), context_(pContext)
    {
        //Get GPU device.
        cudaDeviceProp prop;
        int device;
        cudaGetDevice(&device);
        cudaGetDeviceProperties(&prop, device);
        
        deviceProps_.warpSize_ = prop.warpSize;
        deviceProps_.numMPs_ = prop.multiProcessorCount;
        deviceProps_.maxThreadsXBlock_ = prop.maxThreadsPerBlock;
        deviceProps_.maxThreadsXMP_ = prop.maxThreadsPerMultiProcessor;
        deviceProps_.maxRegistersXBlock_ = prop.regsPerBlock;
        deviceProps_.maxRegistersXMP_ = prop.regsPerMultiprocessor;
        deviceProps_.sharedMemXBlock_ = prop.sharedMemPerBlock;
        deviceProps_.sharedMemXMP_ = prop.sharedMemPerMultiprocessor;
        deviceProps_.majorVersion_ = prop.major;
        deviceProps_.minorVersion_ = prop.minor;
    }

    TFGPUDevice::~TFGPUDevice()
    {}

    void TFGPUDevice::memset(void* pDest, int pVal, size_t pSize)
    {
        cudaMemsetAsync(pDest, pVal, pSize, context_->eigen_device<Eigen::GpuDevice>().stream());
    }

    void TFGPUDevice::memcpy_device_to_device(void* pDest, void* pSrc, size_t pSize)
    {
        cudaMemcpyAsync(pDest, pSrc, pSize, cudaMemcpyDeviceToDevice, context_->eigen_device<Eigen::GpuDevice>().stream());
    }

    void TFGPUDevice::memcpy_device_to_host(void* pDest, void* pSrc, size_t pSize)
    {
        cudaMemcpyAsync(pDest, pSrc, pSize, cudaMemcpyDeviceToHost, context_->eigen_device<Eigen::GpuDevice>().stream());
    }

    void TFGPUDevice::memcpy_host_to_device(void* pDest, void* pSrc, size_t pSize)
    {
        cudaMemcpyAsync(pDest, pSrc, pSize, cudaMemcpyHostToDevice, context_->eigen_device<Eigen::GpuDevice>().stream());
    }

    void TFGPUDevice::memcpy_host_to_symbol(void* pDest, void* pSrc, size_t pSize)
    {
        cudaMemcpyToSymbolAsync(pDest, pSrc, pSize, 0, cudaMemcpyHostToDevice, context_->eigen_device<Eigen::GpuDevice>().stream());
    }

    int TFGPUDevice::get_max_active_block_x_sm(
                const unsigned int pBlockSize, 
                const void* pFunct,
                const size_t pSharedMemXBlock)
    {
        int outputNumBlocks;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor ( 
            &outputNumBlocks, pFunct, pBlockSize, pSharedMemXBlock);
        return outputNumBlocks;
    }

    void TFGPUDevice::check_error(
        const char* pFile, 
        int pLine)
    {
        cudaError_t errorCode = cudaPeekAtLastError();
        if (errorCode != cudaSuccess) 
        {
            fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(errorCode), pFile, pLine);
            exit(errorCode);
            //TODO - Proper error handling, exceptions.
        }
    }

    float* TFGPUDevice::getFloatTmpGPUBuffer(const unsigned int pSize, bool pCPUManaged)
    {
        return this->getTmpGPUBuffer<float>(pSize, pCPUManaged);
    }

    int* TFGPUDevice::getIntTmpGPUBuffer(const unsigned int pSize, bool pCPUManaged)
    {
        return this->getTmpGPUBuffer<int>(pSize, pCPUManaged);
    }

    int64_m* TFGPUDevice::getInt64TmpGPUBuffer(const unsigned int pSize, bool pCPUManaged)
    {
        return this->getTmpGPUBuffer<mccnn::int64_m>(pSize, pCPUManaged);
    }

    const cudaStream_t& TFGPUDevice::getCUDAStream()
    {
        return context_->eigen_device<Eigen::GpuDevice>().stream();
    }

    template<class T>
    T* TFGPUDevice::getTmpGPUBuffer(const unsigned int pSize, bool pCPUManaged)
    {
        tensorflow::AllocatorAttributes allocatorAtt;
        if(pCPUManaged){
            allocatorAtt.set_on_host(true);
            allocatorAtt.set_gpu_compatible(true);
        }
        
        std::unique_ptr<tensorflow::Tensor> pTmpTensor = make_unique<tensorflow::Tensor>();
        TensorShape tmpShape = TensorShape{pSize};
        if(!TF_PREDICT_TRUE(context_->allocate_temp(
            DataTypeToEnum<T>::value, tmpShape, pTmpTensor.get(), allocatorAtt).ok())){
            fprintf(stderr,"Error allocating temporal tensor of %ld bytes.\n", sizeof(T)*pSize);
            exit(-1);
            //TODO - Proper error handling, exceptions.
        }
        auto tmpTensorFlat = pTmpTensor->flat<T>();
        tmpTensors_.push_back(std::move(pTmpTensor));
        return &(tmpTensorFlat(0));
    }

    template int* TFGPUDevice::getTmpGPUBuffer<int>(const unsigned int pSize, bool pCPUManaged);
    template mccnn::int64_m* TFGPUDevice::getTmpGPUBuffer<mccnn::int64_m>(const unsigned int pSize, bool pCPUManaged);
    template float* TFGPUDevice::getTmpGPUBuffer<float>(const unsigned int pSize, bool pCPUManaged);

}