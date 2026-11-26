/////////////////////////////////////////////////////////////////////////////
// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
/////////////////////////////////////////////////////////////////////////////
/// \brief Declaration of the tensorflow gpu device.
/////////////////////////////////////////////////////////////////////////////

#ifndef TF_GPU_DEVICE_H_
#define TF_GPU_DEVICE_H_

#include <vector>
#include <memory>

#include "gpu_device.hpp"

namespace tensorflow{
    class OpKernelContext;
    class Tensor;
}

namespace mccnn{

    /**
     *  Tensorflow GPU Device object.
     */
    class TFGPUDevice: public IGPUDevice
    {
        public:

            /**
             *  Constructor.
             *  @param  pContext    Operation context.
             */
            explicit TFGPUDevice(
                tensorflow::OpKernelContext* pContext);

            /**
             *  Destructor.
             */
            virtual ~TFGPUDevice(void);

            /**
             *  Method to set a block of gpu memory to a certain value.
             *  @param  pDest   Pointer to the gpu memory we want to change.
             *  @param  pVal    Value we want to initialize the memory to.
             *  @param  pSize   Size of the buffer.
             */
            virtual void memset(void* pDest, int pVal, size_t pSize);

            /**
             *  Method to copy memory from gpu to gpu.
             *  @param  pDest   Destiantion pointer.
             *  @param  pSrc    Source pointer.
             *  @param  pSize   Number of bytes to copy.  
             */
            virtual void memcpy_device_to_device(void* pDest, void* pSrc, size_t pSize);

            /**
             *  Method to copy memory from gpu to cpu.
             *  @param  pDest   Destiantion pointer.
             *  @param  pSrc    Source pointer.
             *  @param  pSize   Number of bytes to copy.  
             */
            virtual void memcpy_device_to_host(void* pDest, void* pSrc, size_t pSize);

            /**
             *  Method to copy memory from cpu to gpu.
             *  @param  pDest   Destiantion pointer.
             *  @param  pSrc    Source pointer.
             *  @param  pSize   Number of bytes to copy.  
             */
            virtual void memcpy_host_to_device(void* pDest, void* pSrc, size_t pSize);

            /**
             *  Method to copy memory from cpu to symbol.
             *  @param  pDest   Destiantion pointer.
             *  @param  pSrc    Source pointer.
             *  @param  pSize   Number of bytes to copy.  
             */
            virtual void memcpy_host_to_symbol(void* pDest, void* pSrc, size_t pSize);

            /**
             *  Method to get the maximum number of active blocks per multiprocessor.
             *  @param  pBlockSize          Number of Threads per block.
             *  @param  pFunct              Function for which compute the number of blocks.
             *  @param  pSharedMemXBlock    Share memory per block.
             *  @return Number of active blocks per multiprocessor.
             */
            virtual int get_max_active_block_x_sm(
                const unsigned int pBlockSize, 
                const void* pFunct,
                const size_t pSharedMemXBlock);

            /**
             *  Method to evaluate if was an error in the last operation.
             *  @param  pFile   File from which the cuda operation was called.
             *  @param  pLine   Line of the file from which the cuda operation
             *      was called.
             */
            virtual void check_error(
                const char* pFile, 
                int pLine);

            /**
             *  Method to get a temporal gpu memory buffer of floats.
             *  @param  pSize   Number of elements in the buffer.
             *  @param  pCPUManaged Booleant that indicates if the buffer will
             *  be allocated on CPU for host device data transfers.
             *  @paramt Type of the elements in the buffer.
             *  @return Pointer to the buffer in memory.
             */
            virtual float* getFloatTmpGPUBuffer(const unsigned int pSize, bool pCPUManaged = false);

            /**
             *  Method to get a temporal gpu memory buffer of ints.
             *  @param  pSize   Number of elements in the buffer.
             *  @param  pCPUManaged Booleant that indicates if the buffer will
             *  be allocated on CPU for host device data transfers.
             *  @paramt Type of the elements in the buffer.
             *  @return Pointer to the buffer in memory.
             */
            virtual int* getIntTmpGPUBuffer(const unsigned int pSize, bool pCPUManaged = false);

            /**
             *  Method to get a temporal gpu memory buffer of int64_m.
             *  @param  pSize   Number of elements in the buffer.
             *  @param  pCPUManaged Booleant that indicates if the buffer will
             *  be allocated on CPU for host device data transfers.
             *  @paramt Type of the elements in the buffer.
             *  @return Pointer to the buffer in memory.
             */
            virtual mccnn::int64_m* getInt64TmpGPUBuffer(const unsigned int pSize, bool pCPUManaged = false);

            /**
             *  Method to get the cuda stream used.
             *  @return Cuda stream.
             */
            virtual const cudaStream_t& getCUDAStream();

        private:

            /**
             *  Private method to get a temporal gpu memory buffer.
             *  @param  pSize       Number of elements in the buffer.
             *  @param  pCPUManaged Booleant that indicates if the buffer will
             *  be allocated on CPU for host device data transfers.
             *  @paramt Type of the elements in the buffer.
             *  @return Pointer to the buffer in memory.
             */
            template<class T>
            T* getTmpGPUBuffer(const unsigned int pSize, bool pCPUManaged);

            /**Operation context.*/
            tensorflow::OpKernelContext*                        context_;
            /**Vector of temporal tensors allocated.*/
            std::vector<std::unique_ptr<tensorflow::Tensor>>    tmpTensors_;

    };
}

#endif