set(TAICHI_PROJECT_NAME "differentiable_mpm")

file(GLOB_RECURSE PROJECT_SOURCES "src/*.cpp" "src/*.h" "src/*/*.h" "src/*/*.cpp" "src/*.cuh")
file(GLOB_RECURSE PROJECT_OP_SOURCES "src/*.cc")
file(GLOB_RECURSE PROJECT_SOURCES_CUDA "src/*.cu")

set(SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")

set(TAICHI_PROJECT_LIBRARIES ${TAICHI_PROJECT_LIBRARIES} ${TAICHI_PROJECT_NAME} PARENT_SCOPE)

execute_process(COMMAND ${PYTHON_EXECUTABLE} -c
        "import tensorflow as tf; import sys;\
        sys.stdout.write(tf.sysconfig.get_include())"
        OUTPUT_VARIABLE TF_INCLUDE_DIRS)

execute_process(COMMAND ${PYTHON_EXECUTABLE} -c
        "import tensorflow as tf; import sys;\
        sys.stdout.write(tf.sysconfig.get_lib())"
        OUTPUT_VARIABLE TF_LIBRARY_DIR)



message("Including tensorflow include dir: ${TF_INCLUDE_DIRS}")
message("          tensorflow library dir: ${TF_LIBRARY_DIR}")
include_directories(${TF_INCLUDE_DIRS})

include_directories(external/partio/include)

if (NOT $ENV{CUDA_ARCH})
    set(${CUDA_ARCH} 61)
else()
    set(${CUDA_ARCH} $ENV{CUDA_ARCH})
endif()

message("CUDA Arch: ${CUDA_ARCH}")
find_package(CUDA 8.0 REQUIRED)
set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
#set(CUDA_NVCC_FLAGS ${CMAKE_CXX_FLAGS} ${CUDA_NVCC_FLAGS} -arch=compute_$ENV{CUDA_ARCH} -code=sm_$ENV{CUDA_ARCH} -Xcompiler "-fPIC" --maxrregcount 64 --use_fast_math --ptxas-options=-allow-expensive-optimizations=true,-O3)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Xcompiler "-fPIC" --use_fast_math --ptxas-options=-allow-expensive-optimizations=true,-O3)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -D__CUDA_ARCH___=${CUDA_ARCH}0 -std=c++14)
message("NVCC Flags: ${CUDA_NVCC_FLAGS}")
set(CUDA_PROPAGATE_HOST_FLAGS ON)
cuda_add_library(cudmpm SHARED ${PROJECT_SOURCES_CUDA})

add_subdirectory(external/partio)

add_library(taichi_tf_${TAICHI_PROJECT_NAME} SHARED ${PROJECT_OP_SOURCES})
add_library(taichi_${TAICHI_PROJECT_NAME} SHARED ${PROJECT_SOURCES})

target_link_libraries(taichi_${TAICHI_PROJECT_NAME} ${CORE_LIBRARY_NAME} ${SHARED_LIBS} partio cudmpm)
target_link_libraries(taichi_tf_${TAICHI_PROJECT_NAME} cudmpm ${TF_LIBRARY_DIR}/libtensorflow_framework.so)
find_package(ZLIB)
if (ZLIB_FOUND)
    target_link_libraries(taichi_${TAICHI_PROJECT_NAME} z)
endif(ZLIB_FOUND)
