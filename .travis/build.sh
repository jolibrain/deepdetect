#!/bin/bash
set -e
set -x

LOCAL_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
ROOT_DIR=$(dirname "$LOCAL_DIR")
cd "$ROOT_DIR"

mkdir build
cd build

# Configure
CMAKE_ARGS=('-DCMAKE_VERBOSE_MAKEFILE=ON')
CMAKE_ARGS+=('-DCMAKE_INSTALL_PREFIX=../install')
if [ "$BUILD_CUDA" = 'true' ]; then
    CMAKE_ARGS+=('-DUSE_CUDNN=ON')
    CMAKE_ARGS+=('-DCUDA_NVCC_EXECUTABLE=/usr/local/bin/nvcc')
    CMAKE_ARGS+=('-DCUDA_ARCH=\"-gencode arch=compute_61,code=sm_61\"')
    export PATH="/usr/local/cuda/bin:${PATH}"
else
    CMAKE_ARGS+=('-DUSE_CPU_ONLY=ON')
    CMAKE_ARGS+=('-DUSE_XGBOOST=ON')
fi
cmake .. ${CMAKE_ARGS[*]}
make
