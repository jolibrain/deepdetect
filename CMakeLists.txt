cmake_minimum_required(VERSION 2.8.8)
project(deepdetect)


set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

include(ProcessorCount)
ProcessorCount(N)
include(ExternalProject)

set (deepdetect_VERSION_MAJOR 0)
set (deepdetect_VERSION_MINOR 1)

# options
OPTION(BUILD_TESTS "Should the tests be built")
 
# Get the current working branch
execute_process(
  COMMAND git rev-parse --abbrev-ref HEAD
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_BRANCH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Get the latest abbreviated commit hash of the working branch
execute_process(
  COMMAND git log -1 --format=%H
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_COMMIT_HASH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

set (CMAKE_CXX_FLAGS "-g -Wall -Wextra -fopenmp -fPIC -std=c++11 -O2 -DUSE_OPENCV -DUSE_LMDB")

# configure a header file to pass some of the CMake settings
# to the source code
configure_file (
  "${PROJECT_SOURCE_DIR}/dd_config.h.in"
  "${PROJECT_BINARY_DIR}/dd_config.h"
  )

configure_file(
  "${PROJECT_SOURCE_DIR}/src/githash.h.in"
  "${PROJECT_BINARY_DIR}/githash.h"
)

# dependency on Eigen for confusion matrix fast computation
if (USE_TF)
else()
   find_package(PkgConfig)
   pkg_search_module(Eigen3 REQUIRED eigen3)
endif()

set(TENSORFLOW_CC_DIR ${CMAKE_BINARY_DIR}/tensorflow_cc/src/tensorflow_cc/tensorflow_cc/build/)

set(EIGEN3_INCLUDE_DIR ${TENSORFLOW_CC_DIR}/tensorflow/tensorflow/contrib/makefile/downloads/eigen/)
include_directories("${EIGEN3_INCLUDE_DIR}")

# dependency on Boost
find_package(Boost 1.54 REQUIRED COMPONENTS filesystem thread system iostreams)

# optional packages
if (USE_CPU_ONLY)
else()
include(cmake/Cuda.cmake) # cuda + cudnn
endif()

if (USE_TF)
  # protobuf with tf
  set(PROTOBUF_INCLUDE_DIR ${CMAKE_BINARY_DIR}/protobuf/include/)
  set(PROTOBUF_LIB_DIR ${CMAKE_BINARY_DIR}/protobuf/lib/)
  include_directories("${PROTOBUF_INCLUDE_DIR}")

  # Tensorflow
  message(STATUS "Fetching Tensorflow")
  add_definitions(-DUSE_TF)
  if (CUDA_FOUND AND NOT USE_TF_CPU_ONLY)
    ExternalProject_Add(
      tensorflow_cc
      PREFIX tensorflow_cc
      INSTALL_DIR ${CMAKE_BINARY_DIR}
      DOWNLOAD_COMMAND git clone https://github.com/beniz/tensorflow_cc.git
      CONFIGURE_COMMAND cd tensorflow_cc && mkdir build && cd build && cmake -DTENSORFLOW_STATIC=OFF -DTENSORFLOW_SHARED=ON .. && make && ln -s ${CMAKE_BINARY_DIR}/tensorflow_cc/src/tensorflow_cc/tensorflow_cc/build/tensorflow/tensorflow/contrib/makefile/gen/protobuf ${CMAKE_BINARY_DIR}/protobuf
      BUILD_COMMAND ""
      INSTALL_COMMAND ""
      BUILD_IN_SOURCE 1
      )
  else()
    add_definitions(-DUSE_TF_CPU_ONLY)
    ExternalProject_Add(
      tensorflow_cc
      PREFIX tensorflow_cc
      INSTALL_DIR ${CMAKE_BINARY_DIR}
      DOWNLOAD_COMMAND git clone https://github.com/beniz/tensorflow_cc.git
      CONFIGURE_COMMAND cd tensorflow_cc && mkdir build && cd build && cmake -DTENSORFLOW_STATIC=OFF -DTENSORFLOW_SHARED_CPU=ON .. && make && ln -s ${CMAKE_BINARY_DIR}/tensorflow_cc/src/tensorflow_cc/tensorflow_cc/build/tensorflow/tensorflow/contrib/makefile/gen/protobuf ${CMAKE_BINARY_DIR}/protobuf
      BUILD_COMMAND ""
      INSTALL_COMMAND ""
      BUILD_IN_SOURCE 1
      )
  endif()

set(TF_INC_DIR ${TENSORFLOW_CC_DIR}/tensorflow/ ${TENSORFLOW_CC_DIR}/tensorflow/bazel-genfiles/ ${TENSORFLOW_CC_DIR}/tensorflow/tensorflow/contrib/makefile/downloads/nsync/public/)
set(TF_LIB_DIR ${TENSORFLOW_CC_DIR}/tensorflow/bazel-bin/tensorflow/)

include_directories("${TF_INC_DIR}")

endif() # USE_TF

# OpenCV
find_package(OpenCV REQUIRED)
set(OPENCV_VERSION ${OpenCV_VERSION_MAJOR})
include_directories(${OpenCV_INCLUDE_DIRS})
message(STATUS "OpenCV ${OPENCV_VERSION} (${OpenCV_VERSION}) found (${OpenCV_CONFIG_PATH})")

# customized Caffe as external project
if (CAFFE_INC_DIR AND CAFFE_LIB_DIR)
  # do nothing
else()
  message(STATUS "Configuring customized caffe")
  if (CUDA_FOUND AND NOT USE_CAFFE_CPU_ONLY)
    if (CUDA_ARCH)
    else()
      if (NOT ${CUDA_VERSION} MATCHES "8.0")
	set(CUDA_ARCH "-gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52")
      else()
	set(CUDA_ARCH "-gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_61,code=compute_61")
      endif()
    endif()
    if (HAVE_CUDNN AND USE_CUDNN)
      if (USE_TF)
	ExternalProject_Add(
	  caffe_dd
	  PREFIX caffe_dd
	  INSTALL_DIR ${CMAKE_BINARY_DIR}
	  URL https://github.com/beniz/caffe/archive/master.tar.gz
	  CONFIGURE_COMMAND ln -sf Makefile.config.gpu.cudnn Makefile.config && echo "OPENCV_VERSION:=${OPENCV_VERSION}" >> Makefile.config && ${CMAKE_COMMAND} -E env PATH=${CMAKE_BINARY_DIR}/protobuf/bin:$ENV{PATH} make CUDA_ARCH=${CUDA_ARCH} -j${N}
	  INSTALL_COMMAND ""
	  BUILD_IN_SOURCE 1
	  )
      elseif(JETSON)
	ExternalProject_Add(
	  caffe_dd
	  PREFIX caffe_dd
	  INSTALL_DIR ${CMAKE_BINARY_DIR}
	  URL https://github.com/beniz/caffe/archive/master.tar.gz
	  CONFIGURE_COMMAND ln -sf Makefile.config.gpu.cudnn.jetson Makefile.config && echo "OPENCV_VERSION:=${OPENCV_VERSION}" >> Makefile.config && make CUDA_ARCH=${CUDA_ARCH} -j${N}
	  INSTALL_COMMAND ""
	  BUILD_IN_SOURCE 1
	  )
      else() # USE_TF
	ExternalProject_Add(
	  caffe_dd
	  PREFIX caffe_dd
	  INSTALL_DIR ${CMAKE_BINARY_DIR}
	  URL https://github.com/beniz/caffe/archive/master.tar.gz
	  CONFIGURE_COMMAND ln -sf Makefile.config.gpu.cudnn Makefile.config && echo "OPENCV_VERSION:=${OPENCV_VERSION}" >> Makefile.config && make CUDA_ARCH=${CUDA_ARCH} -j${N}
	  INSTALL_COMMAND ""
	  BUILD_IN_SOURCE 1
	  )
      endif()
    else() # HAVE_CUDNN
      if (USE_TF)
	ExternalProject_Add(
	  caffe_dd
	  PREFIX caffe_dd
	  INSTALL_DIR ${CMAKE_BINARY_DIR}
	  URL https://github.com/beniz/caffe/archive/master.tar.gz
	  CONFIGURE_COMMAND ln -sf Makefile.config.gpu Makefile.config && echo "OPENCV_VERSION:=${OPENCV_VERSION}" >> Makefile.config && ${CMAKE_COMMAND} -E env PATH=${TENSORFLOW_CC_DIR}/tensorflow/tensorflow/contrib/makefile/gen/protobuf/bin:$ENV{PATH} make CUDA_ARCH=${CUDA_ARCH} -j${N}
	  INSTALL_COMMAND ""
	  BUILD_IN_SOURCE 1
	  )
      else() # USE_TF
	ExternalProject_Add(
	  caffe_dd
	  PREFIX caffe_dd
	  INSTALL_DIR ${CMAKE_BINARY_DIR}
	  URL https://github.com/beniz/caffe/archive/master.tar.gz
	  CONFIGURE_COMMAND ln -sf Makefile.config.gpu Makefile.config && echo "OPENCV_VERSION:=${OPENCV_VERSION}" >> Makefile.config && make CUDA_ARCH=${CUDA_ARCH} -j${N}
	  INSTALL_COMMAND ""
	  BUILD_IN_SOURCE 1
	  )
      endif()
    endif() # HAVE_CUDNN
  else() # CUDA_FOUND
    add_definitions(-DUSE_CAFFE_CPU_ONLY)
    if (USE_TF)
      ExternalProject_Add(
	caffe_dd
	PREFIX caffe_dd
	INSTALL_DIR ${CMAKE_BINARY_DIR}
	URL https://github.com/beniz/caffe/archive/master.tar.gz
	CONFIGURE_COMMAND ln -sf Makefile.config.cpu Makefile.config && echo "OPENCV_VERSION:=${OPENCV_VERSION}" >> Makefile.config && ${CMAKE_COMMAND} -E env PATH=${CMAKE_BINARY_DIR}/protobuf/bin:$ENV{PATH} make -j${N}
	INSTALL_COMMAND ""
	BUILD_IN_SOURCE 1
	)
    else() # USE_TF
      ExternalProject_Add(
	caffe_dd
	PREFIX caffe_dd
	INSTALL_DIR ${CMAKE_BINARY_DIR}
	URL https://github.com/beniz/caffe/archive/master.tar.gz
	CONFIGURE_COMMAND ln -sf Makefile.config.cpu Makefile.config && echo "OPENCV_VERSION:=${OPENCV_VERSION}" >> Makefile.config && make -j${N}
	INSTALL_COMMAND ""
	BUILD_IN_SOURCE 1
	)
    endif()
  endif()
  if (JETSON)
    set(HDF5_LIB /usr/lib/aarch64-linux-gnu/hdf5/serial)
  else()
    set(HDF5_LIB /usr/lib/x86_64-linux-gnu/hdf5/serial)
  endif()
  if (CUDA_FOUND)
    set(CAFFE_INC_DIR ${CMAKE_BINARY_DIR}/caffe_dd/src/caffe_dd/include ${CMAKE_BINARY_DIR}/caffe_dd/src/caffe_dd/build/src ${CUDA_INCLUDE_DIRS} ${PROTOBUF_INCLUDE_DIR})
    set(CAFFE_LIB_DIR ${CMAKE_BINARY_DIR}/caffe_dd/src/caffe_dd/build/lib ${CMAKE_BINARY_DIR}/caffe_dd/src/caffe_dd/build/src ${HDF5_LIB} ${PROTOBUF_LIB_DIR} ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_curand_LIBRARY})
  else()
    set(CAFFE_INC_DIR ${CMAKE_BINARY_DIR}/caffe_dd/src/caffe_dd/include ${CMAKE_BINARY_DIR}/caffe_dd/src/caffe_dd/build/src /usr/include/hdf5/serial ${PROTOBUF_INCLUDE_DIR})
    set(CAFFE_LIB_DIR $ENV{HOME}/lib ${CMAKE_BINARY_DIR}/caffe_dd/src/caffe_dd/build/lib ${CMAKE_BINARY_DIR}/caffe_dd/src/caffe_dd/build/src ${HDF5_LIB} ${PROTOBUF_LIB_DIR})
  endif()
endif()

if (USE_TF)
  add_dependencies(caffe_dd tensorflow_cc)
endif()

# XGBoost
if (USE_XGBOOST)
  message(STATUS "Configuring XGBoost")
  add_definitions(-DUSE_XGBOOST)
  if (CUDA_FOUND AND USE_XGBOOST_GPU)
    add_definitions(-DUSE_XGBOOST_GPU)
#    set(XGB_NVCC_FLAGS,"--expt-extended-lambda;-gencode arch=compute_35,code=compute_35")
    ExternalProject_Add(
      xgboost
      PREFIX xgboost
      INSTALL_DIR ${CMAKE_BINARY_DIR}
      DOWNLOAD_COMMAND git clone --recursive https://github.com/dmlc/xgboost.git
      CONFIGURE_COMMAND wget https://github.com/NVlabs/cub/archive/1.6.4.zip && unzip 1.6.4.zip && rm 1.6.4.zip && mkdir build && cd build &&
      cmake .. -DPLUGIN_UPDATER_GPU=ON -DCUB_DIRECTORY=${CMAKE_BINARY_DIR}/xgboost/src/xgboost/cub-1.6.4/ -DCUDA_NVCC_FLAGS=-Xcompiler\ -fPIC\ --expt-extended-lambda\ -gencode\ arch=compute_30,code=compute_30\ -gencode\ arch=compute_35,code=compute_35\ -gencode\ arch=compute_50,code=compute_50\ -gencode\ arch=compute_52,code=compute_52\ -gencode\ arch=compute_61,code=compute_61 && make && make
      INSTALL_COMMAND ""
      BUILD_IN_SOURCE 1
      )
  else()
    ExternalProject_Add(
      xgboost
      PREFIX xgboost
      INSTALL_DIR ${CMAKE_BINARY_DIR}
      DOWNLOAD_COMMAND git clone --recursive https://github.com/dmlc/xgboost.git
      CONFIGURE_COMMAND mkdir build && cd build && cmake .. && make -j ${N}
      INSTALL_COMMAND ""
      BUILD_IN_SOURCE 1
      )
  endif()
  set(XGBOOST_INC_DIR ${CMAKE_BINARY_DIR}/xgboost/src/xgboost/include ${CMAKE_BINARY_DIR}/xgboost/src/xgboost/src ${CMAKE_BINARY_DIR}/xgboost/src/xgboost/rabit/include ${CMAKE_BINARY_DIR}/xgboost/src/xgboost/dmlc-core/include ${CMAKE_BINARY_DIR}/xgboost/src/xgboost/dmlc-core/src/)
  set(XGBOOST_LIB_DIR ${CMAKE_BINARY_DIR}/xgboost/src/xgboost/lib ${CMAKE_BINARY_DIR}/xgboost/src/xgboost/build/ ${CMAKE_BINARY_DIR}/xgboost/src/xgboost/build/dmlc-core)
endif()

if (USE_TSNE)
  message(STATUS "Configuring T-SNE")
  add_definitions(-DUSE_TSNE)
      ExternalProject_Add(
      Multicore-TSNE
      PREFIX Multicore-TSNE
      INSTALL_DIR ${CMAKE_BINARY_DIR}
      DOWNLOAD_COMMAND git clone https://github.com/beniz/Multicore-TSNE.git
      CONFIGURE_COMMAND cd multicore_tsne && mkdir build && cd build && cmake ..
      BUILD_COMMAND cd multicore_tsne/build/ && make
      INSTALL_COMMAND ""
      BUILD_IN_SOURCE 1
      )
    set(TSNE_INC_DIR ${CMAKE_BINARY_DIR}/Multicore-TSNE/src/Multicore-TSNE/multicore_tsne)
    set(TSNE_LIB_DIR ${CMAKE_BINARY_DIR}/Multicore-TSNE/src/Multicore-TSNE/multicore_tsne/build)
endif()

# add the binary tree to the search path for include files
# so that we will find dd_config.h
include_directories("${PROJECT_BINARY_DIR}")
include_directories(${CAFFE_INC_DIR} ${XGBOOST_INC_DIR} ${TSNE_INC_DIR})

# main library, main & tests
include_directories ("${PROJECT_SOURCE_DIR}/src")
add_subdirectory (src) 
add_subdirectory(main)

# templates
file(COPY "templates" DESTINATION ".")

# examples
file(COPY "examples" DESTINATION ".")

# patches
file(COPY "patches" DESTINATION ".")

# unit testing
if (BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

# status
MESSAGE(STATUS "Build Tests          : ${BUILD_TESTS}")
