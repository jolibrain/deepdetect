cmake_minimum_required(VERSION 2.8.8)
project(deepdetect)


set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

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
#find_package(Eigen3 REQUIRED)
#include_directories("${EIGEN3_INCLUDE_DIR}")
#include_directories("${EIGEN3_INCLUDE_DIR}")

set(eigen_archive_hash "50812b426b7c")

#include_directories("$ENV{HOME}/include")

set(EIGEN3_INCLUDE_DIR ${CMAKE_BINARY_DIR}/tensorflow/src/tensorflow/dbuild/external/eigen_archive/eigen-eigen-${eigen_archive_hash} ${CMAKE_BINARY_DIR}/tensorflow/src/tensorflow/dbuild/external/eigen_archive)
include_directories("${EIGEN3_INCLUDE_DIR}")

# dependency on Boost
find_package(Boost 1.54 REQUIRED COMPONENTS filesystem thread system)

# optional packages -> deactivated in current state of having TF + Caffe in the same boat
#include(cmake/Cuda.cmake) # cuda + cudnn

# protobuf with tf
set(PROTOBUF_INCLUDE_DIR ${CMAKE_BINARY_DIR}/protobuf/include/)
set(PROTOBUF_LIB_DIR ${CMAKE_BINARY_DIR}/protobuf/lib/)
include_directories("${PROTOBUF_INCLUDE_DIR}")

# Tensorflow
message(STATUS "Fetching Tensorflow")
ExternalProject_Add(
  tensorflow
  PREFIX tensorflow
  INSTALL_DIR ${CMAKE_BINARY_DIR}
  DOWNLOAD_COMMAND git clone --recurse-submodules https://github.com/tensorflow/tensorflow
  CONFIGURE_COMMAND mkdir -p ${CMAKE_BINARY_DIR}/protobuf && mkdir -p dbuild && ../../../../patches/test_patch.sh ../../../../patches/tf/libtf_link.patch && ./configure && bazel --output_base dbuild/ build //tensorflow:libtensorflow.so //tensorflow/examples/label_image //tensorflow/python/tools:freeze_graph && cd dbuild/external/protobuf && ./autogen.sh && ./configure --prefix=${CMAKE_BINARY_DIR}/protobuf && make && make install
  INSTALL_COMMAND ""
  BUILD_COMMAND ""
  BUILD_IN_SOURCE 1
  )
set(TF_INC_DIR ${CMAKE_BINARY_DIR}/tensorflow/src/tensorflow/ ${CMAKE_BINARY_DIR}/tensorflow/src/tensorflow/dbuild/tensorflow/ ${CMAKE_BINARY_DIR}/tensorflow/src/tensorflow/dbuild/execroot/tensorflow/bazel-out/local-fastbuild/genfiles/)
set(TF_LIB_DIR ${CMAKE_BINARY_DIR}/tensorflow/src/tensorflow/dbuild/execroot/tensorflow/bazel-out/local-fastbuild/bin/tensorflow/)
include_directories("${TF_INC_DIR}")

# customized Caffe as external project
if (CAFFE_INC_DIR AND CAFFE_LIB_DIR)
  # do nothing
else()
  message(STATUS "Fetching and compiling customized caffe")
  if (CUDA_FOUND)
    if (HAVE_CUDNN)
      ExternalProject_Add(
	caffe_dd
	PREFIX caffe_dd
	INSTALL_DIR ${CMAKE_BINARY_DIR}
	URL https://github.com/beniz/caffe/archive/caffe_tf.tar.gz
	CONFIGURE_COMMAND ln -sf Makefile.config.gpu.cudnn Makefile.config && make
	INSTALL_COMMAND ""
	BUILD_IN_SOURCE 1
	)
    else()
ExternalProject_Add(
	caffe_dd
	PREFIX caffe_dd
	INSTALL_DIR ${CMAKE_BINARY_DIR}
	URL https://github.com/beniz/caffe/archive/caffe_tf.tar.gz
	CONFIGURE_COMMAND ln -sf Makefile.config.gpu Makefile.config && make
	INSTALL_COMMAND ""
	BUILD_IN_SOURCE 1
	)
    endif()
  else()
    ExternalProject_Add(
      caffe_dd
      PREFIX caffe_dd
      INSTALL_DIR ${CMAKE_BINARY_DIR}
      URL https://github.com/beniz/caffe/archive/caffe_tf.tar.gz
      CONFIGURE_COMMAND ln -sf Makefile.config.cpu Makefile.config && ${CMAKE_COMMAND} -E env PATH=${CMAKE_BINARY_DIR}/protobuf/bin:$ENV{PATH} make
      INSTALL_COMMAND ""
      BUILD_IN_SOURCE 1
      )
  endif()
  if (CUDA_FOUND)
    set(CAFFE_INC_DIR ${CMAKE_BINARY_DIR}/caffe_dd/src/caffe_dd/include ${CMAKE_BINARY_DIR}/caffe_dd/src/caffe_dd/build/src ${CUDA_INCLUDE_DIRS} ${PROTOBUF_INCLUDE_DIR})
    set(CAFFE_LIB_DIR ${CMAKE_BINARY_DIR}/caffe_dd/src/caffe_dd/build/lib ${CMAKE_BINARY_DIR}/caffe_dd/src/caffe_dd/build/src /usr/lib/x86_64-linux-gnu/hdf5/serial ${PROTOBUF_LIB_DIR} ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_curand_LIBRARY})
  else()
    set(CAFFE_INC_DIR ${CMAKE_BINARY_DIR}/caffe_dd/src/caffe_dd/include ${CMAKE_BINARY_DIR}/caffe_dd/src/caffe_dd/build/src /usr/include/hdf5/serial ${PROTOBUF_INCLUDE_DIR})
    set(CAFFE_LIB_DIR $ENV{HOME}/lib ${CMAKE_BINARY_DIR}/caffe_dd/src/caffe_dd/build/lib ${CMAKE_BINARY_DIR}/caffe_dd/src/caffe_dd/build/src /usr/lib/x86_64-linux-gnu/hdf5/serial ${PROTOBUF_LIB_DIR})
  endif()
endif()

add_dependencies(caffe_dd tensorflow)

# add the binary tree to the search path for include files
# so that we will find dd_config.h
include_directories("${PROJECT_BINARY_DIR}")
include_directories("${CAFFE_INC_DIR}")

# main library, main & tests
include_directories ("${PROJECT_SOURCE_DIR}/src")
add_subdirectory (src) 
add_subdirectory(main)

# templates
file(COPY "templates" DESTINATION ".")

# examples
file(COPY "examples" DESTINATION ".")

# unit testing
if (BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

# status
MESSAGE(STATUS "Build Tests          : ${BUILD_TESTS}")
