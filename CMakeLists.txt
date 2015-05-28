cmake_minimum_required(VERSION 2.8.8)
project(deepdetect)
include(ExternalProject)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

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

set (CMAKE_CXX_FLAGS "-g -Wall -Wextra -fopenmp -fPIC -std=c++11")

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
find_package(Eigen3 REQUIRED)
include_directories("${EIGEN3_INCLUDE_DIR}")

# dependency on Boost
find_package(Boost 1.54 REQUIRED COMPONENTS filesystem thread system)

# optional packages
find_package(CUDA)

# customized Caffe as external project
message(STATUS "Fetching and compiling customized caffe")
if (CUDA_FOUND)
  ExternalProject_Add(
    caffe_dd
    PREFIX caffe_dd
    INSTALL_DIR ${CMAKE_BINARY_DIR}
    URL https://github.com/beniz/caffe/archive/master_dd_integ.tar.gz
    CONFIGURE_COMMAND ln -sf Makefile.config.gpu Makefile.config && make
    INSTALL_COMMAND ""
    BUILD_IN_SOURCE 1
    )
else()
  ExternalProject_Add(
    caffe_dd
    PREFIX caffe_dd
    INSTALL_DIR ${CMAKE_BINARY_DIR}
    URL https://github.com/beniz/caffe/archive/master_dd_integ.tar.gz
    CONFIGURE_COMMAND ln -sf Makefile.config.cpu Makefile.config && make
    INSTALL_COMMAND ""
    BUILD_IN_SOURCE 1
    )
endif()
if (CUDA_FOUND)
  set(CAFFE_INC_DIR ${CMAKE_BINARY_DIR}/caffe_dd/src/caffe_dd/include ${CMAKE_BINARY_DIR}/caffe_dd/src/caffe_dd/build/src ${CUDA_INCLUDE_DIRS})
  set(CAFFE_LIB_DIR ${CMAKE_BINARY_DIR}/caffe_dd/src/caffe_dd/build/lib ${CMAKE_BINARY_DIR}/caffe_dd/src/caffe_dd/build/src /usr/lib/x86_64-linux-gnu/hdf5/serial ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_curand_LIBRARY})
else()
  set(CAFFE_INC_DIR ${CMAKE_BINARY_DIR}/caffe_dd/src/caffe_dd/include ${CMAKE_BINARY_DIR}/caffe_dd/src/caffe_dd/build/src /usr/include/hdf5/serial)
  set(CAFFE_LIB_DIR ${CMAKE_BINARY_DIR}/caffe_dd/src/caffe_dd/build/lib ${CMAKE_BINARY_DIR}/caffe_dd/src/caffe_dd/build/src /usr/lib/x86_64-linux-gnu/hdf5/serial)
endif()

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