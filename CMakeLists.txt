cmake_minimum_required(VERSION 2.8.8)
project(deepdetect)

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

set (CMAKE_CXX_FLAGS "-g -Wall -Wextra -fopenmp -fPIC -std=c++1y")

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

# dependency to Eigen for confusion matrix fast computation
find_package(Eigen3 REQUIRED)
include_directories("${EIGEN3_INCLUDE_DIR}")

# add the binary tree to the search path for include files
# so that we will find dd_config.h
include_directories("${PROJECT_BINARY_DIR}")
include_directories("${CAFFE_INC_DIR}")

# optional packages
find_package(CUDA)

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