cmake_minimum_required(VERSION 3.14)
project(deepdetect)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(RELEASE "release mode" OFF)
option(WARNING "enable warning" ON)
option(USE_NCNN "use NCNN backend" OFF)
option(USE_TORCH "use libtorch backend" ON)
option(USE_PYTORCH_WORKER "use managed Python PyTorch worker backend" OFF)
option(USE_PREBUILT_TORCH "use an installed LibTorch package" OFF)
option(USE_HDF5 "use HDF5" ON)
option(USE_TENSORRT "use TensorRT backend" OFF)
option(USE_TENSORRT_OSS "alias of USE_TENSORRT" OFF)
option(USE_XGBOOST "use XGBOOST backend" OFF)
option(USE_DLIB "use Dlib backend" OFF)
option(USE_CUDNN "enable CUDNN" ON)
option(USE_TSNE "enable TSNE" OFF)
option(USE_CUDA_CV "use CUDA with OpenCV (Requires OpenCV build for CUDA)" OFF)
option(USE_SIMSEARCH "build index and search services" OFF)
option(USE_ANNOY "use annoy as indexer" OFF)
option(USE_FAISS "use FAISS as indexer" ON)
option(BUILD_SPDLOG "build SPDLOG instead of using system library" ON)
option(BUILD_PROTOBUF "build PROTOBUF instead of using system library" ON)
option(USE_BOOST_BACKTRACE "use boost backtrace" ON)
option(USE_OPENMP "use OpenMP" ON)

set(TORCH_LOCATION "" CACHE PATH
  "Deprecated LibTorch prefix; use Torch_DIR or CMAKE_PREFIX_PATH")
set(TORCHVISION_SOURCE_DIR "" CACHE PATH
  "Optional existing torchvision source tree")
option(TORCHVISION_UPDATE_DISCONNECTED
  "Do not contact the torchvision git remote during ExternalProject updates" OFF)
if (TORCH_LOCATION)
  set(USE_PREBUILT_TORCH ON CACHE BOOL
    "use an installed LibTorch package" FORCE)
endif()

foreach(RETIRED_BACKEND_FLAG
    USE_CAFFE USE_CAFFE_DEBUG USE_CAFFE_CPU_ONLY
    USE_CAFFE2 USE_CAFFE2_CPU_ONLY
    USE_TF USE_TF_CPU_ONLY)
  if (DEFINED ${RETIRED_BACKEND_FLAG} AND ${RETIRED_BACKEND_FLAG})
    message(FATAL_ERROR
      "${RETIRED_BACKEND_FLAG}=ON is no longer supported: the Caffe, Caffe2, "
      "and TensorFlow backends were removed. Migrate services to Torch or "
      "ONNX/TensorRT before upgrading.")
  endif()
endforeach()

if (USE_TENSORRT_OSS)
  set(USE_TENSORRT ON)
endif()

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

include(ProcessorCount)
ProcessorCount(N)
include(ExternalProject)
include(CheckLibraryExists)

# options
OPTION(BUILD_TESTS "Should the tests be built")
OPTION(BUILD_TOOLS "Should the tools be built")
OPTION(USE_COMMAND_LINE "build command line JSON API" ON)
OPTION(USE_JSON_API "build internal JSON API" ON)
OPTION(USE_HTTP_SERVER "build cppnet-lib version of http JSON API " OFF)
OPTION(USE_HTTP_SERVER_OATPP "build oatpp version of http JSON API" ON)
OPTION(USE_OATPP_SWAGGER "Enable Swagger documentation endpoint" ON)

# user set variables
set(USE_OPENCV_VERSION CACHE STRING "OpenCV Version used by DeepDetect")

# Get the current working branch
execute_process(
  COMMAND git rev-parse --abbrev-ref HEAD
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_BRANCH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Get the current version
execute_process(
  COMMAND git describe --tags --broken
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_VERSION
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Get the latest abbreviated commit hash of the working branch
execute_process(
  COMMAND git log -1 --format=%H
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_COMMIT_HASH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

if (NOT EXISTS ${CMAKE_BINARY_DIR}/src)
  execute_process(
	COMMAND  bash -c "mkdir ${CMAKE_BINARY_DIR}/src")
endif()

if (USE_OPENMP)
  if (APPLE)
    set(FOPENMP "-Xclang -fopenmp")
    find_library(OPENMP_LIBRARIES REQUIRED
      NAMES omp libomp
      HINTS /opt/homebrew/Cellar/libomp/*/lib/)

    message(STATUS "OpenMP location: ${OPENMP_LIBRARIES}")
    find_path(OPENMP_INCLUDE_DIRS REQUIRED
      NAMES omp.h
      HINTS /opt/homebrew/Cellar/libomp/*/include/)
    message(STATUS "OpenMP header ${OPENMP_INCLUDE_DIRS}")
    include_directories(SYSTEM ${OPENMP_INCLUDE_DIRS})
  else()
    set(FOPENMP "-fopenmp")
  endif()
else()
  set(FOPENMP "")
endif()

if (USE_PREBUILT_TORCH)
    set(DEEPDETECT_CXX_STANDARD 20)
else()
    set(DEEPDETECT_CXX_STANDARD 17)
endif()

if (APPLE)
    set(CMAKE_CXX_FLAGS "-g -O2 -Wall -Wextra ${FOPENMP} -fPIC -std=c++${DEEPDETECT_CXX_STANDARD}")

    message(STATUS "Apple build: CUDA is disabled")
    set(USE_CUDNN OFF)
    set(USE_CUDA_CV OFF)
else()
    set(CMAKE_CXX_FLAGS "-g -O2 -Wall -Wextra ${FOPENMP} -fPIC -std=c++${DEEPDETECT_CXX_STANDARD} -Wl,--no-as-needed -Wno-dangling-reference -Wno-redundant-move -Wno-error=array-bounds -Wno-class-memaccess -ltcmalloc_minimal")
endif()

set(CMAKE_CXX_STANDARD ${DEEPDETECT_CXX_STANDARD})
add_definitions("-DUSE_OPENCV" "-DUSE_LMDB")

if(WARNING)
  string(APPEND CMAKE_CXX_FLAGS " -Werror")
endif()

if(RELEASE)
    set(BUILD_TYPE release)
else()
    set(BUILD_TYPE dev)
endif()

if (USE_COMMAND_LINE)
  set(USE_JSON_API ON)
  string(APPEND CMAKE_CXX_FLAGS " -DUSE_COMMAND_LINE")
endif()

if (USE_HTTP_SERVER)
  set(USE_JSON_API ON)
  string(APPEND CMAKE_CXX_FLAGS " -DUSE_HTTP_SERVER")
endif()
if (USE_HTTP_SERVER_OATPP)
  set(USE_JSON_API ON)
  string(APPEND CMAKE_CXX_FLAGS " -DUSE_HTTP_SERVER_OATPP")
endif()

if (USE_JSON_API)
  string(APPEND CMAKE_CXX_FLAGS " -DUSE_JSON_API")
endif()

# VARIANT

if (APPLE)
    # Variant is not on brew
    set(VARIANT_VERSION v1.2.0)
    message(STATUS "Download Variant library ${VARIANT_VERSION}")
    ExternalProject_Add(
        variant
        PREFIX variant
        UPDATE_DISCONNECTED 1
        URL https://github.com/mapbox/variant/archive/refs/tags/${VARIANT_VERSION}.tar.gz
        URL_HASH SHA256=7059f4420d504c4bc96f8a462a0f6d029c5be914ba55cc030a0a773366dd7bc8
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
    )
    set(VARIANT_INCLUDE_DIRS "${CMAKE_BINARY_DIR}/variant/src/variant/include/")
endif()

# PROTOBUF

# use protobuf same version as pytorch one
set(PROTOBUF_VERSION 3.20.3 CACHE STRING
  "Protobuf version to build when BUILD_PROTOBUF=ON")
set(PROTOBUF_URL_HASH
  e51cc8fc496f893e2a48beb417730ab6cbcb251142ad8b2cd1951faa5c76fe3d
  CACHE STRING "SHA256 hash for the bundled protobuf source archive")
if (BUILD_PROTOBUF)
    ExternalProject_Add(
        protobuf
        PREFIX protobuf
        UPDATE_DISCONNECTED 1
        URL https://github.com/protocolbuffers/protobuf/releases/download/v${PROTOBUF_VERSION}/protobuf-cpp-${PROTOBUF_VERSION}.tar.gz
        URL_HASH SHA256=${PROTOBUF_URL_HASH}
        SOURCE_SUBDIR cmake
        CMAKE_ARGS
        -DCMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE
        -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
        -DBUILD_SHARED_LIBS:BOOL=ON
        -Dprotobuf_BUILD_TESTS:BOOL=OFF
        -Dprotobuf_BUILD_EXAMPLES:BOOL=OFF
        INSTALL_COMMAND ""
    )
    set(PROTOBUF_INCLUDE_DIR ${CMAKE_BINARY_DIR}/protobuf/src/protobuf/src)
    set(PROTOBUF_LIB_DIR ${CMAKE_BINARY_DIR}/protobuf/src/protobuf-build/)
    set(PROTOBUF_PROTOC ${PROTOBUF_LIB_DIR}/protoc)

    if (APPLE)
      set(PROTOBUF_LIB_DEPS ${PROTOBUF_LIB_DIR}/libprotobuf.dylib ${PROTOBUF_LIB_DIR}/libprotoc.dylib ${PROTOBUF_LIB_DIR}/libprotobuf-lite.dylib)
    else()
      set(PROTOBUF_LIB_DEPS ${PROTOBUF_LIB_DIR}/libprotobuf.so ${PROTOBUF_LIB_DIR}/libprotoc.so ${PROTOBUF_LIB_DIR}/libprotobuf-lite.so)
    endif()

    link_directories(${PROTOBUF_LIB_DIR})
    include_directories(SYSTEM ${PROTOBUF_INCLUDE_DIR})
else()
    if (USE_TORCH)
        find_package(Protobuf REQUIRED ${PROTOBUF_VERSION})
    else()
        find_package(Protobuf REQUIRED)
    endif()

    set(PROTOBUF_PROTOC ${Protobuf_PROTOC_EXECUTABLE})
    set(PROTOBUF_INCLUDE_DIR ${Protobuf_INCLUDE_DIRS})
    set(PROTOBUF_LIB_DEPS protobuf::libprotobuf)
    add_custom_target(protobuf)

    # absl for protobuf 22+
    if (${Protobuf_VERSION} VERSION_GREATER_EQUAL 4.22.0)
      message(STATUS "Add Abseil dependency required by Protobuf V${Protobuf_VERSION}")
      if(NOT TARGET absl::strings)
        find_package(absl CONFIG)
      endif()
      set_target_properties(protobuf::libprotobuf PROPERTIES
        INTERFACE_LINK_LIBRARIES "absl::absl_check;absl::absl_log;absl::algorithm;absl::base;absl::bind_front;absl::bits;absl::btree;absl::cleanup;absl::cord;absl::core_headers;absl::debugging;absl::die_if_null;absl::dynamic_annotations;absl::flags;absl::flat_hash_map;absl::flat_hash_set;absl::function_ref;absl::hash;absl::layout;absl::log_initialize;absl::log_severity;absl::memory;absl::node_hash_map;absl::node_hash_set;absl::optional;absl::span;absl::status;absl::statusor;absl::strings;absl::synchronization;absl::time;absl::type_traits;absl::utility;absl::variant"
      )
    endif()
endif()

file(
    DOWNLOAD https://raw.githubusercontent.com/jolibrain/caffe/master/src/caffe/proto/caffe.proto ${CMAKE_BINARY_DIR}/caffe.proto
    EXPECTED_HASH SHA256=e82a1d51635be93e964c6debcb4cd76514fada46348fa7b30355c14850767360
)
add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/src/caffe.pb.h ${CMAKE_BINARY_DIR}/src/caffe.pb.cc
    COMMAND LD_LIBRARY_PATH=${PROTOBUF_LIB_DIR} ${PROTOBUF_PROTOC} --proto_path=${CMAKE_BINARY_DIR} --cpp_out=${CMAKE_BINARY_DIR}/src ${CMAKE_BINARY_DIR}/caffe.proto
    MAIN_DEPENDENCY ${CMAKE_BINARY_DIR}/caffe.proto
    COMMENT Generating caffe.pb.h and caffe.pb.cc
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    DEPENDS protobuf
)
add_custom_target(
    caffe_pb_h
    DEPENDS ${CMAKE_BINARY_DIR}/src/caffe.pb.h ${CMAKE_BINARY_DIR}/src/caffe.pb.cc
)

# read documentation
file(
    READ "${CMAKE_SOURCE_DIR}/docs/api_swagger.md"
    API_DOC
)

# dependency on Eigen for confusion matrix fast computation
find_package(PkgConfig)
pkg_search_module(Eigen3 REQUIRED eigen3)
set(EIGEN3_INCLUDE_DIR "/usr/include/eigen3")

set(eigen_archive_hash "50812b426b7c")

include_directories("${EIGEN3_INCLUDE_DIR}")

# spdlog, usefull for ubuntu20.04/rhel/centos that ship a bugged version (missing header, don't compile with nvcc, ...)
if (BUILD_SPDLOG)
    ExternalProject_Add(
        spdlog
        UPDATE_DISCONNECTED 1
        PREFIX spdlog
        URL https://github.com/gabime/spdlog/archive/refs/tags/v1.15.3.tar.gz
        URL_HASH SHA256=15a04e69c222eb6c01094b5c7ff8a249b36bb22788d72519646fb85feb267e67
        BUILD_IN_SOURCE true
        CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/spdlog/build -DCMAKE_POSITION_INDEPENDENT_CODE=ON 
    )
    set(SPDLOG_LIB_DEPS ${CMAKE_BINARY_DIR}/spdlog/build/lib/libspdlog.a)
    set(SPDLOG_LIB_DIR ${CMAKE_BINARY_DIR}/spdlog/build/lib)
    set(SPDLOG_INCLUDE_DIR ${CMAKE_BINARY_DIR}/spdlog/build/include)
else()
    find_package(spdlog 1.8.2 CONFIG REQUIRED)
    if (USE_PREBUILT_TORCH AND TARGET spdlog::spdlog_header_only)
      set(SPDLOG_TARGET spdlog::spdlog_header_only)
    else()
      set(SPDLOG_TARGET spdlog::spdlog)
    endif()
    set(SPDLOG_LIB_DEPS ${SPDLOG_TARGET})
    get_target_property(SPDLOG_INCLUDE_DIR ${SPDLOG_TARGET}
      INTERFACE_INCLUDE_DIRECTORIES)
endif()
include_directories(BEFORE SYSTEM ${SPDLOG_INCLUDE_DIR})

# hdf5
if (USE_HDF5)
  if (RPI3)
    set(HDF5_LIB /usr/lib/arm-linux-gnueabihf/hdf5/serial)
  else()
    set(HDF5_LIB /usr/lib/x86_64-linux-gnu/hdf5/serial)
  endif()
  set (HDF5_INCLUDE /usr/include/hdf5/serial)
  include_directories(${HDF5_INCLUDE})
endif()

# dependency on Boost
if (USE_BOOST_BACKTRACE)
    add_definitions(-DUSE_BOOST_BACKTRACE)
    find_package(Boost 1.54 REQUIRED COMPONENTS filesystem thread system iostreams stacktrace_backtrace)
    string(APPEND CMAKE_CXX_FLAGS " -DBOOST_STACKTRACE_USE_BACKTRACE -DBOOST_STACKTRACE_LINK")
else()
    find_package(Boost 1.54 REQUIRED COMPONENTS filesystem thread system iostreams)
endif()


# CUDA OR CPU options
option(USE_XGBOOST_CPU_ONLY "use XGBOOST without CUDA" ON)
option(USE_TORCH_CPU_ONLY "use TORCH without CUDA" OFF)
option(USE_DLIB_CPU_ONLY "use DLIB without CUDA" OFF)
option(USE_FAISS_CPU_ONLY "use FAISS without CUDA" OFF)

# Dlib AVX option
option(USE_DLIB_AVX "build dlib with AVX enabled" OFF)

# Deprecated flag
if (USE_XGBOOST_GPU)
  set(USE_XGBOOST_CPU_ONLY OFF)
endif()

if (USE_CPU_ONLY)
  set(USE_XGBOOST_CPU_ONLY ON)
  set(USE_DLIB_CPU_ONLY ON)
  set(USE_TORCH_CPU_ONLY ON)
  set(USE_FAISS_CPU_ONLY ON)
  set(USE_CUDNN OFF)
else()
  include(cmake/Cuda.cmake) # cuda + cudnn
endif()

# CUDA validation
if (NOT APPLE AND NOT CUDA_FOUND)
  if (USE_TENSORRT)
      message(FATAL, "USE_TENSORRT=ON needs CUDA installed")
  endif()
  if (USE_DLIB AND NOT USE_DLIB_CPU_ONLY)
      message(FATAL, "USE_DLIB=ON needs CUDA installed")
      if (USE_CUDNN)
          message(FATAL "Dlib GPU enabled without cuDNN, please set -DUSE_CUDNN=ON")
      endif()
  endif()
  if (USE_TORCH AND NOT USE_TORCH_CPU_ONLY)
      message(FATAL, "USE_TORCH=ON needs CUDA installed")
  endif()
  if (USE_FAISS AND NOT USE_FAISS_CPU_ONLY)
      message(FATAL, "USE_FAISS=ON needs CUDA installed")
  endif()
endif()

# annoy
if (USE_TORCH)
  if (USE_SIMSEARCH)
    if (USE_FAISS AND USE_ANNOY)
      message (STATUS "ANNOY selected, using ANNOY as simsearch backend")
      set(USE_FAISS OFF)
    endif()
    if (NOT USE_FAISS AND NOT USE_ANNOY)
      message (STATUS "FAISS deselected , using ANNOY as simssearch backend")
      set(USE_ANNOY ON)
    endif()
    IF (USE_ANNOY)
      message(STATUS "Fetching Annoy")
      add_definitions(-DUSE_SIMSEARCH)
      add_definitions(-DUSE_ANNOY)
      ExternalProject_Add(
        annoy
        PREFIX annoy
        GIT_REPOSITORY https://github.com/spotify/annoy.git
        CONFIGURE_COMMAND python setup.py build
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
        BUILD_IN_SOURCE 1
      )

      set(ANNOY_INCLUDE_DIR ${CMAKE_BINARY_DIR}/annoy/src/annoy/src/)
      include_directories(SYSTEM "${ANNOY_INCLUDE_DIR}")
    endif()
    if(USE_FAISS)
      if (USE_FAISS_CPU_ONLY)
	    set(CONFIGURE_OPTS "-DFAISS_ENABLE_GPU=OFF" "-DBUILD_TESTING=OFF" "-DFAISS_ENABLE_PYTHON=OFF")
      else()
        set(CONFIGURE_OPTS "-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc" "-DBUILD_TESTING=OFF" "-DFAISS_ENABLE_PYTHON=OFF")
      endif()
      message(STATUS "Fetching FAISS")
      add_definitions(-DUSE_FAISS)
      add_definitions(-DUSE_SIMSEARCH)
      ExternalProject_Add(
        faisslib
        PREFIX faiss
        SOURCE_DIR ${CMAKE_BINARY_DIR}/faiss/src/faiss
        GIT_REPOSITORY https://github.com/facebookresearch/faiss
        GIT_TAG main
        UPDATE_DISCONNECTED 1
        CONFIGURE_COMMAND cmake ${CONFIGURE_OPTS} -B build .
        BUILD_COMMAND make -C build -j${N}
        INSTALL_COMMAND ""
        BUILD_IN_SOURCE 1
      )

      set(FAISS_INCLUDE_DIR ${CMAKE_BINARY_DIR}/faiss/src/faiss)
      include_directories("${FAISS_INCLUDE_DIR}")
      set(FAISS_LIB_DIR ${CMAKE_BINARY_DIR}/faiss/src/faiss/build/faiss)
      set(FAISS_LIB_DEPS openblas faiss)
    endif()
  endif()
else()
  set(USE_SIMSEARCH OFF)
endif()


if (USE_CUDNN)
  find_package(CUDNN REQUIRED)
  set(DEEPDETECT_CUDNN_VERSION ${CUDNN_VERSION})
endif()

# CUDA dependencies
if (CUDA_FOUND)
  set(CUDA_LIB_DEPS ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_curand_LIBRARY} ${CUDA_CUDART_LIBRARY} ${CUDA_cusolver_LIBRARY})
  if (USE_CUDNN)
    set(CUDA_LIB_DEPS ${CUDA_LIB_DEPS} ${CUDNN_LIBRARY})
  endif()
  if ("${CUDA_ARCH}" STREQUAL "")
    message(STATUS "CUDA_ARCH not set, using detected set ${NVCC_ARCH_EXTRA}")

    # Convert arch to correct format
    foreach(arch ${NVCC_ARCH_EXTRA})
      string(REGEX REPLACE "(.*)(.)" "\\1.\\2" arch_with_dots "${arch}")
      string(APPEND CUDA_ARCH "${arch_with_dots};")
    endforeach()
    # remove last semi-colon if any
    string(REGEX REPLACE ".$" "" CUDA_ARCH "${CUDA_ARCH}")
  endif()
  message(STATUS "CUDA_ARCH=${CUDA_ARCH}")
else()
  set(CUDA_LIB_DEPS "")

  if (NOT APPLE)
    # XXX: On Apple platforms MPS can be used, so we'd need a different condition
    add_definitions(-DCPU_ONLY)
  endif()
endif()

find_library(LMDB_LIBRARY NAMES lmdb REQUIRED)

if (USE_DLIB)
    # Dlib

    # Currently supported release version of dlib
    set(DLIB_RELEASE_VERSION v19.24)
    message(STATUS "Using Dlib version ${DLIB_RELEASE_VERSION}")
    add_definitions(-DUSE_DLIB)
    set(DLIB_LIB_DEPS -ldlib -lopenblas)

    if (USE_DLIB_CPU_ONLY)
        ExternalProject_Add(
            dlib
            PREFIX dlib
            URL https://github.com/davisking/dlib/archive/${DLIB_RELEASE_VERSION}.tar.gz
            CONFIGURE_COMMAND cd dlib && mkdir -p build && cd build && cmake .. -DUSE_AVX_INSTRUCTIONS=${USE_DLIB_AVX} -DDLIB_NO_GUI_SUPPORT=ON -DDLIB_USE_CUDA=OFF -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/dlib/build && cmake --build . --config Release --target install
            BUILD_COMMAND ""
            INSTALL_COMMAND ""
            BUILD_IN_SOURCE 1
        )
    else()
        if (NOT USE_CUDNN)
            message(FATAL_ERROR "Dlib with CUDA must also enable cuDNN. Use -DUSE_CUDNN=ON")
        endif()
        ExternalProject_Add(
          dlib
          PREFIX dlib
          URL https://github.com/davisking/dlib/archive/${DLIB_RELEASE_VERSION}.tar.gz
          CONFIGURE_COMMAND cd dlib && mkdir -p build && cd build && cmake .. -DUSE_AVX_INSTRUCTIONS=${USE_DLIB_AVX} -DDLIB_NO_GUI_SUPPORT=ON -DDLIB_USE_CUDA=ON -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/dlib/build && cmake --build . --config Release --target install
          BUILD_COMMAND ""
          INSTALL_COMMAND ""
          BUILD_IN_SOURCE 1
        )
    endif()
    set(DLIB_INC_DIR ${CMAKE_BINARY_DIR}/dlib/build/include)
    set(DLIB_LIB_DIR ${CMAKE_BINARY_DIR}/dlib/build/lib)
    include_directories("${DLIB_INC_DIR}")
    message(STATUS "Dlib will be built")
endif()

# OpenCV
set(OPENCV_MODULES core imgproc highgui imgcodecs videoio)
if (USE_CUDA_CV)
  list(APPEND OPENCV_MODULES cudaimgproc cudaarithm cudawarping cudacodec)
endif()

if (USE_OPENCV_VERSION STREQUAL "")
  find_package(OpenCV 3 QUIET COMPONENTS ${OPENCV_MODULES})
  if (NOT OpenCV_FOUND)
    find_package(OpenCV 4 QUIET COMPONENTS ${OPENCV_MODULES})
  endif()
  if (NOT OpenCV_FOUND)
    find_package(OpenCV 2 REQUIRED COMPONENTS ${OPENCV_MODULES})
  endif()
else()
  find_package(OpenCV ${USE_OPENCV_VERSION} REQUIRED COMPONENTS ${OPENCV_MODULES})
endif()

set(OPENCV_VERSION ${OpenCV_VERSION_MAJOR})
include_directories(${OpenCV_INCLUDE_DIRS})
message(STATUS "OpenCV ${OPENCV_VERSION} (${OpenCV_VERSION}) found (${OpenCV_CONFIG_PATH})")

if (USE_CUDA_CV)
  message(STATUS "Using CUDA OpenCV")
  add_definitions(-DUSE_CUDA_CV)
endif()

# NCNN
if (USE_NCNN)
  message(STATUS "Configuring NCNN")
  add_definitions(-DUSE_NCNN)

  set(NCNN_PATCHES_PATH ${CMAKE_BINARY_DIR}/patches/ncnn)
  set(NCNN_PATCHES
    ${NCNN_PATCHES_PATH}/ncnn_inner_keep_h.patch
    ${NCNN_PATCHES_PATH}/ncnn_lstm_cont.patch
    ${NCNN_PATCHES_PATH}/ncnn_variable_h.patch
    ${NCNN_PATCHES_PATH}/ncnn_flatten_axis.patch
    ${NCNN_PATCHES_PATH}/ncnn_cont_indic.patch
    )

  set(NCNN_VERSION "20201218")
  set(NCNN_TARBALL_HASH "1dac4571168b83ee99efaba969c01e8054cd8e3d9b67ca4893f32b767ab4ec48")
  set(NCNN_BUILD_FLAGS -DCMAKE_BUILD_TYPE=Release -DNCNN_BUILD_EXAMPLES=OFF -DNCNN_BUILD_BENCHMARK=OFF -DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_TESTS=OFF)

  if (RPI3)
    ExternalProject_Add(
      ncnn
      PREFIX ncnn
      URL https://github.com/Tencent/ncnn/releases/download/${NCNN_VERSION}/ncnn-${NCNN_VERSION}-full-source.zip
      URL_HASH SHA256=${NCNN_TARBALL_HASH}
      PATCH_COMMAND ${CMAKE_SOURCE_DIR}/cmake/patches-apply.sh ${NCNN_PATCHES}
      CMAKE_ARGS -DPI3=ON -DCMAKE_TOOLCHAIN_FILE=../toolchains/pi3.toolchain.cmake ${NCNN_BUILD_FLAGS}
      INSTALL_COMMAND ""
    )
  else ()
    ExternalProject_Add(
      ncnn
      PREFIX ncnn
      URL https://github.com/Tencent/ncnn/releases/download/${NCNN_VERSION}/ncnn-${NCNN_VERSION}-full-source.zip
      URL_HASH SHA256=${NCNN_TARBALL_HASH}
      PATCH_COMMAND  ${CMAKE_SOURCE_DIR}/cmake/patches-apply.sh ${NCNN_PATCHES}
      CMAKE_ARGS ${NCNN_BUILD_FLAGS}
      INSTALL_COMMAND ""
      )
  endif()

  set(NCNN_LIB_DEPS ${CMAKE_BINARY_DIR}/ncnn/src/ncnn-build/src/libncnn.a)
  set(NCNN_INC_DIR ${CMAKE_BINARY_DIR}/ncnn/src/ncnn/src ${CMAKE_BINARY_DIR}/ncnn/src/ncnn-build/src)
  set(NCNN_LIB_DIR ${CMAKE_BINARY_DIR}/ncnn/src/ncnn-build/src)
endif()

if (USE_PYTORCH_WORKER)
  message(STATUS "Configuring managed Python PyTorch worker backend")
  add_definitions(-DUSE_PYTORCH_WORKER)
endif()

# Torch
if (USE_TORCH)
  if (!APPLE)
    set(CMAKE_EXE_LINKER_FLAGS "-Wl,--no-as-needed")
  endif()

  set(PYTORCH_PATCHES_PATH ${CMAKE_SOURCE_DIR}/patches/pytorch)

  set(PYTORCH_PATCHES
    ${PYTORCH_PATCHES_PATH}/pytorch_212_new_logger.patch
    ${PYTORCH_PATCHES_PATH}/pytorch_212_use_new_logger.patch
    ${PYTORCH_PATCHES_PATH}/pytorch_212_cloneable.patch
    ${PYTORCH_PATCHES_PATH}/pytorch_212_list_inl_typename.patch
  )
  if (APPLE)
    list(APPEND PYTORCH_PATCHES ${PYTORCH_PATCHES_PATH}/pytorch_113_apple_includes.patch)
  endif()

  message(STATUS "Configuring libtorch")
  add_definitions(-DUSE_TORCH)
  add_definitions(-DFMT_HEADER_ONLY)

  if (USE_PREBUILT_TORCH)
    add_definitions(-DDEEPDETECT_PREBUILT_TORCH)

    if (TORCH_LOCATION)
      list(PREPEND CMAKE_PREFIX_PATH "${TORCH_LOCATION}")
    endif()

    find_package(Torch 2.12.1 EXACT CONFIG REQUIRED)
    get_filename_component(TORCH_LOCATION
      "${Torch_DIR}/../../.." ABSOLUTE)

    if (TARGET torch_cuda)
      set(PYTORCH_USE_CUDA 1)
      set(DEEPDETECT_TORCH_MODE "cuda")
    else()
      set(PYTORCH_USE_CUDA 0)
      set(DEEPDETECT_TORCH_MODE "cpu")
    endif()

    if (USE_TORCH_CPU_ONLY AND PYTORCH_USE_CUDA)
      message(FATAL_ERROR
        "USE_TORCH_CPU_ONLY=ON requires a CPU-only LibTorch package, but "
        "${TORCH_LOCATION} contains torch_cuda.")
    elseif (NOT USE_TORCH_CPU_ONLY AND NOT PYTORCH_USE_CUDA)
      message(FATAL_ERROR
        "GPU Torch support requires a CUDA LibTorch package. Select a CUDA "
        "package or configure with USE_TORCH_CPU_ONLY=ON.")
    endif()

    if (PYTORCH_USE_CUDA)
      if (NOT CUDA_VERSION MATCHES "^(12\\.6|13\\.)")
        message(FATAL_ERROR
          "The supported prebuilt CUDA LibTorch packages require CUDA 12.6 "
          "or CUDA 13; found CUDA ${CUDA_VERSION}.")
      endif()

      set(TORCH_NVIDIA_RUNTIME_HINTS)
      foreach(TORCH_NVIDIA_RUNTIME_ROOT
          "$ENV{VIRTUAL_ENV}"
          "$ENV{CONDA_PREFIX}"
          "$ENV{HOME}/venv")
        if (TORCH_NVIDIA_RUNTIME_ROOT)
          file(GLOB TORCH_NVIDIA_RUNTIME_ROOT_HINTS
            "${TORCH_NVIDIA_RUNTIME_ROOT}/lib/python*/site-packages/nvidia/*/lib")
          list(APPEND TORCH_NVIDIA_RUNTIME_HINTS
            ${TORCH_NVIDIA_RUNTIME_ROOT_HINTS})
        endif()
      endforeach()
      file(GLOB TORCH_NVIDIA_SYSTEM_HINTS
        "/usr/local/lib/python*/site-packages/nvidia/*/lib"
        "/usr/local/lib/python*/dist-packages/nvidia/*/lib"
        "/usr/lib/python*/site-packages/nvidia/*/lib"
        "/usr/lib/python*/dist-packages/nvidia/*/lib")
      list(APPEND TORCH_NVIDIA_RUNTIME_HINTS ${TORCH_NVIDIA_SYSTEM_HINTS})
      if (TORCH_NVIDIA_RUNTIME_HINTS)
        list(REMOVE_DUPLICATES TORCH_NVIDIA_RUNTIME_HINTS)
      endif()

      if (TORCH_CUDNN_LIBRARY)
        get_filename_component(TORCH_CUDNN_LIBRARY_CACHED_REAL
          "${TORCH_CUDNN_LIBRARY}" REALPATH)
        if (NOT TORCH_CUDNN_LIBRARY_CACHED_REAL MATCHES "libcudnn\\.so\\.9")
          unset(TORCH_CUDNN_LIBRARY CACHE)
          unset(TORCH_CUDNN_LIBRARY)
        endif()
      endif()
      find_library(TORCH_CUDNN_LIBRARY
        NAMES libcudnn.so.9
        HINTS ${TORCH_NVIDIA_RUNTIME_HINTS}
        NO_DEFAULT_PATH)
      if (NOT TORCH_CUDNN_LIBRARY)
        find_library(TORCH_CUDNN_LIBRARY
          NAMES libcudnn.so.9
          REQUIRED)
      endif()
      find_library(TORCH_NCCL_LIBRARY
        NAMES libnccl.so.2 nccl
        HINTS ${TORCH_NVIDIA_RUNTIME_HINTS}
        NO_DEFAULT_PATH)
      if (NOT TORCH_NCCL_LIBRARY)
        find_library(TORCH_NCCL_LIBRARY
          NAMES libnccl.so.2 nccl
          REQUIRED)
      endif()
      find_library(TORCH_CUSPARSELT_LIBRARY
        NAMES libcusparseLt.so.0 cusparseLt
        HINTS ${TORCH_NVIDIA_RUNTIME_HINTS}
        NO_DEFAULT_PATH)
      if (NOT TORCH_CUSPARSELT_LIBRARY)
        find_library(TORCH_CUSPARSELT_LIBRARY
          NAMES libcusparseLt.so.0 cusparseLt
          REQUIRED)
      endif()
      find_library(TORCH_NVSHMEM_LIBRARY
        NAMES libnvshmem_host.so.3 nvshmem_host
        HINTS ${TORCH_NVIDIA_RUNTIME_HINTS}
        NO_DEFAULT_PATH)
      if (NOT TORCH_NVSHMEM_LIBRARY)
        find_library(TORCH_NVSHMEM_LIBRARY
          NAMES libnvshmem_host.so.3 nvshmem_host
          REQUIRED)
      endif()
      foreach(TORCH_RUNTIME_LIBRARY
          TORCH_CUDNN_LIBRARY
          TORCH_NCCL_LIBRARY
          TORCH_CUSPARSELT_LIBRARY
          TORCH_NVSHMEM_LIBRARY)
        get_filename_component(TORCH_RUNTIME_LIBRARY_REAL
          "${${TORCH_RUNTIME_LIBRARY}}" REALPATH)
        if (TORCH_RUNTIME_LIBRARY STREQUAL "TORCH_CUDNN_LIBRARY"
            AND NOT TORCH_RUNTIME_LIBRARY_REAL MATCHES "libcudnn\\.so\\.9")
          message(FATAL_ERROR
            "Prebuilt LibTorch requires cuDNN 9, but "
            "${TORCH_RUNTIME_LIBRARY_REAL} was selected.")
        endif()
      endforeach()
      get_filename_component(TORCH_CUDNN_LIBRARY_REAL
        "${TORCH_CUDNN_LIBRARY}" REALPATH)
      get_filename_component(TORCH_CUDNN_LIBRARY_NAME
        "${TORCH_CUDNN_LIBRARY_REAL}" NAME)
      string(REGEX MATCH "libcudnn\\.so\\.([0-9]+(\\.[0-9]+)*)"
        TORCH_CUDNN_VERSION_MATCH "${TORCH_CUDNN_LIBRARY_NAME}")
      set(DEEPDETECT_CUDNN_VERSION "${CMAKE_MATCH_1}")
      unset(TORCH_NCCL_HAS_COMM_RESUME CACHE)
      unset(TORCH_NCCL_HAS_ALLTOALL CACHE)
      check_library_exists("${TORCH_NCCL_LIBRARY}" ncclCommResume ""
        TORCH_NCCL_HAS_COMM_RESUME)
      check_library_exists("${TORCH_NCCL_LIBRARY}" ncclAlltoAll ""
        TORCH_NCCL_HAS_ALLTOALL)
      if (NOT TORCH_NCCL_HAS_ALLTOALL
          OR (CUDA_VERSION VERSION_GREATER_EQUAL 13
              AND NOT TORCH_NCCL_HAS_COMM_RESUME))
        message(FATAL_ERROR
          "The selected NCCL library is incompatible with LibTorch "
          "${Torch_VERSION}: ${TORCH_NCCL_LIBRARY}. Install the NCCL runtime "
          "matching the official LibTorch package and include its directory "
          "in CMAKE_LIBRARY_PATH.")
      endif()

      # The legacy CUDA probe may have selected a different cuDNN major.
      # LibTorch and DeepDetect must load only the validated cuDNN 9 runtime.
      if (CUDNN_LIBRARY)
        list(REMOVE_ITEM CUDA_LIB_DEPS "${CUDNN_LIBRARY}")
      endif()
      set(TORCH_RUNTIME_LIBS
        ${TORCH_CUDNN_LIBRARY}
        ${TORCH_NCCL_LIBRARY}
        ${TORCH_CUSPARSELT_LIBRARY}
        ${TORCH_NVSHMEM_LIBRARY})
    endif()

    set(TORCH_LIB_DEPS ${TORCH_LIBRARIES} ${TORCH_RUNTIME_LIBS} -llmdb)
    set(TORCH_INC_DIR ${TORCH_INCLUDE_DIRS})
    set(TORCH_LIB_DIR ${TORCH_LOCATION}/lib)
    set(DEEPDETECT_TORCH_VERSION ${Torch_VERSION})
  else()
    set(PYTORCH_COMMIT v2.12.1)
    set(PYTORCH_COMPLETE ${CMAKE_BINARY_DIR}/CMakeFiles/pytorch-complete)

    if (APPLE)
      if(USE_TORCH_CPU_ONLY)
        set(PYTORCH_USE_MPS 0)
      else()
        set(PYTORCH_USE_MPS 1)
        add_definitions("-DUSE_MPS")
      endif()

      message(STATUS "PROTOBUF_PROTOC_EXECUTABLE=${PROTOBUF_PROTOC} PROTOBUF_INCLUDE_DIRS=${PROTOBUF_INCLUDE_DIR} PROTOBUF_LIBRARIES=${PROTOBUF_LIB_DEPS}")
      ExternalProject_Add(
        pytorch
        PREFIX pytorch
        GIT_REPOSITORY https://github.com/pytorch/pytorch.git
        GIT_TAG ${PYTORCH_COMMIT}
        GIT_CONFIG advice.detachedHead=false
        UPDATE_DISCONNECTED 1
        PATCH_COMMAND "" test -f ${PYTORCH_COMPLETE} && echo Skipping || git apply ${PYTORCH_PATCHES} && echo Applying ${PYTORCH_PATCHES}
CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        COMMAND test -f ${PYTORCH_COMPLETE} && echo Skipping || ${CMAKE_COMMAND} -E env PATH=${PROTOBUF_LIB_DIR}:${PROTOBUF_INCLUDE_DIR}:$ENV{PATH} BUILD_CUSTOM_PROTOBUF=0 BUILD_TEST=0 USE_MPS=${PYTORCH_USE_MPS} USE_DDLOG=1 USE_TENSORRT=0 "CMAKE_CXX_FLAGS=-isystem ${SPDLOG_INCLUDE_DIR} -isystem ${PROTOBUF_INCLUDE_DIR}" "CMAKE_CUDA_FLAGS=-isystem ${SPDLOG_INCLUDE_DIR} -isystem ${PROTOBUF_INCLUDE_DIR}" CMAKE_PREFIX_PATH="${PROTOBUF_LIB_DIR}/cmake" MAX_JOBS=8 python3 ../pytorch/tools/build_libtorch.py
        INSTALL_COMMAND ""
        DEPENDS spdlog protobuf
      )
      set(TORCH_BINARY_LOCATION ${CMAKE_BINARY_DIR}/pytorch/src/pytorch-build/build)
    else()
      if(USE_TORCH_CPU_ONLY)
        set(PYTORCH_USE_CUDA 0)
      else()
        set(PYTORCH_USE_CUDA 1)
      endif()

      file(WRITE ${CMAKE_BINARY_DIR}/build_pytorch.sh "set -x\n${CMAKE_COMMAND} -E env PYTHONPATH=$ENV{PYTHONPATH}:${CMAKE_BINARY_DIR}/pytorch/src/pytorch PATH=${PROTOBUF_LIB_DIR}:$ENV{PATH} BUILD_CUSTOM_PROTOBUF=0 GLIBCXX_USE_CXX11_ABI=1 BUILD_TEST=0 USE_CUDA=${PYTORCH_USE_CUDA}  BUILD_CAFFE2=1 BUILD_CAFFE2_OPS=1 BUILD_CAFFE2_MOBILE=0 USE_DDLOG=1 USE_TENSORRT=0 CAFFE2_LINK_LOCAL_PROTOBUF=0 \"CMAKE_CXX_FLAGS=-isystem ${SPDLOG_INCLUDE_DIR} -isystem ${PROTOBUF_INCLUDE_DIR} -isystem /usr/lib/x86_64-linux-gnu/openmpi/include\" \"CMAKE_CUDA_FLAGS=-isystem ${SPDLOG_INCLUDE_DIR} -isystem ${PROTOBUF_INCLUDE_DIR} -isystem /usr/lib/x86_64-linux-gnu/openmpi/include\" TORCH_CUDA_ARCH_LIST=\"${CUDA_ARCH}\" \"CMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc\" CMAKE_PREFIX_PATH=${PROTOBUF_LIB_DIR}/cmake MAX_JOBS=4 python3 ../pytorch/tools/build_libtorch.py")
      ExternalProject_Add(
        pytorch
        PREFIX pytorch
        GIT_REPOSITORY https://github.com/pytorch/pytorch.git
        GIT_TAG ${PYTORCH_COMMIT}
        GIT_CONFIG advice.detachedHead=false
        UPDATE_DISCONNECTED 1
        PATCH_COMMAND "" test -f ${PYTORCH_COMPLETE} && echo Skipping || git apply ${PYTORCH_PATCHES} && echo Applying ${PYTORCH_PATCHES}
CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        COMMAND test -f ${PYTORCH_COMPLETE} && echo Skipping || sh ${CMAKE_BINARY_DIR}/build_pytorch.sh
        INSTALL_COMMAND ""
        DEPENDS spdlog protobuf
      )
    endif()
    set(TORCH_LOCATION ${CMAKE_BINARY_DIR}/pytorch/src/pytorch/torch)

    if (APPLE)
      set(TORCH_LIB_DEPS ${TORCH_BINARY_LOCATION}/lib/libtorch.dylib ${TORCH_BINARY_LOCATION}/lib/libtorch_cpu.dylib ${TORCH_BINARY_LOCATION}/lib/libc10.dylib ${TORCH_BINARY_LOCATION}/lib/libtorch_global_deps.dylib -llmdb)
    else()
      set(TORCH_LIB_DEPS ${TORCH_LOCATION}/lib/libtorch.so ${TORCH_LOCATION}/lib/libtorch_cpu.so ${TORCH_LOCATION}/lib/libc10.so -llmdb)

      if (USE_TORCH_CPU_ONLY)
        list(APPEND TORCH_LIB_DEPS)
        set(DEEPDETECT_TORCH_MODE "cpu")
      else()
        list(APPEND TORCH_LIB_DEPS ${TORCH_LOCATION}/lib/libc10_cuda.so ${TORCH_LOCATION}/lib/libtorch_cuda.so)
        set(DEEPDETECT_TORCH_MODE "cuda")
      endif()
    endif()

    set(TORCH_INC_DIR ${TORCH_LOCATION}/include/ ${TORCH_LOCATION}/include/torch/csrc/api/include/ ${CMAKE_BINARY_DIR}/pytorch/src/pytorch/torch/include/torch/csrc/api/include ${TORCH_LOCATION}/.. ${CMAKE_BINARY_DIR}/src)
    set(TORCH_LIB_DIR ${TORCH_LOCATION}/lib/)
    set(DEEPDETECT_TORCH_VERSION ${PYTORCH_COMMIT})
  endif()

  # TORCH VISION
  message(STATUS "Configuring Pytorch Vision")
  set(PYTORCH_VISION_COMMIT "v0.27.1")
  set(PYTORCH_VISION_PATCHES_PATH ${CMAKE_BINARY_DIR}/patches/pytorch/vision)
  # set(PYTORCH_VISION_PATCHES
  #   ${PYTORCH_VISION_PATCHES_PATH}/xxx.patch # add patches here
  # )

  set(PYTORCH_VISION_COMPLETE ${CMAKE_BINARY_DIR}/CMakeFiles/pytorch_vision-complete)
  set(TORCHVISION_LOCATION "${CMAKE_BINARY_DIR}/pytorch_vision/src/pytorch_vision-install")

  if (TORCHVISION_SOURCE_DIR)
    set(PYTORCH_VISION_SOURCE_ARGS
      SOURCE_DIR "${TORCHVISION_SOURCE_DIR}"
      DOWNLOAD_COMMAND ""
      UPDATE_COMMAND "")
  else()
    set(PYTORCH_VISION_SOURCE_ARGS
      GIT_REPOSITORY https://github.com/pytorch/vision.git
      GIT_TAG ${PYTORCH_VISION_COMMIT}
      GIT_CONFIG advice.detachedHead=false
      UPDATE_DISCONNECTED ${TORCHVISION_UPDATE_DISCONNECTED})
  endif()

  if (USE_PREBUILT_TORCH)
    set(PYTORCH_VISION_DEPENDS protobuf)
    set(PYTORCH_VISION_EXTRA_FLAGS "")
  else()
    set(PYTORCH_VISION_DEPENDS protobuf pytorch)
    set(PYTORCH_VISION_EXTRA_FLAGS
      "-isystem ${CMAKE_BINARY_DIR}/pytorch/src/pytorch/")
    if (SPDLOG_INCLUDE_DIR)
      string(APPEND PYTORCH_VISION_EXTRA_FLAGS
        " -isystem ${SPDLOG_INCLUDE_DIR}")
    endif()
    if (PROTOBUF_INCLUDE_DIR)
      string(APPEND PYTORCH_VISION_EXTRA_FLAGS
        " -isystem ${PROTOBUF_INCLUDE_DIR}")
    endif()
  endif()
  string(REPLACE ";" "|" PYTORCH_VISION_CUDA_ARCH "${CUDA_ARCH}")

  ExternalProject_Add(
    pytorch_vision
    PREFIX pytorch_vision
    LIST_SEPARATOR |
    ${PYTORCH_VISION_SOURCE_ARGS}
    PATCH_COMMAND "" #test -f ${PYTORCH_VISION_COMPLETE} && echo Skipping || git apply ${PYTORCH_VISION_PATCHES} && echo Applying ${PYTORCH_VISION_PATCHES}
    CMAKE_ARGS
      ${PYTORCH_VISION_FLAGS}
      -DWITH_CUDA=${PYTORCH_USE_CUDA}
      -DCMAKE_PREFIX_PATH=${TORCH_LOCATION}
      -DCMAKE_INSTALL_PREFIX=${TORCHVISION_LOCATION}
      -DCMAKE_INCLUDE_PATH=${PROTOBUF_INCLUDE_DIR}
      -DCMAKE_LIBRARY_PATH=${PROTOBUF_LIB_DIR}
      "-DCMAKE_CXX_FLAGS=${PYTORCH_VISION_EXTRA_FLAGS}"
      "-DCMAKE_CUDA_FLAGS=${PYTORCH_VISION_EXTRA_FLAGS}"
      "-DCMAKE_INSTALL_RPATH=$ORIGIN|$ORIGIN/../torch/lib"
      -DCMAKE_CXX_STANDARD=${DEEPDETECT_CXX_STANDARD}
      -DCMAKE_CUDA_STANDARD=${DEEPDETECT_CXX_STANDARD}
      "-DTORCH_CUDA_ARCH_LIST=${PYTORCH_VISION_CUDA_ARCH}"
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
    BUILD_COMMAND
      ${CMAKE_COMMAND} --build <BINARY_DIR> --target install --parallel ${N}
    INSTALL_COMMAND ""
    DEPENDS ${PYTORCH_VISION_DEPENDS}
  )

  list(APPEND TORCH_INC_DIR "${TORCHVISION_LOCATION}/include/")
  list(APPEND TORCH_LIB_DIR "${TORCHVISION_LOCATION}/lib/")
  list(APPEND TORCH_LIB_DEPS torchvision)
else()
    ExternalProject_Add(
      fmt
      PREFIX fmt
      UPDATE_DISCONNECTED 1
      URL https://github.com/fmtlib/fmt/archive/refs/tags/11.2.0.tar.gz
      URL_HASH SHA256=bc23066d87ab3168f27cef3e97d545fa63314f5c79df5ea444d41d56f962c6af
      CMAKE_ARGS -DFMT_TEST=OFF -DFMT_DOC=OFF -DFMT_INSTALL=OFF
      INSTALL_COMMAND ""
    )
  set(FMT_INC_DIR ${CMAKE_BINARY_DIR}/fmt/src/fmt/include)
  set(FMT_LIB_DIR ${CMAKE_BINARY_DIR}/fmt/src/fmt/lib)
endif()

# XGBoost
if (USE_XGBOOST)
  message(STATUS "Configuring XGBoost")
  add_definitions(-DUSE_XGBOOST)
  set(XGBOOST_LIB_DEPS -Wl,--whole-archive ${CMAKE_BINARY_DIR}/xgboost/src/xgboost-build/libxgboost.a -Wl,--no-whole-archive ${CMAKE_BINARY_DIR}/xgboost/src/xgboost-build/dmlc-core/libdmlc.a ${CMAKE_BINARY_DIR}/xgboost/src/xgboost-build/librabit.a)

  if (NOT USE_XGBOOST_CPU_ONLY)
    add_definitions(-DUSE_XGBOOST_GPU)
    list(APPEND XGBOOST_LIB_DEPS ${CMAKE_BINARY_DIR}/xgboost/src/xgboost/build/CMakeFiles/gpuxgboost.dir/plugin/updater_gpu/src/gpuxgboost_generated_updater_gpu.cu.o ${CMAKE_BINARY_DIR}/xgboost/src/xgboost/build/nccl/libnccl.a)
    ExternalProject_Add(
      xgboost_cub
      PREFIX xgboost_cub
      URL https://github.com/NVlabs/cub/archive/1.6.4.zip
      URL_HASH SHA256=4198e9c447a1e2a963b9e0e4d861df48baa47fb02e5e4fc507d1834afc99185a
      INSTALL_COMMAND ""
      BUILD_COMMAND ""
      CONFIGURE_COMMAND ""
    )
    ExternalProject_Add(
      xgboost
      PREFIX xgboost
      GIT_REPOSITORY https://github.com/dmlc/xgboost.git
      GIT_TAG 84d992babcc370bff6cf4fe89985f072b0b91a64
      GIT_CONFIG advice.detachedHead=false
      PATCH_COMMAND
        git reset --hard &&
        patch -p1 < ${CMAKE_SOURCE_DIR}/patches/xgboost/static-lib.patch &&
        cd dmlc-core && git checkout 13d5acb8ba7e79550bbf2f730f1a3944ff0fa68b
      CMAKE_ARGS
        -DPLUGIN_UPDATER_GPU=ON
        -DCUB_DIRECTORY=${CMAKE_BINARY_DIR}/xgboost_cub/src/xgboost_cub
        -DCUDA_NVCC_FLAGS="-Xcompiler -fPIC --expt-extended-lambda $CUDA_ARCH_FLAGS"
      INSTALL_COMMAND ""
      DEPENDS xgboost_cub
    )
  else()
    ExternalProject_Add(
      xgboost
      PREFIX xgboost
      GIT_REPOSITORY https://github.com/dmlc/xgboost.git
      GIT_TAG 84d992babcc370bff6cf4fe89985f072b0b91a64
      GIT_CONFIG advice.detachedHead=false
      UPDATE_DISCONNECTED 1
      PATCH_COMMAND
        git reset --hard &&
        patch -p1 < ${CMAKE_SOURCE_DIR}/patches/xgboost/static-lib.patch &&
        cd dmlc-core && git checkout 13d5acb8ba7e79550bbf2f730f1a3944ff0fa68b
      INSTALL_COMMAND ""
    )
  endif()
  set(XGBOOST_INC_DIR ${CMAKE_BINARY_DIR}/xgboost/src/xgboost/include ${CMAKE_BINARY_DIR}/xgboost/src/xgboost/src ${CMAKE_BINARY_DIR}/xgboost/src/xgboost/rabit/include ${CMAKE_BINARY_DIR}/xgboost/src/xgboost/dmlc-core/include ${CMAKE_BINARY_DIR}/xgboost/src/xgboost/dmlc-core/src/)
  set(XGBOOST_LIB_DIR ${CMAKE_BINARY_DIR}/xgboost/src/xgboost-build/lib ${CMAKE_BINARY_DIR}/xgboost/src/xgboost-build/ ${CMAKE_BINARY_DIR}/xgboost/src/xgboost-build/dmlc-core)
endif()

if (USE_TSNE)
  message(STATUS "Configuring T-SNE")
  add_definitions(-DUSE_TSNE)
  set(TSNE_LIB_DEPS -ltsne_multicore -lglog)
  ExternalProject_Add(
      Multicore-TSNE
      PREFIX Multicore-TSNE
      UPDATE_DISCONNECTED 1
      GIT_REPOSITORY https://github.com/beniz/Multicore-TSNE.git
      SOURCE_SUBDIR multicore_tsne
      CMAKE_ARGS -Wno-dev
      INSTALL_COMMAND ""
   )
  set(TSNE_INC_DIR ${CMAKE_BINARY_DIR}/Multicore-TSNE/src/Multicore-TSNE/multicore_tsne)
  set(TSNE_LIB_DIR ${CMAKE_BINARY_DIR}/Multicore-TSNE/src/Multicore-TSNE-build)
    
endif()

if (USE_TENSORRT)
  add_definitions(-DUSE_TENSORRT)
  if (EXISTS ${TENSORRT_DIR}/libprotobuf.a)
    message(ERROR "there is a protobuf in ${TENSORRT_DIR}, it is very likely to cause link problem, you should remove it, we use system or internal ones")
  endif()


  if (JETSON)
    set(TRTTESTDIR /usr/lib/aarch64-linux-gnu)
    set(TENSORRT_LIB_DIR /usr/lib/aarch64-linux-gnu)
    set(TENSORRT_INC_DIR /usr/include/aarch64-linux-gnu)
  elseif(DEFINED TENSORRT_DIR)
    set(TRTTESTDIR ${TENSORRT_DIR}/lib)
    set(TENSORRT_LIB_DIR  ${TENSORRT_DIR}/lib )
    set(TENSORRT_INC_DIR ${TENSORRT_DIR}/include)
  elseif (DEFINED TENSORRT_LIB_DIR AND DEFINED TENSORRT_INC_DIR)
    set(TRTTESTDIR ${TENSORRT_LIB_DIR})
  else()
    set(TRTTESTDIR /usr/lib/x86_64-linux-gnu)
    set(TENSORRT_LIB_DIR  /usr/lib/x86_64-linux-gnu)
    set(TENSORRT_INC_DIR /usr/include/x86_64-linux-gnu)
  endif()

  find_library(TENSORRT_NVINFER_LIB
    NAMES nvinfer
    HINTS ${TENSORRT_LIB_DIR} ${TRTTESTDIR}
    NO_DEFAULT_PATH)
  if (NOT TENSORRT_NVINFER_LIB)
    file(GLOB TENSORRT_NVINFER_LIB_CANDIDATES
      "${TRTTESTDIR}/libnvinfer.so*"
      "${TENSORRT_LIB_DIR}/libnvinfer.so*")
    list(FILTER TENSORRT_NVINFER_LIB_CANDIDATES EXCLUDE REGEX ".*libnvinfer_(plugin|vc_plugin|lean|dispatch|safe).*")
    list(SORT TENSORRT_NVINFER_LIB_CANDIDATES)
    list(REVERSE TENSORRT_NVINFER_LIB_CANDIDATES)
    list(LENGTH TENSORRT_NVINFER_LIB_CANDIDATES TENSORRT_NVINFER_LIB_CANDIDATE_COUNT)
    if (TENSORRT_NVINFER_LIB_CANDIDATE_COUNT GREATER 0)
      list(GET TENSORRT_NVINFER_LIB_CANDIDATES 0 TENSORRT_NVINFER_LIB)
    endif()
  endif()

  if (NOT TENSORRT_NVINFER_LIB)
    message(FATAL_ERROR "Could not find TensorRT libnvinfer.so in ${TRTTESTDIR}, please provide tensorRT location as TENSORRT_DIR or (TENSORRT_LIB_DIR _and_ TENSORRT_INC_DIR)")
  endif()
  message(STATUS "Found TensorRT library: ${TENSORRT_NVINFER_LIB}")

    set(TENSORRT_LIB_DIR ${CMAKE_BINARY_DIR}/tensorrt-oss/bin/ ${TENSORRT_LIB_DIR})
	set(TENSORRT_LIBS ${TENSORRT_NVINFER_LIB} nvinfer_plugin nvonnxparser )

    if (TENSORRT_VERSION STREQUAL "null")
        unset(TENSORRT_VERSION CACHE)
    endif()

    if (NOT TENSORRT_VERSION)
        find_file(TENSORRT_VERSION_HEADER
          NAMES NvInferVersion.h
          HINTS ${TENSORRT_INC_DIR} ${TENSORRT_INC_DIR}/include)

        if (TENSORRT_VERSION_HEADER)
            file(STRINGS "${TENSORRT_VERSION_HEADER}" TENSORRT_MAJOR_LINE
              REGEX "^#define NV_TENSORRT_MAJOR +[0-9]+")
            file(STRINGS "${TENSORRT_VERSION_HEADER}" TENSORRT_MINOR_LINE
              REGEX "^#define NV_TENSORRT_MINOR +[0-9]+")
            string(REGEX REPLACE ".*NV_TENSORRT_MAJOR +([0-9]+).*" "\\1"
              TENSORRT_MAJOR "${TENSORRT_MAJOR_LINE}")
            string(REGEX REPLACE ".*NV_TENSORRT_MINOR +([0-9]+).*" "\\1"
              TENSORRT_MINOR "${TENSORRT_MINOR_LINE}")
        endif()

        if (TENSORRT_MAJOR MATCHES "^[0-9]+$" AND TENSORRT_MINOR MATCHES "^[0-9]+$")
            set(TENSORRT_VERSION "v${TENSORRT_MAJOR}.${TENSORRT_MINOR}")
            message(STATUS "Found TensorRT version ${TENSORRT_MAJOR}.${TENSORRT_MINOR} from ${TENSORRT_VERSION_HEADER}")
        else()
            get_filename_component(TENSORRT_NVINFER_REALPATH
              "${TENSORRT_NVINFER_LIB}" REALPATH)
            if (TENSORRT_NVINFER_REALPATH MATCHES "libnvinfer[.]so[.]([0-9]+)[.]([0-9]+)")
                set(TENSORRT_VERSION "v${CMAKE_MATCH_1}.${CMAKE_MATCH_2}")
                message(STATUS "Found TensorRT version ${CMAKE_MATCH_1}.${CMAKE_MATCH_2} from ${TENSORRT_NVINFER_REALPATH}")
            else()
                message(FATAL_ERROR
                  "Could not determine TensorRT version from ${TENSORRT_NVINFER_LIB}; set TENSORRT_VERSION explicitly")
            endif()
        endif()
    endif()
    add_definitions(-DTENSORRT_VERSION=${TENSORRT_VERSION})

    set(TRT_FLAGS
      -DTRT_OUT_DIR=${CMAKE_BINARY_DIR}/tensorrt-oss/bin
      -Dnvinfer_LIB_PATH=${TENSORRT_NVINFER_LIB}
      -DCMAKE_CXX_STANDARD=17
      -DCMAKE_CXX_STANDARD_REQUIRED=ON
      -DCMAKE_CXX_EXTENSIONS=OFF
      -DCMAKE_CUDA_STANDARD=17
      -DCMAKE_CUDA_STANDARD_REQUIRED=ON
      "-DCMAKE_CXX_FLAGS=-std=c++17"
      "-DCMAKE_CUDA_FLAGS=--std=c++17"
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
      -DBUILD_SAMPLES=OFF
      -DPY_VERSION=3
    )

    if (JETSON)
      list(APPEND TRT_FLAGS -DCMAKE_C_COMPILER=/usr/bin/cc)
    endif()

    # Tensorrt uses its own protobuf version (3.0.0) and this cannot be overrided
    # At best we can set the version it will internally build
    set(TENSORRT_COMPLETE ${CMAKE_BINARY_DIR}/CMakeFiles/tensorrt-oss-complete)
    ExternalProject_Add(
      tensorrt-oss
      PREFIX tensorrt-oss
      GIT_REPOSITORY https://github.com/NVIDIA/TensorRT.git
      GIT_TAG ${TENSORRT_VERSION}
      GIT_CONFIG advice.detachedHead=false
      UPDATE_DISCONNECTED 1
      CONFIGURE_COMMAND ""
      BUILD_COMMAND sh ${CMAKE_SOURCE_DIR}/cmake/build_external.sh TENSORRT ${CMAKE_BINARY_DIR}/tensorrt-oss ${TENSORRT_COMPLETE} src/tensorrt-oss src/tensorrt-oss-build ${N} all ${CMAKE_COMMAND} ${TRT_FLAGS}
      INSTALL_COMMAND ""
    )

    # Use our tensorrt-oss header versions first
    list(INSERT TENSORRT_INC_DIR 0 ${CMAKE_BINARY_DIR}/tensorrt-oss/src/tensorrt-oss/include)

    # onnx.pb.h
    set(ONNX_PROTO ${CMAKE_BINARY_DIR}/tensorrt-oss/src/tensorrt-oss/parsers/onnx/third_party/onnx/onnx/onnx.proto)
    add_custom_command(
        OUTPUT ${CMAKE_BINARY_DIR}/src/onnx.pb.h ${CMAKE_BINARY_DIR}/src/onnx.pb.cc
        COMMAND LD_LIBRARY_PATH=${PROTOBUF_LIB_DIR} ${PROTOBUF_PROTOC} --proto_path=${CMAKE_BINARY_DIR}/tensorrt-oss/src/tensorrt-oss/parsers/onnx/third_party/onnx/onnx/ --cpp_out=${CMAKE_BINARY_DIR}/src ${ONNX_PROTO}
        COMMENT Generating onnx.pb.h and onnx.pb.cc
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        DEPENDS protobuf tensorrt-oss
    )
    add_custom_target(
        onnx_pb_h
        DEPENDS ${CMAKE_BINARY_DIR}/src/onnx.pb.h ${CMAKE_BINARY_DIR}/src/onnx.pb.cc
    )
endif()

set(OATPP_VERSION "1.3.0")
# 1.3 + allow to change api path
set(OATPP_SWAGGER_VERSION "ed5251c580e2e98beb50d818bcea8ddc91419d8c")
# 1.3 + memory leak patch
set(OATPP_ZLIB_VERSION "e7318068ae1b24b7d690f374ffbcd9c8f691336b")
ExternalProject_Add(
    oatpp
    PREFIX oatpp
    UPDATE_DISCONNECTED 1
    URL https://github.com/oatpp/oatpp/archive/${OATPP_VERSION}.tar.gz
    URL_HASH SHA256=e1f80fa8fd7a74da6737e7fee1a4db68b4d7085a3f40e7d550752d6ff5714583
    BUILD_IN_SOURCE true
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DBUILD_SHARED_LIBS=OFF -DOATPP_INSTALL=OFF -DOATPP_BUILD_TESTS=OFF
    INSTALL_COMMAND ""
)
ExternalProject_Add(
    oatpp-zlib
    PREFIX oatpp-zlib
    URL https://github.com/oatpp/oatpp-zlib/archive/${OATPP_ZLIB_VERSION}.tar.gz
    URL_HASH SHA256=17d85ace758ac3b924802d10473a8ed4cf6ee2ffa125022d5a64debf564cf3c3
    CMAKE_ARGS
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
        -DBUILD_SHARED_LIBS=OFF
        -DOATPP_BUILD_TESTS=OFF
        -DOATPP_INSTALL=OFF
        -DOATPP_MODULES_LOCATION=CUSTOM
        -DOATPP_DIR_SRC=${CMAKE_BINARY_DIR}/oatpp/src/oatpp/src
        -DOATPP_DIR_LIB=${CMAKE_BINARY_DIR}/oatpp/src/oatpp/src
    BUILD_IN_SOURCE true
    INSTALL_COMMAND ""
    DEPENDS oatpp
)

set(OATPP_LIB_DEPS
    ${CMAKE_BINARY_DIR}/oatpp-zlib/src/oatpp-zlib/src/liboatpp-zlib.a
)
include_directories(
    SYSTEM ${CMAKE_BINARY_DIR}/oatpp/src/oatpp/src
    SYSTEM ${CMAKE_BINARY_DIR}/oatpp-zlib/src/oatpp-zlib/src
)
set(OATPP_INCLUDE_DIRS
    ${CMAKE_BINARY_DIR}/oatpp/src/oatpp/src
    ${CMAKE_BINARY_DIR}/oatpp-zlib/src/oatpp-zlib/src
)

if(USE_OATPP_SWAGGER)
    ExternalProject_Add(
        oatpp-swagger
        PREFIX oatpp-swagger
        URL https://github.com/oatpp/oatpp-swagger/archive/${OATPP_SWAGGER_VERSION}.tar.gz
        URL_HASH SHA256=afe026d6ba10f5cf43e65578acb80bb82f189d311ed6a0d7c4b8f4bd2b07e55f
        BUILD_IN_SOURCE true
        INSTALL_COMMAND ""
        CMAKE_ARGS
            -DCMAKE_BUILD_TYPE=Release
            -DCMAKE_POSITION_INDEPENDENT_CODE=ON
            -DBUILD_SHARED_LIBS=OFF
            -DOATPP_BUILD_TESTS=OFF
            -DOATPP_INSTALL=OFF
            -DOATPP_MODULES_LOCATION=CUSTOM
            -DOATPP_DIR_SRC=${CMAKE_BINARY_DIR}/oatpp/src/oatpp/src
            -DOATPP_DIR_LIB=${CMAKE_BINARY_DIR}/oatpp/src/oatpp/src
        DEPENDS oatpp
    )

    list(APPEND OATPP_LIB_DEPS
        ${CMAKE_BINARY_DIR}/oatpp-swagger/src/oatpp-swagger/src/liboatpp-swagger.a
    )
    include_directories(
        SYSTEM ${CMAKE_BINARY_DIR}/oatpp-swagger/src/oatpp-swagger/src
    )
    list(APPEND OATPP_INCLUDE_DIRS
        ${CMAKE_BINARY_DIR}/oatpp-swagger/src/oatpp-swagger/src
    )
    # FIXME(sileht): we should copy res outside build dir and dynamically looks at runtime at different places
    add_definitions("-DUSE_OATPP_SWAGGER")
    add_definitions("-DOATPP_SWAGGER_RES_PATH=\"${CMAKE_BINARY_DIR}/oatpp-swagger/src/oatpp-swagger/res/\"")
endif()

list(APPEND OATPP_LIB_DEPS
    ${CMAKE_BINARY_DIR}/oatpp/src/oatpp/src/liboatpp.a
    z) # zlib

if (USE_HTTP_SERVER)
        ExternalProject_Add(
        cpp-netlib
        PREFIX cpp-netlib
        URL http://downloads.cpp-netlib.org/0.11.2/cpp-netlib-0.11.2-final.tar.gz
        URL_HASH SHA256=71953379c5a6fab618cbda9ac6639d87b35cab0600a4450a7392bc08c930f2b1
        BUILD_IN_SOURCE true
        CMAKE_ARGS
            -DCMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE
            -DCPP-NETLIB_BUILD_EXAMPLES=OFF
            -DCPP-NETLIB_BUILD_TESTS=OFF
        INSTALL_COMMAND ""
    )
    set(HTTP_LIB_DEPS ${CMAKE_BINARY_DIR}/cpp-netlib/src/cpp-netlib/libs/network/src/libcppnetlib-uri.a crypto ssl)
    set(HTTP_INCLUDE_DIR ${CMAKE_BINARY_DIR}/cpp-netlib/src/cpp-netlib)
    include_directories(SYSTEM ${HTTP_INCLUDE_DIR})
endif()

# main library, main & tests
if (APPLE)
   include_directories("/opt/homebrew/include/")
   include_directories("/opt/homebrew/opt/libarchive/include/")
   include_directories("/opt/homebrew/include/eigen3")
   include_directories("/opt/homebrew/include/utf8cpp/")

   link_directories("/opt/homebrew/lib/")
endif()

include_directories("${PROJECT_SOURCE_DIR}/src")
# add the binary tree to the search path for include files
# so that we will find dd_config.h
include_directories("${PROJECT_BINARY_DIR}")
# caffe.pb.h, ...
include_directories(${CMAKE_BINARY_DIR}/src)

include_directories(
    ${XGBOOST_INC_DIR}
    ${TSNE_INC_DIR}
    ${TORCH_INC_DIR}
    ${NCNN_INC_DIR}
    ${FMT_INC_DIR}
)
include_directories(BEFORE SYSTEM ${PROTOBUF_INCLUDE_DIR})
include_directories(
    SYSTEM ${TENSORRT_INC_DIR}
)
include_directories(
    ${CMAKE_SOURCE_DIR}/backends/xgb
    ${CMAKE_SOURCE_DIR}/backends/dlib
    ${CMAKE_SOURCE_DIR}/backends/tsne
)
include_directories(${VARIANT_INCLUDE_DIRS})

add_subdirectory(src)

set(COMMON_INCLUDE_DIRS
    ${VARIANT_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${OPENMP_INCLUDE_DIRS}
    ${OATPP_INCLUDE_DIRS}
    ${HTTP_INCLUDE_DIR}
    ${PROTOBUF_INCLUDE_DIR}
    ${SPDLOG_INCLUDE_DIR}
    ${XGBOOST_INC_DIR}
    ${TSNE_INC_DIR}
    ${TORCH_INC_DIR}
    ${TENSORRT_INC_DIR}
    ${NCNN_INC_DIR}

    ${CMAKE_SOURCE_DIR}/backends/xgb
    ${CMAKE_SOURCE_DIR}/backends/dlib
    ${CMAKE_SOURCE_DIR}/backends/tsne
)

if (CUDA_FOUND)
    list(APPEND COMMON_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS})
endif()

set(COMMON_LINK_DIRS
    ${DLIB_LIB_DIR}
    ${TENSORRT_LIB_DIR}
    ${XGBOOST_LIB_DIR}
    ${TSNE_LIB_DIR}
    ${NCNN_LIB_DIR}
    ${FAISS_LIB_DIR}
    ${TORCH_LIB_DIR}
)

set(COMMON_LINK_LIBS
    ddetect ${DLIB_LIB_DEPS} ${TENSORRT_LIBS} ${CUDA_LIB_DEPS} gflags ${OpenCV_LIBS} curlpp curl ${Boost_LIBRARIES} archive ${LMDB_LIBRARY} ${OPENMP_LIBRARIES}
    ${PROTOBUF_LIB_DEPS}
    ${HTTP_LIB_DEPS}
    ${SPDLOG_LIB_DEPS}
    ${OATPP_LIB_DEPS}
    ${XGBOOST_LIB_DEPS}
    ${TSNE_LIB_DEPS}
    ${NCNN_LIB_DEPS}
    ${FAISS_LIB_DEPS}
    ${TORCH_LIB_DEPS}
)
if (USE_HDF5)
    find_library(HDF5_LIBRARY NAMES hdf5_cpp hdf5_serial_cpp)
    message(STATUS "HDF5 library at ${HDF5_LIBRARY}")
    list(APPEND COMMON_LINK_LIBS ${HDF5_LIBRARY})
endif()

# configure config.cmake for external application
get_directory_property(COMMON_DEFINITIONS COMPILE_DEFINITIONS)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/DeepDetectConfig.cmake.in
    ${CMAKE_BINARY_DIR}/DeepDetectConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/DeepDetect
)

write_basic_package_version_file(
    ${CMAKE_BINARY_DIR}/DeepDetectConfigVersion.cmake
    VERSION 0.1.0
    COMPATIBILITY SameMajorVersion)

set(DDETECT_PRIVATE_LINK_LIBS ${COMMON_LINK_LIBS})
list(REMOVE_ITEM DDETECT_PRIVATE_LINK_LIBS ddetect)
target_link_directories(ddetect PRIVATE ${COMMON_LINK_DIRS})
target_link_libraries(ddetect PRIVATE ${DDETECT_PRIVATE_LINK_LIBS})

install(TARGETS ddetect
    EXPORT DeepDetectTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
if (BUILD_PROTOBUF)
  set(PROTOBUF_INSTALL_LIBS ${PROTOBUF_LIB_DEPS})
  foreach(PROTOBUF_LIB_DEP ${PROTOBUF_LIB_DEPS})
    get_filename_component(PROTOBUF_LIB_DEP_REAL
      "${PROTOBUF_LIB_DEP}" REALPATH)
    list(APPEND PROTOBUF_INSTALL_LIBS "${PROTOBUF_LIB_DEP_REAL}")
  endforeach()
  list(REMOVE_DUPLICATES PROTOBUF_INSTALL_LIBS)
  install(FILES ${PROTOBUF_INSTALL_LIBS}
      DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()
if (USE_TORCH AND USE_PREBUILT_TORCH)
  install(FILES
      ${TORCHVISION_LOCATION}/${CMAKE_INSTALL_LIBDIR}/libtorchvision.so
      DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()
install(FILES include/deepdetect/runtime.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/deepdetect)
install(EXPORT DeepDetectTargets
    FILE DeepDetectTargets.cmake
    NAMESPACE DeepDetect::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/DeepDetect)
install(FILES
    ${CMAKE_BINARY_DIR}/DeepDetectConfig.cmake
    ${CMAKE_BINARY_DIR}/DeepDetectConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/DeepDetect)

add_subdirectory(main)

# templates
execute_process(COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/templates/ ${CMAKE_BINARY_DIR}/templates)

# examples
execute_process(COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/examples/ ${CMAKE_BINARY_DIR}/examples)

# patches
execute_process(COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/patches/ ${CMAKE_BINARY_DIR}/patches)

# unit testing
if (BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

# unit testing
if (BUILD_TOOLS)
  add_subdirectory(tools)
endif()

# Get all project files
file(GLOB_RECURSE ALL_SOURCE_FILES src/*.cpp src/*.hpp src/*.h src/*.c tests/*.cc src/*.cc)
list(FILTER ALL_SOURCE_FILES EXCLUDE REGEX "^${CMAKE_SOURCE_DIR}/src/ext/.*$")

# ubuntu 20: clang-format-10, ubuntu 22/24: clang-format-14
find_program(CLANG_FORMAT_BIN NAMES clang-format-14 clang-format clang-format-10)
add_custom_target(
        clang-format
        COMMAND ${CLANG_FORMAT_BIN}
        -style=file
        -i
        ${ALL_SOURCE_FILES}
)
add_custom_target(
        clang-format-check
        COMMAND ${CLANG_FORMAT_BIN}
        --dry-run
        --Werror
        --ferror-limit=10
        -style=file
        -i
        ${ALL_SOURCE_FILES}
)

option(INSTALL_GIT_HOOKS "install client side git hook" OFF)
if (INSTALL_GIT_HOOKS)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/tools/git-pre-commit-hook
                   ${CMAKE_CURRENT_SOURCE_DIR}/.git/hooks/pre-commit @ONLY)
endif()

# configure a header file to pass some of the CMake settings
# to the source code
configure_file (
  "${PROJECT_SOURCE_DIR}/src/dd_config.h.in"
  "${PROJECT_BINARY_DIR}/dd_config.h"
)
configure_file (
  "${PROJECT_SOURCE_DIR}/src/dd_config.cc.in"
  "${PROJECT_BINARY_DIR}/dd_config.cc"
)

# status
message(STATUS "RELEASE:               ${RELEASE}")
message(STATUS "WARNING:               ${WARNING}")
message(STATUS "USE_COMMAND_LINE:      ${USE_COMMAND_LINE}")
message(STATUS "USE_JSON_API:          ${USE_JSON_API}")
message(STATUS "USE_HTTP_SERVER:       ${USE_HTTP_SERVER}")
message(STATUS "USE_HTTP_SERVER_OATPP: ${USE_HTTP_SERVER_OATPP}")
message(STATUS "USE_CPU_ONLY:          ${USE_CPU_ONLY}")
message(STATUS "BUILD_SPDLOG:          ${BUILD_SPDLOG}")
message(STATUS "BUILD_PROTOBUF:        ${BUILD_PROTOBUF}")
message(STATUS "BUILD_TESTS:           ${BUILD_TESTS}")
message(STATUS "BUILD_TOOLS:           ${BUILD_TOOLS}")
message(STATUS "USE_TORCH:             ${USE_TORCH}")
message(STATUS "USE_PYTORCH_WORKER:    ${USE_PYTORCH_WORKER}")
message(STATUS "USE_PREBUILT_TORCH:    ${USE_PREBUILT_TORCH}")
message(STATUS "USE_TORCH_CPU_ONLY:    ${USE_TORCH_CPU_ONLY}")
if (USE_TORCH)
  message(STATUS "TORCH_VERSION:          ${DEEPDETECT_TORCH_VERSION}")
  message(STATUS "TORCH_MODE:             ${DEEPDETECT_TORCH_MODE}")
  message(STATUS "TORCH_LOCATION:         ${TORCH_LOCATION}")
endif()
message(STATUS "USE_SIMSEARCH:         ${USE_SIMSEARCH}")
message(STATUS "USE_NCNN:              ${USE_NCNN}")
message(STATUS "USE_HDF5:              ${USE_HDF5}")
message(STATUS "USE_TENSORRT:          ${USE_TENSORRT}")
message(STATUS "USE_DLIB:              ${USE_DLIB}")
message(STATUS "USE_DLIB_CPU_ONLY:     ${USE_DLIB_CPU_ONLY}")
message(STATUS "USE_DLIB_AVX:          ${USE_DLIB_AVX}")
message(STATUS "USE_ANNOY:             ${USE_ANNOY}")
message(STATUS "USE_FAISS:             ${USE_FAISS}")
message(STATUS "USE_FAISS_CPU_ONLY:    ${USE_FAISS_CPU_ONLY}")
message(STATUS "USE_CUDNN:             ${USE_CUDNN}")
if (USE_CUDNN)
  message(STATUS "CUDNN_VERSION:          ${DEEPDETECT_CUDNN_VERSION}")
endif()
message(STATUS "USE_XGBOOST:           ${USE_XGBOOST}")
message(STATUS "USE_XGBOOST_CPU_ONLY:  ${USE_XGBOOST_CPU_ONLY}")
message(STATUS "USE_TSNE:              ${USE_TSNE}")
message(STATUS "USE_BOOST_BACKTRACE:   ${USE_BOOST_BACKTRACE}")
message(STATUS "USE_OPENMP:            ${USE_OPENMP}")
message(STATUS "USE_CUDA_CV:           ${USE_CUDA_CV}")
message(STATUS "OPENCV_VERSION:        ${OPENCV_VERSION}")
message(STATUS "CUDA_ARCH:             ${CUDA_ARCH}")
