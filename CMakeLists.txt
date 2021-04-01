project(deepdetect)
cmake_minimum_required(VERSION 3.10)

option(RELEASE "release mode" OFF)
option(WARNING "enable warning" ON)
option(USE_CAFFE "build caffe backend" ON)
option(USE_CAFFE_DEBUG "enable caffe debug" OFF)
option(USE_CAFFE2 "build caffe2 backend" OFF)
option(USE_TF "use TF backend" OFF)
option(USE_NCNN "use NCNN backend" OFF)
option(USE_TORCH "use libtorch backend" OFF)
option(USE_HDF5 "use HDF5" ON)
option(USE_TENSORRT "use TensorRT backend" OFF)
option(USE_TENSORRT_OSS "alias of USE_TENSORRT" OFF)
option(USE_XGBOOST "use XGBOOST backend" OFF)
option(USE_DLIB "use Dlib backend" OFF)
option(USE_CUDNN "enable CUDNN" OFF)
option(USE_TSNE "enable TSNE" OFF)
option(USE_CUDA_CV "use CUDA with OpenCV (Requires OpenCV build for CUDA)" OFF)
option(USE_SIMSEARCH "build index and search services" OFF)
option(USE_ANNOY "use annoy as indexer" OFF)
option(USE_FAISS "use FAISS as indexer" ON)
option(BUILD_SPDLOG "build SPDLOG instead of using system library" ON)
option(BUILD_PROTOBUF "build PROTOBUF instead of using system library" ON)
option(USE_BOOST_BACKTRACE "use boost backtrace" ON)

if (USE_CAFFE)
  add_definitions(-DUSE_CAFFE)
endif()

if (USE_TENSORRT_OSS)
  set(USE_TENSORRT ON)
endif()

if (USE_TF AND USE_CAFFE2)
  message(FATAL_ERROR "Building with tensorflow AND caffe2 can't be build together")
endif()

if (USE_TORCH AND USE_CAFFE2)
  message(FATAL_ERROR "Building with torch AND caffe2 can't be build together")
endif()

if (USE_TF AND USE_TORCH)
  message(FATAL_ERROR "Building with tensorflow AND torch can't be build together")
endif()

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

include(ProcessorCount)
ProcessorCount(N)
include(ExternalProject)

# options
OPTION(BUILD_TESTS "Should the tests be built")
OPTION(BUILD_TOOLS "Should the tools be built")
OPTION(USE_COMMAND_LINE "build command line JSON API" ON)
OPTION(USE_JSON_API "build internal JSON API" ON)
OPTION(USE_HTTP_SERVER "build cppnet-lib version of http JSON API " OFF)
OPTION(USE_HTTP_SERVER_OATPP "build oatpp version of http JSON API" ON)

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

set(CMAKE_CXX_FLAGS "-g -O2 -Wall -Wextra -fopenmp -fPIC -std=c++14 -DUSE_OPENCV -DUSE_LMDB")

if(WARNING)
  string(APPEND CMAKE_CXX_FLAGS " -Werror")
endif()

if(RELEASE)
    set(BUILD_TYPE release)
else()
    set(BUILD_TYPE dev)
endif()

if (USE_COMMAND_LINE)
  if (NOT USE_CAFFE)
    set(USE_JSON_API ON)
  endif()
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

# PROTOBUF

# use protobuf same version as pytorch one
if (BUILD_PROTOBUF)
    set(PROTOBUF_VERSION 3.11.4)
    ExternalProject_Add(
        protobuf
        PREFIX protobuf
        URL https://github.com/protocolbuffers/protobuf/releases/download/v${PROTOBUF_VERSION}/protobuf-cpp-${PROTOBUF_VERSION}.tar.gz
        URL_HASH SHA256=ba4bbc3e6b58d2ccfe406e616576ef48710a2aae205f463618025fc691549cfe
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
    set(PROTOBUF_LIB_DEPS ${PROTOBUF_LIB_DIR}/libprotobuf.so ${PROTOBUF_LIB_DIR}/libprotoc.so ${PROTOBUF_LIB_DIR}/libprotobuf-lite.so)
    link_directories(${PROTOBUF_LIB_DIR})
    include_directories(SYSTEM ${PROTOBUF_INCLUDE_DIR})
else()
    if (USE_CAFFE OR USE_TORCH)
        find_package(Protobuf REQUIRED 3.11.4)
    else()
        find_package(Protobuf REQUIRED)
    endif()

    set(PROTOBUF_PROTOC ${Protobuf_PROTOC_EXECUTABLE})
    set(PROTOBUF_INCLUDE_DIR ${Protobuf_INCLUDE_DIRS})
    set(PROTOBUF_LIB_DEPS protobuf::libprotobuf)
    add_custom_target(protobuf)
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


# dependency on Eigen for confusion matrix fast computation
if (USE_TF)
  set(TENSORFLOW_CC_DIR ${CMAKE_BINARY_DIR}/tensorflow_cc/src/tensorflow_cc-build/)
  set(EIGEN3_INCLUDE_DIR ${TENSORFLOW_CC_DIR}/tensorflow/tensorflow/contrib/makefile/downloads/eigen/)
elseif (USE_CAFFE2)
  set(PYTORCH_PATH ${CMAKE_BINARY_DIR}/pytorch/src/pytorch)
  set(DETECTRON_PATH ${CMAKE_BINARY_DIR}/detectron/src/detectron)
  set(EIGEN3_INCLUDE_DIR ${PYTORCH_PATH}/third_party/eigen)
else()
   find_package(PkgConfig)
   pkg_search_module(Eigen3 REQUIRED eigen3)
   set(EIGEN3_INCLUDE_DIR "/usr/include/eigen3")
endif()

set(eigen_archive_hash "50812b426b7c")

include_directories("${EIGEN3_INCLUDE_DIR}")

# spdlog, usefull for ubuntu20.04/rhel/centos that ship a bugged version (missing header, don't compile with nvcc, ...)
if (BUILD_SPDLOG)
    ExternalProject_Add(
        spdlog
        PREFIX spdlog
        URL https://github.com/gabime/spdlog/archive/v1.8.2.tar.gz
        URL_HASH SHA256=e20e6bd8f57e866eaf25a5417f0a38a116e537f1a77ac7b5409ca2b180cec0d5
        BUILD_IN_SOURCE true
        CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/spdlog/build -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    )
    set(SPDLOG_LIB_DEPS ${CMAKE_BINARY_DIR}/spdlog/build/lib/libspdlog.a)
    set(SPDLOG_LIB_DIR ${CMAKE_BINARY_DIR}/spdlog/build/lib)
    set(SPDLOG_CAFFE_LIB_DEPS :libspdlog.a)
    set(SPDLOG_INCLUDE_DIR ${CMAKE_BINARY_DIR}/spdlog/build/include)
    include_directories(SYSTEM ${SPDLOG_INCLUDE_DIR})
else()
    find_package(spdlog 1.8.2 CONFIG REQUIRED)
    set(SPDLOG_LIB_DEPS spdlog::spdlog)
endif()

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
option(USE_TF_CPU_ONLY "use TF without CUDA" OFF)
option(USE_TORCH_CPU_ONLY "use TORCH without CUDA" OFF)
option(USE_CAFFE_CPU_ONLY "use CAFFE without CUDA" OFF)
option(USE_DLIB_CPU_ONLY "use DLIB without CUDA" OFF)
option(USE_CAFFE2_CPU_ONLY "use CAFFE2 without CUDA" OFF)
option(USE_FAISS_CPU_ONLY "use FAISS without CUDA" OFF)

# Deprecated flag
if (USE_XGBOOST_GPU)
  set(USE_XGBOOST_CPU_ONLY OFF)
endif()

if (USE_CPU_ONLY)
  set(USE_XGBOOST_CPU_ONLY ON)
  set(USE_CAFFE_CPU_ONLY ON)
  set(USE_DLIB_CPU_ONLY ON)
  set(USE_TF_CPU_ONLY ON)
  set(USE_CAFFE2_CPU_ONLY ON)
  set(USE_TORCH_CPU_ONLY ON)
  set(USE_FAISS_CPU_ONLY ON)
else()
  cmake_minimum_required(VERSION 3.14)
  include(cmake/Cuda.cmake) # cuda + cudnn
endif()

# CUDA validation
if (NOT CUDA_FOUND)
  if (USE_TENSORRT)
      message(FATAL, "USE_TENSORRT=ON needs CUDA installed")
  endif()
  if (USE_TF AND NOT USE_TF_CPU_ONLY)
      message(FATAL, "USE_TF=ON needs CUDA installed")
  endif()
  if (USE_CAFFE AND NOT USE_CAFFE_CPU_ONLY)
      message(FATAL, "USE_CAFFE=ON needs CUDA installed")
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
  if (USE_CAFFE2 AND NOT USE_CAFFE2_CPU_ONLY)
      message(FATAL, "USE_CAFFE2=ON needs CUDA installed")
  endif()
  if (USE_FAISS AND NOT USE_FAISS_CPU_ONLY)
      message(FATAL, "USE_FAISS=ON needs CUDA installed")
  endif()
endif()

# annoy
if (USE_CAFFE)
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
        set(CONFIGURE_OPTS "--without-cuda")
      else()
        string(REPLACE "/include" "" CUDA_PREFIX ${CUDA_INCLUDE_DIRS})
        if (NOT CUDA_ARCH)
          string(REPLACE ";" " " CUDA_ARCH "${CUDA_NVCC_FLAGS}")
        endif()
        set(CONFIGURE_OPTS --with-cuda=${CUDA_PREFIX} --with-cuda-arch=${CUDA_ARCH})
        add_definitions(-DUSE_GPU_FAISS)
      endif()
      message(STATUS "Fetching FAISS")
      add_definitions(-DUSE_FAISS)
      add_definitions(-DUSE_SIMSEARCH)
      ExternalProject_Add(
        faisslib
        PREFIX faiss
        SOURCE_DIR ${CMAKE_BINARY_DIR}/faiss/src/faiss
        GIT_REPOSITORY https://github.com/jolibrain/faiss
        CONFIGURE_COMMAND ./configure ${CONFIGURE_OPTS}
        INSTALL_COMMAND ""
        BUILD_IN_SOURCE 1
      )

      set(FAISS_INCLUDE_DIR ${CMAKE_BINARY_DIR}/faiss/src/)
      include_directories("${FAISS_INCLUDE_DIR}")
      set(FAISS_LIB_DIR ${CMAKE_BINARY_DIR}/faiss/src/faiss)
      set(FAISS_LIB_DEPS faiss)
    endif()
  endif()
else()
  set(USE_SIMSEARCH OFF)
endif()


if (USE_CUDNN)
  find_package(CUDNN REQUIRED)
endif()

# CUDA dependencies
if (CUDA_FOUND)
  set(CUDA_LIB_DEPS ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_curand_LIBRARY} ${CUDA_CUDART_LIBRARY} ${CUDA_cusolver_LIBRARY})
  if (USE_CUDNN)
    set(CUDA_LIB_DEPS ${CUDA_LIB_DEPS} ${CUDNN_LIBRARY})
  endif()
else()
  set(CUDA_LIB_DEPS "")
  add_definitions(-DCPU_ONLY)
endif()

if (USE_DLIB)
    # Dlib

    # Currently supported release version of dlib
    set(DLIB_RELEASE_VERSION v19.18)
    message(STATUS "Using Dlib version ${DLIB_RELEASE_VERSION}")
    add_definitions(-DUSE_DLIB)
    set(DLIB_LIB_DEPS -ldlib -lopenblas)

    if (USE_DLIB_CPU_ONLY)
        ExternalProject_Add(
            dlib
            PREFIX dlib
            URL https://github.com/davisking/dlib/archive/${DLIB_RELEASE_VERSION}.tar.gz
            CONFIGURE_COMMAND cd dlib && mkdir -p build && cd build && cmake .. -DDLIB_NO_GUI_SUPPORT=ON -DDLIB_USE_CUDA=OFF -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/dlib/build && cmake --build . --config Release --target install
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
          CONFIGURE_COMMAND cd dlib && mkdir -p build && cd build && cmake .. -DDLIB_NO_GUI_SUPPORT=ON -DDLIB_USE_CUDA=ON -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/dlib/build && cmake --build . --config Release --target install
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

set(CAFFE_DD_USER jolibrain)
set(CAFFE_DD_BRANCH master)

if (USE_CAFFE2)

  message(DEPRECATION "CAFFE2 backend is deprecated and will be removed in the future")

  add_definitions(-DUSE_CAFFE2)
  list(APPEND CAFFE2_LIB_DEPS -lcaffe2)
  if (NOT USE_CAFFE2_CPU_ONLY)
    list(APPEND CAFFE2_LIB_DEPS -lcaffe2_gpu)
  endif()

  include_directories(
    ${PYTORCH_PATH}
    ${PYTORCH_PATH}-build
    ${PYTORCH_PATH}/aten/src
  )
  set(CAFFE2_PATCHES ${CMAKE_BINARY_DIR}/patches/caffe2)

  set(PYTORCH_SUPPORTED_COMMIT ff608a9) # Pre-release v1.0rc1
  set(PYTORCH_PATCHES
    ${CAFFE2_PATCHES}/pytorch/0001-eigen.patch
    ${CAFFE2_PATCHES}/pytorch/0002-logging.patch
    ${CAFFE2_PATCHES}/pytorch/0003-collect_proposals.patch
    )

  set(DETECTRON_SUPPORTED_COMMIT 8181a32) # Nov 7, 2018
  set(DETECTRON_PATCHES
    ${CAFFE2_PATCHES}/detectron/0001-dependencies.patch
    ${CAFFE2_PATCHES}/detectron/0002-compiled.patch
    ${CAFFE2_PATCHES}/detectron/0003-ops.patch
    ${CAFFE2_PATCHES}/detectron/0004-import.patch
    ${CAFFE2_PATCHES}/detectron/0005-visual_genome.patch
    ${CAFFE2_PATCHES}/detectron/0006-pkl_cache.patch
    ${CAFFE2_PATCHES}/detectron/0007-weight_transfer.patch
    )

  list(APPEND CAFFE2_OPS
    ${CAFFE2_PATCHES}/custom_ops/bbox_to_roi_op.h
    ${CAFFE2_PATCHES}/custom_ops/bbox_to_roi_op.cc
    ${CAFFE2_PATCHES}/custom_ops/segment_mask_op.h
    ${CAFFE2_PATCHES}/custom_ops/segment_mask_op.cc
    ${CAFFE2_PATCHES}/custom_ops/multi_level_roi_op.h
    ${CAFFE2_PATCHES}/custom_ops/multi_level_roi_op.cc
  )

  list(APPEND PYTORCH_FLAGS

    -DBUILD_CUSTOM_PROTOBUF=OFF
    -DCAFFE2_LINK_LOCAL_PROTOBUF=0

    -DUSE_OPENMP=ON
    -DUSE_MPI=OFF

    -DUSE_GFLAGS=OFF
    -DUSE_GLOG=OFF

    -DBUILD_TEST=OFF
    -DBUILD_BINARY=OFF
    -DBUILD_DOCS=OFF
  )

  # Only a few submodules are currently used by caffe2
  # (No need to log, benchmark, compile for IOS, make python libraries, ...)
  list(APPEND PYTORCH_SUBMODULES
    third_party/cpuinfo
    third_party/cub
    third_party/eigen
    third_party/FP16
    third_party/FXdiv
    third_party/gloo
    third_party/NNPACK
    third_party/onnx
    third_party/psimd
    third_party/pthreadpool
    third_party/pybind11
    third_party/python-peachpy
    third_party/sleef
    )

  # Pytorch
  set(PYTORCH_COMPLETE ${CMAKE_BINARY_DIR}/CMakeFiles/pytorch-complete)
  ExternalProject_Add(
    pytorch
    PREFIX pytorch
    GIT_REPOSITORY https://github.com/pytorch/pytorch.git
    GIT_SUBMODULES ${PYTORCH_SUBMODULES}
    UPDATE_DISCONNECTED 1
    GIT_TAG ${PYTORCH_SUPPORTED_COMMIT}
    PATCH_COMMAND test -f ${PYTORCH_COMPLETE} && echo Skipping || echo cp modules/detectron/*_op.* caffe2/operators | bash && cp ${CAFFE2_OPS} caffe2/operators && git am ${PYTORCH_PATCHES}
    CONFIGURE_COMMAND test -f ${PYTORCH_COMPLETE} && echo Skipping || PATH=${PROTOBUF_LIB_DIR}:$ENV{PATH} cmake ../pytorch ${PYTORCH_FLAGS}
    BUILD_COMMAND test -f ${PYTORCH_COMPLETE} && echo Skipping || make -j${N}
    INSTALL_COMMAND ""
    DEPENDS protobuf
  )

  # Detectron
  set(DETECTRON_COMPLETE ${CMAKE_BINARY_DIR}/CMakeFiles/detectron-complete)
  ExternalProject_Add(
    detectron
    PREFIX detectron
    GIT_REPOSITORY https://github.com/facebookresearch/Detectron
    UPDATE_DISCONNECTED 1
    GIT_TAG ${DETECTRON_SUPPORTED_COMMIT}
    PATCH_COMMAND test -f ${DETECTRON_COMPLETE} && echo Skipping || git am ${DETECTRON_PATCHES}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    DEPENDS protobuf
  )

  # Python path
  if (NOT EXISTS ${CMAKE_BINARY_DIR}/python_path)
    execute_process(COMMAND mkdir -p ${CMAKE_BINARY_DIR}/python_path)
    execute_process(COMMAND ln -s ${PYTORCH_PATH}-build
      ${CMAKE_BINARY_DIR}/python_path/pytorch)
    execute_process(COMMAND ln -s ${DETECTRON_PATH}
      ${CMAKE_BINARY_DIR}/python_path/detectron)
  endif()

endif()

if (USE_TF)
  # Tensorflow
  message(STATUS "Fetching Tensorflow")
  add_definitions(-DUSE_TF)
  set(TF_LIB_DEPS -ltensorflow_cc)

  if (USE_TF_CPU_ONLY)
    add_definitions(-DUSE_TF_CPU_ONLY)
    ExternalProject_Add(
      tensorflow_cc
      PREFIX tensorflow_cc
      GIT_REPOSITORY https://github.com/FloopCZ/tensorflow_cc.git
      GIT_TAG v1.15.0
      SOURCE_SUBDIR tensorflow_cc
      CMAKE_ARGS -DTENSORFLOW_STATIC=OFF -DTENSORFLOW_SHARED=ON -DALLOW_CUDA=OFF
      INSTALL_COMMAND ""
      BUILD_IN_SOURCE 1
      )
  else()
    ExternalProject_Add(
      tensorflow_cc
      PREFIX tensorflow_cc
      GIT_REPOSITORY https://github.com/FloopCZ/tensorflow_cc.git
      GIT_TAG v1.15.0
      SOURCE_SUBDIR tensorflow_cc
      CMAKE_ARGS -DTENSORFLOW_STATIC=OFF -DTENSORFLOW_SHARED=ON
      INSTALL_COMMAND ""
      BUILD_IN_SOURCE 1
      )
  endif()

  set(TF_INC_DIR ${TENSORFLOW_CC_DIR}/tensorflow/ ${TENSORFLOW_CC_DIR}/tensorflow/bazel-genfiles/ ${TENSORFLOW_CC_DIR}/tensorflow/tensorflow/contrib/makefile/downloads/nsync/public/ ${TENSORFLOW_CC_DIR}/tensorflow/tensorflow/contrib/makefile/downloads/absl/)
  set(TF_LIB_DIR ${TENSORFLOW_CC_DIR}/tensorflow/bazel-bin/tensorflow/)

  include_directories("${TF_INC_DIR}")

endif() # USE_TF

# OpenCV
find_package(OpenCV 3 QUIET COMPONENTS core imgproc highgui imgcodecs)
if (NOT OpenCV_FOUND)
  find_package(OpenCV 4 QUIET COMPONENTS core imgproc highgui imgcodecs)
endif()
if (NOT OpenCV_FOUND)
  find_package(OpenCV 2 REQUIRED COMPONENTS core imgproc highgui imgcodecs)
endif()
set(OPENCV_VERSION ${OpenCV_VERSION_MAJOR})
include_directories(${OpenCV_INCLUDE_DIRS})
message(STATUS "OpenCV ${OPENCV_VERSION} (${OpenCV_VERSION}) found (${OpenCV_CONFIG_PATH})")

if (USE_CUDA_CV)
  message(STATUS "Using CUDA OpenCV")
  string(APPEND CMAKE_CXX_FLAGS " -DUSE_CUDA_CV")
endif()

# customized Caffe as external project
if (CAFFE_INC_DIR AND CAFFE_LIB_DIR)
  # do nothing
elseif(USE_CAFFE)
  message(STATUS "Configuring customized caffe")

  if (USE_DD_SYSLOG)
    set(USE_SYSLOG 1)
  else()
    set(USE_SYSLOG 0)
  endif()


  # Set config file
  if (USE_CAFFE_CPU_ONLY)
    add_definitions(-DUSE_CAFFE_CPU_ONLY)
    set(CAFFE_DD_CONFIG_FILE Makefile.config.cpu)
  else()
    if (HAVE_CUDNN AND USE_CUDNN)
      if(JETSON)
	set(CAFFE_DD_CONFIG_FILE Makefile.config.gpu.cudnn.jetson)
      else()
	set(CAFFE_DD_CONFIG_FILE Makefile.config.gpu.cudnn)
      endif()
    else() # HAVE_CUDNN
      set(CAFFE_DD_CONFIG_FILE Makefile.config.gpu)
    endif() # HAVE_CUDNN
  endif()

  if (RPI3)
    set(CAFFE_DD_CONFIG_FILE Makefile.config.pi3)
  endif()

  # Set caffe DEBUG mode
  if (USE_CAFFE_DEBUG)
    set(CAFFE_DEBUG 1)
  else()
    set(CAFFE_DEBUG 0)
  endif()

  message(STATUS "include_dirs : ${INCLUDE_DIRS}")

  ExternalProject_Add(
    caffe_dd
    PREFIX caffe_dd
    URL https://github.com/${CAFFE_DD_USER}/caffe/archive/${CAFFE_DD_BRANCH}.tar.gz
    CONFIGURE_COMMAND
        sh -c "\
            diff ${CMAKE_BINARY_DIR}/caffe_dd/src/caffe_dd/.build_release/src/caffe/proto/caffe.pb.cc ${CMAKE_CURRENT_BINARY_DIR}/src/caffe.pb.cc > /dev/null 2>&1 \
            && ( echo 'caffe version of caffe.pb.h/caffe.pb.cc is already up to date') \
            || ( echo 'caffe.pb.h/caffe.pb.cc have change rebuilding caffe' && rm -rf ${CMAKE_BINARY_DIR}/caffe_dd/src/caffe_dd/.build_release ) \
        " &&
        ln -sf ${CAFFE_DD_CONFIG_FILE} Makefile.config &&
        echo "OPENCV_VERSION:=${OPENCV_VERSION}" >> Makefile.config &&
        echo "USE_SYSLOG:=${USE_SYSLOG}" >> Makefile.config &&
        echo "DEBUG:=${CAFFE_DEBUG}" >> Makefile.config &&
        echo "INCLUDE_DIRS+=${SPDLOG_INCLUDE_DIR}" >> Makefile.config &&
        echo "INCLUDE_DIRS+=${PROTOBUF_INCLUDE_DIR}" >> Makefile.config &&
        echo "CUDA_ARCH:=${CUDA_ARCH}" >> Makefile.config &&
        echo "LIBRARIES+=${SPDLOG_CAFFE_LIB_DEPS}" >> Makefile.config &&
        echo "LIBRARY_DIRS+=${SPDLOG_LIB_DIR}" >> Makefile.config &&
        echo "LIBRARY_DIRS+=${PROTOBUF_LIB_DIR}" >> Makefile.config
    BUILD_COMMAND 
        ${CMAKE_COMMAND} -E env LD_LIBRARY_PATH=${PROTOBUF_LIB_DIR}:$ENV{LD_LIBRARY_PATH} PATH=${PROTOBUF_LIB_DIR}:$ENV{PATH} make -j${N} proto &&
        ${CMAKE_COMMAND} -E env LD_LIBRARY_PATH=${PROTOBUF_LIB_DIR}:$ENV{LD_LIBRARY_PATH} PATH=${PROTOBUF_LIB_DIR}:$ENV{PATH} make -j${N} all
    INSTALL_COMMAND ""
    BUILD_IN_SOURCE 1
    DEPENDS caffe_pb_h protobuf spdlog
  )

  if (USE_HDF5)
    if (JETSON)
      set(HDF5_LIB /usr/lib/aarch64-linux-gnu/hdf5/serial)
    elseif (RPI3)
      set(HDF5_LIB /usr/lib/arm-linux-gnueabihf/hdf5/serial)
    else()
      set(HDF5_LIB /usr/lib/x86_64-linux-gnu/hdf5/serial)
    endif()
  endif()

  if (USE_HDF5)
    if (USE_CAFFE_CPU_ONLY)
      set(CAFFE_INC_DIR ${CMAKE_BINARY_DIR}/caffe_dd/src/caffe_dd/include ${CMAKE_BINARY_DIR}/caffe_dd/src/caffe_dd/build/src /usr/include/hdf5/serial)
      set(CAFFE_LIB_DIR $ENV{HOME}/lib ${CMAKE_BINARY_DIR}/caffe_dd/src/caffe_dd/build/lib ${CMAKE_BINARY_DIR}/caffe_dd/src/caffe_dd/build/src ${HDF5_LIB})
    else()
      set(CAFFE_INC_DIR ${CMAKE_BINARY_DIR}/caffe_dd/src/caffe_dd/include ${CMAKE_BINARY_DIR}/caffe_dd/src/caffe_dd/build/src ${CUDA_INCLUDE_DIRS})
      set(CAFFE_LIB_DIR ${CMAKE_BINARY_DIR}/caffe_dd/src/caffe_dd/build/lib ${CMAKE_BINARY_DIR}/caffe_dd/src/caffe_dd/build/src ${HDF5_LIB})
    endif()
  else()
    if (USE_CAFFE_CPU_ONLY)
      set(CAFFE_INC_DIR ${CMAKE_BINARY_DIR}/caffe_dd/src/caffe_dd/include ${CMAKE_BINARY_DIR}/caffe_dd/src/caffe_dd/build/src)
      set(CAFFE_LIB_DIR $ENV{HOME}/lib ${CMAKE_BINARY_DIR}/caffe_dd/src/caffe_dd/build/lib ${CMAKE_BINARY_DIR}/caffe_dd/src/caffe_dd/build/src)
    else()
      set(CAFFE_INC_DIR ${CMAKE_BINARY_DIR}/caffe_dd/src/caffe_dd/include ${CMAKE_BINARY_DIR}/caffe_dd/src/caffe_dd/build/src ${CUDA_INCLUDE_DIRS})
      set(CAFFE_LIB_DIR ${CMAKE_BINARY_DIR}/caffe_dd/src/caffe_dd/build/lib ${CMAKE_BINARY_DIR}/caffe_dd/src/caffe_dd/build/src)
    endif()
  endif()
endif()

# Caffe dependencies
if (USE_CAFFE)
  set(CAFFE_LIB_DEPS -lleveldb -lsnappy -llmdb -lopenblas -lcaffe)
  if (USE_HDF5)
    string(APPEND CAFFE_LIB_DEPS " -lhdf5_hl -lhdf5")
  endif()
endif()

# NCNN
if (USE_NCNN)
  message(STATUS "Configuring NCNN")
  add_definitions(-DUSE_NCNN)
  include_directories(${CMAKE_CURRENT_BINARY_DIR})

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
      PATCH_COMMAND ${CMAKE_SOURCE_DIR}/cmake/patches-apply.sh ${NCNN_PATCHES}
      CMAKE_ARGS ${NCNN_BUILD_FLAGS}
      INSTALL_COMMAND ""
      )
  endif()

  set(NCNN_LIB_DEPS ${CMAKE_BINARY_DIR}/ncnn/src/ncnn-build/src/libncnn.a)
  set(NCNN_INC_DIR ${CMAKE_BINARY_DIR}/ncnn/src/ncnn/src ${CMAKE_BINARY_DIR}/ncnn/src/ncnn-build/src)
  set(NCNN_LIB_DIR ${CMAKE_BINARY_DIR}/ncnn/src/ncnn-build/src)
  include_directories(${NCNN_INC_DIR})
endif()

# Torch
if (USE_TORCH)

  set(CMAKE_EXE_LINKER_FLAGS "-Wl,--no-as-needed")

  set(PYTORCH_PATCHES_PATH ${CMAKE_BINARY_DIR}/patches/pytorch)

  set(PYTORCH_PATCHES
    ${PYTORCH_PATCHES_PATH}/pytorch_15_compile.patch
    ${PYTORCH_PATCHES_PATH}/pytorch_16_new_logger.patch
    ${PYTORCH_PATCHES_PATH}/pytorch_16_use_new_logger.patch
    ${PYTORCH_PATCHES_PATH}/pytorch_14_namespace.patch
  )

  message(STATUS "Configuring libtorch")
  add_definitions(-DUSE_TORCH)

  if (NOT TORCH_LOCATION)
    #set(PYTORCH_COMMIT ee77ccbb6da4e2efd83673e798acf7081bc03564) # 1.3.1
    #set(PYTORCH_COMMIT 7f73f1d591afba823daa4a99a939217fb54d7688) # 1.4
    #set(PYTORCH_COMMIT 4ff3872a2099993bf7e8c588f7182f3df777205b)  # 1.5
	#set(PYTORCH_COMMIT b31f58de6fa8bbda5353b3c77d9be4914399724d)  # 1.6
    set(PYTORCH_COMMIT v1.8.0)
    set(PYTORCH_COMPLETE ${CMAKE_BINARY_DIR}/CMakeFiles/pytorch-complete)

    if(USE_TORCH_CPU_ONLY)
      set(PYTORCH_USE_CUDA 0)
    else()
      set(PYTORCH_USE_CUDA 1)
    endif()

    ExternalProject_Add(
      pytorch
      PREFIX pytorch
      GIT_REPOSITORY https://github.com/pytorch/pytorch.git
      GIT_TAG ${PYTORCH_COMMIT}
      GIT_CONFIG advice.detachedHead=false
      UPDATE_DISCONNECTED 1
      PATCH_COMMAND test -f ${PYTORCH_COMPLETE} && echo Skipping || git apply ${PYTORCH_PATCHES} && echo Applying ${PYTORCH_PATCHES}
      CONFIGURE_COMMAND ""
      BUILD_COMMAND ""
      COMMAND test -f ${PYTORCH_COMPLETE} && echo Skipping || ${CMAKE_COMMAND} -E env PATH=${PROTOBUF_LIB_DIR}:$ENV{PATH} BUILD_CUSTOM_PROTOBUF=0 GLIBCXX_USE_CXX11_ABI=1 BUILD_TEST=0 USE_CUDA=${PYTORCH_USE_CUDA} BUILD_CAFFE2_OPS=0 USE_DDLOG=1 CAFFE2_LINK_LOCAL_PROTOBUF=0 "CMAKE_CXX_FLAGS=-isystem${SPDLOG_INCLUDE_DIR} -isystem${PROTOBUF_INCLUDE_DIR}" CMAKE_PREFIX_PATH=${PROTOBUF_LIB_DIR}/cmake MAX_JOBS=8 python3 ../pytorch/tools/build_libtorch.py
      INSTALL_COMMAND ""
    )

    set(TORCH_LOCATION ${CMAKE_BINARY_DIR}/pytorch/src/pytorch/torch)
  endif()

  set(TORCH_LIB_DEPS ${TORCH_LOCATION}/lib/libtorch.so ${TORCH_LOCATION}/lib/libtorch_cpu.so ${TORCH_LOCATION}/lib/libc10.so -llmdb)

  if (USE_TORCH_CPU_ONLY)
    list(APPEND TORCH_LIB_DEPS ${TORCH_LOCATION}/lib/libtorch_cpu.so iomp5)
  else()
    list(APPEND TORCH_LIB_DEPS ${TORCH_LOCATION}/lib/libc10_cuda.so ${TORCH_LOCATION}/lib/libtorch_cuda.so)
  endif()

  set(TORCH_INC_DIR ${TORCH_LOCATION}/include/ ${TORCH_LOCATION}/include/torch/csrc/api/include/ ${CMAKE_BINARY_DIR}/pytorch/src/pytorch/torch/include/torch/csrc/api/include ${TORCH_LOCATION}/.. ${CMAKE_BINARY_DIR}/src)
  set(TORCH_LIB_DIR ${TORCH_LOCATION}/lib/)

  # TORCH VISION
  message(STATUS "Configuring Pytorch Vision")
  # set(PYTORCH_VISION_COMMIT "v0.6.0") # 0.6 & torch 1.5
  #set(PYTORCH_VISION_COMMIT "v0.7.0") # 0.7 & torch 1.6
  #set(PYTORCH_VISION_COMMIT "v0.8.2") # 0.8.2 & torch 1.7.1
  set(PYTORCH_VISION_COMMIT "v0.9.0") # 0.9.0 & torch 1.8
  set(TORCHVISION_LOCATION "${CMAKE_BINARY_DIR}/pytorch_vision/src/pytorch_vision-install")
  ExternalProject_Add(
    pytorch_vision
    PREFIX pytorch_vision
    INSTALL_DIR ${CMAKE_BINARY_DIR}
    GIT_REPOSITORY https://github.com/pytorch/vision.git
    GIT_TAG ${PYTORCH_VISION_COMMIT}
    GIT_CONFIG advice.detachedHead=false
    CMAKE_ARGS -DWITH_CUDA=${PYTORCH_USE_CUDA} -DCMAKE_PREFIX_PATH=${TORCH_LOCATION} -DCMAKE_INSTALL_PREFIX=${TORCHVISION_LOCATION}
      -DCMAKE_INCLUDE_PATH=${PROTOBUF_INCLUDE_DIR} -DCMAKE_LIBRARY_PATH=${PROTOBUF_LIB_DIR}
      "-DCMAKE_CXX_FLAGS=-isystem ${CMAKE_BINARY_DIR}/pytorch/src/pytorch/ -isystem ${SPDLOG_INCLUDE_DIR} -isystem ${PROTOBUF_INCLUDE_DIR}"
      "-DCMAKE_CUDA_FLAGS=-isystem ${CMAKE_BINARY_DIR}/pytorch/src/pytorch/ -isystem ${SPDLOG_INCLUDE_DIR} -isystem ${PROTOBUF_INCLUDE_DIR}"
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
    DEPENDS protobuf pytorch
  )

  list(APPEND TORCH_INC_DIR "${TORCHVISION_LOCATION}/include/")
  list(APPEND TORCH_LIB_DIR "${TORCHVISION_LOCATION}/lib/")
  list(APPEND TORCH_LIB_DEPS torchvision)
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
        -DCUDA_NVCC_FLAGS="-Xcompiler -fPIC --expt-extended-lambda $CUDA_ARCH"
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
      GIT_REPOSITORY https://github.com/beniz/Multicore-TSNE.git
      SOURCE_SUBDIR multicore_tsne
      CMAKE_ARGS -Wno-dev
      INSTALL_COMMAND ""
   )
  set(TSNE_INC_DIR ${CMAKE_BINARY_DIR}/Multicore-TSNE/src/Multicore-TSNE/multicore_tsne)
  set(TSNE_LIB_DIR ${CMAKE_BINARY_DIR}/Multicore-TSNE/src/Multicore-TSNE-build)
endif()

# add the binary tree to the search path for include files
# so that we will find dd_config.h
include_directories("${PROJECT_BINARY_DIR}")
include_directories(${CAFFE_INC_DIR} ${XGBOOST_INC_DIR} ${TSNE_INC_DIR} ${TORCH_INC_DIR})
include_directories(${CMAKE_SOURCE_DIR}/src/backends/caffe ${CMAKE_SOURCE_DIR}/backends/xgb ${CMAKE_SOURCE_DIR}/backends/tf ${CMAKE_SOURCE_DIR}/backends/dlib ${CMAKE_SOURCE_DIR}/backends/tsne)

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
    set(TRTTESTDIR TENSORRT_LIB_DIR)
  else()
    set(TRTTESTDIR /usr/lib/x86_64-linux-gnu)
    set(TENSORRT_LIB_DIR  /usr/lib/x86_64-linux-gnu)
    set(TENSORRT_INC_DIR /usr/include/x86_64-linux-gnu)
  endif()

  if (NOT EXISTS "${TRTTESTDIR}/libnvinfer.so")
    message(FATAL_ERROR "Could not find TensorRT ${TENSORRT_LIB_DIR}/libnvinfer.so, please provide tensorRT location as TENSORRT_DIR or (TENSORRT_LIB_DIR _and_ TENSORRT_INC_DIR)")
  else()
    message(STATUS "Found TensorRT libraries : ${TRTTESTDIR}/libnvinfer.so")
  endif()

    set(TENSORRT_LIB_DIR ${CMAKE_BINARY_DIR}/tensorrt-oss/bin/ ${TENSORRT_LIB_DIR})
    include_directories(${CMAKE_CURRENT_BINARY_DIR})
	set(TENSORRT_LIBS nvinfer nvcaffeparser nvinfer_plugin nvonnxparser )

    if (NOT TENSORRT_VERSION)
        if (EXISTS "${TRTTESTDIR}/libnvinfer.so.7.1.3")
            set(TENSORRT_VERSION 20.09)
            message(STATUS "Found TensorRT libraries version 7.1.3")
        # TODO(sileht): Update trt-oss patches for 7.2.2
        # elseif (EXISTS "${TRTTESTDIR}/libnvinfer.so.7.2.2")
        #     set(TENSORRT_VERSION 20.12)
        #     message(STATUS "Found TensorRT libraries version 7.2.2")
        elseif (EXISTS "${TRTTESTDIR}/libnvinfer.so.7")
            set(TENSORRT_VERSION 20.09)
            message(WARNING "No supported TensorRT version found assuming 20.09/7.1.3")
        else()
            message(FATAL_ERROR "No supported TensorRT version found")
        endif()
    endif()
    add_definitions(-DTENSORRT_VERSION=${TENSORRT_VERSION})

    set(TRT_FLAGS
      -DTRT_OUT_DIR=${CMAKE_BINARY_DIR}/tensorrt-oss/bin
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
      -DBUILD_SAMPLES=OFF
    )

    if (JETSON)
      list(APPEND TRT_FLAGS -DCMAKE_C_COMPILER=/usr/bin/cc)
    endif()

	set(TRT_PATCHES_PATH ${CMAKE_BINARY_DIR}/patches/trt)
	set(TRT_PATCHES ${TRT_PATCHES_PATH}/0001-support-for-refinedet.patch)

    # Tensorrt uses its own protobuf version (3.0.0) and this cannot be overrided
    # At best we can set the version it will internally build
    ExternalProject_Add(
      tensorrt-oss
      PREFIX tensorrt-oss
      GIT_REPOSITORY https://github.com/NVIDIA/TensorRT.git
      GIT_TAG ${TENSORRT_VERSION}
      GIT_CONFIG advice.detachedHead=false
	  PATCH_COMMAND git reset --hard && git apply ${TRT_PATCHES}
      CMAKE_ARGS ${TRT_FLAGS}
      INSTALL_COMMAND ""
    )

    # Use our tensorrt-oss header versions first
    include_directories(SYSTEM ${CMAKE_BINARY_DIR}/tensorrt-oss/src/tensorrt-oss/include)
    include_directories(${TENSORRT_INC_DIR})
endif()

if (USE_HTTP_SERVER_OATPP)
    set(OATPP_VERSION "6b4413552acf4ebcd0d3df411d6a2a6fc78cabc1") # pre-1.2.5
    set(OATPP_SWAGGER_VERSION "8b70945f0e9bcf09e16b5cb4be08e752097b0288") # pre-1.2.5
    set(OATPP_ZLIB_VERSION "1846f759b7483e2386236a75728c1cbbb935b2a6") # pre-1.2.5
    ExternalProject_Add(
        oatpp
        PREFIX oatpp
        URL https://github.com/oatpp/oatpp/archive/${OATPP_VERSION}.tar.gz
        URL_HASH SHA256=4517457270d0fb0096ea2796297f6901c619650843d242c2748284d89450217b
        BUILD_IN_SOURCE true
        CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DOATPP_INSTALL=OFF -DOATPP_BUILD_TESTS=OFF
        INSTALL_COMMAND ""
    )
    ExternalProject_Add(
        oatpp-zlib
        PREFIX oatpp-zlib
        URL https://github.com/oatpp/oatpp-zlib/archive/${OATPP_ZLIB_VERSION}.tar.gz
        URL_HASH SHA256=405961fda17ae8fd1be1381e505b0a078b92d11c6f743a01e1827ab95c1ee810
        CMAKE_ARGS
            -DCMAKE_BUILD_TYPE=Release
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
    ExternalProject_Add(
        oatpp-swagger
        PREFIX oatpp-swagger
        URL https://github.com/oatpp/oatpp-swagger/archive/${OATPP_SWAGGER_VERSION}.tar.gz
        URL_HASH SHA256=561a98c54550a703c3c2bde9658ab426d4da92345a4841ba68f5477feece42dc
        BUILD_IN_SOURCE true
        INSTALL_COMMAND ""
        CMAKE_ARGS
            -DCMAKE_BUILD_TYPE=Release
            -DBUILD_SHARED_LIBS=OFF
            -DOATPP_BUILD_TESTS=OFF
            -DOATPP_INSTALL=OFF
            -DOATPP_MODULES_LOCATION=CUSTOM
            -DOATPP_DIR_SRC=${CMAKE_BINARY_DIR}/oatpp/src/oatpp/src
            -DOATPP_DIR_LIB=${CMAKE_BINARY_DIR}/oatpp/src/oatpp/src
        DEPENDS oatpp
    )

    # FIXME(sileht): we should copy res outside build dir and dynamically looks at runtime at different places
    add_definitions("-DOATPP_SWAGGER_RES_PATH=\"${CMAKE_BINARY_DIR}/oatpp-swagger/src/oatpp-swagger/res/\"")

    set(OATPP_LIB_DEPS
        ${CMAKE_BINARY_DIR}/oatpp/src/oatpp/src/liboatpp.a
        ${CMAKE_BINARY_DIR}/oatpp-zlib/src/oatpp-zlib/src/liboatpp-zlib.a
        ${CMAKE_BINARY_DIR}/oatpp-swagger/src/oatpp-swagger/src/liboatpp-swagger.a
        z  # zlib
    )
    include_directories(
        SYSTEM ${CMAKE_BINARY_DIR}/oatpp/src/oatpp/src
        SYSTEM ${CMAKE_BINARY_DIR}/oatpp-zlib/src/oatpp-zlib/src
        SYSTEM ${CMAKE_BINARY_DIR}/oatpp-swagger/src/oatpp-swagger/src
    )
endif()

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
include_directories("${PROJECT_SOURCE_DIR}/src")
include_directories(${CMAKE_BINARY_DIR}/src)
add_subdirectory(src)

set(COMMON_INCLUDE_DIRS ${Boost_INCLUDE_DIRS} ${HTTP_INCLUDE_DIR} ${PROTOBUF_INCLUDE_DIR} ${SPDLOG_INCLUDE_DIR})

if (CUDA_FOUND)
    list(APPEND COMMON_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS})
endif()

set(COMMON_LINK_DIRS
    ${DLIB_LIB_DIR}
    ${TENSORRT_LIB_DIR}
    ${CAFFE_LIB_DIR}
    ${CAFFE2_LIB_DIR}
    ${TF_LIB_DIR}
    ${XGBOOST_LIB_DIR}
    ${TSNE_LIB_DIR}
    ${NCNN_LIB_DIR}
    ${FAISS_LIB_DIR}
    ${TORCH_LIB_DIR}
)

set(COMMON_LINK_LIBS
    ddetect ${DLIB_LIB_DEPS} ${TENSORRT_LIBS} ${CUDA_LIB_DEPS} glog gflags ${OpenCV_LIBS} curlpp curl ${Boost_LIBRARIES} archive
    ${PROTOBUF_LIB_DEPS}
    ${HTTP_LIB_DEPS}
    ${SPDLOG_LIB_DEPS}
    ${OATPP_LIB_DEPS}
    ${CAFFE_LIB_DEPS}
    ${CAFFE2_LIB_DEPS}
    ${TF_LIB_DEPS}
    ${XGBOOST_LIB_DEPS}
    ${TSNE_LIB_DEPS}
    ${NCNN_LIB_DEPS}
    ${FAISS_LIB_DEPS}
    ${TORCH_LIB_DEPS}
)
if (USE_HDF5)
    list(APPEND COMMON_LINK_LIBS hdf5_cpp)
endif()

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


find_program(CLANG_FORMAT_BIN clang-format-10 clang-format)
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

# status
message(STATUS "RELEASE:               ${RELEASE}")
message(STATUS "WARNING:               ${WARNING}")
message(STATUS "USE_COMMAND_LINE:      ${USE_COMMAND_LINE}")
message(STATUS "USE_JSON_API:          ${USE_JSON_API}")
message(STATUS "USE_HTTP_SERVER:       ${USE_HTTP_SERVER}")
message(STATUS "USE_HTTP_SERVER_OATPP: ${USE_HTTP_SERVER_OATPP}")
message(STATUS "BUILD_SPDLOG:          ${BUILD_SPDLOG}")
message(STATUS "BUILD_PROTOBUF:        ${BUILD_PROTOBUF}")
message(STATUS "BUILD_TESTS:           ${BUILD_TESTS}")
message(STATUS "BUILD_TOOLS:           ${BUILD_TOOLS}")
message(STATUS "USE_CAFFE:             ${USE_CAFFE}")
message(STATUS "USE_CAFFE_CPU_ONLY:    ${USE_CAFFE_CPU_ONLY}")
message(STATUS "USE_CAFFE_DEBUG:       ${USE_CAFFE_DEBUG}")
message(STATUS "USE_CAFFE2:            ${USE_CAFFE2}")
message(STATUS "USE_CAFFE2_CPU_ONLY:   ${USE_CAFFE2_CPU_ONLY}")
message(STATUS "USE_TORCH:             ${USE_TORCH}")
message(STATUS "USE_TORCH_CPU_ONLY:    ${USE_TORCH_CPU_ONLY}")
message(STATUS "USE_SIMSEARCH:         ${USE_SIMSEARCH}")
message(STATUS "USE_TF:                ${USE_TF}")
message(STATUS "USE_TF_CPU_ONLY:       ${USE_TF_CPU_ONLY}")
message(STATUS "USE_NCNN:              ${USE_NCNN}")
message(STATUS "USE_HDF5:              ${USE_HDF5}")
message(STATUS "USE_TENSORRT:          ${USE_TENSORRT}")
message(STATUS "USE_DLIB:              ${USE_DLIB}")
message(STATUS "USE_DLIB_CPU_ONLY:     ${USE_DLIB_CPU_ONLY}")
message(STATUS "USE_ANNOY:             ${USE_ANNOY}")
message(STATUS "USE_FAISS:             ${USE_FAISS}")
message(STATUS "USE_FAISS_CPU_ONLY:    ${USE_FAISS_CPU_ONLY}")
message(STATUS "USE_CUDNN:             ${USE_CUDNN}")
message(STATUS "USE_XGBOOST:           ${USE_XGBOOST}")
message(STATUS "USE_XGBOOST_CPU_ONLY:  ${USE_XGBOOST_CPU_ONLY}")
message(STATUS "USE_TSNE:              ${USE_TSNE}")
message(STATUS "USE_BOOST_BACKTRACE:   ${USE_BOOST_BACKTRACE}")
