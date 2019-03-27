cmake_minimum_required(VERSION 2.8.8)
project(deepdetect)

option(USE_CAFFE2 "build caffe2 backend")
option(USE_TF "use TF backend")
option(USE_NCNN "use NCNN backend")
option(USE_HDF5 "use HDF5" ON)
option(USE_CAFFE "use Caffe backend" ON)

if (USE_CAFFE)
  add_definitions(-DUSE_CAFFE)
endif()

if (USE_TF AND USE_CAFFE2)
  message(FATAL_ERROR "Building with tensorflow AND caffe2 causes conflict
(The two libraries have their own protobuf embed)")
endif()

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

include(ProcessorCount)
ProcessorCount(N)
include(ExternalProject)

set (deepdetect_VERSION_MAJOR 0)
set (deepdetect_VERSION_MINOR 1)

# options
OPTION(BUILD_TESTS "Should the tests be built")
OPTION(BUILD_TOOLS "Should the tools be built")
 
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

set(CMAKE_CXX_FLAGS "-g -Wall -Wextra -fopenmp -O2 -fPIC -std=c++11  -DUSE_OPENCV -DUSE_LMDB")
  
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
  set(TENSORFLOW_CC_DIR ${CMAKE_BINARY_DIR}/tensorflow_cc/src/tensorflow_cc/tensorflow_cc/build/)
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
find_package(Boost 1.54 REQUIRED COMPONENTS filesystem thread system iostreams)

# annoy
if (USE_CAFFE)
  if (USE_SIMSEARCH)
    message(STATUS "Fetching Annoy")
    add_definitions(-DUSE_SIMSEARCH)
    ExternalProject_Add(
      annoy
      PREFIX annoy
      INSTALL_DIR ${CMAKE_BINARY_DIR}
      DOWNLOAD_COMMAND git clone https://github.com/spotify/annoy.git
      CONFIGURE_COMMAND python setup.py build
      BUILD_COMMAND ""
      INSTALL_COMMAND ""
      BUILD_IN_SOURCE 1
      )
    
    set(ANNOY_INCLUDE_DIR ${CMAKE_BINARY_DIR}/annoy/src/annoy/src/)
    include_directories("${ANNOY_INCLUDE_DIR}")
  endif()
else()
  set(USE_SIMSEARCH OFF)
endif()
  
# optional packages
if (USE_CPU_ONLY)
  set(USE_CAFFE_CPU_ONLY ON)
  if (USE_TF)
    set(USE_TF_CPU_ONLY ON)
  endif()
  if (USE_DLIB)
    set(USE_DLIB_CPU_ONLY ON)
  endif()
else()
    if (USE_DLIB AND NOT USE_DLIB_CPU_ONLY)
        message(STATUS "Dlib enabled with CUDA requires cuDNN - enabling cuDNN")
        set(USE_CUDNN ON)
    endif()
include(cmake/Cuda.cmake) # cuda + cudnn
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
    set(DLIB_RELEASE_VERSION v19.17)
    message(STATUS "Using Dlib version ${DLIB_RELEASE_VERSION}")
    add_definitions(-DUSE_DLIB)
    set(DLIB_LIB_DEPS -ldlib)

    if (CUDA_FOUND AND NOT USE_DLIB_CPU_ONLY)
        if (NOT USE_CUDNN)
            message(FATAL_ERROR "Dlib with CUDA must also enable cuDNN. Use -DUSE_CUDNN=ON")
        endif()
        ExternalProject_Add(
          dlib
          PREFIX dlib
          INSTALL_DIR ${CMAKE_BINARY_DIR}
          URL https://github.com/davisking/dlib/archive/${DLIB_RELEASE_VERSION}.tar.gz
          CONFIGURE_COMMAND cd dlib && mkdir -p build && cd build && cmake .. -DDLIB_NO_GUI_SUPPORT=ON -DDLIB_USE_CUDA=ON -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/dlib/build && cmake --build . --config Release --target install
          BUILD_COMMAND ""
          INSTALL_COMMAND ""
          BUILD_IN_SOURCE 1
        )
        set(DLIB_INC_DIR ${CMAKE_BINARY_DIR}/dlib/build/include)
        set(DLIB_LIB_DIR ${CMAKE_BINARY_DIR}/dlib/build/lib)
    else()
        ExternalProject_Add(
            dlib
            PREFIX dlib
            INSTALL_DIR ${CMAKE_BINARY_DIR}
            URL https://github.com/davisking/dlib/archive/${DLIB_RELEASE_VERSION}.tar.gz
            CONFIGURE_COMMAND cd dlib && mkdir -p build && cd build && cmake .. -DDLIB_NO_GUI_SUPPORT=ON -DDLIB_USE_CUDA=OFF -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/dlib/build && cmake --build . --config Release --target install
            BUILD_COMMAND ""
            INSTALL_COMMAND ""
            BUILD_IN_SOURCE 1
        )
        set(DLIB_INC_DIR ${CMAKE_BINARY_DIR}/dlib/build/include)
        set(DLIB_LIB_DIR ${CMAKE_BINARY_DIR}/dlib/build/lib)
    endif()

    include_directories("${DLIB_INC_DIR}")
    message(STATUS "Dlib will be built")
endif()

set(CAFFE_DD_USER jolibrain)
set(CAFFE_DD_BRANCH master)

if (USE_CAFFE2)

  add_definitions(-DUSE_CAFFE2)
  list(APPEND CAFFE2_LIB_DEPS -lcaffe2)
  if (CUDA_FOUND AND NOT USE_CPU_ONLY)
    list(APPEND CAFFE2_LIB_DEPS -lcaffe2_gpu)
  endif()

  if (NOT USE_CAFFE)
    list(APPEND CAFFE2_LIB_DEPS -lprotobuf)
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
    third_party/protobuf
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
    INSTALL_DIR ${CMAKE_BINARY_DIR}
    GIT_REPOSITORY https://github.com/pytorch/pytorch.git
    GIT_SUBMODULES ${PYTORCH_SUBMODULES}
    UPDATE_DISCONNECTED 1
    GIT_TAG ${PYTORCH_SUPPORTED_COMMIT}
    PATCH_COMMAND test -f ${PYTORCH_COMPLETE} && echo Skipping || echo cp modules/detectron/*_op.* caffe2/operators | bash && cp ${CAFFE2_OPS} caffe2/operators && git am ${PYTORCH_PATCHES}
    CONFIGURE_COMMAND test -f ${PYTORCH_COMPLETE} && echo Skipping || cmake ../pytorch ${PYTORCH_FLAGS}
    BUILD_COMMAND test -f ${PYTORCH_COMPLETE} && echo Skipping || make -j${N}
    INSTALL_COMMAND ""
    )

  # Compilation path
  set(PROTOBUF_INCLUDE_DIR ${CMAKE_BINARY_DIR}/protobuf/include)
  set(PROTOBUF_LIB_DIR ${CMAKE_BINARY_DIR}/protobuf/lib)
  if (NOT EXISTS ${CMAKE_BINARY_DIR}/protobuf)
    execute_process(COMMAND mkdir -p ${CMAKE_BINARY_DIR}/protobuf)
    execute_process(COMMAND ln -s ${PYTORCH_PATH}/third_party/protobuf/src ${PROTOBUF_INCLUDE_DIR})
    execute_process(COMMAND ln -s ${PYTORCH_PATH}-build/lib ${PROTOBUF_LIB_DIR})
    execute_process(COMMAND ln -s ${PYTORCH_PATH}-build/bin ${CMAKE_BINARY_DIR}/protobuf/bin)
  endif()

  # Detectron
  set(DETECTRON_COMPLETE ${CMAKE_BINARY_DIR}/CMakeFiles/detectron-complete)
  ExternalProject_Add(
    detectron
    PREFIX detectron
    INSTALL_DIR ${CMAKE_BINARY_DIR}
    GIT_REPOSITORY https://github.com/facebookresearch/Detectron
    UPDATE_DISCONNECTED 1
    GIT_TAG ${DETECTRON_SUPPORTED_COMMIT}
    PATCH_COMMAND test -f ${DETECTRON_COMPLETE} && echo Skipping || git am ${DETECTRON_PATCHES}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    )
  add_dependencies(detectron pytorch)

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
  # protobuf with tf
  set(PROTOBUF_INCLUDE_DIR ${CMAKE_BINARY_DIR}/protobuf/include/)
  set(PROTOBUF_LIB_DIR ${CMAKE_BINARY_DIR}/protobuf/lib/)
  include_directories("${PROTOBUF_INCLUDE_DIR}")

  # Tensorflow
  message(STATUS "Fetching Tensorflow")
  add_definitions(-DUSE_TF)
  set(TF_LIB_DEPS -ltensorflow_cc)
  if (CUDA_FOUND AND NOT USE_TF_CPU_ONLY)
    ExternalProject_Add(
      tensorflow_cc
      PREFIX tensorflow_cc
      INSTALL_DIR ${CMAKE_BINARY_DIR}
      DOWNLOAD_COMMAND git clone https://github.com/beniz/tensorflow_cc.git
      CONFIGURE_COMMAND cd tensorflow_cc && mkdir -p build && cd build && cmake -DTENSORFLOW_STATIC=OFF -DTENSORFLOW_SHARED=ON .. && make && ln -s ${CMAKE_BINARY_DIR}/tensorflow_cc/src/tensorflow_cc/tensorflow_cc/build/tensorflow/tensorflow/contrib/makefile/gen/protobuf ${CMAKE_BINARY_DIR}/protobuf
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
      CONFIGURE_COMMAND cd tensorflow_cc && mkdir -p build && cd build && cmake -DTENSORFLOW_STATIC=OFF -DTENSORFLOW_SHARED_CPU=ON .. && make && ln -s ${CMAKE_BINARY_DIR}/tensorflow_cc/src/tensorflow_cc/tensorflow_cc/build/tensorflow/tensorflow/contrib/makefile/gen/protobuf ${CMAKE_BINARY_DIR}/protobuf
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
elseif(USE_CAFFE)
  message(STATUS "Configuring customized caffe")

  if (USE_DD_SYSLOG)
    set(USE_SYSLOG 1)
  else()
    set(USE_SYSLOG 0)
  endif()

  # Set protoc path
  if (CUDA_FOUND
      AND NOT USE_CAFFE_CPU_ONLY
      AND NOT (HAVE_CUDNN AND USE_CUDNN)
      AND USE_TF)
    list(APPEND CAFFE_DD_MAKE
      ${CMAKE_COMMAND} -E env PATH=${TENSORFLOW_CC_DIR}/tensorflow/tensorflow/contrib/makefile/gen/protobuf/bin:$ENV{PATH}
      )
  elseif (USE_TF OR USE_CAFFE2)
    list(APPEND CAFFE_DD_MAKE
      ${CMAKE_COMMAND} -E env PATH=${CMAKE_BINARY_DIR}/protobuf/bin:$ENV{PATH}
      )
  endif()
  list(APPEND CAFFE_DD_MAKE make)

  # Set cuda arch
  if (CUDA_FOUND AND NOT USE_CAFFE_CPU_ONLY)
    if (NOT CUDA_ARCH)
      string(REPLACE ";" " " CUDA_ARCH "${CUDA_NVCC_FLAGS}")
    endif()
    list(APPEND CAFFE_DD_MAKE_ARG CUDA_ARCH=${CUDA_ARCH})
  endif()

  # Set config file
  if (CUDA_FOUND AND NOT USE_CAFFE_CPU_ONLY)
    if (HAVE_CUDNN AND USE_CUDNN)
      if(JETSON)
	set(CAFFE_DD_CONFIG_FILE Makefile.config.gpu.cudnn.jetson)
      else()
	set(CAFFE_DD_CONFIG_FILE Makefile.config.gpu.cudnn)
      endif()
    else() # HAVE_CUDNN
      set(CAFFE_DD_CONFIG_FILE Makefile.config.gpu)
    endif() # HAVE_CUDNN
  else() # CUDA_FOUND
    add_definitions(-DUSE_CAFFE_CPU_ONLY)
    set(CAFFE_DD_CONFIG_FILE Makefile.config.cpu)
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

  list(APPEND CAFFE_DD_CONFIGURE_COMMAND
    ln -sf ${CAFFE_DD_CONFIG_FILE} Makefile.config &&
    echo "OPENCV_VERSION:=${OPENCV_VERSION}" >> Makefile.config &&
    echo "USE_SYSLOG:=${USE_SYSLOG}" >> Makefile.config &&
    echo "DEBUG:=${CAFFE_DEBUG}" >> Makefile.config &&
    ${CAFFE_DD_MAKE} ${CAFFE_DD_MAKE_ARG} -j${N}
    )

  ExternalProject_Add(
    caffe_dd
    PREFIX caffe_dd
    INSTALL_DIR ${CMAKE_BINARY_DIR}
    URL https://github.com/${CAFFE_DD_USER}/caffe/archive/${CAFFE_DD_BRANCH}.tar.gz
    CONFIGURE_COMMAND ${CAFFE_DD_CONFIGURE_COMMAND}
    INSTALL_COMMAND ""
    BUILD_IN_SOURCE 1
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
    if (CUDA_FOUND)
      set(CAFFE_INC_DIR ${CMAKE_BINARY_DIR}/caffe_dd/src/caffe_dd/include ${CMAKE_BINARY_DIR}/caffe_dd/src/caffe_dd/build/src ${CUDA_INCLUDE_DIRS} ${PROTOBUF_INCLUDE_DIR})
      set(CAFFE_LIB_DIR ${CMAKE_BINARY_DIR}/caffe_dd/src/caffe_dd/build/lib ${CMAKE_BINARY_DIR}/caffe_dd/src/caffe_dd/build/src ${HDF5_LIB} ${PROTOBUF_LIB_DIR})
    else()
      set(CAFFE_INC_DIR ${CMAKE_BINARY_DIR}/caffe_dd/src/caffe_dd/include ${CMAKE_BINARY_DIR}/caffe_dd/src/caffe_dd/build/src /usr/include/hdf5/serial ${PROTOBUF_INCLUDE_DIR})
      set(CAFFE_LIB_DIR $ENV{HOME}/lib ${CMAKE_BINARY_DIR}/caffe_dd/src/caffe_dd/build/lib ${CMAKE_BINARY_DIR}/caffe_dd/src/caffe_dd/build/src ${HDF5_LIB} ${PROTOBUF_LIB_DIR})
    endif()
  else()
    if (CUDA_FOUND)
      set(CAFFE_INC_DIR ${CMAKE_BINARY_DIR}/caffe_dd/src/caffe_dd/include ${CMAKE_BINARY_DIR}/caffe_dd/src/caffe_dd/build/src ${CUDA_INCLUDE_DIRS} ${PROTOBUF_INCLUDE_DIR})
      set(CAFFE_LIB_DIR ${CMAKE_BINARY_DIR}/caffe_dd/src/caffe_dd/build/lib ${CMAKE_BINARY_DIR}/caffe_dd/src/caffe_dd/build/src ${PROTOBUF_LIB_DIR})
    else()
      set(CAFFE_INC_DIR ${CMAKE_BINARY_DIR}/caffe_dd/src/caffe_dd/include ${CMAKE_BINARY_DIR}/caffe_dd/src/caffe_dd/build/src ${PROTOBUF_INCLUDE_DIR})
      set(CAFFE_LIB_DIR $ENV{HOME}/lib ${CMAKE_BINARY_DIR}/caffe_dd/src/caffe_dd/build/lib ${CMAKE_BINARY_DIR}/caffe_dd/src/caffe_dd/build/src ${PROTOBUF_LIB_DIR})
    endif()
  endif()
endif()

# Caffe dependencies
if (USE_CAFFE)
  if (USE_HDF5)
    set(CAFFE_LIB_DEPS -lleveldb -lsnappy -llmdb -lhdf5_hl -lhdf5 -lopenblas -lcaffe -lprotobuf)
  else()
    set(CAFFE_LIB_DEPS -lleveldb -lsnappy -llmdb -lopenblas -lcaffe -lprotobuf)
  endif()
endif()

if (USE_TF)
  if (USE_CAFFE)
    add_dependencies(caffe_dd tensorflow_cc)
  endif()
endif()

if (USE_CAFFE2)
  if (USE_CAFFE)
    add_dependencies(caffe_dd pytorch)
    add_dependencies(caffe_dd detectron)
  else()
    set(CAFFE2_INC_DIR ${PROTOBUF_INCLUDE_DIR})
    set(CAFFE2_LIB_DIR ${PROTOBUF_LIB_DIR})
  endif()
endif()

# NCNN
if (USE_NCNN)
  message(STATUS "Configuring NCNN")
  add_definitions(-DUSE_NCNN)
  set(NCNN_LIB_DEPS ${CMAKE_BINARY_DIR}/ncnn/src/ncnn/build/src/libncnn.a)
  set(NCNN_COMMIT_HASH "569be5a19651373cca68933f6430c781d7d6fd97")
  if (RPI3)
    ExternalProject_Add(
      ncnn
      PREFIX ncnn
      INSTALL_DIR ${CMAKE_BINARY_DIR}
      DOWNLOAD_COMMAND git clone --recursive https://github.com/jolibrain/ncnn.git
      CONFIGURE_COMMAND mkdir -p build && cd build && cmake .. -DPI3=ON -DCMAKE_TOOLCHAIN_FILE=../toolchains/pi3.toolchain.cmake
      BUILD_COMMAND cd build/ && make -j${N}
      INSTALL_COMMAND ""
      BUILD_IN_SOURCE 1
    )
  else ()
    ExternalProject_Add(
      ncnn
      PREFIX ncnn
      INSTALL_DIR ${CMAKE_BINARY_DIR}
      DOWNLOAD_COMMAND git clone --recursive https://github.com/jolibrain/ncnn.git
      CONFIGURE_COMMAND mkdir -p build && cd build && cmake ..
      BUILD_COMMAND cd build/ && make -j${N}
      INSTALL_COMMAND ""
      BUILD_IN_SOURCE 1
    )
  endif()
  set(NCNN_INC_DIR ${CMAKE_BINARY_DIR}/ncnn/src/ncnn/src ${CMAKE_BINARY_DIR}/ncnn/src/ncnn/build/src)
  set(NCNN_LIB_DIR ${CMAKE_BINARY_DIR}/ncnn/src/ncnn/build/src)
endif()

# XGBoost
if (USE_XGBOOST)
  message(STATUS "Configuring XGBoost")
  add_definitions(-DUSE_XGBOOST)
  set(XGBOOST_LIB_DEPS -Wl,--whole-archive ${CMAKE_BINARY_DIR}/xgboost/src/xgboost/lib/libxgboost.a -Wl,--no-whole-archive ${CMAKE_BINARY_DIR}/xgboost/src/xgboost/build/dmlc-core/libdmlc.a ${CMAKE_BINARY_DIR}/xgboost/src/xgboost/build/librabit.a)

  if (CUDA_FOUND AND USE_XGBOOST_GPU)
    add_definitions(-DUSE_XGBOOST_GPU)
    list(APPEND XGBOOST_LIB_DEPS ${CMAKE_BINARY_DIR}/xgboost/src/xgboost/build/CMakeFiles/gpuxgboost.dir/plugin/updater_gpu/src/gpuxgboost_generated_updater_gpu.cu.o ${CMAKE_BINARY_DIR}/xgboost/src/xgboost/build/nccl/libnccl.a)
#    set(XGB_NVCC_FLAGS,"--expt-extended-lambda;-gencode arch=compute_35,code=compute_35")
    ExternalProject_Add(
      xgboost
      PREFIX xgboost
      INSTALL_DIR ${CMAKE_BINARY_DIR}
      DOWNLOAD_COMMAND git clone --recursive https://github.com/dmlc/xgboost.git
      CONFIGURE_COMMAND wget https://github.com/NVlabs/cub/archive/1.6.4.zip && unzip 1.6.4.zip && rm 1.6.4.zip && mkdir -p build && cd build &&
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
  set(TSNE_LIB_DEPS -ltsne_multicore -lglog)
  ExternalProject_Add(
      Multicore-TSNE
      PREFIX Multicore-TSNE
      INSTALL_DIR ${CMAKE_BINARY_DIR}
      DOWNLOAD_COMMAND git clone https://github.com/beniz/Multicore-TSNE.git
      CONFIGURE_COMMAND cd multicore_tsne && mkdir -p build && cd build && cmake ..
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
include_directories(${CAFFE_INC_DIR} ${CAFFE2_INC_DIR} ${XGBOOST_INC_DIR} ${TSNE_INC_DIR})
include_directories(${CMAKE_SOURCE_DIR}/src/backends/caffe ${CMAKE_SOURCE_DIR}/backends/xgb ${CMAKE_SOURCE_DIR}/backends/tf ${CMAKE_SOURCE_DIR}/backends/dlib ${CMAKE_SOURCE_DIR}/backends/tsne)

if (USE_NCNN)
  include_directories(${NCNN_INC_DIR})
endif()

# main library, main & tests
include_directories ("${PROJECT_SOURCE_DIR}/src")
add_subdirectory (src)

# Flags used by the subdirectories
set(COMMON_INCLUDE_DIRS ${Boost_INCLUDE_DIRS})
set(COMMON_LINK_DIRS
  ${CAFFE_LIB_DIR}
  ${CAFFE2_LIB_DIR}
  ${TF_LIB_DIR}
  ${XGBOOST_LIB_DIR}
  ${TSNE_LIB_DIR}
  ${NCNN_LIB_DIR}
  ${DLIB_LIB_DIR})
if (USE_HDF5)
  set(COMMON_LINK_LIBS
    ddetect ${CUDA_LIB_DEPS} glog gflags ${OpenCV_LIBS} curlpp curl hdf5_cpp ${Boost_LIBRARIES} archive
    ${CAFFE_LIB_DEPS}
    ${CAFFE2_LIB_DEPS}
    ${TF_LIB_DEPS}
    ${XGBOOST_LIB_DEPS}
    ${TSNE_LIB_DEPS}
    ${NCNN_LIB_DEPS}
    ${DLIB_LIB_DEPS})
else()
  set(COMMON_LINK_LIBS
    ddetect ${CUDA_LIB_DEPS} glog gflags ${OpenCV_LIBS} curlpp curl ${Boost_LIBRARIES} archive
    ${CAFFE_LIB_DEPS}
    ${CAFFE2_LIB_DEPS}
    ${TF_LIB_DEPS}
    ${XGBOOST_LIB_DEPS}
    ${TSNE_LIB_DEPS}
    ${NCNN_LIB_DEPS}
    ${DLIB_LIB_DEPS})
endif()
set(HTTP_LINK_LIBS cppnetlib-uri crypto ssl)

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

# unit testing
if (BUILD_TOOLS)
  add_subdirectory(tools)
endif()

# status
message(STATUS "Build Tests          : ${BUILD_TESTS}")
message(STATUS "Caffe DEBUG          : ${USE_CAFFE_DEBUG}")
