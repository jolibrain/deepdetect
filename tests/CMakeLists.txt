include_directories(${COMMON_INCLUDE_DIRS})
link_directories(${COMMON_LINK_DIRS})
link_directories("/opt/homebrew/lib/")
link_directories("/usr/local/lib") # for gtest

find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# Optional headers can be passed (stored in ARGN)
function (REGISTER_TEST _NAME _FILE)
  if (NOT TARGET ${_NAME})
    add_executable(${_NAME} ${_FILE} ${ARGN})
    add_dependencies(${_NAME} protobuf)
    target_link_libraries(${_NAME} ${COMMON_LINK_LIBS} gtest gtest_main ${OATPP_LIB_DEPS})
  endif()

  add_test(NAME ${_NAME} COMMAND ${_NAME} "--gtest_filter=-*multigpu*")
endfunction ()

function (REGISTER_TEST_MULTIGPU _NAME _FILE)
  if (NOT TARGET ${_NAME})
    add_executable(${_NAME} ${_FILE} ${ARGN})
    add_dependencies(${_NAME} protobuf)
    target_link_libraries(${_NAME} ${COMMON_LINK_LIBS} gtest gtest_main)
  endif()

  add_test(NAME ${_NAME}_multigpu COMMAND ${_NAME} "--gtest_filter=*multigpu*")
endfunction ()

function (DOWNLOAD_DATASET _NAME _URL _WORKDIR _DEST _IN)
  set(_OUT "${ARGV5}") # Optional
  set(_FORCE_RECREATE OFF)
  set(_DATASET_READY OFF)
  if (IS_ABSOLUTE "${_WORKDIR}")
    set(_DATASET_WORKDIR "${_WORKDIR}")
  else()
    set(_DATASET_WORKDIR "${CMAKE_BINARY_DIR}/${_WORKDIR}")
  endif()
  file(MAKE_DIRECTORY "${_DATASET_WORKDIR}")

  if (EXISTS "${_DATASET_WORKDIR}/${_IN}")
    set(_DATASET_READY ON)
  endif()
  if (NOT "${_OUT}" STREQUAL "" AND EXISTS "${_DATASET_WORKDIR}/${_OUT}")
    set(_DATASET_READY ON)
  endif()

  if (EXISTS "${_DATASET_WORKDIR}/${_DEST}")
    file(SIZE "${_DATASET_WORKDIR}/${_DEST}" _DATASET_SIZE)
    if (_DATASET_SIZE EQUAL 0)
      file(REMOVE "${_DATASET_WORKDIR}/${_DEST}")
    endif()
  endif()

  if (_DATASET_READY)
    message(STATUS "Reusing prepared ${_NAME} dataset")
  elseif (EXISTS "${_DATASET_WORKDIR}/${_DEST}")
    message(STATUS "Reusing downloaded ${_NAME} dataset")
  else()
    message(STATUS "Downloading ${_NAME}")
    file(DOWNLOAD ${_URL} ${_DATASET_WORKDIR}/${_DEST}
      STATUS _DOWNLOAD_STATUS)
    list(GET _DOWNLOAD_STATUS 0 _DOWNLOAD_STATUS_CODE)
    list(GET _DOWNLOAD_STATUS 1 _DOWNLOAD_STATUS_MSG)
    if (NOT _DOWNLOAD_STATUS_CODE EQUAL 0)
      message(FATAL_ERROR
        "Download ${_NAME} failed: ${_DOWNLOAD_STATUS_MSG}")
    endif()
    set(_FORCE_RECREATE ON)
  endif()
  if (_DATASET_READY)
    message(STATUS "Skipping already prepared ${_DATASET_WORKDIR}/${_IN}")
  elseif ("${_IN}" STREQUAL "${_DEST}")
    message(STATUS "Skipping not compressed ${_DATASET_WORKDIR}/${_IN}")
  else()
    if (NOT ${_FORCE_RECREATE} AND EXISTS "${_DATASET_WORKDIR}/${_IN}")
      message(STATUS "Reusing uncompressed ${_DATASET_WORKDIR}/${_DEST}")
    else()
      string(FIND ${_DEST} ".tar." _tar_index)
      if (${_tar_index} EQUAL -1)
        execute_process(
          WORKING_DIRECTORY ${_DATASET_WORKDIR}
          COMMAND bzip2 -d ${_DEST}
          RESULT_VARIABLE _UNCOMPRESS_RESULT
        )
      else()
        execute_process(
          WORKING_DIRECTORY ${_DATASET_WORKDIR}
          COMMAND ${CMAKE_COMMAND} -E tar xf ${_DEST}
          RESULT_VARIABLE _UNCOMPRESS_RESULT
        )
      endif()
      if (NOT _UNCOMPRESS_RESULT EQUAL 0)
        message(FATAL_ERROR "Untar ${_DATASET_WORKDIR}/${_DEST} failed")
      endif()
      set(_FORCE_RECREATE ON)
    endif()
  endif()
  if (NOT "${_OUT}" STREQUAL "")
    if (EXISTS "${_DATASET_WORKDIR}/${_OUT}")
      message(STATUS "Reusing already present ${_DATASET_WORKDIR}/${_OUT}")
    else()
      execute_process(
        WORKING_DIRECTORY ${_DATASET_WORKDIR}
        COMMAND ${CMAKE_COMMAND} -E create_symlink ${_IN} ${_OUT}
        RESULT_VARIABLE _RENAME_RESULT
        )
      if (NOT _RENAME_RESULT EQUAL 0)
        message(FATAL_ERROR "create_symlink ${_DATASET_WORKDIR}/${_IN} failed")
      endif()
    endif()
  endif()
endfunction()

function (DOWNLOAD_FILES _BASEURL _DEST)
  set(_files "${ARGN}")
  if (IS_ABSOLUTE "${_DEST}")
    set(_FILES_DEST "${_DEST}")
  else()
    set(_FILES_DEST "${CMAKE_BINARY_DIR}/${_DEST}")
  endif()
  file(MAKE_DIRECTORY "${_FILES_DEST}")
  foreach(_file IN LISTS _files)
    if (EXISTS "${_FILES_DEST}/${_file}")
      file(SIZE "${_FILES_DEST}/${_file}" _FILE_SIZE)
      if (_FILE_SIZE EQUAL 0)
        file(REMOVE "${_FILES_DEST}/${_file}")
      endif()
    endif()
    if (EXISTS "${_FILES_DEST}/${_file}")
      message(STATUS "Reuse downloaded ${_FILES_DEST}/${_file}")
    else()
      message(STATUS "Downloading ${_FILES_DEST}/${_file}")
      file(DOWNLOAD ${_BASEURL}/${_file} ${_FILES_DEST}/${_file}
        STATUS _DOWNLOAD_STATUS)
      list(GET _DOWNLOAD_STATUS 0 _DOWNLOAD_STATUS_CODE)
      list(GET _DOWNLOAD_STATUS 1 _DOWNLOAD_STATUS_MSG)
      if (NOT _DOWNLOAD_STATUS_CODE EQUAL 0)
        message(FATAL_ERROR
          "Download ${_FILES_DEST}/${_file} failed: ${_DOWNLOAD_STATUS_MSG}")
      endif()
    endif()
  endforeach()
endfunction()

find_package(PythonInterp 3 REQUIRED)

REGISTER_TEST(ut_common ut-common.cc)

if (USE_JSON_API)
  REGISTER_TEST(ut_apidata ut-apidata.cc)
  REGISTER_TEST(ut_dto ut-dto.cc)
  REGISTER_TEST(ut_embedded_runtime ut-embedded-runtime.cc)
  REGISTER_TEST(ut_retired_backends ut-retired-backends.cc)
endif()

DOWNLOAD_DATASET(
  "CSV (test) dataset"
  "http://www.deepdetect.com/dd/examples/all/forest_type/train.csv.tar.bz2"
  "examples/all/forest_type"
  "train.csv.tar.bz2"
  "train.csv"
  )
DOWNLOAD_DATASET(
  "SVM (test) dataset"
  "http://www.deepdetect.com/dd/examples/all/farm_ads/farm-ads.svm"
  "examples/all/farm_ads"
  "farm-ads.svm"
  "farm-ads.svm"
  )

DOWNLOAD_DATASET(
  "Text (test) dataset"
  "http://www.deepdetect.com/dd/examples/all/n20/news20.tar.bz2"
  "examples/all/n20"
  "news20.tar.bz2"
  "news20"
  )
DOWNLOAD_DATASET(
  "Test videos"
  "http://www.deepdetect.com/dd/examples/all/video/video1.tar.gz"
  "examples/all/video"
  "video1.tar.gz"
  "video1.mp4"
)
DOWNLOAD_DATASET(
  "Test images"
  "http://www.deepdetect.com/dd/examples/all/images/all_test_images0.tar.gz"
  "examples/all/images"
  "all_test_images0.tar.gz"
  "face.jpg"
)

set(SINUS_EXAMPLE_PATH ${CMAKE_SOURCE_DIR}/examples/all/sinus/)
set(SINUS_EXAMPLE_OUT "${CMAKE_BINARY_DIR}/examples/all/sinus")
set(SINUS_EXAMPLE_TRAIN_OUT "${SINUS_EXAMPLE_OUT}/train")
if (NOT EXISTS ${SINUS_EXAMPLE_TRAIN_OUT})
  message(STATUS "Generating sinus data for time series tests")
  file(MAKE_DIRECTORY ${SINUS_EXAMPLE_OUT})
  file(COPY ${SINUS_EXAMPLE_PATH}/gen.py DESTINATION ${SINUS_EXAMPLE_OUT})
  execute_process(
    COMMAND ${PYTHON_EXECUTABLE} ./gen.py
    WORKING_DIRECTORY ${SINUS_EXAMPLE_OUT}
    RESULT_VARIABLE _SINUS_GEN_RESULT
  )
  if (NOT _SINUS_GEN_RESULT EQUAL 0)
    message(FATAL_ERROR "Sinus ./gen.py failed")
  endif()
else()
  message(STATUS "Reusing generated sinu data for time series tests")
endif()

DOWNLOAD_DATASET(
  "Regression test model"
  "http://www.deepdetect.com/dd/examples/all/sflare/flare.csv"
  "examples/all/sflare"
  "flare.csv"
  "flare.csv"
)

if (USE_HTTP_SERVER_OATPP)
  REGISTER_TEST(ut_oatpp ut-oatpp.cc ut-oatpp.h)
  target_link_libraries(ut_oatpp ${OATPP_LIB_DEPS} ${CMAKE_BINARY_DIR}/oatpp/src/oatpp/src/liboatpp-test.a)
endif()

if (USE_DLIB)
  DOWNLOAD_DATASET(
    "Dlib test face model"
    "http://dlib.net/files/mmod_human_face_detector.dat.bz2"
    "examples/dlib/face"
    "mmod_human_face_detector.dat.bz2"
    "mmod_human_face_detector.dat"
  )
  DOWNLOAD_FILES(
    "https://www.deepdetect.com/models/tf"
    "examples/dlib/face/"
    "grace_hopper.jpg"
    "cat.jpg"
  )

  DOWNLOAD_DATASET(
    "Dlib test object detector model"
    "http://dlib.net/files/mmod_front_and_rear_end_vehicle_detector.dat.bz2"
    "examples/dlib/obj"
    "mmod_front_and_rear_end_vehicle_detector.dat"
    "mmod_front_and_rear_end_vehicle_detector.dat"
  )

  DOWNLOAD_FILES(
    "https://github.com/davisking/dlib/raw/master/examples/"
    "examples/dlib/obj/"
    "mmod_cars_test_image2.jpg"
  )

  REGISTER_TEST(ut_dlibapi ut-dlibapi.cc)
endif()

if (USE_NCNN)
  DOWNLOAD_DATASET(
    "NCNN test Squeezenet SSD model"
    "https://www.deepdetect.com/dd/examples/ncnn/squeezenet-ssd-ncnn-caffe.tar.gz"
    "examples/ncnn"
    "squeezenet-ssd-ncnn-caffe.tar.gz"
    "squeezenet_ssd_ncnn"
    )

    DOWNLOAD_DATASET(
    "NCNN test Squeezenet classification model"
    "https://www.deepdetect.com/dd/examples/ncnn/squeezenet-ncnn-caffe.tar.gz"
    "examples/ncnn"
    "squeezenet-ncnn-caffe.tar.gz"
    "squeezenet_ncnn"
    )

  DOWNLOAD_DATASET(
    "NCNN OCR model"
    "https://www.deepdetect.com/dd/examples/ncnn/multiword_ocr_ncnn.tar.gz"
    "examples/ncnn"
    "multiword_ocr_ncnn.tar.gz"
    "ocr"
    )

  if(USE_JSON_API)
    REGISTER_TEST(ut_ncnnapi ut-ncnnapi.cc)
  endif()
endif()

if (USE_TENSORRT)
  DOWNLOAD_DATASET(
    "TensorRT test Squeezenet SSD model"
    "https://www.deepdetect.com/dd/examples/tensorrt/squeezenet-ssd-trt.tar.gz"
    "examples/trt"
    "squeezenet-ssd-trt.tar.gz"
    "squeezenet_ssd_trt"
    )
  DOWNLOAD_DATASET(
    "TensorRT test RefineDet model"
    "https://www.deepdetect.com/dd/examples/tensorrt/faces-512-trt.tar.gz"
    "examples/trt"
    "faces-512-trt.tar.gz"
    "faces_512"
  )
  DOWNLOAD_DATASET(
    "Embedded ImageNet classification"
    "https://deepdetect.com/models/init/embedded/images/classification/squeezenet_v1.tar.gz"
    "examples/trt/squeezenet_v1"
    "squeezenet_v1.tar.gz"
    "deploy.prototxt"
  )
  DOWNLOAD_DATASET(
    "ONNX resnet model"
    "https://deepdetect.com/models/init/desktop/images/classification/resnet_onnx_trt.tar.gz"
    "examples/trt"
    "resnet_onnx_trt.tar.gz"
    "resnet_onnx_trt"
    )
#  DOWNLOAD_DATASET(
#    "ONNX yolox model"
#    "https://deepdetect.com/models/init/desktop/images/detection/yolox_onnx_trt.tar.gz"
#    "examples/trt"
#    "yolox_onnx_trt.tar.gz"
#    "yolox_onnx_trt"
#    )
  DOWNLOAD_DATASET(
    "ONNX yolox model without wrapper"
    "https://deepdetect.com/models/init/desktop/images/detection/yolox_onnx_trt_nowrap.tar.gz"
    "examples/trt"
    "yolox_onnx_trt_nowrap.tar.gz"
    "yolox_onnx_trt_nowrap"
    )
  DOWNLOAD_DATASET(
    "ONNX CycleGAN model"
    "https://deepdetect.com/dd/examples/tensorrt/cyclegan_resnet_attn_onnx_trt.tar.gz"
    "examples/trt"
    "cyclegan_resnet_attn_onnx_trt.tar.gz"
    "cyclegan_resnet_attn_onnx_trt"
  )
  DOWNLOAD_DATASET(
    "ONNX Consistency model"
    "https://deepdetect.com/dd/examples/tensorrt/noglasses2glasses_cm_128.tar.gz"
    "examples/trt"
    "noglasses2glasses_cm_128.tar.gz"
    "noglasses2glasses_cm_128"
  )

  if(USE_JSON_API)
    REGISTER_TEST(ut_tensorrtapi ut-tensorrtapi.cc)
  endif()
endif()

if (USE_TORCH)
  DOWNLOAD_DATASET(
      "Torch test Resnet 50 model"
      "https://www.deepdetect.com/dd/examples/torch/resnet50_torch241.tar.gz"
      "examples/torch"
      "resnet50_torch241.tar.gz"
      "resnet50_torch241"
    )
  DOWNLOAD_DATASET(
    "Torch training Resnet 50 model"
    "https://www.deepdetect.com/dd/examples/torch/resnet50_training_torch241_small.tar.gz"
    "examples/torch"
    "resnet50_training_torch241_small.tar.gz"
    "resnet50_training_torch241_small"
    )
  DOWNLOAD_DATASET(
    "Torch native Resnet 50 model"
    "https://www.deepdetect.com/dd/examples/torch/resnet50_native_torch.tar.gz"
    "examples/torch"
    "resnet50_native_torch.tar.gz"
    "resnet50_native_torch"
    )
  DOWNLOAD_DATASET(
    "Torchvision test Faster RCNN model"
    "https://www.deepdetect.com/dd/examples/torch/fasterrcnn_torch.tar.gz"
    "examples/torch"
    "fasterrcnn_torch.tar.gz"
    "fasterrcnn_torch"
    )
  # below special version with woraround for jit bug in torch 1..1
  DOWNLOAD_DATASET(
    "Torchvision training Faster RCNN model & cars dataset"
    "https://www.deepdetect.com/dd/examples/torch/fasterrcnn_train_torch111_bs2.tar.gz"
    "examples/torch"
    "fasterrcnn_train_torch111_bs2.tar.gz"
    "fasterrcnn_train_torch111"
    )
  # DOWNLOAD_DATASET(
  #   "Torchvision training Faster RCNN model & cars dataset"
  #   "https://www.deepdetect.com/dd/examples/torch/fasterrcnn_train_torch_bs2.tar.gz"
  #   "examples/torch"
  #   "fasterrcnn_train_torch_bs2.tar.gz"
  #   "fasterrcnn_train_torch"
  #   )
  DOWNLOAD_DATASET(
    "Torch training YoloX Model"
    "https://www.deepdetect.com/dd/examples/torch/yolox_train_torch.tar.gz"
    "examples/torch"
    "yolox_train_torch.tar.gz"
    "yolox_train_torch"
    )
  DOWNLOAD_DATASET(
    "Torch training DETR Model"
    "https://www.deepdetect.com/dd/examples/torch/detr_train_torch.tar.gz"
    "examples/torch"
    "detr_train_torch.tar.gz"
    "detr_train_torch"
    )
  DOWNLOAD_DATASET(
    "Torch training RT-DETRv2 Model"
    "https://www.deepdetect.com/dd/examples/torch/rtdetrv2_train_torch.tar.gz"
    "examples/torch"
    "rtdetrv2_train_torch.tar.gz"
    "rtdetrv2_train_torch"
    )
  DOWNLOAD_DATASET(
    "Torchvision training Resnet18 backbone for OCR"
    "https://www.deepdetect.com/dd/examples/torch/resnet18_training_torch_ocr.tar.gz"
    "examples/torch"
    "resnet18_training_torch_ocr.tar.gz"
    "resnet18_training_torch_ocr"
    )
  DOWNLOAD_DATASET(
    "Torchvision inference DeepLabV3 Resnet50 model"
    "https://www.deepdetect.com/dd/examples/torch/deeplabv3_torch.tar.gz"
    "examples/torch"
    "deeplabv3_torch.tar.gz"
    "deeplabv3_torch"
    )
    DOWNLOAD_DATASET(
    "Torchvision training DeepLabV3 Resnet50 model"
    "https://www.deepdetect.com/dd/examples/torch/deeplabv3_training_torch.tar.gz"
    "examples/torch"
    "deeplabv3_training_torch.tar.gz"
    "deeplabv3_training_torch"
    )
   DOWNLOAD_DATASET(
    "Torch training Segformer model"
    "https://www.deepdetect.com/dd/examples/torch/segformer_training_torch.tar.gz"
    "examples/torch"
    "segformer_training_torch.tar.gz"
    "segformer_training_torch"
    )
  DOWNLOAD_DATASET(
    "Torch BERT classification test model"
    "https://www.deepdetect.com/dd/examples/torch/bert_inference_torch241.tar.gz"
    "examples/torch"
    "bert_inference_torch241.tar.gz"
    "bert_inference_torch241"
    )
  DOWNLOAD_DATASET(
    "Torch Torch BERT training test model"
    "https://www.deepdetect.com/dd/examples/torch/bert_training_torch_140_transformers_251.tar.gz"
    "examples/torch"
    "bert_training_torch_140_transformers_251.tar.gz"
    "bert_training_torch_140_transformers_251"
    )

  if(USE_JSON_API)
    REGISTER_TEST(ut_torchapi ut-torchapi.cc)
    REGISTER_TEST_MULTIGPU(ut_torchapi ut-torchapi.cc)
  endif()
  REGISTER_TEST(ut_graph ut-graph.cc)
endif()

if(USE_TORCH OR USE_TENSORRT)
  REGISTER_TEST(ut_chain ut-chain.cc)
  REGISTER_TEST(ut_video ut-video.cc)
endif()


if (USE_XGBOOST)
  REGISTER_TEST(ut_xgbapi ut-xgbapi.cc)
endif()

if (USE_SIMSEARCH)
  if (USE_ANNOY)
    REGISTER_TEST(ut_simsearch ut-simsearch.cc)
  endif()
  if (USE_FAISS)
    REGISTER_TEST(ut_simsearch_faiss ut-simsearch-faiss.cc)
  endif()

endif()

# Python tests
add_test(NAME ut_python
       WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/tests"
       COMMAND python3 -m unittest ut_python -v)
