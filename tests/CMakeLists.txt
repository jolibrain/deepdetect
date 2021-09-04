include_directories(${COMMON_INCLUDE_DIRS})
link_directories(${COMMON_LINK_DIRS})

find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# Optional headers can be passed (stored in ARGN)
function (REGISTER_TEST _NAME _FILE)
  if (NOT TARGET ${_NAME})
    add_executable(${_NAME} ${_FILE} ${ARGN})
    add_dependencies(${_NAME} protobuf)
    target_link_libraries(${_NAME} ${COMMON_LINK_LIBS} gtest gtest_main)
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

  if (EXISTS "${_WORKDIR}/${_DEST}")
    message(STATUS "Reusing downloaded ${_NAME} dataset")
  else()
    message(STATUS "Downloading ${_NAME}")
    file(DOWNLOAD ${_URL} ${_WORKDIR}/${_DEST})
    set(_FORCE_RECREATE ON)
  endif()
  if (${_IN} STREQUAL ${_DEST})
      message(STATUS "Skipping not compressed ${_WORKDIR}/${_IN}")
  else()
    if (NOT ${_FORCE_RECREATE} AND EXISTS "${_WORKDIR}/${_IN}")
      message(STATUS "Reusing uncompressed ${_WORKDIR}/${_DEST}")
    else()
      string(FIND ${_DEST} ".tar." _tar_index)
      if (${_tar_index} EQUAL -1)
        execute_process(
          WORKING_DIRECTORY ${_WORKDIR}
          COMMAND bzip2 -d ${_DEST}
          RESULT_VARIABLE _UNCOMPRESS_RESULT
        )
      else()
        execute_process(
          WORKING_DIRECTORY ${_WORKDIR}
          COMMAND ${CMAKE_COMMAND} -E tar xvf ${_DEST}
          RESULT_VARIABLE _UNCOMPRESS_RESULT
        )
      endif()
      if (NOT _UNCOMPRESS_RESULT EQUAL 0)
        message(FATAL_ERROR "Untar ${_WORKDIR}/${_DEST} failed")
      endif()
      set(_FORCE_RECREATE ON)
    endif()
  endif()
  if (NOT ${_OUT} STREQUAL "")
    if (EXISTS "${_WORKDIR}/${_OUT}")
      message(STATUS "Reusing already present ${_WORKDIR}/${_OUT}")
    else()
      execute_process(
        WORKING_DIRECTORY ${_WORKDIR}
        COMMAND ${CMAKE_COMMAND} -E create_symlink ${_IN} ${_OUT}
        RESULT_VARIABLE _RENAME_RESULT
        )
      if (NOT _RENAME_RESULT EQUAL 0)
        message(FATAL_ERROR "create_symlink ${_WORKDIR}/${_IN} failed")
      endif()
    endif()
  endif()
endfunction()

function (DOWNLOAD_FILES _BASEURL _DEST)
  set(_files "${ARGN}")
  foreach(_file IN LISTS _files)
    if (EXISTS "${_WORKDIR}/${_file}")
      message(STATUS "Reuse downloaded ${_DEST}/${_file}")
    else()
      message(STATUS "Downloading ${_DEST}/${_file}")
      file(DOWNLOAD ${_BASEURL}/${_file} ${_DEST}/${_file})
    endif()
  endforeach()
endfunction()

find_package(PythonInterp 2.7 REQUIRED)

if (USE_JSON_API)
  REGISTER_TEST(ut_apidata ut-apidata.cc)
  REGISTER_TEST(ut_dto ut-dto.cc)
endif()
if (USE_CAFFE)
  if (USE_JSON_API)
    REGISTER_TEST(ut_conn ut-conn.cc)
    REGISTER_TEST(ut_jsonapi ut-jsonapi.cc)
  endif()
endif()

if (USE_TF)
  add_executable(opencv_tensor opencv_tensor.cc)
  if (USE_HDF5)
    target_link_libraries(opencv_tensor ${OpenCV_LIBS} boost_thread boost_system crypto ssl hdf5_cpp ${TF_LIB_DEPS})
  else()
    target_link_libraries(opencv_tensor ${OpenCV_LIBS} boost_thread boost_system crypto ssl ${TF_LIB_DEPS})
  endif()
endif()

if (USE_CAFFE)
  REGISTER_TEST(ut_caffe_mlp ut-caffe-mlp.cc)
endif()

if (USE_CAFFE)
  DOWNLOAD_DATASET(
    "MNIST (train) dataset"
    "http://www.deepdetect.com/dd/examples/caffe/mnist/mnist_train_lmdb.tar.bz2"
    "examples/caffe/mnist"
    "mnist_train_lmdb.tar.bz2"
    "mnist_train_lmdb"
    "train.lmdb"
    )

  DOWNLOAD_DATASET(
    "MNIST (test) dataset"
    "http://www.deepdetect.com/dd/examples/caffe/mnist/mnist_test_lmdb.tar.bz2"
    "examples/caffe/mnist"
    "mnist_test_lmdb.tar.bz2"
    "mnist_test_lmdb"
    "test.lmdb"
    )

  DOWNLOAD_DATASET(
    "Image (test) dataset"
    "http://www.deepdetect.com/dd/examples/caffe/plankton/train.tar.bz2"
    "examples/caffe/plankton"
    "train.tar.bz2"
    "train"
    )
  DOWNLOAD_DATASET(
    "image segmentation"
    "http://www.deepdetect.com/dd/examples/caffe/camvid/camvid_square.tar.gz"
    "examples/caffe/camvid"
    "camvid_square.tar.gz"
    "CamVid_square"
    )
  DOWNLOAD_DATASET(
    "object detection test model"
    "http://deepdetect.com/dd/examples/caffe/voc/voc0712_dd.tar.gz"
    "examples/caffe/voc"
    "voc0712_dd.tar.gz"
    "voc"
  )
  DOWNLOAD_DATASET(
    "word detection test model"
    "https://www.deepdetect.com/dd/examples/caffe/ocr/word_detect_v2_test.tar.gz"
    "examples/caffe/word_detect_v2"
    "word_detect_v2_test.tar.gz"
    "deploy.prototxt"
  )
  DOWNLOAD_DATASET(
    "ocr test model"
    "https://deepdetect.com/models/init/desktop/images/ocr/multiword_ocr.tar.gz"
    "examples/caffe/multiword_ocr"
    "multiword_ocr.tar.gz"
    "deploy.prototxt"
  )
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

set(SINUS_EXAMPLE_PATH ${CMAKE_SOURCE_DIR}/examples/all/sinus/)
set(SINUS_EXAMPLE_OUT "examples/all/sinus")
set(SINUS_EXAMPLE_TRAIN_OUT "examples/all/sinus/train")
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

if (USE_CAFFE)
  if (USE_JSON_API)
    REGISTER_TEST(ut_caffeapi ut-caffeapi.cc)
  endif()
  if (USE_HTTP_SERVER_OATPP)
    REGISTER_TEST(ut_oatpp ut-oatpp.cc ut-oatpp.h)
    target_link_libraries(ut_oatpp ${OATPP_LIB_DEPS} ${CMAKE_BINARY_DIR}/oatpp/src/oatpp/src/liboatpp-test.a)
  endif()
endif()

if (USE_TF)
  DOWNLOAD_FILES(
    "http://www.deepdetect.com/models/tf/"
    "examples/tf/inception/"
    "inception_v1.pb"
    "grace_hopper.jpg"
    "cat.jpg"
    "corresp_inception_clean.txt"
  )

  REGISTER_TEST(ut_tfapi ut-tfapi.cc)
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
  DOWNLOAD_DATASET(
    "ONNX CycleGAN model"
    "https://deepdetect.com/dd/examples/tensorrt/cyclegan_resnet_attn_onnx_trt.tar.gz"
    "examples/trt"
    "cyclegan_resnet_attn_onnx_trt.tar.gz"
    "cyclegan_resnet_attn_onnx_trt"
  )

  if(USE_JSON_API)
    REGISTER_TEST(ut_tensorrtapi ut-tensorrtapi.cc)
  endif()
endif()

if (USE_TORCH)
  DOWNLOAD_DATASET(
    "Torch test Resnet 50 model"
    "https://www.deepdetect.com/dd/examples/torch/resnet50_torch.tar.gz"
    "examples/torch"
    "resnet50_torch.tar.gz"
    "resnet50_torch"
    )
  DOWNLOAD_DATASET(
    "Torch training Resnet 50 model"
    "https://www.deepdetect.com/dd/examples/torch/resnet50_training_torch_small.tar.gz"
    "examples/torch"
    "resnet50_training_torch_small.tar.gz"
    "resnet50_training_torch_small"
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
  DOWNLOAD_DATASET(
    "Torchvision training Faster RCNN model & cars dataset"
    "https://www.deepdetect.com/dd/examples/torch/fasterrcnn_train_torch_bs2.tar.gz"
    "examples/torch"
    "fasterrcnn_train_torch_bs2.tar.gz"
    "fasterrcnn_train_torch"
    )
  DOWNLOAD_DATASET(
    "Torchvision training DeepLabV3 Resnet50 model"
    "https://www.deepdetect.com/dd/examples/torch/deeplabv3_torch.tar.gz"
    "examples/torch"
    "deeplabv3_torch.tar.gz"
    "deeplabv3_torch"
    )
  DOWNLOAD_DATASET(
    "Torch BERT classification test model"
    "https://www.deepdetect.com/dd/examples/torch/bert_inference_torch.tar.gz"
    "examples/torch"
    "bert_inference_torch.tar.gz"
    "bert_inference_torch"
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

if(USE_TORCH OR USE_CAFFE OR USE_TENSORRT)
  REGISTER_TEST(ut_chain ut-chain.cc)
endif()

if (USE_CAFFE2)
  function (ASSERT_EXISTS _MODEL _FILE _UNTAR)
    set(_FINAL_PATH "examples/caffe2/${_MODEL}/${_FILE}") # File to get
    set(_LOCAL_PATH "${_FINAL_PATH}")
    if (_UNTAR)
      set(_LOCAL_PATH "${_LOCAL_PATH}.tar.gz") # File to download
    endif ()
    set(_REMOTE_PATH "http://www.deepdetect.com/dd/${_LOCAL_PATH}") # Url of the file
    if (NOT EXISTS ${_LOCAL_PATH}) # Download
      message(STATUS "Downloading Caffe2 ${_MODEL} ${_FILE}")
      file(DOWNLOAD ${_REMOTE_PATH} ${_LOCAL_PATH})
    endif ()
    if (_UNTAR AND NOT EXISTS ${_FINAL_PATH}) # Untar
      message(STATUS "Extracting Caffe2 ${_MODEL} ${_FILE}")
      execute_process(COMMAND tar xvzf ${_LOCAL_PATH})
      execute_process(COMMAND mv ${_FILE} ${_FINAL_PATH})
    endif ()
  endfunction (ASSERT_EXISTS)

  function (ASSERT_MODEL_EXISTS _MODEL)
    ASSERT_EXISTS(${_MODEL} "corresp.txt" 0)
    ASSERT_EXISTS(${_MODEL} "init_net.pb" 0)
    ASSERT_EXISTS(${_MODEL} "predict_net.pb" 0)
  endfunction (ASSERT_MODEL_EXISTS)

  # Tar GZ files

  ASSERT_EXISTS("boats_and_cars" "imgs" 1)
  ASSERT_EXISTS("detectron" "imgs" 1)
  ASSERT_EXISTS("detectron_mask" "ext" 1)
  ASSERT_EXISTS("resnet_50_imagenet" "imgs" 1)
  ASSERT_EXISTS("resnet_50_imagenet" "fish.lmdb" 1)

  # Models

  ASSERT_MODEL_EXISTS("detectron")
  ASSERT_MODEL_EXISTS("detectron_mask")
  ASSERT_MODEL_EXISTS("resnet_50_imagenet")

  # Tests

  REGISTER_TEST(ut_caffe2api ut-caffe2api.cc)

endif()

if (USE_XGBOOST)
  REGISTER_TEST(ut_xgbapi ut-xgbapi.cc)
endif()

if (USE_SIMSEARCH)
  DOWNLOAD_DATASET(
    "object detection test model with rois"
    "http://www.deepdetect.com/dd/examples/caffe/voc_roi/voc0712_dd_roi.tar.gz"
    "examples/caffe/voc_roi"
    "voc0712_dd_roi.tar.gz"
    "voc_roi"
    )
  if (USE_ANNOY)
    REGISTER_TEST(ut_simsearch ut-simsearch.cc)
  endif()
  if (USE_FAISS)
    REGISTER_TEST(ut_simsearch_faiss ut-simsearch-faiss.cc)
  endif()

endif()
