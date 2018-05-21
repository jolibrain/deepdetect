include_directories(${Boost_INCLUDE_DIRS})# ${PROTOBUF_INCLUDE_DIRS})
if (CUDA_FOUND)
  set(CUDA_LIB_DEPS ${CUDA_LIBRARIES})
else()
  set(CUDA_LIB_DEPS "")
  add_definitions(-DCPU_ONLY)
endif()
set(CAFFE_LIB_DEPS -lleveldb -lsnappy -llmdb -lhdf5_hl -lhdf5 -lopenblas -lcaffe -lprotobuf)
if (USE_TF)
  set(TF_LIB_DEPS -ltensorflow_cc)
else()
  set(TF_LIB_DEPS)
endif()
link_directories(${CAFFE_LIB_DIR})
link_directories(${TF_LIB_DIR})
link_directories(${XGBOOST_LIB_DIR})
link_directories(${TSNE_LIB_DIR})

if (USE_XGBOOST)
  if (CUDA_FOUND AND USE_XGBOOST_GPU)
    set(XGBOOST_LIB_DEPS -Wl,--whole-archive ${CMAKE_BINARY_DIR}/xgboost/src/xgboost/lib/libxgboost.a -Wl,--no-whole-archive ${CMAKE_BINARY_DIR}/xgboost/src/xgboost/build/dmlc-core/libdmlc.a ${CMAKE_BINARY_DIR}/xgboost/src/xgboost/build/librabit.a  ${CMAKE_BINARY_DIR}/xgboost/src/xgboost/build/CMakeFiles/gpuxgboost.dir/plugin/updater_gpu/src/gpuxgboost_generated_updater_gpu.cu.o ${CMAKE_BINARY_DIR}/xgboost/src/xgboost/build/nccl/libnccl.a)
  else()
    set(XGBOOST_LIB_DEPS -Wl,--whole-archive ${CMAKE_BINARY_DIR}/xgboost/src/xgboost/lib/libxgboost.a -Wl,--no-whole-archive ${CMAKE_BINARY_DIR}/xgboost/src/xgboost/build/dmlc-core/libdmlc.a ${CMAKE_BINARY_DIR}/xgboost/src/xgboost/build/librabit.a)
  endif()
else()
  set(XGBOOST_LIB_DEPS)
endif()

if (USE_TSNE)
  set(TSNE_LIB_DEPS -ltsne_multicore)
endif()

find_package(GTest)
include_directories(${GTEST_INCLUDE_DIRS})
if (GTEST_FOUND)
  add_executable(ut_apidata ut-apidata.cc)
  target_link_libraries(ut_apidata ddetect ${CUDA_LIB_DEPS} glog gflags gtest gtest_main ${OpenCV_LIBS} curlpp curl hdf5_cpp ${Boost_LIBRARIES} ${CAFFE_LIB_DEPS} ${TF_LIB_DEPS} ${XGBOOST_LIB_DEPS} ${TSNE_LIB_DEPS})
  add_test(
    NAME ut_apidata
    COMMAND ut_apidata
    )
  
  add_executable(ut_conn ut-conn.cc)
  target_link_libraries(ut_conn ddetect ${CUDA_LIB_DEPS} glog gflags gtest gtest_main ${OpenCV_LIBS} curlpp curl hdf5_cpp ${Boost_LIBRARIES} ${CAFFE_LIB_DEPS} ${TF_LIB_DEPS} ${XGBOOST_LIB_DEPS} ${TSNE_LIB_DEPS})
  add_test(
    NAME ut_conn
    COMMAND ut_conn
    )
  
  add_executable(ut_jsonapi ut-jsonapi.cc)
  target_link_libraries(ut_jsonapi ddetect ${CUDA_LIB_DEPS} glog gflags gtest gtest_main ${OpenCV_LIBS} curlpp curl hdf5_cpp ${Boost_LIBRARIES} ${CAFFE_LIB_DEPS} ${TF_LIB_DEPS} ${XGBOOST_LIB_DEPS} ${TSNE_LIB_DEPS})
  add_test(
    NAME ut_jsonapi
    COMMAND ut_jsonapi
    )

  if (USE_TF)
    add_executable(opencv_tensor opencv_tensor.cc)
    target_link_libraries(opencv_tensor ${OpenCV_LIBS} boost_thread boost_system crypto ssl hdf5_cpp ${TF_LIB_DEPS})
  endif()
  
  add_executable(test_server test-server.cc)
  target_link_libraries(test_server boost_thread cppnetlib-uri cppnetlib-client-connections boost_system crypto ssl)

  add_executable(ut_caffe_mlp ut-caffe-mlp.cc)
  target_link_libraries(ut_caffe_mlp ddetect ${CUDA_LIB_DEPS} glog gflags gtest gtest_main ${OpenCV_LIBS} curlpp curl hdf5_cpp ${Boost_LIBRARIES} ${CAFFE_LIB_DEPS} ${TF_LIB_DEPS} ${XGBOOST_LIB_DEPS} ${TSNE_LIB_DEPS})
  add_test(
    NAME ut_caffe_mlp
    COMMAND ut_caffe_mlp
    )

  set(MNIST_EXAMPLE_PATH "examples/caffe/mnist/")
  set(MNIST_EXAMPLE_TRAIN_ARCHIVE "mnist_train_lmdb.tar.bz2")
  set(MNIST_EXAMPLE_TEST_ARCHIVE "mnist_test_lmdb.tar.bz2")
  set(MNIST_EXAMPLE_TRAIN_OUT "mnist_train_lmdb")
  set(MNIST_EXAMPLE_TEST_OUT "mnist_test_lmdb")
  set(MNIST_EXAMPLE_TRAIN_OUT_T "train.lmdb")
  set(MNIST_EXAMPLE_TEST_OUT_T "test.lmdb")
  if (NOT EXISTS "${MNIST_EXAMPLE_PATH}${MNIST_EXAMPLE_TRAIN_ARCHIVE}")
    message(STATUS "Downloading MNIST dataset")
    file(DOWNLOAD "http://www.deepdetect.com/dd/examples/caffe/mnist/mnist_train_lmdb.tar.bz2" "${MNIST_EXAMPLE_PATH}/${MNIST_EXAMPLE_TRAIN_ARCHIVE}")
    file(DOWNLOAD "http://www.deepdetect.com/dd/examples/caffe/mnist/mnist_test_lmdb.tar.bz2" "${MNIST_EXAMPLE_PATH}/${MNIST_EXAMPLE_TEST_ARCHIVE}")
    execute_process(COMMAND ${CMAKE_COMMAND} -E tar xvjf ${MNIST_EXAMPLE_PATH}${MNIST_EXAMPLE_TRAIN_ARCHIVE} -C ${MNIST_EXAMPLE_PATH}) # XXX: output directory doesn't work here, maybe because of special cmake handling
    execute_process(COMMAND ${CMAKE_COMMAND} -E tar xvjf ${MNIST_EXAMPLE_PATH}${MNIST_EXAMPLE_TEST_ARCHIVE} -C ${MNIST_EXAMPLE_PATH})
    execute_process(COMMAND ${CMAKE_COMMAND} -E rename ${MNIST_EXAMPLE_TRAIN_OUT} ${MNIST_EXAMPLE_PATH}${MNIST_EXAMPLE_TRAIN_OUT_T})
    execute_process(COMMAND ${CMAKE_COMMAND} -E rename ${MNIST_EXAMPLE_TEST_OUT} ${MNIST_EXAMPLE_PATH}${MNIST_EXAMPLE_TEST_OUT_T})
  endif()

  set(FOREST_EXAMPLE_PATH "examples/all/forest_type/")
  set(FOREST_EXAMPLE_TRAIN_ARCHIVE "train.csv.tar.bz2")
  set(FOREST_EXAMPLE_TRAIN_OUT "train.csv")
  if (NOT EXISTS "${FOREST_EXAMPLE_PATH}${FOREST_EXAMPLE_TRAIN_ARCHIVE}")
    message(STATUS "Downloading CSV test dataset")
    file(DOWNLOAD "http://www.deepdetect.com/dd/examples/all/forest_type/train.csv.tar.bz2" "${FOREST_EXAMPLE_PATH}${FOREST_EXAMPLE_TRAIN_ARCHIVE}")
    execute_process(COMMAND ${CMAKE_COMMAND} -E tar xvjf ${FOREST_EXAMPLE_PATH}${FOREST_EXAMPLE_TRAIN_ARCHIVE} -C ${FOREST_EXAMPLE_PATH})
    execute_process(COMMAND ${CMAKE_COMMAND} -E rename ${FOREST_EXAMPLE_TRAIN_OUT} ${FOREST_EXAMPLE_PATH}${FOREST_EXAMPLE_TRAIN_OUT})
  endif()

  set(FARM_EXAMPLE_PATH "examples/all/farm_ads/")
  set(FARM_EXAMPLE_TRAIN_OUT "farm-ads.svm")
  if (NOT EXISTS "${FARM_EXAMPLE_PATH}${FARM_EXAMPLE_TRAIN_OUT}")
    message(STATUS "Downloading SVM test dataset")
    file(DOWNLOAD "http://www.deepdetect.com/dd/examples/all/farm_ads/farm-ads.svm" "${FARM_EXAMPLE_PATH}${FARM_EXAMPLE_TRAIN_OUT}")
  endif()

  set(PLANK_EXAMPLE_PATH "examples/caffe/plankton/")
  set(PLANK_EXAMPLE_TRAIN_ARCHIVE "train.tar.bz2")
  set(PLANK_EXAMPLE_TRAIN_OUT "train")
  if (NOT EXISTS "${PLANK_EXAMPLE_PATH}${PLANK_EXAMPLE_TRAIN_ARCHIVE}")
    message(STATUS "Downloading image test dataset")
    file(DOWNLOAD "http://www.deepdetect.com/dd/examples/caffe/plankton/${PLANK_EXAMPLE_TRAIN_ARCHIVE}" "${PLANK_EXAMPLE_PATH}${PLANK_EXAMPLE_TRAIN_ARCHIVE}")
    execute_process(COMMAND ${CMAKE_COMMAND} -E tar xvjf ${PLANK_EXAMPLE_PATH}${PLANK_EXAMPLE_TRAIN_ARCHIVE} -C ${PLANK_EXAMPLE_PATH})
    execute_process(COMMAND ${CMAKE_COMMAND} -E rename ${PLANK_EXAMPLE_TRAIN_OUT} ${PLANK_EXAMPLE_PATH}${PLANK_EXAMPLE_TRAIN_OUT})
    execute_process(COMMAND ${CMAKE_COMMAND} -E rename file-lst-ml.txt ${PLANK_EXAMPLE_PATH}file-lst-ml.txt)
    execute_process(COMMAND ${CMAKE_COMMAND} -E rename file-lst-test-ml.txt ${PLANK_EXAMPLE_PATH}file-lst-test-ml.txt)
    execute_process(COMMAND ${CMAKE_COMMAND} -E rename file-lst.txt ${PLANK_EXAMPLE_PATH}file-lst.txt)
    execute_process(COMMAND ${CMAKE_COMMAND} -E rename file-lst-test.txt ${PLANK_EXAMPLE_PATH}file-lst-test.txt)
  endif()

  set(N20_EXAMPLE_PATH "examples/all/n20/")
  set(N20_EXAMPLE_TRAIN_ARCHIVE "news20.tar.bz2")
  set(N20_EXAMPLE_TRAIN_OUT "news20")
  if (NOT EXISTS "${N20_EXAMPLE_PATH}${N20_EXAMPLE_TRAIN_ARCHIVE}")
    message(STATUS "Downloading text test dataset")
    file(DOWNLOAD "http://www.deepdetect.com/dd/examples/all/n20/${N20_EXAMPLE_TRAIN_ARCHIVE}" "${N20_EXAMPLE_PATH}${N20_EXAMPLE_TRAIN_ARCHIVE}")
    execute_process(COMMAND ${CMAKE_COMMAND} -E tar xvjf ${N20_EXAMPLE_PATH}${N20_EXAMPLE_TRAIN_ARCHIVE} -C ${N20_EXAMPLE_PATH})
    execute_process(COMMAND ${CMAKE_COMMAND} -E rename ${N20_EXAMPLE_TRAIN_OUT} ${N20_EXAMPLE_PATH}${N20_EXAMPLE_TRAIN_OUT})
  endif()

  set(SFLARE_EXAMPLE_PATH "examples/all/sflare/")
  set(SFLARE_EXAMPLE_TRAIN_ARCHIVE "flare.csv")
  set(SFLARE_EXAMPLE_TRAIN_OUT "sflare")
  if (NOT EXISTS "${SFLARE_EXAMPLE_PATH}${SFLARE_EXAMPLE_TRAIN_ARCHIVE}")
    message(STATUS "Downloading regression test model")
    file(DOWNLOAD "http://www.deepdetect.com/dd/examples/all/sflare/${SFLARE_EXAMPLE_TRAIN_ARCHIVE}" "${SFLARE_EXAMPLE_PATH}${SFLARE_EXAMPLE_TRAIN_ARCHIVE}")
  endif()

  set(CAMVID_EXAMPLE_PATH "examples/caffe/camvid/")
  set(CAMVID_EXAMPLE_TRAIN_ARCHIVE "camvid_square.tar.gz")
  set(CAMVID_EXAMPLE_TRAIN_OUT "CamVid_square")
  if (NOT EXISTS "${CAMVID_EXAMPLE_PATH}${CAMVID_EXAMPLE_TRAIN_ARCHIVE}")
    message(STATUS "Downloading image segmentation dataset")
    file(DOWNLOAD "http://www.deepdetect.com/dd/examples/caffe/camvid/${CAMVID_EXAMPLE_TRAIN_ARCHIVE}" "${CAMVID_EXAMPLE_PATH}${CAMVID_EXAMPLE_TRAIN_ARCHIVE}")
    execute_process(COMMAND ${CMAKE_COMMAND} -E tar xvzf ${CAMVID_EXAMPLE_PATH}${CAMVID_EXAMPLE_TRAIN_ARCHIVE} -C ${CAMVID_EXAMPLE_PATH})
    execute_process(COMMAND ${CMAKE_COMMAND} -E rename ${CAMVID_EXAMPLE_TRAIN_OUT} ${CAMVID_EXAMPLE_PATH}${CAMVID_EXAMPLE_TRAIN_OUT})
  endif()

  set(VOC_EXAMPLE_PATH "examples/caffe/voc/")
  set(VOC_EXAMPLE_MODEL_ARCHIVE "voc0712_dd.tar.gz")
  set(VOC_EXAMPLE_MODEL_OUT "voc")
  if (NOT EXISTS "${VOC_EXAMPLE_PATH}${VOC_EXAMPLE_MODEL_ARCHIVE}")
    message(STATUS "Downloading object detection test model")
    file(DOWNLOAD "http://deepdetect.com/dd/examples/caffe/voc/${VOC_EXAMPLE_MODEL_ARCHIVE}" "${VOC_EXAMPLE_PATH}${VOC_EXAMPLE_MODEL_ARCHIVE}")
    execute_process(COMMAND ${CMAKE_COMMAND} -E tar xvzf ${VOC_EXAMPLE_PATH}${VOC_EXAMPLE_MODEL_ARCHIVE} -C ${VOC_EXAMPLE_PATH})
    execute_process(COMMAND ${CMAKE_COMMAND} -E rename ${VOC_EXAMPLE_MODEL_OUT} ${VOC_EXAMPLE_PATH}${VOC_EXAMPLE_MODEL_OUT})
  endif()
  
  add_executable(ut_caffeapi ut-caffeapi.cc)
  target_link_libraries(ut_caffeapi ddetect ${CUDA_LIB_DEPS} glog gflags gtest gtest_main ${OpenCV_LIBS} curlpp curl hdf5_cpp ${Boost_LIBRARIES} ${CAFFE_LIB_DEPS} ${TF_LIB_DEPS} ${XGBOOST_LIB_DEPS} ${TSNE_LIB_DEPS})
  add_test(
    NAME ut_caffeapi
    COMMAND ut_caffeapi
    )
  
  add_executable(ut_httpapi ut-httpapi.cc)
  target_link_libraries(ut_httpapi ddetect ${CUDA_LIB_DEPS} glog gflags gtest gtest_main ${OpenCV_LIBS} ${Boost_LIBRARIES} ${CAFFE_LIB_DEPS} ${TF_LIB_DEPS}  ${XGBOOST_LIB_DEPS} ${TSNE_LIB_DEPS} cppnetlib-uri curlpp curl crypto ssl hdf5_cpp)
  add_test(
    NAME ut_httpapi
    COMMAND ut_httpapi
    )

if (USE_TF)
  set(INCEPT_EXAMPLE_PATH "examples/tf/inception/")
  set(INCEPT_EXAMPLE_MODEL_OUT "inception_v1.pb")
  set(INCEPT_IMG1 "grace_hopper.jpg")
  set(INCEPT_IMG2 "cat.jpg")
  set(INCEPT_CORRESP "corresp_inception_clean.txt")
  if (NOT EXISTS "${INCEPT_EXAMPLE_PATH}${INCEPT_EXAMPLE_MODEL_OUT}")
    message(STATUS "Downloading TF test model")
    file(DOWNLOAD "http://www.deepdetect.com/models/tf/${INCEPT_EXAMPLE_MODEL_OUT}" "${INCEPT_EXAMPLE_PATH}${INCEPT_EXAMPLE_MODEL_OUT}")
    file(DOWNLOAD "http://www.deepdetect.com/models/tf/${INCEPT_IMG1}" "${INCEPT_EXAMPLE_PATH}${INCEPT_IMG1}")
    file(DOWNLOAD "http://www.deepdetect.com/models/tf/${INCEPT_IMG2}" "${INCEPT_EXAMPLE_PATH}${INCEPT_IMG2}")
    file(DOWNLOAD "http://www.deepdetect.com/models/tf/${INCEPT_CORRESP}" "${INCEPT_EXAMPLE_PATH}${INCEPT_CORRESP}")
  endif()

  add_executable(ut_tfapi ut-tfapi.cc)
  target_link_libraries(ut_tfapi ddetect ${CUDA_LIB_DEPS} glog gflags gtest gtest_main opencv_core opencv_highgui opencv_imgproc curlpp curl ${Boost_LIBRARIES} ${CAFFE_LIB_DEPS} ${TF_LIB_DEPS} ${XGBOOST_LIB_DEPS} ${TSNE_LIB_DEPS})
  add_test(
    NAME ut_tfapi
    COMMAND ut_tfapi
    )
endif()
  
if (USE_XGBOOST)
  add_executable(ut_xgbapi ut-xgbapi.cc)
  target_link_libraries(ut_xgbapi ddetect ${CUDA_LIB_DEPS} glog gflags gtest gtest_main ${OpenCV_LIBS} curlpp curl hdf5_cpp ${Boost_LIBRARIES} ${CAFFE_LIB_DEPS}  ${XGBOOST_LIB_DEPS} ${TF_LIB_DEPS} ${TSNE_LIB_DEPS})
  add_test(
    NAME ut_xgbapi
    COMMAND ut_xgbapi
    )
endif()

if (USE_SIMSEARCH)
  set(VOC_EXAMPLE_PATH "examples/caffe/voc_roi/")
  set(VOC_EXAMPLE_MODEL_ARCHIVE "voc0712_dd_roi.tar.gz")
  set(VOC_EXAMPLE_MODEL_OUT "voc_roi")
  if (NOT EXISTS "${VOC_EXAMPLE_PATH}${VOC_EXAMPLE_MODEL_ARCHIVE}")
    message(STATUS "Downloading object detection test model with rois")
    file(DOWNLOAD "http://deepdetect.com/dd/examples/caffe/voc_roi/${VOC_EXAMPLE_MODEL_ARCHIVE}" "${VOC_EXAMPLE_PATH}${VOC_EXAMPLE_MODEL_ARCHIVE}")
    execute_process(COMMAND ${CMAKE_COMMAND} -E tar xvzf ${VOC_EXAMPLE_PATH}${VOC_EXAMPLE_MODEL_ARCHIVE} -C ${VOC_EXAMPLE_PATH})
    execute_process(COMMAND ${CMAKE_COMMAND} -E rename ${VOC_EXAMPLE_MODEL_OUT} ${VOC_EXAMPLE_PATH}${VOC_EXAMPLE_MODEL_OUT})
  endif()
  
  add_executable(ut_simsearch ut-simsearch.cc)
  target_link_libraries(ut_simsearch ddetect ${CUDA_LIB_DEPS} glog gflags gtest gtest_main ${OpenCV_LIBS} curlpp curl hdf5_cpp ${Boost_LIBRARIES} ${CAFFE_LIB_DEPS} ${TF_LIB_DEPS} ${XGBOOST_LIB_DEPS} ${TSNE_LIB_DEPS})
  add_test(
    NAME ut_simsearch
    COMMAND ut_simsearch
  )
endif()

endif()
