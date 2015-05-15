if (CUDA_FOUND)
  set(CUDA_LIB_DEPS -lcudart -lcublas -lcurand)# -lcudnn)
else()
  add_definitions(-DCPU_ONLY)
endif()
set(CAFFE_LIB_DEPS -lleveldb -lsnappy -llmdb -lboost_system -lhdf5_hl -lhdf5 -lboost_thread -lopenblas -lcaffe -lprotobuf)
link_directories(${CAFFE_LIB_DIR})

find_package(GTest)
include_directories(${GTEST_INCLUDE_DIRS})
if (GTEST_FOUND)
  add_executable(ut_apidata ut-apidata.cc)
  target_link_libraries(ut_apidata ddetect glog gflags gtest gtest_main opencv_core opencv_highgui opencv_imgproc curlpp curl ${CAFFE_LIB_DEPS})
  add_test(
    NAME ut_apidata
    COMMAND ut_apidata
    )
  
  add_executable(ut_conn ut-conn.cc)
  target_link_libraries(ut_conn ddetect glog gflags gtest gtest_main opencv_core opencv_highgui opencv_imgproc curlpp curl ${CAFFE_LIB_DEPS})
  add_test(
    NAME ut_conn
    COMMAND ut_conn
    )
  
  add_executable(ut_jsonapi ut-jsonapi.cc)
  target_link_libraries(ut_jsonapi ddetect glog gflags gtest gtest_main opencv_core opencv_highgui opencv_imgproc curlpp curl ${CAFFE_LIB_DEPS})
  add_test(
    NAME ut_jsonapi
    COMMAND ut_jsonapi
    )

  add_executable(test_server test-server.cc)
  target_link_libraries(test_server boost_thread cppnetlib-uri cppnetlib-client-connections boost_system crypto ssl)

  add_executable(ut_caffe_mlp ut-caffe-mlp.cc)
  target_link_libraries(ut_caffe_mlp ddetect glog gflags gtest gtest_main opencv_core opencv_highgui opencv_imgproc curlpp curl ${CAFFE_LIB_DEPS})
  add_test(
    NAME ut_caffe_mlp
    COMMAND ut_caffe_mlp
    )
  
  set(MNIST_EXAMPLE_PATH "examples/caffe/mnist/")
  set(MNIST_EXAMPLE_TRAIN_ARCHIVE "mnist_train_lmdb.tar.bz2")
  set(MNIST_EXAMPLE_TEST_ARCHIVE "mnist_test_lmdb.tar.bz2")
  set(MNIST_EXAMPLE_TRAIN_OUT "mnist_train_lmdb")
  set(MNIST_EXAMPLE_TEST_OUT "mnist_test_lmdb")
  if (NOT EXISTS "${MNIST_EXAMPLE_PATH}${MNIST_EXAMPLE_TRAIN_ARCHIVE}")
    file(DOWNLOAD "http://juban.free.fr/dd/examples/caffe/mnist/mnist_train_lmdb.tar.bz2" "${MNIST_EXAMPLE_PATH}/${MNIST_EXAMPLE_TRAIN_ARCHIVE}")
    file(DOWNLOAD "http://juban.free.fr/dd/examples/caffe/mnist/mnist_test_lmdb.tar.bz2" "${MNIST_EXAMPLE_PATH}/${MNIST_EXAMPLE_TEST_ARCHIVE}")
    execute_process(COMMAND ${CMAKE_COMMAND} -E tar xvjf ${MNIST_EXAMPLE_PATH}${MNIST_EXAMPLE_TRAIN_ARCHIVE} -C ${MNIST_EXAMPLE_PATH}) # XXX: output directory doesn't work here, maybe because of special cmake handling
    execute_process(COMMAND ${CMAKE_COMMAND} -E tar xvjf ${MNIST_EXAMPLE_PATH}${MNIST_EXAMPLE_TEST_ARCHIVE} -C ${MNIST_EXAMPLE_PATH})
    execute_process(COMMAND ${CMAKE_COMMAND} -E rename ${MNIST_EXAMPLE_TRAIN_OUT} ${MNIST_EXAMPLE_PATH}${MNIST_EXAMPLE_TRAIN_OUT})
    execute_process(COMMAND ${CMAKE_COMMAND} -E rename ${MNIST_EXAMPLE_TEST_OUT} ${MNIST_EXAMPLE_PATH}${MNIST_EXAMPLE_TEST_OUT})
  endif()
  
  set(FOREST_EXAMPLE_PATH "examples/all/forest_type/")
  set(FOREST_EXAMPLE_TRAIN_ARCHIVE "train.csv.tar.bz2")
  set(FOREST_EXAMPLE_TRAIN_OUT "train.csv")
  if (NOT EXISTS "${FOREST_EXAMPLE_PATH}${FOREST_EXAMPLE_TRAIN_ARCHIVE}")
    file(DOWNLOAD "http://juban.free.fr/dd/examples/all/forest_type/train.csv.tar.bz2" "${FOREST_EXAMPLE_PATH}${FOREST_EXAMPLE_TRAIN_ARCHIVE}")
    execute_process(COMMAND ${CMAKE_COMMAND} -E tar xvjf ${FOREST_EXAMPLE_PATH}${FOREST_EXAMPLE_TRAIN_ARCHIVE} -C ${FOREST_EXAMPLE_PATH})
    execute_process(COMMAND ${CMAKE_COMMAND} -E rename ${FOREST_EXAMPLE_TRAIN_OUT} ${FOREST_EXAMPLE_PATH}${FOREST_EXAMPLE_TRAIN_OUT})
  endif()

  set(PLANK_EXAMPLE_PATH "examples/caffe/plankton/")
  set(PLANK_EXAMPLE_TRAIN_ARCHIVE "train.tar.bz2")
  set(PLANK_EXAMPLE_TRAIN_OUT "train")
  if (NOT EXISTS "${PLANK_EXAMPLE_PATH}${PLANK_EXAMPLE_TRAIN_ARCHIVE}")
    file(DOWNLOAD "http://juban.free.fr/dd/examples/caffe/plankton/${PLANK_EXAMPLE_TRAIN_ARCHIVE}" "${PLANK_EXAMPLE_PATH}${PLANK_EXAMPLE_TRAIN_ARCHIVE}")
    execute_process(COMMAND ${CMAKE_COMMAND} -E tar xvjf ${PLANK_EXAMPLE_PATH}${PLANK_EXAMPLE_TRAIN_ARCHIVE} -C ${PLANK_EXAMPLE_PATH})
    execute_process(COMMAND ${CMAKE_COMMAND} -E rename ${PLANK_EXAMPLE_TRAIN_OUT} ${PLANK_EXAMPLE_PATH}${PLANK_EXAMPLE_TRAIN_OUT})
  endif()
  
  add_executable(ut_caffeapi ut-caffeapi.cc)
  target_link_libraries(ut_caffeapi ddetect glog gflags gtest gtest_main opencv_core opencv_highgui opencv_imgproc curlpp curl ${CAFFE_LIB_DEPS})
  add_test(
    NAME ut_caffeapi
    COMMAND ut_caffeapi
    )
  
  add_executable(ut_httpapi ut-httpapi.cc)
  target_link_libraries(ut_httpapi ddetect glog gflags gtest gtest_main opencv_core opencv_highgui opencv_imgproc ${CAFFE_LIB_DEPS} cppnetlib-uri curlpp curl boost_system crypto ssl)
  add_test(
    NAME ut_httpapi
    COMMAND ut_httpapi
    )

endif()