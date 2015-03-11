set(CAFFE_STATIC_LINK -Wl,--whole-archive /media/dd2/projects/deepdetect/apps/caffe/build/lib/libcaffe.a -Wl,--no-whole-archive)
if (CUDA_FOUND)
  set(CUDA_LIB_DEPS -lcudart -lcublas -lcurand -lcudnn)
else()
  add_definitions(-DCPU_ONLY)
endif()
set(CAFFE_LIB_DEPS -lprotobuf -lleveldb -lsnappy -llmdb -lboost_system -lhdf5_hl -lhdf5 -lboost_thread -lopenblas)
link_directories(${CAFFE_LIB_DIR})

find_package(GTest)
include_directories(${GTEST_INCLUDE_DIRS})
if (GTEST_FOUND)
  add_executable(ut_apidata ut-apidata.cc)
  target_link_libraries(ut_apidata ${CAFFE_STATIC_LINK} ${CAFFE_LIB_DEPS} ${CUDA_LIB_DEPS} ddetect glog gflags gtest gtest_main opencv_core opencv_highgui opencv_imgproc)
  add_test(
    NAME ut_apidata
    COMMAND ut_apidata
    )
  
  add_executable(ut_conn ut-conn.cc)
  target_link_libraries(ut_conn ${CAFFE_STATIC_LINK} ${CAFFE_LIB_DEPS} ${CUDA_LIB_DEPS} ddetect glog gflags gtest gtest_main opencv_core opencv_highgui opencv_imgproc)
  add_test(
    NAME ut_conn
    COMMAND ut_conn
    )
  
  add_executable(ut_jsonapi ut-jsonapi.cc)
  target_link_libraries(ut_jsonapi ${CAFFE_STATIC_LINK} ${CAFFE_LIB_DEPS} ${CUDA_LIB_DEPS} ddetect glog gflags gtest gtest_main opencv_core opencv_highgui opencv_imgproc)
  add_test(
    NAME ut_jsonapi
    COMMAND ut_jsonapi
    )
  
  set(MNIST_EXAMPLE_PATH "examples/caffe/mnist/")
  set(MNIST_EXAMPLE_TRAIN_ARCHIVE "mnist_train_lmdb.tar.bz2")
  set(MNIST_EXAMPLE_TEST_ARCHIVE "mnist_test_lmdb.tar.bz2")
  set(MNIST_EXAMPLE_TRAIN_OUT "mnist_train_lmdb")
  set(MNIST_EXAMPLE_TEST_OUT "mnist_test_lmdb")
  if (NOT EXISTS "${MNIST_EXAMPLE_PATH}")
    file(DOWNLOAD "http://juban.free.fr/dd/examples/caffe/mnist/mnist_train_lmdb.tar.bz2" "${MNIST_EXAMPLE_PATH}/${MNIST_EXAMPLE_TRAIN_ARCHIVE}")
    file(DOWNLOAD "http://juban.free.fr/dd/examples/caffe/mnist/mnist_test_lmdb.tar.bz2" "${MNIST_EXAMPLE_PATH}/${MNIST_EXAMPLE_TEST_ARCHIVE}")
    execute_process(COMMAND ${CMAKE_COMMAND} -E tar xvjf ${MNIST_EXAMPLE_PATH}${MNIST_EXAMPLE_TRAIN_ARCHIVE} -C ${MNIST_EXAMPLE_PATH}) # XXX: output directory doesn't work here, maybe because of special cmake handling
    execute_process(COMMAND ${CMAKE_COMMAND} -E tar xvjf ${MNIST_EXAMPLE_PATH}${MNIST_EXAMPLE_TEST_ARCHIVE} -C ${MNIST_EXAMPLE_PATH})
    execute_process(COMMAND ${CMAKE_COMMAND} -E rename ${MNIST_EXAMPLE_TRAIN_OUT} ${MNIST_EXAMPLE_PATH}${MNIST_EXAMPLE_TRAIN_OUT})
    execute_process(COMMAND ${CMAKE_COMMAND} -E rename ${MNIST_EXAMPLE_TEST_OUT} ${MNIST_EXAMPLE_PATH}${MNIST_EXAMPLE_TEST_OUT})
  endif()
  add_executable(ut_caffeapi ut-caffeapi.cc)
  target_link_libraries(ut_caffeapi ${CAFFE_STATIC_LINK} ${CAFFE_LIB_DEPS} ${CUDA_LIB_DEPS} ddetect glog gflags gtest gtest_main opencv_core opencv_highgui opencv_imgproc)
  add_test(
    NAME ut_caffeapi
    COMMAND ut_caffeapi
    )
endif()