set(CAFFE_STATIC_LINK -Wl,--whole-archive /home/beniz/projects/deepdetect/apps/caffe/build/lib/libcaffe.a -Wl,--no-whole-archive)
if (CUDA_FOUND)
  set(CUDA_LIB_DEPS -lcudart -lcublas -lcurand -lcudnn)
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

  add_executable(ut_jsonapi ut-jsonapi.cc)
  target_link_libraries(ut_jsonapi ${CAFFE_STATIC_LINK} ${CAFFE_LIB_DEPS} ${CUDA_LIB_DEPS} ddetect glog gflags gtest gtest_main opencv_core opencv_highgui opencv_imgproc)
  add_test(
    NAME ut_jsonapi
    COMMAND ut_jsonapi
    )

  add_executable(ut_caffeapi ut-caffeapi.cc)
  target_link_libraries(ut_caffeapi ${CAFFE_STATIC_LINK} ${CAFFE_LIB_DEPS} ${CUDA_LIB_DEPS} ddetect glog gflags gtest gtest_main opencv_core opencv_highgui opencv_imgproc)
  add_test(
    NAME ut_caffeapi
    COMMAND ut_caffeapi
    )
endif()