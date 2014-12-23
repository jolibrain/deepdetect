set(CAFFE_STATIC_LINK -Wl,--whole-archive /home/beniz/projects/deepdetect/apps/caffe/build/lib/libcaffe.a -Wl,--no-whole-archive)
set(CAFFE_LIB_DEPS -lprotobuf -lleveldb -lsnappy -llmdb -lboost_system -lhdf5_hl -lhdf5 -lboost_thread -lopenblas)
add_executable(ut_apidata ut-apidata.cc)
target_link_libraries(ut_apidata ${CAFFE_STATIC_LINK} ${CAFFE_LIB_DEPS} ddetect glog gflags gtest gtest_main opencv_core opencv_highgui opencv_imgproc)
add_test(
  NAME ut_apidata
  COMMAND ut_apidata
)