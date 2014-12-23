set(CAFFE_STATIC_LINK -Wl,--whole-archive /home/beniz/projects/deepdetect/apps/caffe/build/lib/libcaffe.a -Wl,--no-whole-archive)
link_directories(${CAFFE_LIB_DIR})
set(CUDA_LIB_DEPS -lcudart -lcublas -lcurand -lcudnn)
set(CAFFE_LIB_DEPS -lprotobuf -lleveldb -lsnappy -llmdb -lboost_system -lhdf5_hl -lhdf5 -lboost_thread -lopenblas)

add_executable (dd dd.cc)
target_link_libraries (dd ${CAFFE_STATIC_LINK} ${CAFFE_LIB_DEPS} ddetect glog gflags opencv_core opencv_highgui opencv_imgproc)