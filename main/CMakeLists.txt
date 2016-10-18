include_directories(${Boost_INCLUDE_DIRS})
link_directories(${CAFFE_LIB_DIR})
link_directories(${TF_LIB_DIR})
if (CUDA_FOUND)
  set(CUDA_LIB_DEPS -lcudart -lcublas -lcurand)# -lcudnn) #TODO: cudnn
else()
  add_definitions(-DCPU_ONLY)
endif()
set(CAFFE_LIB_DEPS -lleveldb -lsnappy -llmdb -lhdf5_hl -lhdf5 -lopenblas -lcaffe -lprotobuf)
set(TF_LIB_DEPS -ltensorflow ${CMAKE_BINARY_DIR}/tensorflow/src/tensorflow/dbuild/execroot/tensorflow/bazel-out/local_linux-opt/bin/tensorflow/cc/libcc_ops.lo ${CMAKE_BINARY_DIR}/tensorflow/src/tensorflow/dbuild/execroot/tensorflow/bazel-out/local_linux-opt/bin/tensorflow/cc/libconst_op.a ${CMAKE_BINARY_DIR}/tensorflow/src/tensorflow/dbuild/execroot/tensorflow/bazel-out/local_linux-opt/bin/tensorflow/cc/libops.a ${CMAKE_BINARY_DIR}/tensorflow/src/tensorflow/dbuild/execroot/tensorflow/bazel-out/local_linux-opt/bin/tensorflow/cc/libscope.a)

add_executable (dede dede.cc)
target_link_libraries (dede ddetect glog gflags opencv_core opencv_highgui opencv_imgproc cppnetlib-uri curlpp curl crypto ssl ${Boost_LIBRARIES} ${CAFFE_LIB_DEPS} ${TF_LIB_DEPS})
