link_directories(${CAFFE_LIB_DIR})
if (CUDA_FOUND)
  set(CUDA_LIB_DEPS -lcudart -lcublas -lcurand)# -lcudnn) #TODO: cudnn
else()
  add_definitions(-DCPU_ONLY)
endif()
set(CAFFE_LIB_DEPS -lleveldb -lsnappy -llmdb -lboost_system -lhdf5_hl -lhdf5 -lboost_thread -lopenblas -lcaffe -lprotobuf)

add_executable (dede dede.cc)
target_link_libraries (dede ddetect glog gflags opencv_core opencv_highgui opencv_imgproc cppnetlib-uri curlpp curl crypto ssl ${CAFFE_LIB_DEPS})