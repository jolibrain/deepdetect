bin_PROGRAMS=dd
dd_SOURCES=dd.cc

CAFFE_STATIC_LINK=-Wl,--whole-archive,/home/beniz/projects/deepdetect/apps/caffe/build/lib/libcaffe.a,--no-whole-archive

AM_CPPFLAGS=-I../src -I/usr/local/cuda/include
AM_CXXFLAGS=-g -Wall -fopenmp -pthread -fPIC
AM_LDFLAGS=-L../src -L/usr/local/cuda/lib64 -L/usr/lib/x86_64-linux-gnu $(CAFFE_STATIC_LINK)
LDADD=-lddetect -lm $(GFLAGS_LIBS) -lopencv_core -lboost_regex -lglog -lcudart -lcublas -lcurand -lglog -lgflags -lprotobuf -lleveldb -lsnappy -llmdb -lboost_system -lhdf5_hl -lhdf5 -lm -lopencv_core -lopencv_highgui -lopencv_imgproc -lboost_thread -lopenblas
