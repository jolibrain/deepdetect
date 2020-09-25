# main library
if (USE_DD_SYSLOG)
  add_definitions(-DUSE_DD_SYSLOG)
endif()

if (USE_HDF5)
  add_definitions(-DUSE_HDF5)
endif()

set(ddetect_SOURCES deepdetect.h deepdetect.cc mllibstrategy.h mlmodel.h
    mlservice.h inputconnectorstrategy.h imginputfileconn.h csvinputfileconn.h
    csvinputfileconn.cc csvtsinputfileconn.h csvtsinputfileconn.cc
    svminputfileconn.h svminputfileconn.cc txtinputfileconn.h
    txtinputfileconn.cc apidata.h apidata.cc chain_actions.h chain_actions.cc
    service_stats.h service_stats.cc chain.h chain.cc ext/rmustache/mustache.h ext/rmustache/mustache.cc)
if (USE_JSON_API)
  list(APPEND ddetect_SOURCES jsonapi.h jsonapi.cc)
endif()
if (USE_HTTP_SERVER)
  list(APPEND ddetect_SOURCES httpjsonapi.cc httpjsonapi.h)
endif()
if (USE_COMMAND_LINE AND USE_JSON_API)
  list(APPEND ddetect_SOURCES commandlinejsonapi.h commandlinejsonapi.cc)
endif()
if (USE_CAFFE)
  list(APPEND ddetect_SOURCES backends/caffe/caffelib.h backends/caffe/caffelib.cc backends/caffe/caffemodel.h backends/caffe/caffemodel.cc backends/caffe/caffeinputconns.h backends/caffe/caffeinputconns.cc generators/net_generator.h generators/net_caffe.h generators/net_caffe.cc generators/net_caffe_mlp.h generators/net_caffe_mlp.cc generators/net_caffe_convnet.h generators/net_caffe_convnet.cc generators/net_caffe_resnet.h generators/net_caffe_resnet.cc generators/net_caffe_recurrent.cc )
  if (USE_COMMAND_LINE)
    list(APPEND ddetect_SOURCES commandlineapi.h commandlineapi.cc)
  endif()
endif()
if (USE_TF)
  list(APPEND ddetect_SOURCES backends/tf/tflib.cc backends/tf/tflib.h backends/tf/tfmodel.cc backends/tf/tfmodel.h backends/tf/tfinputconns.h)
endif()
if (USE_CAFFE2)
  list(APPEND ddetect_SOURCES
    backends/caffe2/caffe2lib.cc
    backends/caffe2/caffe2model.cc
    backends/caffe2/caffe2inputinterface.cc
    backends/caffe2/caffe2inputimg.cc
    backends/caffe2/nettools/fillers_and_optimizers.cc
    backends/caffe2/nettools/devices_and_operators.cc
    backends/caffe2/nettools/workspace_and_nets.cc
    backends/caffe2/nettools/gradients.cc
    backends/caffe2/nettools/netgroup.cc
    backends/caffe2/nettools/internal.cc
    backends/caffe2/nettools/debug.cc
    )
endif()
if (USE_XGBOOST)
  list(APPEND ddetect_SOURCES backends/xgb/xgblib.cc backends/xgb/xgblib.h backends/xgb/xgbmodel.cc backends/xgb/xgbmodel.h backends/xgb/xgbinputconns.cc backends/xgb/xgbinputconns.h)
endif()
if (USE_TSNE)
  list(APPEND ddetect_SOURCES backends/tsne/tsneinputconns.h backends/tsne/tsneinputconns.cc backends/tsne/tsnemodel.h backends/tsne/tsnelib.h backends/tsne/tsnelib.cc)
endif()
if (USE_SIMSEARCH)
  list(APPEND ddetect_SOURCES simsearch.h simsearch.cc)
endif()
if (USE_DLIB)
  list(APPEND ddetect_SOURCES backends/dlib/DNNStructures.h backends/dlib/dliblib.cc backends/dlib/dliblib.h backends/dlib/dlibmodel.cc backends/dlib/dlibmodel.h backends/dlib/dlibinputconns.h backends/dlib/dlib_actions.cpp backends/dlib/dlib_actions.h)
endif()
if (USE_NCNN)
  set_source_files_properties(
    ${CMAKE_BINARY_DIR}/src/caffe.pb.cc
    ${CMAKE_BINARY_DIR}/src/caffe.pb.h
    PROPERTIES GENERATED TRUE)
  set_source_files_properties(
      ${CMAKE_BINARY_DIR}/src/caffe.pb.cc
      ${CMAKE_BINARY_DIR}/src/caffe.pb.h
      PROPERTIES COMPILE_FLAGS -Wno-unused-parameter)

  list(APPEND ddetect_SOURCES
    backends/ncnn/ncnnlib.cc
    backends/ncnn/ncnnmodel.cc
    backends/ncnn/caffe2ncnn.cc
    backends/ncnn/upgrade_proto.cpp
    )

  if (NOT USE_CAFFE)
    list(APPEND ddetect_SOURCES
      ${CMAKE_BINARY_DIR}/src/caffe.pb.cc
      )
  endif()
endif()
if (USE_TORCH)
  list(APPEND ddetect_SOURCES
	generators/net_caffe.cc
	generators/net_caffe_recurrent.cc
    backends/torch/torchlib.cc
    backends/torch/torchmodel.cc
    backends/torch/torchinputconns.cc
    backends/torch/db.cpp
    backends/torch/db_lmdb.cpp
    backends/torch/native/templates/nbeats.cc
	basegraph.cc
	caffegraphinput.cc
	backends/torch/torchgraphbackend.cc
	graph.cc
	)
  if (NOT EXISTS ${CMAKE_SOURCE_DIR}/src/caffe.proto)
	file(DOWNLOAD https://raw.githubusercontent.com/jolibrain/caffe/master/src/caffe/proto/caffe.proto ${CMAKE_SOURCE_DIR}/src/caffe.proto)
	execute_process(COMMAND LD_LIBRARY_PATH=${PROTOBUF_LIB_DIR}:$ENV{LD_LIBRARY_PATH} ${CMAKE_BINARY_DIR}/pytorch/src/pytorch/torch/bin/protoc
      --proto_path=${CMAKE_SOURCE_DIR}/src
      --cpp_out=${CMAKE_BINARY_DIR}/src ${CMAKE_SOURCE_DIR}/src/caffe.proto
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
      )
  endif()
	set_source_files_properties(
      ${CMAKE_BINARY_DIR}/src/caffe.pb.cc
      ${CMAKE_BINARY_DIR}/src/caffe.pb.h
      PROPERTIES GENERATED TRUE)
    set_source_files_properties(
      ${CMAKE_BINARY_DIR}/src/caffe.pb.cc
      ${CMAKE_BINARY_DIR}/src/caffe.pb.h
      PROPERTIES COMPILE_FLAGS -Wno-unused-parameter)

  if (NOT USE_CAFFE)
    list(APPEND ddetect_SOURCES
      ${CMAKE_BINARY_DIR}/src/caffe.pb.cc
      )
  endif()

endif()

if (USE_TENSORRT)

  set_source_files_properties(
    ${CMAKE_BINARY_DIR}/src/caffe.pb.cc
    ${CMAKE_BINARY_DIR}/src/caffe.pb.h
    PROPERTIES GENERATED TRUE)
  set_source_files_properties(
    ${CMAKE_BINARY_DIR}/src/caffe.pb.cc
    ${CMAKE_BINARY_DIR}/src/caffe.pb.h
    PROPERTIES COMPILE_FLAGS -Wno-unused-parameter)


  list(APPEND ddetect_SOURCES
    backends/tensorrt/tensorrtlib.h
    backends/tensorrt/tensorrtlib.cc
    backends/tensorrt/protoUtils.h
    backends/tensorrt/protoUtils.cc
    backends/tensorrt/tensorrtinputconns.h
    backends/tensorrt/tensorrtinputconns.cc
    backends/tensorrt/tensorrtmodel.h
    backends/tensorrt/tensorrtmodel.cc
    )

  if (NOT USE_CAFFE)
    list(APPEND ddetect_SOURCES
      ${CMAKE_BINARY_DIR}/src/caffe.pb.cc
      )
  endif()
endif()

add_library(ddetect ${ddetect_SOURCES})
if (USE_DLIB)
  add_dependencies(ddetect dlib)
endif()
if (USE_NCNN)
  add_dependencies(ddetect ncnn)
endif()
if (USE_CAFFE)
  add_dependencies(ddetect caffe_dd)
endif()
if (USE_TENSORRT)
  if (USE_CAFFE2)
    add_dependencies(ddetect pytorch)
  endif()
endif()
if (USE_TORCH)
  add_dependencies(ddetect pytorch)
endif()
if (USE_SIMSEARCH)
  if (USE_ANNOY)
    add_dependencies(ddetect annoy)
  else()
    add_dependencies(ddetect faisslib)
  endif()
endif()
if (USE_TENSORRT_OSS)
  add_dependencies(ddetect tensorrt-oss)
endif()
if (USE_XGBOOST)
  add_dependencies(ddetect xgboost)
endif()
if (USE_TNSE)
  add_dependencies(ddetect Multicore-TSNE)
endif()

add_library(staticjson ext/staticjson.cpp)
target_link_libraries(ddetect staticjson)
