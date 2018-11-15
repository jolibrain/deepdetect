# main library
if (USE_DD_SYSLOG)
  add_definitions(-DUSE_DD_SYSLOG)
endif()

if (USE_HDF5)
  add_definitions(-DUSE_HDF5)
endif()

set(ddetect_SOURCES deepdetect.h deepdetect.cc mllibstrategy.h mlmodel.h mlservice.h inputconnectorstrategy.h imginputfileconn.h csvinputfileconn.h csvinputfileconn.cc svminputfileconn.h svminputfileconn.cc txtinputfileconn.h txtinputfileconn.cc apidata.h apidata.cc jsonapi.h jsonapi.cc httpjsonapi.cc httpjsonapi.h commandlinejsonapi.h commandlinejsonapi.cc ext/rmustache/mustache.h ext/rmustache/mustache.cc)
if (USE_CAFFE)
  list(APPEND ddetect_SOURCES backends/caffe/caffelib.h backends/caffe/caffelib.cc backends/caffe/caffemodel.h backends/caffe/caffemodel.cc backends/caffe/caffeinputconns.h backends/caffe/caffeinputconns.cc generators/net_generator.h generators/net_caffe.h generators/net_caffe.cc generators/net_caffe_mlp.h generators/net_caffe_mlp.cc generators/net_caffe_convnet.h generators/net_caffe_convnet.cc generators/net_caffe_resnet.h generators/net_caffe_resnet.cc commandlineapi.h commandlineapi.cc)
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
  list(APPEND ddetect_SOURCES backends/dlib/DNNStructures.h backends/dlib/dliblib.cc backends/dlib/dliblib.h backends/dlib/dlibmodel.cc backends/dlib/dlibmodel.h backends/dlib/dlibinputconns.h)
endif()
add_library(ddetect ${ddetect_SOURCES})
if (USE_CAFFE)
  add_dependencies(ddetect caffe_dd)
endif()
