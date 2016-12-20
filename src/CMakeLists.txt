# main library
if (NOT CUDA_FOUND)
  add_definitions(-DCPU_ONLY)
endif()

set(ddetect_SOURCES deepdetect.h deepdetect.cc caffelib.h caffelib.cc mllibstrategy.h mlmodel.h mlservice.h caffemodel.h caffemodel.cc inputconnectorstrategy.h imginputfileconn.h csvinputfileconn.h csvinputfileconn.cc svminputfileconn.h svminputfileconn.cc txtinputfileconn.h txtinputfileconn.cc caffeinputconns.h caffeinputconns.cc commandlineapi.h commandlineapi.cc commandlinejsonapi.h commandlinejsonapi.cc apidata.h apidata.cc jsonapi.h jsonapi.cc httpjsonapi.cc httpjsonapi.h ext/rmustache/mustache.h ext/rmustache/mustache.cc net_generator.h net_caffe.h net_caffe.cc net_caffe_mlp.h net_caffe_mlp.cc net_caffe_convnet.h net_caffe_convnet.cc)
if (USE_TF)
  list(APPEND ddetect_SOURCES tflib.cc tflib.h tfmodel.cc tfmodel.h tfinputconns.h)
endif()
if (USE_XGBOOST)
  list(APPEND ddetect_SOURCES xgblib.cc xgblib.h xgbmodel.cc xgbmodel.h xgbinputconns.cc xgbinputconns.h)
endif()
add_library(ddetect ${ddetect_SOURCES})
