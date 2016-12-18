# main library
if (NOT CUDA_FOUND)
  add_definitions(-DCPU_ONLY)
endif()

set(ddetect_SOURCES deepdetect.h deepdetect.cc caffelib.h caffelib.cc mllibstrategy.h mlmodel.h mlservice.h caffemodel.h caffemodel.cc inputconnectorstrategy.h imginputfileconn.h csvinputfileconn.h csvinputfileconn.cc txtinputfileconn.h txtinputfileconn.cc caffeinputconns.h caffeinputconns.cc commandlineapi.h commandlineapi.cc commandlinejsonapi.h commandlinejsonapi.cc apidata.h apidata.cc jsonapi.h jsonapi.cc httpjsonapi.cc httpjsonapi.h ext/rmustache/mustache.h ext/rmustache/mustache.cc caffe_templates.cc caffe_templates.h)

#add_library(ddetect deepdetect.h deepdetect.cc caffelib.h caffelib.cc mllibstrategy.h mlmodel.h mlservice.h caffemodel.h caffemodel.cc inputconnectorstrategy.h imginputfileconn.h csvinputfileconn.h csvinputfileconn.cc txtinputfileconn.h txtinputfileconn.cc caffeinputconns.h caffeinputconns.cc commandlineapi.h commandlineapi.cc commandlinejsonapi.h commandlinejsonapi.cc apidata.h apidata.cc jsonapi.h jsonapi.cc httpjsonapi.cc httpjsonapi.h ext/rmustache/mustache.h ext/rmustache/mustache.cc)
if (USE_XGBOOST)
  list(APPEND ddetect_SOURCES xgblib.cc xgblib.h xgbmodel.cc xgbmodel.h xgbinputconns.cc xgbinputconns.h)
endif()
add_library(ddetect ${ddetect_SOURCES})
