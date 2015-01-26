# main library
if (NOT CUDA_FOUND)
  add_definitions(-DCPU_ONLY)
endif()

add_library(ddetect deepdetect.h deepdetect.cc caffelib.h caffelib.cc mllibstrategy.h mlmodel.h mlservice.h caffemodel.h caffemodel.cc inputconnectorstrategy.h imginputfileconn.h caffeinputconns.h commandlineapi.h commandlineapi.cc commandlinejsonapi.h commandlinejsonapi.cc apidata.h apidata.cc jsonapi.h jsonapi.cc ext/rmustache/mustache.h ext/rmustache/mustache.cc)