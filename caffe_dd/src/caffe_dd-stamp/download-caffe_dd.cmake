message(STATUS "downloading...
     src='https://github.com/beniz/caffe/archive/master.tar.gz'
     dst='/home/jsaksris/whitefish/deepdetect/caffe_dd/src/master.tar.gz'
     timeout='none'")




file(DOWNLOAD
  "https://github.com/beniz/caffe/archive/master.tar.gz"
  "/home/jsaksris/whitefish/deepdetect/caffe_dd/src/master.tar.gz"
  SHOW_PROGRESS
  # no TIMEOUT
  STATUS status
  LOG log)

list(GET status 0 status_code)
list(GET status 1 status_string)

if(NOT status_code EQUAL 0)
  message(FATAL_ERROR "error: downloading 'https://github.com/beniz/caffe/archive/master.tar.gz' failed
  status_code: ${status_code}
  status_string: ${status_string}
  log: ${log}
")
endif()

message(STATUS "downloading... done")
