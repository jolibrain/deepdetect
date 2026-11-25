include_directories(${COMMON_INCLUDE_DIRS})
link_directories(${COMMON_LINK_DIRS})

if(USE_NCNN)
  add_subdirectory(ncnn)
endif()
