include_directories(${COMMON_INCLUDE_DIRS})
link_directories(${COMMON_LINK_DIRS})

if (APPLE)
  link_directories("/opt/homebrew/lib/")
endif()

if (USE_COMMAND_LINE OR USE_HTTP_SERVER OR USE_HTTP_SERVER_OATPP)
add_executable (dede dede.cc)
add_dependencies(dede protobuf)
target_link_libraries(dede ${COMMON_LINK_LIBS} ${HTTP_LIB_DEPS} ${OATPP_LIB_DEPS})
endif()
