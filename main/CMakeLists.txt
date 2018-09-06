include_directories(${COMMON_INCLUDE_DIRS})
link_directories(${COMMON_LINK_DIRS})
add_executable (dede dede.cc)
target_link_libraries(dede ${COMMON_LINK_LIBS} ${HTTP_LINK_LIBS})
