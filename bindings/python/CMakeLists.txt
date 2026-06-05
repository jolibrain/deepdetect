cmake_minimum_required(VERSION 3.15)
project(deepdetect_python LANGUAGES CXX)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
find_package(DeepDetect CONFIG REQUIRED)

pybind11_add_module(_native MODULE src/native.cpp)
target_compile_features(_native PRIVATE cxx_std_17)
target_link_libraries(_native PRIVATE DeepDetect::ddetect)
set_target_properties(_native PROPERTIES
  BUILD_RPATH_USE_ORIGIN ON
  INSTALL_RPATH "$ORIGIN")

install(TARGETS _native LIBRARY DESTINATION deepdetect)
