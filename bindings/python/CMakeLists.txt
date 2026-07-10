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
  INSTALL_RPATH "$ORIGIN;$ORIGIN/../torch/lib")

install(TARGETS _native LIBRARY DESTINATION deepdetect)
install(FILES
  deepdetect/cli/external-pytorch-detector-default.yaml
  deepdetect/cli/segformer-default.yaml
  deepdetect/cli/torchvision-detector-default.yaml
  deepdetect/cli/vitpose-default.yaml
  deepdetect/cli/yolox-default.yaml
  DESTINATION deepdetect/cli)

option(DEEPDETECT_PYTHON_BUNDLE_NATIVE
  "bundle the staged DeepDetect SDK shared libraries in the wheel" ON)

function(deepdetect_imported_location target output_variable)
  foreach(property_name
      IMPORTED_LOCATION
      IMPORTED_LOCATION_NOCONFIG
      IMPORTED_LOCATION_RELEASE
      IMPORTED_LOCATION_RELWITHDEBINFO
      IMPORTED_LOCATION_MINSIZEREL
      IMPORTED_LOCATION_DEBUG)
    get_target_property(location "${target}" "${property_name}")
    if (location)
      set("${output_variable}" "${location}" PARENT_SCOPE)
      return()
    endif()
  endforeach()
  message(FATAL_ERROR "Could not determine imported location for ${target}")
endfunction()

if (DEEPDETECT_PYTHON_BUNDLE_NATIVE)
  deepdetect_imported_location(DeepDetect::ddetect DEEPDETECT_LIBRARY)
  get_filename_component(DEEPDETECT_LIBRARY_DIR
    "${DEEPDETECT_LIBRARY}" DIRECTORY)
  get_filename_component(DEEPDETECT_LIBRARY_NAME
    "${DEEPDETECT_LIBRARY}" NAME)

  set(DEEPDETECT_BUNDLED_LIBRARIES "${DEEPDETECT_LIBRARY}")
  foreach(alias_name
      libdeepdetect.so.0
      libdeepdetect.so)
    if (EXISTS "${DEEPDETECT_LIBRARY_DIR}/${alias_name}"
        AND NOT "${alias_name}" STREQUAL "${DEEPDETECT_LIBRARY_NAME}")
      list(APPEND DEEPDETECT_BUNDLED_LIBRARIES
        "${DEEPDETECT_LIBRARY_DIR}/${alias_name}")
    endif()
  endforeach()

  foreach(torchvision_name
      libtorchvision.so
      libtorchvision.so.0)
    if (EXISTS "${DEEPDETECT_LIBRARY_DIR}/${torchvision_name}")
      list(APPEND DEEPDETECT_BUNDLED_LIBRARIES
        "${DEEPDETECT_LIBRARY_DIR}/${torchvision_name}")
    endif()
  endforeach()

  file(GLOB DEEPDETECT_PROTOBUF_LIBRARIES
    "${DEEPDETECT_LIBRARY_DIR}/libprotobuf*.so*"
    "${DEEPDETECT_LIBRARY_DIR}/libprotoc*.so*")
  list(APPEND DEEPDETECT_BUNDLED_LIBRARIES
    ${DEEPDETECT_PROTOBUF_LIBRARIES})

  if (DEEPDETECT_BUNDLED_LIBRARIES)
    list(REMOVE_DUPLICATES DEEPDETECT_BUNDLED_LIBRARIES)
    install(FILES ${DEEPDETECT_BUNDLED_LIBRARIES}
      DESTINATION deepdetect)
  endif()
endif()
