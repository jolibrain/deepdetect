# Isolate the upstream build from DeepDetect's private flags and definitions.
include(FetchContent)
set_directory_properties(PROPERTIES COMPILE_DEFINITIONS "" INCLUDE_DIRECTORIES "")
set(CMAKE_CXX_FLAGS "")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(NCNN_VERSION 20261020)
set(NCNN_VULKAN OFF)
set(NCNN_BUILD_TESTS OFF)
set(NCNN_BUILD_EXAMPLES OFF)
set(NCNN_BUILD_TOOLS OFF)
set(NCNN_BUILD_BENCHMARK OFF)
set(NCNN_INSTALL_SDK OFF)
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
FetchContent_Declare(ncnn
  URL https://github.com/Tencent/ncnn/releases/download/20261020/ncnn-20261020-full-source.zip
  URL_HASH SHA256=699d9acc67489eee0993a551ca1ebc247bbf0f4e33babe1dec810e6a9a52b988)
FetchContent_MakeAvailable(ncnn)
