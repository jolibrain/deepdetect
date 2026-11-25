#pragma once

#include <net.h>
#include <opencv2/imgcodecs.hpp>
#include <filesystem>
#include <stdexcept>

namespace example
{
  inline cv::Mat read_image(const char *path)
  {
    cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
    if (image.empty())
      throw std::runtime_error(std::string("Cannot read image ") + path);
    return image;
  }

  // Each example runs a single image through a single-input, single-output
  // graph.
  inline ncnn::Mat infer(const std::filesystem::path &directory,
                         const ncnn::Mat &input)
  {
    ncnn::Net net;
    net.opt.num_threads = 1;
    net.opt.use_fp16_packed = false;
    net.opt.use_fp16_storage = false;
    net.opt.use_fp16_arithmetic = false;
    net.opt.use_bf16_storage = false;
    net.opt.use_local_pool_allocator = false;
    if (net.load_param((directory / "model.ncnn.param").c_str())
        || net.load_model((directory / "model.ncnn.bin").c_str()))
      throw std::runtime_error("Cannot load NCNN model in "
                               + directory.string());

    if (net.input_names().size() != 1 || net.output_names().size() != 1)
      throw std::runtime_error("Expected one input blob and one output blob");
    auto extractor = net.create_extractor();
    ncnn::Mat output;
    if (extractor.input(net.input_names().front(), input)
        || extractor.extract(net.output_names().front(), output)
        || output.empty())
      throw std::runtime_error("NCNN inference failed");
    if (output.elemsize != sizeof(float) || output.elempack != 1)
      throw std::runtime_error("Expected unpacked FP32 output");
    return output;
  }
}
