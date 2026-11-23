#include "inference.h"
#include "segmentation.h"
#include <opencv2/imgproc.hpp>
#include <iostream>

namespace
{
  ncnn::Mat prepare_image(const cv::Mat &image, bool ade20k)
  {
    const int side = ade20k ? 512 : 480;
    cv::Mat resized;
    cv::resize(image, resized, { side, side }, 0, 0,
               ade20k ? cv::INTER_LINEAR : cv::INTER_CUBIC);
    auto input = ncnn::Mat::from_pixels(
        resized.data, ade20k ? ncnn::Mat::PIXEL_BGR2RGB : ncnn::Mat::PIXEL_BGR,
        side, side);
    const float scale = ade20k ? 1.f / 255.f : 0.0039f;
    const float mean[] = { 0.485f, 0.456f, 0.406f };
    const float stddev[] = { 0.229f, 0.224f, 0.225f };
    for (int c = 0; c < 3; ++c)
      {
        float *pixels = input.channel(c);
        for (int i = 0; i < side * side; ++i)
          pixels[i] = (pixels[i] * scale - mean[c]) / stddev[c];
      }
    return input;
  }
}

int main(int argc, char **argv)
{
  if (argc != 5)
    {
      std::cerr << "Usage: ncnn_segformer {deepdetect|ade20k} MODEL_DIR IMAGE "
                   "MASK.png\n";
      return 2;
    }
  try
    {
      std::string profile = argv[1];
      bool ade20k = profile == "ade20k";
      if (!ade20k && profile != "deepdetect")
        throw std::runtime_error("Choose deepdetect or ade20k");
      if (std::filesystem::path(argv[4]).extension() != ".png")
        throw std::runtime_error(
            "Use a .png output path to preserve class IDs");
      cv::setNumThreads(1);
      auto image = example::read_image(argv[3]);
      auto logits = example::infer(argv[2], prepare_image(image, ade20k));
      cv::Mat ids;
      dd::decode_segmentation(logits, ade20k ? 150 : 2, image.size(), ade20k)
          .ids.convertTo(ids, CV_8U);
      if (!cv::imwrite(argv[4], ids))
        throw std::runtime_error("Could not write mask");
    }
  catch (const std::exception &e)
    {
      std::cerr << e.what() << '\n';
      return 1;
    }
}
