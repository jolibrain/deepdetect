#include "inference.h"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <vector>

namespace
{
  constexpr int side = 640;
  constexpr float confidence_threshold = 0.25f;
  constexpr float nms_threshold = 0.45f;

  struct Detection
  {
    int class_id;
    float score;
    cv::Rect2f box;
  };

  std::vector<Detection> detections(const ncnn::Mat &rows)
  {
    if (rows.dims != 2 || rows.w < 6 || rows.h != 8400)
      throw std::runtime_error("Expected 8400 decoded YOLOX rows at 640x640");
    std::vector<Detection> candidates;
    for (int i = 0; i < rows.h; ++i)
      {
        const float *row = rows.row(i);
        const float *best = std::max_element(row + 5, row + rows.w);
        float score = row[4] * *best;
        if (score >= confidence_threshold && row[2] > 0 && row[3] > 0)
          candidates.push_back({ int(best - (row + 5)) + 1,
                                 score,
                                 { row[0] - row[2] / 2, row[1] - row[3] / 2,
                                   row[2], row[3] } });
      }

    std::stable_sort(candidates.begin(), candidates.end(),
                     [](const Detection &a, const Detection &b)
                       { return a.score > b.score; });
    std::vector<Detection> kept;
    for (const Detection &candidate : candidates)
      {
        auto suppresses = [&](const Detection &other)
          {
            if (candidate.class_id != other.class_id)
              return false;
            float intersection = (candidate.box & other.box).area();
            float union_area
                = candidate.box.area() + other.box.area() - intersection;
            return intersection > nms_threshold * union_area;
          };
        if (std::none_of(kept.begin(), kept.end(), suppresses))
          kept.push_back(candidate);
      }
    return kept;
  }
}

int main(int argc, char **argv)
{
  if (argc != 3)
    {
      std::cerr << "Usage: ncnn_yolox MODEL_DIR IMAGE\n";
      return 2;
    }
  try
    {
      cv::setNumThreads(1);
      auto image = example::read_image(argv[2]);
      double ratio = double(side) / std::max(image.cols, image.rows);
      cv::Size content(std::max(1, int(image.cols * ratio)),
                       std::max(1, int(image.rows * ratio)));
      cv::Mat padded(side, side, CV_8UC3, cv::Scalar::all(114));
      cv::resize(image, padded(cv::Rect(cv::Point(), content)), content, 0, 0,
                 cv::INTER_LINEAR);
      auto input = ncnn::Mat::from_pixels(padded.data, ncnn::Mat::PIXEL_BGR,
                                          side, side);
      auto rows = example::infer(argv[1], input);

      // Undo the resize using the actual integer dimensions before padding.
      auto x = [&](float value)
        {
          return std::clamp(value * (float(image.cols) / content.width), 0.f,
                            float(image.cols - 1));
        };
      auto y = [&](float value)
        {
          return std::clamp(value * (float(image.rows) / content.height), 0.f,
                            float(image.rows - 1));
        };
      std::cout << std::setprecision(9);
      for (const auto &d : detections(rows))
        std::cout << d.class_id << ' ' << d.score << ' ' << x(d.box.x) << ' '
                  << y(d.box.y) << ' ' << x(d.box.br().x) << ' '
                  << y(d.box.br().y) << '\n';
    }
  catch (const std::exception &e)
    {
      std::cerr << e.what() << '\n';
      return 1;
    }
}
