// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <mat.h>
#include <opencv2/imgproc.hpp>
#include <limits>
#include <stdexcept>

namespace dd
{
  struct NCNNSegmentation
  {
    cv::Mat ids;
    cv::Mat confidence;
  };

  // NCNN exposes one unpacked channel per class. Process the channels in
  // order so ties go to the lowest class ID, as in torch.argmax.
  inline NCNNSegmentation decode_segmentation(const ncnn::Mat &logits,
                                              int classes, cv::Size original,
                                              bool resize_logits,
                                              bool return_confidence = false,
                                              float threshold = 0.f)
  {
    if (logits.empty() || logits.dims != 3 || logits.c != classes
        || logits.elempack != 1 || logits.elemsize != sizeof(float))
      throw std::invalid_argument(
          "expected unpacked [classes,height,width] segmentation logits");

    const cv::Size grid
        = resize_logits ? original : cv::Size(logits.w, logits.h);
    NCNNSegmentation result{ cv::Mat::zeros(grid, CV_32S), {} };
    cv::Mat best(grid, CV_32F,
                 cv::Scalar(-std::numeric_limits<float>::infinity()));
    cv::Mat resized;
    auto plane = [&](int c) {
      cv::Mat scores(logits.h, logits.w, CV_32F, logits.channel(c).data);
      if (scores.size() == grid)
        return scores;
      cv::resize(scores, resized, grid, 0, 0, cv::INTER_LINEAR);
      return resized;
    };
    for (int c = 0; c < classes; ++c)
      {
        cv::Mat scores = plane(c);
        if (!cv::checkRange(scores))
          throw std::invalid_argument("non-finite NCNN segmentation logits");
        cv::Mat wins = scores > best;
        scores.copyTo(best, wins);
        result.ids.setTo(c, wins);
      }
    if (return_confidence || threshold > 0)
      {
        // Probability of the winning class is 1 / sum(exp(logit - max)).
        cv::Mat sum = cv::Mat::zeros(grid, CV_32F), exponential;
        for (int c = 0; c < classes; ++c)
          {
            cv::exp(plane(c) - best, exponential);
            sum += exponential;
          }
        cv::Mat probability = 1.0 / sum;
        if (return_confidence)
          result.confidence = probability;
        else
          result.ids.setTo(0, probability < threshold);
      }
    if (grid != original)
      {
        cv::resize(result.ids, result.ids, original, 0, 0, cv::INTER_NEAREST);
        if (return_confidence)
          cv::resize(result.confidence, result.confidence, original, 0, 0,
                     cv::INTER_LINEAR);
      }
    return result;
  }
}
