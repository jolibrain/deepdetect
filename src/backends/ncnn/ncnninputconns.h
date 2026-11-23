/**
 * DeepDetect
 * Copyright (c) 2018 Jolibrain
 * Author: Corentin Barreau <corentin.barreau@epitech.eu>
 *
 * This file is part of deepdetect.
 *
 * deepdetect is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * deepdetect is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with deepdetect.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef NCNNINPUTCONNS_H
#define NCNNINPUTCONNS_H

#include "imginputfileconn.h"
#include "net.h"
#include <cmath>

namespace dd
{
  class ImgNCNNInputFileConn : public ImgInputFileConn
  {
  public:
    int width() const
    {
      return _width;
    }
    int height() const
    {
      return _height;
    }
    void init(const APIData &ad)
    {
      ImgInputFileConn::init(ad);
      if (ad.has("ncnn_letterbox"))
        _letterbox = ad.get("ncnn_letterbox").get<bool>();
      if (ad.has("ncnn_normalize"))
        _normalize = ad.get("ncnn_normalize").get<bool>();
    }

    void transform(const APIData &ad)
    {
      fillup_parameters(ad.getobj("parameters").getobj("input"));
      int width = _width, height = _height;
      if (_letterbox)
        {
          if (width <= 0 || height <= 0 || _crop_width || _crop_height
              || _scaled)
            throw InputConnectorBadParamException(
                "NCNN letterbox requires fixed dimensions without crop or "
                "scale resizing");
          _width = _height = -1;
        }
      get_data(ad);
      ImgInputFileConn::transform(nullptr, false);
      for (auto &image : _images)
        {
          if (image.depth() != CV_8U
              || (image.channels() != 1 && image.channels() != 3))
            throw InputConnectorBadParamException(
                "NCNN expects an 8-bit gray or three-channel image");
          if (_letterbox)
            {
              double ratio = std::min(double(width) / image.cols,
                                      double(height) / image.rows);
              cv::Size size(std::max(1, int(image.cols * ratio)),
                            std::max(1, int(image.rows * ratio)));
              cv::Mat padded(height, width, image.type(),
                             cv::Scalar::all(114));
              cv::resize(image, padded(cv::Rect(cv::Point(), size)), size, 0,
                         0, cv::INTER_LINEAR);
              image = padded;
              _content_sizes.push_back(size);
            }
          else
            _content_sizes.push_back(image.size());
          int channels = image.channels();
          ncnn::Mat tensor = ncnn::Mat::from_pixels(
              image.data,
              channels == 1 ? ncnn::Mat::PIXEL_GRAY : ncnn::Mat::PIXEL_BGR,
              image.cols, image.rows, image.step);
          if (tensor.empty())
            throw InputConnectorBadParamException("empty NCNN input");
          if ((!_mean.empty() && int(_mean.size()) != channels)
              || (!_std.empty() && int(_std.size()) != channels))
            throw InputConnectorBadParamException(
                "NCNN mean and std must match the number of channels");
          if (_normalize)
            {
              if (!std::isfinite(_scale) || _scale <= 0)
                throw InputConnectorBadParamException(
                    "NCNN pixel scale must be positive");
              float mean[3], norm[3];
              for (int c = 0; c < channels; ++c)
                {
                  float deviation = _std.empty() ? 1.f : _std[c];
                  if (!std::isfinite(deviation) || deviation <= 0)
                    throw InputConnectorBadParamException(
                        "NCNN std must be positive");
                  mean[c] = _mean.empty() ? 0.f : _mean[c] / _scale;
                  norm[c] = _scale / deviation;
                }
              tensor.substract_mean_normalize(mean, norm);
            }
          else if (!_mean.empty())
            tensor.substract_mean_normalize(
                _mean.data(), _std.empty() ? nullptr : _std.data());
          else if (_scale != 1.)
            {
              float multiplier = 1.f / _scale;
              float norm[3] = { multiplier, multiplier, multiplier };
              tensor.substract_mean_normalize(nullptr, norm);
            }
          _in.push_back(std::move(tensor));
        }
    }
    std::vector<ncnn::Mat> _in;
    std::vector<cv::Size> _content_sizes;

  private:
    bool _letterbox = false;
    bool _normalize = false;
  };
}
#endif
