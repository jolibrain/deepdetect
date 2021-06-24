// tensorrtinputconns.cc ---

// Copyright (C) 2019 Jolibrain http://www.jolibrain.com

// Author: Guillaume Infantes <guillaume.infantes@jolibrain.com>

// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation; either version 3
// of the License, or (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program. If not, see <http://www.gnu.org/licenses/>.

#include "tensorrtinputconns.h"
#include "half.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/math/cstdfloat/cstdfloat_types.hpp>

namespace dd
{

  void ImgTensorRTInputFileConn::CVMatToRTBuffer(cv::Mat &img, int i)
  {
    cv::Mat converted;
    int channels = img.channels();

    if (_has_mean_scalar && _mean.size() != size_t(channels))
      throw InputConnectorBadParamException(
          "mean vector be of size the number of channels ("
          + std::to_string(channels) + ")");

    if (!_std.empty() && _std.size() != size_t(channels))
      throw InputConnectorBadParamException(
          "std vector be of size the number of channels ("
          + std::to_string(channels) + ")");

    bool has_std = !_std.empty();
    int offset = channels * _height * _width * i;
    img.convertTo(converted, CV_32F);
    boost::float32_t *fbuf = (boost::float32_t *)(_buf.data());
    boost::float32_t *cvbuf = (boost::float32_t *)converted.data;

    for (int c = 0; c < channels; ++c)
      for (int h = 0; h < _height; ++h)
        for (int w = 0; w < _width; ++w)
          {
            fbuf[offset]
                = _scale * cvbuf[(converted.cols * h + w) * channels + c];
            if (_has_mean_scalar)
              fbuf[offset] -= _mean[c];
            if (has_std)
              fbuf[offset] /= _std[c];
            ++offset;
          }
  }

  void ImgTensorRTInputFileConn::transform(const APIData &ad)
  {

    if (ad.has("has_mean_file"))
      _logger->warn("mean file cannot be used with tensorRT backend");

    try
      {
        ImgInputFileConn::transform(ad);
      }
    catch (const std::exception &e)
      {
        throw;
      }

    // ids
    bool set_ids = false;
    if (this->_ids.empty())
      set_ids = true;

    for (int i = 0; i < (int)this->_images.size(); i++)
      {
        if (set_ids)
          this->_ids.push_back(this->_uris.at(i));
        _imgs_size.insert(std::pair<std::string, std::pair<int, int>>(
            this->_ids.at(i), this->_images_size.at(i)));
      }
    _batch_size = this->_images.size();
    _batch_index = 0;
  }

  int ImgTensorRTInputFileConn::process_batch(const unsigned int batch_size)
  {
    if (batch_size < this->_images.size())
      this->_logger->warn("you are giving  more images than max_batch_size, "
                          "please double check");
    unsigned int i;
    if (_bw)
      _buf.resize(batch_size * height() * width());
    else
      _buf.resize(batch_size * 3 * height() * width());
    for (i = 0; i < batch_size && _batch_index < (int)this->_images.size();
         i++, _batch_index++)
      {
        cv::Mat img = this->_images.at(_batch_index);
        CVMatToRTBuffer(img, i);
      }
    return i;
  }

}
