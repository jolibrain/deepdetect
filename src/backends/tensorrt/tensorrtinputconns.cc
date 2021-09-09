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
#ifdef USE_CUDA_CV
#include <opencv2/core/cuda_stream_accessor.hpp>
#endif

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
            ++offset;
          }

    if (_has_mean_scalar)
      {
        offset = channels * _height * _width * i;

        for (int c = 0; c < channels; ++c)
          for (int h = 0; h < _height; ++h)
            for (int w = 0; w < _width; ++w)
              {
                fbuf[offset] -= _mean[c];
                ++offset;
              }
      }

    if (has_std)
      {
        offset = channels * _height * _width * i;

        for (int c = 0; c < channels; ++c)
          for (int h = 0; h < _height; ++h)
            for (int w = 0; w < _width; ++w)
              {
                fbuf[offset] /= _std[c];
                ++offset;
              }
      }
  }

#ifdef USE_CUDA_CV
  void ImgTensorRTInputFileConn::GpuMatToRTBuffer(cv::cuda::GpuMat &img, int i)
  {
    cv::cuda::GpuMat converted;
    int channels = img.channels();

    if (_cuda_buf == nullptr)
      throw InputConnectorInternalException(
          "No cuda buffer available to copy the data");
    if (_has_mean_scalar && _mean.size() != size_t(channels))
      throw InputConnectorBadParamException(
          "mean vector be of size the number of channels ("
          + std::to_string(channels) + ")");

    if (!_std.empty() && _std.size() != size_t(channels))
      throw InputConnectorBadParamException(
          "std vector be of size the number of channels ("
          + std::to_string(channels) + ")");

    bool has_std = !_std.empty();

    // TODO use stream for asynchronous version
    // TODO Maybe preallocation too?
    img.convertTo(converted, CV_32F);

    std::vector<cv::cuda::GpuMat> vec_channels;
    cv::cuda::split(converted, vec_channels, *_cuda_stream);

    for (int c = 0; c < channels; ++c)
      {
        auto &channel = vec_channels.at(c);
        cv::cuda::multiply(channel, _scale, channel, 1, -1, *_cuda_stream);

        if (_has_mean_scalar)
          cv::cuda::add(channel, -_mean[c], channel, cv::noArray(), -1,
                        *_cuda_stream);
        if (has_std)
          cv::cuda::multiply(channel, 1.0 / _std[c], channel, 1, -1,
                             *_cuda_stream);

        int offset = _height * _width * (i * channels + c);
        cudaMemcpy2DAsync(_cuda_buf + offset, _width * sizeof(float),
                          channel.ptr<float>(), channel.step,
                          _width * sizeof(float), _height,
                          cudaMemcpyDeviceToDevice,
                          cv::cuda::StreamAccessor::getStream(*_cuda_stream));
      }
  }
#endif

  void ImgTensorRTInputFileConn::transform(
      oatpp::Object<DTO::ServicePredict> input_dto)
  {
    if (input_dto->has_mean_file)
      _logger->warn("mean file cannot be used with tensorRT backend");

    try
      {
        ImgInputFileConn::transform(input_dto);
      }
    catch (const std::exception &e)
      {
        throw;
      }

    // ids
    bool set_ids = false;
    if (this->_ids.empty())
      set_ids = true;
    size_t img_count =
#ifdef USE_CUDA_CV
        _cuda ? this->_cuda_images.size() :
#endif
              this->_images.size();

    for (size_t i = 0; i < img_count; i++)
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

    size_t img_count =
#ifdef USE_CUDA_CV
        _cuda ? this->_cuda_images.size() :
#endif
              this->_images.size();

    for (i = 0; i < batch_size && _batch_index < (int)img_count;
         i++, _batch_index++)
      {
#ifdef USE_CUDA_CV
        if (_cuda)
          {
            cv::cuda::GpuMat img = this->_cuda_images.at(_batch_index);
            GpuMatToRTBuffer(img, i);
          }
        else
#endif
          {
            cv::Mat img = this->_images.at(_batch_index);
            CVMatToRTBuffer(img, i);
          }
      }
    return i;
  }
}
