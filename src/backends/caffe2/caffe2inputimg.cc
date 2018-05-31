/**
 * DeepDetect
 * Copyright (c) 2018 Jolibrain
 * Author: Julien Chicha
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

#include "backends/caffe2/caffe2inputconns.h"

namespace dd {

  void ImgCaffe2InputFileConn::init(const APIData &ad) {
    ImgInputFileConn::init(ad);
    if (ad.has("std"))
      _std = ad.get("std").get<float>();
  }

  void ImgCaffe2InputFileConn::transform(const APIData &ad) {
    if (_train) {
      transform_train(ad);
    } else {
      transform_predict(ad);
    }
  }

  void ImgCaffe2InputFileConn::transform_predict(const APIData &ad) {
    try {
      ImgInputFileConn::transform(ad);
    } catch (InputConnectorBadParamException &e) {
      throw;
    }
    _ids = _uris;
  }

  void ImgCaffe2InputFileConn::transform_train(const APIData &) {
    //TODO
  }

  int ImgCaffe2InputFileConn::get_tensor_test(caffe2::TensorCPU &tensor, int num) {

    int image_count = _images.size();
    if (!image_count) {
      return 0; // No more data
    }
    int w = _images[0].cols;
    int h = _images[0].rows;
    if (image_count > num && num > 0) {
      image_count = num; // Cap the batch size to 'num'
    }

    // Resize the tensor
    std::vector<cv::Mat> chan(channels());
    tensor.Resize(std::vector<caffe2::TIndex>({image_count, channels(), h, w}));
    size_t channel_size = h * w * sizeof(float);
    uint8_t *data = reinterpret_cast<uint8_t *>(tensor.mutable_data<float>());

    auto it_begin(_images.begin());
    auto it_end(it_begin + image_count);
    for (auto it = it_begin; it < it_end; ++it) {

      // Convert from NHWC uint8_t to NCHW float
      it->convertTo(*it, CV_32F);
      cv::split(*it / _std, chan);
      for (cv::Mat &ch : chan) {
	std::memcpy(data, ch.data, channel_size);
	data += channel_size;
      }

    }
    _images.erase(it_begin, it_end);
    return image_count;
  }
}
