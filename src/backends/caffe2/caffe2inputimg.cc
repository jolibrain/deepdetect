/**
 * DeepDetect
 * Copyright (c) 2014-2016 Emmanuel Benazera
 * Author: Emmanuel Benazera <beniz@droidnik.fr>
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

namespace dd
{
  void ImgCaffe2InputFileConn::init(const APIData &ad)
  {
    ImgInputFileConn::init(ad);
  }

  void ImgCaffe2InputFileConn::transform(const APIData &ad) {
    try {
      ImgInputFileConn::transform(ad);
    } catch (InputConnectorBadParamException &e) {
      throw;
    }
    _ids = _uris;
  }

  int ImgCaffe2InputFileConn::get_tensor_test(caffe2::TensorCPU &tensor, int num) {
    if (!_images.size()) {
      return 0;
    }
    auto w = _images[0].cols;
    auto h = _images[0].rows;
    tensor.Resize(std::vector<caffe2::TIndex>({num, 3, h, w}));
    auto data = tensor.mutable_data<float>();
    std::fill(data, data + tensor.size(), 0);
    auto idx(0), size(0);
    for (auto it = _images.begin(); num - size && it != _images.end(); ++size) {
      std::vector<cv::Mat> chan(3);
      cv::split(*it, chan);
      it = _images.erase(it);
      for (auto &ch : chan) {
	for (auto row = 0; row < h; ++row) {
	  for (auto col = 0; col < w; ++col) {
	    data[idx++] = ch.at<unsigned char>(row, col) / 255.0;
	  }
	}
      }
    }
    return size;
  }
}
