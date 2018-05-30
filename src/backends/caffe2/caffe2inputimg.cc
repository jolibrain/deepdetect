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

namespace dd
{
  void ImgCaffe2InputFileConn::init(const APIData &ad)
  {
    ImgInputFileConn::init(ad);
  }

  void ImgCaffe2InputFileConn::transform(const APIData &ad) {
    if (_train) {
      transform_train(ad);
    } else {
      transform_test(ad);
    }
  }

  void ImgCaffe2InputFileConn::transform_test(const APIData &ad) {
    try {
      ImgInputFileConn::transform(ad);
    } catch (InputConnectorBadParamException &e) {
      throw;
    }
    _ids = _uris;
  }

  void ImgCaffe2InputFileConn::transform_train(const APIData &) {
    _shuffle = true;
    //TODO
  }

  int ImgCaffe2InputFileConn::get_tensor_test(caffe2::TensorCPU &tensor, int num) {
    if (!_images.size()) {
      return 0; // No more data
    }
    int w = _images[0].cols;
    int h = _images[0].rows;
    if (num < 0) {
      num = _images.size(); // No fixed batch_size
    }

    // Resize and prefill with 0s
    tensor.Resize(std::vector<caffe2::TIndex>({num, 3, h, w}));
    float *data = tensor.mutable_data<float>();
    std::fill(data, data + tensor.size(), 0);

    int idx(0), size(0);
    for (auto it = _images.begin(); num - size && it != _images.end(); ++size) {
      std::vector<cv::Mat> chan(3);
      cv::split(*it, chan);
      it = _images.erase(it);
      for (cv::Mat &ch : chan) {
	for (int row = 0; row < h; ++row) {
	  for (int col = 0; col < w; ++col) {
	    data[idx++] = ch.at<unsigned char>(row, col) / 255.0;
	  }
	}
      }
    }
    return size;
  }
}
