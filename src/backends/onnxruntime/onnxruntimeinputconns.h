/**
* DeepDetect
 * Copyright (c) 2026 Pixel Forensics, Inc.
 * Author: Cheni Chadowitz <cchadowitz@pixelforensics.com>
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

#ifndef ONNXRUNTIMEINPUTCONNS_H
#define ONNXRUNTIMEINPUTCONNS_H

#include "imginputfileconn.h"

namespace dd
{
  class ONNXRuntimeInputInterface
  {
  public:
    ONNXRuntimeInputInterface()
    {
    }
    ONNXRuntimeInputInterface(const ONNXRuntimeInputInterface &i) : _buf(i._buf)
    {
    }

    virtual ~ONNXRuntimeInputInterface()
    {
    }

    bool _has_mean_file = false;
    std::vector<float> _buf;
    float *_cuda_buf = nullptr;

    float *data()
    {
      return _buf.data();
    }
  };

  class ImgONNXRuntimeInputFileConn : public ImgInputFileConn,
                                      public ONNXRuntimeInputInterface
  {
  public:
    ImgONNXRuntimeInputFileConn() : ImgInputFileConn()
    {
    }
    ImgONNXRuntimeInputFileConn(const ImgONNXRuntimeInputFileConn &i)
        : ImgInputFileConn(i), ONNXRuntimeInputInterface(i),
          _imgs_size(i._imgs_size)
    {
      this->_has_mean_file = i._has_mean_file;
    }
    virtual ~ImgONNXRuntimeInputFileConn()
    {
    }

    int channels() const { return this->_bw ? 1 : 3; }

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
    }

    void transform(oatpp::Object<DTO::ServicePredict> input_dto);

    int process_batch(const unsigned int batch_size);

    std::string _meanfname = "mean.binaryproto";
    std::string _correspname = "corresp.txt";
    int _batch_index = 0;
    int _batch_size = 0;
    std::unordered_map<std::string, std::pair<int, int>>
        _imgs_size; /**< Original image dimensions (rows, cols), used to rescale bounding boxes. */

  private:
    void CVMatToORTBuffer(cv::Mat &img, int i);

#ifdef USE_CUDA_CV
    void GpuMatToORTBuffer(cv::cuda::GpuMat &img, int i);
#endif
  };
}

#endif // ONNXRUNTIMEINPUTCONNS_H
