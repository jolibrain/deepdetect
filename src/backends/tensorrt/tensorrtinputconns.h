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

#ifndef TENSORRTINPUTCONNS_H
#define TENSORRTINPUTCONNS_H

#include "imginputfileconn.h"
#include "csvtsinputfileconn.h"
#include "NvInfer.h"

namespace dd
{
  class TensorRTInputInterface
  {
  public:
    TensorRTInputInterface()
    {
    }
    TensorRTInputInterface(const TensorRTInputInterface &i) : _buf(i._buf)
    {
    }

    ~TensorRTInputInterface()
    {
    }

    bool _has_mean_file = false; /**< image model mean.binaryproto. */
    std::vector<float> _buf;

    float *data()
    {
      return _buf.data();
    }
  };

  class ImgTensorRTInputFileConn : public ImgInputFileConn,
                                   public TensorRTInputInterface
  {
  public:
    ImgTensorRTInputFileConn() : ImgInputFileConn()
    {
    }
    ImgTensorRTInputFileConn(const ImgTensorRTInputFileConn &i)
        : ImgInputFileConn(i), TensorRTInputInterface(i),
          _imgs_size(i._imgs_size)
    {
      this->_has_mean_file = i._has_mean_file;
    }
    ~ImgTensorRTInputFileConn()
    {
    }

    // for API info only
    int width() const
    {
      return _width;
    }

    // for API info only
    int height() const
    {
      return _height;
    }

    void init(const APIData &ad)
    {
      ImgInputFileConn::init(ad);
    }

    void transform(const APIData &ad);

    std::string _meanfname = "mean.binaryproto";
    std::string _correspname = "corresp.txt";
    int _batch_index = 0;
    int _batch_size = 0;
    int process_batch(const unsigned int batch_size);
    std::unordered_map<std::string, std::pair<int, int>>
        _imgs_size; /**< image sizes, used in detection. */

  private:
    void applyMeanToRTBuf(int channels, int i);
    void applyMeanToRTBuf(float *mean, int channels, int i);
    void CVMatToRTBuffer(cv::Mat &img, int i);
  };

}

#endif
