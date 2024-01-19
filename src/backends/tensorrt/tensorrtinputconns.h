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
    TensorRTInputInterface(const TensorRTInputInterface &i)
        : _has_mean_file(i._has_mean_file), _bufs(i._bufs)
    {
    }

    ~TensorRTInputInterface()
    {
    }

    bool _has_mean_file = false; /**< image model mean.binaryproto. */
    std::vector<std::vector<float>> _bufs;
    std::vector<float *> _cuda_bufs;

    float *data(int id = 0)
    {
      return _bufs.at(id).data();
    }
    float *cuda_data(int id = 0)
    {
      return _cuda_bufs.at(id);
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
          _imgs_size(i._imgs_size), _mask_num_channels(i._mask_num_channels)
    {
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

    void transform(oatpp::Object<DTO::ServicePredict> input_dto);

    /** Load batch in memory, either CUDA or CPU */
    int process_batch(const unsigned int batch_size);

    std::string _meanfname = "mean.binaryproto";
    std::string _correspname = "corresp.txt";
    int _batch_index = 0;
    int _batch_size = 0;
    std::unordered_map<std::string, std::pair<int, int>>
        _imgs_size; /**< image sizes, used in detection. */
    std::vector<cv::Mat> _masks;
#ifdef USE_CUDA_CV
    std::vector<cv::cuda::GpuMat> _masks_cuda;
#endif
    /** number of channels of the mask required by the model */
    int _mask_num_channels = 3;

  private:
    /** Copy cv::Mat into a float buffer ready to be sent to the GPU
     * \param i id of the image in the batch
     * \param mask whether the cv::Mat is a mask. */
    void CVMatToRTBuffer(cv::Mat &img, size_t buf_id, int i,
                         bool mask = false);

#ifdef USE_CUDA_CV
    /** Copy GpuMat into cuda buffer
     * \param i id of the image in the batch
     * \param mask whether the cv::cuda::GpuMat is a mask. */
    void GpuMatToRTBuffer(cv::cuda::GpuMat &img, size_t buf_id, int i,
                          bool mask = false);
#endif
  };
}

#endif
