/**
 * DeepDetect
 * Copyright (c) 2021 Jolibrain
 * Authors: Emmanuel Benazera <emmanuel.benazera@jolibrain.com>
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

#ifndef TORCHDATAAUG_H
#define TORCHDATAAUG_H

#include "imgdataaug.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <torch/torch.h>
#pragma GCC diagnostic pop

namespace dd
{
  class TorchImgRandAugCV : public ImgRandAugCV
  {
  public:
    TorchImgRandAugCV() : ImgRandAugCV()
    {
    }

    TorchImgRandAugCV(const bool &mirror, const bool &rotate,
                      const CropParams &crop_params,
                      const CutoutParams &cutout_params,
                      const GeometryParams &geometry_params,
                      const NoiseParams &noise_params,
                      const DistortParams &distort_params)
        : ImgRandAugCV(mirror, rotate, crop_params, cutout_params,
                       geometry_params, noise_params, distort_params)
    {
    }

    void augment_with_bbox(cv::Mat &src, std::vector<torch::Tensor> &targets);
    void augment_test_with_bbox(cv::Mat &src,
                                std::vector<torch::Tensor> &targets);
  };
}

#endif
