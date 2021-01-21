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

#include "torchdataaug.h"

namespace dd
{

  void TorchImgRandAugCV::augment(cv::Mat &src)
  {
    // apply augmentation
    if (_mirror)
      applyMirror(src);
    if (_rotate)
      applyRotate(src);

    // should be last, in this order
    if (_cutout > 0.0)
      applyCutout(src);
    if (_crop_size > 0)
      applyCrop(src);
  }

  void TorchImgRandAugCV::applyMirror(cv::Mat &src)
  {
#pragma omp critical
    {
      if (_bernouilli(_rnd_gen))
        {
          cv::Mat dst;
          cv::flip(src, dst, 1);
          src = dst;
        }
    }
  }

  void TorchImgRandAugCV::applyRotate(cv::Mat &src)
  {
    int rot = 0;
#pragma omp critical
    {
      rot = _uniform_int_rotate(_rnd_gen);
    }
    if (rot == 0)
      return;
    else if (rot == 1) // 90
      {
        cv::Mat dst;
        cv::transpose(src, dst);
        cv::flip(dst, src, 1);
      }
    else if (rot == 2) // 180
      {
        cv::Mat dst;
        cv::flip(src, dst, -1);
        src = dst;
      }
    else if (rot == 3) // 270
      {
        cv::Mat dst;
        cv::transpose(src, dst);
        cv::flip(dst, src, 0);
      }
  }

  void TorchImgRandAugCV::applyCrop(cv::Mat &src)
  {
    int crop_x = 0;
    int crop_y = 0;
#pragma omp critical
    {
      crop_x = _uniform_int_crop_x(_rnd_gen);
      crop_y = _uniform_int_crop_y(_rnd_gen);
    }
    cv::Rect crop(crop_x, crop_y, _crop_size, _crop_size);
    cv::Mat dst = src(crop).clone();
    src = dst;
  }

  void TorchImgRandAugCV::applyCutout(cv::Mat &src)
  {
    // Draw random between 0 and 1
    float r1 = 0.0;
#pragma omp critical
    {
      r1 = _uniform_real_1(_rnd_gen);
    }
    if (r1 > _cutout)
      return;

#pragma omp critical
    {
      // get shape and area to erase
      float s = _uniform_real_cutout_s(_rnd_gen) * _img_width
                * _img_height;                    // area
      float r = _uniform_real_cutout_r(_rnd_gen); // aspect ratio

      int w = std::min(_img_width,
                       static_cast<int>(std::floor(std::sqrt(s / r))));
      int h = std::min(_img_height,
                       static_cast<int>(std::floor(std::sqrt(s * r))));
      std::uniform_int_distribution<int> distx(0, _img_width - w);
      std::uniform_int_distribution<int> disty(0, _img_height - h);
      int rect_x = distx(_rnd_gen);
      int rect_y = disty(_rnd_gen);

      // erase
      cv::Rect rect(rect_x, rect_y, w, h);
      cv::Mat selected_area = src(rect);
      cv::randu(selected_area, cv::Scalar(_cutout_vl, _cutout_vl, _cutout_vl),
                cv::Scalar(_cutout_vh, _cutout_vh, _cutout_vh)); // TODO: bw
    }
  }
}
