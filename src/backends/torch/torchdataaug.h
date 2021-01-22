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

#include <opencv2/opencv.hpp>
#include <random>

namespace dd
{
  class TorchImgRandAugCV
  {
  public:
    TorchImgRandAugCV()
    {
    }

    TorchImgRandAugCV(const int &img_width, const int &img_height,
                      const bool &mirror, const bool &rotate,
                      const int &crop_size, const bool &cutout)
        : _img_width(img_width), _img_height(img_height), _mirror(mirror),
          _rotate(rotate), _crop_size(crop_size), _cutout(cutout),
          _uniform_real_1(0.0, 1.0), _bernouilli(0.5),
          _uniform_int_rotate(0, 3)
    {
      if (_crop_size > 0)
        {
          _uniform_int_crop_x
              = std::uniform_int_distribution<int>(0, _img_width - _crop_size);
          _uniform_int_crop_y = std::uniform_int_distribution<int>(
              0, _img_height - _crop_size);
        }
      if (_cutout > 0.0)
        {
          _uniform_real_cutout_s
              = std::uniform_real_distribution<float>(_cutout_sl, _cutout_sh);
          _uniform_real_cutout_r
              = std::uniform_real_distribution<float>(_cutout_rl, _cutout_rh);
        }
    }

    ~TorchImgRandAugCV()
    {
    }

    void augment(cv::Mat &src);

  protected:
    void applyMirror(cv::Mat &src);
    void applyRotate(cv::Mat &src);
    void applyCrop(cv::Mat &src);
    void applyCutout(cv::Mat &src);

  private:
    int _img_width = 224;
    int _img_height = 224;

    // augmentation options & parameter
    bool _mirror = false;
    bool _rotate = false;
    int _crop_size = -1;
    float _cutout = 0.0;
    float _cutout_sl = 0.02; /**< min proportion of erased area wrt image. */
    float _cutout_sh = 0.4;  /**< max proportion of erased area wrt image. */
    float _cutout_rl = 0.3;  /**< min aspect ratio of erased area. */
    float _cutout_rh = 3.0;  /**< max aspect ratio of erased area. */
    int _cutout_vl = 0;      /**< min erased area pixel value. */
    int _cutout_vh = 255;    /**< max erased area pixel value. */

    // random generators
    std::default_random_engine _rnd_gen;
    std::uniform_real_distribution<float>
        _uniform_real_1; /**< random real uniform between 0 and 1. */
    std::bernoulli_distribution _bernouilli;
    std::uniform_int_distribution<int> _uniform_int_rotate;
    std::uniform_int_distribution<int> _uniform_int_crop_x;
    std::uniform_int_distribution<int> _uniform_int_crop_y;
    std::uniform_real_distribution<float> _uniform_real_cutout_s;
    std::uniform_real_distribution<float> _uniform_real_cutout_r;
  };
}

#endif
