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
    if (_geometry)
      applyGeometry(src);

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
      if (selected_area.channels() == 3)
        cv::randu(selected_area,
                  cv::Scalar(_cutout_vl, _cutout_vl, _cutout_vl),
                  cv::Scalar(_cutout_vh, _cutout_vh, _cutout_vh));
      else
        cv::randu(selected_area, cv::Scalar(_cutout_vl),
                  cv::Scalar(_cutout_vh));
    }
  }

  void TorchImgRandAugCV::getEnlargedImage(const cv::Mat &in_img,
                                           cv::Mat &in_img_enlarged)
  {
    int pad_mode = cv::BORDER_REFLECT101;
    switch (_geometry_pad_mode)
      {
      case 1: // constant
        pad_mode = cv::BORDER_CONSTANT;
        break;
      case 2: // mirrored
        pad_mode = cv::BORDER_REFLECT101;
        break;
      case 3: // repeat nearest
        pad_mode = cv::BORDER_REPLICATE;
        break;
      default:
        break;
      }
    cv::copyMakeBorder(in_img, in_img_enlarged, in_img.rows, in_img.rows,
                       in_img.cols, in_img.cols, pad_mode);
  }

  void TorchImgRandAugCV::getQuads(const int &rows, const int &cols,
                                   cv::Point2f (&inputQuad)[4],
                                   cv::Point2f (&outputQuad)[4])
  {
    // The 4 points that select quadilateral on the input , from top-left in
    // clockwise order These four pts are the sides of the rect box used as
    // input
    float x0, x1, y0, y1;
    x0 = cols;
    x1 = 2 * cols - 1;
    y0 = rows;
    y1 = 2 * rows - 1;
    if (_geometry_zoom_out || _geometry_zoom_in)
      {
        bool zoom_in = _geometry_zoom_in;
        bool zoom_out = _geometry_zoom_out;
        if (_geometry_zoom_out && _geometry_zoom_in)
          {
            if (_bernouilli(_rnd_gen))
              zoom_in = false;
            else
              zoom_out = false;
          }

        float x0min, x0max, y0min, y0max;
        if (zoom_in)
          {
            x0max = cols + cols * _geometry_zoom_factor;
            y0max = rows + rows * _geometry_zoom_factor;
          }
        else
          {
            x0max = x0;
            y0max = y0;
          }
        if (zoom_out)
          {
            x0min = cols - cols * _geometry_zoom_factor;
            y0min = rows - rows * _geometry_zoom_factor;
          }
        else
          {
            x0min = x0;
            y0min = y0;
          }
        x0 = ((x0max - x0min) * _uniform_real_1(_rnd_gen) + x0min);
        x1 = 3 * cols - x0;
        y0 = ((y0max - y0min) * _uniform_real_1(_rnd_gen) + y0min);
        y1 = 3 * rows - y0;
      }

    inputQuad[0] = cv::Point2f(x0, y0);
    inputQuad[1] = cv::Point2f(x1, y0);
    inputQuad[2] = cv::Point2f(x1, y1);
    inputQuad[3] = cv::Point2f(x0, y1);

    // The 4 points where the mapping is to be done , from top-left in
    // clockwise order
    outputQuad[0] = cv::Point2f(0, 0);
    outputQuad[1] = cv::Point2f(cols - 1, 0);
    outputQuad[2] = cv::Point2f(cols - 1, rows - 1);
    outputQuad[3] = cv::Point2f(0, rows - 1);
    if (_geometry_persp_horizontal)
      {
        if (_bernouilli(_rnd_gen))
          {
            // seen from right
            outputQuad[0].y
                = rows * _geometry_persp_factor * _uniform_real_1(_rnd_gen);
            outputQuad[3].y = rows - outputQuad[0].y;
          }
        else
          {
            // seen from left
            outputQuad[1].y
                = rows * _geometry_persp_factor * _uniform_real_1(_rnd_gen);
            outputQuad[2].y = rows - outputQuad[1].y;
          }
      }
    if (_geometry_persp_vertical)
      {
        if (_bernouilli(_rnd_gen))
          {
            // seen from above
            outputQuad[3].x
                = cols * _geometry_persp_factor * _uniform_real_1(_rnd_gen);
            outputQuad[2].x = cols - outputQuad[3].x;
          }
        else
          {
            // seen from below
            outputQuad[0].x
                = cols * _geometry_persp_factor * _uniform_real_1(_rnd_gen);
            outputQuad[1].x = cols - outputQuad[0].x;
          }
      }
  }

  void TorchImgRandAugCV::applyGeometry(cv::Mat &src)
  {
    // enlarge image
    float g1 = 0.0;
#pragma omp critical
    {
      g1 = _uniform_real_1(_rnd_gen);
    }
    if (g1 > _geometry)
      return;

    cv::Mat src_enlarged;
    getEnlargedImage(src, src_enlarged);

    // Input Quadilateral or Image plane coordinates
    cv::Point2f inputQuad[4];
    // Output Quadilateral or World plane coordinates
    cv::Point2f outputQuad[4];

    // get perpective matrix
#pragma omp critical
    {
      getQuads(src.rows, src.cols, inputQuad, outputQuad);
    }

    // warp perspective
    cv::Mat lambda = cv::getPerspectiveTransform(inputQuad, outputQuad);
    cv::warpPerspective(src_enlarged, src, lambda, src.size());
  }
}
