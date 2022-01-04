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
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <torch/torch.h>
#pragma GCC diagnostic pop
#include <random>

namespace dd
{
  class ImgAugParams
  {
  public:
    ImgAugParams() : _img_width(224), _img_height(224)
    {
    }

    ImgAugParams(const int &img_width, const int &img_height)
        : _img_width(img_width), _img_height(img_height)
    {
    }

    ~ImgAugParams()
    {
    }

    int _img_width;
    int _img_height;
  };

  class CropParams : public ImgAugParams
  {
  public:
    CropParams() : ImgAugParams()
    {
    }

    CropParams(const int &crop_size, const int &img_width,
               const int &img_height)
        : ImgAugParams(img_width, img_height), _crop_size(crop_size)
    {
      if (_crop_size > 0)
        {
          _uniform_int_crop_x
              = std::uniform_int_distribution<int>(0, _img_width - _crop_size);
          _uniform_int_crop_y = std::uniform_int_distribution<int>(
              0, _img_height - _crop_size);
        }
    }

    ~CropParams()
    {
    }

    // default params
    int _crop_size = -1;
    std::uniform_int_distribution<int> _uniform_int_crop_x;
    std::uniform_int_distribution<int> _uniform_int_crop_y;
  };

  class CutoutParams : public ImgAugParams
  {
  public:
    CutoutParams() : ImgAugParams()
    {
    }

    CutoutParams(const float &prob, const int &img_width,
                 const int &img_height)
        : ImgAugParams(img_width, img_height), _prob(prob)
    {
      _uniform_real_cutout_s
          = std::uniform_real_distribution<float>(_cutout_sl, _cutout_sh);
      _uniform_real_cutout_r
          = std::uniform_real_distribution<float>(_cutout_rl, _cutout_rh);
    }

    ~CutoutParams()
    {
    }

    // default params
    float _prob = 0.0;
    float _cutout_sl = 0.02; /**< min proportion of erased area wrt image. */
    float _cutout_sh = 0.4;  /**< max proportion of erased area wrt image. */
    float _cutout_rl = 0.3;  /**< min aspect ratio of erased area. */
    float _cutout_rh = 3.0;  /**< max aspect ratio of erased area. */
    int _cutout_vl = 0;      /**< min erased area pixel value. */
    int _cutout_vh = 255;    /**< max erased area pixel value. */

    // randomized params
    int _rect_x = 0;
    int _rect_y = 0;
    int _w = 0;
    int _h = 0;

    std::uniform_real_distribution<float> _uniform_real_cutout_s;
    std::uniform_real_distribution<float> _uniform_real_cutout_r;
  };

  class GeometryParams
  {
  public:
    GeometryParams()
    {
    }

    GeometryParams(const float &prob, const bool &geometry_persp_horizontal,
                   const bool &geometry_persp_vertical,
                   const bool &geometry_zoom_out, const bool &geometry_zoom_in,
                   const int &geometry_pad_mode)
        : _prob(prob), _geometry_persp_horizontal(geometry_persp_horizontal),
          _geometry_persp_vertical(geometry_persp_vertical),
          _geometry_zoom_out(geometry_zoom_out),
          _geometry_zoom_in(geometry_zoom_in),
          _geometry_pad_mode(geometry_pad_mode)
    {
    }

    ~GeometryParams()
    {
    }

    float _prob = 0.0;
    bool _geometry_persp_horizontal
        = true; /**< horizontal perspective change. */
    bool _geometry_persp_vertical = true; /**< vertical perspective change. */
    bool _geometry_zoom_out
        = true; /**< distance change: look from further away. */
    bool _geometry_zoom_in = true;      /**< distance change: look closer. */
    float _geometry_zoom_factor = 0.25; /**< zoom factor: 0.25 means that image
                                           can be *1.25 or /1.25. */
    float _geometry_persp_factor
        = 0.25;                     /**< persp factor: 0.25 means that new
                                      image corners  be in 1.25 or 0.75. */
    uint8_t _geometry_pad_mode = 1; /**< filling around images, 1: constant, 2:
                                       repeat nearest (replicate). */
    float _geometry_bbox_intersect
        = 0.75; /**< warped bboxes must at least have a 75% intersect with the
                   original bbox, otherwise they are filtered out.*/
    cv::Mat _lambda; /**< warp perspective matrix. */
  };

  class NoiseParams
  {
  public:
    NoiseParams()
    {
    }

    NoiseParams(const bool &hist_eq, const bool &inverse,
                const bool &decolorize, const bool &gauss_blur,
                const bool &jpg, const bool &posterize, const bool &erosion,
                const bool &saltpepper, const bool &clahe,
                const bool &convert_to_hsv, const bool &convert_to_lab)
        : _hist_eq(hist_eq), _inverse(inverse), _decolorize(decolorize),
          _gauss_blur(gauss_blur), _jpg(jpg), _posterize(posterize),
          _erosion(erosion), _saltpepper(saltpepper), _clahe(clahe),
          _convert_to_hsv(convert_to_hsv), _convert_to_lab(convert_to_lab)
    {
    }

    // default params
    float _prob = 0.0;                /**< effect probability. */
    bool _hist_eq = true;             /**< histogram equalized. */
    bool _inverse = true;             /**< color inversion. */
    bool _decolorize = true;          /**< grayscape. */
    bool _gauss_blur = true;          /**< Gaussian blur. */
    bool _jpg = true;                 /**< JPEG compression quality. */
    bool _posterize = true;           /**< posterization. */
    bool _erosion = true;             /**< erosion. */
    bool _saltpepper = true;          /**< salt & pepper. */
    float _saltpepper_fraction = 0.1; /**< percentage of pixels. */
    bool _clahe = true;               /**< local histogram equalizaion. */
    bool _convert_to_hsv = true;      /**< color space conversion. */
    bool _convert_to_lab = true;      /**< color space conversion. */
    bool _rgb = false;                /**< whether reference space is RGB. */
  };

  class DistortParams
  {
  public:
    DistortParams()
    {
    }

    DistortParams(const bool &brightness, const bool &contrast,
                  const bool &saturation, const bool &hue,
                  const bool &channel_order)
        : _brightness(brightness), _contrast(contrast),
          _saturation(saturation), _hue(hue), _channel_order(channel_order)
    {
      _uniform_real_brightness = std::uniform_real_distribution<float>(
          -_brightness_delta, _brightness_delta);
      _uniform_real_contrast = std::uniform_real_distribution<float>(
          _contrast_lower, _contrast_upper);
      _uniform_real_saturation = std::uniform_real_distribution<float>(
          _saturation_lower, _saturation_upper);
      _uniform_real_hue
          = std::uniform_real_distribution<float>(-_hue_delta, _hue_delta);
    }

    // default params
    float _prob = 0.0; /**< effect probability. */
    bool _brightness = true;
    float _brightness_delta = 32; /**< amount to add to the pixel values within
                                    [-delta, delta], in [0,255]. */
    bool _contrast = true;
    float _contrast_upper = 1.5;
    float _contrast_lower = 0.5;
    bool _saturation = true;
    float _saturation_upper = 1.5;
    float _saturation_lower = 0.5;
    bool _hue = true;
    float _hue_delta
        = 36; /**< amount to add to the hue channel, within [0,180]. */
    bool _channel_order = true;

    std::uniform_real_distribution<float> _uniform_real_brightness;
    std::uniform_real_distribution<float> _uniform_real_contrast;
    std::uniform_real_distribution<float> _uniform_real_saturation;
    std::uniform_real_distribution<float> _uniform_real_hue;
    bool _rgb = false; /**< whether reference space is RGB. */
  };

  class TorchImgRandAugCV
  {
  public:
    TorchImgRandAugCV()
    {
    }

    TorchImgRandAugCV(const bool &mirror, const bool &rotate,
                      const CropParams &crop_params,
                      const CutoutParams &cutout_params,
                      const GeometryParams &geometry_params,
                      const NoiseParams &noise_params,
                      const DistortParams &distort_params)
        : _mirror(mirror), _rotate(rotate), _crop_params(crop_params),
          _cutout_params(cutout_params), _geometry_params(geometry_params),
          _noise_params(noise_params), _distort_params(distort_params),
          _uniform_real_1(0.0, 1.0), _bernouilli(0.5),
          _uniform_int_rotate(0, 3)
    {
    }

    ~TorchImgRandAugCV()
    {
    }

    void augment(cv::Mat &src);
    void augment_with_bbox(cv::Mat &src, std::vector<torch::Tensor> &targets);
    void augment_with_segmap(cv::Mat &src, cv::Mat &tgt);

  protected:
    bool roll_weighted_dice(const float &prob);
    bool applyMirror(cv::Mat &src, const bool &sample = true);
    void applyMirrorBBox(std::vector<std::vector<float>> &bboxes,
                         const float &img_width);
    int applyRotate(cv::Mat &src, const bool &sample = true, int rot = 0);
    void applyRotateBBox(std::vector<std::vector<float>> &bboxes,
                         const float &img_width, const float &img_height,
                         const int &rot);
    bool applyCrop(cv::Mat &src, CropParams &cp, int &crop_x, int &crop_y,
                   const bool &sample = true);
    void applyCutout(cv::Mat &src, CutoutParams &cp,
                     const bool &store_rparams = false);
    void applyGeometry(cv::Mat &src, GeometryParams &cp,
                       const bool &store_rparams = false,
                       const bool &sample = true);
    void applyGeometryBBox(std::vector<std::vector<float>> &bboxes,
                           const GeometryParams &cp, const int &img_width,
                           const int &img_height);
    void applyNoise(cv::Mat &src);
    void applyDistort(cv::Mat &src);

  private:
    void getEnlargedImage(const cv::Mat &in_img, const GeometryParams &cp,
                          cv::Mat &in_img_enlarged);
    void getQuads(const int &rows, const int &cols, const GeometryParams &cp,
                  cv::Point2f (&inputQuad)[4], cv::Point2f (&outputQuad)[4]);
    void warpBBoxes(std::vector<std::vector<float>> &bboxes, cv::Mat lambda);
    void filterBBoxes(std::vector<std::vector<float>> &bboxes,
                      const GeometryParams &cp, const int &img_width,
                      const int &img_height);
    void applyNoiseDecolorize(cv::Mat &src);
    void applyNoiseGaussianBlur(cv::Mat &src);
    void applyNoiseHistEq(cv::Mat &src);
    void applyNoiseClahe(cv::Mat &src);
    void applyNoiseJPG(cv::Mat &src);
    void applyNoiseErosion(cv::Mat &src);
    void applyNoisePosterize(cv::Mat &src);
    void applyNoiseInverse(cv::Mat &src);
    void applyNoiseSaltpepper(cv::Mat &src);
    void applyNoiseConvertHSV(cv::Mat &src);
    void applyNoiseConvertLAB(cv::Mat &src);
    void applyDistortBrightness(cv::Mat &src);
    void applyDistortContrast(cv::Mat &src);
    void applyDistortSaturation(cv::Mat &src);
    void applyDistortHue(cv::Mat &src);
    void applyDistortOrderChannel(cv::Mat &src);

  private:
    // augmentation options & parameter
    bool _mirror = false;
    bool _rotate = false;

    CropParams _crop_params;
    CutoutParams _cutout_params;
    GeometryParams _geometry_params;
    NoiseParams _noise_params;
    DistortParams _distort_params;

    // random generators
    std::default_random_engine _rnd_gen;
    std::uniform_real_distribution<float>
        _uniform_real_1; /**< random real uniform between 0 and 1. */
    std::bernoulli_distribution _bernouilli;
    std::uniform_int_distribution<int> _uniform_int_rotate;
  };
}

#endif
