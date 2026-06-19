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

#ifndef IMGDATAAUG_H
#define IMGDATAAUG_H

#include <opencv2/opencv.hpp>

#include <cstdint>
#include <random>
#include <string>
#include <vector>

#define DATAAUG_TEST_SEED 23124534

namespace dd
{
  class CropParams
  {
  public:
    CropParams()
    {
    }

    CropParams(const int &crop_size) : _crop_size(crop_size)
    {
    }

    ~CropParams()
    {
    }

    int _crop_size = -1;
    int _test_crop_samples = 1;
  };

  class CutoutParams
  {
  public:
    CutoutParams()
    {
    }

    CutoutParams(const float &prob) : _prob(prob)
    {
      _uniform_real_cutout_s
          = std::uniform_real_distribution<float>(_cutout_sl, _cutout_sh);
      _uniform_real_cutout_r
          = std::uniform_real_distribution<float>(_cutout_rl, _cutout_rh);
    }

    ~CutoutParams()
    {
    }

    float _prob = 0.0;
    float _cutout_sl = 0.02;
    float _cutout_sh = 0.4;
    float _cutout_rl = 0.3;
    float _cutout_rh = 3.0;
    int _cutout_vl = 0;
    int _cutout_vh = 255;

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
                   const bool &geometry_transl_horizontal,
                   const bool &geometry_transl_vertical,
                   const bool &geometry_zoom_out, const bool &geometry_zoom_in,
                   const std::string &geometry_pad_mode_str)
        : _prob(prob), _geometry_persp_horizontal(geometry_persp_horizontal),
          _geometry_persp_vertical(geometry_persp_vertical),
          _geometry_transl_horizontal(geometry_transl_horizontal),
          _geometry_transl_vertical(geometry_transl_vertical),
          _geometry_zoom_out(geometry_zoom_out),
          _geometry_zoom_in(geometry_zoom_in)
    {
      set_pad_mode(geometry_pad_mode_str);
    }

    ~GeometryParams()
    {
    }

    void set_pad_mode(const std::string &geometry_pad_mode_str)
    {
      if (geometry_pad_mode_str == "constant")
        _geometry_pad_mode = 1;
      else if (geometry_pad_mode_str == "mirrored")
        _geometry_pad_mode = 2;
      else if (geometry_pad_mode_str == "repeat_nearest")
        _geometry_pad_mode = 3;
    }

    float _prob = 0.0;
    bool _geometry_persp_horizontal = true;
    bool _geometry_persp_vertical = true;
    bool _geometry_transl_horizontal = false;
    bool _geometry_transl_vertical = false;
    bool _geometry_zoom_out = true;
    bool _geometry_zoom_in = true;
    float _geometry_zoom_factor = 0.25;
    float _geometry_persp_factor = 0.25;
    float _geometry_transl_factor = 0.5;
    uint8_t _geometry_pad_mode = 1;
    float _geometry_bbox_intersect = 0.75;
    cv::Mat _lambda;
  };

  class NoiseParams
  {
  public:
    NoiseParams(bool bw = false)
        : _hist_eq(!bw), _decolorize(!bw), _jpg(!bw), _convert_to_hsv(!bw),
          _convert_to_lab(!bw)
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

    float _prob = 0.0;
    bool _hist_eq = true;
    bool _inverse = true;
    bool _decolorize = true;
    bool _gauss_blur = true;
    bool _jpg = true;
    bool _posterize = true;
    bool _erosion = true;
    bool _saltpepper = true;
    float _saltpepper_fraction = 0.1;
    bool _clahe = true;
    bool _convert_to_hsv = true;
    bool _convert_to_lab = true;
    bool _rgb = false;
  };

  class DistortParams
  {
  public:
    DistortParams(bool bw = false)
        : _saturation(!bw), _hue(!bw), _channel_order(!bw)
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

    float _prob = 0.0;
    bool _brightness = true;
    float _brightness_delta = 32;
    bool _contrast = true;
    float _contrast_upper = 1.5;
    float _contrast_lower = 0.5;
    bool _saturation = true;
    float _saturation_upper = 1.5;
    float _saturation_lower = 0.5;
    bool _hue = true;
    float _hue_delta = 36;
    bool _channel_order = true;

    std::uniform_real_distribution<float> _uniform_real_brightness;
    std::uniform_real_distribution<float> _uniform_real_contrast;
    std::uniform_real_distribution<float> _uniform_real_saturation;
    std::uniform_real_distribution<float> _uniform_real_hue;
    bool _rgb = false;
  };

  class ImgRandAugCV
  {
  public:
    ImgRandAugCV()
    {
    }

    ImgRandAugCV(const bool &mirror, const bool &rotate,
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
      reset_rnd_test_gen();
    }

    ~ImgRandAugCV()
    {
    }

    void reset_rnd_test_gen()
    {
      _rnd_test_gen = std::default_random_engine(DATAAUG_TEST_SEED);
    }

    void seed_rnd_gen(unsigned int seed)
    {
      _rnd_gen = std::default_random_engine(seed);
    }

    void augment(cv::Mat &src);
    void augment_with_bbox(cv::Mat &src,
                           std::vector<std::vector<float>> &bboxes,
                           std::vector<int> &classes);
    void augment_with_segmap(cv::Mat &src, cv::Mat &tgt);

    void augment_test(cv::Mat &src);
    void augment_test_with_bbox(cv::Mat &src,
                                std::vector<std::vector<float>> &bboxes,
                                std::vector<int> &classes);
    void augment_test_with_segmap(cv::Mat &src, cv::Mat &tgt);

  protected:
    bool roll_weighted_dice(const float &prob);
    void applyDuplicateBBox(std::vector<std::vector<float>> &bboxes,
                            std::vector<int> &classes, const float &img_width,
                            const float &img_height);
    bool applyMirror(cv::Mat &src, const bool &sample = true);
    void applyMirrorBBox(std::vector<std::vector<float>> &bboxes,
                         const float &img_width);
    int applyRotate(cv::Mat &src, const bool &sample = true, int rot = 0);
    void applyRotateBBox(std::vector<std::vector<float>> &bboxes,
                         const float &img_width, const float &img_height,
                         const int &rot);
    bool applyCrop(cv::Mat &src, CropParams &cp, int &crop_x, int &crop_y,
                   const bool &sample = true, const bool &test = false);
    void applyCropBBox(std::vector<std::vector<float>> &bboxes,
                       std::vector<int> &classes, const CropParams &cp,
                       const float &img_width, const float &img_height,
                       const float &crop_x, const float &crop_y);
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

  public:
    bool _mirror = false;
    bool _rotate = false;

    CropParams _crop_params;
    CutoutParams _cutout_params;
    GeometryParams _geometry_params;
    NoiseParams _noise_params;
    DistortParams _distort_params;

    std::default_random_engine _rnd_gen;
    std::default_random_engine _rnd_test_gen;
    std::uniform_real_distribution<float> _uniform_real_1;
    std::bernoulli_distribution _bernouilli;
    std::uniform_int_distribution<int> _uniform_int_rotate;
  };

  void write_image_with_bboxes(const cv::Mat &src,
                               const std::vector<std::vector<float>> &bboxes,
                               const std::string fpath, int &ii);
}

#endif
