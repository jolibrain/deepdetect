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

#include "imgdataaug.h"

#include "apidata.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace dd
{
  ImgRandAugCVConfig parse_img_rand_aug_cv_config(const APIData &mllib,
                                                  int width, int height,
                                                  bool bw, bool rgb)
  {
    ImgRandAugCVConfig config;
    config.enabled = mllib.has("mirror") || mllib.has("rotate")
                     || mllib.has("crop_size") || mllib.has("cutout")
                     || mllib.has("geometry") || mllib.has("noise")
                     || mllib.has("distort");
    if (!config.enabled)
      return config;

    config.mirror = mllib.has("mirror") && mllib.get("mirror").get<bool>();
    config.rotate = mllib.has("rotate") && mllib.get("rotate").get<bool>();
    if (width != height && config.rotate)
      {
        config.rotate = false;
        config.rotate_disabled_for_shape = true;
      }

    if (mllib.has("crop_size"))
      {
        config.has_crop_size = true;
        config.crop_size = mllib.get("crop_size").get<int>();
        config.crop_params = CropParams(config.crop_size);
        if (mllib.has("test_crop_samples"))
          config.crop_params._test_crop_samples
              = mllib.get("test_crop_samples").get<int>();
      }

    if (mllib.has("cutout"))
      {
        config.has_cutout = true;
        config.cutout = mllib.get("cutout").get<double>();
        config.cutout_params = CutoutParams(config.cutout);
      }

    APIData geometry = mllib.getobj("geometry");
    if (!geometry.empty())
      {
        config.has_geometry = true;
        config.geometry_params._prob = geometry.get("prob").get<double>();
        if (geometry.has("persp_vertical"))
          config.geometry_params._geometry_persp_vertical
              = geometry.get("persp_vertical").get<bool>();
        if (geometry.has("persp_horizontal"))
          config.geometry_params._geometry_persp_horizontal
              = geometry.get("persp_horizontal").get<bool>();
        if (geometry.has("transl_vertical"))
          config.geometry_params._geometry_transl_vertical
              = geometry.get("transl_vertical").get<bool>();
        if (geometry.has("transl_horizontal"))
          config.geometry_params._geometry_transl_horizontal
              = geometry.get("transl_horizontal").get<bool>();
        if (geometry.has("zoom_out"))
          config.geometry_params._geometry_zoom_out
              = geometry.get("zoom_out").get<bool>();
        if (geometry.has("zoom_in"))
          config.geometry_params._geometry_zoom_in
              = geometry.get("zoom_in").get<bool>();
        if (geometry.has("persp_factor"))
          config.geometry_params._geometry_persp_factor
              = geometry.get("persp_factor").get<double>();
        if (geometry.has("transl_factor"))
          config.geometry_params._geometry_transl_factor
              = geometry.get("transl_factor").get<double>();
        if (geometry.has("zoom_factor"))
          config.geometry_params._geometry_zoom_factor
              = geometry.get("zoom_factor").get<double>();
        if (geometry.has("pad_mode"))
          config.geometry_params.set_pad_mode(
              geometry.get("pad_mode").get<std::string>());
      }

    config.noise_params = NoiseParams(bw);
    config.noise_params._rgb = rgb;
    APIData noise = mllib.getobj("noise");
    if (!noise.empty())
      {
        config.has_noise = true;
        config.noise_params._prob = noise.get("prob").get<double>();
      }

    config.distort_params = DistortParams(bw);
    config.distort_params._rgb = rgb;
    APIData distort = mllib.getobj("distort");
    if (!distort.empty())
      {
        config.has_distort = true;
        config.distort_params._prob = distort.get("prob").get<double>();
      }

    return config;
  }

}
