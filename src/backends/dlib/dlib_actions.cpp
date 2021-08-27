/**
 * DeepDetect
 * Copyright (c) 2019 Pixel Forensics, Inc.
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

#include "dlib_actions.h"
#include "chain_actions.h"

#include "opencv2/opencv.hpp"
#include "dlib/data_io.h"
#include "dlib/image_io.h"
#include "dlib/image_transforms.h"
#include "dlib/image_processing.h"
#include "dlib/opencv/to_open_cv.h"
#include "dlib/opencv/cv_image.h"

#include "utils/utils.hpp"

namespace dd
{
  void DlibAlignCropAction::apply(APIData &model_out, ChainData &cdata)
  {
    std::vector<APIData> vad = model_out.getv("predictions");
    std::vector<cv::Mat> imgs
        = model_out.getobj("input").get("imgs").get<std::vector<cv::Mat>>();
    std::vector<std::pair<int, int>> imgs_size
        = model_out.getobj("input")
              .get("imgs_size")
              .get<std::vector<std::pair<int, int>>>();
    std::vector<cv::Mat> cropped_imgs;
    std::vector<std::string> bbox_ids;

    // check for action parameters
    double bratio
        = _params->padding_ratio != nullptr ? _params->padding_ratio : 0.25;
    int chip_size = _params->chip_size;
    std::vector<APIData> cvad;

    bool save_crops = _params->save_crops;

    // iterate image batch
    for (size_t i = 0; i < vad.size(); i++)
      {
        std::string uri = vad.at(i).get("uri").get<std::string>();

        cv::Mat cvimg = imgs.at(i);
        dlib::matrix<dlib::rgb_pixel> img;
        dlib::assign_image(img, dlib::cv_image<dlib::rgb_pixel>(cvimg));

        std::vector<APIData> ad_cls = vad.at(i).getv("classes");
        std::vector<APIData> cad_cls;

        // iterate bboxes per image
        for (size_t j = 0; j < ad_cls.size(); j++)
          {
            APIData bbox = ad_cls.at(j).getobj("bbox");
            if (bbox.empty())
              {
                _chain_logger->warn(
                    "align/crop action cannot find bbox object for uri "
                    + uri);
                throw ActionBadParamException(
                    "align/crop action cannot find bbox object for uri "
                    + uri);
              }
            APIData ad_shape = bbox.getobj("shape");
            if (ad_shape.empty())
              {
                _chain_logger->warn(
                    "align/crop action cannot find shape object for uri "
                    + uri);
                throw ActionBadParamException(
                    "align/crop action cannot find shape object for uri "
                    + uri);
              }

            // adding bbox id
            std::string bbox_id = genid(uri, "bbox" + std::to_string(j));
            bbox_ids.push_back(bbox_id);
            APIData ad_cid;
            ad_cls.at(j).add(bbox_id, ad_cid);
            cad_cls.push_back(ad_cls.at(j));
            const dlib::rectangle shape_rect(
                ad_shape.get("left").get<long>(),
                ad_shape.get("top").get<long>(),
                ad_shape.get("right").get<long>(),
                ad_shape.get("bottom").get<long>());
            std::vector<dlib::point> parts;
            std::vector<double> points
                = ad_shape.get("points").get<std::vector<double>>();
            for (size_t idx = 0; idx < points.size() - 1; idx += 2)
              {
                parts.push_back(
                    dlib::point(static_cast<long>(points[idx]),
                                static_cast<long>(points[idx + 1])));
              }

            dlib::full_object_detection shape(shape_rect, parts);
            dlib::matrix<dlib::rgb_pixel> r;
            dlib::extract_image_chip(
                img, dlib::get_face_chip_details(shape, chip_size, bratio), r);
            cv::Mat cropped_img = dlib::toMat(r);

            // save crops if requested
            if (save_crops)
              {
                std::string puri = dd_utils::split(uri, '/').back();
                cv::imwrite("crop_" + puri + "_" + std::to_string(j) + ".png",
                            cropped_img);
              }
            cropped_imgs.push_back(std::move(cropped_img));
          }
        APIData ccls;
        ccls.add("uri", uri);
        if (vad.at(i).has("index_uri"))
          ccls.add("index_uri", vad.at(i).get("index_uri").get<std::string>());
        ccls.add("classes", cad_cls);
        cvad.push_back(ccls);
      }
    // store serialized crops into action output store
    APIData action_out;
    action_out.add("data_raw_img", cropped_imgs);
    action_out.add("cids", bbox_ids);
    cdata.add_action_data(_action_id, action_out);

    // updated model data with chain ids
    model_out.add("predictions", cvad);
  }
}
