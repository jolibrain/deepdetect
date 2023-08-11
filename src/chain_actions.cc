/**
 * DeepDetect
 * Copyright (c) 2019 Emmanuel Benazera
 * Author: Emmanuel Benazera <emmanuel.benazera@jolibrain.com>
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

#include "chain_actions.h"
#include <opencv2/opencv.hpp>
#ifdef USE_CUDA_CV
#include <opencv2/cudaimgproc.hpp>
#endif
#if CV_VERSION_MAJOR >= 3
#define CV_BGR2RGB cv::COLOR_BGR2RGB
#define CV_RGB2BGR cv::COLOR_RGB2BGR
#endif
#include <unordered_set>
#include "utils/utils.hpp"
#include "dto/predict_out.hpp"

#ifdef USE_DLIB
#include "backends/dlib/dlib_actions.h"
#endif

namespace dd
{
  void ImgsCropAction::apply(APIData &model_out, ChainData &cdata)
  {
    std::vector<APIData> vad = model_out.getv("predictions");
    APIData input_ad = model_out.getobj("input");
    std::vector<std::pair<int, int>> imgs_size
        = input_ad.get("imgs_size").get<std::vector<std::pair<int, int>>>();
    std::vector<std::string> bbox_ids;

    std::vector<cv::Mat> imgs;
    std::vector<cv::Mat> cropped_imgs;
#ifdef USE_CUDA_CV
    std::vector<cv::cuda::GpuMat> cuda_imgs;
    std::vector<cv::cuda::GpuMat> cropped_cuda_imgs;

    if (input_ad.has("cuda_imgs"))
      {
        cuda_imgs
            = input_ad.get("cuda_imgs").get<std::vector<cv::cuda::GpuMat>>();
      }
    else
#endif
      {
        imgs = input_ad.get("imgs").get<std::vector<cv::Mat>>();
      }

    // check for action parameters
    double bratio = _params->padding_ratio;
    bool save_crops = _params->save_crops;

    std::string save_path = _params->save_path;
    if (!save_path.empty())
      save_path += "/";

    int fixed_width = _params->fixed_width;
    int fixed_height = _params->fixed_height;

    std::vector<APIData> cvad;

    // iterate image batch
    for (size_t i = 0; i < vad.size(); i++)
      {
        std::string uri = vad.at(i).get("uri").get<std::string>();

        int im_cols, im_rows;
#ifdef USE_CUDA_CV
        if (!cuda_imgs.empty())
          {
            im_cols = cuda_imgs.at(i).cols;
            im_rows = cuda_imgs.at(i).rows;
          }
        else
#endif
          {
            im_cols = imgs.at(i).cols;
            im_rows = imgs.at(i).rows;
          }
        int orig_cols = imgs_size.at(i).second;
        int orig_rows = imgs_size.at(i).first;

        std::vector<APIData> ad_cls = vad.at(i).getv("classes");
        std::vector<APIData> cad_cls;

        // iterate bboxes per image
        for (size_t j = 0; j < ad_cls.size(); j++)
          {
            APIData bbox = ad_cls.at(j).getobj("bbox");
            if (bbox.empty())
              throw ActionBadParamException(
                  "crop action cannot find bbox object for uri " + uri);

            double xmin = bbox.get("xmin").get<double>() / orig_cols * im_cols;
            double ymin = bbox.get("ymin").get<double>() / orig_rows * im_rows;
            double xmax = bbox.get("xmax").get<double>() / orig_cols * im_cols;
            double ymax = bbox.get("ymax").get<double>() / orig_rows * im_rows;

            double deltax = bratio * (xmax - xmin);
            double deltay = bratio * (ymax - ymin);

            double cxmin = std::max(0.0, xmin - deltax);
            double cxmax
                = std::min(static_cast<double>(im_cols), xmax + deltax);
            double cymin = std::max(0.0, ymin - deltay);
            double cymax
                = std::min(static_cast<double>(im_rows), ymax + deltay);

            if (_params->min_width > (cxmax - cxmin))
              fixed_width = _params->min_width;

            if (_params->min_height > (cymax - cymin))
              fixed_height = _params->min_height;

            if (_params->force_square)
              {
                if (fixed_width == 0)
                  fixed_width = static_cast<int>(cxmax - cxmin);
                if (fixed_height == 0)
                  fixed_height = static_cast<int>(cymax - cymin);

                if (fixed_height > fixed_width)
                  fixed_width = fixed_height;
                if (fixed_width > fixed_height)
                  fixed_height = fixed_width;
              }

            if (fixed_width > 0)
              {
                double xcenter = cxmin + (cxmax - cxmin) / 2.0;
                cxmin = int(xcenter - fixed_width / 2.0);
                cxmax = int(xcenter + fixed_width / 2.0);

                if (cxmin < 0)
                  {
                    cxmax += -cxmin;
                    cxmin = 0;
                  }
                if (cxmax > im_cols)
                  {
                    cxmin -= cxmax - im_cols;
                    cxmax = im_cols;
                  }
              }
            if (fixed_height > 0)
              {
                double ycenter = cymin + (cymax - cymin) / 2.0;
                cymin = int(ycenter - fixed_height / 2.0);
                cymax = int(ycenter + fixed_height / 2.0);

                if (cymin < 0)
                  {
                    cymax += -cymin;
                    cymin = 0;
                  }
                if (cymax > im_rows)
                  {
                    cymin -= cymax - im_rows;
                    cymax = im_rows;
                  }
              }

            if (cxmin >= im_cols || cymin >= im_rows || cxmax < 0 || cymax < 0)
              {
                _chain_logger->warn("bounding box does not intersect image, "
                                    "skipping crop action");
                continue;
              }

            cv::Rect roi(cxmin, cymin, cxmax - cxmin, cymax - cymin);
#ifdef USE_CUDA_CV
            if (!cuda_imgs.empty())
              {
                cv::cuda::GpuMat cropped_img = cuda_imgs.at(i)(roi).clone();

                // save crops if requested
                if (save_crops)
                  {
                    cv::Mat cropped_img_cpu;
                    cropped_img.download(cropped_img_cpu);
                    std::string puri = dd_utils::split(uri, '/').back();
                    cv::imwrite(save_path + "crop_" + puri + "_"
                                    + std::to_string(j) + ".png",
                                cropped_img_cpu);
                  }

                cropped_cuda_imgs.push_back(std::move(cropped_img));
              }
            else
#endif
              {
                cv::Mat cropped_img = imgs.at(i)(roi).clone();

                // save crops if requested
                if (save_crops)
                  {
                    std::string puri = dd_utils::split(uri, '/').back();
                    cv::imwrite(save_path + "crop_" + puri + "_"
                                    + std::to_string(j) + ".png",
                                cropped_img);
                  }
                cropped_imgs.push_back(std::move(cropped_img));
              }

            // adding bbox id
            std::string bbox_id = genid(uri, "bbox" + std::to_string(j));
            bbox_ids.push_back(bbox_id);
            ad_cls.at(j).add("class_id", bbox_id);
            cad_cls.push_back(ad_cls.at(j));
          }
        APIData ccls;
        ccls.add("uri", uri);
        if (vad.at(i).has("index_uri"))
          ccls.add("index_uri", vad.at(i).get("index_uri").get<std::string>());
        ccls.add("classes", cad_cls);
        cvad.push_back(ccls);
      }
    // store crops into action output store
    APIData action_out;
#ifdef USE_CUDA_CV
    if (!cropped_cuda_imgs.empty())
      {
        action_out.add("data_cuda_img", cropped_cuda_imgs);
      }
    else
#endif
      {
        action_out.add("data_raw_img", cropped_imgs);
      }
    action_out.add("cids", bbox_ids);
    cdata.add_action_data(_action_id, action_out);

    // updated model data with chain ids
    model_out.add("predictions", cvad);
  }

  void ImgsRotateAction::apply(APIData &model_out, ChainData &cdata)
  {
    // get label
    std::vector<APIData> vad = model_out.getv("predictions");
    std::vector<cv::Mat> imgs
        = model_out.getobj("input").get("imgs").get<std::vector<cv::Mat>>();
    std::vector<std::pair<int, int>> imgs_size
        = model_out.getobj("input")
              .get("imgs_size")
              .get<std::vector<std::pair<int, int>>>();
    std::vector<cv::Mat> rimgs;
    std::vector<std::string> uris;

    // check for action parameters
    std::string orientation = _params->orientation;
    bool save_img = _params->save_img;

    std::string save_path = _params->save_path;
    if (!save_path.empty())
      save_path += "/";

    for (size_t i = 0; i < vad.size(); i++) // iterate predictions
      {
        std::string uri = vad.at(i).get("uri").get<std::string>();
        cv::Mat img = imgs.at(i);
        std::vector<APIData> ad_cls = vad.at(i).getv("classes");
        std::vector<APIData> cad_cls;

        // rotate and make image available to next service
        if (ad_cls.size() > 0)
          {
            uris.push_back(uri);
            std::string cat1 = ad_cls.at(0).get("cat").get<std::string>();
            cv::Mat rimg, timg;
            if (cat1 == "0") // all tests in absolute orientation
              {
                rimg = img;
              }
            else if (cat1 == "90")
              {
                cv::transpose(img, timg);
                int orient = 1;
                if (orientation == "relative")
                  orient = 0; // 270
                cv::flip(timg, rimg, orient);
              }
            else if (cat1 == "180")
              {
                cv::flip(img, rimg, -1);
              }
            else if (cat1 == "270")
              {
                cv::transpose(img, timg);
                int orient = 0;
                if (orientation == "relative")
                  orient = 1; // 90
                cv::flip(timg, rimg, orient);
              }
            if (!rimg.empty())
              {
                rimgs.push_back(rimg);

                // save image if requested
                if (save_img)
                  {
                    std::string puri = dd_utils::split(uri, '/').back();
                    cv::imwrite(
                        save_path + "rot_" + puri + "_" + cat1 + ".png", rimg);
                  }
              }
          }
      }
    // store rotated images into action output store
    APIData action_out;
    action_out.add("data_raw_img", rimgs);
    action_out.add("cids", uris);
    cdata.add_action_data(_action_id, action_out);
  }

  void make_even(cv::Mat &mat, int &width, int &height)
  {
    if (width % 2 != 0)
      width -= 1;
    if (height % 2 != 0)
      height -= 1;

    if (width != mat.cols || height != mat.rows)
      {
        cv::Rect roi{ 0, 0, width, height };
        mat = mat(roi);
      }
  }

  void ImgsCropRecomposeAction::apply(APIData &model_out, ChainData &cdata)
  {
    APIData first_model = cdata.get_model_data("0");
    APIData input_ad = first_model.getobj("input");
    // image
    std::vector<cv::Mat> imgs;
#ifdef USE_CUDA_CV
    std::vector<cv::cuda::GpuMat> cuda_imgs;
    std::vector<cv::cuda::GpuMat> cropped_cuda_imgs;

    if (input_ad.has("cuda_imgs"))
      {
        cuda_imgs
            = input_ad.get("cuda_imgs").get<std::vector<cv::cuda::GpuMat>>();
      }
    else
#endif
      {
        imgs = input_ad.get("imgs").get<std::vector<cv::Mat>>();
      }
    std::vector<std::pair<int, int>> imgs_size
        = input_ad.get("imgs_size").get<std::vector<std::pair<int, int>>>();

    // bbox
    std::vector<APIData> vad = first_model.getv("predictions");

    // generated images
    // XXX: Images always are written on RAM first.
    // This may change in the future
    std::map<std::string, cv::Mat> gen_imgs;
    if (model_out.has("dto"))
      {
        auto dto = model_out.get("dto")
                       .get<oatpp::Any>()
                       .retrieve<oatpp::Object<DTO::PredictBody>>();
        for (size_t i = 0; i < dto->predictions->size(); ++i)
          {
            auto images = dto->predictions->at(i)->images;
            if (images->size() == 0)
              throw ActionBadParamException(
                  "Recompose requires output.image = true in previous model");
            gen_imgs.insert(
                { *dto->predictions->at(i)->uri, images->at(0)->get_img() });
          }
      }
    else
      {
        throw ActionBadParamException("Recompose action requires GAN output");
      }

    std::vector<cv::Mat> rimgs;
    std::vector<std::string> uris;

    bool save_img = _params->save_img;
    std::string save_path = _params->save_path;
    if (!save_path.empty())
      save_path += "/";

    auto pred_body = DTO::PredictBody::createShared();

    // need: original image, bbox coordinates, new image
    for (size_t i = 0; i < vad.size(); i++)
      {
        std::string uri = vad.at(i).get("uri").get<std::string>();
        uris.push_back(uri);

        cv::Mat input_img;
        int input_width, input_height;
        cv::Mat rimg;
#ifdef USE_CUDA_CV
        cv::cuda::GpuMat cuda_input_img;
        cv::cuda::GpuMat cuda_rimg;
        if (!cuda_imgs.empty())
          {
            cuda_input_img = cuda_imgs.at(i);
            input_width = cuda_input_img.cols;
            input_height = cuda_input_img.rows;
            cuda_rimg = cuda_input_img.clone();
          }
        else
#endif
          {
            input_img = imgs.at(i);
            input_width = input_img.cols;
            input_height = input_img.rows;
            rimg = input_img.clone();
          }
        int orig_width = imgs_size.at(i).second;
        int orig_height = imgs_size.at(i).first;

        std::vector<APIData> ad_cls = vad.at(i).getv("classes");
        APIData bbox;

        for (size_t j = 0; j < ad_cls.size(); j++)
          {
            bbox = ad_cls.at(j).getobj("bbox");
            std::string cls_id
                = ad_cls.at(j).get("class_id").get<std::string>();

            cv::Mat gen_img = gen_imgs.at(cls_id);
            int gen_width = gen_img.cols;
            int gen_height = gen_img.rows;

            // support odd width & height
            make_even(gen_img, gen_width, gen_height);

            if (gen_width > input_width || gen_height > input_height)
              {
                throw ActionBadParamException(
                    "Recomposing image is impossible, crop is too big: "
                    + std::to_string(gen_width) + ","
                    + std::to_string(gen_height) + "/"
                    + std::to_string(orig_width) + ","
                    + std::to_string(orig_height));
              }

            double xmin
                = bbox.get("xmin").get<double>() / orig_width * input_width;
            double ymin
                = bbox.get("ymin").get<double>() / orig_height * input_height;
            double xmax
                = bbox.get("xmax").get<double>() / orig_width * input_width;
            double ymax
                = bbox.get("ymax").get<double>() / orig_height * input_height;

            int cx = static_cast<int>((xmin + xmax) / 2);
            int cy = static_cast<int>((ymin + ymax) / 2);
            cx = std::min(std::max(cx, gen_width / 2),
                          input_width - gen_width / 2);
            cy = std::min(std::max(cy, gen_height / 2),
                          input_height - gen_height / 2);

            cv::Rect roi{ cx - gen_width / 2, cy - gen_height / 2, gen_width,
                          gen_height };
#ifdef USE_CUDA_CV
            if (cuda_rimg.cols != 0)
              {
                cv::cuda::GpuMat cuda_gen_img;
                cuda_gen_img.upload(gen_img);
                cuda_gen_img.copyTo(cuda_rimg(roi));
              }
            else
#endif
              {
                gen_img.copyTo(rimg(roi));
              }
          }

        rimgs.push_back(rimg);

        auto action_pred = DTO::Prediction::createShared();
        action_pred->uri = uri.c_str();
        action_pred->images = oatpp::Vector<DTO::DTOImage>::createShared();
#ifdef USE_CUDA_CV
        if (cuda_rimg.cols != 0)
          {
            action_pred->images->push_back({ cuda_rimg });
          }
        else
#endif
          {
            action_pred->images->push_back({ rimg });
          }
        pred_body->predictions->push_back(action_pred);

        // save image if requested
        if (save_img)
          {
            std::string puri = dd_utils::split(uri, '/').back();
#ifdef USE_CUDA_CV
            if (cuda_rimg.cols != 0)
              cuda_rimg.download(rimg);
#endif
            cv::imwrite(save_path + "recompose_" + puri + ".png", rimg);
          }
      }

    // Output: new image -> only works in "image" output mode
    APIData action_out;
    action_out.add("data_raw_img", rimgs);
    action_out.add("cids", uris);
    action_out.add("output", pred_body);
    cdata.add_action_data(_action_id, action_out);
  }

  cv::Scalar bbox_palette[]
      = { { 82, 188, 227 }, { 196, 110, 49 }, { 39, 54, 227 },
          { 68, 227, 81 },  { 77, 157, 255 }, { 255, 112, 207 },
          { 240, 228, 65 }, { 94, 242, 151 }, { 236, 121, 242 },
          { 28, 77, 120 } };
  size_t bbox_palette_size = 10;

  void ImgsDrawBBoxAction::apply(APIData &model_out, ChainData &cdata)
  {
    std::vector<APIData> vad = model_out.getv("predictions");
    APIData input_ad = model_out.getobj("input");

    std::vector<cv::Mat> imgs
        = model_out.getobj("input").get("imgs").get<std::vector<cv::Mat>>();
    std::vector<std::pair<int, int>> imgs_size
        = model_out.getobj("input")
              .get("imgs_size")
              .get<std::vector<std::pair<int, int>>>();
    std::vector<cv::Mat> rimgs;
    std::vector<std::string> uris;
    auto pred_body = DTO::PredictBody::createShared();

    bool save_img = _params->save_img;
    int ref_thickness = _params->thickness;

    std::string save_path = _params->save_path;
    if (!save_path.empty())
      save_path += "/";

    for (size_t i = 0; i < vad.size(); i++)
      {
        std::string uri = vad.at(i).get("uri").get<std::string>();
        uris.push_back(uri);

        int im_cols = imgs.at(i).cols;
        int im_rows = imgs.at(i).rows;
        int orig_cols = imgs_size.at(i).second;
        int orig_rows = imgs_size.at(i).first;

        std::vector<APIData> ad_cls = vad.at(i).getv("classes");
        cv::Mat rimg = imgs.at(i).clone();

        // iterate bboxes per image
        for (size_t j = 0; j < ad_cls.size(); j++)
          {
            APIData bbox = ad_cls.at(j).getobj("bbox");
            std::string cat = ad_cls.at(j).get("cat").get<std::string>();
            if (bbox.empty())
              throw ActionBadParamException(
                  "draw action cannot find bbox object for uri " + uri);

            double xmin = bbox.get("xmin").get<double>() / orig_cols * im_cols;
            double ymin = bbox.get("ymin").get<double>() / orig_rows * im_rows;
            double xmax = bbox.get("xmax").get<double>() / orig_cols * im_cols;
            double ymax = bbox.get("ymax").get<double>() / orig_rows * im_rows;

            // draw bbox
            cv::Point pt1{ int(xmin), int(ymin) };
            cv::Point pt2{ int(xmax), int(ymax) };
            size_t cls_hash = std::hash<std::string>{}(cat);
            cv::Scalar color = bbox_palette[cls_hash % bbox_palette_size];
            cv::rectangle(rimg, pt1, pt2, cv::Scalar(255, 255, 255),
                          ref_thickness + 2);
            cv::rectangle(rimg, pt1, pt2, color, ref_thickness);

            // draw class & confidences
            std::string label;
            if (_params->write_cat)
              label = cat;
            if (_params->write_cat && _params->write_prob)
              label += " - ";
            if (_params->write_prob)
              label += std::to_string(ad_cls.at(j).get("prob").get<double>());

            // font size relatively to base opencv font size
            float font_size = 2;
            int x_txt = static_cast<int>(xmin + 5);
            if (x_txt > im_cols - 15)
              x_txt = im_cols - 15;
            int y_txt = std::min(im_rows - 20,
                                 static_cast<int>(ymax + 2 + font_size * 12));

            cv::putText(rimg, label, cv::Point(x_txt, y_txt),
                        cv::FONT_HERSHEY_PLAIN, font_size,
                        cv::Scalar(255, 255, 255), ref_thickness + 2);
            cv::putText(rimg, label, cv::Point(x_txt, y_txt),
                        cv::FONT_HERSHEY_PLAIN, font_size, color,
                        ref_thickness);
          }

        rimgs.push_back(rimg);

        // save image if requested
        if (save_img)
          {
            std::string puri = dd_utils::split(uri, '/').back();
            std::string rimg_path = save_path + "bbox_" + puri + ".png";
            this->_chain_logger->info("draw_bbox: Saved image to path {}",
                                      rimg_path);
            cv::imwrite(rimg_path, rimg);
          }

        if (_params->output_images)
          {
            auto action_pred = DTO::Prediction::createShared();

            action_pred->vals = DTO::DTOVector<uint8_t>(std::vector<uint8_t>(
                rimg.data, rimg.data + (rimg.total() * rimg.elemSize())));
            action_pred->uri = uri.c_str();
            pred_body->predictions->push_back(action_pred);
          }
      }

    APIData action_out;
    action_out.add("data_raw_img", rimgs);
    action_out.add("cids", uris);
    if (_params->output_images)
      {
        action_out.add("output", pred_body);
      }
    cdata.add_action_data(_action_id, action_out);
  }

  void ClassFilter::apply(APIData &model_out, ChainData &cdata)
  {
    if (_params->classes == nullptr)
      {
        throw ActionBadParamException(
            "filter action is missing classes parameter");
      }
    std::unordered_set<std::string> on_classes_us;
    for (auto s : *_params->classes)
      on_classes_us.insert(s);
    std::unordered_set<std::string>::const_iterator usit;

    std::vector<APIData> vad = model_out.getv("predictions");
    std::vector<APIData> cvad;

    for (size_t i = 0; i < vad.size(); i++)
      {
        std::vector<APIData> ad_cls = vad.at(i).getv("classes");
        std::vector<APIData> cad_cls;

        for (size_t j = 0; j < ad_cls.size(); j++)
          {
            std::string cat = ad_cls.at(j).get("cat").get<std::string>();
            if ((usit = on_classes_us.find(cat)) != on_classes_us.end())
              {
                cad_cls.push_back(ad_cls.at(j));
              }
          }
        APIData ccls;
        ccls.add("classes", cad_cls);
        if (vad.at(i).has("index_uri"))
          ccls.add("index_uri", vad.at(i).get("index_uri").get<std::string>());
        ccls.add("uri", vad.at(i).get("uri").get<std::string>());
        cvad.push_back(ccls);
      }

    // empty action data
    cdata.add_action_data(_action_id, APIData());
    // actions_data.push_back(APIData());

    // updated model data
    model_out.add("predictions", cvad);
  }

  void *ChainActionFactory::add_chain_action(const std::string &action_name,
                                             const action_function &func)
  {
    auto &registry = ChainActionFactory::get_action_registry();
    registry[action_name] = func;
    return nullptr;
  }

  std::unordered_map<std::string, action_function> &
  ChainActionFactory::get_action_registry()
  {
    static std::unordered_map<std::string, action_function> registry;
    return registry;
  }

  void ChainActionFactory::apply_action(
      const std::string &action_type, APIData &model_out, ChainData &cdata,
      const std::shared_ptr<spdlog::logger> &chain_logger)
  {
    if (_call_dto->id == nullptr)
      {
        std::string action_id = std::to_string(cdata._action_data.size());
        _call_dto->id = action_id.c_str();
      }

    auto &registry = get_action_registry();
    auto hit = registry.find(action_type);

    if (hit != registry.end())
      {
        hit->second(_call_dto, model_out, cdata, chain_logger);
      }
    else
      {
        throw ActionBadParamException("unknown action " + action_type);
      }
  }

  CHAIN_ACTION("crop", ImgsCropAction)
  CHAIN_ACTION("rotate", ImgsRotateAction)
  CHAIN_ACTION("recompose", ImgsCropRecomposeAction)
  CHAIN_ACTION("draw_bbox", ImgsDrawBBoxAction)
  CHAIN_ACTION("filter", ClassFilter)
#ifdef USE_DLIB
  CHAIN_ACTION("dlib_align_crop", DlibAlignCropAction)
#endif

} // end of namespace
