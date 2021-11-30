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

  void write_image_with_bboxes(const cv::Mat &src,
                               const std::vector<std::vector<float>> &bboxes,
                               const std::string fpath, int &ii)
  {
    cv::Mat src_bb = src.clone();
    for (size_t bb = 0; bb < bboxes.size(); ++bb)
      {
        cv::Rect r(bboxes[bb][0], bboxes[bb][1], bboxes[bb][2] - bboxes[bb][0],
                   bboxes[bb][3] - bboxes[bb][1]);
        cv::rectangle(src_bb, r, cv::Scalar(255, 0, 0), 1, 8, 0);
      }
    cv::imwrite(fpath + "/test_aug_" + std::to_string(ii) + ".png", src_bb);
    ++ii;
  }

  void TorchImgRandAugCV::augment(cv::Mat &src)
  {
    // apply augmentation
    applyGeometry(src, _geometry_params);
    applyCutout(src, _cutout_params);

    // these transforms do affect dimensions
    applyCrop(src, _crop_params);
    applyMirror(src);
    applyRotate(src);
    applyNoise(src);
  }

  void
  TorchImgRandAugCV::augment_with_bbox(cv::Mat &src,
                                       std::vector<torch::Tensor> &targets)
  {
    torch::Tensor t = targets[0];
    int nbbox = t.size(0);
    std::vector<std::vector<float>> bboxes;
    for (int bb = 0; bb < nbbox; ++bb)
      {
        std::vector<float> bbox;
        for (int d = 0; d < 4; ++d)
          {
            bbox.push_back(t[bb][d].item<float>());
          }
        bboxes.push_back(bbox); // add (xmin, ymin, xmax, ymax)
      }

    bool mirror = applyMirror(src);
    if (mirror)
      {
        applyMirrorBBox(bboxes, static_cast<float>(src.cols));
      }
    int rot = applyRotate(src);
    if (rot > 0)
      {
        applyRotateBBox(bboxes, static_cast<float>(src.cols),
                        static_cast<float>(src.rows), rot);
      }
    // XXX: no cutout with bboxes (yet)
    GeometryParams geoparams = _geometry_params;
    cv::Mat src_c = src.clone();
    applyGeometry(src_c, geoparams, true);
    if (!geoparams._lambda.empty())
      {
        // geometry on bboxes
        std::vector<std::vector<float>> bboxes_c = bboxes;
        applyGeometryBBox(bboxes_c, geoparams, src_c.cols,
                          src_c.rows); // uses the stored lambda
        if (!bboxes_c.empty())         // some bboxes remain
          {
            src = src_c;
            bboxes = bboxes_c;
          }
      }

    // replacing the initial bboxes with the transformed ones.
    nbbox = bboxes.size();
    for (int bb = 0; bb < nbbox; ++bb)
      {
        for (int d = 0; d < 4; ++d)
          {
            t[bb][d] = bboxes.at(bb).at(d);
          }
      }
    applyNoise(src);
  }

  void TorchImgRandAugCV::augment_with_segmap(cv::Mat &src, cv::Mat &tgt)
  {
    GeometryParams geoparams = _geometry_params;
    applyGeometry(src, geoparams, true, true);
    if (!geoparams._lambda.empty())
      applyGeometry(tgt, geoparams, false, false); // reuses geoparams

    applyCutout(src, _cutout_params);
    bool mirrored = applyMirror(src);
    if (mirrored)
      applyMirror(tgt, false);
    int rot = applyRotate(src);
    if (rot != 0)
      applyRotate(tgt, false, rot);
    applyNoise(src);
  }

  bool TorchImgRandAugCV::roll_weighted_dice(const float &prob)
  {
    // Draw random between 0 and 1
    float r1 = 0.0;
#pragma omp critical
    {
      r1 = _uniform_real_1(_rnd_gen);
    }
    if (r1 > prob)
      return false;
    else
      return true;
  }

  /*- transforms -*/
  bool TorchImgRandAugCV::applyMirror(cv::Mat &src, const bool &sample)
  {
    if (!_mirror)
      return false;

    bool mirror = false;
#pragma omp critical
    {
      if (sample)
        mirror = _bernouilli(_rnd_gen);
      else
        mirror = true;
    }
    if (mirror)
      {
        cv::Mat dst;
        cv::flip(src, dst, 1);
        src = dst;
      }
    return mirror;
  }

  void
  TorchImgRandAugCV::applyMirrorBBox(std::vector<std::vector<float>> &bboxes,
                                     const float &img_width)
  {
    for (size_t i = 0; i < bboxes.size(); ++i)
      {
        float xmin = bboxes.at(i)[0];
        bboxes.at(i)[0] = img_width - bboxes.at(i)[2]; // xmin = width - xmax
        bboxes.at(i)[2] = img_width - xmin;
      }
  }

  int TorchImgRandAugCV::applyRotate(cv::Mat &src, const bool &sample, int rot)
  {
    if (!_rotate)
      return -1;

#pragma omp critical
    {
      if (sample)
        rot = _uniform_int_rotate(_rnd_gen);
    }
    if (rot == 0)
      return rot;
    else if (rot == 1) // 270
      {
        cv::Mat dst;
        cv::transpose(src, dst);
        cv::flip(dst, src, 0);
      }
    else if (rot == 2) // 180
      {
        cv::Mat dst;
        cv::flip(src, dst, -1);
        src = dst;
      }
    else if (rot == 3) // 90
      {
        cv::Mat dst;
        cv::transpose(src, dst);
        cv::flip(dst, src, 1);
      }
    return rot;
  }

  void
  TorchImgRandAugCV::applyRotateBBox(std::vector<std::vector<float>> &bboxes,
                                     const float &img_width,
                                     const float &img_height, const int &rot)
  {
    std::vector<std::vector<float>> nbboxes;
    for (size_t i = 0; i < bboxes.size(); ++i)
      {
        std::vector<float> bbox = bboxes.at(i);
        std::vector<float> nbox;
        if (rot == 1) // 90
          {
            nbox.push_back(bbox[1]);              // xmin <- ymin
            nbox.push_back(img_height - bbox[2]); // ymin <- height-xmax
            nbox.push_back(bbox[3]);              // xmax <- ymax
            nbox.push_back(img_height - bbox[0]); // ymax <- height-xmin
          }
        else if (rot == 2) // 180
          {
            nbox.push_back(img_width - bbox[2]);  // xmin <- width-xmax
            nbox.push_back(img_height - bbox[3]); // ymin <- height-ymax
            nbox.push_back(img_width - bbox[0]);  // xmax <- width-xmin
            nbox.push_back(img_height - bbox[1]); // ymax <- height-ymin
          }
        else if (rot == 3) // 270
          {
            nbox.push_back(img_width - bbox[3]); // xmin <- width-ymax
            nbox.push_back(bbox[0]);             // ymin <- xmin
            nbox.push_back(img_width - bbox[1]); // xmax <- width-ymin
            nbox.push_back(bbox[2]);             // ymax <- xmax
          }
        nbboxes.push_back(nbox);
      }
    bboxes = nbboxes;
  }

  void TorchImgRandAugCV::applyCrop(cv::Mat &src, CropParams &cp,
                                    const bool &store_rparams)
  {
    if (cp._crop_size <= 0)
      return;

    int crop_x = 0;
    int crop_y = 0;
#pragma omp critical
    {
      crop_x = cp._uniform_int_crop_x(_rnd_gen);
      crop_y = cp._uniform_int_crop_y(_rnd_gen);
    }
    cv::Rect crop(crop_x, crop_y, cp._crop_size, cp._crop_size);
    cv::Mat dst = src(crop).clone();
    src = dst;

    if (store_rparams)
      {
        cp._crop_x = crop_x;
        cp._crop_y = crop_y;
      }
  }

  void TorchImgRandAugCV::applyCutout(cv::Mat &src, CutoutParams &cp,
                                      const bool &store_rparams)
  {
    if (cp._prob == 0.0)
      return;

    if (!roll_weighted_dice(cp._prob))
      return;

#pragma omp critical
    {
      // get shape and area to erase
      int w = 0, h = 0, rect_x = 0, rect_y = 0;
      if (cp._w == 0 && cp._h == 0)
        {
          float s = cp._uniform_real_cutout_s(_rnd_gen) * cp._img_width
                    * cp._img_height;                    // area
          float r = cp._uniform_real_cutout_r(_rnd_gen); // aspect ratio

          w = std::min(cp._img_width,
                       static_cast<int>(std::floor(std::sqrt(s / r))));
          h = std::min(cp._img_height,
                       static_cast<int>(std::floor(std::sqrt(s * r))));
          std::uniform_int_distribution<int> distx(0, cp._img_width - w);
          std::uniform_int_distribution<int> disty(0, cp._img_height - h);
          rect_x = distx(_rnd_gen);
          rect_y = disty(_rnd_gen);
        }

      // erase
      cv::Rect rect(rect_x, rect_y, w, h);
      cv::Mat selected_area = src(rect);
      if (selected_area.channels() == 3)
        cv::randu(selected_area,
                  cv::Scalar(cp._cutout_vl, cp._cutout_vl, cp._cutout_vl),
                  cv::Scalar(cp._cutout_vh, cp._cutout_vh, cp._cutout_vh));
      else
        cv::randu(selected_area, cv::Scalar(cp._cutout_vl),
                  cv::Scalar(cp._cutout_vh));

      if (store_rparams)
        {
          cp._w = w;
          cp._h = h;
          cp._rect_x = rect_x;
          cp._rect_y = rect_y;
        }
    }
  }

  void TorchImgRandAugCV::getEnlargedImage(const cv::Mat &in_img,
                                           const GeometryParams &cp,
                                           cv::Mat &in_img_enlarged)
  {
    int pad_mode = cv::BORDER_REFLECT101;
    switch (cp._geometry_pad_mode)
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
                                   const GeometryParams &cp,
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
    if (cp._geometry_zoom_out || cp._geometry_zoom_in)
      {
        bool zoom_in = cp._geometry_zoom_in;
        bool zoom_out = cp._geometry_zoom_out;
        if (cp._geometry_zoom_out && cp._geometry_zoom_in)
          {
            if (_bernouilli(_rnd_gen))
              zoom_in = false;
            else
              zoom_out = false;
          }

        float x0min, x0max, y0min, y0max;
        if (zoom_in)
          {
            x0max = cols + cols * cp._geometry_zoom_factor;
            y0max = rows + rows * cp._geometry_zoom_factor;
          }
        else
          {
            x0max = x0;
            y0max = y0;
          }
        if (zoom_out)
          {
            x0min = cols - cols * cp._geometry_zoom_factor;
            y0min = rows - rows * cp._geometry_zoom_factor;
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
    if (cp._geometry_persp_horizontal)
      {
        if (_bernouilli(_rnd_gen))
          {
            // seen from right
            outputQuad[0].y
                = rows * cp._geometry_persp_factor * _uniform_real_1(_rnd_gen);
            outputQuad[3].y = rows - outputQuad[0].y;
          }
        else
          {
            // seen from left
            outputQuad[1].y
                = rows * cp._geometry_persp_factor * _uniform_real_1(_rnd_gen);
            outputQuad[2].y = rows - outputQuad[1].y;
          }
      }
    if (cp._geometry_persp_vertical)
      {
        if (_bernouilli(_rnd_gen))
          {
            // seen from above
            outputQuad[3].x
                = cols * cp._geometry_persp_factor * _uniform_real_1(_rnd_gen);
            outputQuad[2].x = cols - outputQuad[3].x;
          }
        else
          {
            // seen from below
            outputQuad[0].x
                = cols * cp._geometry_persp_factor * _uniform_real_1(_rnd_gen);
            outputQuad[1].x = cols - outputQuad[0].x;
          }
      }
  }

  void TorchImgRandAugCV::applyGeometry(cv::Mat &src, GeometryParams &cp,
                                        const bool &store_rparams,
                                        const bool &sample)
  {
    if (!cp._prob)
      return;

    // enlarge image
    if (sample)
      {
        if (!roll_weighted_dice(cp._prob))
          return;
      }

    cv::Mat src_enlarged;
    getEnlargedImage(src, cp, src_enlarged);

    // Input Quadilateral or Image plane coordinates
    cv::Point2f inputQuad[4];
    // Output Quadilateral or World plane coordinates
    cv::Point2f outputQuad[4];

    // get perpective matrix
#pragma omp critical
    {
      if (sample)
        getQuads(src.rows, src.cols, cp, inputQuad, outputQuad);
    }

    // warp perspective
    cv::Mat lambda
        = (sample ? cv::getPerspectiveTransform(inputQuad, outputQuad)
                  : cp._lambda);
    int inter_flag
        = cv::INTER_NEAREST; //(sample ? cv::INTER_LINEAR : cv::INTER_NEAREST);
    int border_mode = (cp._geometry_pad_mode == 1 ? cv::BORDER_CONSTANT
                                                  : cv::BORDER_REPLICATE);
    cv::warpPerspective(src_enlarged, src, lambda, src.size(), inter_flag,
                        border_mode);

    if (store_rparams)
      cp._lambda = lambda;
  }

  void TorchImgRandAugCV::warpBBoxes(std::vector<std::vector<float>> &bboxes,
                                     cv::Mat lambda)
  {
    std::vector<std::vector<float>> nbboxes;
    for (size_t i = 0; i < bboxes.size(); ++i)
      {
        std::vector<float> bbox = bboxes.at(i);
        std::vector<cv::Point2f> origBBox;
        std::vector<cv::Point2f> warpedBBox;

        cv::Point2f p1;
        p1.x = bbox[0]; // xmin
        p1.y = bbox[1]; // ymin
        origBBox.push_back(p1);
        cv::Point2f p2;
        p2.x = bbox[2]; // xmax
        p2.y = bbox[3]; // ymax
        origBBox.push_back(p2);
        cv::Point2f p3;
        p3.x = bbox[0]; // xmin
        p3.y = bbox[3]; // ymax
        origBBox.push_back(p3);
        cv::Point2f p4;
        p4.x = bbox[2]; // xmax
        p4.y = bbox[1]; // ymin
        origBBox.push_back(p4);

        cv::perspectiveTransform(origBBox, warpedBBox, lambda);
        float xmin = warpedBBox[0].x;
        float ymin = warpedBBox[0].y;
        float xmax = warpedBBox[0].x;
        float ymax = warpedBBox[0].y;
        for (int i = 1; i < 4; ++i)
          {
            if (warpedBBox[i].x < xmin)
              xmin = warpedBBox[i].x;
            if (warpedBBox[i].x > xmax)
              xmax = warpedBBox[i].x;
            if (warpedBBox[i].y < ymin)
              ymin = warpedBBox[i].y;
            if (warpedBBox[i].y > ymax)
              ymax = warpedBBox[i].y;
          }

        std::vector<float> nbox = { xmin, ymin, xmax, ymax };
        nbboxes.push_back(nbox);
      }
    bboxes = nbboxes;
  }

  void TorchImgRandAugCV::filterBBoxes(std::vector<std::vector<float>> &bboxes,
                                       const GeometryParams &cp,
                                       const int &img_width,
                                       const int &img_height)
  {
    std::vector<std::vector<float>> nbboxes;
    for (size_t i = 0; i < bboxes.size(); ++i)
      {
        std::vector<float> bbox = bboxes.at(i);
        if (bbox[2] >= 0.0 && bbox[0] <= img_width && bbox[3] >= 0.0
            && bbox[1] <= img_height)
          {
            std::vector<float> nbbox;
            nbbox.push_back(std::max(0.0f, bbox[0])); // xmin
            nbbox.push_back(std::max(0.0f, bbox[1])); // ymin
            nbbox.push_back(
                std::min(static_cast<float>(img_width), bbox[2])); // xmax
            nbbox.push_back(
                std::min(static_cast<float>(img_height), bbox[3])); // ymax
            float surfbb = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]);
            float surfnbb = (nbbox[2] - nbbox[0]) * (nbbox[3] - nbbox[1]);
            if (surfnbb > cp._geometry_bbox_intersect
                              * surfbb) // keep bboxes that are at least 75%
                                        // as big as the original
              {
                nbboxes.push_back(nbbox);
              }
          }
      }
    bboxes = nbboxes;
  }

  void TorchImgRandAugCV::applyGeometryBBox(
      std::vector<std::vector<float>> &bboxes, const GeometryParams &cp,
      const int &img_width, const int &img_height)
  {
    // XXX: fix (enlarged bboxes for constant padding)
    for (size_t i = 0; i < bboxes.size(); ++i)
      {
        bboxes[i][0] += img_width;
        bboxes[i][2] += img_width;
        bboxes[i][1] += img_height;
        bboxes[i][3] += img_height;
      }

    // use cp lambda on bboxes
    warpBBoxes(bboxes, cp._lambda);

    // filter bboxes
    filterBBoxes(bboxes, cp, img_width, img_height);
  }

  void TorchImgRandAugCV::applyNoise(cv::Mat &src)
  {
    if (_noise_params._rgb)
      {
        cv::Mat bgr;
        cv::cvtColor(src, bgr, cv::COLOR_RGB2BGR);
        src = bgr;
      }

    if (_noise_params._decolorize)
      applyNoiseDecolorize(src);
    if (_noise_params._gauss_blur)
      applyNoiseGaussianBlur(src);
    if (_noise_params._hist_eq)
      applyNoiseHistEq(src);
    if (_noise_params._clahe)
      applyNoiseClahe(src);
    if (_noise_params._jpg)
      applyNoiseJPG(src);
    if (_noise_params._erosion)
      applyNoiseErosion(src);
    if (_noise_params._posterize)
      applyNoisePosterize(src);
    if (_noise_params._inverse)
      applyNoiseInverse(src);
    if (_noise_params._saltpepper)
      applyNoiseSaltpepper(src);
    if (_noise_params._convert_to_hsv)
      applyNoiseConvertHSV(src);
    if (_noise_params._convert_to_lab)
      applyNoiseConvertLAB(src);

    if (_noise_params._rgb)
      {
        cv::Mat rgb;
        cv::cvtColor(src, rgb, cv::COLOR_BGR2RGB);
        src = rgb;
      }
  }

  void TorchImgRandAugCV::applyNoiseDecolorize(cv::Mat &src)
  {
    if (!roll_weighted_dice(_noise_params._prob))
      return;
    if (src.channels() > 1)
      {
        cv::Mat grayscale;
        cv::cvtColor(src, grayscale, cv::COLOR_BGR2GRAY);
        cv::cvtColor(grayscale, src, cv::COLOR_GRAY2BGR);
      }
  }

  void TorchImgRandAugCV::applyNoiseGaussianBlur(cv::Mat &src)
  {
    if (!roll_weighted_dice(_noise_params._prob))
      return;
    cv::Mat out;
    cv::GaussianBlur(src, out, cv::Size(7, 7), 1.5);
    src = out;
  }

  void TorchImgRandAugCV::applyNoiseHistEq(cv::Mat &src)
  {
    if (!roll_weighted_dice(_noise_params._prob))
      return;
    if (src.channels() > 1)
      {
        cv::Mat ycrcb_image;
        cv::cvtColor(src, ycrcb_image, cv::COLOR_BGR2YCrCb);
        // Extract the L channel
        std::vector<cv::Mat> ycrcb_planes(3);
        cv::split(ycrcb_image, ycrcb_planes);
        // now we have the L image in ycrcb_planes[0]
        cv::Mat dst;
        cv::equalizeHist(ycrcb_planes[0], dst);
        ycrcb_planes[0] = dst;
        cv::merge(ycrcb_planes, ycrcb_image);
        // convert back to RGB
        cv::cvtColor(ycrcb_image, src, cv::COLOR_YCrCb2BGR);
      }
    else
      {
        cv::Mat temp_img;
        cv::equalizeHist(src, temp_img);
        src = temp_img;
      }
  }

  void TorchImgRandAugCV::applyNoiseClahe(cv::Mat &src)
  {
    if (!roll_weighted_dice(_noise_params._prob))
      return;
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(4);
    if (src.channels() > 1)
      {
        cv::Mat ycrcb_image;
        cv::cvtColor(src, ycrcb_image, cv::COLOR_BGR2YCrCb);
        // Extract the L channel
        std::vector<cv::Mat> ycrcb_planes(3);
        cv::split(ycrcb_image, ycrcb_planes);
        // now we have the L image in ycrcb_planes[0]
        cv::Mat dst;
        clahe->apply(ycrcb_planes[0], dst);
        ycrcb_planes[0] = dst;
        cv::merge(ycrcb_planes, ycrcb_image);
        // convert back to RGB
        cv::cvtColor(ycrcb_image, src, cv::COLOR_YCrCb2BGR);
      }
    else
      {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
        clahe->setClipLimit(4);
        cv::Mat temp_img;
        clahe->apply(src, temp_img);
        src = temp_img;
      }
  }

  void TorchImgRandAugCV::applyNoiseJPG(cv::Mat &src)
  {
    if (!roll_weighted_dice(_noise_params._prob))
      return;
    std::vector<uchar> buf;
    std::vector<int> params;
    params.push_back(cv::IMWRITE_JPEG_QUALITY);
#pragma omp critical
    {
      params.push_back(_uniform_real_1(_rnd_gen) * 100.0);
    }
    cv::imencode(".jpg", src, buf, params);
    src = cv::imdecode(buf, cv::IMREAD_COLOR);
  }

  void TorchImgRandAugCV::applyNoiseErosion(cv::Mat &src)
  {
    if (!roll_weighted_dice(_noise_params._prob))
      return;
    cv::Mat element
        = cv::getStructuringElement(2, cv::Size(3, 3), cv::Point(1, 1));
    cv::erode(src, src, element);
  }

  void TorchImgRandAugCV::applyNoisePosterize(cv::Mat &src)
  {
    if (!roll_weighted_dice(_noise_params._prob))
      return;
    int div = 64;
    cv::Mat lookUpTable(1, 256, CV_8U);
    uchar *p = lookUpTable.data;
    const int div_2 = div / 2;
    for (int i = 0; i < 256; ++i)
      {
        p[i] = i / div * div + div_2;
      }
    cv::Mat tmp_img;
    cv::LUT(src, lookUpTable, tmp_img);
    src = tmp_img;
  }

  void TorchImgRandAugCV::applyNoiseInverse(cv::Mat &src)
  {
    if (!roll_weighted_dice(_noise_params._prob))
      return;
    cv::Mat tmp_img;
    cv::bitwise_not(src, tmp_img);
    src = tmp_img;
  }

  void TorchImgRandAugCV::applyNoiseSaltpepper(cv::Mat &src)
  {
    if (!roll_weighted_dice(_noise_params._prob))
      return;
    const int noise_pixels_n
        = std::floor(_noise_params._saltpepper_fraction * src.cols * src.rows);
    const std::vector<uchar> val = { 0, 0, 0 };
    if (src.channels() == 1)
      {
#pragma omp critical
        {
          for (int k = 0; k < noise_pixels_n; ++k)
            {
              const int i = _uniform_real_1(_rnd_gen) * src.cols;
              const int j = _uniform_real_1(_rnd_gen) * src.rows;
              uchar *ptr = src.ptr<uchar>(j);
              ptr[i] = val[0];
            }
        }
      }
    else if (src.channels() == 3)
      { // color image
#pragma omp critical
        {
          for (int k = 0; k < noise_pixels_n; ++k)
            {
              const int i = _uniform_real_1(_rnd_gen) * src.cols;
              const int j = _uniform_real_1(_rnd_gen) * src.rows;
              cv::Vec3b *ptr = src.ptr<cv::Vec3b>(j);
              (ptr[i])[0] = val[0];
              (ptr[i])[1] = val[1];
              (ptr[i])[2] = val[2];
            }
        }
      }
  }

  void TorchImgRandAugCV::applyNoiseConvertHSV(cv::Mat &src)
  {
    if (!roll_weighted_dice(_noise_params._prob))
      return;
    cv::Mat hsv_image;
    cv::cvtColor(src, hsv_image, cv::COLOR_BGR2HSV);
    src = hsv_image;
  }

  void TorchImgRandAugCV::applyNoiseConvertLAB(cv::Mat &src)
  {
    if (!roll_weighted_dice(_noise_params._prob))
      return;
    int orig_depth = src.depth();
    cv::Mat lab_image;
    src.convertTo(lab_image, CV_32F);
    lab_image *= 1.0 / 255;
    cv::cvtColor(lab_image, src, cv::COLOR_BGR2Lab);
    src.convertTo(lab_image, orig_depth);
    src = lab_image;
  }
}
