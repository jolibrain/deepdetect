/**
 * DeepDetect
 * Copyright (c) 2018 Jolibrain
 * Author: Corentin Barreau <corentin.barreau@epitech.eu>
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

#include "outputconnectorstrategy.h"
#include "ncnnlib.h"
#include "segmentation.h"
#include "unsupervisedoutputconnector.h"
#include "ops/ops.h"
#include "ncnninputconns.h"
#include "dto/service_predict.hpp"
#include <cmath>
#include <numeric>

namespace dd
{
  namespace
  {
    struct Detection
    {
      int label;
      float score;
      cv::Rect2f box;
    };

    std::vector<Detection> decode_yolox(const ncnn::Mat &out, int classes,
                                        float threshold, float nms, int limit)
    {
      if (out.dims != 2 || classes < 2 || out.w - 5 != classes - 1)
        throw MLLibBadParamException(
            "YOLOX expects rows of decoded cx,cy,w,h,objectness and "
            "foreground scores; nclasses includes background");
      if (!(nms >= 0 && nms <= 1))
        throw MLLibBadParamException(
            "nms_threshold must be between zero and one");
      std::vector<Detection> candidates, selected;
      for (int i = 0; i < out.h; ++i)
        {
          const float *r = out.row(i);
          const float *best = std::max_element(r + 5, r + out.w);
          float score = r[4] * *best;
          if (score < threshold || r[2] <= 0 || r[3] <= 0)
            continue;
          candidates.push_back(
              { int(best - (r + 5)) + 1,
                score,
                { r[0] - r[2] / 2, r[1] - r[3] / 2, r[2], r[3] } });
        }
      std::stable_sort(candidates.begin(), candidates.end(),
                       [](const Detection &a, const Detection &b) {
                         return a.score > b.score;
                       });
      for (const auto &candidate : candidates)
        {
          bool overlaps = std::any_of(
              selected.begin(), selected.end(), [&](const Detection &other) {
                if (candidate.label != other.label)
                  return false;
                float intersection = (candidate.box & other.box).area();
                return intersection
                       > nms
                             * (candidate.box.area() + other.box.area()
                                - intersection);
              });
          if (!overlaps)
            selected.push_back(candidate);
          if (limit > 0 && int(selected.size()) == limit)
            break;
        }
      return selected;
    }

    std::string endpoint(const std::vector<const char *> &names,
                         const oatpp::String &configured)
    {
      if (configured && !configured->empty())
        return *configured;
      if (names.size() != 1)
        throw MLLibBadParamException(
            "ambiguous NCNN endpoint; specify its blob name");
      return names.front();
    }
  }

  template <class I, class O, class M>
  NCNNLib<I, O, M>::NCNNLib(const NCNNModel &model)
      : MLLib<I, O, M>(model), _net(std::make_unique<ncnn::Net>())
  {
    this->_libname = "ncnn";
  }

  template <class I, class O, class M>
  NCNNLib<I, O, M>::NCNNLib(NCNNLib &&other) noexcept
      : MLLib<I, O, M>(std::move(other)), _net(std::move(other._net)),
        _init_dto(std::move(other._init_dto)), _input(std::move(other._input)),
        _output(std::move(other._output)), _format(std::move(other._format))
  {
    this->_libname = "ncnn";
  }

  template <class I, class O, class M>
  void NCNNLib<I, O, M>::init_mllib(const APIData &ad)
  {
    _init_dto = ad.createSharedDTO<DTO::MLLib>();
    if (_init_dto->gpu || _init_dto->datatype != "fp32"
        || _init_dto->threads < 1)
      throw MLLibBadParamException(
          "NCNN requires CPU, fp32 and a positive thread count");
    auto &opt = _net->opt;
    opt.use_local_pool_allocator = false;
    opt.num_threads = *_init_dto->threads;
    opt.lightmode = *_init_dto->lightmode;
    opt.use_fp16_packed = opt.use_fp16_storage = opt.use_fp16_arithmetic
        = false;
    opt.use_bf16_storage = false;
    std::ifstream file(this->_mlmodel._params);
    bool ssd = false;
    for (std::string line; std::getline(file, line);)
      {
        std::istringstream fields(line);
        std::string layer;
        fields >> layer;
        if (layer == "ContinuationIndicator")
          throw MLLibBadParamException(
              "patched NCNN recurrent models require re-export");
        if (layer == "Flatten" || layer == "InnerProduct")
          for (std::string token; fields >> token;)
            if ((layer == "Flatten" && token == "0=2")
                || (layer == "InnerProduct" && token == "3=1"))
              throw MLLibBadParamException("this graph uses a patched NCNN "
                                           "layer and requires re-export");
        if (layer == "DetectionOutput")
          ssd = true;
      }
    ncnn_ops::register_layers(*_net);
    if (_net->load_param(this->_mlmodel._params.c_str())
        || _net->load_model(this->_mlmodel._weights.c_str()))
      throw MLLibBadParamException(
          "failed to load standard NCNN parameters or weights");
    if (_net->input_names().size() != 1)
      throw MLLibBadParamException(
          "NCNN image inference requires one graph input");
    _input = endpoint(_net->input_names(), _init_dto->inputBlob);
    _output = endpoint(_net->output_names(), _init_dto->outputBlob);
    if (_input != _net->input_names().front()
        || std::none_of(
            _net->blobs().begin(), _net->blobs().end(),
            [&](const ncnn::Blob &b) { return b.name == _output; }))
      throw MLLibBadParamException("unknown NCNN input or output blob name");
    _format = _init_dto->ncnn_output ? *_init_dto->ncnn_output
                                     : (ssd ? "ssd" : "probabilities");
    if (_format == "segmentation")
      {
        if (*_init_dto->nclasses < 1 || !this->_inputc.uses_stretch_resize())
          throw MLLibBadParamException("NCNN segmentation requires a positive "
                                       "class count and stretch resize");
        if (_init_dto->ncnn_segmentation_resize != "logits"
            && _init_dto->ncnn_segmentation_resize != "class_ids")
          throw MLLibBadParamException(
              "ncnn_segmentation_resize must be logits or class_ids");
        this->_mltype = "segmentation";
        return;
      }
    if (_format != "logits" && _format != "probabilities" && _format != "ssd"
        && _format != "yolox" && _format != "rtdetrv4")
      throw MLLibBadParamException("unknown ncnn_output");
    this->_mltype
        = (_format == "yolox" || _format == "ssd" || _format == "rtdetrv4")
              ? "detection"
              : "classification";
  }

  template <class I, class O, class M>
  void NCNNLib<I, O, M>::clear_mllib(const APIData &)
  {
  }

  template <class I, class O, class M>
  int NCNNLib<I, O, M>::train(const APIData &, APIData &)
  {
    throw MLLibBadParamException("NCNN supports inference only");
  }

  template <class I, class O, class M>
  oatpp::Object<DTO::PredictBody> NCNNLib<I, O, M>::predict(const APIData &ad)
  {
    auto request = ad.createSharedDTO<DTO::ServicePredict>();
    auto output = request->parameters->output;
    I input(this->_inputc);
    O connector(this->_outputc);
    this->_stats.transform_start();
    input.transform(ad);
    this->_stats.transform_end();
    this->_stats.inc_inference_count(input._in.size());
    std::vector<APIData> results;
    const bool detector = this->_mltype == "detection";
    if (*output->bbox && !detector)
      throw MLLibBadParamException(
          "bbox requires a detection output contract");
    const int classes = *_init_dto->nclasses;
    const bool segmentation = _format == "segmentation";
    const bool with_confidence
        = output->confidences && !output->confidences->empty();
    if (segmentation)
      {
        if (!input.uses_stretch_resize())
          throw MLLibBadParamException(
              "NCNN segmentation requires stretch resize");
        if (with_confidence
            && (output->confidences->size() != 1
                || output->confidences[0] != "best"))
          throw MLLibBadParamException(
              "NCNN segmentation supports confidences [best]");
        if (!(*output->confidence_threshold >= 0
              && *output->confidence_threshold <= 1))
          throw MLLibBadParamException(
              "segmentation confidence threshold must be between 0 and 1");
      }
    for (size_t i = 0; i < input._in.size(); ++i)
      {
        auto ex = _net->create_extractor();
        ncnn::Mat out;
        if (ex.input(_input.c_str(), input._in[i])
            || ex.extract(_output.c_str(), out))
          throw MLLibInternalException("NCNN inference failed for blob "
                                       + _output);
        if (!out.empty()
            && (out.elemsize != sizeof(float) || out.elempack != 1))
          throw MLLibInternalException(
              "expected an unpacked float32 NCNN output");
        if (!out.empty() && (out.dims == 1 || out.dims == 2))
          {
            const float *values = out;
            if (!std::all_of(values, values + out.w * size_t(out.h),
                             [](float v) { return std::isfinite(v); }))
              throw MLLibInternalException("non-finite NCNN output");
          }
        std::vector<double> scores;
        std::vector<std::string> labels;
        std::vector<APIData> boxes;
        APIData result;
        const auto &names = this->_mlmodel._hcorresp;
        auto add = [&](int label, float score) {
          auto it = names.find(label);
          labels.push_back(it == names.end() ? std::to_string(label)
                                             : it->second);
          scores.push_back(score);
        };
        if (segmentation)
          {
            const auto [height, width] = input._images_size.at(i);
            NCNNSegmentation mask;
            try
              {
                mask = decode_segmentation(
                    out, classes, { width, height },
                    _init_dto->ncnn_segmentation_resize == "logits",
                    with_confidence, *output->confidence_threshold);
              }
            catch (const std::invalid_argument &e)
              {
                throw MLLibBadParamException(e.what());
              }
            APIData size;
            size.add("width", width);
            size.add("height", height);
            result.add("imgsize", size);
            const int *ids = mask.ids.ptr<int>();
            result.add("vals",
                       std::vector<double>(ids, ids + mask.ids.total()));
            if (with_confidence)
              {
                APIData confidences;
                const float *prob = mask.confidence.ptr<float>();
                confidences.add(
                    "best",
                    std::vector<double>(prob, prob + mask.confidence.total()));
                result.add("confidences", confidences);
              }
          }
        else if (detector)
          {
            std::vector<Detection> detections;
            if (_format == "yolox")
              detections
                  = decode_yolox(out, classes, output->confidence_threshold,
                                 output->nms_threshold, output->best_bbox);
            else if (!out.empty())
              {
                if (out.dims != 2 || out.w != 6)
                  throw MLLibInternalException(
                      "expected six detection columns");
                for (int j = 0; j < out.h; ++j)
                  {
                    const float *row = out.row(j);
                    if (row[0] < 0 || row[1] < output->confidence_threshold)
                      continue;
                    if (row[0] >= float(std::numeric_limits<int>::max())
                        || std::floor(row[0]) != row[0])
                      throw MLLibInternalException(
                          "invalid class in detection output");
                    detections.push_back({ int(row[0]),
                                           row[1],
                                           { row[2], row[3], row[4] - row[2],
                                             row[5] - row[3] } });
                    if (output->best_bbox > 0
                        && int(detections.size()) == *output->best_bbox)
                      break;
                  }
              }
            const auto original = input._images_size.at(i);
            const auto content = input._content_sizes.at(i);
            float sx = _format == "ssd"
                           ? float(original.second - 1) * input._in[i].w
                                 / content.width
                           : float(original.second) / content.width;
            float sy = _format == "ssd"
                           ? float(original.first - 1) * input._in[i].h
                                 / content.height
                           : float(original.first) / content.height;
            if (_format == "rtdetrv4")
              {
                sx *= input._in[i].w;
                sy *= input._in[i].h;
              }
            for (const auto &d : detections)
              {
                APIData box;
                auto x = [&](float v) {
                  return double(
                      std::clamp(v * sx, 0.f, float(original.second - 1)));
                };
                auto y = [&](float v) {
                  return double(
                      std::clamp(v * sy, 0.f, float(original.first - 1)));
                };
                box.add("xmin", x(d.box.x));
                box.add("xmax", x(d.box.br().x));
                box.add("ymin", y(d.box.y));
                box.add("ymax", y(d.box.br().y));
                boxes.push_back(box);
                add(d.label, d.score);
              }
            result.add("bboxes", boxes);
          }
        else if (*output->ctc)
          {
            if (out.dims != 2 || out.empty())
              throw MLLibInternalException(
                  "expected a time by alphabet CTC output");
            std::string word;
            int previous = *output->blank_label;
            for (int t = 0; t < out.h; ++t)
              {
                const float *row = out.row(t);
                int id = std::max_element(row, row + out.w) - row;
                if (id != previous && id != *output->blank_label)
                  {
                    auto it = names.find(id);
                    word += char(it == names.end()
                                     ? id
                                     : std::atoi(it->second.c_str()));
                  }
                previous = id;
              }
            scores.push_back(1.);
            labels.push_back(word);
          }
        else
          {
            if (out.empty() || out.dims != 1
                || (classes > 0 && classes != out.w))
              throw MLLibInternalException(
                  "expected one NCNN score per configured class");
            std::vector<float> values(static_cast<const float *>(out),
                                      static_cast<const float *>(out) + out.w);
            if (_format == "logits")
              {
                float maximum
                    = *std::max_element(values.begin(), values.end());
                double sum = 0;
                for (float &v : values)
                  {
                    v = std::exp(v - maximum);
                    sum += v;
                  }
                for (float &v : values)
                  v /= sum;
              }
            int count = output->best && *output->best >= 0
                            ? std::min(*output->best, out.w)
                            : out.w;
            std::vector<int> ids(out.w);
            std::iota(ids.begin(), ids.end(), 0);
            std::partial_sort(ids.begin(), ids.begin() + count, ids.end(),
                              [&](int a, int b) {
                                return values[a] == values[b]
                                           ? a < b
                                           : values[a] > values[b];
                              });
            for (int j = 0; j < count; ++j)
              if (values[ids[j]] >= output->confidence_threshold)
                add(ids[j], values[ids[j]]);
          }
        result.add("uri", input._uris.at(i));
        result.add("loss", 0.0);
        if (!segmentation)
          {
            result.add("probs", scores);
            result.add("cats", labels);
          }
        results.push_back(std::move(result));
      }
    OutputConnectorConfig conf;
    conf._nclasses = classes;
    conf._has_bbox = detector;
    if ((_format == "yolox" || _format == "rtdetrv4")
        && output->best == nullptr)
      output->best = classes;
    oatpp::Object<DTO::PredictBody> body;
    if (segmentation)
      {
        UnsupervisedOutput masks;
        masks.add_results(results, true);
        body = masks.finalize(output, conf,
                              static_cast<MLModel *>(&this->_mlmodel));
      }
    else
      {
        connector.add_results(results);
        body = connector.finalize(output, conf,
                                  static_cast<MLModel *>(&this->_mlmodel));
      }
    if (ad.has("chain") && ad.get("chain").get<bool>())
      {
        body->_chain_input._imgs
            = input._orig_images.empty() ? input._images : input._orig_images;
        body->_chain_input._img_sizes = input._images_size;
      }
    return body;
  }
  template class NCNNLib<ImgNCNNInputFileConn, SupervisedOutput, NCNNModel>;
}
