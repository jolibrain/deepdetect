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
#include <thread>
#include <algorithm>
#include <iostream>

// NCNN
#include "cpu.h"
#include "net.h"

#include "ncnnlib.h"
#include "ncnninputconns.h"
#include "dto/service_predict.hpp"

namespace dd
{
  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  ncnn::UnlockedPoolAllocator
      NCNNLib<TInputConnectorStrategy, TOutputConnectorStrategy,
              TMLModel>::_blob_pool_allocator;
  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  ncnn::PoolAllocator
      NCNNLib<TInputConnectorStrategy, TOutputConnectorStrategy,
              TMLModel>::_workspace_pool_allocator;

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  NCNNLib<TInputConnectorStrategy, TOutputConnectorStrategy,
          TMLModel>::NCNNLib(const NCNNModel &cmodel)
      : MLLib<TInputConnectorStrategy, TOutputConnectorStrategy, NCNNModel>(
          cmodel)
  {
    this->_libname = "ncnn";
    _net = new ncnn::Net();
    _net->opt.num_threads = 1;
    _net->opt.blob_allocator = &_blob_pool_allocator;
    _net->opt.workspace_allocator = &_workspace_pool_allocator;
    _net->opt.lightmode = true;
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  NCNNLib<TInputConnectorStrategy, TOutputConnectorStrategy,
          TMLModel>::NCNNLib(NCNNLib &&tl) noexcept
      : MLLib<TInputConnectorStrategy, TOutputConnectorStrategy, NCNNModel>(
          std::move(tl))
  {
    this->_libname = "ncnn";
    _net = tl._net;
    tl._net = nullptr;
    _timeserie = tl._timeserie;
    _old_height = tl._old_height;
    _init_dto = tl._init_dto;
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  NCNNLib<TInputConnectorStrategy, TOutputConnectorStrategy,
          TMLModel>::~NCNNLib()
  {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdelete-non-virtual-dtor"
    delete _net;
#pragma GCC diagnostic pop
    _net = nullptr;
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  void NCNNLib<TInputConnectorStrategy, TOutputConnectorStrategy,
               TMLModel>::init_mllib(const APIData &ad)
  {
    _init_dto = ad.createSharedDTO<DTO::MLLib>();

    bool use_fp32 = (_init_dto->datatype == "fp32");
    _net->opt.use_fp16_packed = !use_fp32;
    _net->opt.use_fp16_storage = !use_fp32;
    _net->opt.use_fp16_arithmetic = !use_fp32;

    int res = _net->load_param(this->_mlmodel._params.c_str());
    if (res != 0)
      {
        this->_logger->error(
            "problem while loading ncnn params {} from repo {}",
            this->_mlmodel._params, this->_mlmodel._repo);
        throw MLLibBadParamException("could not load ncnn params ["
                                     + this->_mlmodel._params + "] from repo ["
                                     + this->_mlmodel._repo + "]");
      }
    res = _net->load_model(this->_mlmodel._weights.c_str());
    if (res != 0)
      {
        this->_logger->error(
            "problem while loading ncnn weigths {} from repo {}",
            this->_mlmodel._weights, this->_mlmodel._repo);
        throw MLLibBadParamException(
            "could not load ncnn weights [" + this->_mlmodel._weights
            + "] from repo [" + this->_mlmodel._repo + "]");
      }
    _old_height = this->_inputc.height();
    _net->set_input_h(_old_height);

    _timeserie = this->_inputc._timeserie;
    if (_timeserie)
      this->_mltype = "timeserie";

    _net->opt.lightmode = _init_dto->lightmode;
    _blob_pool_allocator.set_size_compare_ratio(0.0f);
    _workspace_pool_allocator.set_size_compare_ratio(0.5f);
    model_type(this->_mlmodel._params, this->_mltype);
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  void NCNNLib<TInputConnectorStrategy, TOutputConnectorStrategy,
               TMLModel>::clear_mllib(const APIData &ad)
  {
    (void)ad;
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  int NCNNLib<TInputConnectorStrategy, TOutputConnectorStrategy,
              TMLModel>::train(const APIData &ad, APIData &out)
  {
    (void)ad;
    (void)out;
    return 0;
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  oatpp::Object<DTO::PredictBody>
  NCNNLib<TInputConnectorStrategy, TOutputConnectorStrategy,
          TMLModel>::predict(const APIData &ad)
  {
    auto predict_dto = ad.createSharedDTO<DTO::ServicePredict>();

    TInputConnectorStrategy inputc(this->_inputc);
    TOutputConnectorStrategy tout(this->_outputc);

    this->_stats.transform_start();
    try
      {
        inputc.transform(ad);
      }
    catch (...)
      {
        throw;
      }
    this->_stats.transform_end();

    this->_stats.inc_inference_count(inputc._ids.size());

    // if height (timestep) changes we need to clear net before recreating an
    // extractor with new height, and to reload params and models after clear()
    if (_old_height != -1 && _old_height != inputc.height())
      {
        // in case of new prediction with different timesteps value, clear old
        // net
        _net->clear();
        _net->load_param(this->_mlmodel._params.c_str());
        _net->load_model(this->_mlmodel._weights.c_str());
        _old_height = inputc.height();
        _net->set_input_h(_old_height);
      }

    auto output_params = predict_dto->parameters->output;

    // Extract detection or classification
    std::string out_blob;
    if (_init_dto->outputBlob != nullptr)
      out_blob = _init_dto->outputBlob;

    if (out_blob.empty())
      {
        if (output_params->bbox == true)
          out_blob = "detection_out";
        else if (output_params->ctc == true)
          out_blob = "probs";
        else if (_timeserie)
          out_blob = "rnn_pred";
        else
          out_blob = "prob";
      }

    // Get best
    if (output_params->best == nullptr || output_params->best == -1
        || output_params->best > _init_dto->nclasses)
      output_params->best = _init_dto->nclasses;

    std::vector<APIData> vrad;

    // for loop around batch size
#pragma omp parallel for num_threads(*_init_dto->threads)
    for (size_t b = 0; b < inputc._ids.size(); b++)
      {
        std::vector<double> probs;
        std::vector<std::string> cats;
        std::vector<APIData> bboxes;
        std::vector<APIData> series;
        APIData rad;

        ncnn::Extractor ex = _net->create_extractor();
        ex.set_num_threads(_init_dto->threads);
        ex.input(_init_dto->inputBlob->c_str(), inputc._in.at(b));

        int ret = ex.extract(out_blob.c_str(), inputc._out.at(b));
        if (ret == -1)
          {
            throw MLLibInternalException("NCNN internal error");
          }

        if (output_params->bbox)
          {
            std::string uri = inputc._ids.at(b);
            auto bit = inputc._imgs_size.find(uri);
            int rows = 1;
            int cols = 1;
            if (bit != inputc._imgs_size.end())
              {
                // original image size
                rows = (*bit).second.first;
                cols = (*bit).second.second;
              }
            else
              {
                throw MLLibInternalException(
                    "Couldn't find original image size for " + uri);
              }
            for (int i = 0; i < inputc._out.at(b).h; i++)
              {
                const float *values = inputc._out.at(b).row(i);
                if (output_params->best_bbox > 0
                    && bboxes.size()
                           >= static_cast<size_t>(output_params->best_bbox))
                  break;
                if (values[1] < output_params->confidence_threshold)
                  break; // output is sorted by confidence

                cats.push_back(this->_mlmodel.get_hcorresp(values[0]));
                probs.push_back(values[1]);

                APIData ad_bbox;
                ad_bbox.add("xmin",
                            static_cast<double>(values[2] * (cols - 1)));
                ad_bbox.add("ymin",
                            static_cast<double>(values[3] * (rows - 1)));
                ad_bbox.add("xmax",
                            static_cast<double>(values[4] * (cols - 1)));
                ad_bbox.add("ymax",
                            static_cast<double>(values[5] * (rows - 1)));
                bboxes.push_back(ad_bbox);
              }
          }
        else if (output_params->ctc)
          {
            int alphabet = inputc._out.at(b).w;
            int time_step = inputc._out.at(b).h;
            std::vector<int> pred_label_seq_with_blank(time_step);
            for (int t = 0; t < time_step; ++t)
              {
                const float *values = inputc._out.at(b).row(t);
                pred_label_seq_with_blank[t] = std::distance(
                    values, std::max_element(values, values + alphabet));
              }

            std::vector<int> pred_label_seq;
            int prev = output_params->blank_label;
            for (int t = 0; t < time_step; ++t)
              {
                int cur = pred_label_seq_with_blank[t];
                if (cur != prev && cur != output_params->blank_label)
                  pred_label_seq.push_back(cur);
                prev = cur;
              }
            std::string outstr;
            std::ostringstream oss;
            for (auto l : pred_label_seq)
              outstr
                  += char(std::atoi(this->_mlmodel.get_hcorresp(l).c_str()));
            cats.push_back(outstr);
            probs.push_back(1.0);
          }
        else if (_timeserie)
          {
            std::vector<int> tsl = inputc._timeseries_lengths;
            for (unsigned int tsi = 0; tsi < tsl.size(); ++tsi)
              {
                for (int ti = 0; ti < tsl[tsi]; ++ti)
                  {
                    std::vector<double> predictions;
                    for (int k = 0; k < inputc._ntargets; ++k)
                      {
                        double res = inputc._out.at(b).row(ti)[k];
                        predictions.push_back(inputc.unscale_res(res, k));
                      }
                    APIData ts;
                    ts.add("out", predictions);
                    series.push_back(ts);
                  }
              }
          }
        else
          {
            std::vector<float> cls_scores;

            cls_scores.resize(inputc._out.at(b).w);
            for (int j = 0; j < inputc._out.at(b).w; j++)
              {
                cls_scores[j] = inputc._out.at(b)[j];
              }
            int size = cls_scores.size();
            std::vector<std::pair<float, int>> vec;
            vec.resize(size);
            for (int i = 0; i < size; i++)
              {
                vec[i] = std::make_pair(cls_scores[i], i);
              }

            std::partial_sort(vec.begin(), vec.begin() + output_params->best,
                              vec.end(),
                              std::greater<std::pair<float, int>>());

            for (int i = 0; i < output_params->best; i++)
              {
                if (vec[i].first < output_params->confidence_threshold)
                  continue;
                cats.push_back(this->_mlmodel.get_hcorresp(vec[i].second));
                probs.push_back(vec[i].first);
              }
          }

        rad.add("uri", inputc._ids.at(b));
        rad.add("loss", 0.0);
        rad.add("cats", cats);
        if (output_params->bbox)
          rad.add("bboxes", bboxes);
        if (_timeserie)
          {
            rad.add("series", series);
            rad.add("probs", std::vector<double>(series.size(), 1.0));
          }
        else
          rad.add("probs", probs);

#pragma omp critical
        {
          vrad.push_back(rad);
        }
      } // end for batch_size

    tout.add_results(vrad);
    int nclasses = this->_init_dto->nclasses;
    OutputConnectorConfig conf;
    conf._nclasses = nclasses;
    if (_timeserie)
      conf._timeseries = true;
    if (output_params->bbox == true)
      conf._has_bbox = true;
    oatpp::Object<DTO::PredictBody> out_dto
        = tout.finalize(predict_dto->parameters->output, conf,
                        static_cast<MLModel *>(&this->_mlmodel));

    // chain compliance
    if (ad.has("chain") && ad.get("chain").get<bool>())
      {
        if (typeid(inputc) == typeid(ImgNCNNInputFileConn))
          {
            if (!reinterpret_cast<ImgNCNNInputFileConn *>(&inputc)
                     ->_orig_images.empty())
              out_dto->_chain_input._imgs
                  = reinterpret_cast<ImgNCNNInputFileConn *>(&inputc)
                        ->_orig_images;
            else
              out_dto->_chain_input._imgs
                  = reinterpret_cast<ImgNCNNInputFileConn *>(&inputc)->_images;
            out_dto->_chain_input._img_sizes
                = reinterpret_cast<ImgNCNNInputFileConn *>(&inputc)
                      ->_images_size;
          }
      }

    // out_dto->status = 0;
    return out_dto;
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  void NCNNLib<TInputConnectorStrategy, TOutputConnectorStrategy,
               TMLModel>::model_type(const std::string &param_file,
                                     std::string &mltype)
  {
    std::ifstream paramf(param_file);
    std::stringstream content;
    content << paramf.rdbuf();

    std::size_t found_detection = content.str().find("DetectionOutput");
    if (found_detection != std::string::npos)
      {
        mltype = "detection";
        return;
      }
    std::size_t found_ocr = content.str().find("ContinuationIndicator");
    if (found_ocr != std::string::npos)
      {
        mltype = "ctc";
        return;
      }
    mltype = "classification";
  }

  template class NCNNLib<ImgNCNNInputFileConn, SupervisedOutput, NCNNModel>;
  template class NCNNLib<CSVTSNCNNInputFileConn, SupervisedOutput, NCNNModel>;
}
