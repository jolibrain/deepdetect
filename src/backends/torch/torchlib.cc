/**
 * DeepDetect
 * Copyright (c) 2019-2022 Jolibrain
 * Author: Louis Jean <louis.jean@jolibrain.com>
 *        Guillaume Infantes <guillaume.infantes@jolibrain.com>
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

#include "torchlib.h"

#include "outputconnectorstrategy.h"

#include "generators/net_caffe.h"
#include "generators/net_caffe_recurrent.h"

#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>

#include "native/native.h"
#include "torchsolver.h"
#include "torchloss.h"
#include "torchutils.h"

#include "dto/mllib.hpp"
#include "utils/bbox.hpp"

using namespace torch;

namespace dd
{

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  TorchLib<TInputConnectorStrategy, TOutputConnectorStrategy,
           TMLModel>::TorchLib(const TorchModel &tmodel)
      : MLLib<TInputConnectorStrategy, TOutputConnectorStrategy, TorchModel>(
          tmodel)
  {
    this->_libname = "torch";
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  TorchLib<TInputConnectorStrategy, TOutputConnectorStrategy,
           TMLModel>::TorchLib(TorchLib &&tl) noexcept
      : MLLib<TInputConnectorStrategy, TOutputConnectorStrategy, TorchModel>(
          std::move(tl))
  {
    this->_libname = "torch";
    _module = std::move(tl._module);
    _template = tl._template;
    _nclasses = tl._nclasses;
    _main_device = tl._main_device;
    _devices = tl._devices;
    _masked_lm = tl._masked_lm;
    _seq_training = tl._seq_training;
    _finetuning = tl._finetuning;
    _best_metrics = tl._best_metrics;
    _best_metric_values = tl._best_metric_values;
    _classification = tl._classification;
    _regression = tl._regression;
    _timeserie = tl._timeserie;
    _bbox = tl._bbox;
    _segmentation = tl._segmentation;
    _ctc = tl._ctc;
    _multi_label = tl._multi_label;
    _concurrent_predict = tl._concurrent_predict;
    _loss = tl._loss;
    _template_params = tl._template_params;
    _dtype = tl._dtype;
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  TorchLib<TInputConnectorStrategy, TOutputConnectorStrategy,
           TMLModel>::~TorchLib()
  {
    _module.free();
    torch_utils::free_gpu_memory();
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  double TorchLib<TInputConnectorStrategy, TOutputConnectorStrategy,
                  TMLModel>::unscale(double val, unsigned int k,
                                     const TInputConnectorStrategy &inputc)
  {
    (void)inputc;
    (void)k;
    // unscaling is input connector specific
    return val;
  }

  // full template specialization
  template <>
  double
  TorchLib<CSVTSTorchInputFileConn, SupervisedOutput, TorchModel>::unscale(
      double val, unsigned int k, const CSVTSTorchInputFileConn &inputc)
  {
    if (!inputc._scale)
      return val;
    if (inputc._scale_type == MINMAX
        && (inputc._min_vals.empty() || inputc._max_vals.empty()))
      {
        this->_logger->info("not unscaling output because no bounds "
                            "data found");
        return val;
      }
    else if (inputc._scale_type == ZNORM
             && (inputc._mean_vals.empty() || inputc._variance_vals.empty()))
      {
        this->_logger->info("not unscaling output because no bounds "
                            "data found");
        return val;
      }
    else
      {
        if (!inputc._dont_scale_labels)
          {
            if (inputc._scale_type == MINMAX)
              {
                double max, min;
                if (inputc._label_pos.size() > 0) // labels case
                  {
                    max = inputc._max_vals[inputc._label_pos[k]];
                    min = inputc._min_vals[inputc._label_pos[k]];
                  }
                else // forecast case
                  {
                    max = inputc._max_vals[k];
                    min = inputc._min_vals[k];
                  }
                if (inputc._scale_between_minus_half_and_half)
                  val += 0.5;
                val = val * (max - min) + min;
              }
            else if (inputc._scale_type == ZNORM)
              {
                double mean, variance;
                if (inputc._label_pos.size() > 0) // labels case
                  {
                    mean = inputc._mean_vals[inputc._label_pos[k]];
                    variance = inputc._variance_vals[inputc._label_pos[k]];
                  }
                else // forecast case
                  {
                    mean = inputc._mean_vals[k];
                    variance = inputc._variance_vals[k];
                  }
                val = val * (sqrt(variance)) + mean;
              }
            else
              throw MLLibInternalException("unknwon scale type");
          }

        return val;
      }
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  void TorchLib<TInputConnectorStrategy, TOutputConnectorStrategy,
                TMLModel>::compute_and_print_model_info()
  {
    _module.compute_and_print_model_info();
    this->_model_params = _module._params_count;
    this->_model_frozen_params = _module._frozen_params_count;
  }

  /*- from mllib -*/
  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  void TorchLib<TInputConnectorStrategy, TOutputConnectorStrategy,
                TMLModel>::init_mllib(const APIData &lib_ad)
  {
    // Get parameters
    auto mllib_dto = lib_ad.createSharedDTO<DTO::MLLib>();

    if (mllib_dto->from_repository != nullptr)
      {
        this->_mlmodel.copy_to_target(mllib_dto->from_repository,
                                      this->_mlmodel._repo, this->_logger);
        this->_mlmodel.read_from_repository(this->_logger);
      }
    _template = mllib_dto->model_template;

    if (mllib_dto->gpu && !torch_utils::is_gpu_available())
      {
        throw MLLibBadParamException(
            "GPU is not available, service could not be created");
      }

    _concurrent_predict = mllib_dto->concurrent_predict;
    std::vector<int> gpuids = mllib_dto->gpuid->_ids;

    if (mllib_dto->nclasses != 0)
      {
        _nclasses = mllib_dto->nclasses;
        this->_inputc._nclasses = _nclasses;
      }
    else if (mllib_dto->ntargets != 0)
      {
        _nclasses = mllib_dto->ntargets;
      }

    std::string self_supervised = mllib_dto->self_supervised;
    int embedding_size = mllib_dto->embedding_size;
    bool freeze_traced = mllib_dto->freeze_traced;
    _finetuning = mllib_dto->finetuning;
    _loss = mllib_dto->loss;

    auto template_params_dto = mllib_dto->template_params;
    if (template_params_dto != nullptr)
      {
        _template_params = lib_ad.getobj("template_params");
      }
    if (mllib_dto->nclasses != 0)
      _template_params.add("nclasses", static_cast<int>(mllib_dto->nclasses));
    if (mllib_dto->timesteps > 0)
      _template_params.add("timesteps",
                           static_cast<int>(mllib_dto->timesteps));

    std::string dt = mllib_dto->datatype;
    if (dt == "fp32")
      {
        _dtype = torch::kFloat32;
        this->_logger->info("will predict in FP32");
      }
    else if (dt == "fp16")
      {
        _dtype = torch::kFloat16;
        this->_logger->info("will predict in FP16");
      }
    else if (dt == "fp64")
      {
        _dtype = torch::kFloat64;
        this->_logger->info("will predict in FP64");
      }
    else
      throw MLLibBadParamException("unknown datatype " + dt);

    _module._dtype = _dtype;

    // Find GPU id
    if (mllib_dto->gpu != true)
      {
        _devices = { torch::Device(DeviceType::CPU) };
      }
    else
      {
        // if gpuids = -1, we use all gpus
        if (gpuids.size() == 1 && gpuids[0] == -1)
          {
            gpuids.resize(torch_utils::get_gpu_count());
            std::iota(gpuids.begin(), gpuids.end(), 0);
          }

        if (gpuids.empty())
          gpuids.push_back(0);

        for (int gpuid : gpuids)
          {
#if USE_MPS
            _devices.push_back(torch::Device(DeviceType::MPS, gpuid));
#else
            _devices.push_back(torch::Device(DeviceType::CUDA, gpuid));
#endif
          }

        if (_devices.size() > 1)
          {
            std::string devices_str = std::to_string(gpuids[0]);
            for (size_t i = 1; i < _devices.size(); ++i)
              devices_str += "," + std::to_string(gpuids[i]);
            this->_logger->info("Running on multiple devices: " + devices_str);
          }
      }

    _main_device = _devices[0];

    // Set model type
    if (mllib_dto->segmentation)
      {
        _segmentation = true;
      }
    else if (mllib_dto->ctc || this->_inputc._ctc)
      {
        _ctc = true;
      }
    else if (mllib_dto->nclasses != 0)
      {
        _classification = true;
      }
    else if (mllib_dto->ntargets != 0)
      {
        _regression = true;
      }

    if (mllib_dto->multi_label)
      {
        _multi_label = true;
      }

    if (_template == "bert")
      {
        if (!self_supervised.empty())
          {
            if (self_supervised != "mask")
              {
                throw MLLibBadParamException(
                    "Bert does not support self_supervised type: "
                    + self_supervised);
              }
            this->_logger->info("Masked Language model");
            _masked_lm = true;
            _seq_training = true;
          }
        else if (!_classification)
          {
            throw MLLibBadParamException(
                "BERT only supports self-supervised or classification");
          }
      }
    else if (_template == "gpt2")
      {
        _seq_training = true;
      }
    else if (_template == "fasterrcnn" || _template == "retinanet"
             || _template == "detr" || _template == "yolox")
      {
        _bbox = true;
        _classification = false;
      }
    else if (_template.find("crnn") != std::string::npos)
      {
        _ctc = true;
      }
    else if (_template.find("recurrent") != std::string::npos
             || NativeFactory::is_timeserie(_template))
      {
        _timeserie = true;
      }
    if (!_regression && !_timeserie && !_bbox && !_segmentation && !_ctc
        && self_supervised.empty())
      _classification = true; // classification is default

    // Set mltype
    if (_bbox)
      this->_mltype = "detection";
    else if (_timeserie)
      this->_mltype = "timeserie";
    else if (_regression)
      this->_mltype = "regression";
    else if (_classification)
      this->_mltype = "classification";
    else if (_segmentation)
      this->_mltype = "segmentation";
    else if (_ctc)
      this->_mltype = "ctc";

    this->_logger->info("mlltype={}", this->_mltype);

    // Create the model
    _module._device = _main_device;
    _module._logger = this->_logger;
    _module._finetuning = _finetuning;

    if (_template == "recurrent")
      {
        // call caffe net generator before everything else
        std::string dest_net
            = this->_mlmodel._repo + '/' + _template + ".prototxt";
        if (!fileops::file_exists(dest_net))
          {
            caffe::NetParameter net_param;
            configure_recurrent_template(lib_ad, this->_inputc, net_param,
                                         this->_logger, true);
            torch_utils::torch_write_proto_to_text_file(net_param, dest_net);
          }
        else
          {
            this->_logger->info("prototxt net definition already present in "
                                "repo : using old file");
          }
        this->_mlmodel._proto = dest_net;
      }

    bool model_not_found = this->_mlmodel._traced.empty()
                           && this->_mlmodel._proto.empty()
                           && !NativeFactory::valid_template_def(_template);

    if (model_not_found)
      throw MLLibInternalException("Use of libtorch backend needs either: "
                                   "traced net, protofile or native template");

    bool multiple_models_found
        = ((!this->_mlmodel._traced.empty() || !this->_mlmodel._proto.empty())
           + NativeFactory::valid_template_def(_template))
          > 1;
    if (multiple_models_found && !_ctc)
      {
        this->_logger->error("traced: {}, proto: {}, template: {}",
                             this->_mlmodel._traced, this->_mlmodel._proto,
                             _template);
        if (_finetuning && NativeFactory::valid_template_def(_template))
          throw MLLibInternalException(
              "If you want to finetune a model using a native template, move "
              "the weights to a .npt file");
        else
          throw MLLibInternalException(
              "Only one of these must be provided: traced net, protofile or "
              "native template");
      }

    // FIXME(louis): out of if(bert) because we allow not to specify template
    // at predict. Should we change this?
    this->_inputc._input_format = "bert";
    if (_template == "bert")
      {
        // No allocation, use traced model in repo
        if (_classification)
          {
            _module._linear_head = nn::Linear(embedding_size, _nclasses);
            _module._linear_head->to(_main_device);
            _module._hidden_states = true;
            _module._linear_in = 1;
          }
      }
    else if (_template == "gpt2")
      {
        // No allocation, use traced model in repo
        this->_inputc._input_format = "gpt2";
      }
    // TorchVision detection models
    else if (_template == "fasterrcnn" || _template == "retinanet"
             || _template == "yolox")
      {
        // torchvision models output is a tuple (Loss, Predictions)
        _module._loss_id = 0;
        _module._linear_in = 1;
      }
    else if (_template == "detr")
      {
      }
    else if (!_template.empty())
      {
        bool model_allocated_at_train = NativeFactory::is_timeserie(_template)
                                        || NativeFactory::is_ctc(_template)
                                        || _template == "recurrent";

        if (model_allocated_at_train)
          this->_logger->info("Model allocated during training");
        else if (NativeFactory::valid_template_def(_template))
          {
            _module.create_native_template<TInputConnectorStrategy>(
                _template, _template_params, this->_inputc, this->_mlmodel,
                _main_device);

            // XXX: greyscale images are not supported by torchvision models
            if (std::is_same<TInputConnectorStrategy,
                             ImgTorchInputFileConn>::value
                && !NativeFactory::template_supports_bw(_template))
              {
                dynamic_cast<ImgTorchInputFileConn *>(&this->_inputc)
                    ->_supports_bw
                    = false;
              }
          }
        else
          throw MLLibBadParamException("invalid torch model template "
                                       + _template);
      }
    if (_classification || _regression || _segmentation)
      {
        _module._nclasses = _nclasses;

        if (_finetuning && !_module._native)
          {
            _module._require_linear_head = true;
            this->_logger->info("Add linear layer on top of the traced model");
          }
      }
    else if (_ctc)
      {
        if (_finetuning && !this->_mlmodel._traced.empty())
          {
            _module._linear_in = 0;
            _module._require_crnn_head = true;
            this->_logger->info("Add CRNN head on top of the traced model");
          }
      }

    // Load weights
    _module.load(this->_mlmodel);
    _module.freeze_traced(freeze_traced);

    // print
    if (_module.is_ready(_template))
      {
        compute_and_print_model_info();
      }

    _best_metrics = { "map", "meaniou",  "mlacc", "delta_score_0.1", "bacc",
                      "f1",  "net_meas", "acc",   "L1_mean_error",   "eucll" };
    _best_metric_values.resize(1, std::numeric_limits<double>::infinity());
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  void TorchLib<TInputConnectorStrategy, TOutputConnectorStrategy,
                TMLModel>::clear_mllib(__attribute__((unused))
                                       const APIData &ad)
  {
    std::vector<std::string> extensions{ ".json", ".pt", ".ptw" };
    fileops::remove_directory_files(this->_mlmodel._repo, extensions);
    this->_logger->info("Torchlib service cleared");
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  bool TorchLib<TInputConnectorStrategy, TOutputConnectorStrategy,
                TMLModel>::is_better(double v1, double v2,
                                     std::string metric_name)
  {
    if (metric_name == "eucll" || metric_name == "delta_score_0.1"
        || metric_name == "L1_mean_error")
      return (v2 >= v1);
    return (v1 >= v2);
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  void TorchLib<TInputConnectorStrategy, TOutputConnectorStrategy, TMLModel>::
      save_if_best(APIData &ad_out, int64_t elapsed_it, TorchSolver &tsolver,
                   std::vector<int64_t> &best_iteration_numbers)
  {
    for (size_t i = 0; i < best_iteration_numbers.size(); ++i)
      {
        APIData meas_out;
        if (i == 0)
          meas_out = ad_out.getobj("measure");
        else
          meas_out = ad_out.getv("measures").at(i - 1);

        double cur_meas = std::numeric_limits<double>::infinity();
        std::string meas;
        for (auto m : _best_metrics)
          {
            if (meas_out.has(m))
              {
                cur_meas = meas_out.get(m).template get<double>();
                meas = m;
                break;
              }
          }
        if (cur_meas == std::numeric_limits<double>::infinity())
          {
            // could not find value for measuring best
            this->_logger->info(
                "could not find any value for measuring best model");
            return;
          }
        if (_best_metric_values[i] == std::numeric_limits<double>::infinity()
            || is_better(cur_meas, _best_metric_values[i], meas))
          {
            if (best_iteration_numbers[i] != -1
                && dd_utils::unique(best_iteration_numbers[i],
                                    best_iteration_numbers))
              {
                remove_model(best_iteration_numbers[i]);
              }
            _best_metric_values[i] = cur_meas;
            this->snapshot(elapsed_it, tsolver);
            try
              {
                std::ofstream bestfile;
                std::string bestfilename;
                if (i >= 1)
                  bestfilename = this->_mlmodel._repo
                                 + fileops::insert_suffix(
                                     "_test_" + std::to_string(i - 1),
                                     this->_mlmodel._best_model_filename);
                else
                  bestfilename = this->_mlmodel._repo
                                 + this->_mlmodel._best_model_filename;

                bestfile.open(bestfilename, std::ios::out);
                bestfile << "iteration:" << elapsed_it << std::endl;
                bestfile << meas << ":" << cur_meas << std::endl;

                if (i >= 1)
                  {
                    bestfile << "test_name: ";
                    if (meas_out.has("test_name"))
                      bestfile << meas_out.get("test_name").get<std::string>();
                    else
                      bestfile << "noname_" + std::to_string(i - 1);
                    bestfile << std::endl;
                  }
                bestfile.close();
              }
            catch (std::exception &e)
              {
                this->_logger->error("could not write best model file");
              }
            best_iteration_numbers[i] = elapsed_it;
          }
      }
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  void TorchLib<TInputConnectorStrategy, TOutputConnectorStrategy,
                TMLModel>::remove_model(int64_t elapsed_it)
  {
    this->_logger->info("Deleting superseeded model {} ", elapsed_it);
    std::remove((this->_mlmodel._repo + "/solver-" + std::to_string(elapsed_it)
                 + ".pt")
                    .c_str());
    std::remove((this->_mlmodel._repo + "/checkpoint-"
                 + std::to_string(elapsed_it) + ".pt")
                    .c_str());
    std::remove((this->_mlmodel._repo + "/checkpoint-"
                 + std::to_string(elapsed_it) + ".npt")
                    .c_str());
    std::remove((this->_mlmodel._repo + "/checkpoint-"
                 + std::to_string(elapsed_it) + ".ptw")
                    .c_str());
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  void TorchLib<TInputConnectorStrategy, TOutputConnectorStrategy,
                TMLModel>::snapshot(int64_t elapsed_it, TorchSolver &tsolver)
  {
    this->_logger->info("Saving checkpoint after {} iterations", elapsed_it);
    // solver is allowed to modify net during eval()/train() => do this call
    // before saving net itself
    tsolver.eval();
    this->_module.save_checkpoint(this->_mlmodel, std::to_string(elapsed_it));
    tsolver.save(this->_mlmodel._repo + "/solver-" + std::to_string(elapsed_it)
                 + ".pt");
    tsolver.train();
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  int TorchLib<TInputConnectorStrategy, TOutputConnectorStrategy,
               TMLModel>::train(const APIData &ad, APIData &out)
  {
    using namespace std::chrono;
    this->_tjob_running.store(true);

    TInputConnectorStrategy inputc(this->_inputc);
    inputc._train = true;

    APIData ad_input = ad.getobj("parameters").getobj("input");
    if (ad_input.has("shuffle"))
      inputc._dataset.set_shuffle(ad_input.get("shuffle").get<bool>());
    inputc._dataset._classification = inputc._test_datasets._classification
        = _classification;

    try
      {
        inputc.transform(ad);
        bool module_was_ready = _module.is_ready(_template);
        _module.post_transform_train<TInputConnectorStrategy>(
            _template, _template_params, inputc, this->_mlmodel, _main_device);
        if (!module_was_ready)
          compute_and_print_model_info();
      }
    catch (...)
      {
        throw;
      }

    // set inputc dataset data augmentation options
    APIData ad_mllib = ad.getobj("parameters").getobj("mllib");
    if (typeid(inputc) == typeid(ImgTorchInputFileConn))
      {
        bool has_data_augmentation
            = ad_mllib.has("mirror") || ad_mllib.has("rotate")
              || ad_mllib.has("crop_size") || ad_mllib.has("cutout")
              || ad_mllib.has("geometry") || ad_mllib.has("noise")
              || ad_mllib.has("distort");
        if (has_data_augmentation)
          {
            bool has_mirror
                = ad_mllib.has("mirror") && ad_mllib.get("mirror").get<bool>();
            this->_logger->info("mirror: {}", has_mirror);
            bool has_rotate
                = ad_mllib.has("rotate") && ad_mllib.get("rotate").get<bool>();

            // disable rotate for non square image size
            if (inputc.width() != inputc.height() && has_rotate)
              {
                has_rotate = 0;
                this->_logger->warn(
                    "rotate augment was not applied. To enable rotate, select "
                    "img_width and img_height to be equal.");
              }
            this->_logger->info("rotate: {}", has_rotate);
            CropParams crop_params;
            if (ad_mllib.has("crop_size"))
              {
                int crop_size = ad_mllib.get("crop_size").get<int>();
                crop_params = CropParams(crop_size);
                if (ad_mllib.has("test_crop_samples"))
                  crop_params._test_crop_samples
                      = ad_mllib.get("test_crop_samples").get<int>();
                this->_logger->info("crop_size : {}", crop_size);
              }
            CutoutParams cutout_params;
            if (ad_mllib.has("cutout"))
              {
                float cutout = ad_mllib.get("cutout").get<double>();
                cutout_params = CutoutParams(cutout);
                this->_logger->info("cutout: {}", cutout);
              }
            GeometryParams geometry_params;
            APIData ad_geometry = ad_mllib.getobj("geometry");
            if (!ad_geometry.empty())
              {
                geometry_params._prob = ad_geometry.get("prob").get<double>();
                this->_logger->info("geometry: {}", geometry_params._prob);
                if (ad_geometry.has("persp_vertical"))
                  geometry_params._geometry_persp_vertical
                      = ad_geometry.get("persp_vertical").get<bool>();
                if (ad_geometry.has("persp_horizontal"))
                  geometry_params._geometry_persp_horizontal
                      = ad_geometry.get("persp_horizontal").get<bool>();
                if (ad_geometry.has("transl_vertical"))
                  geometry_params._geometry_transl_vertical
                      = ad_geometry.get("transl_vertical").get<bool>();
                if (ad_geometry.has("transl_horizontal"))
                  geometry_params._geometry_transl_horizontal
                      = ad_geometry.get("transl_horizontal").get<bool>();
                if (ad_geometry.has("zoom_out"))
                  geometry_params._geometry_zoom_out
                      = ad_geometry.get("zoom_out").get<bool>();
                if (ad_geometry.has("zoom_in"))
                  geometry_params._geometry_zoom_in
                      = ad_geometry.get("zoom_in").get<bool>();
                if (ad_geometry.has("persp_factor"))
                  geometry_params._geometry_persp_factor
                      = ad_geometry.get("persp_factor").get<double>();
                if (ad_geometry.has("transl_factor"))
                  geometry_params._geometry_transl_factor
                      = ad_geometry.get("transl_factor").get<double>();
                if (ad_geometry.has("zoom_factor"))
                  geometry_params._geometry_zoom_factor
                      = ad_geometry.get("zoom_factor").get<double>();
                if (ad_geometry.has("pad_mode"))
                  geometry_params.set_pad_mode(
                      ad_geometry.get("pad_mode").get<std::string>());
              }
            auto *img_ic = reinterpret_cast<ImgTorchInputFileConn *>(&inputc);
            NoiseParams noise_params(img_ic->_bw);
            noise_params._rgb = img_ic->_rgb;
            APIData ad_noise = ad_mllib.getobj("noise");
            if (!ad_noise.empty())
              {
                noise_params._prob = ad_noise.get("prob").get<double>();
                this->_logger->info("noise: {}", noise_params._prob);
              }
            DistortParams distort_params(img_ic->_bw);
            distort_params._rgb = img_ic->_rgb;
            APIData ad_distort = ad_mllib.getobj("distort");
            if (!ad_distort.empty())
              {
                distort_params._prob = ad_distort.get("prob").get<double>();
                this->_logger->info("distort: {}", distort_params._prob);
              }
            inputc._dataset._img_rand_aug_cv = TorchImgRandAugCV(
                has_mirror, has_rotate, crop_params, cutout_params,
                geometry_params, noise_params, distort_params);
            inputc._test_datasets.set_img_rand_aug_cv(TorchImgRandAugCV(
                has_mirror, has_rotate, crop_params, cutout_params,
                geometry_params, noise_params,
                distort_params)); // only uses cropping if enable
          }
      }
    int dataloader_threads = 1;
    if (ad_mllib.has("dataloader_threads"))
      dataloader_threads = ad_mllib.get("dataloader_threads").get<int>();

    Tensor class_weights = {};

    if (ad_mllib.has("class_weights"))
      {
        std::vector<double> cwv
            = ad_mllib.get("class_weights").get<std::vector<double>>();
        if (cwv.size() != _nclasses)
          {
            this->_logger->error("class weights given, but number of "
                                 "weights {} do not match "
                                 "number of classes {}, ignoring",
                                 cwv.size(), _nclasses);
          }
        else
          {
            this->_logger->info("using class weights");
            auto options = torch::TensorOptions().dtype(torch::kFloat64);
            class_weights
                = torch::from_blob(cwv.data(), { _nclasses }, options)
                      .to(torch::kFloat32)
                      .to(_main_device);
          }
      }

    // net params
    int64_t batch_size = 1;
    int64_t test_batch_size = 1;

    if (ad_mllib.has("net"))
      {
        APIData ad_net = ad_mllib.getobj("net");
        if (ad_net.has("batch_size"))
          batch_size = ad_net.get("batch_size").get<int>();
        if (ad_net.has("test_batch_size"))
          test_batch_size = ad_net.get("test_batch_size").get<int>();
        if (ad_net.has("reg_weight"))
          {
            _reg_weight = ad_net.get("reg_weight").get<double>();
            this->_logger->info("reg_weight={}", _reg_weight);
          }
      }

    // solver params
    int64_t iterations = 1;
    int64_t iter_size = 1;
    int64_t test_interval = 1;
    int64_t save_period = 0;

    // loss specific to the model
    if (_module.has_model_loss())
      _loss = _template;

    TorchLoss tloss(_loss, _module.has_model_loss(), _seq_training, _timeserie,
                    _regression, _classification, _segmentation, _ctc,
                    class_weights, _reg_weight, _module, this->_logger);
    TorchSolver tsolver(_module, tloss, this->_logger);

    // logging parameters
    int64_t log_batch_period = 20;

    if (ad_mllib.has("template_params"))
      _template_params = ad_mllib;

    if (ad_mllib.has("solver"))
      {
        APIData ad_solver = ad_mllib.getobj("solver");

        tsolver.configure(ad_solver);

        if (ad_solver.has("iterations"))
          iterations = ad_solver.get("iterations").get<int>();
        if (ad_solver.has("test_interval"))
          test_interval = ad_solver.get("test_interval").get<int>();
        if (ad_solver.has("iter_size"))
          iter_size = ad_solver.get("iter_size").get<int>();
        if (ad_solver.has("snapshot"))
          save_period = ad_solver.get("snapshot").get<int>();
      }

    bool retain_graph = ad_mllib.has("retain_graph")
                            ? ad_mllib.get("retain_graph").get<bool>()
                            : false;

    if (iter_size <= 0)
      iter_size = 1;

    size_t gpu_count = _devices.size();

    // create dataset for evaluation during training
    TorchMultipleDataset &eval_dataset = inputc._test_datasets;
    if (eval_dataset.size() == 0)
      throw MLLibBadParamException("empty test dataset");

    // create solver
    std::vector<int64_t> best_iteration_numbers(eval_dataset.size() + 1, -1);
    _best_metric_values.resize(eval_dataset.size() + 1,
                               std::numeric_limits<double>::infinity());
    this->_test_names = eval_dataset.names();

    int it = tsolver.resume(ad_mllib, this->_mlmodel, _main_device,
                            _best_metric_values, best_iteration_numbers,
                            eval_dataset.names());

    bool skip_training = it >= iterations;
    if (skip_training)
      {
        this->_logger->info(
            "Model was trained for {} iterations, skipping training", it);
      }
    else
      {
        this->_logger->info("Training for {} iterations", iterations - it);
      }

    bool resume = it > 0;
    if (!resume)
      this->clear_all_meas_per_iter();

    // [multigpu] initialize all module
    typedef struct
    {
      std::shared_ptr<TorchModule> module;
      std::shared_ptr<TorchLoss> loss;
    } Rank;

    std::vector<Rank> ranks;

    for (size_t i = 0; i < _devices.size(); ++i)
      {
        ranks.emplace_back();
        auto &r = ranks.back();
        if (_devices[i] != _main_device)
          {
            r.module = _module.clone(_devices[i]);
            r.module->train();
            torch::Tensor class_weights_i
                = torch::numel(class_weights) == 0
                      ? class_weights
                      : class_weights.to(_devices[i]);
            r.loss = std::make_shared<TorchLoss>(
                _loss, r.module->has_model_loss(), _seq_training, _timeserie,
                _regression, _classification, _segmentation, _ctc,
                class_weights_i, _reg_weight, *r.module, this->_logger);
          }
      }
    _module.train();

    // create dataloader
    inputc._dataset.reset();
    size_t dataloader_max_jobs = 2 * iter_size * gpu_count;
    this->_logger->info("Init dataloader with {} threads and {} prefetch size",
                        dataloader_threads, dataloader_max_jobs);
    auto dataloader = torch::data::make_data_loader(
        inputc._dataset, data::DataLoaderOptions(batch_size)
                             .workers(dataloader_threads)
                             .max_jobs(dataloader_max_jobs));

    int batch_id = 0;
    double last_it_time = 0;
    double last_test_time = 0;
    double train_loss = 0;
    std::unordered_map<std::string, double> sub_losses;
    double loss_divider = iter_size * gpu_count;
    auto data_it = dataloader->begin();

    if (data_it == dataloader->end())
      {
        throw MLLibBadParamException("No data found in training dataset");
      }
    auto training_start = steady_clock::now();

    double prev_elapsed_time_ms = this->get_meas("elapsed_time_ms");
    if (std::isnan(prev_elapsed_time_ms))
      prev_elapsed_time_ms = 0;

    // `it` is the iteration count (not epoch)
    while (it < iterations)
      {
        if (!this->_tjob_running.load())
          break;

        auto tstart = steady_clock::now();

        // load batches
        std::vector<TorchBatch> batches;
        batches.reserve(_devices.size());

        for (size_t rank = 0; rank < _devices.size(); ++rank)
          {
            batches.push_back(*data_it);
            ++data_it;

            if (data_it == dataloader->end())
              {
                // +1 epoch
                data_it = dataloader->begin();
              }

            if (_masked_lm)
              {
                batches[rank] = inputc.generate_masked_lm_batch(batches[rank]);
              }
          }

        std::exception_ptr eptr;

#pragma omp parallel for num_threads(_devices.size())
        for (size_t rank = 0; rank < _devices.size(); ++rank)
          {
            double loss_val = 0;
            c10::IValue out_val;
            try
              {
                TorchBatch batch = batches[rank];

                torch::Device device = _devices[rank];
                TorchModule &rank_module
                    = device == _main_device ? _module : *ranks[rank].module;
                TorchLoss &rank_tloss
                    = device == _main_device ? tloss : *ranks[rank].loss;

                // Batch preprocessing
                std::vector<c10::IValue> in_vals;
                for (Tensor tensor : batch.data)
                  {
                    in_vals.push_back(tensor.to(device));
                  }
                if (rank_module.has_model_loss())
                  {
                    // if the model computes the loss then we pass target as
                    // input
                    for (Tensor tensor : batch.target)
                      in_vals.push_back(tensor.to(device));
                  }

                if (batch.target.size() == 0)
                  {
                    throw MLLibInternalException(
                        "Batch " + std::to_string(batch_id) + ": no target");
                  }
                std::vector<Tensor> targets;
                for (auto target : batch.target)
                  targets.push_back(target.to(device));

                // Prediction
                out_val = rank_module.forward(in_vals);

                // Compute loss
                Tensor loss = rank_tloss.loss(out_val, targets, in_vals);

                if (loss_divider != 1)
                  loss = loss / loss_divider;

                // Backward
                loss.backward(
                    {},
                    /*retain_graph=*/c10::optional<bool>(retain_graph),
                    /*create_graph=*/false);
                loss_val = loss.item<double>();
              }
            catch (...)
              {
#pragma omp critical
                {
                  eptr = std::current_exception();
                }
                continue;
              }

#pragma omp critical
            {
              // Retain loss and useful values for statistics
              train_loss += loss_val;

              if (out_val.isGenericDict())
                {
                  auto out_dict = out_val.toGenericDict();
                  for (const auto &e : out_dict)
                    {
                      std::string key = e.key().toStringRef();
                      if (!e.value().isTensor())
                        continue;
                      const torch::Tensor &val_t = e.value().toTensor();

                      // all scalar values are considered as metrics
                      if (val_t.numel() != 1)
                        continue;
                      double value = val_t.item<double>();
                      if (loss_divider != 1)
                        value /= loss_divider;
                      if (sub_losses.find(key) != sub_losses.end())
                        sub_losses[key] += value;
                      else
                        sub_losses[key] = value;
                    }
                }
            }
          }

        try
          {
            if (eptr)
              {
                std::rethrow_exception(eptr);
              }
          }
        catch (const std::exception &e)
          {
            this->_logger->error(std::string("Libtorch error: ") + e.what());
            throw MLLibInternalException(std::string("Libtorch error: ")
                                         + e.what());
          }

        // Reduce gradients on device #0
        auto params = _module.parameters();

        for (size_t j = 0; j < _devices.size(); ++j)
          {
            if (_devices[j] == _main_device)
              continue;

            auto params_j = ranks[j].module->parameters();

            for (size_t pi = 0; pi < params.size(); ++pi)
              {
                Tensor &grad = params[pi].mutable_grad();
                Tensor gradj = params_j[pi].grad();
                grad.add_(gradj.to(_main_device));
              }
          }
        // End sync gradients

        // Timing
        auto tstop = steady_clock::now();
        last_it_time += duration_cast<milliseconds>(tstop - tstart).count();

        if ((batch_id + 1) % iter_size == 0)
          {
            if (!this->_tjob_running.load())
              {
                // Interrupt training if the service was deleted
                break;
              }

            tstart = steady_clock::now();

            try
              {
                tsolver.step();
                tsolver.zero_grad();
              }
            catch (std::exception &e)
              {
                this->_logger->error(std::string("Libtorch error: ")
                                     + e.what());
                throw MLLibInternalException(std::string("Libtorch error: ")
                                             + e.what());
              }

            // Broadcast weights to all
            for (size_t j = 0; j < _devices.size(); ++j)
              {
                if (_devices[j] == _main_device)
                  continue;

                auto params_j = ranks[j].module->parameters();

                for (size_t pi = 0; pi < params.size(); ++pi)
                  {
                    torch::NoGradGuard guard;
                    Tensor weight = params[pi];
                    params_j[pi].copy_(weight.to(_devices.at(j)));
                    params_j[pi].grad().zero_();
                  }
              }

            tstop = steady_clock::now();

            double base_lr = tsolver.base_lr();

            this->add_meas("learning_rate", base_lr);
            this->add_meas("iteration", it);
            // without solver time:
            this->add_meas("batch_duration_ms", last_it_time / iter_size);
            // for backward compatibility:
            this->add_meas("iter_time", last_it_time / iter_size);
            // with solver time:
            last_it_time
                += duration_cast<milliseconds>(tstop - tstart).count();
            this->add_meas("iteration_duration_ms", last_it_time);

            double remain_time_ms = last_it_time * (iterations - it);

            if (test_interval > 0)
              {
                int past_tests = (it / test_interval);
                int total_tests = ((iterations - 1) / test_interval) + 1;
                remain_time_ms += last_test_time * (total_tests - past_tests);
              }
            this->add_meas("remain_time", remain_time_ms / 1000.0);
            this->add_meas("train_loss", train_loss);
            int64_t elapsed_time_ms
                = prev_elapsed_time_ms
                  + duration_cast<milliseconds>(tstop - training_start)
                        .count();
            this->add_meas("elapsed_time_ms", elapsed_time_ms);
            this->add_meas_per_iter("elapsed_time_ms", elapsed_time_ms);
            this->add_meas_per_iter("learning_rate", base_lr);
            this->add_meas_per_iter("train_loss", train_loss);
            for (auto e : sub_losses)
              {
                this->add_meas(e.first, e.second);
                this->add_meas_per_iter(e.first, e.second);
              }
            int64_t elapsed_it = it + 1;
            if (log_batch_period != 0 && elapsed_it % log_batch_period == 0)
              {
                this->_logger->info("Iteration {}/{}: loss is {}", elapsed_it,
                                    iterations, train_loss);
                for (auto e : sub_losses)
                  this->_logger->info("\t{}={}", e.first, e.second);
              }
            last_it_time = 0;

            if ((elapsed_it % test_interval == 0 && eval_dataset.size() != 0)
                || elapsed_it == iterations)
              {
                APIData meas_out;
                this->_logger->info("Start test");
                tstart = steady_clock::now();
                tsolver.eval();
                test(ad, inputc, eval_dataset, test_batch_size, meas_out);
                tsolver.train();
                last_test_time
                    = duration_cast<milliseconds>(steady_clock::now() - tstart)
                          .count();

                APIData meas_obj = meas_out.getobj("measure");
                for (const auto &e : sub_losses)
                  meas_obj.add(e.first, e.second);
                meas_out.add("measure", meas_obj);

                save_if_best(meas_out, elapsed_it, tsolver,
                             best_iteration_numbers);

                // print metrics
                for (size_t i = 0; i < eval_dataset.size() + 1; ++i)
                  {
                    if (i == 0)
                      {
                        meas_obj = meas_out.getobj("measure");
                        this->_logger->info("measures over all test sets");
                      }
                    else
                      {
                        size_t test_id = i - 1;
                        meas_obj = meas_out.getv("measures")[test_id];
                        this->_logger->info("measures on test set "
                                            + std::to_string(test_id) + " : "
                                            + eval_dataset.name(test_id));
                      }

                    std::vector<std::string> meas_names = meas_obj.list_keys();
                    for (auto name : meas_names)
                      {
                        std::string metric_name;
                        if (i == 0)
                          metric_name = name;
                        else
                          metric_name = name + "_test" + std::to_string(i - 1);

                        if (name != "cmdiag" && name != "cmfull"
                            && name != "clacc" && name != "cliou"
                            && name != "labels" && name != "test_id"
                            && name != "test_name")
                          {
                            double mval = meas_obj.get(name).get<double>();
                            this->_logger->info("{}={}", metric_name, mval);
                            this->add_meas(metric_name, mval);
                            this->add_meas_per_iter(metric_name, mval);
                          }
                        else if (name == "cmdiag" || name == "clacc"
                                 || name == "cliou")
                          {
                            std::vector<double> mdiag
                                = meas_obj.get(name)
                                      .get<std::vector<double>>();
                            std::vector<std::string> cnames;
                            std::string mdiag_str;
                            for (size_t j = 0; j < mdiag.size(); j++)
                              {
                                mdiag_str
                                    += this->_mlmodel.get_hcorresp(j) + ":"
                                       + std::to_string(mdiag.at(j)) + " ";
                                this->add_meas_per_iter(
                                    metric_name + '_'
                                        + this->_mlmodel.get_hcorresp(j),
                                    mdiag.at(j));
                                cnames.push_back(
                                    this->_mlmodel.get_hcorresp(j));
                              }
                            this->_logger->info("{}=[{}]", metric_name,
                                                mdiag_str);
                            this->add_meas(metric_name, mdiag, cnames);
                          }
                      }
                  }

                out.add("measure", meas_out.getobj("measure"));
                out.add("measures", meas_out.getv("measures"));
              }

            train_loss = 0;
            sub_losses.clear();

            if ((save_period != 0 && elapsed_it % save_period == 0)
                || elapsed_it == iterations)
              {
                bool snapshotted = false;
                for (size_t i = 0; i < best_iteration_numbers.size(); ++i)
                  {

                    if (best_iteration_numbers[i] == elapsed_it)
                      // current model already snapshoted as best model,
                      // do not remove regular snapshot if it is  best
                      // model
                      {
                        best_iteration_numbers[i] = -1;
                        snapshotted = true;
                      }
                  }
                if (!snapshotted)
                  {
                    snapshot(elapsed_it, tsolver);
                  }
              }
            ++it;

            if (it >= iterations)
              break;
          }

        ++batch_id;
      }

    if (!this->_tjob_running.load())
      {
        int64_t elapsed_it = it + 1;
        this->_logger->info("Training job interrupted at iteration {}",
                            elapsed_it);
        bool snapshotted = false;
        for (size_t i = 0; i < best_iteration_numbers.size(); ++i)
          {

            if (best_iteration_numbers[i] == elapsed_it)
              // current model already snapshoted as best model,
              // do not remove regular snapshot if it is  best
              // model
              {
                best_iteration_numbers[i] = -1;
                snapshotted = true;
              }
          }
        if (!snapshotted)
          snapshot(elapsed_it, tsolver);
        torch_utils::free_gpu_memory();
        return -1;
      }

    if (skip_training)
      test(ad, inputc, inputc._test_datasets, test_batch_size, out);
    torch_utils::free_gpu_memory();

    // Update model after training
    this->_mlmodel.read_from_repository(this->_logger);
    this->_mlmodel.read_corresp_file();

    inputc.response_params(out);
    this->_logger->info("Training done.");

    return 0;
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  oatpp::Object<DTO::PredictBody>
  TorchLib<TInputConnectorStrategy, TOutputConnectorStrategy,
           TMLModel>::predict(const APIData &ad_in)
  {
    std::unique_ptr<std::lock_guard<std::mutex>> lock;
    if (!_concurrent_predict)
      {
        // concurrent calls can use more memory on gpu than initially expected
        lock = std::make_unique<std::lock_guard<std::mutex>>(_net_mutex);
        this->_logger->info("Locking torch service for predict");
      }
    oatpp::Object<DTO::ServicePredict> predict_dto;

    // XXX: until everything is DTO, we consider the two cases:
    // - either ad embeds a DTO, so we just have to retrieve it
    // - or it's an APIData that must be converted to DTO
    if (ad_in.has("dto"))
      {
        // cast to ServicePredict...
        auto any = ad_in.get("dto").get<oatpp::Any>();
        predict_dto = oatpp::Object<DTO::ServicePredict>(
            std::static_pointer_cast<typename DTO::ServicePredict>(any->ptr));
      }
    else
      {
        predict_dto = ad_in.createSharedDTO<DTO::ServicePredict>();

        if (ad_in.has("chain") && ad_in.get("chain").get<bool>())
          predict_dto->_chain = true;
        if (ad_in.has("data_raw_img"))
          predict_dto->_data_raw_img
              = ad_in.get("data_raw_img").get<std::vector<cv::Mat>>();
        if (ad_in.has("ids"))
          predict_dto->_ids = ad_in.get("ids").get<std::vector<std::string>>();
        if (ad_in.has("meta_uris"))
          predict_dto->_meta_uris
              = ad_in.get("meta_uris").get<std::vector<std::string>>();
        if (ad_in.has("index_uris"))
          predict_dto->_index_uris
              = ad_in.get("index_uris").get<std::vector<std::string>>();
      }

    auto params = predict_dto->parameters;
    auto input_params = params->input;
    auto output_params = params->output;
    auto mllib_params = params->mllib;

    int64_t predict_batch_size = mllib_params->net->test_batch_size;
    std::string extract_layer = mllib_params->extract_layer;

    bool extract_last = false;
    if (extract_layer == "last")
      extract_last = true;
    std::string forward_method = mllib_params->forward_method;

    std::string dt = mllib_params->datatype;
    if (dt == "fp32")
      _dtype = torch::kFloat32;
    else if (dt == "fp16")
      {
        if (_main_device == torch::Device("cpu"))
          throw MLLibBadParamException(
              "fp16 inference can be done only on GPU");
        _dtype = torch::kFloat16;
      }
    else if (dt == "fp64")
      _dtype = torch::kFloat64;
    else
      throw MLLibBadParamException("unknown datatype " + dt);

    bool bbox = output_params->bbox;
    bool ctc = output_params->ctc;
    double confidence_threshold = output_params->confidence_threshold;

    int best_count = _nclasses;
    if (ctc) // ctc = only one result
      best_count = 1;
    // allow "best: -1" to represent all classes and "best: 0" to still be 0
    else if (output_params->best != nullptr && output_params->best > -1)
      best_count = output_params->best;

    int best_bbox = output_params->best_bbox;

    std::vector<std::string> confidences;
    if (output_params->confidences != nullptr)
      {
        for (oatpp::String conf : *output_params->confidences)
          confidences.push_back(conf);
      }

    bool lstm_continuation = input_params->continuation;
    TInputConnectorStrategy inputc(this->_inputc);

    this->_stats.transform_start();
    TOutputConnectorStrategy outputc(this->_outputc);
    outputc._best = best_count;
    try
      {
        if (std::is_same<TInputConnectorStrategy,
                         ImgTorchInputFileConn>::value)
          {
            inputc.transform(predict_dto);
          }
        else
          {
            // XXX: torchinputconn does not fully support DTOs yet
            inputc.transform(ad_in);
          }
        bool module_was_ready = _module.is_ready(_template);
        _module.post_transform_predict(_template, _template_params, inputc,
                                       this->_mlmodel, _main_device,
                                       predict_dto);
        if (!module_was_ready)
          compute_and_print_model_info();
      }
    catch (...)
      {
        throw;
      }
    this->_stats.transform_end();
    _module.to(_dtype);
    torch::Device cpu("cpu");
    _module.eval();

    if (!extract_last && !extract_layer.empty()
        && !_module.extractable(extract_layer))
      {
        std::string els;
        for (const auto &el : _module.extractable_layers())
          els += el + " ";
        this->_logger->error("Unknown extract layer " + extract_layer
                             + "   candidates are " + els);
        // or we could set back extract_layer to empty string and continue
        throw MLLibBadParamException("unknown extract layer " + extract_layer);
      }

    if (!output_params->measure->empty())
      {
        APIData meas_out;
        test(ad_in, inputc, inputc._dataset, 1, meas_out);
        meas_out.erase("iteration");
        meas_out.erase("train_loss");
        auto out_dto = DTO::PredictBody::createShared();
        out_dto->measure = meas_out;
        torch_utils::free_gpu_memory();
        return out_dto;
      }

    inputc._dataset.reset(false);

    int batch_size = predict_batch_size;
    if (lstm_continuation)
      batch_size = 1;

    auto dataloader = torch::data::make_data_loader(
        std::move(inputc._dataset), data::DataLoaderOptions(batch_size));

    std::vector<APIData> results_ads;
    int nsample = 0;

    for (TorchBatch batch : *dataloader)
      {
        std::vector<c10::IValue> in_vals;
        for (Tensor tensor : batch.data)
          {
            if (tensor.scalar_type() == torch::kFloat32)
              tensor = tensor.to(_dtype);
            in_vals.push_back(tensor.to(_main_device));
          }
        this->_stats.inc_inference_count(batch.data[0].size(0));

        c10::IValue out_ivalue;
        Tensor output;
        try
          {
            if (extract_layer.empty() || extract_last)
              out_ivalue = _module.forward(in_vals, forward_method);
            else
              out_ivalue = _module.extract(in_vals, extract_layer);

            if (!bbox && !_segmentation)
              {
                output = torch_utils::to_tensor_safe(out_ivalue);

                // TODO move this code somewhere else
                if (_template == "gpt2")
                  {
                    // Keep only the prediction for the last token
                    Tensor input_ids = batch.data[0];
                    std::vector<Tensor> outputs;
                    for (int i = 0; i < input_ids.size(0); ++i)
                      {
                        // output is (n_batch * sequence_length * vocab_size)
                        // With gpt2, last token is endoftext so we need to
                        // take the previous output.
                        outputs.push_back(
                            output[i][inputc._lengths.at(i) - 2]);
                      }
                    output = torch::stack(outputs);
                  }

                // XXX: why is (!_timeserie) needed here? Aren't _timeserie and
                // _classification mutually exclusive?
                if (extract_layer.empty() && !_timeserie && _classification)
                  {
                    if (_multi_label)
                      output = torch::sigmoid(output).to(cpu);
                    else
                      output = torch::softmax(output, 1).to(cpu);
                  }
                else if (extract_layer.empty() && !_timeserie && ctc)
                  output = torch::softmax(output, 2).to(cpu);
                else
                  output = output.to(cpu);
              }
          }
        catch (std::exception &e)
          {
            throw MLLibInternalException(std::string("Libtorch error:")
                                         + e.what());
          }

        // Output

        if (!extract_layer.empty())
          {
            for (int j = 0; j < batch_size; j++)
              {
                APIData rad;
                if (!inputc._ids.empty())
                  rad.add("uri", inputc._ids.at(results_ads.size()));
                else
                  rad.add("uri", std::to_string(results_ads.size()));
                if (!inputc._meta_uris.empty())
                  rad.add("meta_uri",
                          inputc._meta_uris.at(results_ads.size()));
                if (!inputc._index_uris.empty())
                  rad.add("index_uri",
                          inputc._index_uris.at(results_ads.size()));
                rad.add("loss", static_cast<double>(0.0));
                torch::Tensor fo = torch::flatten(output)
                                       .contiguous()
                                       .to(torch::kFloat64)
                                       .to(torch::Device("cpu"));

                double *startout = fo.data_ptr<double>();
                std::vector<double> vals(startout,
                                         startout + torch::numel(fo));

                rad.add("vals", vals);
                results_ads.push_back(rad);
              }
          }
        else
          {
            if (_module._native != nullptr)
              output = _module._native->cleanup_output(output);

            if (bbox)
              {
                // Supporting only torchvision output format at the moment.
                if (!out_ivalue.isList())
                  throw MLLibBadParamException(
                      "Could not read output of detection model. Check that "
                      "mllib.template is set correctly.");
                auto out_dicts = out_ivalue.toList();

                for (size_t i = 0; i < out_dicts.size(); ++i)
                  {
                    int img_id = results_ads.size();
                    std::string uri = inputc._ids.at(img_id);
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
                    int src_width
                        = inputc.width() > 0 ? inputc.width() : cols - 1;
                    int src_height
                        = inputc.height() > 0 ? inputc.height() : rows - 1;

                    APIData results_ad;
                    std::vector<double> probs;
                    std::vector<std::string> cats;
                    std::vector<APIData> bboxes;

                    if (!out_dicts.get(i).isGenericDict())
                      throw MLLibBadParamException(
                          "Could not read output of detection model. Check "
                          "that mllib.template is set correctly.");
                    auto out_dict = out_dicts.get(i).toGenericDict();
                    Tensor bboxes_tensor
                        = torch_utils::to_tensor_safe(out_dict.at("boxes"))
                              .to(cpu);
                    Tensor labels_tensor
                        = torch_utils::to_tensor_safe(out_dict.at("labels"))
                              .to(cpu);
                    Tensor score_tensor
                        = torch_utils::to_tensor_safe(out_dict.at("scores"))
                              .to(cpu);

                    auto bboxes_acc = bboxes_tensor.accessor<float, 2>();
                    auto labels_acc = labels_tensor.accessor<int64_t, 1>();
                    auto score_acc = score_tensor.accessor<float, 1>();

                    for (int j = 0; j < labels_tensor.size(0); ++j)
                      {
                        if (best_bbox > 0
                            && bboxes.size() >= static_cast<size_t>(best_bbox))
                          break;

                        double score = score_acc[j];
                        if (score < confidence_threshold)
                          continue;

                        probs.push_back(score);
                        cats.push_back(
                            this->_mlmodel.get_hcorresp(labels_acc[j]));

                        double bbox[] = {
                          bboxes_acc[j][0] / src_width * (cols - 1),
                          bboxes_acc[j][1] / src_height * (rows - 1),
                          bboxes_acc[j][2] / src_width * (cols - 1),
                          bboxes_acc[j][3] / src_height * (rows - 1),
                        };

                        // clamp bbox
                        bbox[0] = std::max(0.0, bbox[0]);
                        bbox[1] = std::max(0.0, bbox[1]);
                        bbox[2]
                            = std::min(static_cast<double>(cols - 1), bbox[2]);
                        bbox[3]
                            = std::min(static_cast<double>(rows - 1), bbox[3]);

                        APIData ad_bbox;
                        ad_bbox.add("xmin", bbox[0]);
                        ad_bbox.add("ymin", bbox[1]);
                        ad_bbox.add("xmax", bbox[2]);
                        ad_bbox.add("ymax", bbox[3]);
                        bboxes.push_back(ad_bbox);
                      }

                    results_ad.add("uri", inputc._uris.at(results_ads.size()));
                    results_ad.add("loss", 0.0);
                    results_ad.add("probs", probs);
                    results_ad.add("cats", cats);
                    results_ad.add("bboxes", bboxes);
                    results_ads.push_back(results_ad);
                  }
              }
            else if (ctc)
              {
                int timestep = output.size(0);
                std::tuple<Tensor, Tensor> sorted_output
                    = output.sort(2, true);
                auto probs_acc
                    = std::get<0>(sorted_output).accessor<float, 3>();
                auto indices_acc
                    = std::get<1>(sorted_output).accessor<int64_t, 3>();

                for (int i = 0; i < output.size(1); ++i)
                  {
                    std::vector<int> pred_label_seq;
                    int prev = output_params->blank_label;
                    float prob = 1;

                    for (int j = 0; j < timestep; ++j)
                      {
                        int cur = indices_acc[j][i][0];
                        if (cur != prev && cur != output_params->blank_label)
                          pred_label_seq.push_back(cur);
                        prev = cur;
                        prob *= probs_acc[j][i][0];
                      }

                    std::ostringstream oss;
                    for (int l : pred_label_seq)
                      oss << char(
                          std::atoi(this->_mlmodel.get_hcorresp(l).c_str()));

                    APIData results_ad;
                    results_ad.add("uri", inputc._uris.at(results_ads.size()));
                    results_ad.add("loss", 0.0);
                    results_ad.add("cats",
                                   std::vector<std::string>{ oss.str() });
                    results_ad.add("probs", std::vector<double>{ prob });
                    results_ad.add("nclasses", (int)_nclasses);
                    results_ads.push_back(results_ad);
                  }
              }
            else if (_classification)
              {
                std::tuple<Tensor, Tensor> sorted_output
                    = output.sort(1, true);
                Tensor probsf = std::get<0>(sorted_output).to(torch::kFloat);
                auto probs_acc = probsf.accessor<float, 2>();
                auto indices_acc
                    = std::get<1>(sorted_output).accessor<int64_t, 2>();

                for (int i = 0; i < output.size(0); ++i)
                  {
                    APIData results_ad;
                    std::vector<double> probs;
                    std::vector<std::string> cats;

                    for (int j = 0; j < best_count; ++j)
                      {
                        if (probs_acc[i][j] < confidence_threshold)
                          {
                            // break because probs are sorted
                            break;
                          }

                        probs.push_back(probs_acc[i][j]);
                        int index = indices_acc[i][j];
                        if (_seq_training)
                          {
                            cats.push_back(inputc.get_word(index));
                          }
                        else
                          {
                            cats.push_back(this->_mlmodel.get_hcorresp(index));
                          }
                      }

                    results_ad.add("uri", inputc._uris.at(results_ads.size()));
                    results_ad.add("loss", 0.0);
                    results_ad.add("cats", cats);
                    results_ad.add("probs", probs);
                    results_ad.add("nclasses", (int)_nclasses);

                    results_ads.push_back(results_ad);
                  }
              }
            else if (_segmentation)
              {
                if (out_ivalue.isGenericDict())
                  {
                    auto out_dict = out_ivalue.toGenericDict();
                    output = torch_utils::to_tensor_safe(out_dict.at("out"));
                  }
                else
                  output = torch_utils::to_tensor_safe(out_ivalue);
                output = torch::softmax(output, 1);

                int imgsize = inputc.width() * inputc.height();
                torch::Tensor segmap;
                torch::Tensor confmap;
                std::tuple<torch::Tensor, torch::Tensor> maxmap;
                if (!confidences.empty()) // applies "best" confidence lookup
                  {
                    maxmap = torch::max(output.clone().squeeze(), 0, false);
                    confmap = torch::flatten(std::get<0>(maxmap))
                                  .contiguous()
                                  .to(torch::kFloat64)
                                  .to(cpu);
                    segmap = torch::flatten(std::get<1>(maxmap))
                                 .contiguous()
                                 .to(torch::kFloat64)
                                 .to(cpu);
                  }
                else
                  segmap = torch::flatten(torch::argmax(output.squeeze(), 0))
                               .contiguous()
                               .to(torch::kFloat64)
                               .to(cpu); // squeeze removes the batch size

                APIData rad;
                std::string uri;
                if (!inputc._ids.empty())
                  uri = inputc._ids.at(results_ads.size());
                else
                  uri = std::to_string(results_ads.size());
                rad.add("uri", uri);
                rad.add("loss", static_cast<double>(0.0));
                double *startout = segmap.data_ptr<double>();
                std::vector<double> vals(startout,
                                         startout + torch::numel(segmap));
                std::vector<double> confs;
                if (!confidences.empty())
                  {
                    startout = confmap.data_ptr<double>();
                    confs = std::vector<double>(
                        startout, startout + torch::numel(confmap));
                  }

                auto bit = inputc._imgs_size.find(uri);
                APIData ad_imgsize;
                ad_imgsize.add("height", (*bit).second.first);
                ad_imgsize.add("width", (*bit).second.second);
                rad.add("imgsize", ad_imgsize);

                if (imgsize
                    != (*bit).second.first
                           * (*bit).second.second) // resizing output
                                                   // segmentation array
                  {
                    vals = ImgInputFileConn::img_resize_vector(
                        vals, inputc.height(), inputc.width(),
                        (*bit).second.first, (*bit).second.second, true);
                    if (!confidences.empty())
                      confs = ImgInputFileConn::img_resize_vector(
                          confs, inputc.height(), inputc.width(),
                          (*bit).second.first, (*bit).second.second, false);
                  }

                rad.add("vals", vals);
                if (!confidences.empty())
                  {
                    APIData vconfs;
                    vconfs.add("best", confs);
                    rad.add("confidences", vconfs);
                  }
                results_ads.push_back(rad);
              }
            else if (_regression)
              {
                auto probs_acc = output.accessor<float, 2>();

                for (int i = 0; i < output.size(0); ++i)
                  {
                    APIData results_ad;
                    std::vector<double> probs;
                    std::vector<std::string> cats;

                    for (size_t j = 0; j < _nclasses; ++j)
                      {
                        probs.push_back(probs_acc[i][j]);
                        cats.push_back(this->_mlmodel.get_hcorresp(j));
                      }

                    results_ad.add("uri", inputc._uris.at(results_ads.size()));
                    results_ad.add("loss", 0.0);
                    results_ad.add("cats", cats);
                    results_ad.add("probs", probs);
                    results_ad.add("nclasses", (int)_nclasses);

                    results_ads.push_back(results_ad);
                  }
              }
            else if (_timeserie)
              {
                output = output.to(cpu);
                auto output_acc = output.accessor<float, 3>();
                for (int j = 0; j < output.size(0); ++j)
                  {
                    std::vector<APIData> series;
                    for (int t = 0; t < output.size(1); ++t)
                      {
                        std::vector<double> preds;
                        for (unsigned int k = 0; k < inputc._ntargets; ++k)
                          {
                            double res = output_acc[j][t][k];
                            preds.push_back(unscale(res, k, inputc));
                          }
                        APIData ts;
                        ts.add("out", preds);
                        series.push_back(ts);
                      }
                    APIData result_ad;
                    if (!inputc._ids.empty())
                      result_ad.add("uri", inputc._ids.at(nsample++));
                    else
                      result_ad.add("uri", std::to_string(nsample++));
                    result_ad.add("series", series);
                    result_ad.add("probs",
                                  std::vector<double>(series.size(), 1.0));
                    result_ad.add("loss", 0.0);
                    results_ads.push_back(result_ad);
                  }
              }
          }
      }

    oatpp::Object<DTO::PredictBody> out_dto;
    OutputConnectorConfig conf;
    if (extract_layer.empty() && !_segmentation)
      {
        outputc.add_results(results_ads);

        if (_timeserie)
          conf._timeseries = true;
        if (_regression)
          conf._regression = true;
        conf._has_bbox = bbox;
        conf._nclasses = static_cast<int>(_nclasses);
        out_dto = outputc.finalize(output_params, conf,
                                   static_cast<MLModel *>(&this->_mlmodel));
      }
    else
      {
        UnsupervisedOutput unsupo;
        unsupo.add_results(results_ads);
        out_dto = unsupo.finalize(output_params, conf,
                                  static_cast<MLModel *>(&this->_mlmodel));
      }

    if (predict_dto->_chain)
      {
        if (std::is_same<TInputConnectorStrategy,
                         ImgTorchInputFileConn>::value)
          {
            // can't do reinterpret_cast because virtual inheritance
            InputConnectorStrategy *inputc_base = &inputc;
            auto *img_ic = dynamic_cast<ImgTorchInputFileConn *>(inputc_base);
            if (!img_ic->_orig_images.empty())
              out_dto->_chain_input._imgs = img_ic->_orig_images;
            else
              out_dto->_chain_input._imgs = img_ic->_images;
            out_dto->_chain_input._img_sizes = img_ic->_images_size;
          }
      }

    // out_dto->status = 0;
    return out_dto;
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  int TorchLib<TInputConnectorStrategy, TOutputConnectorStrategy,
               TMLModel>::test(const APIData &ad,
                               TInputConnectorStrategy &inputc,
                               TorchMultipleDataset &testsets, int batch_size,
                               APIData &out)
  {
    for (size_t i = 0; i < testsets.size(); ++i)
      test(ad, inputc, testsets[i], batch_size, out, i, testsets.name(i));

    SupervisedOutput::aggregate_multiple_testsets(out);
    return 0;
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  int TorchLib<TInputConnectorStrategy, TOutputConnectorStrategy,
               TMLModel>::test(const APIData &ad,
                               TInputConnectorStrategy &inputc,
                               TorchDataset &dataset, int batch_size,
                               APIData &out, size_t test_id,
                               const std::string &test_name)
  {
    APIData ad_res;
    APIData ad_bbox;
    APIData ad_out = ad.getobj("parameters").getobj("output");
    int nclasses = _masked_lm ? inputc.vocab_size() : _nclasses;

    // reset data aug test random generator
    dataset._img_rand_aug_cv.reset_rnd_test_gen();

    // confusion matrix is irrelevant to masked_lm training
    if (_masked_lm && ad_out.has("measure"))
      {
        auto meas = ad_out.get("measure").get<std::vector<std::string>>();
        std::vector<std::string>::iterator it;
        if ((it = std::find(meas.begin(), meas.end(), "cmfull")) != meas.end())
          meas.erase(it);
        if ((it = std::find(meas.begin(), meas.end(), "cmdiag")) != meas.end())
          meas.erase(it);
        ad_out.add("measure", meas);
      }

    std::vector<int> iou_thresholds;
    std::map<int, APIData> ad_bbox_per_iou;
    if (_bbox)
      {
        auto meas = ad_out.get("measure").get<std::vector<std::string>>();
        SupervisedOutput::find_ap_iou_thresholds(meas, iou_thresholds);

        for (int i : iou_thresholds)
          ad_bbox_per_iou[i] = APIData();
      }

    auto dataloader = torch::data::make_data_loader(
        dataset, data::DataLoaderOptions(batch_size));
    torch::Device cpu("cpu");

    _module.eval();
    int entry_id = 0;
    for (TorchBatch batch : *dataloader)
      {
        if (_masked_lm)
          {
            batch = inputc.generate_masked_lm_batch(batch);
          }
        std::vector<c10::IValue> in_vals;
        for (Tensor tensor : batch.data)
          in_vals.push_back(tensor.to(_main_device));

        Tensor output;
        c10::IValue out_ivalue;
        try
          {
            out_ivalue = _module.forward(in_vals);
            if (!_bbox && !_segmentation)
              {
                output = torch_utils::to_tensor_safe(out_ivalue);
              }
          }
        catch (std::exception &e)
          {
            throw MLLibInternalException(std::string("Libtorch error: ")
                                         + e.what());
          }

        if (batch.target.empty())
          throw MLLibBadParamException("Missing label on data while testing");
        Tensor labels;
        if (_timeserie)
          {
            if (_module._native != nullptr)
              output = _module._native->cleanup_output(output);
            // iterate over data in batch
            labels = batch.target[0];
            output = output.to(cpu);
            labels = labels.to(cpu);
            auto output_acc = output.accessor<float, 3>();
            auto target_acc = labels.accessor<float, 3>();

            // tensors are test_batch_size x timesteps x ntargets
            for (int j = 0; j < labels.size(0); ++j)
              {
                std::vector<double> targets;
                std::vector<double> predictions;
                std::vector<double> targets_unscaled;
                std::vector<double> predictions_unscaled;
                for (int t = 0; t < labels.size(1); ++t)
                  for (unsigned int k = 0; k < inputc._ntargets; ++k)
                    {
                      targets.push_back(target_acc[j][t][k]);
                      predictions.push_back(output_acc[j][t][k]);
                      targets_unscaled.push_back(
                          unscale(target_acc[j][t][k], k, inputc));
                      predictions_unscaled.push_back(
                          unscale(output_acc[j][t][k], k, inputc));
                    }
                APIData bad;
                bad.add("target", targets);
                bad.add("pred", predictions);
                bad.add("target_unscaled", targets_unscaled);
                bad.add("pred_unscaled", predictions_unscaled);
                ad_res.add(std::to_string(entry_id), bad);
                ++entry_id;
              }
          }
        else if (_bbox)
          {
            // Supporting only Faster RCNN format at the moment.
            auto out_dicts = out_ivalue.toList();
            Tensor targ_ids = batch.target.at(0);
            auto targ_ids_acc = targ_ids.accessor<int, 1>();
            Tensor targ_bboxes = batch.target.at(1);
            Tensor targ_labels = batch.target.at(2);

            int stop = 0;

            for (size_t i = 0; i < out_dicts.size(); ++i)
              {
                auto out_dict = out_dicts.get(i).toGenericDict();
                Tensor bboxes_tensor
                    = torch_utils::to_tensor_safe(out_dict.at("boxes"))
                          .to(cpu);
                Tensor labels_tensor
                    = torch_utils::to_tensor_safe(out_dict.at("labels"))
                          .to(cpu);
                Tensor score_tensor
                    = torch_utils::to_tensor_safe(out_dict.at("scores"))
                          .to(cpu);

                int start = stop;
                while (stop < static_cast<int>(targ_ids.size(0))
                       && targ_ids_acc[stop] == static_cast<int>(i))
                  {
                    ++stop;
                  }

                for (int iou_thres : iou_thresholds)
                  {
                    double iou_thres_d = static_cast<double>(iou_thres) / 100;
                    std::vector<APIData> vbad = get_bbox_stats(
                        targ_bboxes.index(
                            { torch::indexing::Slice(start, stop) }),
                        targ_labels.index(
                            { torch::indexing::Slice(start, stop) }),
                        bboxes_tensor, labels_tensor, score_tensor,
                        iou_thres_d);
                    ad_bbox_per_iou[iou_thres].add(std::to_string(entry_id),
                                                   vbad);
                  }
                ++entry_id;
              }
          }
        else if (_ctc)
          {
            output = torch::softmax(output, 2).to(cpu);
            std::tuple<Tensor, Tensor> sorted_output = output.sort(2, true);
            auto indices_acc
                = std::get<1>(sorted_output).accessor<int64_t, 3>();
            at::Tensor target = batch.target.at(0).to(cpu);
            at::Tensor target_length = batch.target.at(1).to(cpu);
            int blank_label = 0;
            int timestep = output.size(0);

            for (int i = 0; i < output.size(1); ++i)
              {
                // compute best sequence
                std::vector<int64_t> pred_label_seq;
                int prev = blank_label;

                for (int j = 0; j < timestep; ++j)
                  {
                    int cur = indices_acc[j][i][0];
                    if (cur != prev && cur != blank_label)
                      pred_label_seq.push_back(cur);
                    prev = cur;
                  }

                // compare to target
                torch::Tensor pred_tensor = torch::from_blob(
                    pred_label_seq.data(), pred_label_seq.size(),
                    at::TensorOptions(at::ScalarType::Long));
                torch::Tensor targ_tensor = target.index(
                    { i, torch::indexing::Slice(
                             0, target_length[i].item<int>()) });

                std::vector<double> pred_vec;
                if (torch::equal(pred_tensor, targ_tensor))
                  pred_vec = { 1.0, 0.0 };
                else
                  pred_vec = { 0.0, 1.0 };

                APIData bad;
                bad.add("pred", pred_vec);
                bad.add("target", 0.0);
                ad_res.add(std::to_string(entry_id), bad);
                ++entry_id;
              }
          }
        else if (_segmentation)
          {
            if (out_ivalue.isGenericDict())
              {
                auto out_dict = out_ivalue.toGenericDict();
                output = torch_utils::to_tensor_safe(out_dict.at("out"));
              }
            else
              output = torch_utils::to_tensor_safe(out_ivalue);
            output = torch::softmax(output, 1);
            torch::Tensor target = batch.target.at(0).to(torch::kFloat64);
            torch::Tensor segmap
                = torch::flatten(torch::argmax(output.squeeze(), 1))
                      .contiguous()
                      .to(torch::kFloat64)
                      .to(cpu); // squeeze removes the batch size
            double *startout = segmap.data_ptr<double>();
            double *target_arr = target.data_ptr<double>();
            int tensormap_size = output.size(2) * output.size(3);

            for (int j = 0; j < output.size(0); ++j)
              {
                APIData bad;
                std::vector<double> vals(
                    startout,
                    startout + tensormap_size); // TODO: classes as channels ?
                startout += tensormap_size;
                std::vector<double> targs(target_arr,
                                          target_arr + tensormap_size);
                target_arr += tensormap_size;
                bad.add("target", targs);
                bad.add("pred", vals);
                ad_res.add(std::to_string(entry_id), bad);
                ++entry_id;
              }
          }
        else
          {
            if (_masked_lm)
              {
                // Convert [n_batch, sequence_length, vocab_size] to [n_batch
                // * sequence_length, vocab_size]
                output = output.view(IntList{ -1, output.size(2) });
              }
            if (_classification || _seq_training)
              {
                labels = batch.target[0].view(IntList{ -1 });
                output = torch::softmax(output, 1).to(cpu);
                auto output_acc = output.accessor<float, 2>();
                auto labels_acc = labels.accessor<int64_t, 1>();

                for (int j = 0; j < labels.size(0); ++j)
                  {
                    if (_masked_lm && labels_acc[j] == -1)
                      continue;

                    APIData bad;
                    std::vector<double> predictions;
                    for (int c = 0; c < nclasses; ++c)
                      {
                        predictions.push_back(output_acc[j][c]);
                      }
                    bad.add("target", static_cast<double>(labels_acc[j]));
                    bad.add("pred", predictions);
                    ad_res.add(std::to_string(entry_id), bad);
                    ++entry_id;
                  }
              }
            else if (_regression)
              {
                output = output.to(cpu);
                labels = batch.target[0];
                unsigned int ntargets = nclasses;
                auto output_acc = output.accessor<float, 2>();
                auto labels_acc = labels.accessor<float, 2>();
                for (int j = 0; j < labels.size(0); ++j)
                  {
                    APIData bad;
                    std::vector<double> predictions;
                    if (ntargets == 1)
                      {
                        predictions.push_back(output_acc[j][0]);
                        bad.add("target",
                                static_cast<double>(labels_acc[j][0]));
                      }
                    else
                      {
                        std::vector<double> targets;
                        for (unsigned int t = 0; t < ntargets; ++t)
                          {
                            predictions.push_back(output_acc[j][t]);
                            targets.push_back(labels_acc[j][t]);
                          }
                        bad.add("target", targets);
                      }
                    bad.add("pred", predictions);
                    ad_res.add(std::to_string(entry_id), bad);
                    ++entry_id;
                  }
              }
          }
        // this->_logger->info("Testing: {}/{} entries processed", entry_id,
        // test_size);
      }

    ad_res.add("iteration",
               static_cast<double>(this->get_meas("iteration") + 1));
    ad_res.add("train_loss", this->get_meas("train_loss"));
    if (_timeserie)
      {
        ad_res.add("timeserie", true);
        ad_res.add("timeseries", (int)inputc._ntargets);
      }
    else
      {
        std::vector<std::string> clnames;
        for (int i = 0; i < nclasses; i++)
          clnames.push_back(this->_mlmodel.get_hcorresp(i));
        ad_res.add("clnames", clnames);
        ad_res.add("nclasses", nclasses);
      }
    if (_bbox)
      {
        ad_res.add("bbox", true);
        ad_res.add("pos_count", entry_id);

        for (int iou_thres : iou_thresholds)
          {
            ad_bbox.add("map-" + std::to_string(iou_thres),
                        ad_bbox_per_iou[iou_thres]);
          }
        ad_res.add("0", ad_bbox);
      }
    else if (_segmentation)
      ad_res.add("segmentation", true);
    ad_res.add("batch_size",
               entry_id); // here batch_size = tested entries count
    SupervisedOutput::measure(ad_res, ad_out, out, test_id, test_name);
    _module.train();
    return 0;
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  std::vector<APIData>
  TorchLib<TInputConnectorStrategy, TOutputConnectorStrategy,
           TMLModel>::get_bbox_stats(const at::Tensor &targ_bboxes,
                                     const at::Tensor &targ_labels,
                                     const at::Tensor &bboxes_tensor,
                                     const at::Tensor &labels_tensor,
                                     const at::Tensor &score_tensor,
                                     float overlap_threshold)
  {
    auto targ_bboxes_acc = targ_bboxes.accessor<float, 2>();
    auto targ_labels_acc = targ_labels.accessor<int64_t, 1>();
    auto bboxes_acc = bboxes_tensor.accessor<float, 2>();
    auto labels_acc = labels_tensor.accessor<int64_t, 1>();
    auto score_acc = score_tensor.accessor<float, 1>();

    APIData bad; // see caffe
    const int targ_bbox_count = targ_labels.size(0);
    const int pred_bbox_count = labels_tensor.size(0);
    std::vector<bool> match_used(targ_bbox_count, false);

    struct eval_info
    {
      // every positive score
      std::vector<double> tp_d;
      // 1 if true pos, 0 otherwise
      std::vector<int> tp_i;
      // every positive score
      std::vector<double> fp_d;
      // 1 if false pos, 0 otherwise
      std::vector<int> fp_i;
      int num_pos;
      // XXX: fp variables are useless since tp_d == fp_d & tp_i == !fp_i
      // This could be refactored along with supervised output
      // connector
    };

    std::vector<eval_info> eval_infos(_nclasses);

    for (int j = 0; j < pred_bbox_count; ++j)
      {
        double score = score_acc[j];
        int cls = labels_acc[j];

        if (cls >= static_cast<int>(_nclasses))
          {
            throw MLLibInternalException(
                "Model output class: " + std::to_string(cls)
                + " is invalid for nclasses = " + std::to_string(_nclasses));
          }

        std::vector<double> bbox{
          bboxes_acc[j][0],
          bboxes_acc[j][1],
          bboxes_acc[j][2],
          bboxes_acc[j][3],
        };

        bool has_cls = false;
        float overlap_max = -1.0;
        int bmax = -1;

        for (int k = 0; k < targ_bbox_count; ++k)
          {
            if (targ_labels_acc[k] != cls)
              continue;
            else
              has_cls = true;

            if (!match_used[k])
              {
                std::vector<double> targ_bbox{
                  targ_bboxes_acc[k][0],
                  targ_bboxes_acc[k][1],
                  targ_bboxes_acc[k][2],
                  targ_bboxes_acc[k][3],
                };

                float overlap = bbox_utils::iou(bbox, targ_bbox);
                if (overlap > overlap_max)
                  {
                    overlap_max = overlap;
                    bmax = k;
                  }
              }
          }

        if (has_cls)
          {
            if (bmax != -1 && !match_used[bmax]
                && overlap_max >= overlap_threshold)
              {
                match_used[bmax] = true;
                eval_infos[cls].tp_d.push_back(score);
                eval_infos[cls].tp_i.push_back(1);
                eval_infos[cls].fp_d.push_back(score);
                eval_infos[cls].fp_i.push_back(0);
              }
            else
              {
                eval_infos[cls].tp_d.push_back(score);
                eval_infos[cls].tp_i.push_back(0);
                eval_infos[cls].fp_d.push_back(score);
                eval_infos[cls].fp_i.push_back(1);
              }
          }
      }

    for (int k = 0; k < targ_bbox_count; ++k)
      {
        eval_infos[targ_labels_acc[k]].num_pos++;
      }
    std::vector<APIData> vbad;
    for (size_t label = 1; label < _nclasses; ++label)
      {
        APIData lbad;
        lbad.add("tp_d", eval_infos[label].tp_d);
        lbad.add("tp_i", eval_infos[label].tp_i);
        lbad.add("fp_d", eval_infos[label].fp_d);
        lbad.add("fp_i", eval_infos[label].fp_i);
        lbad.add("num_pos", eval_infos[label].num_pos);
        lbad.add("label", static_cast<int>(label));
        vbad.push_back(lbad);
      }
    return vbad;
  }

  template class TorchLib<ImgTorchInputFileConn, SupervisedOutput, TorchModel>;
  template class TorchLib<VideoTorchInputFileConn, SupervisedOutput,
                          TorchModel>;
  template class TorchLib<TxtTorchInputFileConn, SupervisedOutput, TorchModel>;
  template class TorchLib<CSVTSTorchInputFileConn, SupervisedOutput,
                          TorchModel>;
} // namespace dd
