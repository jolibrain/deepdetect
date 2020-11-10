/**
 * DeepDetect
 * Copyright (c) 2019-2020 Jolibrain
 * Author: Louis Jean <ljean@etud.insa-toulouse.fr>
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

#if !defined(CPU_ONLY)
#include <c10/cuda/CUDACachingAllocator.h>
#endif

#include "outputconnectorstrategy.h"

#include "generators/net_caffe.h"
#include "generators/net_caffe_recurrent.h"

#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>

#include "native/native.h"
#include "torchsolver.h"
#include "torchutils.h"

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
    _best_metric_value = tl._best_metric_value;
    _classification = tl._classification;
    _timeserie = tl._timeserie;
    _loss = tl._loss;
    _template_params = tl._template_params;
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  TorchLib<TInputConnectorStrategy, TOutputConnectorStrategy,
           TMLModel>::~TorchLib()
  {
    _module.free();
    torch_utils::empty_cuda_cache();
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
    if (inputc._min_vals.empty() || inputc._max_vals.empty())
      {
        this->_logger->info("not unscaling output because no bounds "
                            "data found");
        return val;
      }
    else
      {
        if (!inputc._dont_scale_labels)
          {
            double max = inputc._max_vals[inputc._label_pos[k]];
            double min = inputc._min_vals[inputc._label_pos[k]];
            if (inputc._scale_between_minus1_and_1)
              val += 0.5;
            val = val * (max - min) + min;
          }
        return val;
      }
  }

  /*- from mllib -*/
  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  void TorchLib<TInputConnectorStrategy, TOutputConnectorStrategy,
                TMLModel>::init_mllib(const APIData &lib_ad)
  {
    bool gpu = false;
    std::vector<int> gpuids;
    bool freeze_traced = false;
    int embedding_size = 768;
    std::string self_supervised = "";

    if (lib_ad.has("template"))
      _template = lib_ad.get("template").get<std::string>();
    if (lib_ad.has("gpu"))
      gpu = lib_ad.get("gpu").get<bool>();
    if (gpu && !torch::cuda::is_available())
      {
        throw MLLibBadParamException(
            "GPU is not available, service could not be created");
      }
    if (lib_ad.has("gpuid"))
      {
        if (lib_ad.get("gpuid").is<int>())
          gpuids = { lib_ad.get("gpuid").get<int>() };
        else
          gpuids = lib_ad.get("gpuid").get<std::vector<int>>();
      }
    if (lib_ad.has("nclasses"))
      {
        _classification = true;
        _nclasses = lib_ad.get("nclasses").get<int>();
      }
    if (lib_ad.has("self_supervised"))
      self_supervised = lib_ad.get("self_supervised").get<std::string>();
    if (lib_ad.has("embedding_size"))
      embedding_size = lib_ad.get("embedding_size").get<int>();
    if (lib_ad.has("finetuning"))
      _finetuning = lib_ad.get("finetuning").get<bool>();
    if (lib_ad.has("freeze_traced"))
      freeze_traced = lib_ad.get("freeze_traced").get<bool>();
    if (lib_ad.has("loss"))
      _loss = lib_ad.get("loss").get<std::string>();

    if (!gpu)
      {
        _main_device = torch::Device(DeviceType::CPU);
        _devices = { _main_device };
      }
    else
      {
        // if gpuids = -1, we use all gpus
        if (gpuids.size() == 1 && gpuids[0] == -1)
          {
            gpuids.resize(torch::cuda::device_count());
            std::iota(gpuids.begin(), gpuids.end(), 0);
          }

        if (gpuids.empty())
          gpuids.push_back(0);

        for (int gpuid : gpuids)
          _devices.push_back(torch::Device(DeviceType::CUDA, gpuid));
        _main_device = _devices[0];

        if (_devices.size() > 1)
          {
            std::string devices_str = std::to_string(gpuids[0]);
            for (size_t i = 1; i < _devices.size(); ++i)
              devices_str += "," + std::to_string(gpuids[i]);
            this->_logger->info("Running on multiple devices: " + devices_str);
          }
      }
    _module._device = _main_device;
    _module._logger = this->_logger;

    if (_template.find("recurrent") != std::string::npos)
      {
        // call caffe net generator before verything else
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

    bool unsupported_model_configuration
        = this->_mlmodel._traced.empty() && this->_mlmodel._proto.empty()
          && !NativeFactory::valid_template_def(_template);

    if (unsupported_model_configuration)
      throw MLLibInternalException("Use of libtorch backend needs either: "
                                   "traced net, protofile or native template");
    // Create the model
    this->_inputc._input_format = "bert";
    if (_template == "bert")
      {
        if (_classification)
          {
            _module._classif = nn::Linear(embedding_size, _nclasses);
            _module._classif->to(_main_device);
            _module._hidden_states = true;
            _module._classif_in = 1;
          }
        else if (!self_supervised.empty())
          {
            if (self_supervised != "mask")
              {
                throw MLLibBadParamException("self_supervised");
              }
            this->_logger->info("Masked Language model");
            _masked_lm = true;
            _seq_training = true;
          }
        else
          {
            throw MLLibBadParamException(
                "BERT only supports self-supervised or classification");
          }
      }
    else if (_template == "gpt2")
      {
        this->_inputc._input_format = "gpt2";
        _seq_training = true;
      }
    else if (_template.find("recurrent") != std::string::npos
             || NativeFactory::is_timeserie(_template))
      {
        _timeserie = true;
      }
    else if (!_template.empty())
      {
        throw MLLibBadParamException("template");
      }

    if (_classification)
      {
        this->_mltype = "classification";
        _module._nclasses = _nclasses;

        if (_finetuning)
          {
            _module._require_classif_layer = true;
            this->_logger->info(
                "Add classification layer on top of the traced model");
          }
      }

    // Load weights
    _module.load(this->_mlmodel);
    _module.freeze_traced(freeze_traced);

    _best_metrics = { "map", "meaniou",  "mlacc", "delta_score_0.1", "bacc",
                      "f1",  "net_meas", "acc",   "L1_mean_error",   "eucll" };
    _best_metric_value = std::numeric_limits<double>::infinity();

    if (lib_ad.has("template_params"))
      _template_params = lib_ad;
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
      return (v2 > v1);
    return (v1 > v2);
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  int64_t TorchLib<TInputConnectorStrategy, TOutputConnectorStrategy,
                   TMLModel>::save_if_best(APIData &meas_out,
                                           int64_t elapsed_it,
                                           TorchSolver &tsolver,
                                           int64_t best_to_remove)
  {
    double cur_meas = std::numeric_limits<double>::infinity();
    std::string meas;
    for (auto m : _best_metrics)
      {
        if (meas_out.has(m))
          {
            cur_meas = meas_out.get(m).get<double>();
            meas = m;
            break;
          }
      }
    if (cur_meas == std::numeric_limits<double>::infinity())
      {
        // could not find value for measuring best
        this->_logger->info(
            "could not find any value for measuring best model");
        return false;
      }
    if (_best_metric_value == std::numeric_limits<double>::infinity()
        || is_better(cur_meas, _best_metric_value, meas))
      {
        if (best_to_remove != -1)
          {
            remove_model(best_to_remove);
          }
        _best_metric_value = cur_meas;
        this->snapshot(elapsed_it, tsolver);
        try
          {
            std::ofstream bestfile;
            std::string bestfilename
                = this->_mlmodel._repo + this->_mlmodel._best_model_filename;
            bestfile.open(bestfilename, std::ios::out);
            bestfile << "iteration:" << elapsed_it << std::endl;
            bestfile << meas << ":" << cur_meas << std::endl;
            bestfile.close();
          }
        catch (std::exception &e)
          {
            this->_logger->error("could not write best model file");
          }
        return elapsed_it;
      }
    return best_to_remove;
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
                 + std::to_string(elapsed_it) + ".ptw")
                    .c_str());
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  void TorchLib<TInputConnectorStrategy, TOutputConnectorStrategy,
                TMLModel>::snapshot(int64_t elapsed_it, TorchSolver &tsolver)
  {
    this->_logger->info("Saving checkpoint after {} iterations", elapsed_it);
    this->_module.save_checkpoint(this->_mlmodel, std::to_string(elapsed_it));
    // Save optimizer
    tsolver.save(this->_mlmodel._repo + "/solver-" + std::to_string(elapsed_it)
                 + ".pt");
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

    try
      {
        inputc.transform(ad);
        _module.post_transform_train<TInputConnectorStrategy>(
            _template, _template_params, inputc, this->_mlmodel, _main_device);
      }
    catch (...)
      {
        throw;
      }

    APIData ad_mllib = ad.getobj("parameters").getobj("mllib");

    // solver params
    int64_t iterations = 1;
    int64_t batch_size = 1;
    int64_t iter_size = 1;
    int64_t test_batch_size = 1;
    int64_t test_interval = 1;
    int64_t save_period = 0;
    TorchSolver tsolver(this->_logger);

    Tensor class_weights = {};

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

    if (ad_mllib.has("net"))
      {
        APIData ad_net = ad_mllib.getobj("net");
        if (ad_net.has("batch_size"))
          batch_size = ad_net.get("batch_size").get<int>();
        if (ad_net.has("test_batch_size"))
          test_batch_size = ad_net.get("test_batch_size").get<int>();
      }

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

    if (iter_size <= 0)
      iter_size = 1;

    size_t gpu_count = _devices.size();
    if (gpu_count > 1)
      batch_size *= gpu_count;

    // create dataset for evaluation during training
    TorchDataset eval_dataset;
    if (!inputc._test_dataset.empty())
      {
        eval_dataset = inputc._test_dataset; //.split(0, 0.1);
      }

    // create solver
    tsolver.create(_module);

    int it = 0;
    // reload solver and set it value accordingly
    it = tsolver.load(this->_mlmodel._sstate, _main_device);

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

    tsolver.zero_grad();
    _module.train();

    // create dataloader
    inputc._dataset.reset();
    auto dataloader = torch::data::make_data_loader(
        std::move(inputc._dataset), data::DataLoaderOptions(batch_size));

    int batch_id = 0;
    int64_t best_to_remove = -1;

    // it is the iteration count (not epoch)
    while (it < iterations)
      {
        if (!this->_tjob_running.load())
          {
            break;
          }

        double train_loss = 0;
        double avg_it_time = 0;

        for (TorchBatch batch : *dataloader)
          {
            auto tstart = system_clock::now();
            if (_masked_lm)
              {
                batch = inputc.generate_masked_lm_batch(batch);
              }
            std::vector<c10::IValue> in_vals;
            for (Tensor tensor : batch.data)
              {
                in_vals.push_back(tensor.to(_main_device));
              }

            if (batch.target.size() == 0)
              {
                throw MLLibInternalException(
                    "Batch " + std::to_string(batch_id) + ": no target");
              }
            Tensor y = batch.target.at(0).to(_main_device);

            c10::IValue y_pred_ivalue;
            try
              {
                y_pred_ivalue = _module.forward_on_devices(in_vals, _devices);
              }
            catch (std::exception &e)
              {
                this->_logger->error(std::string("Libtorch error: ")
                                     + e.what());
                throw MLLibInternalException(std::string("Libtorch error: ")
                                             + e.what());
              }

            Tensor loss;
            // As CrossEntropy is not available (Libtorch 1.1) we use
            // nllloss
            // + log_softmax
            if (_seq_training)
              {
                // Convert [n_batch, sequence_length, vocab_size] to
                // [n_batch
                // * sequence_length, vocab_size]
                // + ignore non-masked tokens (== -1 in target)
                Tensor y_pred = torch_utils::to_tensor_safe(y_pred_ivalue);
                loss = torch::nll_loss(
                    torch::log_softmax(
                        y_pred.view(IntList{ -1, y_pred.size(2) }), 1),
                    y.view(IntList{ -1 }), class_weights, Reduction::Mean, -1);
              }
            else if (_timeserie)
              {
                if (_module._native != nullptr)
                  loss = _module._native->loss(_loss, in_vals, y_pred_ivalue,
                                               y);
                else
                  {
                    Tensor y_pred = torch_utils::to_tensor_safe(y_pred_ivalue);
                    if (_loss.empty() || _loss == "L1" || _loss == "l1")
                      loss = torch::l1_loss(y_pred, y);
                    else if (_loss == "L2" || _loss == "l2" || _loss == "eucl")
                      loss = torch::mse_loss(y_pred, y);
                    else
                      throw MLLibBadParamException("unknown loss " + _loss);
                  }
              }
            else
              {
                Tensor y_pred = torch_utils::to_tensor_safe(y_pred_ivalue);
                loss = torch::nll_loss(torch::log_softmax(y_pred, 1),
                                       y.view(IntList{ -1 }), class_weights);
              }
            if (iter_size > 1)
              loss /= iter_size;

            double loss_val = loss.item<double>();
            train_loss += loss_val;
            loss.backward();
            auto tstop = system_clock::now();
            avg_it_time += duration_cast<milliseconds>(tstop - tstart).count();

            if ((batch_id + 1) % iter_size == 0)
              {
                if (!this->_tjob_running.load())
                  {
                    break;
                  }
                try
                  {
                    tsolver.step();
                    tsolver.zero_grad();
                  }
                catch (std::exception &e)
                  {
                    this->_logger->error(std::string("Libtorch error: ")
                                         + e.what());
                    throw MLLibInternalException(
                        std::string("Libtorch error: ") + e.what());
                  }

                avg_it_time /= iter_size;
                this->add_meas("learning_rate", tsolver.base_lr());
                this->add_meas("iteration", it);
                this->add_meas("iter_time", avg_it_time);
                this->add_meas("remain_time", avg_it_time * iter_size
                                                  * (iterations - it)
                                                  / 1000.0);
                this->add_meas("train_loss", train_loss);
                this->add_meas_per_iter("learning_rate", tsolver.base_lr());
                this->add_meas_per_iter("train_loss", train_loss);
                int64_t elapsed_it = it + 1;
                if (log_batch_period != 0
                    && elapsed_it % log_batch_period == 0)
                  {
                    this->_logger->info("Iteration {}/{}: loss is {}",
                                        elapsed_it, iterations, train_loss);
                  }
                avg_it_time = 0;
                train_loss = 0;

                if ((elapsed_it % test_interval == 0 && !eval_dataset.empty())
                    || elapsed_it == iterations)
                  {
                    // Free memory
                    loss = torch::empty(1);
                    y_pred_ivalue = torch::empty(1);
                    y = torch::empty(1);
                    in_vals.clear();

                    APIData meas_out;
                    this->_logger->info("Start test");
                    test(ad, inputc, eval_dataset, test_batch_size, meas_out);
                    APIData meas_obj = meas_out.getobj("measure");
                    std::vector<std::string> meas_names = meas_obj.list_keys();

                    best_to_remove = save_if_best(meas_obj, elapsed_it,
                                                  tsolver, best_to_remove);

                    for (auto name : meas_names)
                      {
                        if (name != "cmdiag" && name != "cmfull"
                            && name != "labels")
                          {
                            double mval = meas_obj.get(name).get<double>();
                            this->_logger->info("{}={}", name, mval);
                            this->add_meas(name, mval);
                            this->add_meas_per_iter(name, mval);
                          }
                        else if (name == "cmdiag")
                          {
                            std::vector<double> mdiag
                                = meas_obj.get(name)
                                      .get<std::vector<double>>();
                            std::vector<std::string> cnames;
                            std::string mdiag_str;
                            for (size_t i = 0; i < mdiag.size(); i++)
                              {
                                mdiag_str
                                    += this->_mlmodel.get_hcorresp(i) + ":"
                                       + std::to_string(mdiag.at(i)) + " ";
                                this->add_meas_per_iter(
                                    name + '_'
                                        + this->_mlmodel.get_hcorresp(i),
                                    mdiag.at(i));
                                cnames.push_back(
                                    this->_mlmodel.get_hcorresp(i));
                              }
                            this->_logger->info("{}=[{}]", name, mdiag_str);
                            this->add_meas(name, mdiag, cnames);
                          }
                      }

                    if (elapsed_it == iterations)
                      out = meas_out;
                  }

                if ((save_period != 0 && elapsed_it % save_period == 0)
                    || elapsed_it == iterations)
                  {
                    if (best_to_remove == elapsed_it)
                      // current model already snapshoted as best model,
                      // do not remove regular snapshot if it is  best
                      // model
                      best_to_remove = -1;
                    else
                      snapshot(elapsed_it, tsolver);
                  }
                ++it;

                if (it >= iterations)
                  break;
              }

            ++batch_id;
          }
      }

    if (!this->_tjob_running.load())
      {
        this->_logger->info("Training job interrupted at iteration {}", it);
        torch_utils::empty_cuda_cache();
        return -1;
      }

    if (skip_training)
      test(ad, inputc, inputc._test_dataset, test_batch_size, out);
    torch_utils::empty_cuda_cache();

    // Update model after training
    this->_mlmodel.read_from_repository(this->_logger);
    this->_mlmodel.read_corresp_file();

    inputc.response_params(out);
    this->_logger->info("Training done.");
    return 0;
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  int TorchLib<TInputConnectorStrategy, TOutputConnectorStrategy,
               TMLModel>::predict(const APIData &ad, APIData &out)
  {
    int64_t predict_batch_size = 1;
    APIData params = ad.getobj("parameters");
    APIData output_params = params.getobj("output");
    std::string extract_layer;

    if (params.has("mllib"))
      {
        APIData ad_mllib = ad.getobj("parameters").getobj("mllib");
        if (ad_mllib.has("net"))
          {
            APIData ad_net = ad_mllib.getobj("net");
            if (ad_net.has("test_batch_size"))
              predict_batch_size = ad_net.get("test_batch_size").get<int>();
          }
        if (ad_mllib.has("extract_layer"))
          extract_layer = ad_mllib.get("extract_layer").get<std::string>();
      }

    bool lstm_continuation = false;
    TInputConnectorStrategy inputc(this->_inputc);
    if (_module._native != nullptr)
      _module._native->update_input_connector(inputc);

    this->_stats.transform_start();
    TOutputConnectorStrategy outputc(this->_outputc);
    try
      {
        inputc.transform(ad);
        _module.post_transform_predict(_template, _template_params, inputc,
                                       this->_mlmodel, _main_device, ad);
        if (ad.getobj("parameters").getobj("input").has("continuation")
            && ad.getobj("parameters")
                   .getobj("input")
                   .get("continuation")
                   .get<bool>())
          lstm_continuation = true;
        else
          lstm_continuation = false;
      }
    catch (...)
      {
        throw;
      }
    this->_stats.transform_end();

    torch::Device cpu("cpu");
    _module.eval();

    if (!extract_layer.empty() && !_module.extractable(extract_layer))
      {
        std::string els;
        for (const auto &el : _module.extractable_layers())
          els += el + " ";
        this->_logger->error("Unknown extract layer " + extract_layer
                             + "   candidates are " + els);
        // or we could set back extract_layer to empty string and continue
        throw MLLibBadParamException("unknown extract layer " + extract_layer);
      }

    if (output_params.has("measure"))
      {
        APIData meas_out;
        test(ad, inputc, inputc._dataset, 1, meas_out);
        meas_out.erase("iteration");
        meas_out.erase("train_loss");
        out.add("measure", meas_out.getobj("measure"));
        torch_utils::empty_cuda_cache();
        return 0;
      }

    inputc._dataset.reset(false);

    //		int batch_size = inputc._dataset.cache_size();
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
            in_vals.push_back(tensor.to(_main_device));
          }
        this->_stats.inc_inference_count(batch.data[0].size(0));

        c10::IValue output_ivalue;

        try
          {
            if (extract_layer.empty())
              output_ivalue = _module.forward(in_vals);
            else
              output_ivalue = _module.extract(in_vals, extract_layer);
            if (_timeserie)
              {
                // DO NOTHING
              }
            else
              {
                if (_template == "gpt2")
                  {
                    // Keep only the prediction for the last token
                    Tensor input_ids = batch.data[0];
                    Tensor output = torch_utils::to_tensor_safe(output_ivalue);
                    std::vector<Tensor> outputs;
                    for (int i = 0; i < input_ids.size(0); ++i)
                      {
                        // output is (n_batch * sequence_length * vocab_size)
                        // With gpt2, last token is endoftext so we need to
                        // take the previous output.
                        outputs.push_back(
                            output[i][inputc._lengths.at(i) - 2]);
                      }
                    output_ivalue = torch::stack(outputs);
                  }
                output_ivalue
                    = torch::softmax(
                          torch_utils::to_tensor_safe(output_ivalue), 1)
                          .to(cpu);
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

                std::vector<double> vals;
                _module.to_extracted_vals(output_ivalue, vals);

                rad.add("vals", vals);
                results_ads.push_back(rad);
              }
          }
        else
          {

            if (_module._native != nullptr)
              output_ivalue = _module._native->cleanup_output(output_ivalue);

            Tensor output = torch_utils::to_tensor_safe(output_ivalue);

            if (output_params.has("best"))
              {
                const int best_count = output_params.get("best").get<int>();
                std::tuple<Tensor, Tensor> sorted_output
                    = output.sort(1, true);
                auto probs_acc
                    = std::get<0>(sorted_output).accessor<float, 2>();
                auto indices_acc
                    = std::get<1>(sorted_output).accessor<int64_t, 2>();

                for (int i = 0; i < output.size(0); ++i)
                  {
                    APIData results_ad;
                    std::vector<double> probs;
                    std::vector<std::string> cats;

                    for (int j = 0; j < best_count; ++j)
                      {
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
                        for (unsigned int k = 0; k < this->_inputc._ntargets;
                             ++k)
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

    if (extract_layer.empty())
      {
        outputc.add_results(results_ads);

        if (_timeserie)
          out.add("timeseries", true);
        outputc.finalize(output_params, out,
                         static_cast<MLModel *>(&this->_mlmodel));
      }
    else
      {
        UnsupervisedOutput unsupo;
        unsupo.add_results(results_ads);
        unsupo.finalize(ad.getobj("parameters").getobj("output"), out,
                        static_cast<MLModel *>(&this->_mlmodel));
      }
    out.add("status", 0);
    return 0;
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  int TorchLib<TInputConnectorStrategy, TOutputConnectorStrategy,
               TMLModel>::test(const APIData &ad,
                               TInputConnectorStrategy &inputc,
                               TorchDataset &dataset, int batch_size,
                               APIData &out)
  {
    APIData ad_res;
    APIData ad_out = ad.getobj("parameters").getobj("output");
    int nclasses = _masked_lm ? inputc.vocab_size() : _nclasses;

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

        c10::IValue output_ivalue;
        try
          {
            output_ivalue = _module.forward(in_vals);
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
              output_ivalue = _module._native->cleanup_output(output_ivalue);
            torch::Tensor output = torch_utils::to_tensor_safe(output_ivalue);
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
                for (int t = 0; t < labels.size(1); ++t)
                  for (unsigned int k = 0; k < this->_inputc._ntargets; ++k)
                    {
                      targets.push_back(target_acc[j][t][k]);
                      predictions.push_back(output_acc[j][t][k]);
                    }
                APIData bad;
                bad.add("target", targets);
                bad.add("pred", predictions);
                ad_res.add(std::to_string(entry_id), bad);
                ++entry_id;
              }
          }
        else
          {
            torch::Tensor output = torch_utils::to_tensor_safe(output_ivalue);
            labels = batch.target[0].view(IntList{ -1 });
            if (_masked_lm)
              {
                // Convert [n_batch, sequence_length, vocab_size] to [n_batch
                // * sequence_length, vocab_size]
                output = output.view(IntList{ -1, output.size(2) });
              }
            output = torch::softmax(output, 1).to(cpu);
            auto output_acc = output.accessor<float, 2>();
            auto labels_acc = labels.accessor<int64_t, 1>();

            for (int j = 0; j < labels.size(0); ++j)
              {
                if (_masked_lm && labels_acc[j] == -1)
                  continue;

                APIData bad;
                std::vector<double> predictions;
                for (int c = 0; c < nclasses; c++)
                  {
                    predictions.push_back(output_acc[j][c]);
                  }
                bad.add("target", static_cast<double>(labels_acc[j]));
                bad.add("pred", predictions);
                ad_res.add(std::to_string(entry_id), bad);
                ++entry_id;
              }
          }
        // this->_logger->info("Testing: {}/{} entries processed", entry_id,
        // test_size);
      }

    ad_res.add("iteration", this->get_meas("iteration") + 1);
    ad_res.add("train_loss", this->get_meas("train_loss"));
    if (_timeserie)
      {
        ad_res.add("timeserie", true);
        ad_res.add("timeseries", (int)this->_inputc._ntargets);
      }
    else
      {
        std::vector<std::string> clnames;
        for (int i = 0; i < nclasses; i++)
          clnames.push_back(this->_mlmodel.get_hcorresp(i));
        ad_res.add("clnames", clnames);
        ad_res.add("nclasses", nclasses);
      }
    ad_res.add("batch_size",
               entry_id); // here batch_size = tested entries count
    SupervisedOutput::measure(ad_res, ad_out, out);
    _module.train();
    return 0;
  }

  template class TorchLib<ImgTorchInputFileConn, SupervisedOutput, TorchModel>;
  template class TorchLib<TxtTorchInputFileConn, SupervisedOutput, TorchModel>;
  template class TorchLib<CSVTSTorchInputFileConn, SupervisedOutput,
                          TorchModel>;
} // namespace dd
