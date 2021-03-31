/**
 * DeepDetect
 * Copyright (c) 2019-2020 Jolibrain
 * Author:  Guillaume Infantes <guillaume.infantes@jolibrain.com>
 *          Louis Jean <ljean@etud.insa-toulouse.fr>
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
#include "torchmodule.h"

#include "graph/graph.h"
#include "native/native.h"
#include "torchutils.h"

namespace dd
{
  // ======= TORCH MODULE

  TorchModule::TorchModule() : _device{ "cpu" }
  {
  }

  void TorchModule::to(torch::Device device)
  {
    _device = device;
    if (_graph)
      _graph->to(device);
    if (_native)
      _native->to(device);
    if (_traced)
      _traced->to(device);
    if (_linear)
      _linear->to(device);
  }

  void TorchModule::to(torch::Dtype dtype)
  {
    _dtype = dtype;
    if (_graph)
      _graph->to(dtype);
    if (_native)
      _native->to(dtype);
    if (_traced)
      _traced->to(dtype);
    if (_linear)
      _linear->to(dtype);
  }

  void TorchModule::to(torch::Device device, torch::Dtype dtype)
  {
    _device = device;
    _dtype = dtype;
    if (_graph)
      _graph->to(device, dtype);
    if (_native)
      _native->to(device, dtype);
    if (_traced)
      _traced->to(device, dtype);
    if (_linear)
      _linear->to(device, dtype);
  }

  void TorchModule::proto_model_load(const TorchModel &model)
  {
    _logger->info("loading " + model._proto);
    try
      {
        _graph = std::make_shared<CaffeToTorch>(model._proto);
        _graph->to(_device);
      }
    catch (std::exception &e)
      {
        _logger->info("unable to load " + model._proto);
        throw;
      }
  }

  void TorchModule::graph_model_load(const TorchModel &tmodel)
  {
    if (!tmodel._traced.empty() && _graph->needs_reload())
      {
        _logger->info("loading " + tmodel._traced);
        try
          {
            torch::load(_graph, tmodel._traced, _device);
          }
        catch (std::exception &e)
          {
            _logger->error("unable to load " + tmodel._traced);
            throw;
          }
        std::vector<int> idims = _graph->get_input_dims_from_loaded();
        std::string idimss;
        for (int i : idims)
          idimss += std::to_string(i) + " ";
        _logger->info("input dims of loaded model: " + idimss);
      }
  }

  void TorchModule::native_model_load(const TorchModel &tmodel)
  {
    if (!_native)
      {
        _logger->warn(
            "trying to load weights before allocating native module");
        return;
      }

    if (!tmodel._native.empty())
      {
        _logger->info("loading " + tmodel._native);
        try
          {
            torch_utils::load_weights(*_native, tmodel._native, _device,
                                      _logger);
          }
        catch (std::exception &e)
          {
            _logger->error("unable to load " + tmodel._native);
            throw;
          }
      }
  }

  void TorchModule::linear_model_load(const TorchModel &model)
  {
    _logger->info("loading " + model._weights);
    try
      {
        torch::load(_linear, model._weights, _device);
      }
    catch (std::exception &e)
      {
        _logger->error("unable to load " + model._weights);
        throw;
      }
  }

  void TorchModule::linear_layer_load()
  {
    if (!_linear_layer_file.empty())
      {
        _logger->info("loading " + _linear_layer_file);
        torch::load(_linear, _linear_layer_file, _device);
      }
  }

  void TorchModule::traced_model_load(TorchModel &model)
  {
    _logger->info("loading " + model._traced);
    try
      {
        _traced = std::make_shared<torch::jit::script::Module>(
            torch::jit::load(model._traced, _device));
      }
    catch (std::exception &e)
      {
        _logger->error("unable to load " + model._traced);
        throw;
      }
  }

  template <class TInputConnectorStrategy>
  void TorchModule::create_native_template(
      const std::string &tmpl, const APIData &lib_ad,
      const TInputConnectorStrategy &inputc, const TorchModel &tmodel,
      const torch::Device &device)
  {
    _device = device; // TODO: should be set with set_device elsewhere
    this->_native = std::shared_ptr<NativeModule>(
        NativeFactory::from_template<TInputConnectorStrategy>(
            tmpl, lib_ad, inputc, _logger));

    if (_native)
      {
        _logger->info("created net using template " + tmpl);
        native_model_load(tmodel);
      }
  }

  template <class TInputConnectorStrategy>
  void TorchModule::post_transform(const std::string tmpl,
                                   const APIData &template_params,
                                   const TInputConnectorStrategy &inputc,
                                   const TorchModel &tmodel,
                                   const torch::Device &device)
  {
    if (!_native)
      create_native_template<TInputConnectorStrategy>(tmpl, template_params,
                                                      inputc, tmodel, device);
    if (_graph)
      {
        std::vector<long int> dims = inputc._dataset.datasize(0);
        std::string d;
        for (long int di : dims)
          d += std::to_string(di) + " ";
        _logger->info("input data dimensions : " + d);
        dims.insert(dims.begin(), 1); // dummy batch size
        _graph->finalize(dims);
        if (_graph->needs_reload())
          _logger->info("net was reallocated due to input dim changes");
        // reload params after finalize
        graph_model_load(tmodel);
      }
    to(_device);

    if (_require_linear_layer && !_linear)
      {
        try
          {
            // TODO const cast because getting an input example actually
            // *modifies* the connector (it must be reset after that)
            // -> find a way to get an example without modifying the dataset?
            setup_linear_layer(_nclasses,
                               const_cast<TInputConnectorStrategy &>(inputc)
                                   .get_input_example(device));
            _linear->to(_device);
          }
        catch (std::exception &e)
          {
            throw MLLibInternalException(std::string("Libtorch error: ")
                                         + e.what());
          }
      }
  }

  template <class TInputConnectorStrategy>
  void TorchModule::post_transform_train(const std::string tmpl,
                                         const APIData &template_params,
                                         const TInputConnectorStrategy &inputc,
                                         const TorchModel &tmodel,
                                         const torch::Device &device)
  {
    post_transform(tmpl, template_params, inputc, tmodel, device);
  }

  template <class TInputConnectorStrategy>
  void TorchModule::post_transform_predict(
      const std::string tmpl, const APIData &template_params,
      const TInputConnectorStrategy &inputc, const TorchModel &tmodel,
      const torch::Device &device, const APIData &ad)
  {
    post_transform(tmpl, template_params, inputc, tmodel, device);

    if (_graph)
      {
        if (ad.getobj("parameters").getobj("input").has("continuation")
            && ad.getobj("parameters")
                   .getobj("input")
                   .get("continuation")
                   .get<bool>())
          _graph->lstm_continues(true);
        else
          _graph->lstm_continues(false);
      }
  }

  c10::IValue TorchModule::forward(std::vector<c10::IValue> source,
                                   const std::string &forward_method)
  {
    // graph and native modules take only one tensor as input for now
    if (_graph)
      return _graph->forward(torch_utils::to_tensor_safe(source[0]));
    if (_native)
      return _native->forward(torch_utils::to_tensor_safe(source[0]));
    if (_traced)
      {
        if (!forward_method.empty())
          {
            if (auto method = _traced->find_method(forward_method))
              {
                _logger->info("found forward method {}", method->name());
                auto output = (*method)(std::move(source));
                source = torch_utils::unwrap_c10_vector(output);
              }
            else
              throw MLLibBadParamException("Method " + forward_method
                                           + " not found in traced model");
          }
        else
          {
            auto output = _traced->forward(source);
            source = torch_utils::unwrap_c10_vector(output);
          }
      }
    c10::IValue out_val = source.at(_linear_in);
    if (_hidden_states)
      {
        // out_val is a tuple containing tensors of dimension n_batch *
        // sequence_length * n_features We want a tensor of size n_batch *
        // n_features from the last hidden state
        auto &elems = out_val.toTuple()->elements();
        out_val = elems.back().toTensor().slice(1, 0, 1).squeeze(1);
      }

    if (_linear)
      {
        out_val = _linear->forward(torch_utils::to_tensor_safe(out_val));
      }
    return out_val;
  }

  c10::IValue
  TorchModule::forward_on_devices(std::vector<c10::IValue> source,
                                  const std::vector<torch::Device> &devices)
  {
    // Normal forward if only one device
#if !defined(CPU_ONLY)
    if (devices.size() <= 1)
      {
#endif
        if (!devices.empty())
          to(devices[0]);
        return forward(source);
#if !defined(CPU_ONLY)
      }
#endif

#if !defined(CPU_ONLY)
    // graph and native modules take only one tensor as input for now
    if (_graph)
      {
        throw MLLibBadParamException(
            "Multigpu is not yet supported with graphs.");
      }
    if (_native)
      {
        return torch::nn::parallel::data_parallel(
            _native, torch_utils::to_tensor_safe(source[0]), devices, _device);
      }
    if (_traced)
      {
        throw MLLibBadParamException(
            "Multigpu is not yet supported with traced models");
      }
    c10::IValue out_val = source.at(_linear_in);
    if (_hidden_states)
      {
        // out_val is a tuple containing tensors of dimension n_batch *
        // sequence_length * n_features We want a tensor of size n_batch *
        // n_features from the last hidden state
        auto &elems = out_val.toTuple()->elements();
        out_val = elems.back().toTensor().slice(1, 0, 1).squeeze(1);
      }
    if (_linear)
      {
        out_val = torch::nn::parallel::data_parallel(
            _linear, torch_utils::to_tensor_safe(out_val), devices, _device);
      }
    return out_val;
#endif
  }

  c10::IValue TorchModule::extract(std::vector<c10::IValue> source,
                                   std::string extract_layer)
  {
    if (_graph) // native modules take only one tensor as input for now
      return _graph->extract(torch_utils::to_tensor_safe(source[0]),
                             extract_layer);
    if (_native)
      return _native->extract(torch_utils::to_tensor_safe(source[0]),
                              extract_layer);
    auto output = _traced->forward(source);
    source = torch_utils::unwrap_c10_vector(output);

    c10::IValue out_val = source.at(_linear_in);

    if (_hidden_states)
      {
        // out_val is a tuple containing tensors of dimension n_batch *
        // sequence_length * n_features We want a tensor of size n_batch *
        // n_features from the last hidden state
        auto &elems = out_val.toTuple()->elements();
        out_val = elems.back().toTensor().slice(1, 0, 1).squeeze(1);
      }
    return out_val;
  }

  bool TorchModule::extractable(std::string extract_layer) const
  {
    if (_graph)
      return _graph->extractable(extract_layer);
    if (_native)
      return _native->extractable(extract_layer);
    if (_traced)
      return extract_layer == "final";
    return false;
  }

  std::vector<std::string> TorchModule::extractable_layers() const
  {
    if (_graph)
      return _graph->extractable_layers();
    if (_native)
      return _native->extractable_layers();
    if (_traced)
      {
        std::vector<std::string> ret;
        ret.push_back("final");
        return ret;
      }
    return std::vector<std::string>();
  }

  void TorchModule::freeze_traced(bool freeze)
  {
    if (freeze != _freeze_traced)
      {
        _freeze_traced = freeze;
        std::vector<torch::Tensor> params;
        torch_utils::add_parameters(_traced, params, false);
        for (auto &param : params)
          {
            param.set_requires_grad(!freeze);
          }
      }
  }

  void TorchModule::setup_linear_layer(int nclasses,
                                       std::vector<c10::IValue> input_example)
  {
    _linear = nullptr;
    // First dimension is batch id
    int outdim
        = torch_utils::to_tensor_safe(forward(input_example)).sizes()[1];
    _linear = torch::nn::Linear(outdim, nclasses);
    linear_layer_load();
  }

  std::vector<torch::Tensor> TorchModule::parameters()
  {
    if (_graph)
      return _graph->parameters();
    if (_native)
      return _native->parameters();
    std::vector<torch::Tensor> params;
    if (_traced)
      torch_utils::add_parameters(_traced, params);
    if (_linear)
      {
        auto linear_params = _linear->parameters();
        params.insert(params.end(), linear_params.begin(),
                      linear_params.end());
      }
    return params;
  }

  void TorchModule::save_checkpoint(TorchModel &model, const std::string &name)
  {
    if (_traced)
      _traced->save(model._repo + "/checkpoint-" + name + ".pt");
    if (_linear)
      torch::save(_linear, model._repo + "/checkpoint-" + name + ".ptw");
    if (_graph)
      torch::save(_graph, model._repo + "/checkpoint-" + name + ".pt");
    if (_native)
      torch::save(_native, model._repo + "/checkpoint-" + name + ".npt");
  }

  void TorchModule::load(TorchModel &model)
  {
    if (!model._native.empty() && !model._proto.empty())
      {
        throw MLLibBadParamException(
            "Found both native and graph model in repository");
      }

    if (!model._traced.empty() && model._proto.empty())
      traced_model_load(model);

    if (!model._weights.empty())
      {
        if (_linear)
          {
            linear_model_load(model);
          }
        else if (_require_linear_layer)
          {
            _linear_layer_file = model._weights;
          }
      }
    if (!model._proto.empty())
      {
        proto_model_load(model);
        graph_model_load(model);
      }

    if (!model._native.empty())
      native_model_load(model);
  }

  void TorchModule::eval()
  {
    if (_graph)
      _graph->eval();
    if (_traced)
      _traced->eval();
    if (_linear)
      _linear->eval();
    if (_native)
      _native->eval();
  }

  void TorchModule::train()
  {
    if (_graph)
      _graph->train();
    if (_traced)
      _traced->train();
    if (_linear)
      _linear->train();
    if (_native)
      _native->train();
  }

  void TorchModule::free()
  {
    _graph = nullptr;
    _traced = nullptr;
    _linear = nullptr;
    _native = nullptr;
  }

  template void TorchModule::post_transform(
      const std::string tmpl, const APIData &template_params,
      const ImgTorchInputFileConn &inputc, const TorchModel &tmodel,
      const torch::Device &device);

  template void TorchModule::post_transform_train(
      const std::string tmpl, const APIData &template_params,
      const ImgTorchInputFileConn &inputc, const TorchModel &tmodel,
      const torch::Device &device);

  template void TorchModule::post_transform_predict(
      const std::string tmpl, const APIData &template_params,
      const ImgTorchInputFileConn &inputc, const TorchModel &tmodel,
      const torch::Device &device, const APIData &ad);

  template void TorchModule::post_transform(
      const std::string tmpl, const APIData &template_params,
      const TxtTorchInputFileConn &inputc, const TorchModel &tmodel,
      const torch::Device &device);

  template void TorchModule::post_transform_train(
      const std::string tmpl, const APIData &template_params,
      const TxtTorchInputFileConn &inputc, const TorchModel &tmodel,
      const torch::Device &device);

  template void TorchModule::post_transform_predict(
      const std::string tmpl, const APIData &template_params,
      const TxtTorchInputFileConn &inputc, const TorchModel &tmodel,
      const torch::Device &device, const APIData &ad);

  template void TorchModule::post_transform(
      const std::string tmpl, const APIData &template_params,
      const CSVTSTorchInputFileConn &inputc, const TorchModel &tmodel,
      const torch::Device &device);

  template void TorchModule::post_transform_train(
      const std::string tmpl, const APIData &template_params,
      const CSVTSTorchInputFileConn &inputc, const TorchModel &tmodel,
      const torch::Device &device);

  template void TorchModule::post_transform_predict(
      const std::string tmpl, const APIData &template_params,
      const CSVTSTorchInputFileConn &inputc, const TorchModel &tmodel,
      const torch::Device &device, const APIData &ad);
}
