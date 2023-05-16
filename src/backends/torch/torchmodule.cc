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
    to(device, _dtype);
  }

  void TorchModule::to(torch::Dtype dtype)
  {
    to(_device, dtype);
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
    if (_linear_head)
      _linear_head->to(device, dtype);
    if (_crnn_head)
      _crnn_head->to(device, dtype);
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
            // finetuning relaxes strictness when loading weights
            torch_utils::load_weights(*_native, tmodel._native, _device,
                                      _logger, !_finetuning);
          }
        catch (std::exception &e)
          {
            _logger->error("unable to load " + tmodel._native);
            throw;
          }
      }
  }

  void TorchModule::crnn_head_load(const TorchModel &model)
  {
    _logger->info("loading " + model._head_weights);
    try
      {
        torch::load(_crnn_head, model._head_weights, _device);
      }
    catch (std::exception &e)
      {
        _logger->error("unable to load " + model._head_weights);
        throw;
      }
  }

  void TorchModule::linear_head_load(const TorchModel &model)
  {
    _logger->info("loading " + model._head_weights);
    try
      {
        torch::load(_linear_head, model._head_weights, _device);
      }
    catch (std::exception &e)
      {
        _logger->error("unable to load " + model._head_weights);
        throw;
      }
  }

  void TorchModule::crnn_head_load()
  {
    if (!_head_weights.empty())
      {
        _logger->info("loading " + _head_weights);
        torch::load(_crnn_head, _head_weights, _device);
      }
  }

  void TorchModule::linear_head_load()
  {
    if (!_head_weights.empty())
      {
        _logger->info("loading " + _head_weights);
        torch::load(_linear_head, _head_weights, _device);
      }
  }

  void TorchModule::traced_model_load(TorchModel &model)
  {
    _logger->info("loading " + model._traced);
    try
      {
#ifdef USE_MPS
        // XXX(louisj): this is a workaround for torch v2.0.1, subsequent
        // versions should have this fixed
        if (_device.type() == torch::DeviceType::MPS)
          {
            std::cout << "LOADING en MPS" << std::endl;
            _traced = std::make_shared<torch::jit::script::Module>(
                torch::jit::load(model._traced,
                                 torch::Device(torch::DeviceType::CPU)));
            _traced->to(torch::Device(torch::DeviceType::MPS));
          }
        else
#endif
          {
            _traced = std::make_shared<torch::jit::script::Module>(
                torch::jit::load(model._traced, _device));
          }
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
      {
        create_native_template<TInputConnectorStrategy>(
            tmpl, template_params, inputc, tmodel, device);
      }
    if (_graph)
      {
        std::vector<int64_t> dims = inputc._dataset.datasize(0);
        std::string d;
        for (int64_t di : dims)
          d += std::to_string(di) + " ";
        _logger->info("input data dimensions : " + d);
        dims.insert(dims.begin(), 1); // dummy batch size
        _graph->finalize(dims);
        if (_graph->needs_reload())
          {
            _logger->info("net was reallocated due to input dim changes");
          }
        // reload params after finalize
        graph_model_load(tmodel);
      }

    if (_require_linear_head && !_linear_head)
      {
        try
          {
            // TODO const cast because getting an input example actually
            // *modifies* the connector (it must be reset after that)
            // -> find a way to get an example without modifying the dataset?
            setup_linear_head(_nclasses,
                              const_cast<TInputConnectorStrategy &>(inputc)
                                  .get_input_example(device));
          }
        catch (std::exception &e)
          {
            throw MLLibInternalException(std::string("Libtorch error: ")
                                         + e.what());
          }
      }
    if (_require_crnn_head && !_crnn_head)
      {
        try
          {
            setup_crnn_head(template_params,
                            const_cast<TInputConnectorStrategy &>(inputc)
                                .get_input_example(device),
                            inputc._alphabet_size);
          }
        catch (std::exception &e)
          {
            throw MLLibInternalException(std::string("Libtorch error: ")
                                         + e.what());
          }
      }

    to(device);
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
      const torch::Device &device,
      const oatpp::Object<DTO::ServicePredict> &pred_dto)
  {
    post_transform(tmpl, template_params, inputc, tmodel, device);

    if (_graph)
      _graph->lstm_continues(pred_dto->parameters->input->continuation);
  }

  bool TorchModule::is_ready(const std::string &tmplate) const
  {
    if (!_native && NativeFactory::valid_template_def(tmplate))
      return false;
    if (_graph && _graph->needs_reload())
      return false;
    if (_require_linear_head && !_linear_head)
      return false;
    if (_require_crnn_head && !_crnn_head)
      return false;

    return true;
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

    if (_training && _loss_id >= 0)
      {
        // if we are in training mode and model does output the loss (eg
        // traced detection models), then we return the loss.
        return source.at(_loss_id);
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

    // graph and native modules take only one tensor as input for now
    if (_linear_head)
      {
        out_val = _linear_head->forward(torch_utils::to_tensor_safe(out_val));
      }
    else if (_crnn_head)
      {
        out_val = _crnn_head->forward(torch_utils::to_tensor_safe(out_val));
      }
    return out_val;
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

  bool TorchModule::has_model_loss() const
  {
    return _loss_id >= 0;
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

  void TorchModule::setup_linear_head(int nclasses,
                                      std::vector<c10::IValue> input_example)
  {
    _linear_head = nullptr;
    // First dimension is batch id
    int outdim
        = torch_utils::to_tensor_safe(forward(input_example)).sizes()[1];
    _linear_head = torch::nn::Linear(outdim, nclasses);
    linear_head_load();
  }

  void TorchModule::setup_crnn_head(const APIData &template_params,
                                    std::vector<c10::IValue> input_example,
                                    int output_size)
  {
    _crnn_head = nullptr;

    if (!_traced)
      throw MLLibInternalException("No traced model");

    auto outdims = torch_utils::to_tensor_safe(forward(input_example)).sizes();
    std::stringstream ss;
    ss << "Backbone output dimensions = " << outdims;
    _logger->info(ss.str());
    _crnn_head = CRNNHead(template_params, outdims.vec(), output_size);
    crnn_head_load();
  }

  std::vector<torch::Tensor> TorchModule::parameters()
  {
    std::vector<torch::Tensor> params;
    if (_graph)
      return _graph->parameters();
    else if (_native)
      return _native->parameters();

    if (_traced)
      torch_utils::add_parameters(_traced, params);
    if (_linear_head)
      {
        auto linear_params = _linear_head->parameters();
        params.insert(params.end(), linear_params.begin(),
                      linear_params.end());
      }
    else if (_crnn_head)
      {
        auto crnn_params = _crnn_head->parameters();
        params.insert(params.end(), crnn_params.begin(), crnn_params.end());
      }
    return params;
  }

  void TorchModule::save_checkpoint(TorchModel &model, const std::string &name)
  {
    if (_traced)
      _traced->save(model._repo + "/checkpoint-" + name + ".pt");
    if (_linear_head)
      torch::save(_linear_head, model._repo + "/checkpoint-" + name + ".ptw");
    if (_crnn_head)
      torch::save(_crnn_head, model._repo + "/checkpoint-" + name + ".ptw");
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

    if (!model._head_weights.empty())
      {
        _head_weights = model._head_weights;
        if (_linear_head)
          {
            linear_head_load(model);
          }
        else if (_crnn_head)
          {
            crnn_head_load(model);
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
    if (_linear_head)
      _linear_head->eval();
    if (_crnn_head)
      _crnn_head->eval();
    if (_native)
      _native->eval();

    _training = false;
  }

  void TorchModule::train()
  {
    if (_graph)
      _graph->train();
    if (_traced)
      _traced->train();
    if (_linear_head)
      _linear_head->train();
    if (_crnn_head)
      _crnn_head->train();
    if (_native)
      _native->train();

    _training = true;
  }

  void TorchModule::free()
  {
    _graph = nullptr;
    _traced = nullptr;
    _linear_head = nullptr;
    _crnn_head = nullptr;
    _native = nullptr;
  }

  std::shared_ptr<TorchModule> TorchModule::clone(torch::Device device)
  {
    auto cloned = std::make_shared<TorchModule>(*this);

    if (_native)
      {
        cloned->_native
            = std::dynamic_pointer_cast<NativeModule>(_native->clone(device));
      }
    if (_traced)
      {
        cloned->_traced
            = std::make_shared<torch::jit::script::Module>(_traced->clone());
        cloned->_traced->to(device);
        for (auto param : cloned->_traced->parameters())
          {
            param.detach_().requires_grad_();
          }
      }
    if (_linear_head)
      {
        cloned->_linear_head
            = std::dynamic_pointer_cast<torch::nn::LinearImpl>(
                _linear_head->clone(device));
      }
    if (_crnn_head)
      {
        cloned->_crnn_head = std::dynamic_pointer_cast<CRNNHeadImpl>(
            _crnn_head->clone(device));
      }
    if (_graph)
      {
        throw MLLibBadParamException("MultiGPU is not supported on non "
                                     "cloneable models (graph models)");
      }
    cloned->to(device);
    return cloned;
  }

  std::string long_number_to_str(int64_t number)
  {
    std::string number_str = std::to_string(number);
    std::reverse(number_str.begin(), number_str.end());
    std::stringstream ss;
    for (size_t i = 0; i < number_str.size(); i++)
      {
        if (i != 0 && i % 3 == 0)
          ss << ",";
        ss << number_str[i];
      }
    number_str = ss.str();
    std::reverse(number_str.begin(), number_str.end());
    return number_str;
  }

  void print_native_params(std::shared_ptr<spdlog::logger> logger,
                           const std::string &name,
                           const torch::nn::Module &module,
                           int64_t &param_count, int64_t &frozen_count)
  {
    logger->info("## {} parameters", name);
    param_count = 0;
    frozen_count = 0;
    for (const auto &p : module.named_parameters())
      {
        std::stringstream sstream;
        sstream << "name=" << p.key() << ", size=" << p.value().sizes()
                << ", requires_grad="
                << (p.value().requires_grad() ? "true" : "false");
        logger->info(sstream.str());

        // Count parameters
        int count = 1;
        for (int s : p.value().sizes())
          {
            count *= s;
          }
        param_count += count;
        if (!p.value().requires_grad())
          frozen_count += count;
      }
    logger->info("{} parameters count: {}", name,
                 long_number_to_str(param_count));
    if (frozen_count != 0)
      {
        logger->info("\tfrozen = {}", long_number_to_str(frozen_count));
      }
  }

  void TorchModule::compute_and_print_model_info()
  {
    int64_t total_param_count = 0;
    int64_t total_frozen_count = 0;
    if (_graph)
      {
        int64_t graph_param_count, graph_frozen_count;
        print_native_params(_logger, "Graph", *_graph, graph_param_count,
                            graph_frozen_count);
        total_param_count += graph_param_count;
        total_frozen_count += graph_frozen_count;
      }
    if (_native)
      {
        int64_t native_param_count, native_frozen_count;
        print_native_params(_logger, "Native", *_native, native_param_count,
                            native_frozen_count);
        total_param_count += native_param_count;
        total_frozen_count += native_frozen_count;
      }
    if (_traced)
      {
        _logger->info("## Traced parameters");
        int64_t traced_param_count = 0;
        int64_t traced_frozen_count = 0;
        for (const auto &p : _traced->named_parameters())
          {
            std::stringstream sstream;
            sstream << "name=" << p.name << ", size=" << p.value.sizes()
                    << ", requires_grad="
                    << (p.value.requires_grad() ? "true" : "false");
            _logger->info(sstream.str());

            // Count parameters
            int count = 1;
            for (int s : p.value.sizes())
              {
                count *= s;
              }
            traced_param_count += count;
            if (!p.value.requires_grad())
              traced_frozen_count += count;
          }
        _logger->info("Traced parameters count: {}",
                      long_number_to_str(traced_param_count));
        if (traced_frozen_count != 0)
          {
            _logger->info("\tfrozen = {}",
                          long_number_to_str(traced_frozen_count));
          }
        total_param_count += traced_param_count;
        total_frozen_count += traced_frozen_count;
      }
    if (_linear_head)
      {
        int64_t linear_param_count, linear_frozen_count;
        print_native_params(_logger, "Linear", *_linear_head,
                            linear_param_count, linear_frozen_count);
        total_param_count += linear_param_count;
        total_frozen_count += linear_frozen_count;
      }
    if (_crnn_head)
      {
        int64_t crnn_param_count, crnn_frozen_count;
        print_native_params(_logger, "CRNN", *_crnn_head, crnn_param_count,
                            crnn_frozen_count);
        total_param_count += crnn_param_count;
        total_frozen_count += crnn_frozen_count;
      }
    _logger->info("## Total number of parameters: {}",
                  long_number_to_str(total_param_count));
    _params_count = total_param_count;
    _frozen_params_count = total_frozen_count;
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
      const torch::Device &device,
      const oatpp::Object<DTO::ServicePredict> &pred_dto);

  template void TorchModule::create_native_template(
      const std::string &tmpl, const APIData &lib_ad,
      const ImgTorchInputFileConn &inputc, const TorchModel &tmodel,
      const torch::Device &device);

  template void TorchModule::post_transform(
      const std::string tmpl, const APIData &template_params,
      const VideoTorchInputFileConn &inputc, const TorchModel &tmodel,
      const torch::Device &device);

  template void TorchModule::post_transform_train(
      const std::string tmpl, const APIData &template_params,
      const VideoTorchInputFileConn &inputc, const TorchModel &tmodel,
      const torch::Device &device);

  template void TorchModule::post_transform_predict(
      const std::string tmpl, const APIData &template_params,
      const VideoTorchInputFileConn &inputc, const TorchModel &tmodel,
      const torch::Device &device,
      const oatpp::Object<DTO::ServicePredict> &pred_dto);

  template void TorchModule::create_native_template(
      const std::string &tmpl, const APIData &lib_ad,
      const VideoTorchInputFileConn &inputc, const TorchModel &tmodel,
      const torch::Device &device);

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
      const torch::Device &device,
      const oatpp::Object<DTO::ServicePredict> &pred_dto);

  template void TorchModule::create_native_template(
      const std::string &tmpl, const APIData &lib_ad,
      const TxtTorchInputFileConn &inputc, const TorchModel &tmodel,
      const torch::Device &device);

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
      const torch::Device &device,
      const oatpp::Object<DTO::ServicePredict> &pred_dto);

  template void TorchModule::create_native_template(
      const std::string &tmpl, const APIData &lib_ad,
      const CSVTSTorchInputFileConn &inputc, const TorchModel &tmodel,
      const torch::Device &device);
}
