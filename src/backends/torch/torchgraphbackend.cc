/**
 * DeepDetect
 * Copyright (c) 2020 Jolibrain
 * Author: Guillaume Infantes <guillaume.infantes@jolibrain.com>
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

#include "torchgraphbackend.h"
#include "mllibstrategy.h"

namespace dd
{
  using torch::Tensor;
  using torch::nn::AnyModule;
  using torch::nn::Linear;
  using torch::nn::LinearOptions;
  using torch::nn::LSTM;
  using torch::nn::LSTMOptions;
  using torch::nn::PReLU;
  using torch::nn::RNN;
  using torch::nn::RNNOptions;

  void TorchGraphBackend::set_input(torch::Tensor input)
  {
    std::vector<int> dim(input.sizes().begin(), input.sizes().end());
    this->set_input_dim(dim);
    _variables[_inputname] = input;
    BaseGraph::finalize();
    allocate_modules();
  }

  void TorchGraphBackend::finalize()
  {
    BaseGraph::finalize();
    allocate_modules();
  }

  void TorchGraphBackend::finalize(at::IntArrayRef dim)
  {
    finalize(dim.vec());
  }

  void TorchGraphBackend::finalize(std::vector<int> dim)
  {
    this->set_input_dim(dim);
    BaseGraph::finalize();
    allocate_modules();
  }

  void TorchGraphBackend::finalize(std::vector<int64_t> dim)
  {
    std::vector<int> dimint;
    for (auto d : dim)
      dimint.push_back(d);
    finalize(dimint);
  }

  bool TorchGraphBackend::extractable(std::string extract_layer)
  {
    for (BaseGraph::Vertex v : _sortedVars)
      if (varname(v) == extract_layer)
        return true;
    return false;
  }

  std::vector<std::string> TorchGraphBackend::extractable_layers() const
  {
    std::vector<std::string> allvars;
    for (BaseGraph::Vertex v : _sortedVars)
      allvars.push_back(varname(v));
    return allvars;
  }

  torch::Tensor TorchGraphBackend::extract(torch::Tensor inputTensor,
                                           std::string extract_layer)
  {
    set_input(inputTensor);
    for (BaseGraph::Vertex o : _sortedOps)
      {
        std::vector<torch::Tensor> out = forward(o);
        std::vector<BaseGraph::Vertex> outputVars = this->outputs(o);
        for (unsigned int i = 0; i < outputVars.size(); ++i)
          {
            _variables[varname(outputVars[i])] = out[i];
            if (varname(outputVars[i]) == extract_layer)
              return out[i];
          }
      }
    throw TorchGraphException(
        "could not extract layer, extract layer could not be computed from "
        "input please check graph structure");
  }

  torch::Tensor TorchGraphBackend::forward(torch::Tensor inputTensor)
  {
    set_input(inputTensor);
    bool output_computed = false;
    for (BaseGraph::Vertex o : _sortedOps)
      {
        std::vector<torch::Tensor> out = forward(o);
        std::vector<BaseGraph::Vertex> outputVars = this->outputs(o);
        for (unsigned int i = 0; i < outputVars.size(); ++i)
          {
            if (varname(outputVars[i]) == _outputname)
              output_computed = true;
            _variables[varname(outputVars[i])] = out[i];
          }
      }
    if (!output_computed)
      throw TorchGraphException(
          "did not compute output, please check NN graph");
    return _variables[_outputname];
  }

  std::vector<torch::Tensor> TorchGraphBackend::forward(BaseGraph::Vertex v)
  {

    std::vector<torch::Tensor> inputsTensor;
    for (BaseGraph::Vertex vi : this->inputs(v))
      {
        inputsTensor.push_back(_variables[varname(vi)]);
      }

    auto opname_v = opname(v);
    std::vector<torch::Tensor> output;
    std::string optype = this->optype(v);

    if (optype == "RNN")
      {
        throw MLLibInternalException(
            "RNN layer type not supported, use LSTM instead");
      }
    else if (optype == "LSTM")
      {
        std::tuple<torch::Tensor, std::tuple<torch::Tensor, torch::Tensor>>
            full_output;
        if (_lstm_continuation && _rnn_has_memories[opname_v])
          {
            full_output
                = _modules[opname_v]
                      .forward<std::tuple<Tensor, std::tuple<Tensor, Tensor>>>(
                          inputsTensor[0],
                          torch::optional<
                              std::tuple<torch::Tensor, torch::Tensor>>(
                              _rnn_memories[opname_v]));
          }
        else
          full_output
              = _modules[opname_v]
                    .forward<std::tuple<Tensor, std::tuple<Tensor, Tensor>>>(
                        inputsTensor[0]);
        _autoencoder_timesteps = std::get<0>(full_output).size(1);
        output.push_back(std::get<0>(full_output)); // all outputs
        output.push_back(
            std::get<0>(std::get<1>(full_output))); // last hidden value
        output.push_back(
            std::get<1>(std::get<1>(full_output))); // last memory / c  value
        if (_lstm_continuation)
          {
            _rnn_memories[opname_v] = std::get<1>(full_output);
            _rnn_has_memories[opname_v] = true;
          }
      }
    else if (optype == "InnerProduct")
      output.push_back(_modules[opname_v].forward(inputsTensor[0]));
    else if (optype == "ReLU")
      output.push_back(_modules[opname_v].forward(inputsTensor[0]));
    else if (optype == "Tile")
      {
        torch::Tensor x = inputsTensor[0];
        std::vector<long int> rssizes = x.sizes().vec();
        rssizes.erase(rssizes.begin()); // remove first dim because it is
        // 1 : num_layers * num_directions
        rssizes.insert(rssizes.begin() + _graph[v].axis, 1L);
        torch::Tensor y = x.reshape(rssizes);
        std::vector<long int> tiless(rssizes.size(), 1);
        if (_graph[v].outputsdims[0][_graph[v].axis]
            < 0) // to be autodetermined : autoencoder LSTM case only for now
          tiless[_graph[v].axis] = _autoencoder_timesteps;
        else
          tiless[_graph[v].axis] = _graph[v].outputsdims[0][_graph[v].axis];
        output.push_back(y.repeat(tiless));
      }
    else
      throw TorchGraphException("unknown optype " + optype + " for operator "
                                + opname_v);
    return output;
  }

  void TorchGraphBackend::allocate_modules()
  {
    _allocation_done = false;
    for (BaseGraph::Vertex v : _sortedOps)
      {
        if (!_graph[v].alloc_needed)
          continue;
        if (_parameters_used)
          throw TorchGraphException(
              "parameters reallocation necessary while they are used "
              "elsewhere. You should module.forward() / module.set_input() "
              "/ "
              "module.finalize() with correct input dimensions before "
              "modules.parameters() or  module.parameters_release() if you "
              "know what you are doing");
        std::string optype = this->optype(v);
        std::string opname = this->opname(v);
        if (_modules.find(opname) != _modules.end())
          unregister_module(opname);

        torch::nn::AnyModule m;
        if (optype == "LSTM")
          {
            // dim(v,0,2) is 2nd dimension of input 0 of v, ie datadim for
            // lstm
            LSTM m = register_module(
                opname, LSTM(LSTMOptions(dim(v, 0, 2), num_output(v))
                                 .num_layers(1)
                                 .batch_first(true)
                                 .bidirectional(false)));

            _modules[opname] = AnyModule(m);
            _graph[v].alloc_needed = false;
            _rnn_has_memories[opname] = false;
            _allocation_done = true;
          }
        else if (optype == "RNN")
          {
            // dim(v,0,2) is 2nd dimension of input 0 of v, ie datadim for
            // lstm
            RNN m = register_module(opname,
                                    RNN(RNNOptions(dim(v, 0, 2), num_output(v))
                                            .num_layers(1)
                                            .batch_first(true)
                                            .bidirectional(false)));
            _modules[opname] = AnyModule(m);
            _graph[v].alloc_needed = false;
            _rnn_has_memories[opname] = false;
            _allocation_done = true;
          }
        else if (optype == "InnerProduct")
          {
            // dim(v,0,2) is 2nd dimension of input 0 of v, ie datadim for
            // lstm output
            Linear m = register_module(
                opname,
                Linear(LinearOptions(dim(v, 0, 2), num_output(v)).bias(true)));
            _modules[opname] = AnyModule(m);
            _graph[v].alloc_needed = false;
            _allocation_done = true;
          }
        else if (optype == "Tile")
          _graph[v].alloc_needed = false;
        else if (optype == "ReLU")
          {
            PReLU m = register_module(opname, PReLU());
            _modules[opname] = AnyModule(m);
            _graph[v].alloc_needed = false;
            _allocation_done = true;
          }
      }
    to(_device, _dtype);
  }

  void TorchGraphBackend::to(torch::Device device, torch::Dtype dtype,
                             bool non_blocking)
  {
    if (!allocated())
      throw TorchGraphException(
          "trying to move NN to gpu/cpu / cast before its effective "
          "allocation. finalize(inputdim) or forward() it once before to()");
    else
      {
        torch::nn::Module::to(device, dtype, non_blocking);
        _device = device;
        _dtype = dtype;
      }
  }

  void TorchGraphBackend::to(torch::Dtype dtype, bool non_blocking)
  {
    if (!allocated())
      throw TorchGraphException(
          "trying to cast params before their effective allocation. "
          "finalize(inputdim) or forward() it once before to()");
    else
      {
        torch::nn::Module::to(dtype, non_blocking);
        _dtype = dtype;
      }
  }

  void TorchGraphBackend::to(torch::Device device, bool non_blocking)
  {
    if (!allocated())
      throw TorchGraphException(
          "trying to move NN to gpu/cpu before its effective allocation. "
          "finalize(inputdim) or forward() it once before to()");
    else
      {
        torch::nn::Module::to(device, non_blocking);
        _device = device;
      }
  }

  std::vector<torch::Tensor> TorchGraphBackend::parameters(bool recurse)
  {
    if (!allocated())
      throw TorchGraphException(
          "trying to get parameters (for optim?) before their effective "
          "allocation. finalize(inputdim) or forward() it once before to()");
    else
      {
        _parameters_used = true;
        return torch::nn::Module::parameters(recurse);
      }
  }

  std::vector<int> TorchGraphBackend::get_input_dims(
      std::string optype,
      torch::OrderedDict<std::string, torch::Tensor> params)
  {
    std::vector<int> dims;
    if (optype == "LSTM")
      {
        torch::Tensor *val = params.find("weight_ih_l0");
        dims.push_back(val->size(1));
      }
    else if (optype == "InnerProduct")
      {
        torch::Tensor *val = params.find("weight");
        dims.push_back(val->size(1));
      }
    else
      {
        // TODO : others types are not yet implemented
        dims.push_back(-1);
      }
    return dims;
  }

  std::vector<int> TorchGraphBackend::get_input_dims_from_loaded()
  {
    BaseGraph::Vertex o = _sortedOps[0];
    auto opname_o = opname(o);
    torch::nn::AnyModule am = _modules[opname_o];
    std::shared_ptr<torch::nn::Module> m = am.ptr();
    std::string optype_o = optype(o);
    auto params = m->named_parameters();

    return get_input_dims(optype_o, params);
  }
}
