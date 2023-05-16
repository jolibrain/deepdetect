/**
 * DeepDetect
 * Copyright (c) 2022 Jolibrain
 * Author:  Louis Jean <louis.jean@jolibrain.com>
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

#include "crnn_head.hpp"

#include "../../torchlib.h"

namespace dd
{
  void CRNNHeadImpl::get_params(const APIData &ad_params,
                                const std::vector<int64_t> &input_dims,
                                int output_size)
  {
    if (ad_params.has("timesteps"))
      _timesteps = ad_params.get("timesteps").get<int>();
    if (ad_params.has("hidden_size"))
      _hidden_size = ad_params.get("hidden_size").get<int>();
    if (ad_params.has("num_layers"))
      _num_layers = ad_params.get("num_layers").get<int>();
    if (ad_params.has("bidirectional"))
      _bidirectional = ad_params.get("bidirectional").get<bool>();

    if (output_size > 0)
      _output_size = output_size;

    // compute input size
    at::Tensor dummy = torch::zeros(
        std::vector<int64_t>(input_dims.begin(), input_dims.end()));
    int bs = dummy.size(0);
    int ts = _timesteps;
    if (ts <= 0)
      ts = dummy.size(-1);
    dummy = dummy.reshape({ bs, -1, ts });
    _input_size = dummy.size(1);
    std::cout << "LSTM input shape: " << dummy.sizes() << std::endl;
  }

  void CRNNHeadImpl::init()
  {
    if (_lstm)
      {
        unregister_module("lstm");
        _lstm = nullptr;
      }
    if (_proj)
      {
        unregister_module("proj");
        _proj = nullptr;
      }

    _lstm = register_module(
        "lstm",
        torch::nn::LSTM(torch::nn::LSTMOptions(_input_size, _hidden_size)
                            .num_layers(_num_layers)));
    int proj_size = _output_size;
    int d = _bidirectional ? 2 : 1;
    _proj = register_module("proj",
                            torch::nn::Linear(d * _hidden_size, proj_size));
  }

  void CRNNHeadImpl::set_output_size(int output_size)
  {
    if (_output_size != output_size)
      {
        _output_size = output_size;
        init();
      }
  }

  torch::Tensor CRNNHeadImpl::forward(torch::Tensor feats)
  {
    // Input: feature map from resnet
    // Output: LSTM results
    int bs = feats.size(0);
    int ts = _timesteps;
    if (ts == 0)
      ts = feats.size(-1);

    feats = feats.reshape({ bs, -1, ts });
    // timesteps first
    feats = feats.permute({ 2, 0, 1 });

    // std::cout << "feats before: " << feats.sizes() << std::endl;
    std::tuple<at::Tensor, std::tuple<at::Tensor, at::Tensor>> outputs
        = _lstm->forward(feats);
    auto out = _proj->forward(std::get<0>(outputs));
    // std::cout << "feats after: " << out.sizes() << std::endl;

    return out;
  }
}
