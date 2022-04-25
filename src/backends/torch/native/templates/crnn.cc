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

#include "crnn.hpp"

#include "../../torchlib.h"

namespace dd
{
  void CRNN::get_params(const APIData &ad_params,
                        const std::vector<long int> &input_dims,
                        int output_size)
  {
    if (ad_params.has("timesteps"))
      _timesteps = ad_params.get("timesteps").get<int>();
    if (ad_params.has("hidden_size"))
      _hidden_size = ad_params.get("hidden_size").get<int>();
    if (ad_params.has("num_layers"))
      _num_layers = ad_params.get("num_layers").get<int>();

    if (output_size > 0)
      _output_size = output_size;

    at::Tensor dummy = torch::zeros(
        std::vector<int64_t>(input_dims.begin(), input_dims.end()));
    int batch_size = dummy.size(0);
    dummy = dummy.reshape({ batch_size, -1, _timesteps });
    _input_size = dummy.size(1);
  }

  void CRNN::init()
  {
    if (_lstm)
      {
        unregister_module("lstm");
        _lstm = nullptr;
      }

    uint32_t hidden_size, proj_size;
    if (_hidden_size > 0 || _hidden_size == _output_size)
      {
        hidden_size = _hidden_size;
        proj_size = _output_size;
      }
    else
      {
        hidden_size = _output_size;
        proj_size = 0;
      }

    _lstm = register_module(
        "lstm",
        torch::nn::LSTM(torch::nn::LSTMOptions(_input_size, hidden_size)
                            .num_layers(_num_layers)
                            .proj_size(proj_size)));
  }

  void CRNN::set_output_size(int output_size)
  {
    if (_output_size != output_size)
      {
        _output_size = output_size;
        init();
      }
  }

  torch::Tensor CRNN::forward(torch::Tensor feats)
  {
    // Input: feature map from resnet
    // Output: LSTM results
    int batch_size = feats.size(0);
    feats = feats.reshape({ batch_size, -1, _timesteps });
    // timesteps first
    feats = feats.permute({ 2, 0, 1 });

    // std::cout << "feats before: " << feats.sizes() << std::endl;
    std::tuple<at::Tensor, std::tuple<at::Tensor, at::Tensor>> outputs
        = _lstm->forward(feats);
    // std::cout << "feats after: " << std::get<0>(outputs).sizes() <<
    // std::endl;

    return std::get<0>(outputs);
  }

  torch::Tensor CRNN::loss(std::string loss, torch::Tensor input,
                           torch::Tensor output, torch::Tensor target)
  {
    (void)loss;
    (void)input;
    (void)output;
    (void)target;
    throw MLLibInternalException("CRNN::loss not implemented");
  }
}
