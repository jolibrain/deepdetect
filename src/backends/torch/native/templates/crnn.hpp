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

#ifndef CRNN_H
#define CRNN_H

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "torch/torch.h"
#pragma GCC diagnostic pop
#include "../../torchinputconns.h"
#include "../native_net.h"

namespace dd
{
  class CRNN : public NativeModuleImpl<CRNN>
  {
  public:
    torch::nn::LSTM _lstm = nullptr;
    int _timesteps = 32;
    int _num_layers = 3;
    int _hidden_size = 64;

    // dataset / backbone dependent variables
    int _input_size = 64;
    int _output_size = 2;

  public:
    CRNN(int timesteps = 32, int num_layers = 3, int hidden_size = 0,
         int input_size = 64, int output_size = 2)
        : _timesteps(timesteps), _num_layers(num_layers),
          _hidden_size(hidden_size), _input_size(input_size),
          _output_size(output_size)
    {
      init();
    }

    CRNN(const APIData &ad_params, const std::vector<long int> &input_dims,
         int output_size = -1)
    {
      get_params(ad_params, input_dims, output_size);
      init();
    }

    CRNN(const CRNN &other)
        : torch::nn::Module(other), _timesteps(other._timesteps),
          _num_layers(other._num_layers), _hidden_size(other._hidden_size),
          _input_size(other._input_size), _output_size(other._output_size)
    {
      init();
    }

    CRNN &operator=(const CRNN &other)
    {
      _timesteps = other._timesteps;
      _num_layers = other._num_layers;
      _hidden_size = other._hidden_size;

      _input_size = other._input_size;
      _output_size = other._output_size;
      init();
      return *this;
    }

    void init();

    void get_params(const APIData &ad_params,
                    const std::vector<long int> &input_dims, int output_size);

    void set_output_size(int output_size);

    torch::Tensor forward(torch::Tensor x) override;

    void reset() override
    {
      init();
    }

    torch::Tensor extract(torch::Tensor x, std::string extract_layer) override
    {
      (void)x;
      (void)extract_layer;
      return torch::Tensor();
    }

    bool extractable(std::string extract_layer) const override
    {
      (void)extract_layer;
      return false;
    }

    std::vector<std::string> extractable_layers() const override
    {
      return std::vector<std::string>();
    }

    torch::Tensor cleanup_output(torch::Tensor output) override
    {
      return output;
    }

    torch::Tensor loss(std::string loss, torch::Tensor input,
                       torch::Tensor output, torch::Tensor target) override;
  };
}

#endif // CRNN_H
