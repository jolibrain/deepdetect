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

#ifndef CRNNHead_H
#define CRNNHead_H

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "torch/torch.h"
#pragma GCC diagnostic pop
#include "../../torchinputconns.h"
#include "../native_net.h"

namespace dd
{
  class CRNNHeadImpl : public torch::nn::Cloneable<CRNNHeadImpl>
  {
  public:
    int _timesteps = 0;
    int _num_layers = 3;
    int _hidden_size = 256;
    bool _bidirectional = false;

    // dataset / backbone dependent variables
    int _input_size = 64;
    int _output_size = 2;

    torch::nn::LSTM _lstm = nullptr;
    torch::nn::Linear _proj = nullptr;

  public:
    CRNNHeadImpl(int timesteps = 0, int num_layers = 3, int hidden_size = 256,
                 bool bidirectional = false, int input_size = 64,
                 int output_size = 2)
        : _timesteps(timesteps), _num_layers(num_layers),
          _hidden_size(hidden_size), _bidirectional(bidirectional),
          _input_size(input_size), _output_size(output_size)
    {
      init();
    }

    CRNNHeadImpl(const APIData &ad_params,
                 const std::vector<int64_t> &input_dims, int output_size = -1)
    {
      get_params(ad_params, input_dims, output_size);
      init();
    }

    CRNNHeadImpl(const CRNNHeadImpl &other)
        : torch::nn::Module(other), _timesteps(other._timesteps),
          _num_layers(other._num_layers), _hidden_size(other._hidden_size),
          _bidirectional(other._bidirectional), _input_size(other._input_size),
          _output_size(other._output_size)
    {
      init();
    }

    CRNNHeadImpl &operator=(const CRNNHeadImpl &other)
    {
      _timesteps = other._timesteps;
      _num_layers = other._num_layers;
      _hidden_size = other._hidden_size;
      _bidirectional = other._bidirectional;

      _input_size = other._input_size;
      _output_size = other._output_size;
      init();
      return *this;
    }

    void init();

    void get_params(const APIData &ad_params,
                    const std::vector<int64_t> &input_dims, int output_size);

    void set_output_size(int output_size);

    torch::Tensor forward(torch::Tensor x);

    void reset() override
    {
      init();
    }
  };
  typedef torch::nn::ModuleHolder<CRNNHeadImpl> CRNNHead;
}

#endif // CRNNHead_H
