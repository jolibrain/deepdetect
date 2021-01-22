/**
 * DeepDetect
 * Copyright (c) 2021 Jolibrain
 * Author:  Guillaume Infantes <guillaume.infantes@jolibrain.com>
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

#include "mllibstrategy.h"
#include "positionalenc.h"
using namespace torch::indexing;

namespace dd
{
  void PositionalEncodingImpl::init()
  {

    if (_dropout_ratio != 0.0)
      _dropout
          = register_module("dropout", torch::nn::Dropout(_dropout_ratio));

    torch::Tensor pet;
    if (_et == PEType::naive)
      {
        pet = torch::arange(0, _seq_len * _datadim,
                            torch::TensorOptions(torch::kFloat32))
                  .reshape({ _seq_len, _datadim });
        pet = (pet / (float)(_seq_len * _datadim)) * 2.0 - 1.0;
      }
    else if (_et == PEType::none)
      {
        pet = torch::zeros(_seq_len * _datadim,
                           torch::TensorOptions(torch::kFloat32))
                  .reshape({ _seq_len, _datadim });
      }
    else if (_et == PEType::sincos)
      {
        pet = torch::zeros({ _seq_len, _datadim });
        torch::Tensor position
            = torch::arange(0, _seq_len, torch::TensorOptions(torch::kFloat32))
                  .unsqueeze(1);
        torch::Tensor div_term
            = torch::exp(torch::arange(0, _datadim, 2).to(torch::kFloat32)
                         * (-log(1E4) / _datadim));
        pet.index({ Slice(), Slice(0, None, 2) })
            = torch::sin(position * div_term);
        pet.index({ Slice(), Slice(1, None, 2) })
            = torch::cos(position * div_term);
      }
    else
      {
        throw MLLibInternalException("unknown PE");
      }
    pet.unsqueeze_(0);
    if (_learn)
      _pet = register_parameter("pe", pet);
    else
      _pet = register_buffer("pe", pet);
  }

  torch::Tensor PositionalEncodingImpl::forward()
  {
    if (_dropout_ratio != 0.0)
      return (_dropout(_pet));
    return _pet;
  }
}
