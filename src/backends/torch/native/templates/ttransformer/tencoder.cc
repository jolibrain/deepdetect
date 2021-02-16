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

#include "tencoder.h"
namespace dd
{
  void TEncoderImpl::init()
  {
    torch::nn::TransformerEncoderLayerOptions opts
        = torch::nn::TransformerEncoderLayerOptions(_embed_dim, _nheads)
              .dim_feedforward(_hidden_dim)
              .dropout(_dropout);
    if (_activation == Activation::relu)
      opts = opts.activation(torch::kReLU);
    else
      opts = opts.activation(torch::kGELU);
    torch::nn::TransformerEncoderLayer encoder_layer
        = torch::nn::TransformerEncoderLayer(opts);
    _encoder = register_module(
        "encoder",
        torch::nn::TransformerEncoder(
            torch::nn::TransformerEncoderOptions(encoder_layer, _nlayers)));
  }

  torch::Tensor TEncoderImpl::forward(torch::Tensor x, torch::Tensor mask)
  {
    x.transpose_(0, 1);
    x = _encoder(x, mask);
    return x.transpose_(0, 1);
  }

}
