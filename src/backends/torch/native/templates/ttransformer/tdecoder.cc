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

#include "tdecoder.h"

namespace dd
{
  void TDecoderImpl::init()
  {
    if (_simple)
      {
        _lins.push_back(register_module(
            "decoder_0", torch::nn::Linear(_embed_dim * _input_len,
                                           _output_dim * _output_len)));
        for (int i = 1; i < _nlayers; ++i)
          {
            std::string lname = "decoder_" + std::to_string(i);
            _lins.push_back(register_module(
                lname, torch::nn::Linear(_output_dim * _output_len,
                                         _output_dim * _output_len)));
          }
        if (_dropout_ratio != 0.0)
          _dropout = register_module("decoder_dropout",
                                     torch::nn::Dropout(_dropout_ratio));
      }
    else
      {
        torch::nn::TransformerDecoderLayerOptions opts
            = torch::nn::TransformerDecoderLayerOptions(_output_dim, _nheads);
        opts = opts.dropout(_dropout_ratio).dim_feedforward(_hidden_dim);
        if (_activation == Activation::relu)
          opts = opts.activation(torch::kReLU);
        else
          opts = opts.activation(torch::kGELU);

        torch::nn::TransformerDecoderLayer decoder_layer
            = torch::nn::TransformerDecoderLayer(opts);
        _transformer_decoder = register_module(
            "decoder",
            torch::nn::TransformerDecoder(
                torch::nn::TransformerDecoderOptions(decoder_layer, _nlayers)
                    .norm(torch::nn::AnyModule(torch::nn::LayerNorm(
                        torch::nn::LayerNormOptions({ _output_dim }))))));
      }
  }

  torch::Tensor TDecoderImpl::forward(torch::Tensor x, torch::Tensor mask)
  {
    torch::Tensor out;
    if (_simple)
      {
        x = x.reshape({ -1, _embed_dim * _input_len });
        x = _lins[0](x);
        for (int i = 1; i < _nlayers; ++i)
          {
            if (_activation == Activation::relu)
              x = torch::relu(x);
            else
              x = torch::gelu(x);
            if (_dropout_ratio != 0.0)
              x = _dropout(x);
            x = _lins[i](x);
          }
        out = x.reshape({ -1, _output_len, _output_dim });
        return out;
      }
    else
      return forward(x, x, mask, mask);
  }

  torch::Tensor TDecoderImpl::forward(torch::Tensor target, torch::Tensor mem,
                                      torch::Tensor tmask, torch::Tensor mmask)
  {
    if (_simple)
      throw MLLibBadParamException("complex forward call for simple decoder");
    torch::Tensor tmem = mem.transpose(0, 1);
    torch::Tensor ttarget = target.transpose(0, 1);
    torch::Tensor out = _transformer_decoder(ttarget, tmask, tmask, mmask);
    out.transpose_(0, 1);
    return out;
  }

}
