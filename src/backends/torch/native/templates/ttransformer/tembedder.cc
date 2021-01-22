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

#include "tembedder.h"

#define SIREN_W0 30.0
#define SIREN_C 6.0

namespace dd
{
  void EmbedderImpl::init()
  {
    // if sine do special init
    if (_etype == EmbedType::step) // embed per timestep
      _lins.push_back(register_module(
          "embed_0", torch::nn::Linear(_input_dim, _embed_dim)));
    else if (_etype == EmbedType::all) // embed whole ts at once
      _lins.push_back(register_module(
          "embed_0", torch::nn::Linear(_input_dim * _input_len,
                                       _embed_dim * _input_len)));
    else
      _lins.push_back(register_module(
          "embed_0", torch::nn::Linear(_input_len, _input_len)));

    if (_eactivation == Activation::siren)
      {
        float num_inputs = static_cast<float>(_lins[0]->weight.size(-1));
        torch::nn::init::uniform_(_lins[0]->weight, -1.0 / num_inputs,
                                  1.0 / num_inputs);
      }
    for (int i = 1; i < _nlayers; ++i)
      {
        std::string lname = "embed_" + std::to_string(i);
        if (_etype == EmbedType::step)
          _lins.push_back(register_module(
              lname, torch::nn::Linear(_embed_dim, _embed_dim)));
        else if (_etype == EmbedType::all)
          _lins.push_back(register_module(
              lname, torch::nn::Linear(_embed_dim * _input_len,
                                       _embed_dim * _input_len)));
        else
          _lins.push_back(register_module(
              lname, torch::nn::Linear(_input_len, _input_len)));
        if (_eactivation == Activation::siren)
          {
            float num_inputs = _lins[i]->weight.size(-1);
            float b = sqrt(SIREN_C / num_inputs) / SIREN_W0;
            torch::nn::init::uniform_(_lins[i]->weight, -b, b);
          }
      }
    if (_etype == EmbedType::serie)
      {
        _serieEmbedder = register_module(
            "embed_final", torch::nn::Linear(_input_dim, _embed_dim));
        if (_eactivation == Activation::siren)
          {
            float num_inputs = _serieEmbedder->weight.size(-1);
            float b = sqrt(SIREN_C / num_inputs) / SIREN_W0;
            torch::nn::init::uniform_(_serieEmbedder->weight, -b, b);
          }
      }
    if (_dropout_ratio != 0.0)
      _dropout = register_module("embed_dropout",
                                 torch::nn::Dropout(_dropout_ratio));
  }

  torch::Tensor EmbedderImpl::forward(torch::Tensor x)
  {
    if (_etype == EmbedType::all)
      x = x.reshape({ -1, _input_len * _input_dim });
    else if (_etype == EmbedType::serie)
      x = x.transpose(1, 2);
    x = _lins[0](x);
    if (_eactivation == Activation::siren)
      x = torch::sin(SIREN_W0 * x);
    else if (_eactivation == Activation::relu)
      x = torch::relu(x);
    else
      x = torch::gelu(x);
    for (int i = 1; i < _nlayers; ++i)
      {
        if (_dropout_ratio != 0.0)
          x = _dropout(x);
        x = _lins[i](x);
        if (_eactivation == Activation::siren)
          x = torch::sin(SIREN_W0 * x);
        else if (_eactivation == Activation::relu)
          x = torch::relu(x);
        else
          x = torch::gelu(x);
      }
    if (_etype == EmbedType::all)
      x = x.reshape({ -1, _input_len, _embed_dim });
    else if (_etype == EmbedType::serie)
      {
        x = x.transpose(1, 2);
        x = _serieEmbedder(x);
        if (_eactivation == Activation::siren)
          x = torch::sin(SIREN_W0 * x);
        else if (_eactivation == Activation::relu)
          x = torch::relu(x);
        else
          x = torch::gelu(x);
      }
    return x;
  }
}
