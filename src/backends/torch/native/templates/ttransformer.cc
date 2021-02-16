/**
 * DeepDetect
 * Copyright (c) 2019-2020 Jolibrain
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

#include "ttransformer.h"
#include <math.h>

namespace dd
{

  void TTransformer::init()
  {

    _logger->info("embedding type: {}  dim: {}  nlayers: {}  dropout:{}",
                  std::to_string(_embed_type), std::to_string(_embed_dim),
                  std::to_string(_embed_nlayers),
                  std::to_string(_embed_dropout));
    _embedder = register_module("embedder",
                                Embedder(_input_dim, _input_len, _embed_type,
                                         _embed_activation, _embed_dim,
                                         _embed_nlayers, _embed_dropout));

    _logger->info("pe type: {}  dropout: {}   learn: {} ",
                  std::to_string(_petype), std::to_string(_pe_dropout),
                  std::to_string(_embed_dropout), std::to_string(_pelearn));
    _pe = register_module("pe",
                          PositionalEncoding(_input_len, _embed_dim, _petype,
                                             _pe_dropout, _pelearn));

    if (_peagg == PEAggregation::sum)
      _logger->info("using classical (sum) pe aggregation");
    else if (_peagg == PEAggregation::cat)
      _logger->info("using cat pe aggregation");
    else
      throw MLLibBadParamException("unkown pe agg");

    _logger->info(
        "encoder nheads: {}  nlayers: {}  hidden_dim: {}  dropout: {}",
        std::to_string(_encoder_nheads), std::to_string(_encoder_nlayers),
        std::to_string(_encoder_hidden_dim), std::to_string(_encoder_dropout));
    if (_peagg == PEAggregation::sum)
      _encoder = register_module(
          "TEncoder", TEncoder(_input_len, _embed_dim, _encoder_nheads,
                               _encoder_nlayers, _encoder_hidden_dim,
                               _encoder_dropout, _encoder_activation));
    else if (_peagg == PEAggregation::cat)
      _encoder = register_module(
          "TEncoder", TEncoder(_input_len, _embed_dim * 2, _encoder_nheads,
                               _encoder_nlayers, _encoder_hidden_dim,
                               _encoder_dropout, _encoder_activation));
    else
      throw MLLibBadParamException("unknow pe aggregation type");

    if (_simple_decoder)
      _logger->info("decoder mlp: nlayers: {}  hidden_dim: {}  dropout: {}",
                    std::to_string(_decoder_nlayers),
                    std::to_string(_decoder_hidden_dim),
                    std::to_string(_encoder_dropout));
    else
      _logger->info(
          "decoder transformer: nheads: {} nlayers: {}  hidden_dim: {}  "
          "dropout: {}",
          std::to_string(_decoder_nheads), std::to_string(_decoder_nlayers),
          std::to_string(_decoder_nlayers),
          std::to_string(_decoder_hidden_dim),
          std::to_string(_encoder_dropout));
    if (_peagg == PEAggregation::sum)
      _decoder = register_module(
          "TDecoder", TDecoder(_simple_decoder, _embed_dim, _input_len,
                               _output_dim, _output_len, _decoder_nheads,
                               _decoder_nlayers, _decoder_hidden_dim,
                               _decoder_dropout, _decoder_activation));
    else if (_peagg == PEAggregation::cat)
      _decoder = register_module(
          "TDecoder", TDecoder(_simple_decoder, _embed_dim * 2, _input_len,
                               _output_dim, _output_len, _decoder_nheads,
                               _decoder_nlayers, _decoder_hidden_dim,
                               _decoder_dropout, _decoder_activation));

    if (!_autoreg)
      _encoder_mask
          = register_buffer("encoder_mask", generate_mask_none(_input_len));
    else
      {
        _encoder_mask
            = register_buffer("encoder_mask", generate_mask_past(_input_len));
        _decoder_mask
            = register_buffer("decoder_mask", generate_mask_past(_output_len));
      }

    if (_embed_dim % _encoder_nheads != 0)
      {
        throw MLLibBadParamException(
            "embed_dim must be divisible by encoder_nheads");
      }
  }

  torch::Tensor TTransformer::generate_mask_none(int len)
  {
    return torch::ones({ len, len }).to(torch::kFloat32);
  }
  torch::Tensor TTransformer::generate_mask_past(int len)
  {
    // TODO : past mask
    return torch::ones({ len, len }).to(torch::kFloat32);
  }

  void TTransformer::set_encoder_mask(torch::Tensor mask)
  {
    // TODO : change dims
    _encoder_mask = mask;
  }

  void TTransformer::set_decoder_mask(torch::Tensor mask)
  {
    // TODO : change dims
    _decoder_mask = mask;
  }

  torch::Tensor TTransformer::forward(torch::Tensor x)
  {
    x = _embedder(x);

    torch::Tensor out;
    if (!_autoreg)
      {
        if (_peagg == PEAggregation::sum)
          x = x + _pe();
        else if (_peagg == PEAggregation::cat)
          x = torch::cat({ x, _pe().repeat({ x.size(0), 1, 1 }) }, -1);
        out = _encoder(x, _encoder_mask);
        out = _decoder->forward(out, _decoder_mask);
        out = out.reshape({ -1, _output_len, _output_dim });
        return out;
      }
    else
      {
        throw MLLibBadParamException("autoreg not implemented yet");
      }
    return out;
  }
}
