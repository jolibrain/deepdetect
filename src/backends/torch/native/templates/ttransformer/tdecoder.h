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

#ifndef TDECODER_H
#define TDECODER_H
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "torch/torch.h"
#pragma GCC diagnostic pop
#include "../../native_net.h"
#include "ttypes.h"
namespace dd
{
  class TDecoderImpl : public torch::nn::Cloneable<TDecoderImpl>
  {
  public:
    TDecoderImpl(bool simple, int embed_dim, int input_len, int output_dim,
                 int output_len, int nheads, int nlayers, int hidden_dim,
                 float dropout, Activation activation)
        : _simple(simple), _embed_dim(embed_dim), _input_len(input_len),
          _output_dim(output_dim), _output_len(output_len), _nheads(nheads),
          _nlayers(nlayers), _hidden_dim(hidden_dim), _dropout_ratio(dropout),
          _activation(activation)
    {
      init();
    }

    TDecoderImpl(const TDecoderImpl &d)
        : torch::nn::Module(d), _simple(d._simple), _embed_dim(d._embed_dim),
          _input_len(d._input_len), _output_dim(d._output_dim),
          _output_len(d._output_len), _nheads(d._nheads), _nlayers(d._nlayers),
          _hidden_dim(d._hidden_dim), _dropout_ratio(d._dropout_ratio),
          _activation(d._activation)
    {
      init();
    }

    void reset() override
    {
      init();
    }

    torch::Tensor forward(torch::Tensor x, torch::Tensor mask);

    torch::Tensor forward(torch::Tensor previous_target,
                          torch::Tensor encoder_mem, torch::Tensor tmask,
                          torch::Tensor mmask);

  protected:
    void init();
    bool _simple;
    int _embed_dim;
    int _input_len;
    int _output_dim;
    int _output_len;
    int _nheads;
    int _nlayers;
    int _hidden_dim;
    float _dropout_ratio;
    torch::nn::Dropout _dropout;
    Activation _activation;
    std::vector<torch::nn::Linear> _lins;
    torch::nn::TransformerDecoder _transformer_decoder{ nullptr };
  };
  typedef torch::nn::ModuleHolder<TDecoderImpl> TDecoder;

}
#endif
