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

#ifndef TENCODER_H
#define TENCODER_H

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "torch/torch.h"
#pragma GCC diagnostic pop
#include "../../native_net.h"

#include "ttypes.h"

namespace dd
{

  class TEncoderImpl : public torch::nn::Cloneable<TEncoderImpl>
  {
  public:
    TEncoderImpl(int input_len, int embed_dim, int nheads, int nlayers,
                 int hidden_dim, float dropout, Activation ea)
        : _input_len(input_len), _embed_dim(embed_dim), _nheads(nheads),
          _nlayers(nlayers), _hidden_dim(hidden_dim), _dropout(dropout),
          _activation(ea)
    {
      init();
    }

    TEncoderImpl(const TEncoderImpl &e)
        : torch::nn::Module(e), _input_len(e._input_len),
          _embed_dim(e._embed_dim), _nheads(e._nheads), _nlayers(e._nlayers),
          _hidden_dim(e._hidden_dim), _dropout(e._dropout),
          _activation(e._activation)
    {
      init();
    }

    torch::Tensor forward(torch::Tensor x, torch::Tensor mask);
    void reset() override
    {
      init();
    }

  protected:
    int _input_len;
    int _embed_dim;
    int _nheads;
    int _nlayers;
    int _hidden_dim;
    float _dropout;
    Activation _activation;
    torch::nn::TransformerEncoder _encoder{ nullptr };
    void init();
  };
  typedef torch::nn::ModuleHolder<TEncoderImpl> TEncoder;
}
#endif
