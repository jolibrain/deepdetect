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

#ifndef TEMBEDDER_H
#define TEMBEDDER_H

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "torch/torch.h"
#pragma GCC diagnostic pop
#include "../../native_net.h"
#include "ttypes.h"

namespace dd
{
  class EmbedderImpl : public torch::nn::Cloneable<EmbedderImpl>
  {
  public:
    EmbedderImpl(int input_dim, int input_len, EmbedType etype,
                 Activation eactivation, int embed_dim, int nlayers,
                 float dropout)
        : _input_dim(input_dim), _input_len(input_len), _etype(etype),
          _eactivation(eactivation), _embed_dim(embed_dim), _nlayers(nlayers),
          _dropout_ratio(dropout)
    {
      init();
    }

    EmbedderImpl(const EmbedderImpl &e)
        : torch::nn::Module(e), _input_dim(e._input_dim),
          _input_len(e._input_len), _etype(e._etype),
          _eactivation(e._eactivation), _embed_dim(e._embed_dim),
          _nlayers(e._nlayers), _dropout_ratio(e._dropout_ratio)
    {
      init();
    }

    EmbedderImpl &operator=(const EmbedderImpl &e)
    {
      torch::nn::Module::operator=(e);
      _input_dim = e._input_dim;
      _input_len = e._input_len;
      _etype = e._etype;
      _eactivation = e._eactivation;
      _embed_dim = e._embed_dim;
      _nlayers = e._nlayers;
      _dropout_ratio = e._dropout_ratio;
      _dropout = e._dropout;
      _lins = e._lins;
      _serieEmbedder = e._serieEmbedder;
      return *this;
    }

    torch::Tensor forward(torch::Tensor x);
    void reset() override
    {
      init();
    }

  protected:
    void init();
    int _input_dim;
    int _input_len;
    EmbedType _etype;
    Activation _eactivation;
    int _embed_dim;
    int _nlayers;
    float _dropout_ratio;
    torch::nn::Dropout _dropout{ nullptr };
    std::vector<torch::nn::Linear> _lins;
    torch::nn::Linear _serieEmbedder{ nullptr };
  };
  typedef torch::nn::ModuleHolder<EmbedderImpl> Embedder;
}
#endif
