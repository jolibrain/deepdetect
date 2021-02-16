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

#ifndef POSITIONAL_ENC_H
#define POSITIONAL_ENC_H
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "torch/torch.h"
#pragma GCC diagnostic pop
#include "../../native_net.h"
#include "ttypes.h"
namespace dd
{
  class PositionalEncodingImpl
      : public torch::nn::Cloneable<PositionalEncodingImpl>
  {
  public:
    PositionalEncodingImpl(int seq_len, int datadim, PEType et, float dropout,
                           bool learn)
        : _seq_len(seq_len), _datadim(datadim), _et(et),
          _dropout_ratio(dropout), _learn(learn)
    {
      init();
    }

    PositionalEncodingImpl(const PositionalEncodingImpl &pe)
        : torch::nn::Module(pe), _seq_len(pe._seq_len), _datadim(pe._datadim),
          _et(pe._et), _dropout_ratio(pe._dropout_ratio), _learn(pe._learn)
    {
      init();
    }

    torch::Tensor forward();
    void reset() override
    {
      init();
    }

  protected:
    torch::Tensor _pet;
    torch::nn::Dropout _dropout{ nullptr };
    void init();
    int _seq_len;
    int _datadim;
    PEType _et;
    float _dropout_ratio;
    bool _learn;
  };

  typedef torch::nn::ModuleHolder<PositionalEncodingImpl> PositionalEncoding;
}
#endif
