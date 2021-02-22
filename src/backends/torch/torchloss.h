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

#ifndef TORCH_LOSS_H
#define TORCH_LOSS_H

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <torch/torch.h>
#pragma GCC diagnostic pop

#include "apidata.h"
#include "torchmodule.h"

namespace dd
{

  /**
   * \brief this class is a wrapper around torch native solvers/optimizer and
   * our own versions
   */
  class TorchLoss
  {
  public:
    /**
     * \brief simple constructor
     */
    TorchLoss(std::string loss, bool seq_training, bool timeserie,
              bool regression, bool classification, TorchModule &module,
              std::shared_ptr<spdlog::logger> logger)
        : _loss(loss), _seq_training(seq_training), _timeserie(timeserie),
          _regression(regression), _classification(classification),
          _logger(logger)
    {
      _native = module._native;
    }

    void set_class_weights(torch::Tensor cw)
    {
      _class_weights = cw;
    }

    torch::Tensor loss(torch::Tensor y_pred, torch::Tensor y,
                       std::vector<c10::IValue> &x);
    torch::Tensor reloss(torch::Tensor y_pred);

    std::vector<c10::IValue> getLastInputs()
    {
      return _ivx;
    }

  protected:
    std::string _loss;
    bool _seq_training;
    bool _timeserie;
    bool _regression;
    bool _classification;
    torch::Tensor _class_weights = {};
    std::shared_ptr<NativeModule> _native;
    std::shared_ptr<spdlog::logger> _logger;
    torch::Tensor _y_pred;
    torch::Tensor _y;
    std::vector<c10::IValue> _ivx;
  };
}
#endif
