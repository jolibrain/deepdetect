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
    TorchLoss(std::string loss, bool model_loss, bool seq_training,
              bool timeserie, bool regression, bool classification,
              bool segmentation, torch::Tensor class_weights,
              double reg_weight, TorchModule &module,
              std::shared_ptr<spdlog::logger> logger)
        : _loss(loss), _model_loss(model_loss), _seq_training(seq_training),
          _timeserie(timeserie), _regression(regression),
          _classification(classification), _segmentation(segmentation),
          _class_weights(class_weights), _reg_weight(reg_weight),
          _logger(logger)
    {
      _native = module._native;
    }

    torch::Tensor loss(c10::IValue model_out, torch::Tensor target,
                       std::vector<c10::IValue> &x);
    torch::Tensor reloss(c10::IValue model_out);

    std::vector<c10::IValue> getLastInputs()
    {
      return _ivx;
    }

  protected:
    std::string _loss;
    bool _model_loss; /** < wether loss is provided by the model */
    bool _seq_training;
    bool _timeserie;
    bool _regression;
    bool _classification;
    bool _segmentation;
    torch::Tensor _class_weights = {};
    double _reg_weight = 1; /** < on detection models, weight to apply to bbox
                               regression loss */
    std::shared_ptr<NativeModule> _native;
    std::shared_ptr<spdlog::logger> _logger;
    torch::Tensor _y_pred;
    torch::Tensor _y;
    std::vector<c10::IValue> _ivx;
    long int _num_batches = 0;
  };
}
#endif
