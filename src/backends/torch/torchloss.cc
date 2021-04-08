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

#include "torchloss.h"
#pragma GCC diagnostic pop

namespace dd
{
  torch::Tensor TorchLoss::reloss(torch::Tensor y_pred)
  {
    return loss(y_pred, _y, _ivx);
  }

  torch::Tensor TorchLoss::loss(torch::Tensor y_pred, torch::Tensor y,
                                std::vector<c10::IValue> &ivx)
  {
    // blow memorize to be able to redo loss call (in case of solver.sam)
    _y = y;
    _ivx = ivx;
    torch::Tensor x = ivx[0].toTensor();

    torch::Tensor loss;

    if (_model_loss)
      {
        loss = y_pred;
      }
    else if (_seq_training)
      {
        // Convert [n_batch, sequence_length, vocab_size] to
        // [n_batch
        // * sequence_length, vocab_size]
        // + ignore non-masked tokens (== -1 in target)
        loss = torch::nll_loss(
            torch::log_softmax(
                y_pred.view(torch::IntList{ -1, y_pred.size(2) }), 1),
            y.view(torch::IntList{ -1 }), _class_weights,
            torch::Reduction::Mean, -1);
      }
    else if (_timeserie)
      {
        if (_native != nullptr)
          loss = _native->loss(_loss, x, y_pred, y);
        else
          {
            if (_loss.empty() || _loss == "L1" || _loss == "l1")
              loss = torch::l1_loss(y_pred, y);
            else if (_loss == "L2" || _loss == "l2" || _loss == "eucl")
              loss = torch::mse_loss(y_pred, y);
            else
              throw MLLibBadParamException("unknown loss " + _loss);
          }
      }
    else if (_regression)
      {
        if (_loss.empty() || _loss == "L1" || _loss == "l1")
          loss = torch::l1_loss(y_pred, y);
        else if (_loss == "L2" || _loss == "l2" || _loss == "eucl")
          loss = torch::mse_loss(y_pred, y);
        else
          throw MLLibBadParamException("unknown loss " + _loss);
      }
    else if (_classification)
      {
        // As CrossEntropy is not available (Libtorch 1.1) we use
        // nllloss
        // + log_softmax
        loss = torch::nll_loss(torch::log_softmax(y_pred, 1),
                               y.view(torch::IntList{ -1 }), _class_weights);
      }
    else
      {
        throw MLLibBadParamException("unexpected model type");
      }
    return loss;
  }
}
