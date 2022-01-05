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
#include <iostream>

namespace dd
{
  torch::Tensor TorchLoss::reloss(c10::IValue y_pred)
  {
    return loss(y_pred, _y, _ivx);
  }

  torch::Tensor TorchLoss::loss(c10::IValue model_out, torch::Tensor y,
                                std::vector<c10::IValue> &ivx)
  {
    // blow memorize to be able to redo loss call (in case of solver.sam)
    _y = y;
    _ivx = ivx;
    torch::Tensor x = ivx[0].toTensor();

    torch::Tensor y_pred;
    torch::Tensor loss;

    if (model_out.isGenericDict())
      {
        auto out_dict = model_out.toGenericDict();
        if (_segmentation)
          y_pred = torch_utils::to_tensor_safe(out_dict.at("out"));
        else if (_loss == "yolox")
          {
            torch::Tensor iou_loss
                = torch_utils::to_tensor_safe(out_dict.at("iou_loss"));
            torch::Tensor l1_loss
                = torch_utils::to_tensor_safe(out_dict.at("l1_loss"));
            torch::Tensor conf_loss
                = torch_utils::to_tensor_safe(out_dict.at("conf_loss"));
            torch::Tensor cls_loss
                = torch_utils::to_tensor_safe(out_dict.at("cls_loss"));
            y_pred = iou_loss * _reg_weight + l1_loss + conf_loss + cls_loss;
          }
        else // _model_loss = true
          y_pred = torch_utils::to_tensor_safe(out_dict.at("total_loss"));
      }
    else
      {
        y_pred = torch_utils::to_tensor_safe(model_out);
      }

    // sanity check
    if (!y_pred.defined() || y_pred.numel() == 0)
      throw MLLibInternalException("The model returned an empty tensor");

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
    else if (_segmentation)
      {
        if (_loss.empty())
          {

            loss = torch::nn::functional::cross_entropy(
                y_pred, y.squeeze(1).to(torch::kLong)); // TODO: options
          }
        else if (_loss == "dice" || _loss == "dice_multiclass"
                 || _loss == "dice_weighted" || _loss == "dice_weighted_batch"
                 || _loss == "dice_weighted_all")
          {
            // see https://arxiv.org/abs/1707.03237
            double smooth = 1e-7;
            torch::Tensor y_true_f
                = torch::one_hot(y.to(torch::kInt64), y_pred.size(1))
                      .squeeze(1)
                      .permute({ 0, 3, 1, 2 })
                      .flatten(2)
                      .to(torch::kFloat32);
            torch::Tensor y_pred_f = torch::flatten(torch::sigmoid(y_pred), 2);

            torch::Tensor intersect;
            torch::Tensor denom;

            if (_loss == "dice" || _loss == "dice_multiclass")
              {
                intersect = torch::sum(y_true_f * y_pred_f, { 2 });
                denom = torch::sum(y_true_f + y_pred_f, { 2 });
              }
            else if (_loss == "dice_weighted")
              {
                torch::Tensor sum = torch::sum(y_true_f, { 2 }) + 1.0;
                torch::Tensor weights = 1.0 / sum / sum;
                intersect = torch::sum(y_true_f * y_pred_f, { 2 }) * weights;
                denom = torch::sum(y_true_f + y_pred_f, { 2 }) * weights;
              }
            else if (_loss == "dice_weighted_batch"
                     || _loss == "dice_weighted_all")
              {
                torch::Tensor sum
                    = torch::sum(y_true_f, std::vector<int64_t>({ 0, 2 }))
                      + 1.0;
                torch::Tensor weights = 1.0 / sum / sum;
                if (_loss == "dice_weighted_all")
                  {
                    if (_num_batches == 0)
                      _class_weights = weights;
                    else
                      {
                        weights = (_class_weights * _num_batches + weights)
                                  / (_num_batches + 1);
                        _class_weights = weights;
                      }
                    _num_batches++;
                  }
                intersect = torch::sum(y_true_f * y_pred_f,
                                       std::vector<int64_t>({ 0, 2 }))
                            * weights;
                denom = torch::sum(y_true_f + y_pred_f,
                                   std::vector<int64_t>({ 0, 2 }))
                        * weights;
              }

            return 1.0 - torch::mean(2.0 * intersect / (denom + smooth));
          }
        else
          throw MLLibBadParamException("unknown loss: " + _loss);
      }
    else
      {
        throw MLLibBadParamException("unexpected model type");
      }
    return loss;
  }
}
