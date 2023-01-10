/**
 * DeepDetect
 * Copyright (c) 2019-2023 Jolibrain
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

/*this is largely inspired/adapted from adam torch/c++ implementation, ie
 * pytorch/torch/csrc/api/src/optim/adam.cpp */

#include "./radam.h"
#include "mllibstrategy.h"

#include <torch/csrc/autograd/variable.h>
#include <torch/nn/module.h>
#include <torch/serialize/archive.h>
#include <torch/utils.h>

#include <ATen/ATen.h>
#include <c10/util/irange.h>

#include <cmath>
#include <functional>

namespace dd
{
  RAdamOptions::RAdamOptions(double lr) : lr_(lr)
  {
  }

  bool operator==(const RAdamOptions &lhs, const RAdamOptions &rhs)
  {
    return (lhs.lr() == rhs.lr())
           && (std::get<0>(lhs.betas()) == std::get<0>(rhs.betas()))
           && (std::get<1>(lhs.betas()) == std::get<1>(rhs.betas()))
           && (lhs.eps() == rhs.eps())
           && (lhs.weight_decay() == rhs.weight_decay());
  }

  void RAdamOptions::serialize(torch::serialize::OutputArchive &archive) const
  {
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(lr);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(betas);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(eps);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(weight_decay);
  }

  void RAdamOptions::serialize(torch::serialize::InputArchive &archive)
  {
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, lr);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(betas_t, betas);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, eps);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, weight_decay);
  }

  double RAdamOptions::get_lr() const
  {
    return lr();
  }

  void RAdamOptions::set_lr(const double lr)
  {
    this->lr(lr);
  }

  bool operator==(const RAdamParamState &lhs, const RAdamParamState &rhs)
  {
    return (lhs.step() == rhs.step())
           && torch::equal(lhs.exp_avg(), rhs.exp_avg())
           && torch::equal(lhs.exp_avg_sq(), rhs.exp_avg_sq());
  }

  void
  RAdamParamState::serialize(torch::serialize::OutputArchive &archive) const
  {
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(step);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(exp_avg);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(exp_avg_sq);
  }

  void RAdamParamState::serialize(torch::serialize::InputArchive &archive)
  {
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(int64_t, step);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(torch::Tensor, exp_avg);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(torch::Tensor, exp_avg_sq);
  }

  torch::Tensor RAdam::step(LossClosure closure)
  {
    torch::NoGradGuard no_grad;
    torch::Tensor loss = {};
    if (closure != nullptr)
      {
        at::AutoGradMode enable_grad(true);
        loss = closure();
      }
    for (auto &group : param_groups_)
      {
        for (auto &p : group.params())
          {
            if (!p.grad().defined())
              {
                continue;
              }
            auto grad = p.grad();
            TORCH_CHECK(
                !grad.is_sparse(), "RAdam does not support sparse gradients" /*, please consider SparseRAdam instead*/);
            auto param_state
                = state_.find(c10::guts::to_string(p.unsafeGetTensorImpl()));
            auto &options = static_cast<RAdamOptions &>(group.options());

            // State initialization
            if (param_state == state_.end())
              {
                auto state = std::make_unique<RAdamParamState>();
                state->step(0);
                // Exponential moving average of gradient values
                state->exp_avg(
                    torch::zeros_like(p, torch::MemoryFormat::Preserve));
                // Exponential moving average of squared gradient values
                state->exp_avg_sq(
                    torch::zeros_like(p, torch::MemoryFormat::Preserve));
                state_[c10::guts::to_string(p.unsafeGetTensorImpl())]
                    = std::move(state);
              }

            auto &state = static_cast<RAdamParamState &>(
                *state_[c10::guts::to_string(p.unsafeGetTensorImpl())]);
            auto &exp_avg = state.exp_avg();
            auto &exp_avg_sq = state.exp_avg_sq();

            state.step(state.step() + 1);
            auto beta1 = std::get<0>(options.betas());
            auto beta2 = std::get<1>(options.betas());

            auto bias_correction1 = 1 - std::pow(beta1, state.step());
            auto bias_correction2 = 1 - std::pow(beta2, state.step());

            if (options.weight_decay() != 0)
              {
                grad = grad.add(p, options.weight_decay());
              }

            // Decay the first and second moment running average coefficient
            exp_avg.mul_(beta1).add_(grad, 1 - beta1);
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, 1 - beta2);

            auto N_sma_max = 2.0 / (1.0 - beta2) - 1.0; // rho_inf
            auto beta2_t = std::pow(beta2, (float)state.step());
            auto N_sma
                = N_sma_max
                  - 2.0 * (float)state.step() * beta2_t / bias_correction2;
            if (N_sma >= 5.0)
              {
                auto step_size
                    = sqrt((1.0 - beta2_t) * (N_sma - 4.0) * (N_sma - 2.0)
                           * N_sma_max / (N_sma_max - 4.0) / (N_sma_max - 2.0)
                           / N_sma)
                      / bias_correction1;
                torch::Tensor denom = exp_avg_sq.add_(options.eps()).sqrt();
                auto perturb = exp_avg / denom;
                step_size *= options.lr();
                p.add_(perturb, -step_size);
              }
            else
              {
                auto step_size = options.lr() / bias_correction1;
                auto perturb = exp_avg;
                p.add_(perturb, -step_size);
              }
          }
      }
    return loss;
  }

  void RAdam::save(torch::serialize::OutputArchive &archive) const
  {
    serialize(*this, archive);
  }

  void RAdam::load(torch::serialize::InputArchive &archive)
  {
    torch::IValue pytorch_version;
    if (archive.try_read("pytorch_version", pytorch_version))
      {
        serialize(*this, archive);
      }
    else
      { // deserializing archives saved in old format (prior to
        // version 1.5.0)
        TORCH_WARN("Your serialized RAdam optimizer is still using the old "
                   "serialization format. "
                   "You should re-save your RAdam optimizer to use the new "
                   "serialization format.");
      }
  }
} // namespace dd
