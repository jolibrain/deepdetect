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

/*this is largely inspired/adapted from adam torch/c++ implementation, ie
 * pytorch/torch/csrc/api/src/optim/adam.cpp */

#include "./ranger.h"
#include <torch/csrc/autograd/variable.h>
#include <torch/nn/module.h>
#include <torch/serialize/archive.h>
#include <torch/utils.h>

#include <ATen/ATen.h>

#include <cmath>
#include <functional>

namespace dd
{

  RangerOptions::RangerOptions(double lr) : lr_(lr)
  {
  }

  bool operator==(const RangerOptions &lhs, const RangerOptions &rhs)
  {
    return (lhs.lr() == rhs.lr())
           && (std::get<0>(lhs.betas()) == std::get<0>(rhs.betas()))
           && (std::get<1>(lhs.betas()) == std::get<1>(rhs.betas()))
           && (lhs.eps() == rhs.eps())
           && (lhs.weight_decay() == rhs.weight_decay())
           && (lhs.rectified() == rhs.rectified())
           && (lhs.decoupled_wd() == rhs.decoupled_wd())
           && (lhs.lookahead() == rhs.lookahead())
           && (lhs.adabelief() == rhs.adabelief())
           && (lhs.gradient_centralization() == rhs.gradient_centralization())
           && (lhs.lsteps() == rhs.lsteps()) && (lhs.lalpha() == rhs.lalpha());
  }

  void RangerOptions::serialize(torch::serialize::OutputArchive &archive) const
  {
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(lr);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(betas);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(eps);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(weight_decay);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(decoupled_wd);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(rectified);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(lookahead);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(adabelief);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(gradient_centralization);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(lsteps);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(lalpha);
  }

  void RangerOptions::serialize(torch::serialize::InputArchive &archive)
  {
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, lr);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(betas_t, betas);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, eps);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, weight_decay);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(bool, decoupled_wd);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(bool, rectified);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(bool, lookahead);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(bool, adabelief);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(bool, gradient_centralization);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(int, lsteps);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, lalpha);
  }

  bool operator==(const RangerParamState &lhs, const RangerParamState &rhs)
  {
    return ((lhs.step() == rhs.step())
            && torch::equal(lhs.exp_avg(), rhs.exp_avg())
            && torch::equal(lhs.exp_avg_sq(), rhs.exp_avg_sq())
            && torch::equal(lhs.slow_buffer(), rhs.slow_buffer()));
  }

  void
  RangerParamState::serialize(torch::serialize::OutputArchive &archive) const
  {
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(step);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(exp_avg);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(exp_avg_sq);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(slow_buffer);
  }

  void RangerParamState::serialize(torch::serialize::InputArchive &archive)
  {
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(int64_t, step);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(torch::Tensor, exp_avg);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(torch::Tensor, exp_avg_sq);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(torch::Tensor, slow_buffer);
  }

  torch::Tensor Ranger::step(LossClosure closure)
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

            TORCH_CHECK(!grad.is_sparse(),
                        "Ranger does not support sparse gradients");
            auto param_state
                = state_.find(c10::guts::to_string(p.unsafeGetTensorImpl()));
            auto &options = static_cast<RangerOptions &>(group.options());

            // State initialization
            if (param_state == state_.end())
              {
                auto state = std::make_unique<RangerParamState>();
                state->step(0);
                // Exponential moving average of gradient values
                state->exp_avg(
                    torch::zeros_like(p, torch::MemoryFormat::Preserve));
                // Exponential moving average of squared gradient values
                state->exp_avg_sq(
                    torch::zeros_like(p, torch::MemoryFormat::Preserve));
                state->slow_buffer(
                    torch::empty_like(p, torch::MemoryFormat::Preserve));
                state->slow_buffer().copy_(p.data());
                state_[c10::guts::to_string(p.unsafeGetTensorImpl())]
                    = std::move(state);
              }

            auto &state = static_cast<RangerParamState &>(
                *state_[c10::guts::to_string(p.unsafeGetTensorImpl())]);
            auto &exp_avg = state.exp_avg();
            auto &exp_avg_sq = state.exp_avg_sq();

            state.step(state.step() + 1);
            auto beta1 = std::get<0>(options.betas());
            auto beta2 = std::get<1>(options.betas());

            auto bias_correction1 = 1.0 - std::pow(beta1, state.step());
            auto bias_correction2 = 1.0 - std::pow(beta2, state.step());

            if (options.weight_decay() != 0) // weight decay not decoupled !!
              grad = grad.add(p, options.weight_decay());

            if (options.gradient_centralization())
              {

                std::vector<long int> dim;
                for (long int i = 1; i < grad.dim(); ++i)
                  dim.push_back(i);
                grad.add_(-grad.mean(torch::IntArrayRef(dim), true));
              }

            exp_avg.mul_(beta1).add_(grad, 1 - beta1); // m_t

            if (options.adabelief())
              exp_avg_sq.mul_(beta2).addcmul_(grad - exp_avg, grad - exp_avg,
                                              1 - beta2); // v_t
            else
              exp_avg_sq.mul_(beta2).addcmul_(grad, grad, 1 - beta2); // v_t
            torch::Tensor denom;
            if (!options.rectified())
              {
                denom = (exp_avg_sq.sqrt() / sqrt(bias_correction2))
                            .add_(options.eps());

                auto step_size = options.lr() / bias_correction1;
                p.addcdiv_(exp_avg, denom, -step_size);
              }
            else
              {
                auto N_sma_max = 2.0 / (1.0 - beta2) - 1.0; // rho_inf
                auto beta2_t = std::pow(beta2, state.step());
                auto N_sma
                    = N_sma_max
                      - 2.0 * (float)state.step() * beta2_t / bias_correction2;
                double step_size;
                if (N_sma >= 5)
                  {
                    step_size = sqrt((1.0 - beta2_t) * (N_sma - 4.0)
                                     / (N_sma_max - 4.0) * (N_sma - 2.0)
                                     / N_sma * N_sma_max / (N_sma_max - 2.0))
                                / bias_correction1;
                    if (options.adabelief())
                      denom = exp_avg_sq.add_(options.eps())
                                  .sqrt()
                                  .add_(options.eps());
                    else
                      denom = exp_avg_sq.sqrt().add_(options.eps());
                    p.addcdiv_(exp_avg, denom, -step_size * options.lr());
                  }
                else
                  {
                    step_size = 1.0 / bias_correction1;
                    p.add_(exp_avg, -step_size * options.lr());
                  }
              }
            if (state.step() % options.lsteps() == 0 && options.lookahead())
              {
                auto slow_p = state.slow_buffer();
                slow_p.add_(p.data() - slow_p, options.lalpha());
                p.data().copy_(slow_p);
              }
          }
      }
    return loss;
  }

  void Ranger::save(torch::serialize::OutputArchive &archive) const
  {
    serialize(*this, archive);
  }

  void Ranger::load(torch::serialize::InputArchive &archive)
  {
    torch::IValue pytorch_version;
    if (archive.try_read("pytorch_version", pytorch_version))
      {
        serialize(*this, archive);
      }
    else
      { // deserializing archives saved in old format (prior to
        // version 1.5.0)
        TORCH_WARN("Your serialized Ranger optimizer is still using the old "
                   "serialization format. "
                   "You should re-save your Ranger optimizer to use the new "
                   "serialization format.");
      }
  }
} // namespace dd
