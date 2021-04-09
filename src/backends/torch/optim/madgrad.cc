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

#include "./madgrad.h"
#include "mllibstrategy.h"
#include <torch/csrc/autograd/variable.h>
#include <torch/nn/module.h>
#include <torch/serialize/archive.h>
#include <torch/utils.h>

#include <ATen/ATen.h>

#include <cmath>
#include <functional>

namespace dd
{

  MadgradOptions::MadgradOptions(double lr) : lr_(lr)
  {
  }

  bool operator==(const MadgradOptions &lhs, const MadgradOptions &rhs)
  {
    return (lhs.lr() == rhs.lr()) && (lhs.eps() == rhs.eps())
           && (lhs.weight_decay() == rhs.weight_decay())
           && (lhs.momentum() == rhs.momentum()) && (lhs.eps() == rhs.eps())
           && (lhs.lookahead() == rhs.lookahead())
           && (lhs.lsteps() == rhs.lsteps()) && (lhs.lalpha() == rhs.lalpha())
           && (lhs.swa() == rhs.swa());
  }

  void
  MadgradOptions::serialize(torch::serialize::OutputArchive &archive) const
  {
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(lr);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(eps);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(weight_decay);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(momentum);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(lookahead);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(lsteps);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(lalpha);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(swa);
  }

  void MadgradOptions::serialize(torch::serialize::InputArchive &archive)
  {
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, lr);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, eps);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, momentum);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, weight_decay);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(bool, lookahead);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(int, lsteps);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, lalpha);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(bool, swa);
  }

  bool operator==(const MadgradParamState &lhs, const MadgradParamState &rhs)
  {
    return ((lhs.k() == rhs.k()) && torch::equal(lhs.s(), rhs.s())
            && torch::equal(lhs.grad_sum_sq(), rhs.grad_sum_sq())
            && torch::equal(lhs.x0(), rhs.x0())
            && torch::equal(lhs.slow_buffer(), rhs.slow_buffer())
            && torch::equal(lhs.swa_buffer(), rhs.swa_buffer()));
  }

  void
  MadgradParamState::serialize(torch::serialize::OutputArchive &archive) const
  {
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(k);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(s);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(grad_sum_sq);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(x0);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(slow_buffer);
    _TORCH_OPTIM_SERIALIZE_TORCH_ARG(swa_buffer);
  }

  void MadgradParamState::serialize(torch::serialize::InputArchive &archive)
  {
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(int64_t, k);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(torch::Tensor, s);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(torch::Tensor, grad_sum_sq);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(torch::Tensor, x0);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(torch::Tensor, slow_buffer);
    _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(torch::Tensor, swa_buffer);
  }

  torch::Tensor Madgrad::step(LossClosure closure)
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
                        "Madgrad does not support sparse gradients");
            auto param_state
                = state_.find(c10::guts::to_string(p.unsafeGetTensorImpl()));
            auto &options = static_cast<MadgradOptions &>(group.options());

            // State initialization
            if (param_state == state_.end())
              {
                auto state = std::make_unique<MadgradParamState>();
                state->k(0);
                state->s(torch::zeros_like(p.data(),
                                           torch::MemoryFormat::Preserve));
                state->grad_sum_sq(torch::zeros_like(
                    p.data(), torch::MemoryFormat::Preserve));
                if (options.momentum() != 0.0)
                  state->x0(torch::clone(p.data()));
                state->slow_buffer(torch::empty_like(
                    p.data(), torch::MemoryFormat::Preserve));
                state->slow_buffer().copy_(p.data());
                state_[c10::guts::to_string(p.unsafeGetTensorImpl())]
                    = std::move(state);
                if (options.swa())
                  state->swa_buffer(torch::zeros_like(
                      p.data(), torch::MemoryFormat::Preserve));
              }

            auto &state = static_cast<MadgradParamState &>(
                *state_[c10::guts::to_string(p.unsafeGetTensorImpl())]);

            auto &grad_sum_sq = state.grad_sum_sq();
            auto &s = state.s();

            double lamb = options.lr() * pow(double(state.k() + 1), 0.5);
            double ck = 1.0 - options.momentum();
            state.k(state.k() + 1);

            if (options.weight_decay() != 0.0)
              grad.add_(p.data(), options.weight_decay());

            if (grad.is_sparse())
              throw MLLibInternalException(
                  "madgrad for sparse gradient no implemented yet");

            torch::Tensor x0;
            if (options.momentum() == 0.0)
              {
                torch::Tensor rms
                    = grad_sum_sq.pow(1.0 / 3.0).add_(options.eps());
                x0 = p.data().addcdiv(s, rms, 1.0);
              }
            else
              {
                x0 = state.x0();
              }
            grad_sum_sq.addcmul_(grad, grad, lamb);
            torch::Tensor rms = grad_sum_sq.pow(1.0 / 3.0).add_(options.eps());

            s.data().add_(grad, lamb);

            if (options.momentum() == 0.0)
              {
                p.data().copy_(x0.addcdiv(s, rms, -1.0));
              }
            else
              {
                torch::Tensor z = x0.addcdiv(s, rms, -1.0);
                p.data().mul_(1.0 - ck).add_(z, ck);
              }

            if (state.k() % options.lsteps() == 0 && options.lookahead())
              {
                auto slow_p = state.slow_buffer();
                slow_p.add_(p.data() - slow_p, options.lalpha());
                p.data().copy_(slow_p);
              }

            if (options.swa())
              {
                auto &swa_buf = state.swa_buffer();
                double swa_decay = 1.0 / (state.k() + 1);
                torch::Tensor diff = (p.data() - swa_buf) * swa_decay;
                swa_buf.add_(diff);
              }
          }
      }
    return loss;
  }

  void Madgrad::swap_swa_sgd()
  {
    for (auto &group : param_groups_)
      {
        auto &options = static_cast<MadgradOptions &>(group.options());
        if (!options.swa())
          continue;
        for (auto &p : group.params())
          {
            auto &state = static_cast<MadgradParamState &>(
                *state_[c10::guts::to_string(p.unsafeGetTensorImpl())]);
            auto &swa_buf = state.swa_buffer();

            auto tmp = torch::empty_like(p.data());
            tmp.copy_(p.data());
            p.data().copy_(swa_buf);
            swa_buf.copy_(tmp);
          }
      }
  }

  void Madgrad::save(torch::serialize::OutputArchive &archive) const
  {
    serialize(*this, archive);
  }

  void Madgrad::load(torch::serialize::InputArchive &archive)
  {
    torch::IValue pytorch_version;
    if (archive.try_read("pytorch_version", pytorch_version))
      {
        serialize(*this, archive);
      }
    else
      { // deserializing archives saved in old format (prior to
        // version 1.5.0)
        TORCH_WARN("Your serialized Madgrad optimizer is still using the old "
                   "serialization format. "
                   "You should re-save your Madgrad optimizer to use the new "
                   "serialization format.");
      }
  }
} // namespace dd
