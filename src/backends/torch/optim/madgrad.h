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

#ifndef MADGRAD_H
#define MADGRAD_H

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <torch/arg.h>
#include <torch/nn/module.h>
#include <torch/optim/optimizer.h>
#include <torch/optim/serialize.h>
#pragma GCC diagnostic pop

#include <utility>
#include <vector>

namespace torch
{
  namespace serialize
  {
    class OutputArchive;
    class InputArchive;
  } // namespace serialize
} // namespace torch

namespace dd
{
  struct TORCH_API MadgradOptions
      : public torch::optim::OptimizerCloneableOptions<MadgradOptions>
  {
    MadgradOptions(double lr = 1e-3);
    TORCH_ARG(double, lr) = 1e-2;
    TORCH_ARG(double, momentum) = 0.9;
    TORCH_ARG(double, weight_decay) = 0.0;
    TORCH_ARG(double, eps) = 1e-8;

    TORCH_ARG(bool, lookahead) = true;
    TORCH_ARG(int, lsteps) = 6;
    TORCH_ARG(double, lalpha) = 0.5;
    TORCH_ARG(bool, swa) = false;

  public:
    void serialize(torch::serialize::InputArchive &archive) override;
    void serialize(torch::serialize::OutputArchive &archive) const override;
    TORCH_API friend bool operator==(const MadgradOptions &lhs,
                                     const MadgradOptions &rhs);
    ~MadgradOptions() = default;
  };

  struct TORCH_API MadgradParamState
      : public torch::optim::OptimizerCloneableParamState<MadgradParamState>
  {
    TORCH_ARG(int64_t, k) = 0;
    TORCH_ARG(torch::Tensor, s);
    TORCH_ARG(torch::Tensor, grad_sum_sq);
    TORCH_ARG(torch::Tensor, x0);
    TORCH_ARG(torch::Tensor, slow_buffer);
    TORCH_ARG(torch::Tensor, swa_buffer);

  public:
    void serialize(torch::serialize::InputArchive &archive) override;
    void serialize(torch::serialize::OutputArchive &archive) const override;
    TORCH_API friend bool operator==(const MadgradParamState &lhs,
                                     const MadgradParamState &rhs);
    ~MadgradParamState() = default;
  };

  class TORCH_API Madgrad : public torch::optim::Optimizer
  {
  public:
    explicit Madgrad(
        std::vector<torch::optim::OptimizerParamGroup> param_groups,
        MadgradOptions defaults = {})
        : Optimizer(std::move(param_groups),
                    std::make_unique<MadgradOptions>(defaults))
    {
      TORCH_CHECK(defaults.lr() >= 0,
                  "Invalid learning rate: ", defaults.lr());
      TORCH_CHECK(defaults.momentum() >= 0,
                  "Invalid learning rate: ", defaults.momentum());
      TORCH_CHECK(defaults.eps() >= 0,
                  "Invalid epsilon value: ", defaults.eps());
      TORCH_CHECK(defaults.weight_decay() >= 0,
                  "Invalid weight_decay value: ", defaults.weight_decay());
      TORCH_CHECK(defaults.lsteps() >= 0,
                  "Invalid lookahead steps: ", defaults.lsteps());
      TORCH_CHECK(defaults.lalpha() >= 0,
                  "Invalid lookahead alpha: ", defaults.lalpha());
    }
    explicit Madgrad(std::vector<torch::Tensor> params,
                     MadgradOptions defaults = {})
        : Madgrad({ std::move(torch::optim::OptimizerParamGroup(params)) },
                  defaults)
    {
    }

    torch::Tensor step(LossClosure closure = nullptr) override;
    void save(torch::serialize::OutputArchive &archive) const override;
    void load(torch::serialize::InputArchive &archive) override;

    void swap_swa_sgd();

  private:
    template <typename Self, typename Archive>
    static void serialize(Self &self, Archive &archive)
    {
      _TORCH_OPTIM_SERIALIZE_WITH_TEMPLATE_ARG(Madgrad);
    }
    bool swa_in_params = false;
  };
} // namespace dd

#endif
