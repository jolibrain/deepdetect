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
#pragma once

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
  struct TORCH_API RAdamOptions
      : public torch::optim::OptimizerCloneableOptions<RAdamOptions>
  {
    RAdamOptions(double lr = 1e-3);
    TORCH_ARG(double, lr) = 1e-3;
    typedef std::tuple<double, double> betas_t;
    TORCH_ARG(betas_t, betas) = std::make_tuple(0.9, 0.999);
    TORCH_ARG(double, eps) = 1e-8;
    TORCH_ARG(double, weight_decay) = 0;

  public:
    void serialize(torch::serialize::InputArchive &archive) override;
    void serialize(torch::serialize::OutputArchive &archive) const override;
    TORCH_API friend bool operator==(const RAdamOptions &lhs,
                                     const RAdamOptions &rhs);
    ~RAdamOptions() override = default;
    double get_lr() const override;
    void set_lr(const double lr) override;
  };

  struct TORCH_API RAdamParamState
      : public torch::optim::OptimizerCloneableParamState<RAdamParamState>
  {
    TORCH_ARG(int64_t, step) = 0;
    TORCH_ARG(torch::Tensor, exp_avg);
    TORCH_ARG(torch::Tensor, exp_avg_sq);
    TORCH_ARG(torch::Tensor, max_exp_avg_sq) = {};

  public:
    void serialize(torch::serialize::InputArchive &archive) override;
    void serialize(torch::serialize::OutputArchive &archive) const override;
    TORCH_API friend bool operator==(const RAdamParamState &lhs,
                                     const RAdamParamState &rhs);
    ~RAdamParamState() override = default;
  };

  class TORCH_API RAdam : public torch::optim::Optimizer
  {
  public:
    explicit RAdam(std::vector<torch::optim::OptimizerParamGroup> param_groups,
                   RAdamOptions defaults = {})
        : Optimizer(std::move(param_groups),
                    std::make_unique<RAdamOptions>(defaults))
    {
      TORCH_CHECK(defaults.lr() >= 0,
                  "Invalid learning rate: ", defaults.lr());
      TORCH_CHECK(defaults.eps() >= 0,
                  "Invalid epsilon value: ", defaults.eps());
      auto betas = defaults.betas();
      TORCH_CHECK(0 <= std::get<0>(betas) && std::get<0>(betas) < 1.0,
                  "Invalid beta parameter at index 0: ", std::get<0>(betas));
      TORCH_CHECK(0 <= std::get<1>(betas) && std::get<1>(betas) < 1.0,
                  "Invalid beta parameter at index 1: ", std::get<1>(betas));
      TORCH_CHECK(defaults.weight_decay() >= 0,
                  "Invalid weight_decay value: ", defaults.weight_decay());
    }
    explicit RAdam(std::vector<torch::Tensor> params,
                   // NOLINTNEXTLINE(performance-move-const-arg)
                   RAdamOptions defaults = {})
        : RAdam({ std::move(torch::optim::OptimizerParamGroup(params)) },
                defaults)
    {
    }

    torch::Tensor step(LossClosure closure = nullptr) override;
    void save(torch::serialize::OutputArchive &archive) const override;
    void load(torch::serialize::InputArchive &archive) override;

  private:
    template <typename Self, typename Archive>
    static void serialize(Self &self, Archive &archive)
    {
      _TORCH_OPTIM_SERIALIZE_WITH_TEMPLATE_ARG(RAdam);
    }
  };
} // namespace dd
