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

#ifndef TORCH_SOLVER_H
#define TORCH_SOLVER_H

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <torch/torch.h>
#pragma GCC diagnostic pop

#include "apidata.h"
#include "torchmodule.h"

namespace dd
{

  class TorchSolver
  {
  public:
    TorchSolver(std::shared_ptr<spdlog::logger> logger) : _logger(logger)
    {
    }
    void configure(APIData ad_solver);
    void create(TorchModule &module);

    int load(std::string sstate);
    void save(std::string sfile);

    void zero_grad()
    {
      _optimizer->zero_grad();
    }

    void step()
    {
      _optimizer->step();
    }

    double base_lr()
    {
      return _base_lr;
    }

  protected:
    std::string _solver_type = "SGD";
    double _base_lr = 0.0001;
    double _beta1 = 0.9;
    double _beta2 = 0.999;
    bool _rectified = true;
    bool _lookahead = true;
    bool _adabelief = false;
    bool _gc = false;
    int _lsteps = 5;
    double _lalpha = 0.5;
    double _weight_decay = 0.0;
    bool _decoupled_wd = false;

    std::shared_ptr<spdlog::logger> _logger; /**< mllib logger. */

    std::unique_ptr<torch::optim::Optimizer> _optimizer;
  };
}
#endif
