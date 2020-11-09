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

  /**
   * \brief this class is a wrapper around torch native solvers/optimizer and
   * our own versions
   */
  class TorchSolver
  {
  public:
    /**
     * \brief simple constructor
     */
    TorchSolver(std::shared_ptr<spdlog::logger> logger) : _logger(logger)
    {
    }

    /**
     * \brief configure solver from api data
     */
    void configure(APIData ad_solver);

    /**
     * \brief allocates solver for real
     */
    void create(TorchModule &module);

    /**
     * \brief reload solver state
     */
    int load(std::string sstate, torch::Device device);

    /**
     * \brief dump solver state
     */
    void save(std::string sfile);

    /**
     * \brief zero_grad() indirection in order to mimic native optimizer
     * behavior
     */
    void zero_grad()
    {
      _optimizer->zero_grad();
    }

    /**
     * \brief step() indirection in order to mimic native optimizer behavior
     */
    void step()
    {
      _optimizer->step();
    }

    /**
     * \brief get base lr for logging purposes
     */
    double base_lr()
    {
      return _base_lr;
    }

  protected:
    std::string _solver_type
        = "SGD"; /**< id of solver in {SGD, ADAM, RMSPROP, ADAGRAD, RANGER}*/
    double _base_lr = 0.0001; /**< base learning rate*/
    double _beta1 = 0.9;      /**< for ADAM and RANGER : beta1 param */
    double _beta2 = 0.999;    /**< for ADAM and RANGER : beta2 param */
    bool _rectified = true; /**< for RANGER : use rectified version, ie RADAM*/
    bool _lookahead = true; /**< for RANGER : use hinton's lookahead */
    bool _adabelief = false; /**< for RANGER : use  ADABELIEF version */
    bool _gc = false;        /**< for RANGER : use gradient centralization */
    int _lsteps
        = 5; /**< for RANGER, if lookahead: number of lookahead steps */
    double _lalpha = 0.5; /**< for RANGER, if lookahead: weight of lookahead */
    double _weight_decay = 0.0; /**< weight decay value*/
    bool _decoupled_wd = false; /**< for RANGER : use decoupled weight decay,
                                   NOT YET IMPLEMENTED */

    std::shared_ptr<spdlog::logger> _logger; /**< mllib logger. */

    std::unique_ptr<torch::optim::Optimizer>
        _optimizer; /**< the real opitmizer */
  };
}
#endif
