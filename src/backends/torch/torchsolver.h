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
#include "torchloss.h"
#include "optim/ranger.h"

#define DEFAULT_CLIP_VALUE 5.0
#define DEFAULT_CLIP_NORM 100.0
#define DEFAULT_SAM_RHO 0.05

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
    TorchSolver(TorchModule &module, TorchLoss &loss,
                std::shared_ptr<spdlog::logger> logger)
        : _module(module), _tloss(loss), _logger(logger)
    {
    }

    /**
     * \brief configure solver from api data
     */
    void configure(APIData ad_solver);

    /**
     * \brief reload solver state
     */
    int load(std::string sstate, torch::Device device);

    /**
     * \brief dump solver state
     */
    void save(std::string sfile);

    /**
     * \brief restore solver state, checks solverstate presence  and returns
     * iteration number, best metric value and corresponding iteration number
     * according to ad_mllib.resume param
     */

    int resume(const APIData &ad_mllib, const TorchModel &mlmodel,
               const torch::Device &main_device,
               std::vector<double> &best_metric_values,
               std::vector<int64_t> &best_iteration_number,
               const std::vector<std::string> &set_names);

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
     * also applies gradient clipping if asked for
     */
    void step();

    /**
     * \brief get base lr for logging purposes
     */
    double base_lr()
    {
      return _base_lr;
    }

    void eval()
    {
      swap_swa_sgd();
    }

    void train()
    {
      swap_swa_sgd();
    }

  protected:
    /**
     * \brief allocates solver for real
     */
    void create();
    void real_step();
    void override_options();
    void sam_first_step();
    void sam_second_step();

    void swap_swa_sgd()
    {
      if (_swa)
        (reinterpret_cast<Ranger *>(_optimizer.get()))->swap_swa_sgd();
    }

    std::vector<torch::Tensor> _sam_ew;

    std::vector<at::Tensor> _params; /**< list of parameter to optimize,
                   storing it here for gradient clipping */

    std::string _solver_type
        = "SGD"; /**< id of solver in {SGD, ADAM, RMSPROP, ADAGRAD, RANGER}*/
    double _base_lr = 0.0001; /**< base learning rate*/
    double _momentum = 0.9;   /**< momentum for madgrad*/
    double _beta1 = 0.9;      /**< for ADAM and RANGER : beta1 param */
    double _beta2 = 0.999;    /**< for ADAM and RANGER : beta2 param */
    bool _rectified = true; /**< for RANGER : use rectified version, ie RADAM*/
    bool _lookahead = true; /**< for RANGER : use hinton's lookahead */
    bool _adabelief = false; /**< for RANGER : use  ADABELIEF version */
    bool _gc = false;        /**< for RANGER : use gradient centralization */
    bool _adamp = false;     /**< for RANGER : ADAMP variant */
    int _lsteps
        = 5; /**< for RANGER, if lookahead: number of lookahead steps */
    double _lalpha = 0.5; /**< for RANGER, if lookahead: weight of lookahead */
    bool _clip = false;   /**<  clip gradients */
    double _clip_value = -1.0;  /**< value to clip gradients to */
    double _clip_norm = -1.0;   /**<  norm to clip gradients to */
    double _weight_decay = 0.0; /**< weight decay value*/
    bool _decoupled_wd = false; /**< for RANGER : use decoupled weight decay,
                                   NOT YET IMPLEMENTED */
    bool _sam = false;
    double _sam_rho = DEFAULT_SAM_RHO;

    bool _swa = false; /**< stochastic weights averaging 1803.05407 */

    TorchModule &_module;
    TorchLoss &_tloss;
    std::shared_ptr<spdlog::logger> _logger; /**< mllib logger. */

    std::unique_ptr<torch::optim::Optimizer>
        _optimizer; /**< the real opitmizer */
  };
}
#endif
