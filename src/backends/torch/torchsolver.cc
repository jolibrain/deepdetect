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

#include "torchsolver.h"
#include "optim/ranger.h"

namespace dd
{
  void TorchSolver::configure(APIData ad_solver)
  {
    if (ad_solver.has("solver_type"))
      _solver_type = ad_solver.get("solver_type").get<std::string>();

    if (_solver_type == "RANGER" || _solver_type == "RANGER_PLUS")
      _clip = true;

    if (_solver_type == "RANGER_PLUS")
      {
        _adabelief = true;
        _gc = true;
      }

    if (ad_solver.has("base_lr"))
      _base_lr = ad_solver.get("base_lr").get<double>();
    if (ad_solver.has("beta1"))
      _beta1 = ad_solver.get("beta1").get<double>();
    if (ad_solver.has("beta"))
      _beta2 = ad_solver.get("beta2").get<double>();
    if (ad_solver.has("clip"))
      _clip = ad_solver.get("clip").get<bool>();
    if (ad_solver.has("clip_value"))
      _clip_value = ad_solver.get("clip_value").get<double>();
    if (ad_solver.has("clip_norm"))
      _clip_norm = ad_solver.get("clip_norm").get<double>();
    if (ad_solver.has("rectified"))
      _rectified = ad_solver.get("rectified").get<bool>();
    if (ad_solver.has("lookahead"))
      _lookahead = ad_solver.get("lookahead").get<bool>();
    if (ad_solver.has("adabelief"))
      _adabelief = ad_solver.get("adabelief").get<bool>();
    if (ad_solver.has("gradient_centralization"))
      _gc = ad_solver.get("gradient_centralization").get<bool>();
    if (ad_solver.has("lookahead_steps"))
      _lsteps = ad_solver.get("lookahead_steps").get<int>();
    if (ad_solver.has("lookahead_alpha"))
      _lalpha = ad_solver.get("lookahead_alpha").get<double>();
    if (ad_solver.has("weight_decay"))
      _weight_decay = ad_solver.get("weight_decay").get<double>();
    if (ad_solver.has("decoupled_wd"))
      _decoupled_wd = ad_solver.get("decoupled_wd").get<bool>();
  }

  void TorchSolver::create(TorchModule &module)
  {
    this->_logger->info("Selected solver type: {}", _solver_type);

    _params = module.parameters();

    if (_solver_type == "ADAM")
      {
        _optimizer
            = std::unique_ptr<torch::optim::Optimizer>(new torch::optim::Adam(
                _params, torch::optim::AdamOptions(_base_lr)
                             .betas(std::make_tuple(_beta1, _beta2))
                             .weight_decay(_weight_decay)));
        this->_logger->info("base_lr: {}", _base_lr);
      }
    else if (_solver_type == "RMSPROP")
      {
        _optimizer = std::unique_ptr<torch::optim::Optimizer>(
            new torch::optim::RMSprop(
                _params, torch::optim::RMSpropOptions(_base_lr).weight_decay(
                             _weight_decay)));
        this->_logger->info("base_lr: {}", _base_lr);
      }
    else if (_solver_type == "ADAGRAD")
      {
        _optimizer = std::unique_ptr<torch::optim::Optimizer>(
            new torch::optim::Adagrad(
                _params, torch::optim::AdagradOptions(_base_lr).weight_decay(
                             _weight_decay)));
        this->_logger->info("base_lr: {}", _base_lr);
      }
    else if (_solver_type == "RANGER" || _solver_type == "RANGER_PLUS")
      {
        _optimizer = std::unique_ptr<torch::optim::Optimizer>(
            new Ranger(_params, RangerOptions(_base_lr)
                                    .betas(std::make_tuple(_beta1, _beta2))
                                    .weight_decay(_weight_decay)
                                    .decoupled_wd(_decoupled_wd)
                                    .rectified(_rectified)
                                    .lookahead(_lookahead)
                                    .adabelief(_adabelief)
                                    .gradient_centralization(_gc)
                                    .lsteps(_lsteps)
                                    .lalpha(_lalpha)));
        this->_logger->info("base_lr: {}", _base_lr);
        this->_logger->info("beta_1: {}", _beta1);
        this->_logger->info("beta_2: {}", _beta2);
        this->_logger->info("weight_decay: {}", _weight_decay);
        this->_logger->info("rectified: {}", _rectified);
        this->_logger->info("lookahead: {}", _lookahead);
        this->_logger->info("adabelief: {}", _adabelief);
        this->_logger->info("gradient_centralization: {}", _gc);
        if (_lookahead)
          {
            this->_logger->info("lookahead steps: {}", _lsteps);
            this->_logger->info("lookahead alpha: {}", _lalpha);
          }
      }
    else
      {
        if (_solver_type != "SGD")
          this->_logger->warn("Solver type {} not found, using SGD",
                              _solver_type);
        _optimizer
            = std::unique_ptr<torch::optim::Optimizer>(new torch::optim::SGD(
                _params, torch::optim::SGDOptions(_base_lr)));
        this->_logger->info("base_lr: {}", _base_lr);
      }
    this->_logger->info("clip: {}", _clip);
    if (_clip)
      {
        if (_clip_value >= 0.0)
          this->_logger->info("clip_value: {}", _clip_value);
        if (_clip_norm >= 0.0)
          this->_logger->info("clip_norm: {}", _clip_norm);
        if (_clip_norm < 0.0 && _clip_value < 0.0)
          {
            this->_logger->warn(
                "Gradient clipping selected, but no value given, "
                "will clip gradient value to {} and norm to {}",
                DEFAULT_CLIP_VALUE, DEFAULT_CLIP_NORM);
          }
      }
  }

  void TorchSolver::step()
  {
    if (_clip)
      {
        if (_clip_value > 0.0)
          torch::nn::utils::clip_grad_value_(_params, _clip_value);
        if (_clip_norm > 0.0)
          torch::nn::utils::clip_grad_norm_(_params, _clip_norm);
        if (_clip_value <= 0.0 && _clip_norm <= 0.0)
          {
            torch::nn::utils::clip_grad_value_(_params, DEFAULT_CLIP_VALUE);
            torch::nn::utils::clip_grad_norm_(_params, DEFAULT_CLIP_NORM);
          }
      }
    _optimizer->step();
  }

  void TorchSolver::save(std::string sfile)
  {
    torch::save(*_optimizer, sfile);
  }

  int TorchSolver::load(std::string sstate, torch::Device device)
  {
    if (!sstate.empty())
      {
        _logger->info("Reload solver from {}", sstate);
        size_t start = sstate.rfind("-") + 1;
        size_t end = sstate.rfind(".");
        int it = std::stoi(sstate.substr(start, end - start));
        _logger->info("Restarting optimization from iter {}", it);
        _logger->info("loading " + sstate);
        try
          {
            torch::load(*_optimizer, sstate, device);
          }
        catch (std::exception &e)
          {
            this->_logger->error("unable to load " + sstate);
            throw;
          }
        return it;
      }
    return 0;
  }
}
