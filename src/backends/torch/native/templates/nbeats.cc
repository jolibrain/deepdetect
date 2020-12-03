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

#include "nbeats.h"
#include <cmath>
#include <string>

namespace dd
{
  void NBeats::BlockImpl::init_block()
  {
    _fc1 = register_module(
        "fc1", torch::nn::Linear(_backcast_length * _data_size, _units));
    _fc2 = register_module("fc2", torch::nn::Linear(_units, _units));
    _fc3 = register_module("fc3", torch::nn::Linear(_units, _units));
    _fc4 = register_module("fc4", torch::nn::Linear(_units, _units));
    _theta_f_fc = register_module(
        "theta_f_fc",
        torch::nn::Linear(
            torch::nn::LinearOptions(_units, _thetas_dim).bias(false)));
    if (_share_thetas)
      _theta_b_fc = _theta_f_fc;
    else
      _theta_b_fc = register_module(
          "theta_b_fc",
          torch::nn::Linear(
              torch::nn::LinearOptions(_units, _thetas_dim).bias(false)));
  }

  torch::Tensor NBeats::BlockImpl::first_forward(torch::Tensor x)
  {
    x = x.reshape({ x.size(0), x.size(1) * x.size(2) });
    x = _fc1->forward(x);
    x = torch::relu(x);
    x = torch::relu(_fc2->forward(x));
    x = torch::relu(_fc3->forward(x));
    x = torch::relu(_fc4->forward(x));
    return x;
  }

  torch::Tensor NBeats::BlockImpl::first_extract(torch::Tensor x,
                                                 std::string extract_layer)
  {
    x = x.reshape({ x.size(0), x.size(1) * x.size(2) });
    x = _fc1->forward(x);
    if (extract_layer == "fc1")
      return x;
    x = torch::relu(x);
    x = _fc2->forward(x);
    if (extract_layer == "fc2")
      return x;
    x = torch::relu(x);
    x = _fc3->forward(x);
    if (extract_layer == "fc3")
      return x;
    x = torch::relu(x);
    x = _fc4->forward(x);
    if (extract_layer == "fc4")
      return x;
    x = torch::relu(x);
    return x;
  }

  std::tuple<torch::Tensor, torch::Tensor>
  NBeats::SeasonalityBlockImpl::forward(torch::Tensor x)
  {
    x = BlockImpl::first_forward(x);
    torch::Tensor tbfc = _theta_b_fc->forward(x);
    torch::Tensor backcast = tbfc.mm(_bS);
    torch::Tensor tffc = _theta_f_fc->forward(x);
    torch::Tensor forecast = tffc.mm(_fS);
    return std::make_tuple(
        backcast.reshape({ backcast.size(0), _backcast_length, _data_size }),
        forecast.reshape({ forecast.size(0), _forecast_length, _data_size }));
  }

  torch::Tensor
  NBeats::SeasonalityBlockImpl::extract(torch::Tensor x,
                                        std::string extract_layer)
  {
    x = BlockImpl::first_extract(x, extract_layer);
    if (extract_layer != "theta_b_fc" && extract_layer != "theta_f_fc")
      return x;
    torch::Tensor tbfc = _theta_b_fc->forward(x);
    torch::Tensor backcast = tbfc.mm(_bS);
    if (extract_layer == "theta_b_fc")
      return backcast;

    torch::Tensor tffc = _theta_f_fc->forward(x);
    torch::Tensor forecast = tffc.mm(_fS);
    return forecast;
  }

  std::tuple<torch::Tensor, torch::Tensor>
  NBeats::create_sin_basis(int thetas_dim)
  {
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    unsigned int p = thetas_dim;
    unsigned int p1 = p / 2;
    unsigned int p2 = (p % 2 == 0) ? p / 2 : p / 2 + 1;
    std::vector<float> tdata;

    for (unsigned int i = 0; i < p1; ++i)
      for (unsigned int j = 0; j < _forecast_linspace.size(); ++j)
        for (unsigned int d2 = 0; d2 < _data_size; ++d2)
          tdata.push_back(std::cos(2 * M_PI * i * _forecast_linspace[j]));
    torch::Tensor s1
        = torch::from_blob(
              tdata.data(),
              { p1, static_cast<long int>(_forecast_linspace.size())
                        * _data_size },
              options)
              .clone();

    tdata.clear();
    for (unsigned int i = 0; i < p2; ++i)
      for (unsigned int j = 0; j < _forecast_linspace.size(); ++j)
        for (unsigned int d2 = 0; d2 < _data_size; ++d2)
          tdata.push_back(std::sin(2 * M_PI * i * _forecast_linspace[j]));
    torch::Tensor s2
        = torch::from_blob(
              tdata.data(),
              { p2, static_cast<long int>(_forecast_linspace.size())
                        * _data_size },
              options)
              .clone();
    torch::Tensor fS = register_buffer("fS", torch::cat({ s1, s2 }));

    tdata.clear();
    for (unsigned int i = 0; i < p1; ++i)
      for (unsigned int j = 0; j < _backcast_linspace.size(); ++j)
        for (unsigned int d2 = 0; d2 < _data_size; ++d2)
          tdata.push_back(std::cos(2 * M_PI * i * _backcast_linspace[j]));
    torch::Tensor ss1
        = torch::from_blob(
              tdata.data(),
              { p1, static_cast<long int>(_backcast_linspace.size())
                        * _data_size },
              options)
              .clone();

    tdata.clear();
    for (unsigned int i = 0; i < p2; ++i)
      for (unsigned int j = 0; j < _backcast_linspace.size(); ++j)
        for (unsigned int d2 = 0; d2 < _data_size; ++d2)
          tdata.push_back(std::sin(2 * M_PI * i * _backcast_linspace[j]));
    torch::Tensor ss2
        = torch::from_blob(
              tdata.data(),
              { p2, static_cast<long int>(_backcast_linspace.size())
                        * _data_size },
              options)
              .clone();

    torch::Tensor bS = register_buffer("bS", torch::cat({ ss1, ss2 }));
    return std::make_tuple(bS, fS);
  }

  std::tuple<torch::Tensor, torch::Tensor>
  NBeats::create_exp_basis(int thetas_dim)
  {
    torch::Tensor bT, fT;
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    unsigned int p = thetas_dim;
    std::vector<float> tdata;

    for (unsigned int i = 0; i < p; ++i)
      for (unsigned int j = 0; j < _forecast_linspace.size(); ++j)
        for (unsigned int d2 = 0; d2 < _data_size; ++d2)
          {
            tdata.push_back(static_cast<float>(
                powf(_forecast_linspace[j], static_cast<float>(i))));
            ;
          }
    fT = register_buffer(
        "fT",
        torch::from_blob(tdata.data(),
                         { static_cast<long int>(p),
                           static_cast<long int>(_forecast_linspace.size())
                               * static_cast<long int>(_data_size) },
                         options)
            .clone());

    tdata.clear();
    for (unsigned int i = 0; i < p; ++i)
      for (unsigned int j = 0; j < _backcast_linspace.size(); ++j)
        for (unsigned int d2 = 0; d2 < _data_size; ++d2)
          tdata.push_back(static_cast<float>(
              powf(_backcast_linspace[j], static_cast<float>(i))));
    bT = register_buffer(
        "bT",
        torch::from_blob(tdata.data(),
                         { static_cast<long int>(p),
                           static_cast<long int>(_backcast_linspace.size())
                               * static_cast<long int>(_data_size) },
                         options)
            .clone());
    return std::make_tuple(bT, fT);
  }

  torch::Tensor NBeats::TrendBlockImpl::extract(torch::Tensor x,
                                                std::string extract_layer)
  {
    x = BlockImpl::first_extract(x, extract_layer);
    if (extract_layer != "theta_b_fc" && extract_layer != "theta_f_fc")
      return x;

    torch::Tensor tbfc = _theta_b_fc->forward(x);
    torch::Tensor backcast = tbfc.mm(_bT);
    if (extract_layer == "theta_b_fc")
      return backcast;

    torch::Tensor tffc = _theta_b_fc->forward(x);
    torch::Tensor forecast = tffc.mm(_fT);
    return forecast;
  }

  std::tuple<torch::Tensor, torch::Tensor>
  NBeats::TrendBlockImpl::forward(torch::Tensor x)
  {
    x = BlockImpl::first_forward(x);
    torch::Tensor tbfc = _theta_b_fc->forward(x);
    torch::Tensor tffc = _theta_b_fc->forward(x);
    torch::Tensor backcast = tbfc.mm(_bT);
    torch::Tensor forecast = tffc.mm(_fT);

    return std::make_tuple(
        backcast.reshape({ backcast.size(0), _backcast_length, _data_size }),
        forecast.reshape({ forecast.size(0), _forecast_length, _data_size }));
  }

  torch::Tensor NBeats::GenericBlockImpl::extract(torch::Tensor x,
                                                  std::string extract_layer)
  {
    x = BlockImpl::first_extract(x, extract_layer);
    if (extract_layer != "theta_b_fc" && extract_layer != "theta_f_fc")
      return x;

    x = _theta_b_fc->forward(x);
    if (extract_layer == "theta_b_fc")
      return x;

    torch::Tensor theta_b = torch::relu(x);
    torch::Tensor y = _theta_f_fc->forward(x);
    if (extract_layer == "theta_f_fc")
      return y;
    torch::Tensor theta_f = torch::relu(y);
    torch::Tensor backcast = _backcast_fc->forward(theta_b);
    if (extract_layer == "backcast_fc")
      return backcast;
    torch::Tensor forecast = _forecast_fc->forward(theta_f);
    return forecast;
  }

  std::tuple<torch::Tensor, torch::Tensor>
  NBeats::GenericBlockImpl::forward(torch::Tensor x)
  {
    x = BlockImpl::first_forward(x);
    torch::Tensor theta_b = torch::relu(_theta_b_fc->forward(x));
    torch::Tensor theta_f = torch::relu(_theta_f_fc->forward(x));
    torch::Tensor backcast = _backcast_fc->forward(theta_b);
    torch::Tensor forecast = _forecast_fc->forward(theta_f);
    return std::make_tuple(
        backcast.reshape({ backcast.size(0), _backcast_length, _data_size }),
        forecast.reshape({ forecast.size(0), _forecast_length, _data_size }));
  }

  void NBeats::update_params(const CSVTSTorchInputFileConn &inputc)
  {
    _data_size = inputc._datadim - inputc._label.size();
    if (inputc._forecast_timesteps > 0)
      {
        _forecast_length = inputc._forecast_timesteps;
        _backcast_length = inputc._backcast_timesteps;
      }
    else
      _forecast_length = _backcast_length = inputc._timesteps;
  }

  void NBeats::create_nbeats()
  {
    _stacks.clear();
    float back_step = 1.0 / (float)(_backcast_length);
    _backcast_linspace.clear();
    for (unsigned int i = 0; i < _backcast_length; ++i)
      _backcast_linspace.push_back(back_step * static_cast<float>(i));
    float fore_step = 1.0 / (float)(_forecast_length);
    _forecast_linspace.clear();
    for (unsigned int i = 0; i < _forecast_length; ++i)
      _forecast_linspace.push_back(fore_step * static_cast<float>(i));

    std::tuple<torch::Tensor, torch::Tensor> S;
    std::tuple<torch::Tensor, torch::Tensor> T;

    for (unsigned int stack_id = 0; stack_id < _stack_types.size(); ++stack_id)
      {
        BlockType bt = _stack_types[stack_id];
        Stack s;
        switch (bt)
          {
          case seasonality:
            S = create_sin_basis(_thetas_dims[stack_id]);
            for (unsigned int block_id = 0; block_id < _nb_blocks_per_stack;
                 ++block_id)
              s.push_back(torch::nn::AnyModule(register_module(
                  "seasonalityBlock_" + std::to_string(block_id) + "_stack_"
                      + std::to_string(stack_id),
                  SeasonalityBlock(_hidden_layer_units, _thetas_dims[stack_id],
                                   _backcast_length, _forecast_length,
                                   _data_size, std::get<0>(S),
                                   std::get<1>(S)))));
            break;
          case trend:
            T = create_exp_basis(_thetas_dims[stack_id]);
            for (unsigned block_id = 0; block_id < _nb_blocks_per_stack;
                 ++block_id)
              s.push_back(torch::nn::AnyModule(register_module(
                  "trendBlock_" + std::to_string(block_id) + "_stack_"
                      + std::to_string(stack_id),
                  TrendBlock(_hidden_layer_units, _thetas_dims[stack_id],
                             _backcast_length, _forecast_length, _data_size,
                             std::get<0>(T), std::get<1>(T)))));
            break;
          case generic:
            for (unsigned int block_id = 0; block_id < _nb_blocks_per_stack;
                 ++block_id)
              s.push_back(torch::nn::AnyModule(register_module(
                  "genericBlock_" + std::to_string(block_id) + "_stack_"
                      + std::to_string(stack_id),
                  GenericBlock(_hidden_layer_units, _thetas_dims[stack_id],
                               _backcast_length, _forecast_length,
                               _data_size))));
            break;
          default:
            break;
          }
        _stacks.push_back(s);
      }
    _finit = register_buffer(
        "finit", torch::zeros({ 1, _forecast_length, _data_size }));
  }

  void NBeats::reset()
  {
    create_nbeats();
  }

  torch::Tensor NBeats::forward(torch::Tensor x)
  {
    torch::Tensor b = x;
    torch::Tensor f = _finit.repeat({ x.size(0), 1, 1 });

    int stack_counter = 0;
    for (Stack s : _stacks)
      {
        for (torch::nn::AnyModule m : s)
          {
            auto bf = m.forward<std::tuple<torch::Tensor, torch::Tensor>>(b);
            b = b - std::get<0>(bf);
            f = f + std::get<1>(bf);
          }
        stack_counter++;
      }

    return torch::cat({ b, f }, 1);
  }

  bool NBeats::extractable(std::string extract_layer) const
  {
    std::vector<std::string> els = extractable_layers();
    return std::find(els.begin(), els.end(), extract_layer) != els.end();
  }

  std::vector<std::string> NBeats::extractable_layers() const
  {
    std::vector<std::string> els;
    for (unsigned long int si = 0; si < _stacks.size(); ++si)
      for (unsigned long int bi = 0; bi < _stacks[si].size(); ++bi)
        {
          for (auto item : _stacks[si][bi].ptr()->named_children().keys())
            els.push_back(std::to_string(si) + ":" + std::to_string(bi) + ":"
                          + item);
          els.push_back(std::to_string(si) + ":" + std::to_string(bi)
                        + ":end");
        }
    return els;
  }

  torch::Tensor NBeats::extract(torch::Tensor x, std::string extract_layer)
  {

    std::vector<std::string> subst;
    std::string item;
    size_t pos_start = 0, pos_end;
    while ((pos_end = extract_layer.find(":", pos_start)) != std::string::npos)
      {
        subst.push_back(extract_layer.substr(pos_start, pos_end - pos_start));
        pos_start = pos_end + 1;
      }
    subst.push_back(extract_layer.substr(pos_start));

    int num_stack = std::stoi(subst[0]);
    int num_block = std::stoi(subst[1]);
    bool endofblock = subst[2] == "end";

    torch::Tensor b = x;
    torch::Tensor f = _finit.repeat({ x.size(0), 1, 1 });

    int stack_counter = 0;
    for (Stack s : _stacks)
      {
        int block_counter = 0;
        for (torch::nn::AnyModule m : s)
          {
            if (num_stack == stack_counter && num_block == block_counter
                && !endofblock)
              {
                if (_stack_types[stack_counter] == trend)
                  return m.get<TrendBlock>()->extract(b, subst[2]);
                if (_stack_types[stack_counter] == seasonality)
                  return m.get<SeasonalityBlock>()->extract(b, subst[2]);
                if (_stack_types[stack_counter] == generic)
                  return m.get<GenericBlock>()->extract(b, subst[2]);
              }
            else
              {
                auto bf
                    = m.forward<std::tuple<torch::Tensor, torch::Tensor>>(b);

                b = b - std::get<0>(bf);
                f = f + std::get<1>(bf);
                if (num_stack == stack_counter && num_block == block_counter
                    && endofblock)
                  return torch::cat({ b, f }, 1);
                block_counter++;
              }
          }
        stack_counter++;
      }
    return torch::cat({ b, f }, 1);
  }
}
