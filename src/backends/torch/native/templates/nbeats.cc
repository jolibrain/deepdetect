#include "nbeats.h"
#include <cmath>

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
        "theta_f_fc", torch::nn::Linear(torch::nn::LinearOptions(
                                            _units, _thetas_dim * _data_size)
                                            .bias(false)));
    if (_share_thetas)
      _theta_b_fc = _theta_f_fc;
    else
      _theta_b_fc = register_module(
          "theta_b_fc", torch::nn::Linear(torch::nn::LinearOptions(
                                              _units, _thetas_dim * _data_size)
                                              .bias(false)));
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

  std::tuple<torch::Tensor, torch::Tensor>
  NBeats::SeasonalityBlockImpl::forward(torch::Tensor x)
  {
    x = BlockImpl::first_forward(x);
    torch::Tensor tbfc = _theta_b_fc->forward(x);
    torch::Tensor backcast = tbfc.mm(_bS.to(_device));
    torch::Tensor tffc = _theta_f_fc->forward(x);
    torch::Tensor forecast = tffc.mm(_fS.to(_device));
    return std::make_tuple(
        backcast.reshape({ backcast.size(0), _backcast_length, _data_size }),
        forecast.reshape({ forecast.size(0), _forecast_length, _data_size }));
  }

  std::tuple<torch::Tensor, torch::Tensor>
  NBeats::create_sin_basis(int thetas_dim)
  {
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    unsigned int p = thetas_dim;
    unsigned int p1 = p / 2;
    unsigned int p2 = (p % 2 == 0) ? p / 2 : p / 2 + 1;
    std::vector<float> tdata;

    for (unsigned int d1 = 0; d1 < _data_size; ++d1)
      for (unsigned int i = 0; i < p1; ++i)
        for (unsigned int d2 = 0; d2 < _data_size; ++d2)
          for (unsigned int j = 0; j < _forecast_linspace.size(); ++j)
            tdata.push_back(std::cos(2 * M_PI * i * _forecast_linspace[j]));
    torch::Tensor s1
        = torch::from_blob(tdata.data(),
                           { _data_size * p1,
                             static_cast<long int>(_forecast_linspace.size())
                                 * _data_size },
                           options)
              .clone();

    tdata.clear();
    for (unsigned int d1 = 0; d1 < _data_size; ++d1)
      for (unsigned int i = 0; i < p2; ++i)
        for (unsigned int d2 = 0; d2 < _data_size; ++d2)
          for (unsigned int j = 0; j < _forecast_linspace.size(); ++j)
            tdata.push_back(std::sin(2 * M_PI * i * _forecast_linspace[j]));
    torch::Tensor s2
        = torch::from_blob(tdata.data(),
                           { _data_size * p2,
                             static_cast<long int>(_forecast_linspace.size())
                                 * _data_size },
                           options)
              .clone();
    torch::Tensor fS = torch::cat({ s1, s2 });

    tdata.clear();
    for (unsigned int d1 = 0; d1 < _data_size; ++d1)
      for (unsigned int i = 0; i < p1; ++i)
        for (unsigned int d2 = 0; d2 < _data_size; ++d2)
          for (unsigned int j = 0; j < _backcast_linspace.size(); ++j)
            tdata.push_back(std::cos(2 * M_PI * i * _backcast_linspace[j]));
    torch::Tensor ss1
        = torch::from_blob(tdata.data(),
                           { _data_size * p1,
                             static_cast<long int>(_backcast_linspace.size())
                                 * _data_size },
                           options)
              .clone();

    tdata.clear();
    for (unsigned int d1 = 0; d1 < _data_size; ++d1)
      for (unsigned int i = 0; i < p2; ++i)
        for (unsigned int d2 = 0; d2 < _data_size; ++d2)
          for (unsigned int j = 0; j < _backcast_linspace.size(); ++j)
            tdata.push_back(std::sin(2 * M_PI * i * _backcast_linspace[j]));
    torch::Tensor ss2
        = torch::from_blob(tdata.data(),
                           { _data_size * p2,
                             static_cast<long int>(_backcast_linspace.size())
                                 * _data_size },
                           options)
              .clone();

    torch::Tensor bS = torch::cat({ ss1, ss2 });
    return std::make_tuple(bS.to(_device), fS.to(_device));
  }

  std::tuple<torch::Tensor, torch::Tensor>
  NBeats::create_exp_basis(int thetas_dim)
  {
    torch::Tensor bT, fT;
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    unsigned int p = thetas_dim;
    std::vector<float> tdata;

    for (unsigned int d1 = 0; d1 < _data_size; ++d1)
      for (unsigned int i = 0; i < p; ++i)
        for (unsigned int d2 = 0; d2 < _data_size; ++d2)
          for (unsigned int j = 0; j < _forecast_linspace.size(); ++j)
            {
              tdata.push_back(static_cast<float>(
                  powf(_forecast_linspace[j], static_cast<float>(i))));
              ;
            }
    fT = torch::from_blob(
             tdata.data(),
             { static_cast<long int>(p) * static_cast<long int>(_data_size),
               static_cast<long int>(_forecast_linspace.size())
                   * static_cast<long int>(_data_size) },
             options)
             .clone();

    tdata.clear();
    for (unsigned int d1 = 0; d1 < _data_size; ++d1)
      for (unsigned int i = 0; i < p; ++i)
        for (unsigned int d2 = 0; d2 < _data_size; ++d2)
          for (unsigned int j = 0; j < _backcast_linspace.size(); ++j)
            tdata.push_back(static_cast<float>(
                powf(_backcast_linspace[j], static_cast<float>(i))));
    bT = torch::from_blob(
             tdata.data(),
             { static_cast<long int>(p) * static_cast<long int>(_data_size),
               static_cast<long int>(_backcast_linspace.size())
                   * static_cast<long int>(_data_size) },
             options)
             .clone();
    return std::make_tuple(bT.to(_device), fT.to(_device));
  }

  std::tuple<torch::Tensor, torch::Tensor>
  NBeats::TrendBlockImpl::forward(torch::Tensor x)
  {
    x = BlockImpl::first_forward(x);
    torch::Tensor tbfc = _theta_b_fc->forward(x);
    torch::Tensor tffc = _theta_b_fc->forward(x);
    torch::Tensor backcast = tbfc.mm(_bT.to(_device));
    torch::Tensor forecast = tffc.mm(_fT.to(_device));

    return std::make_tuple(
        backcast.reshape({ backcast.size(0), _backcast_length, _data_size }),
        forecast.reshape({ forecast.size(0), _forecast_length, _data_size }));
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
    _output_size = inputc._label.size();
    _data_size = inputc._datadim - _output_size;
    _backcast_length = inputc._timesteps;
    // per dd timeserie / LSTM logic, there is one output per input
    _forecast_length = inputc._timesteps;
  }

  void NBeats::create_nbeats()
  {
    float step = (float)(_backcast_length + _forecast_length)
                 / (float)(_backcast_length + _forecast_length - 1);
    for (unsigned int i = 0; i < _backcast_length; ++i)
      _backcast_linspace.push_back(
          (-(float)_backcast_length + (float)step * (float)i));
    for (unsigned int i = 0; i < _forecast_length; ++i)
      _forecast_linspace.push_back(_backcast_linspace[_backcast_length - 1]
                                   + (float)(i + 1) * (float)step);

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
    _fcn = register_module("fcn", torch::nn::Linear(_data_size, _output_size));
  }

  torch::Tensor NBeats::forward(torch::Tensor x)
  {
    torch::Tensor b = x;
    torch::Tensor f = torch::zeros({ x.size(0), _forecast_length, _data_size })
                          .to(_device);

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
    return torch::stack({ b, f }, 0);
  }
}
