#include "nbeats.h"

namespace dd
{
  void NBeats::Block::init_block()
  {
	_fc1 = register_module("fc1", torch::nn::Linear(_backcast_length, _units));
	_fc2 = register_module("fc2", torch::nn::Linear(_units, _units));
	_fc3 = register_module("fc3", torch::nn::Linear(_units, _units));
	_fc4 = register_module("fc4", torch::nn::Linear(_units, _units));
	_theta_f_fc =
	  register_module(
					  "theta_f_fc",
					  torch::nn::Linear(torch::nn::LinearOptions(_units, _thetas_dim).bias(false)));
	if (_share_thetas)
	  _theta_b_fc = _theta_f_fc;
	else
	  _theta_f_fc =
		register_module("theta_f_fc",
						torch::nn::Linear(torch::nn::LinearOptions(_units, _thetas_dim).bias(false)));

	float step = (float)(_backcast_length + _forecast_length ) /
	  (float)(_backcast_length + _forecast_length-1);
	for (unsigned int i = 0; i<_backcast_length; ++i)
	  _backcast_linspace.push_back(-_backcast_length + step*i);
	for (unsigned int i =0; i< _forecast_length; ++i)
	  _forecast_linspace.push_back(_backcast_linspace[_backcast_length-1] + (i+1)*step);
  }

  torch::Tensor NBeats::Block::forward(torch::Tensor x)
  {
	x = torch::relu(_fc1->forward(x));
	x = torch::relu(_fc2->forward(x));
	x = torch::relu(_fc3->forward(x));
	x = torch::relu(_fc4->forward(x));
	return x;
  }

  std::tuple<torch::Tensor,torch::Tensor> NBeats::SeasonalityBlock::forward(torch::Tensor x)
  {
	x = Block::forward(x);
	torch::Tensor backcast = seasonality_model(_theta_b_fc->forward(x),_backcast_linspace);
	torch::Tensor forecast = seasonality_model(_theta_f_fc->forward(x),_forecast_linspace);
	return std::make_tuple(backcast,forecast);
  }

  torch::Tensor NBeats::SeasonalityBlock::seasonality_model(torch::Tensor x, const std::vector<float>& times)
  {
	//TODO
	int p = x.sizes().back();
	int p1 = p / 2;
	int p2 = (p % 2 == 0) ? p/2 : p/2 + 1;
  }



  std::tuple<torch::Tensor,torch::Tensor> NBeats::TrendBlock::forward(torch::Tensor x)
  {
	x = Block::forward(x);
	torch::Tensor backcast = trend_model(_theta_b_fc->forward(x),_backcast_linspace);
	torch::Tensor forecast = trend_model(_theta_f_fc->forward(x),_forecast_linspace);
	return std::make_tuple(backcast,forecast);
  }

  torch::Tensor NBeats::TrendBlock::trend_model(torch::Tensor x, const std::vector<float>& times)
  {
	//TODO
  }


  std::tuple<torch::Tensor,torch::Tensor> NBeats::GenericBlock::forward(torch::Tensor x)
  {
	x = Block::forward(x);
	torch::Tensor theta_b =	torch::relu(_theta_b_fc->forward(x));
	torch::Tensor theta_f =	torch::relu(_theta_f_fc->forward(x));

	torch::Tensor backcast = _backcast_fc->forward(theta_b);
	torch::Tensor forecast = _forecast_fc->forward(theta_f);
	return std::make_tuple(backcast,forecast);
  }

  void NBeats::update_params(const APIData &adlib, const CSVTSTorchInputFileConn &inputc)
  {
	//TODO
  }


  void NBeats::create_nbeats()
  {
	//TODO
  }


  torch::Tensor NBeats::forward(torch::Tensor x)
  {
	//TODO
  }
}
