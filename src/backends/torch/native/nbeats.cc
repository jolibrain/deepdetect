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
	auto options = torch::TensorOptions().dtype(torch::kFloat32);
	unsigned int p = x.sizes().back();
	unsigned int p1 = p / 2;
	unsigned int p2 = (p % 2 == 0) ? p/2 : p/2 + 1;
	std::vector<float> data;
	for (unsigned int i = 0; i<p1; ++i)
	  for (unsigned int j = 0; j< times.size(); ++j)
		data.push_back(std::cos(2*M_PI * i * times[j]));
	torch::Tensor s1 = torch::from_blob(data.data(),{p1,times.size()},options);
	data.clear();
	for (unsigned int i = 0; i<p2; ++i)
	  for (unsigned int j = 0; j< times.size(); ++j)
		data.push_back(std::sin(2*M_PI * i * times[j]));
	torch::Tensor s2 = torch::from_blob(data.data(),{p2,times.size()},options);
	torch::Tensor S = torch::cat({s1,s2});
	return x.mm(S.to(this->_dtype).to(this->_device));
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
	auto options = torch::TensorOptions().dtype(torch::kFloat32);
	unsigned int p = x.sizes().back();
	std::vector<float> data;
	for (unsigned int i = 0; i<p; ++i)
	  for (unsigned int j = 0; j< times.size(); ++j)
		data.push_back(std::pow(times[j],i));
	torch::Tensor T = torch::from_blob(data.data(),{p,times.size()},options);
	return x.mm(T.to(_dtype).to(_device));
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
