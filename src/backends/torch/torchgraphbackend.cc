/**
 * DeepDetect
 * Copyright (c) 2020 Jolibrain
 * Author: Guillaume Infantes <guillaume.infantes@jolibrain.com>
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



#include "torchgraphbackend.h"


namespace dd
{
  using torch::nn::AnyModule;
  using torch::nn::RNN;
  using torch::nn::RNNOptions;
  using torch::nn::LSTM;
  using torch::nn::LSTMOptions;
  using torch::nn::Linear;
  using torch::nn::LinearOptions;
  using torch::Tensor;


  void TorchGraphBackend::set_input(torch::Tensor input)
  {
	std::vector<int> dim(input.sizes().begin(),input.sizes().end());
	this->set_input_dim(dim);
	_variables[_inputname] = input;
	BaseGraph::finalize();
	allocate_modules();
  }

  void TorchGraphBackend::finalize()
  {
	BaseGraph::finalize();
	allocate_modules();
  }


  void TorchGraphBackend::finalize(at::IntArrayRef dim)
  {
	finalize(dim.vec());
  }

  void TorchGraphBackend::finalize(std::vector<int> dim)
  {
	this->set_input_dim(dim);
	BaseGraph::finalize();
	allocate_modules();
  }


  void TorchGraphBackend::finalize(std::vector<int64_t> dim)
  {
	std::vector<int> dimint;
	for (auto d:dim)
	  dimint.push_back(d);
	finalize(dimint);
  }




  torch::Tensor TorchGraphBackend::forward(torch::Tensor inputTensor)
  {
	set_input(inputTensor);
	bool output_computed = false;
	for (BaseGraph::Vertex o: _sortedOps)
	  {
		std::vector<torch::Tensor> out = forward(o);
		std::vector<BaseGraph::Vertex> outputVars = this->outputs(o);
		for (unsigned int i=0; i<outputVars.size(); ++i)
		  {
			if (varname(outputVars[i]) == _outputname)
			  output_computed = true;
			_variables[varname(outputVars[i])] = out[i];
		  }
	  }
	if (!output_computed)
	  throw TorchGraphException("did not compute output, please check NN graph");
	return _variables[_outputname];
  }

  std::vector<torch::Tensor> TorchGraphBackend::forward(BaseGraph::Vertex v)
  {

	std::vector<torch::Tensor> inputsTensor;
	for (BaseGraph::Vertex vi: this->inputs(v))
	  {
		inputsTensor.push_back(_variables[varname(vi)]);
	  }

    auto opname_v = opname(v);
	std::vector<torch::Tensor> output;
	std::string optype = this->optype(v);
	if (optype == "LSTM" || optype == "RNN")
	  {
		//get<0>(f()) for all outputs / hidden
		//get<0>(get<1>(f)) for last output
		//get<1>(get<1>(f)) for last internal state
		std::tuple<torch::Tensor, std::tuple<torch::Tensor, torch::Tensor>> full_output;
		if (_lstm_continuation && _rnn_has_memories[opname_v])
		  {
			full_output = _modules[opname_v].
			  forward<std::tuple<Tensor,std::tuple<Tensor,Tensor>>>
			  (inputsTensor[0],
			   torch::optional<std::tuple<torch::Tensor,torch::Tensor>>(_rnn_memories[opname_v]));
		  }
		else
		  full_output = _modules[opname_v].
			forward<std::tuple<Tensor,std::tuple<Tensor,Tensor>>>
			(inputsTensor[0]);
		output.push_back(std::get<0>(full_output));
		if (_lstm_continuation)
		  {
			_rnn_memories[opname_v] = std::get<1>(full_output);
			_rnn_has_memories[opname_v] = true;
		  }

	  }
	else if (optype == "InnerProduct")
	  output.push_back(_modules[opname_v].forward(inputsTensor[0]));
	else
	  throw TorchGraphException("unknown optype " + optype + " for operator " + opname_v);
	return output;
  }


  void TorchGraphBackend::allocate_modules()
  {
	for (BaseGraph::Vertex v: _sortedOps)
	  {
		if (!_graph[v].alloc_needed)
		  continue;
		if (_parameters_used)
		  throw TorchGraphException("parameters reallocation necessary while they are used elsewhere. You should module.forward() / module.set_input() / module.finalize() with correct input dimensions before modules.parameters() or  module.parameters_release() if you know what you are doing");
		std::string optype = this->optype(v);
		std::string opname = this->opname(v);
		if (_modules.find(opname) != _modules.end())
		  unregister_module(opname);

		torch::nn::AnyModule m;
		if (optype == "LSTM")
		  {
			//dim(v,0,2) is 2nd dimension of input 0 of v, ie datadim for lstm
			LSTM m = register_module(opname, LSTM(LSTMOptions(dim(v,0,2),
															  num_output(v))
												  .num_layers(1)
												  .batch_first(true)
												  .bidirectional(false)));

			_modules[opname] = AnyModule(m);
			_graph[v].alloc_needed = false;
			_rnn_has_memories[opname] = false;
		  }
		else if (optype == "RNN")
		  {
			//dim(v,0,2) is 2nd dimension of input 0 of v, ie datadim for lstm
			RNN m = register_module(opname,RNN(RNNOptions(dim(v,0,2),
														  num_output(v))
											   .num_layers(1)
											   .batch_first(true)
											   .bidirectional(false)));
			_modules[opname] = AnyModule(m);
			_graph[v].alloc_needed = false;
			_rnn_has_memories[opname] = false;
		  }
		else if (optype == "InnerProduct")
		  {
			//dim(v,0,2) is 2nd dimension of input 0 of v, ie datadim for lstm output
			Linear m = register_module(opname,
									   Linear(LinearOptions(dim(v,0,2),
															num_output(v))
											  .bias(true)));
			_modules[opname] = AnyModule(m);
			_graph[v].alloc_needed = false;
		  }
	  }
	to(_device,_dtype);
  }

  void TorchGraphBackend::to(torch::Device device, torch::Dtype dtype, bool non_blocking)
  {
	if (!allocated())
	  throw TorchGraphException("trying to move NN to gpu/cpu / cast before its effective allocation. finalize(inputdim) or forward() it once before to()");
	else
	  {
		torch::nn::Module::to(device, dtype, non_blocking);
		_device = device;
		_dtype = dtype;
	  }
  }

  void TorchGraphBackend::to(torch::Dtype dtype, bool non_blocking)
  {
	if (!allocated())
	  throw TorchGraphException("trying to cast params before their effective allocation. finalize(inputdim) or forward() it once before to()");
	else
	  {
		torch::nn::Module::to(dtype, non_blocking);
		_dtype = dtype;
	  }
  }

  void TorchGraphBackend::to(torch::Device device, bool non_blocking)
  {
	if (!allocated())
	  throw TorchGraphException("trying to move NN to gpu/cpu before its effective allocation. finalize(inputdim) or forward() it once before to()");
	else
	  {
		torch::nn::Module::to(device, non_blocking);
		_device = device;
	  }
  }

  std::vector<torch::Tensor> TorchGraphBackend::parameters(bool recurse)
  {
  	if (!allocated())
  	  throw TorchGraphException("trying to get parameters (for optim?) before their effective allocation. finalize(inputdim) or forward() it once before to()");
  	else
  	  {
  		_parameters_used =true;
  		return torch::nn::Module::parameters(recurse);
  	  }
  }


}
