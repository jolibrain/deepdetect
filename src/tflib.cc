/**
 * DeepDetect
 * Copyright (c) 2016 Emmanuel Benazera
 * Author: Emmanuel Benazera <beniz@droidnik.fr>
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

#include <string>
#include "tflib.h"
#include "imginputfileconn.h"
#include "outputconnectorstrategy.h"

#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/graph/default_device.h" 

namespace dd
{

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  TFLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::TFLib(const TFModel &cmodel)
    :MLLib<TInputConnectorStrategy,TOutputConnectorStrategy,TFModel>(cmodel)
  {
    this->_libname = "tensorflow";
  }
  
  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  TFLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::TFLib(TFLib &&cl) noexcept
    :MLLib<TInputConnectorStrategy,TOutputConnectorStrategy,TFModel>(std::move(cl))
  {
    this->_libname = "tensorflow";
    _nclasses = cl._nclasses;
    _regression = cl._regression;
    _ntargets = cl._ntargets;
    _inputLayer = cl._inputLayer;
    _outputLayer = cl._outputLayer;
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  TFLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::~TFLib()
  {
    //TODO: mutex in case of concurrent session Run calls ?
    //TODO: delete structures, if any
    if (_session)
      {
	_session->Close();
	_session.reset();
      }
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  void TFLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::init_mllib(const APIData &ad)
  {
    if (ad.has("nclasses"))
      _nclasses = ad.get("nclasses").get<int>();
    if (ad.has("regression") && ad.get("regression").get<bool>())
      {
	_regression = true;
	_nclasses = 1;
      }
    // setting the value of Input Layer for Tensorflow graph
    if (ad.has("inputlayer"))
      {
	_inputLayer = ad.get("inputlayer").get<std::string>();
      }
    // setting the final Output Layer for Tensorflow graph
    if (ad.has("outputlayer"))
    {
      _outputLayer = ad.get("outputlayer").get<std::string>();
    }
    if (ad.has("ntargets"))
      _ntargets = ad.get("ntargets").get<int>();
    if (_nclasses == 0)
      throw MLLibBadParamException("number of classes is unknown (nclasses == 0)");
    if (_regression && _ntargets == 0)
      throw MLLibBadParamException("number of regression targets is unknown (ntargets == 0)");
    this->_mlmodel.read_from_repository(this->_mlmodel._repo);
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  void TFLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::clear_mllib(const APIData &ad)
  {
    //TODO
  }
  
  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  int TFLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::train(const APIData &ad,
									       APIData &out)
  {
    //TODO
    return 0;
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  int TFLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::predict(const APIData &ad,
										   APIData &out)
  {
    //std::string device = "/gpu:0";
    TInputConnectorStrategy inputc(this->_inputc);
    TOutputConnectorStrategy tout;
    APIData cad = ad;
    cad.add("model_repo",this->_mlmodel._repo);
    try
      {
	inputc.transform(cad);
      }
    catch (std::exception &e)
      {
	throw;
      }
    
    int batch_size = inputc.batch_size();

    if (!_session)
      {
	tensorflow::GraphDef graph_def;
	std::string graphFile = this->_mlmodel._graphName;
	if (graphFile.empty())
	  throw MLLibBadParamException("No pre-trained model found in model repository");
	std::cout << "graphFile dir=" << graphFile<< std::endl;
	// Loading the graph to the given variable
	tensorflow::Status graphLoadedStatus = ReadBinaryProto(tensorflow::Env::Default(),graphFile,&graph_def);
	
	if (!graphLoadedStatus.ok())
	  {
	    std::cerr << graphLoadedStatus.ToString()<<std::endl;
	    LOG(ERROR) << "failed loading tensorflow graph with status=" << graphLoadedStatus.ToString() << std::endl;
	    throw MLLibBadParamException("failed loading tensorflow graph with status=" + graphLoadedStatus.ToString());
	  }
	/*std::cerr << "graph load status=" << graphLoadedStatus.ToString() << std::endl;
	  std::cerr << "graph def node size=" << graph_def.node_size() << std::endl;
	  for (size_t i=0;i<graph_def.node_size();i++)
	  std::cerr << graph_def.node(i).name() << std::endl;*/

	if (_inputLayer.empty())
	  {
	    _inputLayer = graph_def.node(0).name();
	    LOG(INFO) << "using input layer=" << _inputLayer << std::endl;
	  }
	if (_outputLayer.empty())
	  {
	    _outputLayer = graph_def.node(graph_def.node_size()-1).name();
	    LOG(INFO) << "using output layer=" << _outputLayer << std::endl;
	  }

	//tensorflow::graph::SetDefaultDevice(device, &graph_def);
	
	// creating a session with the graph
	tensorflow::SessionOptions options;
	tensorflow::ConfigProto &config = options.config;
	config.mutable_gpu_options()->set_allow_growth(true); // default is we prevent tf from holding all memory across all GPUs
	_session = std::unique_ptr<tensorflow::Session>(tensorflow::NewSession(options));
	tensorflow::Status session_create_status = _session->Create(graph_def);
	
	if (!session_create_status.ok())
	  {
	    std::cout << session_create_status.ToString()<<std::endl;
	    _session = nullptr;
	    throw MLLibInternalException(session_create_status.ToString());
	  }
      }
    
    // vector for storing  the outputAPI of the file 
    std::vector<APIData> vrad;
    
    
    std::vector<tensorflow::Tensor> vtfinputs;
    if (batch_size > 1)
      {
	/*
	  as incredible as it seems, code below is the bloated tf way of concatenating
	  tensors to produce the intput to the neural net graph.
	*/
	auto root = tensorflow::Scope::NewRootScope();
	std::string concat_name = "concatenated";
	std::vector<tensorflow::ops::Input> ops_inputs;
	for (int i=0;i<batch_size;i++)
	  ops_inputs.push_back(tensorflow::ops::Input(inputc._dv[i]));
	tensorflow::gtl::ArraySlice<tensorflow::ops::Input> ipl(&ops_inputs[0],ops_inputs.size());
	tensorflow::ops::InputList toil(ipl);
	auto concatout = tensorflow::ops::Concat(root.WithOpName(concat_name),0,toil);
	std::unique_ptr<tensorflow::Session> concat_session(tensorflow::NewSession(tensorflow::SessionOptions()));
	tensorflow::GraphDef graph;
	root.ToGraphDef(&graph);
	concat_session->Create(graph);
	tensorflow::Status concat_run_status = concat_session->Run({}, {concat_name}, {}, &vtfinputs);
	if (!concat_run_status.ok())
	  {
	    std::cout << concat_run_status.ToString() << std::endl;
	    throw MLLibInternalException(concat_run_status.ToString());
	  }
      }
    else vtfinputs = {inputc._dv.at(0)};
    
    // running the loded graph and saving the generated output 
    std::vector<tensorflow::Tensor> finalOutput; // To save the final Output generated by the tensorflow
    tensorflow::Status run_status  = _session->Run({{_inputLayer,*(vtfinputs.begin())}},{_outputLayer},{},&finalOutput);
    if (!run_status.ok())
      {
	std::cout <<run_status.ToString()<<std::endl;
	throw MLLibInternalException(run_status.ToString()); //TODO: separate bad param and internal errors
      }
    tensorflow::Tensor output = std::move(finalOutput.at(0));
    APIData rad;
    auto scores = output.flat<float>();
    for (int i=0;i<batch_size;i++)
      {
	rad.add("uri",inputc._ids.at(i));
	std::vector<double> probs;
	std::vector<std::string> cats;
	for (int c=0;c<_nclasses;c++)
	  {
	    //std::cerr << "score=" << scores(c) << " / c=" << c << std::endl;
	    probs.push_back(scores(i*_nclasses+c));
	    cats.push_back(this->_mlmodel.get_hcorresp(c));
	  }
	rad.add("probs",probs);
	rad.add("cats",cats);
	rad.add("loss",0.0);
	vrad.push_back(rad);
      }
    tout.add_results(vrad);
    tout.finalize(ad.getobj("parameters").getobj("output"),out);
    out.add("status",0);  
    
    return 0;
  }
  
  template class TFLib<ImgTFInputFileConn,SupervisedOutput,TFModel>;
  template class TFLib<ImgTFInputFileConn,UnsupervisedOutput,TFModel>;
}
