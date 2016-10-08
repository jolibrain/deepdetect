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
    //TODO: delete structures, if any
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

    }else{
        _inputLayer = "Mul"; // Default Input Layer Name of TensorFlow          
    }
    std::cerr << "inputlayer=" << _inputLayer << std::endl;
    // setting the final Output Layer for Tensorflow graph
    if (ad.has("outputlayer"))
    {
      _outputLayer = ad.get("outputlayer").get<std::string>();

    }else{
        _outputLayer = "softmax"; // Default Input Layer Name of TensorFlow
              
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
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  int TFLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::predict(const APIData &ad,
										   APIData &out)
  {
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

  tensorflow::GraphDef graph_def;
  std::string graphFile = this->_mlmodel._graphName;
  if (graphFile.empty())
    throw MLLibBadParamException("No pre-trained model found in model repository");
  std::cout << "graphFile dir=" << graphFile<< std::endl;
  // Loading the graph to the given variable
  tensorflow::Status graphLoadedStatus = ReadBinaryProto(tensorflow::Env::Default(),graphFile,&graph_def);

  if (!graphLoadedStatus.ok()){
    std::cerr << graphLoadedStatus.ToString()<<std::endl;
    return 1;
  }
  /*std::cerr << "graph load status=" << graphLoadedStatus.ToString() << std::endl;
  std::cerr << "graph def node size=" << graph_def.node_size() << std::endl;
  for (size_t i=0;i<graph_def.node_size();i++)
  std::cerr << graph_def.node(i).name() << std::endl;*/

  // creating a session with the graph
  std::unique_ptr<tensorflow::Session> session(tensorflow::NewSession(tensorflow::SessionOptions()));
  tensorflow::Status session_create_status = session->Create(graph_def);
  
  if (!session_create_status.ok()){
    std::cout <<session_create_status.ToString()<<std::endl;
    return 1;
  }

  // vector for storing  the outputAPI of the file 
  std::vector<APIData> vrad;
  // running the loded graph and saving the generated output 
  std::vector<tensorflow::Tensor>::iterator it = inputc._dv.begin();
  for (int i =0  ; i<batch_size; i++){ //TODO: pass data in batch
    std::vector<tensorflow::Tensor> finalOutput; // To save the final Output generated by the tensorflow
    tensorflow::Status run_status  = session->Run({{_inputLayer,*it}},{_outputLayer},{},&finalOutput);
    if (!run_status.ok()){
      std::cout <<run_status.ToString()<<std::endl;
      throw MLLibInternalException(run_status.ToString()); //TODO: separate bad param and internal errors
    }
    tensorflow::Tensor output = std::move(finalOutput.at(0));
    APIData rad;
    rad.add("uri",inputc._ids.at(i));
    generatedLabel(output,rad);
    rad.add("loss",0.0);
    vrad.push_back(rad);
    
    ++it;
  }
  tout.add_results(vrad);
  tout.finalize(ad.getobj("parameters").getobj("output"),out);
  out.add("status",0);  

  return 0;
  }
  
  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  void TFLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::generatedLabel(const tensorflow::Tensor &output, APIData &out)
  {
    // file for reading the label file and marking the output accordingly
    std::string labelfile = this->_mlmodel._labelName;
    std::ifstream label(labelfile); 
    std::string line;

    auto scores = output.flat<float>();
    
    // sorting the file to find the top labels
    std::vector<std::pair<float,std::string>> sorted;

    for (unsigned int i =0; i<_nclasses ;++i){
      std::getline(label,line);
      sorted.emplace_back(scores(i),line);
    }
    
    std::sort(sorted.begin(),sorted.end());
    std::reverse(sorted.begin(),sorted.end());
    //selecting the output with top 5 probability
    std::vector<double> probs;
    std::vector<std::string> cats;
    for(unsigned int i =0 ; i< 5;++i){
      //std::cout << sorted[i].first << " "<<sorted[i].second <<std::endl;
      probs.push_back(sorted[i].first);
      cats.push_back(sorted[i].second);  
    }
  out.add("probs",probs);
  out.add("cats",cats);
  }

  template class TFLib<ImgTFInputFileConn,SupervisedOutput,TFModel>;
  template class TFLib<ImgTFInputFileConn,UnsupervisedOutput,TFModel>;
}
