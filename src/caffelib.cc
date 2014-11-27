/**
 * DeepDetect
 * Copyright (c) 2014 Emmanuel Benazera
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

#include "caffelib.h"
#include "imginputfileconn.h"
#include "outputconnectorstrategy.h"
#include <iostream>

using caffe::Caffe;
using caffe::Net;
using caffe::Blob;
using caffe::Datum;

namespace dd
{

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  CaffeLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::CaffeLib(const CaffeModel &cmodel)
    :MLLib<TInputConnectorStrategy,TOutputConnectorStrategy,CaffeModel>(cmodel)
  {
    this->_libname = "caffe";
    if (_gpu)
      {
	Caffe::SetDevice(_gpuid);
	Caffe::set_mode(Caffe::GPU);
      }
    else Caffe::set_mode(Caffe::CPU);
    if (this->_has_predict)
      Caffe::set_phase(Caffe::TEST); // XXX: static within Caffe, cannot go along with training across services.
    if (!this->_mlmodel._def.empty()) // whether in prediction mode...
      {
	_net = new Net<float>(this->_mlmodel._def);
	_net->CopyTrainedLayersFrom(this->_mlmodel._weights);
      }
  }
  
  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  CaffeLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::CaffeLib(CaffeLib &&cl) noexcept
    :MLLib<TInputConnectorStrategy,TOutputConnectorStrategy,CaffeModel>(std::move(cl))
  {
    this->_libname = "caffe";
    _gpu = cl._gpu;
    _gpuid = cl._gpuid;
    _net = cl._net;
    cl._net = nullptr;
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  CaffeLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::~CaffeLib()
  {
    if (_net)
      delete _net; // XXX: for now, crashes.
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  int CaffeLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::train(const APIData &ad,
										 APIData &out)
  //std::string &output)
  {
    static std::string snapshotf = "snapshot";
    //XXX: for now, inputc not used, will be if we run the learning loop from within here in order to collect stats along the way
    //TODO: get solver param (from ad?)
    //std::string solver_file = ad.get(solverf).get<std::string>();
    if (this->_mlmodel._solver.empty())
      {
	LOG(ERROR) << "missing solver file";
	return 1;
      }

    caffe::SolverParameter solver_param;
    caffe::ReadProtoFromTextFileOrDie(this->_mlmodel._solver,&solver_param); //TODO: no die
    
    // optimize
    std::shared_ptr<caffe::Solver<float>> solver(caffe::GetSolver<float>(solver_param));
    std::string snapshot_file = ad.get(snapshotf).get<std::string>();
    if (!snapshot_file.empty())
      solver->Solve(snapshot_file);
    else if (!this->_mlmodel._weights.empty())
      {
	solver->net()->CopyTrainedLayersFrom(this->_mlmodel._weights);
	solver->Solve();
      }
    else 
      {
	LOG(INFO) << "Optimizing model";
	solver->Solve();
      }
    //TODO: output connector.
    //output = "Optimization done.";
    return 0;
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  int CaffeLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::predict(const APIData &ad,
										   APIData &out)
  //std::string &output)
  {
    TInputConnectorStrategy inputc;
    inputc.transform(ad);
    Datum datum;
    //ReadImageToDatum(inputc._imgfname,1,227,227,&datum);
    //CVMatToDatum(inputc._image,&datum);
    //std::vector<Blob<float>*> bottom = {blob};
    float loss = 0.0;
    //std::vector<Datum> dv = {datum};
    std::vector<cv::Mat> dv = {inputc._image};
    std::vector<int> dvl = {0};
    //boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(_net->layers()[0])->AddDatumVector(dv);
    boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(_net->layers()[0])->AddMatVector(dv,dvl);
    
    //std::vector<Blob<float>*> results = _net->Forward(bottom,&loss);
    //TODO: loss ?
    std::vector<Blob<float>*> results = _net->ForwardPrefilled(&loss);
    int slot = results.size() - 1;
    std::cout << "results size=" << results.size() << std::endl;
    std::cout << "count=" << results[slot]->count() << std::endl;
    TOutputConnectorStrategy tout;
    for (int i=0;i<results[slot]->count();i++)
      {
	//std::cout << results[4]->cpu_data()[i] << std::endl;
	tout.add_cat(results[slot]->cpu_data()[i],this->_mlmodel._hcorresp[i]);
      }
    TOutputConnectorStrategy btout;
    tout.best_cats(5,btout);
    //btout.to_str(output);
    btout.to_ad(out);
    out.add("status",0);
  
    return 0;
  }

  template class CaffeLib<ImgInputFileConn,SupervisedOutput,CaffeModel>;
}
