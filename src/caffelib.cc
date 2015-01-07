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
#include <chrono>
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
    
    // XXX: setting the GPU outside of Caffe appears to fuss with the static pointers
    /*if (_gpu)
      {
	Caffe::SetDevice(_gpuid);
	Caffe::set_mode(Caffe::GPU);
      }
    else Caffe::set_mode(Caffe::CPU);
    if (this->_has_predict)
    Caffe::set_phase(Caffe::TEST); // XXX: static within Caffe, cannot go along with training across services.
    else Caffe::set_phase(Caffe::TRAIN);*/
    create_model();
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
  int CaffeLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::create_model()
  {
    if (!this->_mlmodel._def.empty() && !this->_mlmodel._weights.empty()) // whether in prediction mode...
      {
	_net = new Net<float>(this->_mlmodel._def);
	_net->CopyTrainedLayersFrom(this->_mlmodel._weights);
	return 0;
      }
    return 1;
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  int CaffeLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::train(const APIData &ad,
										 APIData &out)
  {
    if (this->_mlmodel._solver.empty())
      {
	LOG(ERROR) << "missing solver file";
	return 1;
      }

    caffe::SolverParameter solver_param;
    caffe::ReadProtoFromTextFileOrDie(this->_mlmodel._solver,&solver_param); //TODO: no die
    update_solver_data_paths(solver_param);
    
    // parameters
    // solver's parameters
    APIData ad_solver = ad.getobj("parameters").getobj("mllib").getobj("solver");
    if (ad_solver.size())
      {
	if (ad_solver.has("iterations"))
	  {
	    int max_iter = ad_solver.get("iterations").get<int>();
	    solver_param.set_max_iter(max_iter);
	  }
	if (ad_solver.has("snapshot")) // iterations between snapshots
	  {
	    int snapshot = ad_solver.get("snapshot").get<int>();
	    solver_param.set_snapshot(snapshot);
	  }
	if (ad_solver.has("snapshot_prefix")) // overrides default internal dd prefix which is model repo
	  {
	    std::string snapshot_prefix = ad_solver.get("snapshot_prefix").get<std::string>();
	    solver_param.set_snapshot_prefix(snapshot_prefix);
	  }
	//TODO: add support for more parameters
	if (ad_solver.has("solver_type"))
	  {
	    std::string solver_type = ad_solver.get("solver_type").get<std::string>();
	    if (strcasecmp(solver_type.c_str(),"SGD"))
	      solver_param.set_solver_type(caffe::SolverParameter_SolverType_SGD);
	    else if (strcasecmp(solver_type.c_str(),"ADAGRAD"))
	      solver_param.set_solver_type(caffe::SolverParameter_SolverType_ADAGRAD);
	    else if (strcasecmp(solver_type.c_str(),"NESTEROV"))
	      solver_param.set_solver_type(caffe::SolverParameter_SolverType_NESTEROV);
	  }
	else if (ad_solver.has("test_iter"))
	  solver_param.set_test_iter(0,ad_solver.get("test_iter").get<int>()); // XXX: 0 might not always be the correct index here.
	else if (ad_solver.has("test_interval"))
	  solver_param.set_test_interval(ad_solver.get("test_interval").get<int>());
	else if (ad_solver.has("test_initialization"))
	  solver_param.set_test_initialization(ad_solver.get("test_initialization").get<bool>());
	else if (ad_solver.has("lr_policy"))
	  solver_param.set_lr_policy(ad_solver.get("lr_policy").get<std::string>());
	else if (ad_solver.has("base_lr"))
	  solver_param.set_base_lr(ad_solver.get("base_lr").get<float>());
	else if (ad_solver.has("gamma"))
	  solver_param.set_gamma(ad_solver.get("gamma").get<float>());
	else if (ad_solver.has("step_size"))
	  solver_param.set_stepsize(ad_solver.get("stepsize").get<int>());
	else if (ad_solver.has("max_iter"))
	  solver_param.set_max_iter(ad_solver.get("max_iter").get<int>());
	else if (ad_solver.has("momentum"))
	  solver_param.set_momentum(ad_solver.get("momentum").get<double>());
	else if (ad_solver.has("power"))
	  solver_param.set_power(ad_solver.get("power").get<double>());
      }
    
    // optimize
    this->_tjob_running = true;
    caffe::Solver<float> *solver = caffe::GetSolver<float>(solver_param);
    if (!this->_mlmodel._weights.empty())
      {
	solver->net()->CopyTrainedLayersFrom(this->_mlmodel._weights);
      }
    Caffe::set_phase(Caffe::TRAIN);
    solver->PreSolve();
    
    std::string snapshot_file = ad.get("snapshot_file").get<std::string>();
    if (!snapshot_file.empty())
      solver->Restore(snapshot_file.c_str());
    
    
    solver->iter_ = 0;
    solver->current_step_ = 0;
    
    const int start_iter = solver->iter_;
    int average_loss = solver->param_.average_loss();
    std::vector<float> losses;
    this->clear_loss_per_iter();
    float smoothed_loss = 0.0;
    std::vector<Blob<float>*> bottom_vec;
    while(solver->iter_ < solver->param_.max_iter()
	  && this->_tjob_running.load())
      {
	// Save a snapshot if needed.
	if (solver->param_.snapshot() && solver->iter_ > start_iter &&
	    solver->iter_ % solver->param_.snapshot() == 0) {
	  solver->Snapshot();
	}
	if (solver->param_.test_interval() && solver->iter_ % solver->param_.test_interval() == 0
	    && (solver->iter_ > 0 || solver->param_.test_initialization())) 
	  {
	    solver->TestAll();
	  }
	float loss = solver->net_->ForwardBackward(bottom_vec);
	if (static_cast<int>(losses.size()) < average_loss) 
	  {
	    losses.push_back(loss);
	    int size = losses.size();
	    smoothed_loss = (smoothed_loss * (size - 1) + loss) / size;
	  } 
	else 
	  {
	    int idx = (solver->iter_ - start_iter) % average_loss;
	    smoothed_loss += (loss - losses[idx]) / average_loss;
	    losses[idx] = loss;
	  }
	this->_loss.store(smoothed_loss);
	
	if (solver->param_.test_interval() && solver->iter_ % solver->param_.test_interval() == 0)
	  this->add_loss_per_iter(loss); // to avoid filling up with possibly millions of entries...
	
	//std::cout << "loss=" << this->_loss << std::endl;
	
	solver->ComputeUpdateValue();
	solver->net_->Update();
	
	solver->iter_++;
      }
    // always save final snapshot.
    if (solver->param_.snapshot_after_train())
      solver->Snapshot();
    if (_net)
      delete _net;
    this->_mlmodel.read_from_repository(this->_mlmodel._repo);
    if (create_model())
      throw MLLibInternalException("no model in " + this->_mlmodel._repo + " for initializing net");
    return 0;
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  int CaffeLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::predict(const APIData &ad,
										   APIData &out)
  {
    TInputConnectorStrategy inputc(this->_inputc);
    inputc.transform(ad); //TODO: catch errors ?
    int batch_size = inputc.size();
    
    // with datum
    /*std::vector<Datum> dv;
    for (int i=0;i<batch_size;i++)
      {      
	Datum datum;
	CVMatToDatum(inputc._images.at(i),&datum);
	dv.push_back(datum);
	}*/
    boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(_net->layers()[0])->AddDatumVector(inputc._dv);
    
    // with addmat (PR)
    //std::vector<int> dvl(batch_size,0.0);
    //boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(_net->layers()[0])->AddMatVector(inputc._images,dvl); // Caffe will crash with gtest or sigsegv here if input size is incorrect.
    //std::vector<Blob<float>*> results = _net->Forward(bottom,&loss);
    
    float loss = 0.0;
    std::chrono::time_point<std::chrono::system_clock> tstart = std::chrono::system_clock::now();
    std::vector<Blob<float>*> results = _net->ForwardPrefilled(&loss); // XXX: on a batch, are we getting the average loss ?
    std::chrono::time_point<std::chrono::system_clock> tstop = std::chrono::system_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(tstop-tstart).count();
    //std::cout << "Caffe prediction time=" << elapsed << std::endl;
    int slot = results.size() - 1;
    /*std::cout << "results size=" << results.size() << std::endl;
      std::cout << "count=" << results[slot]->count() << std::endl;*/
    int scount = results[slot]->count();
    int scperel = scount / batch_size;
    TOutputConnectorStrategy tout;
    for (int j=0;j<batch_size;j++)
      {
	tout.add_result(inputc._uris.at(j),loss);
	for (int i=0;i<scperel;i++)
	  {
	    tout.add_cat(inputc._uris.at(j),results[slot]->cpu_data()[j*scperel+i],this->_mlmodel.get_hcorresp(i));
	  }
      }
    TOutputConnectorStrategy btout(this->_outputc);
    tout.best_cats(ad,btout); //TODO: use output parameter for best cat
    btout.to_ad(out);
    out.add("status",0);
    
    return 0;
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  void CaffeLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::update_solver_data_paths(caffe::SolverParameter &sp)
  {
    // fix net model path.
    sp.set_net(this->_mlmodel._repo + "/" + sp.net());
    
    // fix net snapshot path.
    sp.set_snapshot_prefix(this->_mlmodel._repo + "/model");
    
    // fix source paths in the model.
    caffe::NetParameter *np = sp.mutable_net_param();
    caffe::ReadProtoFromTextFile(sp.net().c_str(),np); //TODO: error on read + use internal caffe ReadOrDie procedure
    for (int i=0;i<np->layers_size();i++)
      {
	caffe::LayerParameter *lp = np->mutable_layers(i);
	if (lp->has_data_param())
	  {
	    caffe::DataParameter *dp = lp->mutable_data_param();
	    if (dp->has_source())
	      {
		dp->set_source(this->_mlmodel._repo + "/" + dp->source());
	      }
	  }
      }
    sp.clear_net();
  }

  template class CaffeLib<ImgCaffeInputFileConn,SupervisedOutput,CaffeModel>;
}
