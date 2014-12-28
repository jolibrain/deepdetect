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
  {
    static std::string snapshotf = "snapshot";
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
	    int max_iter = static_cast<int>(ad_solver.get("iterations").get<double>());
	    solver_param.set_max_iter(max_iter);
	  }
	//TODO: add support for more parameters
      }
    
    // optimize
    this->_tjob_running = true;
    caffe::Solver<float> *solver = caffe::GetSolver<float>(solver_param);
    bool async = !ad.has("async") ? false : ad.get("async").get<bool>();
    if (!async)
      {
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
	    /*delete _net;
	      _net = solver->net().get();*/ // setting up the new model
	    std::vector<Blob<float>*> bottom_vec; // dummy
	    float lloss = 0.0;
	    solver->net()->Forward(bottom_vec,&lloss);
	    this->_loss.store(lloss);
	  }
      }
    else
      {
	Caffe::set_phase(Caffe::TRAIN);
	solver->PreSolve();
	
	solver->iter_ = 0;
	solver->current_step_ = 0;
	//TODO: resume file from ad
	/*if (resume_file) {
	  LOG(INFO) << "Restoring previous solver status from " << resume_file;
	  Restore(resume_file);
	  }*/
	
	const int start_iter = solver->iter_;
	int average_loss = solver->param_.average_loss();
	std::vector<float> losses;
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

	    //std::cout << "loss=" << this->_loss << std::endl;
	    
	    solver->ComputeUpdateValue();
	    solver->net_->Update();
	  
	    solver->iter_++;
	  }
	// always save final snapshot.
	if (solver->param_.snapshot_after_train())
	  solver->Snapshot();
      }    
    //TODO: output connector ?
    return 0;
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  int CaffeLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::predict(const APIData &ad,
										   APIData &out)
  {
    TInputConnectorStrategy inputc(this->_inputc);
    inputc.transform(ad); //TODO: catch errors ?
    Datum datum;
    //ReadImageToDatum(inputc._imgfname,1,227,227,&datum);
    //CVMatToDatum(inputc._image,&datum);
    //std::vector<Blob<float>*> bottom = {blob};
    float loss = 0.0;
    //std::vector<Datum> dv = {datum};
    //std::vector<cv::Mat> dv = {inputc._images.at(0),inputc._images.at(1)};
    int batch_size = inputc.size();
    std::vector<int> dvl(batch_size,0.0);
    //boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(_net->layers()[0])->AddDatumVector(dv);
    boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(_net->layers()[0])->AddMatVector(inputc._images,dvl); // Caffe will crash with gtest or sigsegv here if input size is incorrect.
    //std::vector<Blob<float>*> results = _net->Forward(bottom,&loss);
    //TODO: loss ?
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
    TOutputConnectorStrategy btout;
    tout.best_cats(5,btout);
    btout.to_ad(out);
    out.add("status",0);
    
    return 0;
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  void CaffeLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::update_solver_data_paths(caffe::SolverParameter &sp)
  {
    // fix net model path.
    sp.set_net(this->_mlmodel._repo + "/" + sp.net());
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

  template class CaffeLib<ImgInputFileConn,SupervisedOutput,CaffeModel>;
}
