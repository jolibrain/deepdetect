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
#include "utils/fileops.hpp"
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
  }
  
  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  CaffeLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::CaffeLib(CaffeLib &&cl) noexcept
    :MLLib<TInputConnectorStrategy,TOutputConnectorStrategy,CaffeModel>(std::move(cl))
  {
    this->_libname = "caffe";
    _gpu = cl._gpu;
    _gpuid = cl._gpuid;
    _net = cl._net;
    _nclasses = cl._nclasses;
    cl._net = nullptr;
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  CaffeLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::~CaffeLib()
  {
    if (_net)
      delete _net; // XXX: for now, crashes.
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  void CaffeLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::instantiate_template(const APIData &ad)
  {
    // - locate template repository
    std::string model_tmpl = ad.get("template").get<std::string>();
    this->_mlmodel._model_template = model_tmpl;
    std::cout << "instantiating model template " << model_tmpl << std::endl;

    // - copy files to model repository
    std::string source = std::string(MLMODEL_TEMPLATE_REPO) + "caffe/" + model_tmpl + "/";
    std::cout << "source=" << source << std::endl;
    std::cout << "dest=" << this->_mlmodel._repo + '/' + model_tmpl + ".prototxt" << std::endl;
    if (fileops::copy_file(source + model_tmpl + ".prototxt",
			   this->_mlmodel._repo + '/' + model_tmpl + ".prototxt"))
      throw MLLibBadParamException("failed to locate model template " + source + ".prototxt");
    if (fileops::copy_file(source + model_tmpl + "_solver.prototxt",
			   this->_mlmodel._repo + '/' + model_tmpl + "_solver.prototxt"))
      throw MLLibBadParamException("failed to locate solver template " + source + "_solver.prototxt");
    fileops::copy_file(source + "deploy.prototxt",
		       this->_mlmodel._repo + "/deploy.prototxt");
    this->_mlmodel.read_from_repository(this->_mlmodel._repo);
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  int CaffeLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::create_model()
  {
    if (!this->_mlmodel._def.empty() && !this->_mlmodel._weights.empty()) // whether in prediction mode...
      {
	if (_net)
	  {
	    delete _net;
	    _net = nullptr;
	  }
	_net = new Net<float>(this->_mlmodel._def);
	_net->CopyTrainedLayersFrom(this->_mlmodel._weights);
	return 0;
      }
    else if (this->_mlmodel._def.empty())
      return 2; // missing 'deploy' file.
    return 1;
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  void CaffeLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::init_mllib(const APIData &ad)
  {
    if (ad.has("gpu"))
      _gpu = ad.get("gpu").get<bool>();
    if (ad.has("gpuid"))
      _gpuid = ad.get("gpuid").get<int>();
    if (ad.has("nclasses"))
      _nclasses = ad.get("nclasses").get<int>();
    else std::cerr << "[Warning]: number of classes is undetermined in Caffe\n";
    // instantiate model template here, if any
    if (ad.has("template"))
      instantiate_template(ad);
    else // model template instantiation is defered until training call
      create_model();
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  int CaffeLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::train(const APIData &ad,
										 APIData &out)
  {
    if (this->_mlmodel._solver.empty())
      {
	throw MLLibBadParamException("missing solver file in " + this->_mlmodel._repo);
      }

    TInputConnectorStrategy inputc(this->_inputc);
    inputc._train = true;
    inputc.transform(ad);

    // instantiate model template here, as a defered from service initialization
    // since inputs are necessary in order to fit the inner net input dimension.
    if (!this->_mlmodel._model_template.empty())
      {
	// modifies model structure, template must have been copied at service creation with instantiate_template
	update_protofile_net(this->_mlmodel._repo + '/' + this->_mlmodel._model_template + ".prototxt",
			     this->_mlmodel._repo + "/deploy.prototxt",
			     inputc);
	create_model(); // creates initial net.
      }

    caffe::SolverParameter solver_param;
    caffe::ReadProtoFromTextFileOrDie(this->_mlmodel._solver,&solver_param); //TODO: no die
    update_in_memory_net_and_solver(solver_param,ad);

    // parameters
    APIData ad_mllib = ad.getobj("parameters").getobj("mllib");
#ifndef CPU_ONLY
    if (ad_mllib.has("gpu"))
      {
	bool gpu = ad_mllib.get("gpu").get<bool>();
	if (gpu)
	  {
	    if (ad_mllib.has("gpuid"))
	      Caffe::SetDevice(ad_mllib.get("gpuid").get<int>());
	    Caffe::set_mode(Caffe::GPU);
	  }
	else Caffe::set_mode(Caffe::CPU);
      }
    else
      {
	if (_gpu)
	  {
	    Caffe::SetDevice(_gpuid);
	    Caffe::set_mode(Caffe::GPU);
	  }
	else Caffe::set_mode(Caffe::CPU);
      }
#else
    Caffe::set_mode(Caffe::CPU);
#endif
    
    // solver's parameters
    APIData ad_solver = ad_mllib.getobj("solver");
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

    if (!inputc._dv.empty())
      {
	boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(solver->net()->layers()[0])->AddDatumVector(inputc._dv);
	if (!inputc._dv_test.empty())
	  boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(solver->test_nets().at(0)->layers()[0])->AddDatumVector(inputc._dv_test);
	else boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(solver->test_nets().at(0)->layers()[0])->AddDatumVector(inputc._dv);
      }
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
    //TODO: output data object

    // always save final snapshot.
    if (solver->param_.snapshot_after_train())
      solver->Snapshot();
    
    if (_net)
      {
	delete _net;
	_net = nullptr;
      }
    solver_param = caffe::SolverParameter();
    delete solver;
    this->_mlmodel.read_from_repository(this->_mlmodel._repo);
    int cm = create_model();
    if (cm == 1)
      throw MLLibInternalException("no model in " + this->_mlmodel._repo + " for initializing the net");
    else if (cm == 2)
      throw MLLibBadParamException("no deploy file in " + this->_mlmodel._repo + " for initializing the net");
    
    // test
    APIData ad_out = ad.getobj("parameters").getobj("output");
    if (ad_out.has("measure"))
      {
	boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(_net->layers()[0])->AddDatumVector(inputc._dv_test);
	float loss = 0.0;
	std::vector<Blob<float>*> results = _net->ForwardPrefilled(&loss);
	int batch_size = inputc._dv_test.size();
	int slot = results.size() - 1;
	int scount = results[slot]->count();
	int scperel = scount / batch_size;
	std::vector<double> predictions;
	std::vector<int> targets;
	APIData ad_res;
	ad_res.add("batch_size",batch_size);
	ad_res.add("nclasses",_nclasses);
	for (int j=0;j<batch_size;j++)
	  {
	    APIData bad;
	    std::vector<double> predictions;
	    int target = inputc._dv_test.at(j).label();
	    for (int k=0;k<_nclasses;k++)
	      {
		predictions.push_back(results[slot]->cpu_data()[j*scperel+k]);
	      }
	    bad.add("target",target);
	    bad.add("pred",predictions);
	    std::vector<APIData> vad = { bad };
	    ad_res.add(std::to_string(j),vad);
	  }
	this->_outputc.measure(ad_res,ad_out,out);
      }
    return 0;
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  int CaffeLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::predict(const APIData &ad,
										   APIData &out)
  {
    // check for net
    if (!_net)
      {
	int cm = create_model();
	if (cm == 1)
	  throw MLLibInternalException("no model in " + this->_mlmodel._repo + " for initializing the net");
	else if (cm == 2)
	  throw MLLibBadParamException("no deploy file in " + this->_mlmodel._repo + " for initializing the net");
      }
    
    // parameters
    APIData ad_mllib = ad.getobj("parameters").getobj("mllib");
#ifndef CPU_ONLY
    if (ad_mllib.has("gpu"))
      {
	bool gpu = ad_mllib.get("gpu").get<bool>();
	if (gpu)
	  {
	    if (ad_mllib.has("gpuid"))
	      Caffe::SetDevice(ad_mllib.get("gpuid").get<int>());
	    Caffe::set_mode(Caffe::GPU);
	  }
	else Caffe::set_mode(Caffe::CPU);       
      }
    else
      {
	if (_gpu)
	  {
	    Caffe::SetDevice(_gpuid);
	    Caffe::set_mode(Caffe::GPU);
	  }
	else Caffe::set_mode(Caffe::CPU);
      }
#else
      Caffe::set_mode(Caffe::CPU);
#endif

    TInputConnectorStrategy inputc(this->_inputc);
    inputc.transform(ad); //TODO: catch errors ?
    int batch_size = inputc.batch_size();
    boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(_net->layers()[0])->AddDatumVector(inputc._dv);
    
    // with addmat (PR)
    //std::vector<int> dvl(batch_size,0.0);
    //boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(_net->layers()[0])->AddMatVector(inputc._images,dvl); // Caffe will crash with gtest or sigsegv here if input size is incorrect.
    //std::vector<Blob<float>*> results = _net->Forward(bottom,&loss);
    
    float loss = 0.0;
    //std::chrono::time_point<std::chrono::system_clock> tstart = std::chrono::system_clock::now();
    std::vector<Blob<float>*> results = _net->ForwardPrefilled(&loss); // XXX: on a batch, are we getting the average loss ?
    /*std::chrono::time_point<std::chrono::system_clock> tstop = std::chrono::system_clock::now();
      double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(tstop-tstart).count();*/
    //std::cout << "Caffe prediction time=" << elapsed << std::endl;
    int slot = results.size() - 1;
    /*std::cout << "results size=" << results.size() << std::endl;
      std::cout << "count=" << results[slot]->count() << std::endl;*/
    int scount = results[slot]->count();
    int scperel = scount / batch_size;
    int nclasses = _nclasses > 0 ? _nclasses : scperel; // XXX: beware of scperel as it can refer to the number of neurons is last layer before softmax, which is replaced 'in-place' with probabilities after softmax. Weird by Caffe... */
    TOutputConnectorStrategy tout;
    for (int j=0;j<batch_size;j++)
      {
	tout.add_result(inputc._ids.at(j),loss);
	for (int i=0;i<nclasses;i++)
	  {
	    tout.add_cat(inputc._ids.at(j),results[slot]->cpu_data()[j*scperel+i],this->_mlmodel.get_hcorresp(i));
	  }
      }
    TOutputConnectorStrategy btout(this->_outputc);
    tout.best_cats(ad.getobj("parameters").getobj("output"),btout);
    btout.to_ad(out);
    out.add("status",0);
    
    return 0;
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  void CaffeLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::update_in_memory_net_and_solver(caffe::SolverParameter &sp,
													    const APIData &ad)
  {
    // fix net model path.
    sp.set_net(this->_mlmodel._repo + "/" + sp.net());
    
    // fix net snapshot path.
    sp.set_snapshot_prefix(this->_mlmodel._repo + "/model");
    
    // acquire custom batch size if any
    APIData ad_net = ad.getobj("parameters").getobj("mllib").getobj("net");
    int batch_size = -1;
    if (ad_net.has("batch_size"))
      batch_size = ad_net.get("batch_size").get<int>();

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
		dp->set_source(this->_mlmodel._repo + "/" + dp->source()); // this updates in-memory net
	      }
	    if (dp->has_batch_size() && batch_size > 0)
	      {
		dp->set_batch_size(batch_size);
	      }
	  }
	else if (lp->has_memory_data_param())
	  {
	    caffe::MemoryDataParameter *mdp = lp->mutable_memory_data_param();
	    if (mdp->has_batch_size() && batch_size > 0)
	      {
		mdp->set_batch_size(batch_size);
	      }
	  }
      }
    sp.clear_net();
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  void CaffeLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::update_protofile_net(const std::string &net_file,
												 const std::string &deploy_file,
												 const TInputConnectorStrategy &inputc)
  {
    //TODO: get "parameters/mllib/net" from ad (e.g. for batch_size).

    caffe::NetParameter net_param;
    caffe::ReadProtoFromTextFile(net_file,&net_param);
    net_param.mutable_layers(0)->mutable_memory_data_param()->set_channels(inputc.feature_size());
    net_param.mutable_layers(1)->mutable_memory_data_param()->set_channels(inputc.feature_size()); // test layer
    caffe::WriteProtoToTextFile(net_param,net_file);
    
    caffe::ReadProtoFromTextFile(deploy_file,&net_param);
    net_param.mutable_layers(0)->mutable_memory_data_param()->set_channels(inputc.feature_size());
    caffe::WriteProtoToTextFile(net_param,deploy_file);
  }

  template class CaffeLib<ImgCaffeInputFileConn,SupervisedOutput,CaffeModel>;
  template class CaffeLib<CSVCaffeInputFileConn,SupervisedOutput,CaffeModel>;
}
