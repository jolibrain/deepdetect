/**
 * DeepDetect
 * Copyright (c) 2014-2015 Emmanuel Benazera
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
#include "utils/utils.hpp"
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
    _regression = cl._regression;
    _ntargets = cl._ntargets;
    cl._net = nullptr;
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  CaffeLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::~CaffeLib()
  {
    delete _net;
    _net = nullptr;
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  void CaffeLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::instantiate_template(const APIData &ad)
  {
    // - locate template repository
    std::string model_tmpl = ad.get("template").get<std::string>();
    this->_mlmodel._model_template = model_tmpl;
    std::cout << "instantiating model template " << model_tmpl << std::endl;

    // - copy files to model repository
    std::string source = this->_mlmodel._mlmodel_template_repo + model_tmpl + "/";
    LOG(INFO) << "source=" << source << std::endl;
    LOG(INFO) << "dest=" << this->_mlmodel._repo + '/' + model_tmpl + ".prototxt" << std::endl;
    std::string dest_net = this->_mlmodel._repo + '/' + model_tmpl + ".prototxt";
    std::string dest_deploy_net = this->_mlmodel._repo + "/deploy.prototxt";
    int err = fileops::copy_file(source + model_tmpl + ".prototxt", dest_net);
    if (err == 1)
      throw MLLibBadParamException("failed to locate model template " + source + ".prototxt");
    else if (err == 2)
      throw MLLibBadParamException("failed to create model template destination " + dest_net);
    err = fileops::copy_file(source + model_tmpl + "_solver.prototxt",
			     this->_mlmodel._repo + '/' + model_tmpl + "_solver.prototxt");
    if (err == 1)
      throw MLLibBadParamException("failed to locate solver template " + source + model_tmpl + "_solver.prototxt");
    else if (err == 2)
      throw MLLibBadParamException("failed to create destination template solver file " + this->_mlmodel._repo + '/' + model_tmpl + "_solver.prototxt");
    err = fileops::copy_file(source + "deploy.prototxt", dest_deploy_net);
    if (err == 1)
      throw MLLibBadParamException("failed to locate deploy template " + source + "deploy.prototxt");
    else if (err == 2)
      throw MLLibBadParamException("failed to create destination deploy solver file " + dest_deploy_net);

    // if mlp template, set the net structure as number of layers.
    if (model_tmpl == "mlp" || model_tmpl == "mlp_db")
      {
	caffe::NetParameter net_param,deploy_net_param;
	caffe::ReadProtoFromTextFile(dest_net,&net_param); //TODO: catch parsing error (returns bool true on success)
	caffe::ReadProtoFromTextFile(dest_deploy_net,&deploy_net_param);
	configure_mlp_template(ad,_regression,_ntargets,_nclasses,net_param,deploy_net_param);
	caffe::WriteProtoToTextFile(net_param,dest_net);
	caffe::WriteProtoToTextFile(deploy_net_param,dest_deploy_net);
      }
    else if (model_tmpl == "convnet")
      {
	caffe::NetParameter net_param,deploy_net_param;
	caffe::ReadProtoFromTextFile(dest_net,&net_param); //TODO: catch parsing error (returns bool true on success)
	caffe::ReadProtoFromTextFile(dest_deploy_net,&deploy_net_param);
	configure_convnet_template(ad,_regression,_ntargets,_nclasses,this->_inputc,net_param,deploy_net_param);
	caffe::WriteProtoToTextFile(net_param,dest_net);
	caffe::WriteProtoToTextFile(deploy_net_param,dest_deploy_net);
      }
    else
      {
	caffe::NetParameter net_param,deploy_net_param;
	caffe::ReadProtoFromTextFile(dest_net,&net_param); //TODO: catch parsing error (returns bool true on success)
	caffe::ReadProtoFromTextFile(dest_deploy_net,&deploy_net_param);
	if ((ad.has("rotate") && ad.get("rotate").get<bool>()) 
	    || (ad.has("mirror") && ad.get("mirror").get<bool>()))
	  {
	    caffe::LayerParameter *lparam = net_param.mutable_layer(0); // data input layer
	    lparam->mutable_transform_param()->set_mirror(ad.get("mirror").get<bool>());
	    lparam->mutable_transform_param()->set_rotate(ad.get("rotate").get<bool>());
	  }
	// adapt number of neuron output
	update_protofile_classes(net_param);
	update_protofile_classes(deploy_net_param);
	caffe::WriteProtoToTextFile(net_param,dest_net);
	caffe::WriteProtoToTextFile(deploy_net_param,dest_deploy_net);
      }
    if (ad.has("finetuning") && ad.get("finetuning").get<bool>())
      {
	if (!ad.has("weights"))
	  throw MLLibBadParamException("finetuning requires specifying an existing weights file");	
	caffe::NetParameter net_param,deploy_net_param;
	caffe::ReadProtoFromTextFile(dest_net,&net_param); //TODO: catch parsing error (returns bool true on success)
	caffe::ReadProtoFromTextFile(dest_deploy_net,&deploy_net_param);
	update_protofile_finetune(net_param);
	update_protofile_finetune(deploy_net_param);
	caffe::WriteProtoToTextFile(net_param,dest_net);
	caffe::WriteProtoToTextFile(deploy_net_param,dest_deploy_net);
      }
    
    this->_mlmodel.read_from_repository(this->_mlmodel._repo);
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  void CaffeLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::configure_mlp_template(const APIData &ad,
												   const bool &regression,
												   const int &targets,
												   const int &cnclasses,
												   caffe::NetParameter &net_param,
												   caffe::NetParameter &deploy_net_param)
  {
    std::vector<int> layers = {50};
    std::string activation = "ReLU";
    double elu_alpha = 1.0;
    double dropout = 0.5;
    if (ad.has("layers"))
      layers = ad.get("layers").get<std::vector<int>>();
    if (ad.has("activation"))
      {
	activation = ad.get("activation").get<std::string>();
	if (dd_utils::iequals(activation,"relu"))
	  activation = "ReLU";
	else if (dd_utils::iequals(activation,"prelu"))
	  activation = "PReLU";
	else if (dd_utils::iequals(activation,"elu"))
	  {
	    activation = "ELU";
	    if (ad.has("elu_alpha"))
	      elu_alpha = ad.get("elu_alpha").get<double>();
	  }
	else if (dd_utils::iequals(activation,"sigmoid"))
	  activation = "Sigmoid";
	else if (dd_utils::iequals(activation,"tanh"))
	  activation = "TanH";
      }
    if (ad.has("dropout"))
      dropout = ad.get("dropout").get<double>();
    bool db = false;
    if (ad.has("db") && ad.get("db").get<bool>())
      db = true;
    if (!db && layers.empty() && activation == "ReLU" && dropout == 0.5 && targets == 0)
      return; // nothing to do
    
    int nclasses = 0;
    int rl = 2;
    int drl = 1;
    int max_rl = 8;
    int max_drl = 5;
    caffe::LayerParameter *lparam = nullptr;
    caffe::LayerParameter *dlparam = nullptr;
    std::string prec_ip = "data";
    std::string last_ip = "ip0";
    for (size_t l=0;l<layers.size();l++)
      {
	if (l == 0)
	  {
	    lparam = net_param.mutable_layer(6);
	    if (!cnclasses) // if unknown we keep the default one
	      nclasses = lparam->mutable_inner_product_param()->num_output();
	    else nclasses = cnclasses;
	    if (targets > 1)
	      {
		lparam = net_param.mutable_layer(0);
		lparam->set_top(0,"fulldata");
		lparam->set_top(1,"fake_label");
		if (!db)
		  lparam->mutable_memory_data_param()->set_channels(targets); // XXX: temporary value, set at training time
		else
		  {
		    lparam->clear_memory_data_param();
		    lparam->set_type("Data");
		    caffe::DataParameter *ldparam = lparam->mutable_data_param();
		    ldparam->set_source("train.lmdb");
		    ldparam->set_batch_size(1000); // dummy value, updated before training
		    ldparam->set_backend(caffe::DataParameter_DB_LMDB);
		  }
		lparam = net_param.mutable_layer(1);
		lparam->set_top(0,"fulldata");
		lparam->set_top(1,"fake_label");
		lparam->mutable_memory_data_param()->set_channels(targets);
		dlparam = deploy_net_param.mutable_layer(0);
		dlparam->set_top(0,"fulldata");
		dlparam->set_top(1,"fake_label");
		dlparam->mutable_memory_data_param()->set_channels(targets);
		
		lparam = net_param.mutable_layer(rl);
		lparam->clear_include();
		lparam->clear_top();
		lparam->clear_bottom();
		lparam->clear_inner_product_param();
		lparam->clear_dropout_param();
		lparam->clear_loss_weight();
		lparam->set_name("slice_labels");
		lparam->set_type("Slice");
		lparam->add_bottom("fulldata");
		lparam->add_top(prec_ip);
		lparam->add_top("label");
		caffe::SliceParameter *spp = lparam->mutable_slice_param();
		spp->set_slice_dim(1);
		spp->add_slice_point(nclasses);
		++rl;

		dlparam = deploy_net_param.mutable_layer(drl);
		dlparam->clear_include();
		dlparam->clear_top();
		dlparam->clear_bottom();
		dlparam->clear_inner_product_param();
		dlparam->clear_dropout_param();
		dlparam->clear_loss_weight();
		dlparam->set_name("slice_labels");
		dlparam->set_type("Slice");
		dlparam->add_bottom("fulldata");
		dlparam->add_top(prec_ip);
		dlparam->add_top("label");
		spp = dlparam->mutable_slice_param();
		spp->set_slice_dim(1);
		spp->add_slice_point(nclasses); // XXX: temporary value
		++drl;
	      }
	    else if (db)
	      {
		// fixing input layer so that it takes data in from db
		lparam = net_param.mutable_layer(0);
		lparam->clear_memory_data_param();
		lparam->set_type("Data");
		caffe::DataParameter *ldparam = lparam->mutable_data_param();
		ldparam->set_source("train.lmdb");
		ldparam->set_batch_size(1000); // dummy value, updated before training
		ldparam->set_backend(caffe::DataParameter_DB_LMDB);
	      }
	  }
	else if (l > 0)
	  {
	    prec_ip = "ip" + std::to_string(l-1);
	    last_ip = "ip" + std::to_string(l);
	  }

	if (rl < max_rl)
	  {
	    lparam = net_param.mutable_layer(rl);
	    lparam->clear_include();
	    lparam->clear_top();
	    lparam->clear_bottom();
	    lparam->clear_inner_product_param();
	    lparam->clear_dropout_param();
	    lparam->clear_loss_weight();
	  }
	else lparam = net_param.add_layer();
	lparam->set_name(last_ip);
	lparam->set_type("InnerProduct");
	lparam->add_bottom(prec_ip);
	lparam->add_top(last_ip);
	caffe::InnerProductParameter *ipp = lparam->mutable_inner_product_param();
	ipp->set_num_output(layers.at(l));
	ipp->mutable_weight_filler()->set_type("xavier");
	ipp->mutable_bias_filler()->set_type("constant");
	++rl;
	
	if (drl < max_drl)
	  {
	    dlparam = deploy_net_param.mutable_layer(drl);
	    dlparam->clear_include();
	    dlparam->clear_top();
	    dlparam->clear_bottom();
	    dlparam->clear_inner_product_param();
	    dlparam->clear_dropout_param();
	    dlparam->clear_loss_weight();
	  }
	else dlparam = deploy_net_param.add_layer();
	dlparam->set_name(last_ip);
	dlparam->set_type("InnerProduct");
	dlparam->add_bottom(prec_ip);
	dlparam->add_top(last_ip);
	ipp = dlparam->mutable_inner_product_param();
	ipp->set_num_output(layers.at(l));
	ipp->mutable_weight_filler()->set_type("xavier");
	ipp->mutable_bias_filler()->set_type("constant");
	++drl;
	
	if (rl < max_rl)
	  {
	    lparam = net_param.mutable_layer(rl);
	    lparam->clear_include();
	    lparam->clear_top();
	    lparam->clear_bottom();
	    lparam->clear_loss_weight();
	    lparam->clear_dropout_param();
	    lparam->clear_inner_product_param();
	  }
	else lparam = net_param.add_layer();
	lparam->set_name("act"+std::to_string(l));
	lparam->set_type(activation);
	if (activation == "ELU" && elu_alpha != 1.0)
	  lparam->mutable_elu_param()->set_alpha(elu_alpha);
	lparam->add_bottom(last_ip);
	lparam->add_top(last_ip);
	++rl;

	if (drl < max_drl)
	  {
	    dlparam = deploy_net_param.mutable_layer(drl);
	    dlparam->clear_include();
	    dlparam->clear_top();
	    dlparam->clear_bottom();
	    dlparam->clear_loss_weight();
	    dlparam->clear_dropout_param();
	    dlparam->clear_inner_product_param();
	  }
	else dlparam = deploy_net_param.add_layer();
	dlparam->set_name("act"+std::to_string(l));
	dlparam->set_type(activation);
	if (activation == "ELU" && elu_alpha != 1.0)
	  lparam->mutable_elu_param()->set_alpha(elu_alpha);
	dlparam->add_bottom(last_ip);
	dlparam->add_top(last_ip);
	++drl;
	
	if (dropout > 0.0 && dropout < 1.0)
	  {
	    if (rl < max_rl)
	      {
		lparam = net_param.mutable_layer(rl);
		lparam->clear_include();
		lparam->clear_bottom();
		lparam->clear_top();
		lparam->clear_loss_weight();
		lparam->clear_dropout_param();
		lparam->clear_inner_product_param();
	      }
	    else lparam = net_param.add_layer(); // dropout layer
	    lparam->set_name("drop"+std::to_string(l));
	    lparam->set_type("Dropout");
	    lparam->add_bottom(last_ip);
	    lparam->add_top(last_ip);
	    lparam->mutable_dropout_param()->set_dropout_ratio(dropout);
	    ++rl;
	  }
      }

    // add remaining softmax layers
    prec_ip = "ip" + std::to_string(layers.size()-1);
    last_ip = "ip" + std::to_string(layers.size());
    if (rl < max_rl)
      {
	lparam = net_param.mutable_layer(rl); // last inner product before softmax
	lparam->clear_include();
	lparam->clear_bottom();
	lparam->clear_top();
	lparam->clear_loss_weight();
	lparam->clear_dropout_param();
	lparam->clear_inner_product_param();
      }
    else lparam = net_param.add_layer();
    lparam->set_name(last_ip);
    lparam->set_type("InnerProduct");
    lparam->add_bottom(prec_ip);
    lparam->add_top(last_ip);
    caffe::InnerProductParameter *ipp = lparam->mutable_inner_product_param();
    if (!regression || targets == 0)
      ipp->set_num_output(nclasses);
    else ipp->set_num_output(targets);
    ipp->mutable_weight_filler()->set_type("gaussian");
    ipp->mutable_weight_filler()->set_std(0.1);
    ipp->mutable_bias_filler()->set_type("constant");
    ++rl;

    if (drl < max_drl)
      {
	dlparam = deploy_net_param.mutable_layer(drl);
	dlparam->clear_include();
	dlparam->clear_top();
	dlparam->clear_bottom();
	dlparam->clear_loss_weight();
	dlparam->clear_dropout_param();
	dlparam->clear_inner_product_param();
      }
    else dlparam = deploy_net_param.add_layer();
    dlparam->set_name(last_ip);
    dlparam->set_type("InnerProduct");
    dlparam->add_bottom(prec_ip);
    dlparam->add_top(last_ip);
    caffe::InnerProductParameter *dipp = dlparam->mutable_inner_product_param();
    if (!regression || targets == 0)
      dipp->set_num_output(nclasses);
    else dipp->set_num_output(targets);
    dipp->mutable_weight_filler()->set_type("gaussian");
    dipp->mutable_weight_filler()->set_std(0.1);
    dipp->mutable_bias_filler()->set_type("constant");
    ++drl;

    if (!regression)
      {
	if (rl < max_rl)
	  {
	    lparam = net_param.mutable_layer(rl);
	    lparam->clear_include();
	    lparam->clear_bottom();
	    lparam->clear_top();
	    lparam->clear_loss_weight();
	    lparam->clear_dropout_param();
	    lparam->clear_inner_product_param();
	  }
	else lparam = net_param.add_layer(); // test loss
	lparam->set_name("losst");
        lparam->set_type("Softmax");
	lparam->add_bottom(last_ip);
	lparam->add_top("losst");
	caffe::NetStateRule *nsr = lparam->add_include();
	nsr->set_phase(caffe::TEST);
	++rl;
      }

    if (rl < max_rl)
      {
	lparam = net_param.mutable_layer(rl);
	lparam->clear_include();
	lparam->clear_bottom();
	lparam->clear_top();
	lparam->clear_loss_weight();
	lparam->clear_dropout_param();
	lparam->clear_inner_product_param();
      }
    else lparam = net_param.add_layer(); // training loss
    lparam->set_name("loss");
    if (regression)
      {
	lparam->set_type("EuclideanLoss");
	lparam->add_include()->set_phase(caffe::TRAIN);
      }
    else lparam->set_type("SoftmaxWithLoss");
    lparam->add_bottom(last_ip);
    lparam->add_bottom("label");
    lparam->add_top("loss");
    caffe::NetStateRule *nsr = lparam->add_include();
    nsr->set_phase(caffe::TRAIN);
    ++rl;
    
    if (!regression)
      {
	if (drl < max_drl)
	  {
	    dlparam = deploy_net_param.mutable_layer(drl);
	    dlparam->clear_include();
	    dlparam->clear_top();
	    dlparam->clear_bottom();
	    dlparam->clear_loss_weight();
	    dlparam->clear_dropout_param();
	    dlparam->clear_inner_product_param();
	  }
	else dlparam = deploy_net_param.add_layer();
	dlparam->set_name("loss");
	dlparam->set_type("Softmax");
	dlparam->add_bottom(last_ip);
	dlparam->add_top("loss");
      }
  }
  
  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  void CaffeLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::configure_convnet_template(const APIData &ad,
												       const bool &regression,
												       const int &targets,
												       const int &cnclasses,
												       const TInputConnectorStrategy &inputc,
												       caffe::NetParameter &net_param,
												       caffe::NetParameter &deploy_net_param)
  {
    //- get relevant configuration elements
    std::vector<std::string> layers;
    std::string activation = "ReLU";
    double elu_alpha = 1.0;
    double dropout = 0.5;
    if (ad.has("layers"))
      try
	{
	  layers = ad.get("layers").get<std::vector<std::string>>();
	}
      catch(std::exception &e)
	{
	  throw MLLibBadParamException("convnet template requires specifying a string array of layers");
	}
    if (ad.has("activation"))
      {
	activation = ad.get("activation").get<std::string>();
	if (dd_utils::iequals(activation,"relu"))
	  activation = "ReLU";
	else if (dd_utils::iequals(activation,"prelu"))
	  activation = "PReLU";
	else if (dd_utils::iequals(activation,"elu"))
	  {
	    activation = "ELU";
	    if (ad.has("elu_alpha"))
	      elu_alpha = ad.get("elu_alpha").get<double>();
	  }
	else if (dd_utils::iequals(activation,"sigmoid"))
	  activation = "Sigmoid";
	else if (dd_utils::iequals(activation,"tanh"))
	  activation = "TanH";
      }
    if (ad.has("dropout"))
      dropout = ad.get("dropout").get<double>();
    bool db = inputc._db;
    if (ad.has("db") && ad.get("db").get<bool>())
      db = true;
    if (!db && layers.empty() && activation == "ReLU" && dropout == 0.5)
      return; // nothing to do

    const std::string cr_str = "CR";
    const std::string p_str = "P";
    std::vector<std::pair<int,int>> cr_layers; // conv + activation
    std::vector<int> fc_layers; // fully connected
    for (auto s: layers)
      {
	size_t pos = 0;
	if ((pos=s.find(cr_str))!=std::string::npos)
	  {
	    std::string ncr = s.substr(0,pos);
	    std::string crs = s.substr(pos+cr_str.size());
	    cr_layers.push_back(std::pair<int,int>(std::atoi(ncr.c_str()),std::atoi(crs.c_str())));
	  }
	else
	  {
	    try
	      {
		fc_layers.push_back(std::atoi(s.c_str()));
	      }
	    catch(std::exception &e)
	      {
		throw MLLibBadParamException("convnet template requires fully connected layers size to be specified as a string");
	      }
	  }
      }

    // default params
    uint32_t conv_kernel_size = 3;
    uint32_t conv1d_early_kernel_size = 7;
    std::string conv_wfill_type = "xavier";
    double conv_wfill_std = 0.001;
    std::string conv_b_type = "constant";
    caffe::PoolingParameter_PoolMethod pool_type = caffe::PoolingParameter_PoolMethod_MAX;
    bool flat1dconv = inputc._flat1dconv; // whether the model uses 1d-conv (e.g. character-level convnet for text)
    int pool_kernel_size = 2;
    int pool_stride = 2;
    int nclasses = 0;
    int rl = 2;
    int drl = 1;
    int max_rl = 9;
    int max_drl = 6;
    caffe::LayerParameter *lparam = nullptr;
    caffe::LayerParameter *dlparam = nullptr;
    int ccount = 0;
    std::string prec_ip = "data";
    std::string last_ip = "conv0";
    for (size_t l=0;l<cr_layers.size();l++)
      {
	if (l == 0)
	  {
	    lparam = net_param.mutable_layer(6);
	    if (!cnclasses) // if unknown we keep the default one
	      nclasses = lparam->mutable_inner_product_param()->num_output();
	    else nclasses = cnclasses;
	    if (targets > 1)
	      {
		lparam = net_param.mutable_layer(0);
		lparam->set_top(0,"fulldata");
		lparam->set_top(1,"fake_label");
		if (!db)
		  lparam->mutable_memory_data_param()->set_channels(targets); // XXX: temporary value
		else
		  {
		    if (flat1dconv) // not dealing with images
		      lparam->clear_transform_param();
		    lparam->clear_memory_data_param();
		    lparam->clear_loss_weight();
		    lparam->set_type("Data");
		    caffe::DataParameter *ldparam = lparam->mutable_data_param();
		    ldparam->set_source("train.lmdb");
		    ldparam->set_batch_size(1000); // dummy value, updated before training
		    ldparam->set_backend(caffe::DataParameter_DB_LMDB);
		  }
		lparam = net_param.mutable_layer(1);
		lparam->set_top(0,"fulldata");
		lparam->set_top(1,"fake_label");
		lparam->mutable_memory_data_param()->set_channels(targets);
		dlparam = deploy_net_param.mutable_layer(0);
		dlparam->set_top(0,"fulldata");
		dlparam->set_top(1,"fake_label");
		dlparam->mutable_memory_data_param()->set_channels(targets);
		
		lparam = net_param.mutable_layer(rl);
		lparam->clear_include();
		lparam->clear_top();
		lparam->clear_bottom();
		lparam->clear_inner_product_param();
		lparam->clear_dropout_param();
		lparam->clear_loss_weight();
		lparam->set_name("slice_labels");
		lparam->set_type("Slice");
		lparam->add_bottom("fulldata");
		lparam->add_top("data");
		lparam->add_top("label");
		caffe::SliceParameter *spp = lparam->mutable_slice_param();
		spp->set_slice_dim(1);
		spp->add_slice_point(nclasses); // XXX: temporary value
		++rl;

		dlparam = deploy_net_param.mutable_layer(drl);
		dlparam->clear_include();
		dlparam->clear_top();
		dlparam->clear_bottom();
		dlparam->clear_inner_product_param();
		dlparam->clear_dropout_param();
		dlparam->clear_loss_weight();
		dlparam->set_name("slice_labels");
		dlparam->set_type("Slice");
		dlparam->add_bottom("fulldata");
		dlparam->add_top("data");
		dlparam->add_top("label");
		spp = dlparam->mutable_slice_param();
		spp->set_slice_dim(1);
		spp->add_slice_point(nclasses);
		++drl;
	      }
	    else if (db)
	      {
		// fixing input layer so that it takes data in from db
		lparam = net_param.mutable_layer(0);
		if (!flat1dconv)
		  {
		    if (ad.has("rotate") || ad.has("mirror"))
		      {
			lparam->mutable_transform_param()->set_mirror(ad.get("mirror").get<bool>());
			lparam->mutable_transform_param()->set_rotate(ad.get("rotate").get<bool>());
		      }
		    std::string mf = "mean.binaryproto";
		    lparam->mutable_transform_param()->set_mean_file(mf.c_str());
		  }
		else if (flat1dconv) // not dealing with images
		  lparam->clear_transform_param();
		lparam->clear_memory_data_param();
		lparam->clear_loss_weight();
		lparam->set_type("Data");
		caffe::DataParameter *ldparam = lparam->mutable_data_param();
		ldparam->set_source("train.lmdb");
		ldparam->set_batch_size(1000); // dummy value, updated before training
		ldparam->set_backend(caffe::DataParameter_DB_LMDB);
	      }
	    else if (!db && flat1dconv)
	      {
		// train
		lparam = net_param.mutable_layer(0);
		lparam->mutable_memory_data_param()->set_channels(1);
		lparam->mutable_memory_data_param()->set_height(inputc.height());
		lparam->mutable_memory_data_param()->set_width(inputc.width());
	      }
	    if (flat1dconv)
	      {
		// test
		lparam = net_param.mutable_layer(1);
		lparam->mutable_memory_data_param()->set_channels(1);
		lparam->mutable_memory_data_param()->set_height(inputc.height());
		lparam->mutable_memory_data_param()->set_width(inputc.width());
		// deploy
		lparam = deploy_net_param.mutable_layer(0);
		lparam->mutable_memory_data_param()->set_channels(1);
		lparam->mutable_memory_data_param()->set_height(inputc.height());
		lparam->mutable_memory_data_param()->set_width(inputc.width());
	      }
	  }
	else if (l > 0)
	  {
	    prec_ip = "pool" + std::to_string(l-1);
	    last_ip = "conv" + std::to_string(ccount);
	  }
	int nconv = cr_layers.at(l).first;
	for (int c=0;c<nconv;c++)
	  {
	    if (rl < max_rl)
	      {
		lparam = net_param.mutable_layer(rl);
		lparam->clear_include();
		lparam->clear_top();
		lparam->clear_bottom();
		lparam->clear_inner_product_param();
		lparam->clear_pooling_param();
		lparam->clear_dropout_param();
		lparam->clear_loss_weight();
	      }
	    else lparam = net_param.add_layer();
	    lparam->set_name(last_ip);
	    lparam->set_type("Convolution");
	    lparam->add_bottom(prec_ip);
	    lparam->add_top(last_ip);
	    lparam->mutable_convolution_param()->set_num_output(cr_layers.at(l).second);
	    if (flat1dconv)
	      {
		lparam->mutable_convolution_param()->clear_kernel_size();
		lparam->mutable_convolution_param()->set_kernel_h(ccount < 2 ? conv1d_early_kernel_size : conv_kernel_size);
		if (prec_ip == "data")
		  lparam->mutable_convolution_param()->set_kernel_w(inputc.width());
		else lparam->mutable_convolution_param()->set_kernel_w(1);
	      }
	    else if (!lparam->mutable_convolution_param()->kernel_size_size())
	      lparam->mutable_convolution_param()->add_kernel_size(conv_kernel_size);
	    lparam->mutable_convolution_param()->mutable_weight_filler()->set_type(conv_wfill_type);
	    lparam->mutable_convolution_param()->mutable_weight_filler()->set_std(conv_wfill_std);
	    lparam->mutable_convolution_param()->mutable_bias_filler()->set_type(conv_b_type);
	    //TODO: auto compute best padding value
	    ++rl;
	    
	    if (drl < max_drl)
	      {
		dlparam = deploy_net_param.mutable_layer(drl);
		dlparam->clear_include();
		dlparam->clear_top();
		dlparam->clear_bottom();
		dlparam->clear_inner_product_param();
		dlparam->clear_pooling_param();
		dlparam->clear_dropout_param();
		dlparam->clear_loss_weight();
	      }
	    else dlparam = deploy_net_param.add_layer();
	    dlparam->set_name(last_ip);
	    dlparam->set_type("Convolution");
	    dlparam->add_top(last_ip);
	    dlparam->add_bottom(prec_ip);
	    dlparam->mutable_convolution_param()->set_num_output(cr_layers.at(l).second);
	    if (flat1dconv)
	      {
		dlparam->mutable_convolution_param()->clear_kernel_size();
		dlparam->mutable_convolution_param()->set_kernel_h(ccount < 2 ? conv1d_early_kernel_size : conv_kernel_size);
		if (prec_ip == "data")
		  dlparam->mutable_convolution_param()->set_kernel_w(inputc.width());
		else dlparam->mutable_convolution_param()->set_kernel_w(1);
	      }
	    else if (!dlparam->mutable_convolution_param()->kernel_size_size())
	      dlparam->mutable_convolution_param()->add_kernel_size(conv_kernel_size);
	    dlparam->mutable_convolution_param()->mutable_weight_filler()->set_type(conv_wfill_type);
	    dlparam->mutable_convolution_param()->mutable_weight_filler()->set_std(conv_wfill_std);
	    dlparam->mutable_convolution_param()->mutable_bias_filler()->set_type(conv_b_type);
	    ++drl;
	    
	    if (rl < max_rl)
	      {
		lparam = net_param.mutable_layer(rl);
		lparam->clear_include();
		lparam->clear_top();
		lparam->clear_bottom();
		lparam->clear_loss_weight();
		lparam->clear_dropout_param();
		lparam->clear_inner_product_param();
	      }
	    else lparam = net_param.add_layer();
	    lparam->set_name("act"+std::to_string(ccount));
	    lparam->set_type(activation);
	    if (activation == "ELU" && elu_alpha != 1.0)
	      lparam->mutable_elu_param()->set_alpha(elu_alpha);
	    lparam->add_bottom("conv"+std::to_string(ccount));
	    lparam->add_top("conv"+std::to_string(ccount));
	    ++rl;
	    
	    if (drl < max_drl)
	      {
		dlparam = deploy_net_param.mutable_layer(drl);
		dlparam->clear_include();
		dlparam->clear_top();
		dlparam->clear_bottom();
		dlparam->clear_loss_weight();
		dlparam->clear_dropout_param();
		dlparam->clear_inner_product_param();
	      }
	    else dlparam = deploy_net_param.add_layer();
	    dlparam->set_name("act"+std::to_string(ccount));
	    dlparam->set_type(activation);
	    if (activation == "ELU" && elu_alpha != 1.0)
	      lparam->mutable_elu_param()->set_alpha(elu_alpha);
	    dlparam->add_bottom("conv"+std::to_string(ccount));
	    dlparam->add_top("conv"+std::to_string(ccount));
	    ++drl;
	    
	    prec_ip = "conv" + std::to_string(ccount);
	    ++ccount;
	    last_ip = "conv" + std::to_string(ccount);
	  }
	
	std::string cum = std::to_string(ccount-1);
	std::string lcum = std::to_string(l);
	if (rl < max_rl)
	  {
	    lparam = net_param.mutable_layer(rl);
	    lparam->clear_inner_product_param();
	    lparam->clear_bottom();
	    lparam->clear_top();
	    lparam->clear_include();
	    lparam->clear_loss_weight();
	  }
	else lparam = net_param.add_layer();
	lparam->set_name("pool"+lcum);
	lparam->set_type("Pooling");
	lparam->add_bottom("conv"+cum);
	lparam->add_top("pool"+lcum);
	lparam->mutable_pooling_param()->set_pool(pool_type);
	if (flat1dconv)
	  {
	    lparam->mutable_pooling_param()->clear_stride();
	    lparam->mutable_pooling_param()->clear_kernel_size();
	    lparam->mutable_pooling_param()->set_stride_h(3);
	    lparam->mutable_pooling_param()->set_stride_w(1);
	    lparam->mutable_pooling_param()->set_kernel_h(3);
	    lparam->mutable_pooling_param()->set_kernel_w(1);
	  }
	else 
	  {
	    lparam->mutable_pooling_param()->set_kernel_size(pool_kernel_size);
	    lparam->mutable_pooling_param()->set_stride(pool_stride);
	  }
	++rl;
	
	if (drl < max_drl)
	  {
	    dlparam = deploy_net_param.mutable_layer(drl);
	    dlparam->clear_inner_product_param();
	    dlparam->clear_bottom();
	    dlparam->clear_top();
	    dlparam->clear_include();
	    dlparam->clear_loss_weight();
	  }
	else dlparam = deploy_net_param.add_layer(); // pooling
	dlparam->set_name("pool"+lcum);
	dlparam->set_type("Pooling");
	dlparam->add_bottom("conv"+cum);
	dlparam->add_top("pool"+lcum);
	dlparam->mutable_pooling_param()->set_pool(pool_type);
	if (flat1dconv)
	  {
	    dlparam->mutable_pooling_param()->clear_stride();
	    dlparam->mutable_pooling_param()->clear_kernel_size();
	    dlparam->mutable_pooling_param()->set_stride_h(3);
	    dlparam->mutable_pooling_param()->set_stride_w(1);
	    dlparam->mutable_pooling_param()->set_kernel_h(3);
	    dlparam->mutable_pooling_param()->set_kernel_w(1);
	  }
	else 
	  {
	    dlparam->mutable_pooling_param()->set_kernel_size(pool_kernel_size);
	    dlparam->mutable_pooling_param()->set_stride(pool_stride);
	  }
	++drl;

	/*if (dropout > 0.0 && dropout < 1.0)
	  {
	    if (rl < max_rl)
	      {
		lparam = net_param.mutable_layer(rl);
		lparam->clear_bottom();
		lparam->clear_top();
		lparam->clear_loss_weight();
	      }
	    else lparam = net_param.add_layer(); // dropout layer
	    lparam->set_name("drop"+lcum);
	    lparam->set_type("Dropout");
	    lparam->add_bottom("pool"+lcum);
	    lparam->add_top("pool"+lcum);
	    lparam->mutable_dropout_param()->set_dropout_ratio(dropout);
	    ++rl;
	    }*/
      }

    prec_ip = "pool" + std::to_string(cr_layers.size()-1);
    last_ip = "ip" + std::to_string(cr_layers.size());
    int lfc = cr_layers.size();
    int cact = ccount + 1;
    bool reshaped = false;
    for (auto fc: fc_layers)
      {
	if (flat1dconv && !reshaped)
	  {
	    reshaped = true;
	    if (rl < max_rl)
	      {
		lparam = net_param.mutable_layer(rl);
		lparam->clear_bottom();
		lparam->clear_top();
		lparam->clear_include();
		lparam->clear_loss_weight();
	      }
	    else lparam = net_param.add_layer();
	    lparam->set_name("reshape0");
	    lparam->set_type("Reshape");
	    lparam->add_bottom(prec_ip);
	    lparam->add_top("reshape0");
	    lparam->mutable_reshape_param()->mutable_shape()->add_dim(0);
	    lparam->mutable_reshape_param()->mutable_shape()->add_dim(-1);
	    ++rl;
	
	    if (drl < max_drl)
	      {
		dlparam = deploy_net_param.mutable_layer(drl);
		dlparam->clear_bottom();
		dlparam->clear_top();
		dlparam->clear_include();
		dlparam->clear_loss_weight();
	      }
	    else dlparam = deploy_net_param.add_layer();
	    dlparam->set_name("resphape0");
	    dlparam->set_type("Reshape");
	    dlparam->add_bottom(prec_ip);
	    dlparam->add_top("reshape0");
	    dlparam->mutable_reshape_param()->mutable_shape()->add_dim(0);
	    dlparam->mutable_reshape_param()->mutable_shape()->add_dim(-1);
	    ++drl;
	  }

	if (rl < max_rl)
	  {
	    lparam = net_param.mutable_layer(rl);
	    lparam->clear_bottom();
	    lparam->clear_top();
	    lparam->clear_include();
	    lparam->clear_loss_weight();
	  }
	else lparam = net_param.add_layer();
	lparam->set_name(last_ip);
	lparam->set_type("InnerProduct");
	if (flat1dconv)
	  lparam->add_bottom("reshape0");
	else lparam->add_bottom(prec_ip);
	lparam->add_top(last_ip);
	caffe::InnerProductParameter *ipp = lparam->mutable_inner_product_param();
	ipp->set_num_output(fc);
	ipp->mutable_weight_filler()->set_type("xavier");
	ipp->mutable_bias_filler()->set_type("constant");
	++rl;
	
	if (drl < max_drl)
	  {
	    dlparam = deploy_net_param.mutable_layer(drl);
	    dlparam->clear_bottom();
	    dlparam->clear_top();
	    dlparam->clear_include();
	    dlparam->clear_loss_weight();
	  }
	else dlparam = deploy_net_param.add_layer();
	dlparam->set_name(last_ip);
	dlparam->set_type("InnerProduct");
	if (flat1dconv)
	  dlparam->add_bottom("reshape0");
	else dlparam->add_bottom(prec_ip);
	dlparam->add_top(last_ip);
	caffe::InnerProductParameter *dipp = dlparam->mutable_inner_product_param();
	dipp->set_num_output(fc);
	dipp->mutable_weight_filler()->set_type("xavier");
	dipp->mutable_bias_filler()->set_type("constant");
	++drl;

	if (rl < max_rl)
	  {
	    lparam = net_param.mutable_layer(rl);
	    lparam->clear_include();
	    lparam->clear_top();
	    lparam->clear_bottom();
	    lparam->clear_loss_weight();
	    lparam->clear_inner_product_param();
	  }
	else lparam = net_param.add_layer();
	lparam->set_name("act"+std::to_string(cact));
	lparam->set_type(activation);
	if (activation == "ELU" && elu_alpha != 1.0)
	  lparam->mutable_elu_param()->set_alpha(elu_alpha);
	lparam->add_bottom(last_ip);
	lparam->add_top(last_ip);
	++rl;

	if (drl < max_drl)
	  {
	    dlparam = deploy_net_param.mutable_layer(drl);
	    dlparam->clear_include();
	    dlparam->clear_top();
	    dlparam->clear_bottom();
	    dlparam->clear_loss_weight();
	    dlparam->clear_inner_product_param();
	  }
	else dlparam = deploy_net_param.add_layer();
	dlparam->set_name("act"+std::to_string(cact));
	dlparam->set_type(activation);
	if (activation == "ELU" && elu_alpha != 1.0)
	  lparam->mutable_elu_param()->set_alpha(elu_alpha);
	dlparam->add_bottom(last_ip);
	dlparam->add_top(last_ip);
	++drl;

	if (dropout > 0.0 && dropout < 1.0)
	  {
	    if (rl < max_rl)
	      {
		lparam = net_param.mutable_layer(rl);
		lparam->clear_bottom();
		lparam->clear_top();
	      }
	    else lparam = net_param.add_layer(); // dropout layer
	    std::string drop = "drop" + std::to_string(lfc);
	    lparam->set_name(drop);
	    lparam->set_type("Dropout");
	    lparam->add_bottom(last_ip);
	    lparam->add_top(last_ip);
	    lparam->mutable_dropout_param()->set_dropout_ratio(dropout);
	    ++rl;
	  }
	
	++lfc;
	++cact;
	prec_ip = last_ip;
	last_ip = "ip" + std::to_string(lfc);
      }

    // add remaining inner product softmax layers    
    if (rl < max_rl)
      {
	lparam = net_param.mutable_layer(rl);
	lparam->clear_bottom();
	lparam->clear_top();
	lparam->clear_include();
	lparam->clear_loss_weight();
      }
    else lparam = net_param.add_layer(); // last inner product before softmax
    lparam->set_name(last_ip);
    lparam->set_type("InnerProduct");
    lparam->add_bottom(prec_ip);
    lparam->add_top(last_ip);
    caffe::InnerProductParameter *ipp = lparam->mutable_inner_product_param();
    if (!regression || targets == 0)
      ipp->set_num_output(nclasses);
    else ipp->set_num_output(targets);
    ipp->mutable_weight_filler()->set_type("xavier");
    ipp->mutable_bias_filler()->set_type("constant");
    ++rl;

    if (drl < max_drl)
      {
	dlparam = deploy_net_param.mutable_layer(drl);
	dlparam->clear_bottom();
	dlparam->clear_top();
	dlparam->clear_include();
	dlparam->clear_loss_weight();
      }
    else dlparam = deploy_net_param.add_layer();
    dlparam->set_name(last_ip);
    dlparam->set_type("InnerProduct");
    dlparam->add_bottom(prec_ip);
    dlparam->add_top(last_ip);
    caffe::InnerProductParameter *dipp = dlparam->mutable_inner_product_param();
    if (!regression || targets == 0)
      dipp->set_num_output(nclasses);
    else dipp->set_num_output(targets);
    dipp->mutable_weight_filler()->set_type("xavier");
    dipp->mutable_bias_filler()->set_type("constant");
    ++drl;

    if (!regression)
      {
	if (rl < max_rl)
	  {
	    lparam = net_param.mutable_layer(rl);
	    lparam->clear_bottom();
	    lparam->clear_top();
	    lparam->clear_include();
	  }
	else lparam = net_param.add_layer(); // test loss
	lparam->set_name("losst");
        lparam->set_type("Softmax");
	lparam->add_bottom(last_ip);
	lparam->add_top("losst");
	caffe::NetStateRule *nsr = lparam->add_include();
	nsr->set_phase(caffe::TEST);
	++rl;
      }

    if (rl < max_rl)
      {
	lparam = net_param.mutable_layer(rl);
	lparam->clear_bottom();
	lparam->clear_top();
	lparam->clear_include();
      }
    else lparam = net_param.add_layer(); // training loss
    lparam->set_name("loss");
    if (regression)
      lparam->set_type("EuclideanLoss");
    else lparam->set_type("SoftmaxWithLoss");
    lparam->add_bottom(last_ip);
    lparam->add_bottom("label");
    lparam->add_top("loss");
    caffe::NetStateRule *nsr = lparam->add_include();
    nsr->set_phase(caffe::TRAIN);
    ++rl;
    
    if (!regression)
      {
	if (drl < max_drl)
	  {
	    dlparam = deploy_net_param.mutable_layer(drl);
	    dlparam->clear_bottom();
	    dlparam->clear_top();
	    dlparam->clear_include();
	  }
	else dlparam = deploy_net_param.add_layer();
	dlparam->set_name("loss");
	dlparam->set_type("Softmax");
	dlparam->add_bottom(last_ip);
	dlparam->add_top("loss");
	++drl;
      }
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  int CaffeLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::create_model(const bool &test)
  {
    // create net and fill it up
    if (!this->_mlmodel._def.empty() && !this->_mlmodel._weights.empty())
      {
	delete _net;
	_net = nullptr;
	if (!test)
	  _net = new Net<float>(this->_mlmodel._def,caffe::TRAIN);
	else _net = new Net<float>(this->_mlmodel._def,caffe::TEST);
	LOG(INFO) << "Using pre-trained weights from " << this->_mlmodel._weights << std::endl;
	_net->CopyTrainedLayersFrom(this->_mlmodel._weights);
	return 0;
      }
    // net definition is missing
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
    if (ad.has("regression") && ad.get("regression").get<bool>())
      {
	_regression = true;
	_nclasses = 1;
      }
    if (ad.has("ntargets"))
      _ntargets = ad.get("ntargets").get<int>();
    if (_nclasses == 0)
      throw MLLibBadParamException("number of classes is unknown (nclasses == 0)");
    if (_regression && _ntargets == 0)
      throw MLLibBadParamException("number of regression targets is unknown (ntargets == 0)");
    // instantiate model template here, if any
    if (ad.has("template"))
      instantiate_template(ad);
    else // model template instantiation is defered until training call
      create_model();
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  void CaffeLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::clear_mllib(const APIData &ad)
  {
    (void)ad;
    std::vector<std::string> extensions = {".solverstate",".caffemodel",".json"};
    if (!this->_inputc._db)
      extensions.push_back(".dat"); // e.g., for txt input connector and db, do not delete the vocab.dat since the db is not deleted
    fileops::remove_directory_files(this->_mlmodel._repo,extensions);
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  int CaffeLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::train(const APIData &ad,
										 APIData &out)
  {
    if (this->_mlmodel._solver.empty())
      {
	throw MLLibBadParamException("missing solver file in " + this->_mlmodel._repo);
      }

    std::lock_guard<std::mutex> lock(_net_mutex); // XXX: not mandatory as train calls are locking resources from above
    TInputConnectorStrategy inputc(this->_inputc);
    this->_inputc._dv.clear();
    this->_inputc._dv_test.clear();
    this->_inputc._ids.clear();
    inputc._train = true;
    APIData cad = ad;
    cad.add("model_repo",this->_mlmodel._repo); // pass the model repo so that in case of images, it is known where to save the db

    try
      {
	inputc.transform(cad);
      }
    catch (...)
      {
	throw;
      }
    
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
    caffe::ReadProtoFromTextFile(this->_mlmodel._solver,&solver_param);
    bool has_mean_file = false;
    int user_batch_size, batch_size, test_batch_size, test_iter;
    update_in_memory_net_and_solver(solver_param,cad,inputc,has_mean_file,user_batch_size,batch_size,test_batch_size,test_iter);

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
	    if (strcasecmp(solver_type.c_str(),"SGD") == 0)
	      solver_param.set_solver_type(caffe::SolverParameter_SolverType_SGD);
	    else if (strcasecmp(solver_type.c_str(),"ADAGRAD") == 0)
	      {
		solver_param.set_solver_type(caffe::SolverParameter_SolverType_ADAGRAD);
		solver_param.set_momentum(0.0); // cannot be used with adagrad
	      }
	    else if (strcasecmp(solver_type.c_str(),"NESTEROV") == 0)
	      solver_param.set_solver_type(caffe::SolverParameter_SolverType_NESTEROV);
	    else if (strcasecmp(solver_type.c_str(),"RMSPROP") == 0)
	      {
		solver_param.set_solver_type(caffe::SolverParameter_SolverType_RMSPROP);
		solver_param.set_momentum(0.0); // not supported by Caffe PR
	      }
	    else if (strcasecmp(solver_type.c_str(),"ADADELTA") == 0)
	      solver_param.set_solver_type(caffe::SolverParameter_SolverType_ADADELTA);
	    else if (strcasecmp(solver_type.c_str(),"ADAM") == 0)
	      solver_param.set_solver_type(caffe::SolverParameter_SolverType_ADAM);
	  }
	if (ad_solver.has("test_interval"))
	  solver_param.set_test_interval(ad_solver.get("test_interval").get<int>());
	if (ad_solver.has("test_initialization"))
	  solver_param.set_test_initialization(ad_solver.get("test_initialization").get<bool>());
	if (ad_solver.has("lr_policy"))
	  solver_param.set_lr_policy(ad_solver.get("lr_policy").get<std::string>());
	if (ad_solver.has("base_lr"))
	  solver_param.set_base_lr(ad_solver.get("base_lr").get<double>());
	if (ad_solver.has("gamma"))
	  solver_param.set_gamma(ad_solver.get("gamma").get<double>());
	if (ad_solver.has("stepsize"))
	  solver_param.set_stepsize(ad_solver.get("stepsize").get<int>());
	if (ad_solver.has("momentum") && solver_param.solver_type() != caffe::SolverParameter_SolverType_ADAGRAD)
	  solver_param.set_momentum(ad_solver.get("momentum").get<double>());
	if (ad_solver.has("weight_decay"))
	  solver_param.set_weight_decay(ad_solver.get("weight_decay").get<double>());
	if (ad_solver.has("power"))
	  solver_param.set_power(ad_solver.get("power").get<double>());
	if (ad_solver.has("rms_decay"))
	  solver_param.set_rms_decay(ad_solver.get("rms_decay").get<double>());
	if (ad_solver.has("iter_size"))
	  solver_param.set_iter_size(ad_solver.get("iter_size").get<int>());
      }
    
    // optimize
    this->_tjob_running = true;
    caffe::Solver<float> *solver = nullptr;
    try
      {
	solver = caffe::SolverRegistry<float>::CreateSolver(solver_param);
      }
    catch(std::exception &e)
      {
	delete solver;
	throw;
      }
    if (!inputc._dv.empty())
      {
	LOG(INFO) << "filling up net prior to training\n";
	try {
	  boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(solver->net()->layers()[0])->AddDatumVector(inputc._dv);
	}
	catch(std::exception &e)
	  {
	    delete solver;
	    throw;
	  }
	/*if (!solver->test_nets().empty())
	  {
	    if (!inputc._dv_test.empty())
	      boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(solver->test_nets().at(0)->layers()[0])->AddDatumVector(inputc._dv_test);
	    else boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(solver->test_nets().at(0)->layers()[0])->AddDatumVector(inputc._dv);
	    }*/
	inputc._dv.clear();
	inputc._ids.clear();
      }
    this->_mlmodel.read_from_repository(this->_mlmodel._repo);
    this->_mlmodel.read_corresp_file();
    if (ad_mllib.has("resume") && ad_mllib.get("resume").get<bool>())
      {
	if (this->_mlmodel._sstate.empty())
	  {
	    delete solver;
	    throw MLLibBadParamException("resuming a model requires a .solverstate file in model repository");
	  }
	else 
	  {
	    try
	      {
		solver->Restore(this->_mlmodel._sstate.c_str());
	      }
	    catch(std::exception &e)
	      {
		LOG(ERROR) << "Failed restoring network state\n";
		delete solver;
		throw;
	      }
	  }
      }
    else if (!this->_mlmodel._weights.empty())
      {
	try
	  {
	    solver->net()->CopyTrainedLayersFrom(this->_mlmodel._weights);
	  }
	catch(std::exception &e)
	  {
	    delete solver;
	    throw;
	  }
      }
    else
      {
	solver->iter_ = 0;
	solver->current_step_ = 0;
      }
	
    const int start_iter = solver->iter_;
    int average_loss = solver->param_.average_loss();
    std::vector<float> losses;
    this->clear_all_meas_per_iter();
    float smoothed_loss = 0.0;
    std::vector<Blob<float>*> bottom_vec;
    while(solver->iter_ < solver->param_.max_iter()
	  && this->_tjob_running.load())
      {
	this->add_meas("iteration",solver->iter_);

	solver->net_->ClearParamDiffs();
	
	// Save a snapshot if needed.
	if (solver->param_.snapshot() && solver->iter_ > start_iter &&
	    solver->iter_ % solver->param_.snapshot() == 0) {
	  solver->Snapshot();
	}
	if (solver->param_.test_interval() && solver->iter_ % solver->param_.test_interval() == 0
	    && (solver->iter_ > 0 || solver->param_.test_initialization())) 
	  {
	    APIData meas_out;
	    solver->test_nets().at(0).get()->ShareTrainedLayersWith(solver->net().get());
	    test(solver->test_nets().at(0).get(),ad,inputc,test_batch_size,has_mean_file,meas_out);
	    APIData meas_obj = meas_out.getobj("measure");
	    std::vector<std::string> meas_str = meas_obj.list_keys();
	    LOG(INFO) << "batch size=" << batch_size;
	    for (auto m: meas_str)
	      {
		if (m != "cmdiag" && m != "cmfull") // do not report confusion matrix in server logs
		  {
		    double mval = meas_obj.get(m).get<double>();
		    LOG(INFO) << m << "=" << mval;
		    this->add_meas(m,mval);
		    if (!std::isnan(mval)) // if testing occurs once before training even starts, loss is unknown and we don't add it to history.
		      this->add_meas_per_iter(m,mval);
		  }
		else if (m == "cmdiag")
		  {
		    std::vector<double> mdiag = meas_obj.get(m).get<std::vector<double>>();
		    std::string mdiag_str;
		    for (size_t i=0;i<mdiag.size();i++)
		      mdiag_str += this->_mlmodel.get_hcorresp(i) + ":" + std::to_string(mdiag.at(i)) + " ";
		    LOG(INFO) << m << "=[" << mdiag_str << "]";
		  }
	      }
	  }
	
	float loss = 0.0;
	try
	  {
	    for (size_t i = 0; i < solver->callbacks().size(); ++i) {
	      solver->callbacks()[i]->on_start();
	    }
	    for (int i = 0; i < solver->param_.iter_size(); ++i)
	      loss += solver->net_->ForwardBackward(bottom_vec);
	    loss /= solver->param_.iter_size();
	  }
	catch(std::exception &e)
	  {
	    LOG(ERROR) << "exception while forward/backward pass through the network\n";
	    delete solver;
	    throw;
	  }
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
	this->add_meas("train_loss",smoothed_loss);

	if (solver->param_.test_interval() && solver->iter_ % solver->param_.test_interval() == 0)
	  {
	    this->add_meas_per_iter("train_loss",loss); // to avoid filling up with possibly millions of entries...
	    LOG(INFO) << "smoothed_loss=" << this->get_meas("train_loss");
	  }
	try
	  {
	    for (size_t i = 0; i < solver->callbacks().size(); ++i) {
	      solver->callbacks()[i]->on_gradients_ready();
	    }
	    solver->iter_++;
	    solver->ApplyUpdate();
	  }
	catch (std::exception &e)
	  {
	    LOG(ERROR) << "exception while updating network\n";
	    delete solver;
	    throw;
	  }
      }
    
    // always save final snapshot.
    if (solver->param_.snapshot_after_train())
      solver->Snapshot();
    
    // destroy the net
    delete _net;
    _net = nullptr;
    delete solver;
    
    // bail on forced stop, i.e. not testing the net further.
    if (!this->_tjob_running.load())
      {
	inputc._dv_test.clear();
	return 0;
      }
    
    solver_param = caffe::SolverParameter();
    this->_mlmodel.read_from_repository(this->_mlmodel._repo);
    int cm = create_model();
    if (cm == 1)
      throw MLLibInternalException("no model in " + this->_mlmodel._repo + " for initializing the net");
    else if (cm == 2)
      throw MLLibBadParamException("no deploy file in " + this->_mlmodel._repo + " for initializing the net");
    
    // test
    test(_net,ad,inputc,test_batch_size,has_mean_file,out);
    inputc._dv_test.clear();
        
    // add whatever the input connector needs to transmit out
    inputc.response_params(out);

    // if batch_size has been recomputed, let the user know
    if (user_batch_size != batch_size)
      {
	APIData advb;
	advb.add("batch_size",batch_size);
	std::vector<APIData> vb = { advb };
	if (!out.has("parameters"))
	  {
	    APIData adparams;
	    adparams.add("mllib",vb);
	    std::vector<APIData> vab = { adparams };
	    out.add("parameters",vab);
	  }
	else
	  {
	    APIData adparams = out.getobj("parameters");
	    adparams.add("mllib",vb);
	    std::vector<APIData> vad = { adparams };
	    out.add("parameters",vad);
	  }
      }
    
    return 0;
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  void CaffeLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::test(caffe::Net<float> *net,
										 const APIData &ad,
										 TInputConnectorStrategy &inputc,
										 const int &test_batch_size,
										 const bool &has_mean_file,
										 APIData &out)
  {
    APIData ad_res;
    ad_res.add("iteration",this->get_meas("iteration"));
    ad_res.add("train_loss",this->get_meas("train_loss"));
    APIData ad_out = ad.getobj("parameters").getobj("output");
    if (ad_out.has("measure"))
      {
	float mean_loss = 0.0;
	int tresults = 0;
	int nout = _nclasses;
	if (_regression && _ntargets > 1)
	  nout = _ntargets;
	ad_res.add("nclasses",_nclasses);
	inputc.reset_dv_test();
	std::vector<caffe::Datum> dv;
	while(!(dv=inputc.get_dv_test(test_batch_size,has_mean_file)).empty())
	  {
	    try
	      {
		boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(net->layers()[0])->set_batch_size(dv.size());
		boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(net->layers()[0])->AddDatumVector(dv);
	      }
	    catch(std::exception &e)
	      {
		// XXX: might want to clean up here...
		throw;
	      }
	    float loss = 0.0;
	    std::vector<Blob<float>*> lresults = net->ForwardPrefilled(&loss);
	    int slot = lresults.size() - 1;
	    if (_regression && _ntargets > 1) // slicing is involved
	      slot--; // labels appear to be last
	    int scount = lresults[slot]->count();
	    int scperel = scount / dv.size();
	    for (int j=0;j<(int)dv.size();j++)
	      {
		APIData bad;
		std::vector<double> predictions;
		if (!_regression || _ntargets == 1)
		  {
		    double target = dv.at(j).label();
		    for (int k=0;k<nout;k++)
		      {
			predictions.push_back(lresults[slot]->cpu_data()[j*scperel+k]);
		      }
		    bad.add("target",target);
		  }
		else
		  {
		    std::vector<double> target;
		    for (int k=inputc.channels();k<dv.at(j).float_data_size();k++)
		      target.push_back(dv.at(j).float_data(k));
		    for (int k=0;k<nout;k++)
		      {
			predictions.push_back(lresults[slot]->cpu_data()[j*scperel+k]);
		      }
		    bad.add("target",target);
		  }
		bad.add("pred",predictions);
		std::vector<APIData> vad = { bad };
		ad_res.add(std::to_string(tresults+j),vad);
	      }
	    tresults += dv.size();
	    mean_loss += loss;
	  }
	std::vector<std::string> clnames;
	for (int i=0;i<nout;i++)
	  clnames.push_back(this->_mlmodel.get_hcorresp(i));
	ad_res.add("clnames",clnames);
	ad_res.add("batch_size",tresults);
	if (_regression)
	  ad_res.add("regression",_regression);
	//ad_res.add("loss",mean_loss / static_cast<double>(tresults)); // XXX: Caffe ForwardPrefilled call above return loss = 0.0
      }
    SupervisedOutput::measure(ad_res,ad_out,out);
  }
  
  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  int CaffeLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::predict(const APIData &ad,
										   APIData &out)
  {
    std::lock_guard<std::mutex> lock(_net_mutex); // no concurrent calls since the net is not re-instantiated
        
    // check for net
    if (!_net || _net->phase() == caffe::TRAIN)
      {
	int cm = create_model(true);
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
    try
      {
	boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(_net->layers()[0])->set_batch_size(batch_size);
	boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(_net->layers()[0])->AddDatumVector(inputc._dv);
      }
    catch(std::exception &e)
      {
	throw;
      }
    
    float loss = 0.0;
    std::vector<Blob<float>*> results = _net->ForwardPrefilled(&loss);
    int slot = results.size() - 1;
    if (_regression)
      {
	if (_ntargets > 1)
	  slot = 1;
	else slot = 0; // XXX: more in-depth testing required
      }
    int scount = results[slot]->count();
    int scperel = scount / batch_size;
    int nclasses = scperel;
    TOutputConnectorStrategy tout;
    for (int j=0;j<batch_size;j++)
      {
	tout.add_result(inputc._ids.at(j),loss);
	for (int i=0;i<nclasses;i++)
	  tout.add_cat(inputc._ids.at(j),results[slot]->cpu_data()[j*scperel+i],this->_mlmodel.get_hcorresp(i));
      }
    TOutputConnectorStrategy btout(this->_outputc);
    if (_regression)
      tout._best = nclasses;
    tout.best_cats(ad.getobj("parameters").getobj("output"),btout);
    btout.to_ad(out,_regression);
    out.add("status",0);
    
    return 0;
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  void CaffeLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::update_in_memory_net_and_solver(caffe::SolverParameter &sp,
													    const APIData &ad,
													    const TInputConnectorStrategy &inputc,
													    bool &has_mean_file,
													    int &user_batch_size,
													    int &batch_size,
													    int &test_batch_size,
													    int &test_iter)
  {
    // fix net model path.
    sp.set_net(this->_mlmodel._repo + "/" + sp.net());
    
    // fix net snapshot path.
    sp.set_snapshot_prefix(this->_mlmodel._repo + "/model");
    
    // acquire custom batch size if any
    user_batch_size = batch_size = inputc.batch_size();
    test_batch_size = inputc.test_batch_size();
    test_iter = -1;
    fix_batch_size(ad,inputc,user_batch_size,batch_size,test_batch_size,test_iter);
    if (test_iter != -1) // has changed
      sp.set_test_iter(0,test_iter);
    
    // fix source paths in the model.
    caffe::NetParameter *np = sp.mutable_net_param();
    caffe::ReadProtoFromTextFile(sp.net().c_str(),np); //TODO: error on read + use internal caffe ReadOrDie procedure
    for (int i=0;i<np->layer_size();i++)
      {
	caffe::LayerParameter *lp = np->mutable_layer(i);
	if (lp->has_data_param())
	  {
	    caffe::DataParameter *dp = lp->mutable_data_param();
	    if (dp->has_source())
	      {
		if (i == 0 && ad.has("db")) // training
		  {
		    dp->set_source(ad.getobj("db").get("train_db").get<std::string>());
		  }
		else if (i == 1 && ad.has("db"))
		  {
		    dp->set_source(ad.getobj("db").get("test_db").get<std::string>());
		  }
		else 
		  {
		    dp->set_source(this->_mlmodel._repo + "/" + dp->source()); // this updates in-memory net
		  }
	      }
	    if (dp->has_batch_size() && batch_size != inputc.batch_size() && batch_size > 0)
	      {
		dp->set_batch_size(user_batch_size); // data params seem to handle batch_size that are no multiple of the training set
		batch_size = user_batch_size;
	      }
	  }
	else if (lp->has_memory_data_param())
	  {
	    caffe::MemoryDataParameter *mdp = lp->mutable_memory_data_param();
	    if (mdp->has_batch_size() && batch_size != inputc.batch_size() && batch_size > 0)
	      {
		if (i == 0) // training
		  mdp->set_batch_size(batch_size);
		else mdp->set_batch_size(test_batch_size);
	      }
	  }
	if (lp->has_transform_param())
	  {
	    caffe::TransformationParameter *tp = lp->mutable_transform_param();
	    has_mean_file = tp->has_mean_file();
	    if (tp->has_mean_file())
	      {
		if (ad.has("db"))
		  tp->set_mean_file(ad.getobj("db").get("meanfile").get<std::string>());
		else tp->set_mean_file(this->_mlmodel._repo + "/" + tp->mean_file());
	      }
	  }
      }
    sp.clear_net();
  }

  // XXX: we are no more pre-setting the batch_size to the data set values (e.g. number of samples)
  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  void CaffeLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::update_protofile_net(const std::string &net_file,
												 const std::string &deploy_file,
												 const TInputConnectorStrategy &inputc)
  {
    caffe::NetParameter net_param;
    caffe::ReadProtoFromTextFile(net_file,&net_param); //TODO: catch parsing error (returns bool true on success)
    if (net_param.mutable_layer(0)->has_memory_data_param()
	|| net_param.mutable_layer(1)->has_memory_data_param())
      {
	if (_ntargets == 0)
	  {
	    if (net_param.mutable_layer(0)->has_memory_data_param())
	      net_param.mutable_layer(0)->mutable_memory_data_param()->set_channels(inputc.channels());
	    if (net_param.mutable_layer(1)->has_memory_data_param())
	      net_param.mutable_layer(1)->mutable_memory_data_param()->set_channels(inputc.channels()); // test layer
	  }
	else
	  {
	    if (net_param.mutable_layer(0)->has_memory_data_param())
	      net_param.mutable_layer(0)->mutable_memory_data_param()->set_channels(inputc.channels()+_ntargets);
	    if (net_param.mutable_layer(1)->has_memory_data_param())
	      net_param.mutable_layer(1)->mutable_memory_data_param()->set_channels(inputc.channels()+_ntargets);
	    net_param.mutable_layer(2)->mutable_slice_param()->set_slice_point(0,inputc.channels());
	  }
      }
    
    caffe::NetParameter deploy_net_param;
    caffe::ReadProtoFromTextFile(deploy_file,&deploy_net_param);
    if (deploy_net_param.mutable_layer(0)->has_memory_data_param())
      {
	// no batch size set on deploy model since it is adjusted for every prediction batch
	if (_ntargets == 0)
	  deploy_net_param.mutable_layer(0)->mutable_memory_data_param()->set_channels(inputc.channels());
	else
	  {
	    deploy_net_param.mutable_layer(0)->mutable_memory_data_param()->set_channels(inputc.channels()+_ntargets);
	    deploy_net_param.mutable_layer(1)->mutable_slice_param()->set_slice_point(0,inputc.channels());
	  }
      }
    caffe::WriteProtoToTextFile(net_param,net_file);
    caffe::WriteProtoToTextFile(deploy_net_param,deploy_file);
  }
  
  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  void CaffeLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::update_protofile_classes(caffe::NetParameter &net_param)
  {
    // fix class numbers
    // this procedure looks for the first bottom layer with a 'num_output' field and
    // set it to the number of classes defined by the supervised service.
    for (int l=net_param.layer_size()-1;l>0;l--)
      {
	caffe::LayerParameter *lparam = net_param.mutable_layer(l);
	if (lparam->type() == "Convolution")
	  {
	    if (lparam->has_convolution_param())
	      {
		lparam->mutable_convolution_param()->set_num_output(_nclasses);
		break;
	      }
	  }
	else if (lparam->type() == "InnerProduct")
	  {
	    if (lparam->has_inner_product_param())
	      {
		if (!_regression || _ntargets == 0)
		  lparam->mutable_inner_product_param()->set_num_output(_nclasses);
		else lparam->mutable_inner_product_param()->set_num_output(_ntargets);
		break;
	      }
	  }
      }
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  void CaffeLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::update_protofile_finetune(caffe::NetParameter &net_param)
  {
    // fix class numbers
    // this procedure looks for the first bottom layer with a 'num_output' field and
    // rename the layer so that its weights can be reinitialized and the net finetuned
    int k = net_param.layer_size();
    std::string ft_lname, ft_oldname;
    for (int l=net_param.layer_size()-1;l>0;l--)
      {
	caffe::LayerParameter *lparam = net_param.mutable_layer(l);
	if (lparam->type() == "Convolution")
	  {
	    ft_oldname = lparam->name();
	    ft_lname = lparam->name() + "_ftune";
	    lparam->set_name(ft_lname);
	    lparam->set_top(0,ft_lname);
	    k = l;
	    break;
	  }
	else if (lparam->type() == "InnerProduct")
	  {
	    ft_oldname = lparam->name();
	    ft_lname = lparam->name() + "_ftune";
	    lparam->set_name(ft_lname);
	    lparam->set_top(0,ft_lname);
	    k = l;
	    break;
	  }
      }
    // update relations from other layers
    for (int l=net_param.layer_size()-1;l>k;l--)
      {
	caffe::LayerParameter *lparam = net_param.mutable_layer(l);
	if (lparam->top(0) == ft_oldname)
	  lparam->set_top(0,ft_lname);
	if (lparam->bottom(0) == ft_oldname)
	  lparam->set_bottom(0,ft_lname);
      }
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  void CaffeLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::fix_batch_size(const APIData &ad,
											   const TInputConnectorStrategy &inputc,
											   int &user_batch_size,
											   int &batch_size,
											   int &test_batch_size,
											   int &test_iter)
  {
    // acquire custom batch size if any
    APIData ad_net = ad.getobj("parameters").getobj("mllib").getobj("net");
    //batch_size = inputc.batch_size();
    //test_batch_size = inputc.test_batch_size();
    //test_iter = 1;
    if (ad_net.has("batch_size"))
      {
	// adjust batch size so that it is a multiple of the number of training samples (Caffe requirement)
	user_batch_size = batch_size = test_batch_size = ad_net.get("batch_size").get<int>();
	if (ad_net.has("test_batch_size"))
	  test_batch_size = ad_net.get("test_batch_size").get<int>();
	if (batch_size == 0)
	  throw MLLibBadParamException("batch size set to zero");
	LOG(INFO) << "user batch_size=" << batch_size << " / inputc batch_size=" << inputc.batch_size() << std::endl;

	// code below is required when Caffe (weirdly) requires the batch size 
	// to be a multiple of the training dataset size.
	if (batch_size < inputc.batch_size())
	  {
	    int min_batch_size = 0;
	    for (int i=batch_size;i>=1;i--)
	      if (inputc.batch_size() % i == 0)
		{
		  min_batch_size = i;
		  break;
		}
	    int max_batch_size = 0;
	    for (int i=batch_size;i<inputc.batch_size();i++)
	      {
		if (inputc.batch_size() % i == 0)
		  {
		    max_batch_size = i;
		    break;
		  }
	      }
	    if (fabs(batch_size-min_batch_size) < fabs(max_batch_size-batch_size))
	      batch_size = min_batch_size;
	    else batch_size = max_batch_size;
	    for (int i=test_batch_size;i>1;i--)
	      if (inputc.test_batch_size() % i == 0)
		{
		  test_batch_size = i;
		  break;
		}
	    test_iter = inputc.test_batch_size() / test_batch_size;
	  }
	else batch_size = inputc.batch_size();
	test_iter = inputc.test_batch_size() / test_batch_size;
	
	//debug
	LOG(INFO) << "batch_size=" << batch_size << " / test_batch_size=" << test_batch_size << " / test_iter=" << test_iter << std::endl;
	//debug
      }
  }

  template class CaffeLib<ImgCaffeInputFileConn,SupervisedOutput,CaffeModel>;
  template class CaffeLib<CSVCaffeInputFileConn,SupervisedOutput,CaffeModel>;
  template class CaffeLib<TxtCaffeInputFileConn,SupervisedOutput,CaffeModel>;
}
