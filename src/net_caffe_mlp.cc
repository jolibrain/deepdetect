/**
 * DeepDetect
 * Copyright (c) 2014-2016 Emmanuel Benazera
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

#include "net_caffe_mlp.h"
#include "imginputfileconn.h"

namespace dd
{

  /*- NetInputCaffeMLP -*/
  template<class TInputCaffe>
  void NetInputCaffeMLP<TInputCaffe>::configure_inputs(const APIData &ad_mllib,
						       const TInputCaffe &inputc)
  {
    //NetInputCaffe<TInputCaffe>::configure_inputs(ad);

    //APIData ad_input = ad.getobj("parameters").getobj("input");
    //APIData ad_mllib = ad.getobj("parameters").getobj("mllib");

    int nclasses = -1;
    if (ad_mllib.has("nclasses"))
      nclasses = ad_mllib.get("nclasses").get<int>();
    int ntargets = -1;
    if (ad_mllib.has("ntargets"))
      ntargets = ad_mllib.get("ntargets").get<int>();
    bool db = false;
    if (ad_mllib.has("db"))
      db = ad_mllib.get("db").get<bool>();
    int width = inputc.width();
    int height = inputc.height();
    int channels = inputc.channels();
    
    // train net
    std::string top = "data";
    std::string label = "label";
    if (ntargets > 1)
      {
	top = "fulldata";
	label = "fake_label";
      }

    // train layer
    caffe::LayerParameter *lparam = CaffeCommon::add_layer(this->_net_params,top,label);
    lparam->set_name("data");
    lparam->add_top(top);
    lparam->add_top(label);
    caffe::NetStateRule *nsr = lparam->add_include();
    nsr->set_phase(caffe::TRAIN);
    
    // deploy net
    caffe::LayerParameter *dlparam = CaffeCommon::add_layer(this->_dnet_params,top,label);
    dlparam->set_name("data");
    dlparam->add_top(top);
    dlparam->add_top(label);

    // sources
    if (db)
      {
	lparam->set_type("Data");
	caffe::DataParameter *dparam = lparam->mutable_data_param();
	dparam->set_source("train.lmdb");
	dparam->set_batch_size(32); // dummy value, updated before training
	dparam->set_backend(caffe::DataParameter_DB_LMDB);

	// test
	lparam = this->_net_params->add_layer(); // test layer
	lparam->set_type("Data");
	dparam = lparam->mutable_data_param();
	dparam->set_source("test.lmdb");
	dparam->set_batch_size(32); // dummy value, updated before training
	dparam->set_backend(caffe::DataParameter_DB_LMDB);
	caffe::NetStateRule *nsr = lparam->add_include();
	nsr->set_phase(caffe::TEST);
      }
    else
      {
	lparam->set_type("MemoryData");
	caffe::MemoryDataParameter *mdparam = lparam->mutable_memory_data_param();
	mdparam->set_batch_size(32); // dummy value, updated before training
	mdparam->set_channels(channels);
	mdparam->set_height(height);
	mdparam->set_width(width);

	lparam = this->_net_params->add_layer(); // test layer
	lparam->set_type("MemoryData");
	mdparam = lparam->mutable_memory_data_param();
	mdparam->set_batch_size(32); // dummy value, updated before training
	mdparam->set_channels(channels);
	mdparam->set_height(height);
	mdparam->set_width(width);
	caffe::NetStateRule *nsr = lparam->add_include();
	nsr->set_phase(caffe::TEST);
      }

    // deploy
    dlparam->set_type("MemoryData");
    caffe::MemoryDataParameter *mdparam = dlparam->mutable_memory_data_param();
    mdparam->set_batch_size(1);
    mdparam->set_channels(channels);
    mdparam->set_height(height);
    mdparam->set_width(width);
    
    if (ntargets > 1) // regression
      {
	lparam = CaffeCommon::add_layer(this->_net_params,top,"data");
	lparam->add_top("label");
	lparam->set_type("Slice");
	lparam->set_name("slice_labels");
	caffe::SliceParameter *sparam = lparam->mutable_slice_param();
	sparam->set_slice_dim(1);
	sparam->add_slice_point(1); //TODO: temporay value, NOT nclasses

	dlparam = CaffeCommon::add_layer(this->_dnet_params,top,"data");
	dlparam->add_top("label");
	dlparam->set_type("Slice");
	dlparam->set_name("slice_labels");
	sparam = dlparam->mutable_slice_param();
	sparam->set_slice_dim(1);
	sparam->add_slice_point(1); //TODO: temporay value, NOT nclasses
      }
  }

  /*- NetLayersCaffeMLP -*/
  void NetLayersCaffeMLP::add_basic_block(caffe::NetParameter *net_param,
					  const std::string &bottom,
					  const std::string &top,
					  const int &num_output,
					  const std::string &activation,
					  const double &dropout_ratio,
					  const bool &bn)
  {
    add_fc(net_param,bottom,top,num_output);
    if (bn)
      {
	add_bn(net_param,top);
	add_act(net_param,top,activation);
      }
    else if (dropout_ratio > 0.0)
      {
	add_act(net_param,top,activation);
	add_dropout(net_param,top,dropout_ratio);
      }
    else add_act(net_param,top,activation);
  }

  void NetLayersCaffeMLP::configure_net(const APIData &ad_mllib)
  {
    std::vector<int> layers = {50}; // default
    std::string activation = CaffeCommon::set_activation(ad_mllib);
    double dropout = 0.0; // default
    if (ad_mllib.has("layers"))
      layers = ad_mllib.get("layers").get<std::vector<int>>();
    if (ad_mllib.has("dropout"))
      dropout = ad_mllib.get("dropout").get<double>();
    bool bn = false;
    if (ad_mllib.has("bn"))
      bn = ad_mllib.get("bn").get<bool>();
    int nclasses = -1;
    if (ad_mllib.has("nclasses"))
      nclasses = ad_mllib.get("nclasses").get<int>();
    int ntargets = -1;
    if (ad_mllib.has("ntargets"))
      ntargets = ad_mllib.get("ntargets").get<int>();
    std::string bottom = "data";
    for (size_t l=0;l<layers.size();l++)
      {
	std::string top = "ip" + std::to_string(l);
	add_basic_block(this->_net_params,bottom,top,layers.at(l),activation,dropout,bn);
	add_basic_block(this->_dnet_params,bottom,top,layers.at(l),activation,0.0,bn);
	bottom = top;
      }
    //TODO: to loss ?
    add_softmax(this->_net_params,bottom,"label","losst",nclasses > 0 ? nclasses : ntargets);
    add_softmax(this->_dnet_params,bottom,"","loss",nclasses > 0 ? nclasses : ntargets,true);
  }

  template class NetInputCaffeMLP<ImgCaffeInputFileConn>;
  template class NetInputCaffeMLP<CSVCaffeInputFileConn>;
  template class NetInputCaffeMLP<TxtCaffeInputFileConn>;
  template class NetInputCaffeMLP<SVMCaffeInputFileConn>;
  template class NetCaffe<NetInputCaffeMLP<ImgCaffeInputFileConn>,NetLayersCaffeMLP,NetLossCaffe>;
  template class NetCaffe<NetInputCaffeMLP<CSVCaffeInputFileConn>,NetLayersCaffeMLP,NetLossCaffe>;
  template class NetCaffe<NetInputCaffeMLP<TxtCaffeInputFileConn>,NetLayersCaffeMLP,NetLossCaffe>;
  template class NetCaffe<NetInputCaffeMLP<SVMCaffeInputFileConn>,NetLayersCaffeMLP,NetLossCaffe>;
}
