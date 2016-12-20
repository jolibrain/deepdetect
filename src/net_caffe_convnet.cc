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

#include "net_caffe_convnet.h"
#include "imginputfileconn.h"

namespace dd
{

  /*- NetLayersCaffeConvnet -*/
  void NetLayersCaffeConvnet::parse_conv_layers(const std::vector<std::string> &layers,
						std::vector<std::pair<int,int>> &cr_layers,
						std::vector<int> &fc_layers)
  {
    const std::string cr_str = "CR";
    //const std::string p_str = "P";
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
		//TODO
		//throw MLLibBadParamException("convnet template requires fully connected layers size to be specified as a string");
	      }
	  }
      }
  }

  void NetLayersCaffeConvnet::add_basic_block(caffe::NetParameter *net_param,
					      const std::string &bottom,
					      const std::string &top,
					      const int &nconv,
					      const int &num_output,
					      const int &kernel_size,
					      const int &pad,
					      const int &stride,
					      const std::string &activation,
					      const double &dropout_ratio,
					      const bool &bn,
					      const int &pl_kernel_size,
					      const int &pl_stride,
					      const std::string &pl_type)
  {
    std::string top_conv;
    for (int c=0;c<nconv;c++)
      {
	top_conv = top + "_conv_" + std::to_string(c);
	add_conv(net_param,bottom,top_conv,num_output,kernel_size,pad,stride);
	if (bn)
	  {
	    add_bn(net_param,top_conv);
	    add_act(net_param,top_conv,activation);
	  }
	else if (dropout_ratio > 0.0)
	  {
	    add_act(net_param,top_conv,activation);
	    if (c != nconv-1)
	      add_dropout(net_param,top_conv,dropout_ratio);
	  }
	else add_act(net_param,top_conv,activation);
      }
    // pooling
    add_pooling(net_param,top_conv,top,pl_kernel_size,pl_stride,pl_type);
    /*if (dropout_ratio > 0.0 && !bn)
      add_dropout(net_param,top,dropout_ratio);*/
  }

  void NetLayersCaffeConvnet::configure_net(const APIData &ad_mllib)
  {
    int nclasses = -1;
    if (ad_mllib.has("nclasses"))
      nclasses = ad_mllib.get("nclasses").get<int>();
    int ntargets = -1;
    if (ad_mllib.has("ntargets"))
      ntargets = ad_mllib.get("ntargets").get<int>();
    bool regression = false;
    if (ad_mllib.has("regression"))
      regression = ad_mllib.get("regression").get<bool>();
    
    std::vector<std::string> layers;
    std::string activation = CaffeCommon::set_activation(ad_mllib);
    double dropout = 0.5;
    if (ad_mllib.has("layers"))
      layers = ad_mllib.get("layers").get<std::vector<std::string>>();
    //TODO: else raise exception
    std::vector<std::pair<int,int>> cr_layers;
    std::vector<int> fc_layers;
    parse_conv_layers(layers,cr_layers,fc_layers);
    if (ad_mllib.has("dropout"))
      dropout = ad_mllib.get("dropout").get<double>();
    bool bn = false;
    if (ad_mllib.has("bn"))
      bn = ad_mllib.get("bn").get<bool>();
    bool db = false;
    if (ad_mllib.has("db") && ad_mllib.get("db").get<bool>())
      db = true;
    uint32_t conv_kernel_size = 3;
    uint32_t conv1d_early_kernel_size = 7;
    /*configure_inputs(net_param,ad,nclasses,ntargets,false,db,!text);
      configure_inputs(deploy_net_param,ad,nclasses,ntargets,true,db,!text);*/
    std::string bottom = "data";
    for (size_t l=0;l<cr_layers.size();l++)
      {
	std::string top = "ip" + std::to_string(l);
	add_basic_block(this->_net_params,bottom,top,cr_layers.at(l).first,cr_layers.at(l).second,conv_kernel_size,0,1,activation,0.0,bn,2,2,"MAX"); //TODO: pad=0,stride=1, option, dropout is 0 since no dropout in inner loop
	add_basic_block(this->_dnet_params,bottom,top,cr_layers.at(l).first,cr_layers.at(l).second,conv_kernel_size,0,1,activation,0.0,bn,2,2,"MAX");
	bottom = top;
      }
    for (auto fc: fc_layers)
      {
	std::string top = "fc" + std::to_string(fc);
	NetLayersCaffeMLP::add_basic_block(this->_net_params,bottom,top,fc,activation,dropout,bn);
	NetLayersCaffeMLP::add_basic_block(this->_dnet_params,bottom,top,fc,activation,0.0,bn);
	bottom = top;
      }
    add_softmax(this->_net_params,bottom,"label","losst",nclasses > 0 ? nclasses : ntargets);
    add_softmax(this->_dnet_params,bottom,"","loss",nclasses > 0 ? nclasses : ntargets,true);
  }

  template class NetCaffe<NetInputCaffe<ImgCaffeInputFileConn>,NetLayersCaffeConvnet,NetLossCaffe>;
  template class NetCaffe<NetInputCaffe<CSVCaffeInputFileConn>,NetLayersCaffeConvnet,NetLossCaffe>;
  template class NetCaffe<NetInputCaffe<TxtCaffeInputFileConn>,NetLayersCaffeConvnet,NetLossCaffe>;
  template class NetCaffe<NetInputCaffe<SVMCaffeInputFileConn>,NetLayersCaffeConvnet,NetLossCaffe>;
}
