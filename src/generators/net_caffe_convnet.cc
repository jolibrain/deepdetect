/**
 * DeepDetect
 * Copyright (c) 2014-2017 Emmanuel Benazera
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
#include "mllibstrategy.h"

namespace dd
{

  /*- NetLayersCaffeConvnet -*/
  void NetLayersCaffeConvnet::parse_conv_layers(const std::vector<std::string> &layers,
						std::vector<ConvBlock> &cr_layers,
						std::vector<int> &fc_layers)
  {
    const std::string cr_str = "CR";
    const std::string d_str = "DR";
    const std::string u_str = "UR";
    for (auto s: layers)
      {
	size_t pos = 0;
	if ((pos=s.find(cr_str))!=std::string::npos)
	  {
	    std::string ncr = s.substr(0,pos);
	    std::string crs = s.substr(pos+cr_str.size());
	    cr_layers.push_back(ConvBlock(cr_str,std::atoi(ncr.c_str()),std::atoi(crs.c_str())));
	  }
	else if ((pos=s.find(d_str))!=std::string::npos)
	  {
	    std::string crs = s.substr(pos+cr_str.size());
	    cr_layers.push_back(ConvBlock(d_str,1,std::atoi(crs.c_str())));
	  }
	else if ((pos=s.find(u_str))!=std::string::npos)
	  {
	    std::string crs = s.substr(pos+cr_str.size());
	    cr_layers.push_back(ConvBlock(u_str,1,std::atoi(crs.c_str())));
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
		throw MLLibBadParamException("convnet template requires fully connected layers size to be specified as a string");
	      }
	  }
      }
  }

  std::string NetLayersCaffeConvnet::add_basic_block(caffe::NetParameter *net_param,
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
					      const std::string &pl_type,
					      const int &kernel_w,
					      const int &kernel_h,
					      const int &pl_kernel_w,
					      const int &pl_kernel_h,
					      const int &pl_stride_w,
					      const int &pl_stride_h)
  {
    std::string top_conv;
    std::string bottom_conv = bottom;
    for (int c=0;c<nconv;c++)
      {
	top_conv = top + "_conv_" + std::to_string(c);
	add_conv(net_param,bottom_conv,top_conv,num_output,kernel_size,pad,stride,kernel_w,kernel_h,0,0,top_conv);
	if (bn)
	  {
	    add_bn(net_param,top_conv);
	    add_act(net_param,top_conv,activation);
	  }
	else if (dropout_ratio > 0.0) // though in general, no dropout between convolutions
	  {
	    add_act(net_param,top_conv,activation);
	    if (c != nconv-1)
	      add_dropout(net_param,top_conv,dropout_ratio);
	  }
	else add_act(net_param,top_conv,activation);
	bottom_conv = top_conv;
      }
    // pooling
    if (pl_type != "NONE")
      add_pooling(net_param,top_conv,top,pl_kernel_size,pl_stride,pl_type,pl_kernel_w,pl_kernel_h,pl_stride_w,pl_stride_h);
    /*if (dropout_ratio > 0.0 && !bn)
      add_dropout(net_param,top,dropout_ratio);*/
    return top_conv;
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
    bool autoencoder = false;
    if (ad_mllib.has("autoencoder"))
      autoencoder = ad_mllib.get("autoencoder").get<bool>();
    bool flat1dconv = false;
    if (ad_mllib.has("flat1dconv"))
      flat1dconv = ad_mllib.get("flat1dconv").get<bool>();
    
    std::vector<std::string> layers;
    std::string activation = CaffeCommon::set_activation(ad_mllib);
    double dropout = 0.5;
    if (ad_mllib.has("layers"))
      layers = ad_mllib.get("layers").get<std::vector<std::string>>();
    //TODO: else raise exception
    std::vector<ConvBlock> cr_layers;
    std::vector<int> fc_layers;
    parse_conv_layers(layers,cr_layers,fc_layers);
    if (ad_mllib.has("dropout"))
      dropout = ad_mllib.get("dropout").get<double>();
    bool bn = false;
    if (ad_mllib.has("bn"))
      bn = ad_mllib.get("bn").get<bool>();
    int conv_kernel_size = 3;
    int conv1d_early_kernel_size = 7;
    std::string bottom = "data";
    bool has_deconv = false;
    int width = -1;
    if (flat1dconv)
      width = this->_net_params->mutable_layer(1)->mutable_memory_data_param()->width();
    std::string top_conv;
    //TODO: support for embed layer in inputs
    for (size_t l=0;l<cr_layers.size();l++)
      {
	std::string top = "ip" + std::to_string(l);
	if (cr_layers.at(l)._layer == "CR")
	  {
	    if (!flat1dconv)
	      {
		int conv_pad = 0;
		if (has_deconv)
		  conv_pad = 1;
		top_conv = add_basic_block(this->_net_params,bottom,top,cr_layers.at(l)._nconv,cr_layers.at(l)._num_output,
					   conv_kernel_size,conv_pad,1,activation,0.0,bn,2,2,has_deconv?"NONE":"MAX");
		if (has_deconv && l == cr_layers.size()-1)
		  top_conv = add_basic_block(this->_dnet_params,bottom,top,cr_layers.at(l)._nconv,cr_layers.at(l)._num_output,
					     1,conv_pad,1,activation,0.0,bn,1,1,"NONE");
		else
		  top_conv = add_basic_block(this->_dnet_params,bottom,top,cr_layers.at(l)._nconv,cr_layers.at(l)._num_output,
					     conv_kernel_size,conv_pad,1,activation,0.0,bn,2,2,has_deconv?"NONE":"MAX");


	      }
	    else
	      {
		top_conv = add_basic_block(this->_net_params,bottom,top,cr_layers.at(l)._nconv,cr_layers.at(l)._num_output,
					   0,0,1,activation,0.0,bn,0,0,"MAX",bottom=="data"?width:1,l<2?conv1d_early_kernel_size:conv_kernel_size,1,3,1,3);
		top_conv = add_basic_block(this->_dnet_params,bottom,top,cr_layers.at(l)._nconv,cr_layers.at(l)._num_output,
					   0,0,1,activation,0.0,bn,0,0,"MAX",bottom=="data"?width:1,l<2?conv1d_early_kernel_size:conv_kernel_size,1,3,1,3);
	      }
	  }
	else if (cr_layers.at(l)._layer == "DR")
	  {
	    add_deconv(this->_net_params,top_conv,top,cr_layers.at(l)._num_output,2,0,2);
	    add_act(this->_net_params,top,activation);
	    add_deconv(this->_dnet_params,top_conv,top,cr_layers.at(l)._num_output,2,0,2);
	    add_act(this->_dnet_params,top,activation);
	    has_deconv = true;
	  }
	//TODO: upsampling UR
	bottom = top;
      }
    if (flat1dconv)
      {
	std::string top = "reshape0";
	caffe::ReshapeParameter r_param;
	r_param.mutable_shape()->add_dim(0);
	r_param.mutable_shape()->add_dim(-1);
	add_reshape(this->_net_params,bottom,top,r_param);
	add_reshape(this->_dnet_params,bottom,top,r_param);
	bottom = top;
      }
    bottom = top_conv;
    int l = 0;
    for (auto fc: fc_layers)
      {
	std::string top = "fc" + std::to_string(fc) + "_" + std::to_string(l);
	NetLayersCaffeMLP::add_basic_block(this->_net_params,bottom,top,fc,activation,dropout,bn,false);
	NetLayersCaffeMLP::add_basic_block(this->_dnet_params,bottom,top,fc,activation,0.0,bn,false);
	bottom = top;
	++l;
      }
    if (regression)
      {
	add_euclidean_loss(this->_net_params,bottom,"label","losst",ntargets);
	add_euclidean_loss(this->_net_params,bottom,"","loss",ntargets,true);
      }
    else if (autoencoder)
      {
	add_sigmoid_crossentropy_loss(this->_net_params,bottom,"data","losst",ntargets);
	add_act(this->_dnet_params,bottom,"Sigmoid");
      }
    else
      {
	add_softmax(this->_net_params,bottom,"label","losst",nclasses > 0 ? nclasses : ntargets);
	add_softmax(this->_dnet_params,bottom,"","loss",nclasses > 0 ? nclasses : ntargets,true);
      }
  }

  template class NetCaffe<NetInputCaffe<ImgCaffeInputFileConn>,NetLayersCaffeConvnet,NetLossCaffe>;
  template class NetCaffe<NetInputCaffe<CSVCaffeInputFileConn>,NetLayersCaffeConvnet,NetLossCaffe>;
  template class NetCaffe<NetInputCaffe<TxtCaffeInputFileConn>,NetLayersCaffeConvnet,NetLossCaffe>;
  template class NetCaffe<NetInputCaffe<SVMCaffeInputFileConn>,NetLayersCaffeConvnet,NetLossCaffe>;
}
