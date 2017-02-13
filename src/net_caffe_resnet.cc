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

/*
  ResNet API layer specification
  ResY, where Y is the number of layers, such that 9*n+2 = Y, i.e. n = (Y-2)/9
        -> pre-fixed stages as {16, 64, 128, 256}
	-> do n res blocks per stage 1,2,3 (stage 4 is final)
	-> bn + relu + avg pooling
	-> add softmax
 */

#include "net_caffe_resnet.h"

namespace dd
{

  /*- NetLayersCaffeResnet -*/
  void NetLayersCaffeResnet::parse_res_layers(const std::vector<std::string> &layers,
					      int &depth, int &n)
  {
    const std::string res_str = "Res";
    if (layers.size() == 1) // ResXX generator, for images
      {
	std::string l = layers.at(0); //TODO: error checking
	size_t pos = 0;
	if ((pos=l.find(res_str))!=std::string::npos)
	  {
	    std::string sdepth = l.substr(pos+res_str.size());
	    depth = std::atoi(sdepth.c_str());
	  }
      }
    else // custom MLP resnet
      {
	//TODO
      }
    n = static_cast<int>(std::floor((depth-2)/9));
  }
  
  void NetLayersCaffeResnet::add_init_block(caffe::NetParameter *net_param,
					    const std::string &bottom,
					    std::string &top)
  {
    add_conv(net_param,bottom,top,64,7,3,2);
    /*add_bn(net_param,"conv1");
      add_act(net_param,"conv1","ReLU");*/
    /*top = "pool1";
      add_pooling(net_param,"conv1","pool1",3,2,"MAX");*/ // large images only, e.g. Imagenet
  }
  
  void NetLayersCaffeResnet::add_basic_block(caffe::NetParameter *net_param,
					     const int &block_num,
					     const std::string &bottom,
					     const int &num_output,
					     const std::string &activation,
					     const int &stride,
					     const bool &identity,
					     std::string &top)
  {
    //TODO: basic vs bottleneck case
    // basic = 2 conv 3x3
    // bottleneck = conv 1x1, conv 3x3, conv 1x1 
    add_bn(net_param,bottom);
    add_act(net_param,bottom,activation);
    int kernel_size = 1;
    int pad = 0;
    std::string block_num_str = std::to_string(block_num);
    std::string conv_name = "conv1_branch" + block_num_str;
    //TODO: not conv
    add_conv(net_param,bottom,conv_name,num_output,kernel_size,pad,stride); //TODO: stride as parameter to function

    pad = 1;
    kernel_size = 3;
    add_bn(net_param,conv_name);
    add_act(net_param,conv_name,activation);
    std::string conv2_name = "conv2_branch" + block_num_str;
    add_conv(net_param,conv_name,conv2_name,num_output,kernel_size,pad,1);

    pad = 0;
    kernel_size = 1;
    add_bn(net_param,conv2_name);
    add_act(net_param,conv2_name,activation);
    std::string conv3_name = "conv3_branch" + block_num_str;
    add_conv(net_param,conv2_name,conv3_name,num_output,kernel_size,pad,1);
    
    // resize shortcut if input size != output size
    std::string bottom_scale = bottom;
    if (!identity)
      {
	bottom_scale = bottom + "_branch";
	add_conv(net_param,bottom,bottom_scale,num_output,1,0,stride,0,0,"shortcut_"+bottom);
      }
    
    // residual
    top = "res" + block_num_str;
    add_eltwise(net_param,bottom_scale,conv3_name,top);
  }
  
  void NetLayersCaffeResnet::configure_net(const APIData &ad_mllib)
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
    bool flat1dconv = false;
    if (ad_mllib.has("flat1dconv"))
      flat1dconv = ad_mllib.get("flat1dconv").get<bool>();
    
    std::vector<std::string> layers;
    std::string activation = CaffeCommon::set_activation(ad_mllib);
    double dropout = 0.5;
    if (ad_mllib.has("layers"))
      layers = ad_mllib.get("layers").get<std::vector<std::string>>();
    //TODO: else raise exception
    int depth = -1;
    int n = -1;
    parse_res_layers(layers,depth,n);
    LOG(INFO) << "generating ResNet with depth=" << depth << " / n=" << n << std::endl;
    if (ad_mllib.has("dropout"))
      dropout = ad_mllib.get("dropout").get<double>();
    bool bn = false;
    if (ad_mllib.has("bn"))
      bn = ad_mllib.get("bn").get<bool>();
    bool db = false;
    if (ad_mllib.has("db") && ad_mllib.get("db").get<bool>())
      db = true;
    int conv_kernel_size = 3;
    int conv1d_early_kernel_size = 7;
    std::string bottom = "data";
    int width = -1;
    if (flat1dconv)
      width = this->_net_params->mutable_layer(1)->mutable_memory_data_param()->width();

    std::vector<int> stages = {16, 64, 128, 256};
    std::string top = "conv1";
    add_init_block(this->_net_params,bottom,top);
    add_init_block(this->_dnet_params,bottom,top);
    bottom = top;
    int block_num = 1;
    for (size_t s=0;s<stages.size();s++)
      {
	for (int i=0;i<n;i++)
	  {
	    add_basic_block(this->_net_params,block_num,bottom,stages[s],activation,s==0?1:2,i==0?false:true,top);
	    add_basic_block(this->_dnet_params,block_num,bottom,stages[s],activation,s==0?1:2,i==0?false:true,top);
	    bottom = top;
	    ++block_num;
	  }
      }
    add_bn(this->_net_params,bottom);
    add_bn(this->_dnet_params,bottom);
    add_act(this->_net_params,bottom,activation);
    add_act(this->_dnet_params,bottom,activation);
    add_pooling(this->_net_params,bottom,"pool_final",8,1,"AVE");
    add_pooling(this->_dnet_params,bottom,"pool_final",8,1,"AVE");
    
    if (regression)
      {
	add_euclidean_loss(this->_net_params,bottom,"label","losst",ntargets);
	add_euclidean_loss(this->_net_params,bottom,"","loss",ntargets,true);
      }
    else
      {
	add_softmax(this->_net_params,bottom,"label","losst",nclasses > 0 ? nclasses : ntargets);
	add_softmax(this->_dnet_params,bottom,"","loss",nclasses > 0 ? nclasses : ntargets,true);
      }
  }
  
}
