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
#include "mllibstrategy.h"

namespace dd
{

  /*- NetLayersCaffeResnet -*/
  void NetLayersCaffeResnet::parse_res_layers(const std::vector<std::string> &layers,
					      std::vector<ConvBlock> &cr_layers,
					      std::vector<int> &fc_layers,
					      int &depth, int &n)
  {
    const std::string res_str = "Res";
    const std::string cr_str = "CR";
    if (layers.empty())
      throw MLLibBadParamException("no layers specified for resnet template");
    if (layers.size() == 1) // ResXX generator, for images
      {
	std::string l = layers.at(0); //TODO: error checking
	size_t pos = 0;
	if ((pos=l.find(res_str))!=std::string::npos)
	  {
	    std::string sdepth = l.substr(pos+res_str.size());
	    depth = std::atoi(sdepth.c_str());
	  }
	n = std::max(1,static_cast<int>(std::ceil((depth-2)/9)));
      }
    else // custom convnet resnet
      {
	parse_conv_layers(layers,cr_layers,fc_layers);
	depth = cr_layers.size() / 2 - fc_layers.size();
	n = 2;
      }
  }
  
  void NetLayersCaffeResnet::add_init_block(caffe::NetParameter *net_param,
					    const std::string &bottom,
					    const int &num_output,
					    const int &kernel_size,
					    const int &kernel_w,
					    const int &kernel_h,
					    std::string &top,
					    const bool &act,
					    const bool &pooling)
  {
    add_conv(net_param,bottom,top,num_output,kernel_size,0,2,kernel_w,kernel_h);
    if (act)
      {
	add_bn(net_param,"conv1");
	add_act(net_param,"conv1","ReLU");
      }
    if (pooling)
      {
	top = "pool1";
	add_pooling(net_param,"conv1","pool1",3,2,"MAX"); // large images only, e.g. Imagenet
      }
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
    std::string tmp_top = bottom + "_tmp";
    add_bn(net_param,bottom,tmp_top);
    add_act(net_param,tmp_top,activation);
    int kernel_size = 1;
    int pad = 0;
    std::string block_num_str = std::to_string(block_num);
    std::string conv_name = "conv1_branch" + block_num_str;
    add_conv(net_param,tmp_top,conv_name,num_output,kernel_size,pad,stride); //TODO: stride as parameter to function

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
	add_conv(net_param,bottom,bottom_scale,num_output,1,0,stride,0,0,0,0,"shortcut_"+bottom);
      }
    
    // residual
    top = "res" + block_num_str;
    add_eltwise(net_param,bottom_scale,conv3_name,top);
  }

  void NetLayersCaffeResnet::add_basic_block_flat(caffe::NetParameter *net_param,
						  const int &block_num,
						  const std::string &bottom,
						  const int &num_output,
						  const std::string &activation,
						  const int &stride,
						  const int &kernel_w,
						  const int &kernel_h,
						  const bool &identity,
						  std::string &top)
  {
    int kernel_size = 0; // unset
    int pad = 0;
    std::string block_num_str = std::to_string(block_num);
    std::string conv_name = "conv1_branch" + block_num_str;
    add_conv(net_param,bottom,conv_name,num_output,kernel_size,pad,stride,kernel_w,kernel_h,0,2); //TODO: stride as parameter to function
    add_bn(net_param,conv_name);
    add_act(net_param,conv_name,activation);

    std::string conv2_name = "conv2_branch" + block_num_str;
    add_conv(net_param,conv_name,conv2_name,num_output,kernel_size,pad,stride,kernel_w,kernel_h);
    add_bn(net_param,conv2_name);
    add_act(net_param,conv2_name,activation);

    
    // resize shortcut if input size != output size
    std::string bottom_scale = bottom;
    if (!identity)
      {
	bottom_scale = bottom + "_branch";
	add_conv(net_param,bottom,bottom_scale,num_output,0,0,1,1,1,0,0,"shortcut_"+bottom);
      }
    
    // residual
    top = "res" + block_num_str;
    add_eltwise(net_param,bottom_scale,conv2_name,top);
  }
  
  void NetLayersCaffeResnet::add_basic_block_mlp(caffe::NetParameter *net_param,
						 const int &block_num,
						 const std::string &bottom,
						 const int &num_output,
						 const std::string &activation,
						 const bool &identity,
									 std::string &top)
  {
    std::string tmp_top = bottom + "_tmp";
    add_bn(net_param,bottom,tmp_top); //TODO: not in place as to change the layer top name ?
    add_act(net_param,tmp_top,activation);
    std::string block_num_str = std::to_string(block_num);
    std::string fc_name = "fc_branch" + block_num_str;
    add_fc(net_param,tmp_top,fc_name,num_output);
    
    add_bn(net_param,fc_name);
    add_act(net_param,fc_name,activation);
    std::string fc2_name = "fc2_branch" + block_num_str;
    add_fc(net_param,fc_name,fc2_name,num_output);

    // resize shortcut if input size != output size
    std::string bottom_scale = bottom;
    if (!identity)
      {
	bottom_scale = bottom + "_branch";
	add_fc(net_param,bottom,bottom_scale,num_output);
      }
    
    // residual
    top = "res" + block_num_str;
    add_eltwise(net_param,bottom_scale,fc2_name,top);
  }

  void NetLayersCaffeResnet::configure_net_resarch(const APIData &ad_mllib)
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
    if (ad_mllib.has("layers"))
      layers = ad_mllib.get("layers").get<std::vector<std::string>>();
    int depth = -1;
    int n = -1;
    std::vector<ConvBlock> cr_layers;
    std::vector<int> fc_layers;
    parse_res_layers(layers,cr_layers,fc_layers,depth,n);
    _logger->info("generating ResNet with depth={} / n={}",depth,n);
    std::string bottom = "data";
    
    //std::vector<int> stages = {16, 64, 128, 256};
    std::vector<int> stages = {64, 128, 256, 512};
    std::string top = "conv1";
    add_init_block(this->_net_params,bottom,64,7,0,0,top);
    top = "conv1";
    add_init_block(this->_dnet_params,bottom,64,7,0,0,top);
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
    int pool_size = std::ceil(std::max(1,static_cast<int>(this->_net_params->layer(1).memory_data_param().width()) / 37));
    add_pooling(this->_net_params,bottom,"pool_final",pool_size,1,"AVE"); // could be 8 for 300x300 images
    add_pooling(this->_dnet_params,bottom,"pool_final",pool_size,1,"AVE");
    bottom = "pool_final";
    
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

  void NetLayersCaffeResnet::configure_net_flat(const APIData &ad_mllib)
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
    int depth = -1;
    int n = -1;
    std::vector<ConvBlock> cr_layers;
    std::vector<int> fc_layers;
    parse_res_layers(layers,cr_layers,fc_layers,depth,n);
    _logger->info("generating CharText ResNet with depth={} / n={}",depth,n);
    if (ad_mllib.has("dropout"))
      dropout = ad_mllib.get("dropout").get<double>();
    bool bn = false;
    if (ad_mllib.has("bn"))
      bn = ad_mllib.get("bn").get<bool>();
    std::string bottom = "data";
    int width = this->_net_params->mutable_layer(1)->mutable_memory_data_param()->width();

    std::string top = "conv1";
    add_init_block(this->_net_params,bottom,cr_layers.at(0)._num_output,0,width,7,top,false,false);
    top = "conv1";
    add_init_block(this->_dnet_params,bottom,cr_layers.at(0)._num_output,0,width,7,top,false,false);
    bottom = top;
    int block_num = 1;
    for (size_t l=0;l<cr_layers.size();l++)
      {
	for (int i=0;i<cr_layers.at(l)._nconv;i++)
	  {
	    int stride = 1;
	    int kernel_w = 1;
	    int kernel_h = 3;
	    bool identity = (i == 0 ? false : true);
	    add_basic_block_flat(this->_net_params,block_num,bottom,cr_layers.at(l)._num_output,activation,
				 stride,kernel_w,kernel_h,identity,top);
	    add_basic_block_flat(this->_dnet_params,block_num,bottom,cr_layers.at(l)._num_output,activation,
				 stride,kernel_w,kernel_h,identity,top);
	    bottom = top;
	    ++block_num;
	  }
	top = "pool" + std::to_string(block_num);
	add_pooling(this->_net_params,bottom,top,0,0,"MAX",1,3,1,2);
	add_pooling(this->_dnet_params,bottom,top,0,0,"MAX",1,3,1,2);
	bottom = top;
      }

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
    else
      {
	add_softmax(this->_net_params,bottom,"label","losst",nclasses > 0 ? nclasses : ntargets);
	add_softmax(this->_dnet_params,bottom,"","loss",nclasses > 0 ? nclasses : ntargets,true);
      }
  }

  void NetLayersCaffeResnet::configure_net_mlp(const APIData &ad_mllib)
  {
    int nclasses = -1;
    if (ad_mllib.has("nclasses"))
      nclasses = ad_mllib.get("nclasses").get<int>();
    int ntargets = -1;
    if (ad_mllib.has("ntargets"))
      ntargets = ad_mllib.get("ntargets").get<int>();
    bool sparse = false;
    if (ad_mllib.has("sparse"))
      sparse = ad_mllib.get("sparse").get<bool>();
    bool regression = false;
    if (ad_mllib.has("regression"))
      regression = ad_mllib.get("regression").get<bool>();
    std::string activation = CaffeCommon::set_activation(ad_mllib);
     std::vector<int> layers;
    if (ad_mllib.has("layers"))
      layers = ad_mllib.get("layers").get<std::vector<int>>();
    int depth = layers.size();
    int n = 2;
        
    _logger->info("generating MLP ResNet with depth={} / n={}",depth,n);
    std::string bottom = "data";
   
    std::string top = "fc1";
    if (sparse)
      {
	add_sparse_fc(this->_net_params,bottom,top,layers.at(0));
	add_sparse_fc(this->_dnet_params,bottom,top,layers.at(0));
      }
    else
      {
	add_fc(this->_net_params,bottom,top,layers.at(0));
	add_fc(this->_dnet_params,bottom,top,layers.at(0));
      }
    bottom = top;
    int block_num = 1;
    for (size_t l=1;l<layers.size();l++)
      {
	bool identity = (layers.at(l) == layers.at(l-1));
	add_basic_block_mlp(this->_net_params,block_num,bottom,layers.at(l),activation,identity,top);
	add_basic_block_mlp(this->_dnet_params,block_num,bottom,layers.at(l),activation,identity,top);
	bottom = top;
	++block_num;
      }    
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
  
  void NetLayersCaffeResnet::configure_net(const APIData &ad_mllib)
  {
    bool flat1dconv = false;
    if (ad_mllib.has("flat1dconv"))
      flat1dconv = ad_mllib.get("flat1dconv").get<bool>();
    bool is_mlp = true;
    try
      {
	std::vector<std::string> layers = ad_mllib.get("layers").get<std::vector<std::string>>();
	for (auto s: layers)
	  if (s.find("CR")!=std::string::npos || s.find("Res")!=std::string::npos)
	    {
	      is_mlp = false;
	      break;
	    }
      }
    catch(std::exception &e)
      {
	/*std::vector<int> layers = ad_mllib.get("layers").get<std::vector<int>>();
	  is_mlp = true;*/
      }
    if (!flat1dconv && !is_mlp)
      configure_net_resarch(ad_mllib);
    else if (!is_mlp)
      configure_net_flat(ad_mllib);
    else configure_net_mlp(ad_mllib);
  }  
}
