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

#include "caffe_templates.h"
#include "utils/utils.hpp"

using caffe::Caffe;
using caffe::Net;
using caffe::Blob;
using caffe::Datum;

namespace dd
{
  // inputs
  template<class TInputConnectorStrategy>
  void CaffeTemplates<TInputConnectorStrategy>::configure_inputs(caffe::NetParameter &net_param,
					const APIData &ad,
					const int &nclasses,
					const int &targets,
					const int &channels,
					const int &width,
					const int &height,
					const bool &deploy,
					const bool &db,
					const bool &has_mean_file)
  {
    caffe::LayerParameter *lparam = net_param.add_layer(); // train net input layer
    caffe::LayerParameter *dlparam = nullptr;
    std::string top = "data";
    std::string label = "label";
    if (targets > 1)
      {
	top = "fulldata";
	label = "fake_label";
      }
    if (!deploy)
      {
	dlparam = net_param.add_layer(); // test net input layer
	dlparam->set_name("data");
	dlparam->add_top(top);
	dlparam->add_top(label);
      }
    lparam->set_name("data");
    lparam->add_top(top);
    lparam->add_top(label);
    if (!deploy)
      {
	caffe::NetStateRule *nsr = lparam->add_include();
	nsr->set_phase(caffe::TRAIN);
      }
    if (db)
      {
	lparam->set_type("Data");
	caffe::TransformationParameter *tparam = lparam->mutable_transform_param();
	if (ad.has("rotate"))
	  tparam->set_mirror(ad.get("rotate").get<bool>());
	if (ad.has("mirror"))
	  tparam->set_rotate(ad.get("mirror").get<bool>());
	if (ad.has("crop_size"))
	  tparam->set_crop_size(ad.get("crop_size").get<int>());
	if (has_mean_file)
	  tparam->set_mean_file("mean.binaryproto");
	caffe::DataParameter *dparam = lparam->mutable_data_param();
	dparam->set_source("train.lmdb");
	dparam->set_batch_size(32); // dummy value, updated before training
	dparam->set_backend(caffe::DataParameter_DB_LMDB);
      }
    else
      {
	lparam->set_type("MemoryData");
	caffe::MemoryDataParameter *mdparam = lparam->mutable_memory_data_param();
	mdparam->set_batch_size(32); // dummy value, updated before training
	if (!text)
	mdparam->set_channels(channels);
	mdparam->set_height(height);
	mdparam->set_width(width);
      }
    if (!deploy)
      {
	dlparam->set_type("MemoryData");
	caffe::NetStateRule *dnsr = dlparam->add_include();
	dnsr->set_phase(caffe::TEST);
	caffe::MemoryDataParameter *mdparam = dlparam->mutable_memory_data_param();
	mdparam->set_batch_size(32); // dummy value, updated before training
	mdparam->set_channels(channels);
	mdparam->set_height(height);
	mdparam->set_width(width);
      }
    if (targets > 1)
      {
	lparam = add_layer(net_param,top,"data");
	lparam->add_top("label");
	lparam->set_type("Slice");
	lparam->set_name("slice_labels");
	caffe::SliceParameter *sparam = lparam->mutable_slice_param();
	sparam->set_slice_dim(1);
	sparam->add_slice_point(nclasses); // dummy value, updated before training
      }
  }
  
  // outputs
  void CaffeTemplates::write_model_out(caffe::NetParameter &net_param,
				      std::string &outfile)
  {
    try
      {
	caffe::WriteProtoToTextFile(net_param,outfile);
      }
    catch(...)
      {
	throw;
      }
  }

  // options
  std::string CaffeTemplates::set_activation(const APIData &ad)
  {
    std::string activation = "ReLU"; // default
    if (ad.has("activation"))
      {
	activation = ad.get("activation").get<std::string>();
	if (dd_utils::iequals(activation,"relu"))
	  activation = "ReLU";
	else if (dd_utils::iequals(activation,"prelu"))
	  activation = "PReLU";
	else if (dd_utils::iequals(activation,"elu"))
	  activation = "ELU";
	else if (dd_utils::iequals(activation,"sigmoid"))
	  activation = "Sigmoid";
	else if (dd_utils::iequals(activation,"tanh"))
	  activation = "TanH";
      }
    return activation;
  }

  // basic layers
  caffe::LayerParameter* CaffeTemplates::add_layer(caffe::NetParameter &net_param,
						   const std::string &bottom,
						   const std::string &top)
  {
    caffe::LayerParameter *lparam = net_param.add_layer();
    lparam->add_bottom(bottom);
    lparam->add_top(top);
    return lparam;
  }

  void CaffeTemplates::add_fc(caffe::NetParameter &net_param,
			      const std::string &bottom,
			      const std::string &top,
			      const int &num_output)
  {
    caffe::LayerParameter *lparam = add_layer(net_param,bottom,top);
    lparam->set_type("InnerProduct");
    lparam->set_name("fc_" + bottom);
    caffe::InnerProductParameter *iparam = lparam->mutable_inner_product_param();
    iparam->set_num_output(num_output);
    iparam->mutable_weight_filler()->set_type("xavier"); //TODO: option
    caffe::FillerParameter *fparam = iparam->mutable_bias_filler();
    fparam->set_type("constant");
    fparam->set_value(0.0); //TODO: option
  }
  
  void CaffeTemplates::add_conv(caffe::NetParameter &net_param,
				const std::string &bottom,
				const std::string &top,
				const int &num_output,
				const int &kernel_size,
				const int &pad,
				const int &stride)
  {
    caffe::LayerParameter *lparam = add_layer(net_param,bottom,top);
    lparam->set_name("conv_" + bottom);
    lparam->set_type("Convolution");
    caffe::ConvolutionParameter *cparam = lparam->mutable_convolution_param();
    cparam->set_num_output(num_output);
    cparam->add_kernel_size(kernel_size);
    cparam->add_pad(pad);
    cparam->add_stride(stride);
    cparam->mutable_weight_filler()->set_type("xavier"); //TODO: option
    caffe::FillerParameter *fparam = cparam->mutable_bias_filler();
    fparam->set_type("constant");
    fparam->set_value(0.2); //TODO: option
  }

  void CaffeTemplates::add_act(caffe::NetParameter &net_param,
			       const std::string &bottom,
			       const std::string &activation,
			       const double &elu_alpha)
  {
    caffe::LayerParameter *lparam = add_layer(net_param,bottom,bottom);
    lparam->set_name("act_" + activation + "_" + bottom);
    lparam->set_type(activation);
    if (activation == "ELU" && elu_alpha != 1.0)
      lparam->mutable_elu_param()->set_alpha(elu_alpha); //TODO: pass it on
  }

  void CaffeTemplates::add_pooling(caffe::NetParameter &net_param,
				   const std::string &bottom,
				   const std::string &top,
				   const int &kernel_size,
				   const int &stride,
				   const std::string &type)
  {
    caffe::LayerParameter *lparam = add_layer(net_param,bottom,top);
    lparam->set_name("pool_" + bottom);
    lparam->set_type("Pooling");
    caffe::PoolingParameter *pparam = lparam->mutable_pooling_param();
    pparam->set_kernel_size(kernel_size);
    pparam->set_stride(stride);
    if (type == "MAX")
      pparam->set_pool(caffe::PoolingParameter::MAX);
    else if (type == "AVE")
      pparam->set_pool(caffe::PoolingParameter::AVE);
    else if (type == "STOCHASTIC")
      pparam->set_pool(caffe::PoolingParameter::STOCHASTIC);
  }

  void CaffeTemplates::add_dropout(caffe::NetParameter &net_param,
				   const std::string &bottom,
				   const double &ratio)
  {
    caffe::LayerParameter *lparam = add_layer(net_param,bottom,bottom);
    std::string drop_name = "drop_" + bottom;
    lparam->set_name(drop_name);
    lparam->set_type("Dropout");
    lparam->mutable_dropout_param()->set_dropout_ratio(ratio);
  }
  
  void CaffeTemplates::add_bn(caffe::NetParameter &net_param,
			      const std::string &bottom)
  {
    caffe::LayerParameter *lparam = add_layer(net_param,bottom,bottom);
    std::string bn_name = "bn_" + bottom;
    lparam->set_name(bn_name);
    lparam->set_type("BatchNorm");
    lparam = add_layer(net_param,bottom,bottom);
    lparam->set_name("scale_" + bottom);
    lparam->mutable_scale_param()->set_bias_term(true);
  }

  void CaffeTemplates::add_eltwise(caffe::NetParameter &net_param,
				   const std::string &bottom1,
				   const std::string &bottom2,
				   const std::string &top)
  {
    caffe::LayerParameter *lparam = add_layer(net_param,bottom1,top);
    lparam->add_bottom(bottom2);
    lparam->set_name("elt_"+top);
    lparam->set_type("Eltwise");
  }

  void CaffeTemplates::add_softmax(caffe::NetParameter &net_param,
				   const std::string &bottom,
				   const std::string &label,
				   const std::string &top,
				   const int &num_output,
				   const bool &with_loss,
				   const bool &deploy)
  {
    std::string ln_tmp = "ip_" + top;
    caffe::LayerParameter *lparam = nullptr;
    if (!with_loss)
      {
	add_fc(net_param,bottom,ln_tmp,num_output);
	lparam = add_layer(net_param,ln_tmp,top);
      }
    else lparam = add_layer(net_param,bottom,top);
    if (with_loss)
      {
	lparam->add_bottom(label);
	lparam->set_type("SoftmaxWithLoss");
      }
    else
      {
	lparam->set_type("Softmax");
	if (!deploy)
	  {
	    caffe::NetStateRule *rule = lparam->add_include();
	    rule->set_phase(caffe::TEST);
	  }
      }
    lparam->set_name(top);
  }

  // mlp
  void CaffeTemplates::add_mlp_basic_block(caffe::NetParameter &net_param,
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
  
  void CaffeTemplates::configure_mlp_template(const APIData &ad,
				       const bool &regression,
				       const int &targets,
				       const int &cnclasses,
				       caffe::NetParameter &net_param,
				       caffe::NetParameter &deploy_net_param)
  {
    std::vector<int> layers = {50};
    std::string activation = set_activation(ad);
    double dropout = 0.5;
    if (ad.has("layers"))
      layers = ad.get("layers").get<std::vector<int>>();
    if (ad.has("dropout"))
      dropout = ad.get("dropout").get<double>();
    bool bn = false;
    if (ad.has("bn"))
      bn = ad.get("bn").get<bool>();
    bool db = false;
    if (ad.has("db"))
      db = ad.get("db").get<bool>();
    configure_inputs(net_param,ad,cnclasses,targets,false,db,false);
    configure_inputs(deploy_net_param,ad,cnclasses,targets,true,db,false);
    std::string bottom = "data";
    for (size_t l=0;l<layers.size();l++)
      {
	std::string top = "ip" + std::to_string(l);
	add_mlp_basic_block(net_param,bottom,top,layers.at(l),activation,dropout,bn);
	add_mlp_basic_block(deploy_net_param,bottom,top,layers.at(l),activation,0.0,bn);
	bottom = top;
      }
    add_softmax(net_param,bottom,"","losst",cnclasses,false,false);
    add_softmax(net_param,bottom,"label","loss",cnclasses,true,false);
    add_softmax(deploy_net_param,bottom,"label","loss",cnclasses,false,true);
  }

  // convnet
  void CaffeTemplates::parse_conv_layers(const std::vector<std::string> &layers,
					 std::vector<std::pair<int,int>> &cr_layers,
					 std::vector<int> &fc_layers)
  {
    const std::string cr_str = "CR";
    const std::string p_str = "P";
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
  
  void CaffeTemplates::add_convnet_basic_block(caffe::NetParameter &net_param,
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
  
  void CaffeTemplates::configure_convnet_template(const APIData &ad,
						  const bool &regression,
						  const int &targets,
						  const int &cnclasses,
						  const bool &text,
						  caffe::NetParameter &net_param,
						  caffe::NetParameter &deploy_net_param)
  {
    std::vector<std::string> layers;
    std::string activation = set_activation(ad);
    double dropout = 0.5;
    if (ad.has("layers"))
      layers = ad.get("layers").get<std::vector<std::string>>();
    //TODO: else raise exception
    std::vector<std::pair<int,int>> cr_layers;
    std::vector<int> fc_layers;
    parse_conv_layers(layers,cr_layers,fc_layers);
    if (ad.has("dropout"))
      dropout = ad.get("dropout").get<double>();
    bool bn = false;
    if (ad.has("bn"))
      bn = ad.get("bn").get<bool>();
    bool db = false;
    if (ad.has("db") && ad.get("db").get<bool>())
      db = true;
    uint32_t conv_kernel_size = 3;
    uint32_t conv1d_early_kernel_size = 7;
    configure_inputs(net_param,ad,cnclasses,targets,false,db,!text);
    configure_inputs(deploy_net_param,ad,cnclasses,targets,true,db,!text);
    std::string bottom = "data";
    for (size_t l=0;l<cr_layers.size();l++)
      {
	std::string top = "ip" + std::to_string(l);
	add_convnet_basic_block(net_param,bottom,top,cr_layers.at(l).first,cr_layers.at(l).second,conv_kernel_size,0,1,activation,0.0,bn,2,2,"MAX"); //TODO: pad=0,stride=1, option, dropout is 0 since no dropout in inner loop
	add_convnet_basic_block(deploy_net_param,bottom,top,cr_layers.at(l).first,cr_layers.at(l).second,conv_kernel_size,0,1,activation,0.0,bn,2,2,"MAX");
	bottom = top;
      }
    for (auto fc: fc_layers)
      {
	std::string top = "fc" + std::to_string(fc);
	add_mlp_basic_block(net_param,bottom,top,fc,activation,dropout,bn);
	add_mlp_basic_block(deploy_net_param,bottom,top,fc,activation,0.0,bn);
	bottom = top;
      }
    add_softmax(net_param,bottom,"","losst",cnclasses,false,false);
    add_softmax(net_param,"ip_losst","label","loss",cnclasses,true,false);
    add_softmax(deploy_net_param,bottom,"label","loss",cnclasses,false,true);
  }
  
  // resnet
  void CaffeTemplates::add_resnet_init_block(caffe::NetParameter &net_param,
					     const std::string &bottom,
					     const std::string &top)
  {
    add_conv(net_param,bottom,top,64,7,3,2);
    /*add_bn(net_param,"conv1");
    add_act(net_param,"conv1","ReLU");
    add_pooling(net_param,"conv1","pool1",3,2,"MAX");*/ // large images only, e.g. Imagenet
  }

  void CaffeTemplates::add_resnet_basic_block(caffe::NetParameter &net_param,
					      const int &block_num,
					      const std::string &bottom,
					      const std::string &top,
					      const int &num_output,
					      const std::string &activation,
					      const bool &identity)
  {
    //TODO: bottleneck case
    add_bn(net_param,bottom);
    add_act(net_param,bottom,activation);
    int kernel_size = 3; //TODO: option
    int pad = 1;
    int stride = 1;
    std::string conv_name = "conv1_branch" + std::to_string(block_num);
    add_conv(net_param,bottom,conv_name,num_output,kernel_size,pad,stride);
    
    add_bn(net_param,conv_name);
    add_act(net_param,conv_name,activation);
    std::string conv2_name = "conv2_branch" + std::to_string(block_num);
    add_conv(net_param,conv_name,conv2_name,num_output,kernel_size,pad,stride);

    // resize shortcut if input size != output size
    std::string bottom_scale = bottom;
    if (!identity)
      {
	bottom_scale = bottom + "_branch";
	add_conv(net_param,bottom,bottom_scale,num_output,1,0,stride);
      }
    
    // residual
    add_eltwise(net_param,bottom_scale,conv2_name,"res"+std::to_string(block_num));
  }
  
  void CaffeTemplates::configure_resnet_template(const APIData &ad,
						 const bool &regression,
						 const int &targets,
						 const int &cnclasses,
						 caffe::NetParameter &net_param,
						 caffe::NetParameter &deploy_net_param)
  {
    std::string activation = set_activation(ad);
    std::vector<int> layers;
    //TODO: layer language in vector of string (like convnet)
    if (ad.has("layers"))
      layers = ad.get("layers").get<std::vector<int>>();
    int nlayers = layers.size();
    bool db = false;
    if (ad.has("db"))
      db = ad.get("db").get<bool>();
    
    configure_inputs(net_param,ad,cnclasses,targets,false,db,false);
    // resnet init block
    add_resnet_init_block(net_param,"data","conv1"); //TODO: if long init
    std::string bottom = "conv1";
    for (int l=0;l<nlayers;l++)
      {
	// basic resnet block
	std::string top = "res" + std::to_string(l);
	add_resnet_basic_block(net_param,l,bottom,top,layers.at(l),activation,(l==0 || layers.at(l-1)==layers.at(l))); //TODO: if bottleneck
	bottom = top;
      }
    // average pooling
    add_pooling(net_param,bottom,"pool_final",7,1,"AVE");
    // fc
    add_fc(net_param,"pool_final","fc_final",cnclasses);
    // softmax (or regression)
    add_softmax(net_param,"fc_final","label","prob",cnclasses,true,false);
  }
  
}
