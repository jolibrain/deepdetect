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

#include "net_caffe.h"
#include "utils/utils.hpp"
#include "imginputfileconn.h"

namespace dd
{

  /*- CaffeCommon -*/
  caffe::LayerParameter* CaffeCommon::add_layer(caffe::NetParameter *net_param,
						const std::string &bottom,
						const std::string &top,
						const std::string &name,
						const std::string &type,
						const std::string &label)
  {
    caffe::LayerParameter *lparam = net_param->add_layer();
    if (!bottom.empty())
      lparam->add_bottom(bottom);
    lparam->add_top(top);
    if (!label.empty())
      lparam->add_top(label);
    if (!name.empty())
      lparam->set_name(name);
    if (!type.empty())
      lparam->set_type(type);
    return lparam;
  }

  std::string CaffeCommon::set_activation(const APIData &ad_mllib)
  {
    std::string activation = "ReLU"; // default
    if (ad_mllib.has("activation"))
      {
	activation = ad_mllib.get("activation").get<std::string>();
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
  
  /*- NetInputCaffe -*/
  template <class TInputCaffe>
  void NetInputCaffe<TInputCaffe>::configure_inputs(const APIData &ad_mllib,
						    const TInputCaffe &inputc)
  {
    int ntargets = -1;
    if (ad_mllib.has("ntargets"))
      ntargets = ad_mllib.get("ntargets").get<int>();
    bool db = false;
    if (ad_mllib.has("db")) //TODO: if Caffe + image, db is true
      db = ad_mllib.get("db").get<bool>();
    int width = inputc.width();
    int height = inputc.height();
    int channels = inputc.channels();
    bool flat1dconv = inputc._flat1dconv; // whether the model uses 1d-conv (e.g. character-level convnet for text)
    
    // train net
    std::string top = "data";
    std::string label = "label";
    if (ntargets > 1)
      {
	top = "fulldata";
	label = "fake_label";
      }

    // train layer
    caffe::LayerParameter *lparam = CaffeCommon::add_layer(this->_net_params,"",top,"inputl","",label);
    caffe::NetStateRule *nsr = lparam->add_include();
    nsr->set_phase(caffe::TRAIN);
    
    // deploy net
    caffe::LayerParameter *dlparam = CaffeCommon::add_layer(this->_dnet_params,"",top,"inputl","",label);
    
    // sources
    if (db)
      {
	if (inputc._sparse)
	  lparam->set_type("SparseData");
	else lparam->set_type("Data");
	caffe::DataParameter *dparam = lparam->mutable_data_param();
	dparam->set_source("train.lmdb");
	dparam->set_batch_size(32); // dummy value, updated before training
	dparam->set_backend(caffe::DataParameter_DB_LMDB);
	if (!flat1dconv)
	  {
	    if (ad_mllib.has("mirror"))
	      lparam->mutable_transform_param()->set_mirror(ad_mllib.get("mirror").get<bool>());
	    if (ad_mllib.has("rotate"))
	      lparam->mutable_transform_param()->set_rotate(ad_mllib.get("rotate").get<bool>());
	    //TODO
	    /*std::string mf = "mean.binaryproto";
	      lparam->mutable_transform_param()->set_mean_file(mf.c_str());*/
	  }
	
	// test
	/*lparam = this->_net_params->add_layer(); // test layer
	lparam->set_type("Data");
	dparam = lparam->mutable_data_param();
	dparam->set_source("test.lmdb");
	dparam->set_batch_size(32); // dummy value, updated before training
	dparam->set_backend(caffe::DataParameter_DB_LMDB);
	caffe::NetStateRule *nsr = lparam->add_include();
	nsr->set_phase(caffe::TEST);*/
      }
    else
      {
	if (inputc._sparse)
	  lparam->set_type("MemorySparseData");
	else lparam->set_type("MemoryData");
	caffe::MemoryDataParameter *mdparam = lparam->mutable_memory_data_param();
	mdparam->set_batch_size(32); // dummy value, updated before training
	mdparam->set_channels(channels);
	mdparam->set_height(height);
	mdparam->set_width(width);
      }

    lparam = CaffeCommon::add_layer(this->_net_params,"",top,"inputl",
				    inputc._sparse ? "MemorySparseData" : "MemoryData",label); // test layer
    caffe::MemoryDataParameter *mdparam = lparam->mutable_memory_data_param();
    mdparam->set_batch_size(32); // dummy value, updated before training
    mdparam->set_channels(channels);
    mdparam->set_height(height);
    mdparam->set_width(width);
    nsr = lparam->add_include();
    nsr->set_phase(caffe::TEST);
    
    // deploy
    if (inputc._sparse)
      dlparam->set_type("MemorySparseData");
    else dlparam->set_type("MemoryData");
    mdparam = dlparam->mutable_memory_data_param();
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
	sparam->add_slice_point(1);

	dlparam = CaffeCommon::add_layer(this->_dnet_params,top,"data");
	dlparam->add_top("label");
	dlparam->set_type("Slice");
	dlparam->set_name("slice_labels");
	sparam = dlparam->mutable_slice_param();
	sparam->set_slice_dim(1);
	sparam->add_slice_point(1);
      }
  }

  /*- NetLayersCaffe -*/
  void NetLayersCaffe::add_fc(caffe::NetParameter *net_param,
			      const std::string &bottom,
			      const std::string &top,
			      const int &num_output)
  {
    caffe::LayerParameter *lparam = CaffeCommon::add_layer(net_param,bottom,top,"fc_"+bottom,"InnerProduct");
    caffe::InnerProductParameter *iparam = lparam->mutable_inner_product_param();
    iparam->set_num_output(num_output);
    iparam->mutable_weight_filler()->set_type("xavier"); //TODO: option
    caffe::FillerParameter *fparam = iparam->mutable_bias_filler();
    fparam->set_type("constant");
    fparam->set_value(0.0); //TODO: option
  }

  void NetLayersCaffe::add_sparse_fc(caffe::NetParameter *net_param,
				     const std::string &bottom,
				     const std::string &top,
				     const int &num_output)
  {
    caffe::LayerParameter *lparam = CaffeCommon::add_layer(net_param,bottom,top,"fc_"+bottom,"SparseInnerProduct");
    caffe::InnerProductParameter *iparam = lparam->mutable_inner_product_param();
    iparam->set_num_output(num_output);
    iparam->mutable_weight_filler()->set_type("xavier"); //TODO: option
    caffe::FillerParameter *fparam = iparam->mutable_bias_filler();
    fparam->set_type("constant");
    fparam->set_value(0.0); //TODO: option
  }
  
  void NetLayersCaffe::add_conv(caffe::NetParameter *net_param,
				const std::string &bottom,
				const std::string &top,
				const int &num_output,
				const int &kernel_size,
				const int &pad,
				const int &stride,
				const int &kernel_w,
				const int &kernel_h,
				const int &pad_w,
				const int &pad_h,
				const std::string &name,
				const std::string &init)
  {
    caffe::LayerParameter *lparam = CaffeCommon::add_layer(net_param,bottom,top,name.empty()?"conv_"+bottom:name,"Convolution");
    caffe::ConvolutionParameter *cparam = lparam->mutable_convolution_param();
    cparam->set_num_output(num_output);
    if (kernel_w == 0 && kernel_h == 0)
      cparam->add_kernel_size(kernel_size);
    else
      {
	cparam->set_kernel_w(kernel_w);
	cparam->set_kernel_h(kernel_h);
      }	
    if (pad_w == 0 && pad_h == 0)
      cparam->add_pad(pad);
    else
      {
	cparam->set_pad_w(pad_w);
	cparam->set_pad_h(pad_h);
      }
    cparam->add_stride(stride);
    cparam->mutable_weight_filler()->set_type(init);
    caffe::FillerParameter *fparam = cparam->mutable_bias_filler();
    fparam->set_type("constant");
    //fparam->set_value(0.2); //TODO: option
  }

  void NetLayersCaffe::add_act(caffe::NetParameter *net_param,
			       const std::string &bottom,
			       const std::string &activation,
			       const double &elu_alpha,
			       const double &negative_slope)
  {
    caffe::LayerParameter *lparam = CaffeCommon::add_layer(net_param,bottom,bottom,
							   "act_" + activation + "_" + bottom,activation);
    if (activation == "ELU" && elu_alpha != 1.0)
      lparam->mutable_elu_param()->set_alpha(elu_alpha);
    if (activation == "ReLU" && negative_slope != 0.0)
      lparam->mutable_relu_param()->set_negative_slope(negative_slope);
  }

  void NetLayersCaffe::add_pooling(caffe::NetParameter *net_param,
				   const std::string &bottom,
				   const std::string &top,
				   const int &kernel_size,
				   const int &stride,
				   const std::string &type,
				   const int &kernel_w,
				   const int &kernel_h,
				   const int &stride_w,
				   const int &stride_h)
  {
    caffe::LayerParameter *lparam = CaffeCommon::add_layer(net_param,bottom,top,"pool_"+bottom,"Pooling");
    caffe::PoolingParameter *pparam = lparam->mutable_pooling_param();
    if (kernel_w == 0 && kernel_h == 0)
      pparam->set_kernel_size(kernel_size);
    else
      {
	pparam->set_kernel_w(kernel_w);
	pparam->set_kernel_h(kernel_h);
      }
    if (stride_w == 0 && stride_h == 0)
      pparam->set_stride(stride);
    else
      {
	pparam->set_stride_w(stride_w);
	pparam->set_stride_h(stride_h);
      }
    if (type == "MAX")
      pparam->set_pool(caffe::PoolingParameter::MAX);
    else if (type == "AVE")
      pparam->set_pool(caffe::PoolingParameter::AVE);
    else if (type == "STOCHASTIC")
      pparam->set_pool(caffe::PoolingParameter::STOCHASTIC);
  }

  void NetLayersCaffe::add_dropout(caffe::NetParameter *net_param,
				   const std::string &bottom,
				   const double &ratio)
  {
    caffe::LayerParameter *lparam = CaffeCommon::add_layer(net_param,bottom,bottom,
							   "drop_"+bottom,"Dropout");
    lparam->mutable_dropout_param()->set_dropout_ratio(ratio);
  }
  
  void NetLayersCaffe::add_bn(caffe::NetParameter *net_param,
			      const std::string &bottom,
			      const std::string &top)
  {
    caffe::LayerParameter *lparam = CaffeCommon::add_layer(net_param,bottom,top.empty()?bottom:top,
							   "bn_"+bottom,"BatchNorm");
    lparam = CaffeCommon::add_layer(net_param,top.empty()?bottom:top,top.empty()?bottom:top,
				    "scale_"+bottom,"Scale"); //TODO: add scale
    lparam->mutable_scale_param()->set_bias_term(true);
  }

  void NetLayersCaffe::add_eltwise(caffe::NetParameter *net_param,
				   const std::string &bottom1,
				   const std::string &bottom2,
				   const std::string &top)
  {
    caffe::LayerParameter *lparam = CaffeCommon::add_layer(net_param,bottom1,top,"elt_"+top,"Eltwise");
    lparam->add_bottom(bottom2);
  }

  void NetLayersCaffe::add_reshape(caffe::NetParameter *net_param,
				   const std::string &bottom,
				   const std::string &top,
				   const caffe::ReshapeParameter &r_param)
  {
    //TODO
    caffe::LayerParameter *lparam = CaffeCommon::add_layer(net_param,bottom,top,"reshape_"+top,"Reshape");
    lparam->mutable_reshape_param()->CopyFrom(r_param);
  }
  
  void NetLayersCaffe::add_softmax(caffe::NetParameter *net_param,
				   const std::string &bottom,
				   const std::string &label,
				   const std::string &top,
				   const int &num_output,
				   const bool &deploy)
  {
    std::string ln_tmp = "ip_" + top;
    add_fc(net_param,bottom,ln_tmp,num_output);

    if (!deploy)
      {
	caffe::LayerParameter *lparam = CaffeCommon::add_layer(net_param,ln_tmp,top,
							       "prob","SoftmaxWithLoss"); // train
	lparam->add_bottom(label);
	caffe::NetStateRule *nsr = lparam->add_include();
	nsr->set_phase(caffe::TRAIN);
	
	lparam = CaffeCommon::add_layer(net_param,ln_tmp,top,
					"probt","Softmax"); // test
	nsr = lparam->add_include();
	nsr->set_phase(caffe::TEST);
      }
    else
      {
	CaffeCommon::add_layer(net_param,ln_tmp,top,
			       "prob","Softmax"); // deploy
      }
  }

  void NetLayersCaffe::add_euclidean_loss(caffe::NetParameter *net_param,
					  const std::string &bottom,
					  const std::string &label,
					  const std::string &top,
					  const int &num_output,
					  const bool &deploy)
  {
    std::string ln_tmp = "ip_" + top;
    add_fc(net_param,bottom,ln_tmp,num_output);


    if (!deploy)
      {
	caffe::LayerParameter *lparam = CaffeCommon::add_layer(net_param,ln_tmp,top,
							       "loss","EuclideanLoss"); // train
	lparam->add_bottom(label);
	caffe::NetStateRule *nsr = lparam->add_include();
	nsr->set_phase(caffe::TRAIN);
      }
  }
  
  /*- NetLossCaffe -*/

  /*- NetCaffe -*/
  /*template<class TNetInputCaffe, class TNetLayersCaffe, class TNetLossCaffe>
  NetCaffe<TNetInputCaffe,TNetLayersCaffe,TNetLossCaffe>::NetCaffe(caffe::NetParameter *net_params,
								   caffe::NetParameter *dnet_params)
    :_net_params(net_params),_dnet_params(dnet_params),
     _nic(net_params,dnet_params),_nlac(net_params,dnet_params)//,_nloc(net_params,dnet_params)
  {
    
  }*/

  template class NetInputCaffe<ImgCaffeInputFileConn>;
  template class NetInputCaffe<CSVCaffeInputFileConn>;
  template class NetInputCaffe<TxtCaffeInputFileConn>;
  template class NetInputCaffe<SVMCaffeInputFileConn>;
}
