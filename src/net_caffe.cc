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

namespace dd
{

  /*- CaffeCommon -*/
  caffe::LayerParameter* CaffeCommon::add_layer(caffe::NetParameter *net_param,
						const std::string &bottom,
						const std::string &top,
						const std::string &name,
						const std::string &type)
  {
    caffe::LayerParameter *lparam = net_param->add_layer();
    lparam->add_bottom(bottom);
    lparam->add_top(top);
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
  void NetInputCaffe<TInputCaffe>::configure_inputs(const APIData &ad,
						    const TInputCaffe &inputc)
  {
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
  
  void NetLayersCaffe::add_conv(caffe::NetParameter *net_param,
				const std::string &bottom,
				const std::string &top,
				const int &num_output,
				const int &kernel_size,
				const int &pad,
				const int &stride)
  {
    caffe::LayerParameter *lparam = CaffeCommon::add_layer(net_param,bottom,top,"conv_"+bottom,"Convolution");
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
				   const std::string &type)
  {
    caffe::LayerParameter *lparam = CaffeCommon::add_layer(net_param,bottom,top,"pool_"+bottom,"Pooling");
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

  void NetLayersCaffe::add_dropout(caffe::NetParameter *net_param,
				   const std::string &bottom,
				   const double &ratio)
  {
    caffe::LayerParameter *lparam = CaffeCommon::add_layer(net_param,bottom,bottom,
							   "drop_"+bottom,"Dropout");
    lparam->mutable_dropout_param()->set_dropout_ratio(ratio);
  }
  
  void NetLayersCaffe::add_bn(caffe::NetParameter *net_param,
			      const std::string &bottom)
  {
    caffe::LayerParameter *lparam = CaffeCommon::add_layer(net_param,bottom,bottom,
							   "bn_"+bottom,"BatchNorm");
    lparam = CaffeCommon::add_layer(net_param,bottom,bottom,
				    "scale_"+bottom,"Scale");
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
  
  /*- NetLossCaffe -*/

  /*- NetCaffe -*/
  /*template<class TNetInputCaffe, class TNetLayersCaffe, class TNetLossCaffe>
  NetCaffe<TNetInputCaffe,TNetLayersCaffe,TNetLossCaffe>::NetCaffe(caffe::NetParameter *net_params,
								   caffe::NetParameter *dnet_params)
    :_net_params(net_params),_dnet_params(dnet_params),
     _nic(net_params,dnet_params),_nlac(net_params,dnet_params)//,_nloc(net_params,dnet_params)
  {
    
  }*/
  
}
