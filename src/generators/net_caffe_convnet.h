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

#ifndef NET_CAFFE_CONVNET_H
#define NET_CAFFE_CONVNET_H

#include "generators/net_caffe_mlp.h"

namespace dd
{

  class ConvBlock
  {
  public:
    ConvBlock(const std::string &layer,
	      const int &nconv,
	      const int &num_output)
      :_layer(layer),_nconv(nconv),_num_output(num_output) {}
    ~ConvBlock() {}
    std::string _layer; // from C: conv, D: deconv, U: upsample
    int _nconv;
    int _num_output;
  };
  
  class NetLayersCaffeConvnet: public NetLayersCaffeMLP
  {
  public:
  NetLayersCaffeConvnet(caffe::NetParameter *net_params,
			caffe::NetParameter *dnet_params,
			std::shared_ptr<spdlog::logger> &logger)
    :NetLayersCaffeMLP(net_params,dnet_params,logger) 
      {
	net_params->set_name("convnet");
	dnet_params->set_name("convnet");
      }
    ~NetLayersCaffeConvnet() {}

  protected:
    void parse_conv_layers(const std::vector<std::string> &layers,
			   std::vector<ConvBlock> &cr_layers,
			   std::vector<int> &fc_layers);
  private:
    std::string add_basic_block(caffe::NetParameter *net_param,
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
			 const int &kernel_w=0,
			 const int &kernel_h=0,
			 const int &pl_kernel_w=0,
			 const int &pl_kernel_h=0,
			 const int &pl_stride_w=0,
			 const int &pl_stride_h=0); //TODO: class of convolution parameters ? use caffe proto ?

  public:
    void configure_net(const APIData &ad_mllib);
  };
}

#endif
