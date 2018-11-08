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

#ifndef NET_CAFFE_RESNET_H
#define NET_CAFFE_RESNET_H

#include "generators/net_caffe_convnet.h"

namespace dd
{

  class NetLayersCaffeResnet: public NetLayersCaffeConvnet
  {
  public:
    NetLayersCaffeResnet(caffe::NetParameter *net_params,
			 caffe::NetParameter *dnet_params,
			 std::shared_ptr<spdlog::logger> &logger)
      :NetLayersCaffeConvnet(net_params,dnet_params,logger) 
      {
	net_params->set_name("resnet");
	dnet_params->set_name("resnet");
      }
    ~NetLayersCaffeResnet() {}

  private:
    void parse_res_layers(const std::vector<std::string> &layers,
			  std::vector<ConvBlock> &cr_layers,
			  std::vector<int> &fc_layers,
			  int &depth, int &n);
    
    void add_init_block(caffe::NetParameter *net_param,
			const std::string &bottom,
			const int &num_output,
			const int &kernel_size,
			const int &kernel_w,
			const int &kernel_h,
			std::string &top,
			const bool &bn=true,
			const bool &pooling=true);
    
    void add_basic_block(caffe::NetParameter *net_param,
			 const int &block_num,
			 const std::string &bottom,
			 const int &num_output,
			 const std::string &activation,
			 const int &stride,
			 const bool &identity,
			 std::string &top);

    void add_basic_block_flat(caffe::NetParameter *net_param,
			      const int &block_num,
			      const std::string &bottom,
			      const int &num_output,
			      const std::string &activation,
			      const int &stride,
			      const int &kernel_w,
			      const int &kernel_h,
			      const bool &identity,
			      std::string &top);

    void add_basic_block_mlp(caffe::NetParameter *net_param,
			     const int &block_num,
			     const std::string &bottom,
			     const int &num_output,
			     const std::string &activation,
			     const bool &identity,
			     std::string &top);

    void configure_net_resarch(const APIData &ad_mllib);
    void configure_net_flat(const APIData &ad_mllib);
    void configure_net_mlp(const APIData &ad_mllib);

    
  public:
    void configure_net(const APIData &ad_mllib);
  };
  
}

#endif
