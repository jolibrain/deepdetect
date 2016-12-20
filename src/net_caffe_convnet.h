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

#include "net_caffe_mlp.h"

namespace dd
{

  class NetLayersCaffeConvnet: public NetLayersCaffeMLP
  {
  public:
  NetLayersCaffeConvnet(caffe::NetParameter *net_params,
			caffe::NetParameter *dnet_params)
    :NetLayersCaffeMLP(net_params,dnet_params) {}
    ~NetLayersCaffeConvnet() {}

  private:
    void parse_conv_layers(const std::vector<std::string> &layers,
			   std::vector<std::pair<int,int>> &cr_layers,
			   std::vector<int> &fc_layers);
    
    void add_basic_block(caffe::NetParameter *net_param,
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
			 const std::string &pl_type);

  public:
    void configure_net(const APIData &ad_mllib);
  };
}

#endif
