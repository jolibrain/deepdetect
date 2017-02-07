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

#ifndef NET_CAFFE_MLP_H
#define NET_CAFFE_MLP_H

#include "net_caffe.h"

namespace dd
{

  /*template <class TInputCaffe>
    class NetInputCaffeMLP: public NetInputCaffe<TInputCaffe>
    {
    public:
      NetInputCaffeMLP(caffe::NetParameter *net_params,
		       caffe::NetParameter *dnet_params)
	:NetInputCaffe<TInputCaffe>(net_params,dnet_params) {}
      ~NetInputCaffeMLP() {}

      };*/

  class NetLayersCaffeMLP: public NetLayersCaffe
  {
  public:
  NetLayersCaffeMLP(caffe::NetParameter *net_params,
		    caffe::NetParameter *dnet_params)
    :NetLayersCaffe(net_params,dnet_params) {}
    ~NetLayersCaffeMLP() {}

    void add_basic_block(caffe::NetParameter *net_param,
			 const std::string &bottom,
			 const std::string &top,
			 const int &num_output,
			 const std::string &activation,
			 const double &dropout_ratio,
			 const bool &bn,
			 const bool &sparse);
    
    void configure_net(const APIData &ad_mllib);
  };
  
}

#endif
