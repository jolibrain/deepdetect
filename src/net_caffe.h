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

#ifndef NET_CAFFE_H
#define NET_CAFFE_H

#include "net_generator.h"
#include "caffe/caffe.hpp"

namespace dd
{

  using caffe::Caffe;
  using caffe::Net;
  using caffe::Blob;
  using caffe::Datum;
  
  class NetInputCaffe: public NetInput
  {
  public:
    NetInputCaffe()
      :NetInput() {}
    ~NetInputCaffe() {}
    
    void configure_inputs(const APIData &ad);
  };

  class NetLayersCaffe
  {
  public:
    
  };

  class NetLossCaffe
  {
  public:
    
  };
  
  template <class TNetInputCaffe, class TNetLayersCaffe, class TNetLossCaffe>
    class NetCaffe : public NetGenerator<TNetInputCaffe,TNetLayersCaffe,TNetLossCaffe>
    {
    public:
      NetCaffe(const int &nclasses);
      ~NetCaffe() {}

    public:
      caffe::NetParameter net_params; /**< training net definition. */
      caffe::NetParameter dnet_params; /**< deploy net definition. */
      int _nclasses;
      
    };
  
}

#endif
