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
#include <spdlog/spdlog.h>

namespace dd
{

  using caffe::Caffe;
  using caffe::Net;
  using caffe::Blob;
  using caffe::Datum;

  class CaffeCommon
  {
  public:
    static caffe::LayerParameter* add_layer(caffe::NetParameter *net_param,
					    const std::string &bottom,
					    const std::string &top,
					    const std::string &name="",
					    const std::string &type="",
					    const std::string &label="");
    
    static std::string set_activation(const APIData &ad_mllib);
    
  };
  
  template <class TInputCaffe>
  class NetInputCaffe: public NetInput<TInputCaffe>
  {
  public:
  NetInputCaffe(caffe::NetParameter *net_params,
		caffe::NetParameter *dnet_params)
    :NetInput<TInputCaffe>(),_net_params(net_params),_dnet_params(dnet_params)
    {
      _net_params->set_name("net");
      _dnet_params->set_name("dnet");
    }
    ~NetInputCaffe() {}
    
    void configure_inputs(const APIData &ad_mllib,
			  const TInputCaffe &inputc);
    
    void add_embed(caffe::NetParameter *net_param,
		   const std::string &bottom,
		   const std::string &top,
		   const int &input_dim,
		   const int &num_output);

    caffe::NetParameter *_net_params;
    caffe::NetParameter *_dnet_params;
  };

  class NetLayersCaffe: public NetLayers
  {
  public:
    NetLayersCaffe(caffe::NetParameter *net_params,
		   caffe::NetParameter *dnet_params,
		   std::shared_ptr<spdlog::logger> &logger)
      :NetLayers(),_net_params(net_params),_dnet_params(dnet_params),_logger(logger) {}
    
    //void add_basic_block() {}
    void configure_net(const APIData &ad) { (void)ad; }
    
    // common layers
    void add_fc(caffe::NetParameter *net_param,
		const std::string &bottom,
		const std::string &top,
		const int &num_output);

    void add_sparse_fc(caffe::NetParameter *net_param,
		       const std::string &bottom,
		       const std::string &top,
		       const int &num_output);
    
    void add_conv(caffe::NetParameter *net_param,
		  const std::string &bottom,
		  const std::string &top,
		  const int &num_output,
		  const int &kernel_size,
		  const int &pad,
		  const int &stride,
		  const int &kernel_w=0,
		  const int &kernel_h=0,
		  const int &pad_w=0,
		  const int &pad_h=0,
		  const std::string &name="",
		  const std::string &init="msra");

    void add_deconv(caffe::NetParameter *net_param,
		    const std::string &bottom,
		    const std::string &top,
		    const int &num_output,
		    const int &kernel_size,
		    const int &pad,
		    const int &stride,
		    const int &kernel_w=0,
		    const int &kernel_h=0,
		    const int &pad_w=0,
		    const int &pad_h=0,
		    const std::string &name="",
		    const std::string &init="msra");

    void add_act(caffe::NetParameter *net_param,
		 const std::string &bottom,
		 const std::string &activation,
		 const double &elu_alpha=1.0,
		 const double &negative_slope=0.0,
		 const bool &test=false);

    void add_pooling(caffe::NetParameter *net_param,
		     const std::string &bottom,
		     const std::string &top,
		     const int &kernel_size,
		     const int &stride,
		     const std::string &type,
		     const int &kernel_w=0,
		     const int &kernel_h=0,
		     const int &stride_w=0,
		     const int &stride_h=0);

    void add_dropout(caffe::NetParameter *net_param,
		     const std::string &bottom,
		     const double &ratio);

    void add_lstm(caffe::NetParameter *net_param,
                  const std::string &seq,
                  const std::string &cont,
                  const std::string &name);

    void add_rnn(caffe::NetParameter *net_param,
                 const std::string &seq,
                 const std::string &cont,
                 const std::string &name);


    void add_bn(caffe::NetParameter *net_param,
		const std::string &bottom,
		const std::string &top="");

    void add_eltwise(caffe::NetParameter *net_param,
		     const std::string &bottom1,
		     const std::string &bottom2,
		     const std::string &top);

    void add_reshape(caffe::NetParameter *net_param,
		     const std::string &bottom,
		     const std::string &top,
		     const caffe::ReshapeParameter &r_param); //TODO

    // requires a fully connected layer (all losses ?)
    void add_softmax(caffe::NetParameter *net_param,
		     const std::string &bottom,
		     const std::string &label,
		     const std::string &top,
		     const int &num_output,
		     const bool &deploy=false);

    void add_euclidean_loss(caffe::NetParameter *net_param,
			    const std::string &bottom,
			    const std::string &label,
			    const std::string &top,
			    const int &num_output,
			    const bool &deploy=false);

    void add_sigmoid_crossentropy_loss(caffe::NetParameter *net_param,
				       const std::string &bottom,
				       const std::string &label,
				       const std::string &top,
				       const int &num_output,
				       const bool &deploy=false,
				       const bool &fc=true);

    void add_interp(caffe::NetParameter *net_param,
		    const std::string &bottom,
		    const std::string &top,
		    const int &interp_width,
		    const int &interp_height);

    void add_flatten(caffe::NetParameter *net_param,
		     const std::string &bottom,
		     const std::string &top,
		     const bool &test=false);
    
    caffe::NetParameter *_net_params;
    caffe::NetParameter *_dnet_params;
    std::shared_ptr<spdlog::logger> _logger;
  };

  class NetLossCaffe
  {
  public:
    
  };
  
  template <class TNetInputCaffe, class TNetLayersCaffe, class TNetLossCaffe>
    class NetCaffe : public NetGenerator<TNetInputCaffe,TNetLayersCaffe,TNetLossCaffe>
    {
    public:
      NetCaffe(caffe::NetParameter *net_params,
	       caffe::NetParameter *dnet_params,
	       std::shared_ptr<spdlog::logger> &logger)
	:_net_params(net_params),_dnet_params(dnet_params),
	_nic(net_params,dnet_params),_nlac(net_params,dnet_params,logger),_logger(logger) {}
      ~NetCaffe() {}

    public:
      caffe::NetParameter* _net_params; /**< training net definition. */
      caffe::NetParameter* _dnet_params; /**< deploy net definition. */

      TNetInputCaffe _nic;
      TNetLayersCaffe _nlac;
      //TNetLossCaffe _nloc
      std::shared_ptr<spdlog::logger> _logger;
    };
  
}

#endif
