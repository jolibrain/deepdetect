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

#ifndef CAFFE_TEMPLATES_H
#define CAFFE_TEMPLATES_H

#include "apidata.h"
#include "caffe/caffe.hpp"

namespace dd
{

  template<class TInputConnectorStrategy>
  class CaffeTemplates
  {
  public:

    CaffeTemplates() {}
    ~CaffeTemplates() {}
    
    // inputs
    static void configure_inputs(caffe::NetParameter &net_param,
				 const APIData &ad,
				 const int &nclasses,
				 const int &targets,
				 const int &channels,
				 const int &width,
				 const int &height,
				 const bool &deploy,
				 const bool &db,
				 const bool &has_mean_file);

    // outputs
    static void write_model_out(caffe::NetParameter &net_param,
				std::string &outfile);

    // options
    static std::string set_activation(const APIData &ad);
    
    // basic layers
    static caffe::LayerParameter* add_layer(caffe::NetParameter &net_param,
					    const std::string &bottom,
					    const std::string &top);
    static void add_fc(caffe::NetParameter &net_param,
		       const std::string &bottom,
		       const std::string &top,
		       const int &num_output);
    static void add_conv(caffe::NetParameter &net_param,
			 const std::string &bottom,
			 const std::string &top,
			 const int &num_output,
			 const int &kernel_size,
			 const int &pad,
			 const int &stride);
    static void add_act(caffe::NetParameter &net_param,
			const std::string &bottom,
			const std::string &activation,
			const double &elu_alpha=1.0);
    static void add_pooling(caffe::NetParameter &net_param,
			    const std::string &bottom,
			    const std::string &top,
			    const int &kernel_size,
			    const int &stride,
			    const std::string &type);
    static void add_dropout(caffe::NetParameter &net_param,
			    const std::string &bottom,
			    const double &ratio);
    static void add_bn(caffe::NetParameter &net_param,
		       const std::string &bottom);
    static void add_eltwise(caffe::NetParameter &net_param,
			    const std::string &bottom1,
			    const std::string &bottom2,
			    const std::string &top);
    static void add_softmax(caffe::NetParameter &net_param,
			    const std::string &bottom,
			    const std::string &label,
			    const std::string &top,
			    const int &num_output,
			    const bool &with_loss,
			    const bool &deploy);

    // mlp
    static void add_mlp_basic_block(caffe::NetParameter &net_param,
				    const std::string &bottom,
				    const std::string &top,
				    const int &num_output,
				    const std::string &activation,
				    const double &dropout_ratio,
				    const bool &bn);
    
    static void configure_mlp_template(const APIData &ad,
				       const bool &regression,
				       const int &targets,
				       const int &cnclasses,
				       caffe::NetParameter &net_param,
				       caffe::NetParameter &deploy_net_param);

    // convnet
    static void parse_conv_layers(const std::vector<std::string> &layers,
				  std::vector<std::pair<int,int>> &cr_layers,
				  std::vector<int> &fc_layers);
    
    static void add_convnet_basic_block(caffe::NetParameter &net_param,
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

    static void configure_convnet_template(const APIData &ad,
					   const bool &regression,
					   const int &targets,
					   const int &cnclasses,
					   const bool &text,
					   caffe::NetParameter &net_param,
					   caffe::NetParameter &deploy_net_param);
    
    // charnn / TODO
    static void add_charnn_basic_block(caffe::NetParameter &net_param,
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

    static void configure_charnn_template(const APIData &ad,
					  const bool &regression,
					  const int &targets,
					  const int &cnclasses,
					  const int &alphabet_size, // width
					  const int &sequence, // height
					  caffe::NetParameter &net_param,
					  caffe::NetParameter &deploy_net_param);

    // resnet
    static void add_resnet_init_block(caffe::NetParameter &net_param,
				      const std::string &bottom,
				      const std::string &top);

    static void add_resnet_basic_block(caffe::NetParameter &net_param,
				       const int &block_num,
				       const std::string &bottom,
				       const std::string &top,
				       const int &num_output,
				       const std::string &activation,
				       const bool &identity);

    static void configure_resnet_template(const APIData &ad,
					  const bool &regression,
					  const int &targets,
					  const int &cnclasses,
					  caffe::NetParameter &net_param,
					  caffe::NetParameter &deploy_net_param);

  public:
    //TODO: move some parameters here
  };
  
}

#endif
