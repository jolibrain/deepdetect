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

#include "net_caffe_mlp.h"
#include "imginputfileconn.h"

namespace dd
{

  /*- NetLayersCaffeMLP -*/
  void NetLayersCaffeMLP::add_basic_block(caffe::NetParameter *net_param,
					  const std::string &bottom,
					  const std::string &top,
					  const int &num_output,
					  const std::string &activation,
					  const double &dropout_ratio,
					  const bool &bn,
					  const bool &sparse)
  {
    if (sparse)
      add_sparse_fc(net_param,bottom,top,num_output);
    else add_fc(net_param,bottom,top,num_output);
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

  void NetLayersCaffeMLP::configure_net(const APIData &ad_mllib)
  {
    std::vector<int> layers = {50}; // default
    std::string activation = CaffeCommon::set_activation(ad_mllib);
    double dropout = 0.0; // default
    if (ad_mllib.has("layers"))
      layers = ad_mllib.get("layers").get<std::vector<int>>();
    if (ad_mllib.has("dropout"))
      dropout = ad_mllib.get("dropout").get<double>();
    bool bn = false;
    if (ad_mllib.has("bn"))
      bn = ad_mllib.get("bn").get<bool>();
    int nclasses = -1;
    if (ad_mllib.has("nclasses"))
      nclasses = ad_mllib.get("nclasses").get<int>();
    int ntargets = -1;
    if (ad_mllib.has("ntargets"))
      ntargets = ad_mllib.get("ntargets").get<int>();
    bool sparse = false;
    if (ad_mllib.has("sparse"))
      sparse = ad_mllib.get("sparse").get<bool>();
    bool regression = false;
    if (ad_mllib.has("regression"))
      regression = ad_mllib.get("regression").get<bool>();
    bool autoencoder = false;
    if (ad_mllib.has("autoencoder"))
      autoencoder = ad_mllib.get("autoencoder").get<bool>();
    std::string bottom = "data";
    for (size_t l=0;l<layers.size();l++)
      {
	std::string top = "ip" + std::to_string(l);
	add_basic_block(this->_net_params,bottom,top,layers.at(l),activation,dropout,bn,sparse&&l==0?sparse:false);
	add_basic_block(this->_dnet_params,bottom,top,layers.at(l),activation,0.0,bn,sparse&&l==0?sparse:false);
	bottom = top;
      }
    //TODO: to loss ?
    if (regression)
      {
	add_euclidean_loss(this->_net_params,bottom,"label","losst",ntargets);
	add_euclidean_loss(this->_net_params,bottom,"","loss",ntargets,true);
	add_euclidean_loss(this->_dnet_params,bottom,"","loss",ntargets,true);
      }
    else if (autoencoder) //TODO: sigmoid crossentropy would be only for integer-type targets...
      {
	add_sigmoid_crossentropy_loss(this->_net_params,bottom,"data","losst",ntargets);
	add_act(this->_dnet_params,bottom,"sigmoid");
      }
    else
      {
	add_softmax(this->_net_params,bottom,"label","losst",nclasses > 0 ? nclasses : ntargets);
	add_softmax(this->_dnet_params,bottom,"","loss",nclasses > 0 ? nclasses : ntargets,true);
      }
  }

  template class NetCaffe<NetInputCaffe<ImgCaffeInputFileConn>,NetLayersCaffeMLP,NetLossCaffe>;
  template class NetCaffe<NetInputCaffe<CSVCaffeInputFileConn>,NetLayersCaffeMLP,NetLossCaffe>;
  template class NetCaffe<NetInputCaffe<TxtCaffeInputFileConn>,NetLayersCaffeMLP,NetLossCaffe>;
  template class NetCaffe<NetInputCaffe<SVMCaffeInputFileConn>,NetLayersCaffeMLP,NetLossCaffe>;
}
