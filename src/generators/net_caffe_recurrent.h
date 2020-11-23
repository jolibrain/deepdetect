/**
 * DeepDetect
 * Copyright (c) 2018-2019 Emmanuel Benazera
 * Author: Guillaume Infantes <guillaume.infantes@jolibrain.com>
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

#ifndef NET_CAFFE_RECURRENT_H
#define NET_CAFFE_RECURRENT_H

#include "generators/net_caffe.h"

namespace dd
{

  class NetLayersCaffeRecurrent : public NetLayersCaffe
  {
  public:
    NetLayersCaffeRecurrent(caffe::NetParameter *net_params,
                            caffe::NetParameter *dnet_params,
                            std::shared_ptr<spdlog::logger> &logger)
        : NetLayersCaffe(net_params, dnet_params, logger)
    {
      net_params->set_name("recurrent");
      dnet_params->set_name("recurrent");
    }
    ~NetLayersCaffeRecurrent()
    {
    }

    void add_basic_block(caffe::NetParameter *net_param,
                         const std::vector<std::string> &bottom_seq,
                         const std::string &bottom_cont,
                         const std::string &top, const int &num_output,
                         const double &dropout_ratio,
                         const std::string weight_filler,
                         const std::string bias_filler,
                         const std::string &type, const int id,
                         bool expose_hidden = false);

    void configure_net(const APIData &ad_mllib, bool expose_hidden = false);

    void add_concat(caffe::NetParameter *net_params, std::string name,
                    std::string top, std::vector<std::string> bottoms,
                    int axis);

    void add_slicer(caffe::NetParameter *net_params, int slice_points,
                    std::string bottom, std::string targets_name,
                    std::string inputs_name, std::string cont_seq);

    void add_flatten(caffe::NetParameter *net_params, std::string name,
                     std::string bottom, std::string top, int axis);

    void add_permute(caffe::NetParameter *net_params, std::string top,
                     std::string bottom, int naxis, bool train, bool test);

    void add_affine(caffe::NetParameter *net_params, std::string name,
                    const std::vector<std::string> &bottom, std::string top,
                    const std::string weight_filler,
                    const std::string bias_filler, int nout, int nin,
                    bool relu);

    void add_tile(caffe::NetParameter *net_param, const std::string layer_name,
                  const std::string bottom_name, const std::string top_name);

    void parse_recurrent_layers(const std::vector<std::string> &layers,
                                std::vector<std::string> &r_layers,
                                std::vector<int> &h_sizes);

  private:
    const std::string _lstm_str = "L";
    const std::string _rnn_str = "R";
    const std::string _affine_str = "A";
    const std::string _affine_relu_str = "AR";
    const std::string _tile_str = "T";
  };

  /**
   * \brief	create recurrent.prototxt using api data
   * @param ad input apidata
   * @param inputc input connector
   * @param net_param prototxt infos
   */
  template <class TInputConnectorStrategy>
  void configure_recurrent_template(const APIData &ad,
                                    TInputConnectorStrategy &inputc,
                                    caffe::NetParameter &net_param,
                                    std::shared_ptr<spdlog::logger> &logger,
                                    bool expose_hidden = false);
}

#endif
