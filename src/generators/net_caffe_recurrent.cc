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

#include "net_caffe_recurrent.h"
#include "imginputfileconn.h"
#include "mllibstrategy.h"

namespace dd
{

  void NetLayersCaffeRecurrent::add_tile(caffe::NetParameter *net_param,
                                         const std::string layer_name,
                                         const std::string bottom_name,
                                         const std::string top_name)
  {
    caffe::LayerParameter *lparam = net_param->add_layer();
    lparam->set_type("Tile");
    lparam->set_name(layer_name);
    lparam->add_top(top_name);
    lparam->add_bottom(bottom_name);
    caffe::TileParameter *tparam = lparam->mutable_tile_param();
    tparam->set_axis(1);
    tparam->set_tiles(-1); // -1 means timesteps to be found elsewhere
  }

  void NetLayersCaffeRecurrent::add_basic_block(
      caffe::NetParameter *net_param,
      const std::vector<std::string> &bottom_seqs,
      const std::string &bottom_cont, const std::string &top,
      const int &num_output, const double &dropout_ratio,
      const std::string weight_filler, const std::string bias_filler,
      const std::string &type, const int id, bool expose_hidden)
  {

    std::string bottom_seq;
    std::string ctype = (type == _rnn_str ? "RNN" : "LSTM");
    std::string name = ctype + std::to_string(id);

    if (bottom_seqs.size() > 1)
      {
        // add concat
        caffe::LayerParameter *lparam = net_param->add_layer();
        lparam->set_type("Concat");
        lparam->set_name(name + "_concat");
        lparam->add_top(name + "_concat");
        bottom_seq = name + "_concat";
        for (std::string s : bottom_seqs)
          lparam->add_bottom(s);
        caffe::ConcatParameter *cparam = lparam->mutable_concat_param();
        cparam->set_axis(2);
      }
    else
      bottom_seq = bottom_seqs[0];

    caffe::LayerParameter *lparam;
    if (expose_hidden)
      {
        lparam = net_param->add_layer();
        lparam->set_type("DummyData");
        lparam->set_name(name + "_hidden_init");
        lparam->add_top(name + "_h_init");
        lparam->add_top(name + "_c_init");
        caffe::DummyDataParameter *dparam = lparam->mutable_dummy_data_param();
        caffe::BlobShape *b = dparam->add_shape();
        b->add_dim(32);
        b->add_dim(num_output);
        b = dparam->add_shape();
        b->add_dim(32);
        b->add_dim(num_output);
      }

    lparam = net_param->add_layer();
    lparam->set_type(ctype);
    lparam->set_name(name);
    if (dropout_ratio == 0)
      lparam->add_top(top);
    else
      lparam->add_top(ctype + std::to_string(id) + "_undropped");
    if (expose_hidden)
      {
        lparam->add_top(name + "_final_h");
        lparam->add_top(name + "_final_c");
      }
    lparam->add_bottom(bottom_seq);
    lparam->add_bottom(bottom_cont);
    if (expose_hidden)
      {
        lparam->add_bottom(name + "_h_init");
        lparam->add_bottom(name + "_c_init");
      }
    caffe::RecurrentParameter *rparam = lparam->mutable_recurrent_param();
    rparam->set_num_output(num_output);
    if (expose_hidden)
      rparam->set_expose_hidden(true);
    caffe::FillerParameter *weight_filler_param
        = rparam->mutable_weight_filler();
    weight_filler_param->set_type(weight_filler);
    if (weight_filler == "uniform")
      {
        weight_filler_param->set_min(-1.0 / sqrt(num_output));
        weight_filler_param->set_max(1.0 / sqrt(num_output));
      }
    caffe::FillerParameter *bias_filler_param = rparam->mutable_bias_filler();
    bias_filler_param->set_type(bias_filler);
    if (bias_filler == "uniform")
      {
        bias_filler_param->set_min(-1.0 / sqrt(num_output));
        bias_filler_param->set_max(1.0 / sqrt(num_output));
      }

    if (dropout_ratio != 0)
      {
        lparam = net_param->add_layer();
        lparam->set_type("Dropout");
        lparam->set_name(ctype + std::to_string(id) + "_dropout");
        lparam->add_top(top);
        lparam->add_bottom(ctype + std::to_string(id) + "_undropped");
        caffe::DropoutParameter *drop_param = lparam->mutable_dropout_param();
        drop_param->set_dropout_ratio(dropout_ratio);
      }
  }

  void NetLayersCaffeRecurrent::add_flatten(caffe::NetParameter *net_params,
                                            std::string name,
                                            std::string bottom,
                                            std::string top, int axis)
  {
    caffe::LayerParameter *lparam = net_params->add_layer();
    lparam->set_type("Flatten");
    lparam->set_name(name);
    lparam->add_top(top);
    lparam->add_bottom(bottom);
    caffe::FlattenParameter *fparam = lparam->mutable_flatten_param();
    fparam->set_axis(axis);
  }

  void NetLayersCaffeRecurrent::add_slicer(caffe::NetParameter *net_params,
                                           int slice_point, std::string bottom,
                                           std::string targets_name,
                                           std::string inputs_name,
                                           std::string cont_seq)
  {
    caffe::LayerParameter *lparam = net_params->add_layer();
    lparam->set_type("Slice");
    lparam->set_name("slice_timeseries");

    caffe::SliceParameter *sparam = lparam->mutable_slice_param();
    sparam->set_axis(2);
    sparam->add_slice_point(1);
    sparam->add_slice_point(slice_point);
    lparam->add_top(cont_seq);
    lparam->add_top(targets_name);
    lparam->add_top(inputs_name);
    lparam->add_bottom(bottom);
  }

  void NetLayersCaffeRecurrent::add_permute(caffe::NetParameter *net_params,
                                            std::string top,
                                            std::string bottom, int naxis,
                                            bool train, bool test)
  {
    caffe::LayerParameter *lparam = net_params->add_layer();
    lparam->set_type("Permute");
    lparam->set_name("permute_T_N_" + bottom);
    lparam->add_top(top);
    lparam->add_bottom(bottom);
    caffe::PermuteParameter *pparam = lparam->mutable_permute_param();
    pparam->add_order(1);
    pparam->add_order(0);
    pparam->add_order(2);
    if (naxis == 4)
      pparam->add_order(3);
    if (train)
      {
        caffe::NetStateRule *nsr;
        nsr = lparam->add_include();
        nsr->set_phase(caffe::TRAIN);
      }
    else if (test)
      {
        caffe::NetStateRule *nsr;
        nsr = lparam->add_include();
        nsr->set_phase(caffe::TEST);
      }
  }

  void NetLayersCaffeRecurrent::add_concat(caffe::NetParameter *net_params,
                                           std::string name, std::string top,
                                           std::vector<std::string> bottoms,
                                           int axis)
  {
    caffe::LayerParameter *lparam = net_params->add_layer();
    lparam->set_type("Concat");
    lparam->set_name(name);
    lparam->add_top(top);
    for (auto i : bottoms)
      {
        lparam->add_bottom(i);
      }
    caffe::ConcatParameter *cparam = lparam->mutable_concat_param();
    cparam->set_axis(axis);
  }

  void NetLayersCaffeRecurrent::add_affine(
      caffe::NetParameter *net_params, std::string name,
      const std::vector<std::string> &bottoms, std::string top,
      const std::string weight_filler, const std::string bias_filler, int nout,
      int nin, bool relu)
  {
    std::string bottom;

    if (bottoms.size() > 1)
      {
        // add concat
        caffe::LayerParameter *lparam = net_params->add_layer();
        lparam->set_type("Concat");
        lparam->set_name(name + "_concat");
        lparam->add_top(name + "_concat");
        bottom = name + "_concat";
        for (std::string s : bottoms)
          lparam->add_bottom(s);
        caffe::ConcatParameter *cparam = lparam->mutable_concat_param();
        cparam->set_axis(2);
      }
    else
      bottom = bottoms[0];

    caffe::LayerParameter *lparam = net_params->add_layer();
    lparam->set_type("InnerProduct");
    lparam->set_name(name);
    if (relu)
      lparam->add_top(top + "_unrelued");
    else
      lparam->add_top(top);
    lparam->add_bottom(bottom);
    caffe::InnerProductParameter *cparam
        = lparam->mutable_inner_product_param();
    cparam->set_num_output(nout);
    cparam->set_axis(2);
    caffe::FillerParameter *wfparam = cparam->mutable_weight_filler();
    wfparam->set_type(weight_filler);
    if (weight_filler == "uniform")
      {
        wfparam->set_min(-1.0 / sqrt(nin));
        wfparam->set_max(1.0 / sqrt(nin));
      }
    caffe::FillerParameter *bfparam = cparam->mutable_bias_filler();
    bfparam->set_type(bias_filler);
    if (bias_filler == "uniform")
      {
        bfparam->set_min(-1.0 / sqrt(nin));
        bfparam->set_max(1.0 / sqrt(nin));
      }

    if (relu)
      {
        caffe::LayerParameter *lparam = net_params->add_layer();
        lparam->set_name(name + "_ReLU");
        lparam->set_type("ReLU");
        lparam->add_bottom(top + "_unrelued");
        lparam->add_top(top);
      }
  }

  void NetLayersCaffeRecurrent::parse_recurrent_layers(
      const std::vector<std::string> &layers,
      std::vector<std::string> &r_layers, std::vector<int> &h_sizes)
  {
    for (auto s : layers)
      {
        size_t pos = 0;
        if ((pos = s.find(_affine_relu_str)) != std::string::npos)
          {
            r_layers.push_back(_affine_relu_str);
            h_sizes.push_back(
                std::stoi(s.substr(pos + _affine_relu_str.size())));
          }
        else if ((pos = s.find(_lstm_str)) != std::string::npos)
          {
            r_layers.push_back(_lstm_str);
            h_sizes.push_back(std::stoi(s.substr(pos + _lstm_str.size())));
          }
        else if ((pos = s.find(_rnn_str)) != std::string::npos)
          {
            r_layers.push_back(_rnn_str);
            h_sizes.push_back(std::stoi(s.substr(pos + _rnn_str.size())));
          }
        else if ((pos = s.find(_affine_str)) != std::string::npos)
          {
            r_layers.push_back(_affine_str);
            h_sizes.push_back(std::stoi(s.substr(pos + _affine_str.size())));
          }
        else if ((pos = s.find(_tile_str)) != std::string::npos)
          {
            r_layers.push_back(_tile_str);
            h_sizes.push_back(-1);
          }
        else
          {
            try
              {
                h_sizes.push_back(std::stoi(s));
                r_layers.push_back(_lstm_str);
              }
            catch (std::exception &e)
              {
                throw MLLibBadParamException(
                    "timeseries template requires layers of the form "
                    "\"L50\". "
                    "L for LSTM, R for RNN, A for affine dimension "
                    "reduction, AR for affine + relu"
                    " and 50 for a hidden cell size of 50");
              }
          }
      }
  }

  void NetLayersCaffeRecurrent::configure_net(const APIData &ad_mllib,
                                              bool expose_hidden)
  {

    std::vector<std::string> layers;
    std::vector<int> osize;
    std::vector<double> dropouts; // default
    std::map<int, int> types;
    std::string loss = "L1";
    std::string loss_temp;
    std::string init = "uniform";
    int ntargets;
    //    double sl1sigma = 10;

    if (ad_mllib.has("layers"))
      {
        std::vector<std::string> apilayers
            = ad_mllib.get("layers").get<std::vector<std::string>>();
        parse_recurrent_layers(apilayers, layers, osize);
      }
    if (ad_mllib.has("dropout"))
      {
        try
          {
            double dropout = ad_mllib.get("dropout").get<double>();
            dropouts = std::vector<double>(layers.size(), dropout);
          }
        catch (std::exception &e)
          {
            try
              {
                dropouts = ad_mllib.get("dropout").get<std::vector<double>>();
              }
            catch (std::exception &e)
              {
                throw InputConnectorBadParamException(
                    "wrong type for dropout parameter");
              }
          }
      }
    if (ad_mllib.has("loss"))
      {
        loss_temp = ad_mllib.get("loss").get<std::string>();
        if (loss_temp == "L2" || loss_temp == "euclidean")
          loss = "EuclideanLoss";
      }
    ntargets = ad_mllib.get("ntargets").get<int>();

    bool residual = false;
    if (ad_mllib.has("residual"))
      {
        residual = ad_mllib.get("residual").get<bool>();
      }

    // if (ad_mllib.has("init"))
    //   {
    //     init = ad_mllib.get("init").get<std::string>();
    //   }

    // first permute
    add_permute(this->_net_params, "permuted_data", "data", 4, false, false);
    add_permute(this->_dnet_params, "permuted_data", "data", 4, false, false);

    int slice_point = 1 + ntargets;

    add_slicer(this->_net_params, slice_point, "permuted_data", "target_seq",
               "input_seq", "cont_seq_unshaped");
    add_slicer(this->_dnet_params, slice_point, "permuted_data", "target_seq",
               "input_seq", "cont_seq_unshaped");

    add_flatten(this->_net_params, "shape_cont_seq", "cont_seq_unshaped",
                "cont_seq", 1);
    add_flatten(this->_dnet_params, "shape_cont_seq", "cont_seq_unshaped",
                "cont_seq", 1);

    add_flatten(this->_net_params, "flatten_input", "input_seq",
                "input_seq_flattened", 2);
    add_flatten(this->_dnet_params, "flatten_input", "input_seq",
                "input_seq_flattened", 2);

    std::string type;
    std::vector<std::string> bottoms = { "input_seq_flattened" };
    std::string top;
    for (unsigned int i = 0; i < layers.size(); ++i)
      {
        if (layers[i] == _rnn_str)
          type = "RNN";
        else if (layers[i] == _lstm_str)
          type = "LSTM";
        else if (layers[i] == _affine_str)
          type = "AFFINE";
        else if (layers[i] == _affine_relu_str)
          type = "AFFINE_RELU";
        else if (layers[i] == _tile_str)
          type = "TILE";

        top = type + "_" + std::to_string(i);
        if ((i == layers.size() - 1) && osize[osize.size() - 1] == ntargets)
          top = "rnn_pred";
        else
          top = type + "_" + std::to_string(i);
        if (type == "AFFINE")
          {
            int isize = i == 0 ? osize[i]
                               : osize[i - 1]; // used only for initing weights
            add_affine(this->_net_params, "affine_" + std::to_string(i),
                       bottoms, top, init, init, osize[i], isize, false);
            add_affine(this->_dnet_params, "affine_" + std::to_string(i),
                       bottoms, top, init, init, osize[i], isize, false);
          }
        else if (type == "AFFINE_RELU")
          {
            int isize = i == 0 ? osize[i]
                               : osize[i - 1]; // used only for initing weights
            add_affine(this->_net_params, "affine_" + std::to_string(i),
                       bottoms, top, init, init, osize[i], isize, true);
            add_affine(this->_dnet_params, "affine_" + std::to_string(i),
                       bottoms, top, init, init, osize[i], isize, true);
          }
        else if (type == "TILE")
          {
            // get previous layer name
            std::string prevname;
            if (layers[i - 1] == _rnn_str)
              prevname = "RNN" + std::to_string(i - 1) + "_final_h";
            prevname = "LSTM" + std::to_string(i - 1) + "_final_h";
            add_tile(this->_net_params, "tile_" + std::to_string(i), prevname,
                     top);
            add_tile(this->_dnet_params, "tile_" + std::to_string(i), prevname,
                     top);
            dropouts.insert(dropouts.begin() + i, 0.0);
          }
        else
          {
            add_basic_block(this->_net_params, bottoms, "cont_seq", top,
                            osize[i], dropouts[i], init, init, layers[i], i,
                            expose_hidden);
            add_basic_block(this->_dnet_params, bottoms, "cont_seq", top,
                            osize[i], dropouts[i], init, init, layers[i], i,
                            expose_hidden);
          }
        if (!residual)
          bottoms.clear();
        bottoms.push_back(top);
      }

    // add affine dim reduction only if num of  outputs of last layer  do not
    // match ntarget number
    if (osize[osize.size() - 1] != ntargets)
      {
        add_affine(this->_net_params, "affine_final", bottoms, "rnn_pred",
                   init, init, ntargets, osize[osize.size() - 1], false);
        add_affine(this->_dnet_params, "affine_final", bottoms, "rnn_pred",
                   init, init, ntargets, osize[osize.size() - 1], false);
      }

    add_permute(this->_net_params, "permuted_rnn_pred", "rnn_pred", 3, true,
                false);
    add_permute(this->_net_params, "permuted_target_seq", "target_seq", 3,
                true, false);

    if (loss == "EuclideanLoss")
      {
        caffe::LayerParameter *lparam;
        lparam = CaffeCommon::add_layer(this->_net_params, "permuted_rnn_pred",
                                        "loss", "loss", loss);
        lparam->add_bottom("permuted_target_seq");
        caffe::NetStateRule *nsr;
        nsr = lparam->add_include();
        nsr->set_phase(caffe::TRAIN);
      }
    else
      {

        caffe::LayerParameter *lparam;
        lparam = CaffeCommon::add_layer(
            this->_net_params, "permuted_target_seq",
            "permuted_target_seq_flattened", "Target_Seq_Dim", "Flatten");
        caffe::FlattenParameter *ffp = lparam->mutable_flatten_param();
        ffp->set_axis(2);
        caffe::NetStateRule *nsr;
        nsr = lparam->add_include();
        nsr->set_phase(caffe::TRAIN);

        lparam = CaffeCommon::add_layer(this->_net_params, "permuted_rnn_pred",
                                        "difference", "Loss_Sum_Layer",
                                        "Eltwise");
        lparam->add_bottom("permuted_target_seq_flattened");
        caffe::EltwiseParameter *ep = lparam->mutable_eltwise_param();
        ep->set_operation(caffe::EltwiseParameter::SUM);
        ep->add_coeff(1.0);
        ep->add_coeff(-1.0);
        nsr = lparam->add_include();
        nsr->set_phase(caffe::TRAIN);

        lparam = CaffeCommon::add_layer(this->_net_params, "difference",
                                        "summed_difference", "Loss_Reduction",
                                        "Reduction");
        caffe::ReductionParameter *rp = lparam->mutable_reduction_param();
        rp->set_operation(caffe::ReductionParameter::ASUM);
        rp->set_axis(1);
        nsr = lparam->add_include();
        nsr->set_phase(caffe::TRAIN);

        lparam = CaffeCommon::add_layer(this->_net_params, "summed_difference",
                                        "scaled_difference", "Loss_Scale",
                                        "Scale");
        caffe::ScaleParameter *sp = lparam->mutable_scale_param();
        caffe::FillerParameter *fp = sp->mutable_filler();
        fp->set_type("constant");
        fp->set_value(1.0);
        sp->set_axis(0);
        sp->set_bias_term(false);
        caffe::ParamSpec *ps = lparam->add_param();
        ps->set_lr_mult(0.0);
        ps->set_decay_mult(0.0);
        nsr = lparam->add_include();
        nsr->set_phase(caffe::TRAIN);

        lparam = CaffeCommon::add_layer(this->_net_params, "scaled_difference",
                                        "loss", "Loss_Reduction_Batch",
                                        "Reduction");
        rp = lparam->mutable_reduction_param();
        rp->set_operation(caffe::ReductionParameter::SUM);
        lparam->add_loss_weight(1.0);
        nsr = lparam->add_include();
        nsr->set_phase(caffe::TRAIN);
      }
  }

  template <class TInputConnectorStrategy>
  void configure_recurrent_template(const APIData &ad,
                                    TInputConnectorStrategy &inputc,
                                    caffe::NetParameter &net_param,
                                    std::shared_ptr<spdlog::logger> &logger,
                                    bool expose_hidden)
  {
    caffe::NetParameter dnet_param;
    NetCaffe<NetInputCaffe<TInputConnectorStrategy>, NetLayersCaffeRecurrent,
             NetLossCaffe>
        netcaffe(&net_param, &dnet_param, logger);
    netcaffe._nic.configure_inputs(ad, inputc);
    // add ntargets to ad
    const_cast<APIData &>(ad).add("ntargets", (int)inputc._ntargets);
    netcaffe._nlac.configure_net(ad, expose_hidden);
    // will be changed at predict time
    // small default for debug
    net_param.mutable_layer(0)->mutable_memory_data_param()->set_channels(10);
  }

#ifdef USE_CAFFE
  template class NetCaffe<NetInputCaffe<CSVTSCaffeInputFileConn>,
                          NetLayersCaffeRecurrent, NetLossCaffe>;
#endif
#ifdef USE_TORCH
  template void configure_recurrent_template<CSVTSTorchInputFileConn>(
      const APIData &ad, CSVTSTorchInputFileConn &inputc,
      caffe::NetParameter &net_param, std::shared_ptr<spdlog::logger> &logger,
      bool expose_hidden);
  template void configure_recurrent_template<ImgTorchInputFileConn>(
      const APIData &ad, ImgTorchInputFileConn &inputc,
      caffe::NetParameter &net_param, std::shared_ptr<spdlog::logger> &logger,
      bool expose_hidden);
  template void configure_recurrent_template<TxtTorchInputFileConn>(
      const APIData &ad, TxtTorchInputFileConn &inputc,
      caffe::NetParameter &net_param, std::shared_ptr<spdlog::logger> &logger,
      bool expose_hidden);
#endif
  // template class
  // NetCaffe<NetInputCaffe<ImgCaffeInputFileConn>,NetLayersCaffeRecurrent,NetLossCaffe>;
  // template class
  // NetCaffe<NetInputCaffe<CSVCaffeInputFileConn>,NetLayersCaffeRecurrent,NetLossCaffe>;
  // template class
  // NetCaffe<NetInputCaffe<TxtCaffeInputFileConn>,NetLayersCaffeRecurrent,NetLossCaffe>;
  // template class
  // NetCaffe<NetInputCaffe<SVMCaffeInputFileConn>,NetLayersCaffeRecurrent,NetLossCaffe>;
}
