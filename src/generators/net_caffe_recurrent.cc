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

#include "net_caffe_recurrent.h"
#include "imginputfileconn.h"

namespace dd
{


  void NetLayersCaffeRecurrent::add_basic_block(caffe::NetParameter *net_param,
                                                const std::string &bottom_seq,
                                                const std::string &bottom_cont,
                                                const std::string &top,
                                                const int &num_output,
                                                const double &dropout_ratio,
                                                const std::string &type,
                                                const int id)
  {
    std::string ctype = (type == "R" ?  "RNN" : "LSTM");
    caffe::LayerParameter *lparam = net_param->add_layer();
    lparam->set_type(ctype);
    lparam->set_name(ctype+std::to_string(id));
    if (dropout_ratio ==0)
      lparam->add_top(top);
    else
      lparam->add_top(ctype+std::to_string(id)+"_undropped");
    lparam->add_bottom(bottom_seq);
    lparam->add_bottom(bottom_cont);
    caffe::RecurrentParameter *rparam = lparam->mutable_recurrent_param();
    rparam->set_num_output(num_output);
    caffe::FillerParameter *weight_filler_param = rparam->mutable_weight_filler();
    weight_filler_param->set_type("uniform");
    weight_filler_param->set_min(-1.0);
    weight_filler_param->set_max(1.0);
    caffe::FillerParameter *bias_filler_param = rparam->mutable_bias_filler();
    bias_filler_param->set_type("constant");
    bias_filler_param->set_value(0);

    if (dropout_ratio !=0)
      {
        lparam = net_param->add_layer();
        lparam->set_type("Dropout");
        lparam->set_name(ctype+std::to_string(id)+"_dropout");
        lparam->add_top(top);
        lparam->add_bottom(ctype+std::to_string(id)+"_undropped");
        caffe::DropoutParameter *drop_param = lparam->mutable_dropout_param();
        drop_param->set_dropout_ratio(dropout_ratio);
      }
  }

  void NetLayersCaffeRecurrent::add_flatten(caffe::NetParameter *net_params,
                                            std::string bottom, std::string top, int axis)
  {
    caffe::LayerParameter *lparam = net_params->add_layer();
    lparam->set_type("Flatten");
    lparam->set_name("shape_cont_seq");
    lparam->add_top(top);
    lparam->add_bottom(bottom);
    caffe::FlattenParameter * fparam = lparam->mutable_flatten_param();
    fparam->set_axis(axis);
  }

  void NetLayersCaffeRecurrent::add_slicer(caffe::NetParameter *net_params,
                                           std::set<int> slice_points,
                                           std::vector<std::string> tops,
                                           std::string bottom,
                                           std::string cont_seq)
  {
    caffe::LayerParameter *lparam = net_params->add_layer();
    lparam->set_type("Slice");
    lparam->set_name("slice_timeseries");
    lparam->add_top(cont_seq);

    caffe::SliceParameter *sparam = lparam->mutable_slice_param();
    sparam->set_slice_dim(2);
    //sparam->add_slice_point(1);
    slice_points.insert(1);
    for (std::set<int>::iterator i = slice_points.begin(); i!= slice_points.end(); ++i)
      {
        sparam->add_slice_point(*i);
      }
    for (auto t:tops)
      lparam->add_top(t);
    lparam->add_bottom(bottom);
  }

  void NetLayersCaffeRecurrent::add_permute(caffe::NetParameter *net_params, std::string top, std::string bottom)
  {
    caffe::LayerParameter *lparam = net_params->add_layer();
    lparam->set_type("Permute");
    lparam->set_name("permute_T_N");
    lparam->add_top(top);
    lparam->add_bottom(bottom);
    caffe::PermuteParameter *pparam = lparam->mutable_permute_param();
    pparam->add_order(1);
    pparam->add_order(0);
    pparam->add_order(2);
    pparam->add_order(3);
  }

  void NetLayersCaffeRecurrent::add_concat(caffe::NetParameter *net_params,
                                           std::string name,
                                           std::string top,
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


  void NetLayersCaffeRecurrent::configure_net(const APIData &ad_mllib)
  {

    std::vector<std::string> layers = {"L","L"}; // default
    std::vector<double> dropouts; // default
    double dropout = 0.0;
    std::vector<int> targets = {1}; // index of csv columns to predict
    std::vector<int> inputs = {0}; // index of csv columns to use a input
    std::string loss = "SmoothL1Loss";
    std::string loss_temp;
    double sl1sigma = 0.000000001;

    int ncols = 2;
    int hidden = 50;
    if (ad_mllib.has("layers"))
      layers = ad_mllib.get("layers").get<std::vector<std::string>>();
    if (ad_mllib.has("dropout"))
      dropout = ad_mllib.get("dropout").get<double>();
    if (ad_mllib.has("dropouts"))
      dropouts = ad_mllib.get("dropouts").get<std::vector<double>>();
    else
      dropouts =std::vector<double>(layers.size(), dropout);
    if (ad_mllib.has("targets"))
      targets = ad_mllib.get("targets").get<std::vector<int>>();
    if (ad_mllib.has("inputs"))
      inputs = ad_mllib.get("inputs").get<std::vector<int>>();
    if (ad_mllib.has("ncols"))
      ncols = ad_mllib.get("ncols").get<int>();
    if (ad_mllib.has("hidden"))
      hidden = ad_mllib.get("hidden").get<int>();
    if (ad_mllib.has("loss"))
      {
        loss_temp = ad_mllib.get("loss").get<std::string>();
        if (loss_temp == "L2" || loss_temp == "euclidean")
          loss = "EuclideanLoss";
      }
    if (ad_mllib.has("sl1sigma"))
      sl1sigma = ad_mllib.get("sl1sigma").get<double>();
    else
      sl1sigma = 10.0; //override proto default in order to be sharp L1

    // caffe::LayerParameter *i1param = this->_net_params->mutable_layer(0);
    // caffe::MemoryDataParameter * mparam = i1param->mutable_memory_data_param();
    // mparam->set_height(1+ncols);
    // i1param = this->_net_params->mutable_layer(1);
    // mparam = i1param->mutable_memory_data_param();
    // mparam->set_height(1+ncols);

    std::string bottom = "data";
    std::sort(targets.begin(), targets.end());
    std::sort(inputs.begin(), inputs.end());

    // first permute
    add_permute(this->_net_params, "permuted_data", "data");
    add_permute(this->_dnet_params, "permuted_data", "data");


    // slice
    // one output is continuation indicator,
    // others are separation from input to ouput

    std::set<int> slice_points;
    for (int unsigned i=0; i< targets.size(); ++i)
      {
        slice_points.insert(targets[i]+1);
        slice_points.insert(targets[i]+2);
      }
    for (int unsigned i=0; i< inputs.size(); ++i)
      {
        slice_points.insert(inputs[i]+1);
        slice_points.insert(inputs[i]+2);
      }
    slice_points.erase(ncols+1);

    std::vector<std::string> tops;
    std::vector<std::string> bottoms;
    std::vector<int> ttargets(targets);
    std::vector<int> tinputs(inputs);
    int last_treated = -1;
    while (!ttargets.empty() || !tinputs.empty())
      {
        int first_t = *(ttargets.begin());
        int first_i = *(tinputs.begin());
        if (first_t < first_i || tinputs.empty())
          {
            if (first_t > last_treated+1)
              {
                if (first_t-1 == last_treated+1)
                  tops.push_back("unused_"+std::to_string(first_t -1));
                else
                  tops.push_back("unused_"+std::to_string(last_treated+1)+"_"
                                 +std::to_string(first_t -1));
              }
            tops.push_back("target_"+std::to_string(first_t));
            ttargets.erase(ttargets.begin());
            last_treated = first_t;
          }
        else
          {
            if (first_i > last_treated+1)
              {
                if (first_i-1 == last_treated+1)
                  tops.push_back("unused_"+std::to_string(first_i -1));
                else
                  tops.push_back("unused_"+std::to_string(last_treated+1)+"_"
                                 +std::to_string(first_i -1));
              }
            tops.push_back("input_"+std::to_string(first_i));
            tinputs.erase(tinputs.begin());
            last_treated = first_i;
          }
      }

    add_slicer(this->_net_params, slice_points, tops, "permuted_data","cont_seq_unshaped");
    add_slicer(this->_dnet_params, slice_points, tops, "permuted_data","cont_seq_unshaped");

    add_flatten(this->_net_params, "cont_seq_unshaped","cont_seq", 1);
    add_flatten(this->_dnet_params,"cont_seq_unshaped","cont_seq", 1);

    std::vector<std::string> input_names;
    for (unsigned int i = 0; i< inputs.size(); ++i)
      {
        input_names.push_back("input_"+std::to_string(inputs[i]));
      }

    add_concat(this->_net_params, "concat_inputs", "input_seq",input_names, 2);
    add_concat(this->_dnet_params, "concat_inputs", "input_seq",input_names, 2);

    std::vector<std::string> target_names;
    for (unsigned int i = 0; i< targets.size(); ++i)
      {
        target_names.push_back("target_"+std::to_string(targets[i]));
      }

    add_concat(this->_net_params, "concat_targets", "target_seq",target_names, 2);
    add_concat(this->_dnet_params, "concat_targets", "target_seq",target_names, 2);

    // lstm0

    int first_num_output = layers.size() <=1 ? targets.size() : hidden;
    double first_dropout_ratio = layers.size() <=1 ? 0.0 : dropouts[0];
    std::string type = (layers[0] == "R"? "RNN":"LSTM");
    bottom = type+"_"+std::to_string(0);
    std::string top = bottom;
    add_basic_block(this->_net_params,"input_seq",
                    "cont_seq",bottom,first_num_output, first_dropout_ratio,type, 0);
    add_basic_block(this->_dnet_params,"input_seq",
                    "cont_seq",bottom,first_num_output, first_dropout_ratio,type, 0);
    for (unsigned int i=1; i<layers.size()-1; ++i)
      {
        top = type+"_"+std::to_string(i);
        add_basic_block(this->_net_params,bottom,
                        "cont_seq",top,hidden, dropouts[i],type, i);
        add_basic_block(this->_dnet_params,bottom,
                        "cont_seq",top,hidden, dropouts[i],type, i);
        bottom = top;
      }

    if (layers.size() > 1)
      {
        top =type+"_"+std::to_string(layers.size()-1);
        add_basic_block(this->_net_params,bottom,
                        "cont_seq",top,
                        targets.size(), 0.0,type, layers.size()-1);
        add_basic_block(this->_dnet_params,bottom,
                        "cont_seq",top,
                        targets.size(), 0.0,type, layers.size()-1);
      }


    caffe::LayerParameter *lparam;
    caffe::NetStateRule *nsr;
    lparam = CaffeCommon::add_layer(this->_net_params,top,"loss","loss",loss); // train

    lparam->add_bottom("target_seq");
    nsr = lparam->add_include();
    nsr->set_phase(caffe::TRAIN);
    if (loss == "SmoothL1Loss")
      {
        caffe::SmoothL1LossParameter *lp = lparam->mutable_smooth_l1_loss_param();
        lp->set_sigma(sl1sigma);
      }
  }

  template class NetCaffe<NetInputCaffe<CSVTSCaffeInputFileConn>,NetLayersCaffeRecurrent,NetLossCaffe>;
  // template class NetCaffe<NetInputCaffe<ImgCaffeInputFileConn>,NetLayersCaffeRecurrent,NetLossCaffe>;
  // template class NetCaffe<NetInputCaffe<CSVCaffeInputFileConn>,NetLayersCaffeRecurrent,NetLossCaffe>;
  // template class NetCaffe<NetInputCaffe<TxtCaffeInputFileConn>,NetLayersCaffeRecurrent,NetLossCaffe>;
  // template class NetCaffe<NetInputCaffe<SVMCaffeInputFileConn>,NetLayersCaffeRecurrent,NetLossCaffe>;

}
