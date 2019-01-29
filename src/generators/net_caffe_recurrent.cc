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
    weight_filler_param->set_min(-1.0/sqrt(num_output));
    weight_filler_param->set_max(1.0/sqrt(num_output));
    caffe::FillerParameter *bias_filler_param = rparam->mutable_bias_filler();
    bias_filler_param->set_type("uniform");
    bias_filler_param->set_min(-1.0/sqrt(num_output));
    bias_filler_param->set_max(1.0/sqrt(num_output));

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
                                           int slice_point,
                                           std::string bottom,
                                           std::string targets_name,
                                           std::string inputs_name,
                                           std::string cont_seq)
  {
    caffe::LayerParameter *lparam = net_params->add_layer();
    lparam->set_type("Slice");
    lparam->set_name("slice_timeseries");

    caffe::SliceParameter *sparam = lparam->mutable_slice_param();
    sparam->set_slice_dim(2);
    sparam->add_slice_point(1);
    sparam->add_slice_point(slice_point);
    lparam->add_top(cont_seq);
    lparam->add_top(targets_name);
    lparam->add_top(inputs_name);
    lparam->add_bottom(bottom);
  }

  void NetLayersCaffeRecurrent::add_permute(caffe::NetParameter *net_params, std::string top, std::string bottom, int naxis, bool train, bool test)
  {
    caffe::LayerParameter *lparam = net_params->add_layer();
    lparam->set_type("Permute");
    lparam->set_name("permute_T_N_"+bottom);
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

  void NetLayersCaffeRecurrent::add_affine(caffe::NetParameter *net_params,
                                           std::string name,
                                           std::string bottom,
                                           std::string top,
                                           int nout)
  {
    caffe::LayerParameter *lparam = net_params->add_layer();
    lparam->set_type("InnerProduct");
    lparam->set_name(name);
    lparam->add_top(top);
    lparam->add_bottom(bottom);
    caffe::InnerProductParameter *cparam = lparam->mutable_inner_product_param();
    cparam->set_num_output(nout);
    cparam->set_axis(2);
    caffe::FillerParameter *wfparam = cparam->mutable_weight_filler();
    wfparam->set_type("xavier");
    caffe::FillerParameter *bfparam = cparam->mutable_bias_filler();
    bfparam->set_type("xavier");
  }


  void NetLayersCaffeRecurrent::configure_net(const APIData &ad_mllib)
  {

    std::vector<std::string> layers = {"L","L"}; // default
    std::vector<double> dropouts; // default
    double dropout = 0.0;
    std::map<int,int> types;
    std::string loss = "L1";
    std::string loss_temp;
    int ntargets;
    //    double sl1sigma = 10;

    int hidden = 50;
    if (ad_mllib.has("layers"))
      layers = ad_mllib.get("layers").get<std::vector<std::string>>();
    if (ad_mllib.has("dropout"))
      dropout = ad_mllib.get("dropout").get<double>();
    if (ad_mllib.has("dropouts"))
      dropouts = ad_mllib.get("dropouts").get<std::vector<double>>();
    else
      dropouts =std::vector<double>(layers.size(), dropout);
    if (ad_mllib.has("hidden"))
      hidden = ad_mllib.get("hidden").get<int>();
    if (ad_mllib.has("loss"))
      {
        loss_temp = ad_mllib.get("loss").get<std::string>();
        if (loss_temp == "L2" || loss_temp == "euclidean")
          loss = "EuclideanLoss";
      }
    ntargets = ad_mllib.get("ntargets").get<int>();

    std::string bottom = "data";

    // first permute
    add_permute(this->_net_params, "permuted_data", "data", 4,false,false);
    add_permute(this->_dnet_params, "permuted_data", "data", 4,false,false);


    int slice_point = 1 + ntargets;

    add_slicer(this->_net_params, slice_point, "permuted_data",
               "target_seq","input_seq","cont_seq_unshaped");
    add_slicer(this->_dnet_params, slice_point, "permuted_data",
               "target_seq","input_seq","cont_seq_unshaped");

    add_flatten(this->_net_params, "cont_seq_unshaped","cont_seq", 1);
    add_flatten(this->_dnet_params,"cont_seq_unshaped","cont_seq", 1);

    // lstm0

    int first_num_output = layers.size() <=1 ? ntargets : hidden;
    double first_dropout_ratio = layers.size() <=1 ? 0.0 : dropouts[0];
    std::string type = (layers[0] == "R"? "RNN":"LSTM");
    bottom = type+"_"+std::to_string(0);
    std::string top = bottom;
    add_basic_block(this->_net_params,"input_seq",
                    "cont_seq",bottom,first_num_output, first_dropout_ratio,type, 0);
    add_basic_block(this->_dnet_params,"input_seq",
                    "cont_seq",bottom,first_num_output, first_dropout_ratio,type, 0);
    for (unsigned int i=1; i<layers.size(); ++i)
      {
        top = type+"_"+std::to_string(i);
        add_basic_block(this->_net_params,bottom,
                        "cont_seq",top,hidden, dropouts[i],type, i);
        add_basic_block(this->_dnet_params,bottom,
                        "cont_seq",top,hidden, dropouts[i],type, i);
        bottom = top;
      }


    add_affine(this->_net_params,"affine",bottom,"rnn_pred", ntargets);
    add_affine(this->_dnet_params,"affine", bottom,"rnn_pred",  ntargets);

    add_permute(this->_net_params, "permuted_rnn_pred", "rnn_pred", 3,true,false);
    add_permute(this->_net_params, "permuted_target_seq", "target_seq", 3,true,false);



    if (loss == "EuclideanLoss")
      {
        caffe::LayerParameter *lparam;
        lparam = CaffeCommon::add_layer(this->_net_params,"permuted_rnn_pred","loss","loss",loss);
        lparam->add_bottom("permuted_target_seq");
        caffe::NetStateRule *nsr;
        nsr = lparam->add_include();
        nsr->set_phase(caffe::TRAIN);
      }
    else
      {

        caffe::LayerParameter *lparam;
        lparam = CaffeCommon::add_layer(this->_net_params,"permuted_target_seq", "permuted_target_seq_flattened",
                                        "Target_Seq_Dim","Flatten");
        caffe::FlattenParameter *ffp = lparam->mutable_flatten_param();
        ffp->set_axis(2);
        caffe::NetStateRule *nsr;
        nsr = lparam->add_include();
        nsr->set_phase(caffe::TRAIN);

        lparam = CaffeCommon::add_layer(this->_net_params,"permuted_rnn_pred", "difference",
                                        "Loss_Sum_Layer","Eltwise");
        lparam->add_bottom("permuted_target_seq_flattened");
        caffe::EltwiseParameter *ep = lparam->mutable_eltwise_param();
        ep->set_operation(caffe::EltwiseParameter::SUM);
        ep->add_coeff(1.0);
        ep->add_coeff(-1.0);
        nsr = lparam->add_include();
        nsr->set_phase(caffe::TRAIN);

        lparam = CaffeCommon::add_layer(this->_net_params,"difference","summed_difference",
                                        "Loss_Reduction","Reduction");
        caffe::ReductionParameter *rp = lparam->mutable_reduction_param();
        rp->set_operation(caffe::ReductionParameter::ASUM);
        rp->set_axis(1);
        nsr = lparam->add_include();
        nsr->set_phase(caffe::TRAIN);

        lparam = CaffeCommon::add_layer(this->_net_params,"summed_difference","scaled_difference",
                                        "Loss_Scale","Scale");
        caffe::ScaleParameter *sp = lparam->mutable_scale_param();
        caffe::FillerParameter *fp = sp->mutable_filler();
        fp->set_type("constant");
        fp->set_value(0.000025);
        sp->set_axis(0);
        sp->set_bias_term(false);
        caffe::ParamSpec *ps = lparam->add_param();
        ps->set_lr_mult(0.0);
        ps->set_decay_mult(0.0);
        nsr = lparam->add_include();
        nsr->set_phase(caffe::TRAIN);

        lparam = CaffeCommon::add_layer(this->_net_params,"scaled_difference","loss",
                                        "Loss_Reduction_Batch","Reduction");
        rp = lparam->mutable_reduction_param();
        rp->set_operation(caffe::ReductionParameter::SUM);
        lparam->add_loss_weight(1.0);
        nsr = lparam->add_include();
        nsr->set_phase(caffe::TRAIN);
      }

  }

  template class NetCaffe<NetInputCaffe<CSVTSCaffeInputFileConn>,NetLayersCaffeRecurrent,NetLossCaffe>;
  // template class NetCaffe<NetInputCaffe<ImgCaffeInputFileConn>,NetLayersCaffeRecurrent,NetLossCaffe>;
  // template class NetCaffe<NetInputCaffe<CSVCaffeInputFileConn>,NetLayersCaffeRecurrent,NetLossCaffe>;
  // template class NetCaffe<NetInputCaffe<TxtCaffeInputFileConn>,NetLayersCaffeRecurrent,NetLossCaffe>;
  // template class NetCaffe<NetInputCaffe<SVMCaffeInputFileConn>,NetLayersCaffeRecurrent,NetLossCaffe>;

}
