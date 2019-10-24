#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include <map>
#include <string>

#include "src/caffe.pb.h"
#include "upgrade_proto.hpp"



bool NetNeedsUpgrade(const caffe::NetParameter& net_param) {
  return NetNeedsV0ToV1Upgrade(net_param) || NetNeedsV1ToV2Upgrade(net_param)
      || NetNeedsDataUpgrade(net_param) || NetNeedsInputUpgrade(net_param)
      || NetNeedsBatchNormUpgrade(net_param);
}

bool UpgradeNetAsNeeded(caffe::NetParameter* param) {
  bool success = true;
  if (NetNeedsV0ToV1Upgrade(*param)) {
    caffe::NetParameter original_param(*param);
    if (!UpgradeV0Net(original_param, param)) {
      success = false;
    }
  }
  // NetParameter uses old style data transformation fields; try to upgrade it.
  if (NetNeedsDataUpgrade(*param)) {
    UpgradeNetDataTransformation(param);
  }
  if (NetNeedsV1ToV2Upgrade(*param)) {
    caffe::NetParameter original_param(*param);
    if (!UpgradeV1Net(original_param, param)) {
      success = false;
    }
  }
  // NetParameter uses old style input fields; try to upgrade it.
  if (NetNeedsInputUpgrade(*param)) {
    UpgradeNetInput(param);
  }
  // NetParameter uses old style batch norm layers; try to upgrade it.
  if (NetNeedsBatchNormUpgrade(*param)) {
    UpgradeNetBatchNorm(param);
  }
  return success;
}



bool NetNeedsV0ToV1Upgrade(const caffe::NetParameter& net_param) {
  for (int i = 0; i < net_param.layers_size(); ++i) {
    if (net_param.layers(i).has_layer()) {
      return true;
    }
  }
  return false;
}

bool NetNeedsV1ToV2Upgrade(const caffe::NetParameter& net_param) {
  return net_param.layers_size() > 0;
}

bool UpgradeV0Net(const caffe::NetParameter& v0_net_param_padding_layers,
                  caffe::NetParameter* net_param) {
  // First upgrade padding layers to padded conv layers.
  caffe::NetParameter v0_net_param;
  UpgradeV0PaddingLayers(v0_net_param_padding_layers, &v0_net_param);
  // Now upgrade layer parameters.
  bool is_fully_compatible = true;
  net_param->Clear();
  if (v0_net_param.has_name()) {
    net_param->set_name(v0_net_param.name());
  }
  for (int i = 0; i < v0_net_param.layers_size(); ++i) {
    is_fully_compatible &= UpgradeV0LayerParameter(v0_net_param.layers(i),
                                                   net_param->add_layers());
  }
  for (int i = 0; i < v0_net_param.input_size(); ++i) {
    net_param->add_input(v0_net_param.input(i));
  }
  for (int i = 0; i < v0_net_param.input_dim_size(); ++i) {
    net_param->add_input_dim(v0_net_param.input_dim(i));
  }
  if (v0_net_param.has_force_backward()) {
    net_param->set_force_backward(v0_net_param.force_backward());
  }
  return is_fully_compatible;
}

void UpgradeV0PaddingLayers(const caffe::NetParameter& param,
                            caffe::NetParameter* param_upgraded_pad) {
  // Copy everything other than the layers from the original param.
  param_upgraded_pad->Clear();
  param_upgraded_pad->CopyFrom(param);
  param_upgraded_pad->clear_layers();
  // Figure out which layer each bottom blob comes from.
  std::map<std::string, int> blob_name_to_last_top_idx;
  for (int i = 0; i < param.input_size(); ++i) {
    const std::string& blob_name = param.input(i);
    blob_name_to_last_top_idx[blob_name] = -1;
  }
  for (int i = 0; i < param.layers_size(); ++i) {
    const caffe::V1LayerParameter& layer_connection = param.layers(i);
    const caffe::V0LayerParameter& layer_param = layer_connection.layer();
    // Add the layer to the new net, unless it's a padding layer.
    if (layer_param.type() != "padding") {
      param_upgraded_pad->add_layers()->CopyFrom(layer_connection);
    }
    for (int j = 0; j < layer_connection.bottom_size(); ++j) {
      const std::string& blob_name = layer_connection.bottom(j);
      if (blob_name_to_last_top_idx.find(blob_name) ==
          blob_name_to_last_top_idx.end()) {
        return;
      }
      const int top_idx = blob_name_to_last_top_idx[blob_name];
      if (top_idx == -1) {
        continue;
      }
      const caffe::V1LayerParameter& source_layer = param.layers(top_idx);
      if (source_layer.layer().type() == "padding") {
        int layer_index = param_upgraded_pad->layers_size() - 1;
        param_upgraded_pad->mutable_layers(layer_index)->mutable_layer()
            ->set_pad(source_layer.layer().pad());
        param_upgraded_pad->mutable_layers(layer_index)
            ->set_bottom(j, source_layer.bottom(0));
      }
    }
    for (int j = 0; j < layer_connection.top_size(); ++j) {
      const std::string& blob_name = layer_connection.top(j);
      blob_name_to_last_top_idx[blob_name] = i;
    }
  }
}

bool UpgradeV0LayerParameter(const caffe::V1LayerParameter& v0_layer_connection,
                             caffe::V1LayerParameter* layer_param) {
  bool is_fully_compatible = true;
  layer_param->Clear();
  for (int i = 0; i < v0_layer_connection.bottom_size(); ++i) {
    layer_param->add_bottom(v0_layer_connection.bottom(i));
  }
  for (int i = 0; i < v0_layer_connection.top_size(); ++i) {
    layer_param->add_top(v0_layer_connection.top(i));
  }
  if (v0_layer_connection.has_layer()) {
    const caffe::V0LayerParameter& v0_layer_param = v0_layer_connection.layer();
    if (v0_layer_param.has_name()) {
      layer_param->set_name(v0_layer_param.name());
    }
    const std::string& type = v0_layer_param.type();
    if (v0_layer_param.has_type()) {
      layer_param->set_type(UpgradeV0LayerType(type));
    }
    for (int i = 0; i < v0_layer_param.blobs_size(); ++i) {
      layer_param->add_blobs()->CopyFrom(v0_layer_param.blobs(i));
    }
    for (int i = 0; i < v0_layer_param.blobs_lr_size(); ++i) {
      layer_param->add_blobs_lr(v0_layer_param.blobs_lr(i));
    }
    for (int i = 0; i < v0_layer_param.weight_decay_size(); ++i) {
      layer_param->add_weight_decay(v0_layer_param.weight_decay(i));
    }
    if (v0_layer_param.has_num_output()) {
      if (type == "conv") {
        layer_param->mutable_convolution_param()->set_num_output(
            v0_layer_param.num_output());
      } else if (type == "innerproduct") {
        layer_param->mutable_inner_product_param()->set_num_output(
            v0_layer_param.num_output());
      } else {
        is_fully_compatible = false;
      }
    }
    if (v0_layer_param.has_biasterm()) {
      if (type == "conv") {
        layer_param->mutable_convolution_param()->set_bias_term(
            v0_layer_param.biasterm());
      } else if (type == "innerproduct") {
        layer_param->mutable_inner_product_param()->set_bias_term(
            v0_layer_param.biasterm());
      } else {
        is_fully_compatible = false;
      }
    }
    if (v0_layer_param.has_weight_filler()) {
      if (type == "conv") {
        layer_param->mutable_convolution_param()->
            mutable_weight_filler()->CopyFrom(v0_layer_param.weight_filler());
      } else if (type == "innerproduct") {
        layer_param->mutable_inner_product_param()->
            mutable_weight_filler()->CopyFrom(v0_layer_param.weight_filler());
      } else {
        is_fully_compatible = false;
      }
    }
    if (v0_layer_param.has_bias_filler()) {
      if (type == "conv") {
        layer_param->mutable_convolution_param()->
            mutable_bias_filler()->CopyFrom(v0_layer_param.bias_filler());
      } else if (type == "innerproduct") {
        layer_param->mutable_inner_product_param()->
            mutable_bias_filler()->CopyFrom(v0_layer_param.bias_filler());
      } else {
        is_fully_compatible = false;
      }
    }
    if (v0_layer_param.has_pad()) {
      if (type == "conv") {
        layer_param->mutable_convolution_param()->add_pad(v0_layer_param.pad());
      } else if (type == "pool") {
        layer_param->mutable_pooling_param()->set_pad(v0_layer_param.pad());
      } else {
        is_fully_compatible = false;
      }
    }
    if (v0_layer_param.has_kernelsize()) {
      if (type == "conv") {
        layer_param->mutable_convolution_param()->add_kernel_size(
            v0_layer_param.kernelsize());
      } else if (type == "pool") {
        layer_param->mutable_pooling_param()->set_kernel_size(
            v0_layer_param.kernelsize());
      } else {
        is_fully_compatible = false;
      }
    }
    if (v0_layer_param.has_group()) {
      if (type == "conv") {
        layer_param->mutable_convolution_param()->set_group(
            v0_layer_param.group());
      } else {
        is_fully_compatible = false;
      }
    }
    if (v0_layer_param.has_stride()) {
      if (type == "conv") {
        layer_param->mutable_convolution_param()->add_stride(
            v0_layer_param.stride());
      } else if (type == "pool") {
        layer_param->mutable_pooling_param()->set_stride(
            v0_layer_param.stride());
      } else {
        is_fully_compatible = false;
      }
    }
    if (v0_layer_param.has_pool()) {
      if (type == "pool") {
        caffe::V0LayerParameter_PoolMethod pool = v0_layer_param.pool();
        switch (pool) {
        case caffe::V0LayerParameter_PoolMethod_MAX:
          layer_param->mutable_pooling_param()->set_pool(
                                                         caffe::PoolingParameter_PoolMethod_MAX);
          break;
        case caffe::V0LayerParameter_PoolMethod_AVE:
          layer_param->mutable_pooling_param()->set_pool(
                                                         caffe::PoolingParameter_PoolMethod_AVE);
          break;
        case caffe::V0LayerParameter_PoolMethod_STOCHASTIC:
          layer_param->mutable_pooling_param()->set_pool(
                                                         caffe::PoolingParameter_PoolMethod_STOCHASTIC);
          break;
        default:
          is_fully_compatible = false;
        }
      } else {
        is_fully_compatible = false;
      }
    }
    if (v0_layer_param.has_dropout_ratio()) {
      if (type == "dropout") {
        layer_param->mutable_dropout_param()->set_dropout_ratio(
            v0_layer_param.dropout_ratio());
      } else {
        is_fully_compatible = false;
      }
    }
    if (v0_layer_param.has_local_size()) {
      if (type == "lrn") {
        layer_param->mutable_lrn_param()->set_local_size(
            v0_layer_param.local_size());
      } else {
        is_fully_compatible = false;
      }
    }
    if (v0_layer_param.has_alpha()) {
      if (type == "lrn") {
        layer_param->mutable_lrn_param()->set_alpha(v0_layer_param.alpha());
      } else {
        is_fully_compatible = false;
      }
    }
    if (v0_layer_param.has_beta()) {
      if (type == "lrn") {
        layer_param->mutable_lrn_param()->set_beta(v0_layer_param.beta());
      } else {
        is_fully_compatible = false;
      }
    }
    if (v0_layer_param.has_k()) {
      if (type == "lrn") {
        layer_param->mutable_lrn_param()->set_k(v0_layer_param.k());
      } else {
        is_fully_compatible = false;
      }
    }
    if (v0_layer_param.has_source()) {
      if (type == "data") {
        layer_param->mutable_data_param()->set_source(v0_layer_param.source());
      } else if (type == "hdf5_data") {
        layer_param->mutable_hdf5_data_param()->set_source(
            v0_layer_param.source());
      } else if (type == "images") {
        layer_param->mutable_image_data_param()->set_source(
            v0_layer_param.source());
      } else if (type == "window_data") {
        layer_param->mutable_window_data_param()->set_source(
            v0_layer_param.source());
      } else if (type == "infogain_loss") {
        layer_param->mutable_infogain_loss_param()->set_source(
            v0_layer_param.source());
      } else {
        is_fully_compatible = false;
      }
    }
    if (v0_layer_param.has_scale()) {
      layer_param->mutable_transform_param()->
          set_scale(v0_layer_param.scale());
    }
    if (v0_layer_param.has_meanfile()) {
      layer_param->mutable_transform_param()->
          set_mean_file(v0_layer_param.meanfile());
    }
    if (v0_layer_param.has_batchsize()) {
      if (type == "data") {
        layer_param->mutable_data_param()->set_batch_size(
            v0_layer_param.batchsize());
      } else if (type == "hdf5_data") {
        layer_param->mutable_hdf5_data_param()->set_batch_size(
            v0_layer_param.batchsize());
      } else if (type == "images") {
        layer_param->mutable_image_data_param()->set_batch_size(
            v0_layer_param.batchsize());
      } else if (type == "window_data") {
        layer_param->mutable_window_data_param()->set_batch_size(
            v0_layer_param.batchsize());
      } else {
        is_fully_compatible = false;
      }
    }
    if (v0_layer_param.has_cropsize()) {
      layer_param->mutable_transform_param()->
          set_crop_size(v0_layer_param.cropsize());
    }
    if (v0_layer_param.has_mirror()) {
      layer_param->mutable_transform_param()->
          set_mirror(v0_layer_param.mirror());
    }
    if (v0_layer_param.has_rand_skip()) {
      if (type == "data") {
        layer_param->mutable_data_param()->set_rand_skip(
            v0_layer_param.rand_skip());
      } else if (type == "images") {
        layer_param->mutable_image_data_param()->set_rand_skip(
            v0_layer_param.rand_skip());
      } else {
        is_fully_compatible = false;
      }
    }
    if (v0_layer_param.has_shuffle_images()) {
      if (type == "images") {
        layer_param->mutable_image_data_param()->set_shuffle(
            v0_layer_param.shuffle_images());
      } else {
        is_fully_compatible = false;
      }
    }
    if (v0_layer_param.has_new_height()) {
      if (type == "images") {
        layer_param->mutable_image_data_param()->set_new_height(
            v0_layer_param.new_height());
      } else {
        is_fully_compatible = false;
      }
    }
    if (v0_layer_param.has_new_width()) {
      if (type == "images") {
        layer_param->mutable_image_data_param()->set_new_width(
            v0_layer_param.new_width());
      } else {
        is_fully_compatible = false;
      }
    }
    if (v0_layer_param.has_concat_dim()) {
      if (type == "concat") {
        layer_param->mutable_concat_param()->set_concat_dim(
            v0_layer_param.concat_dim());
      } else {
        is_fully_compatible = false;
      }
    }
    if (v0_layer_param.has_det_fg_threshold()) {
      if (type == "window_data") {
        layer_param->mutable_window_data_param()->set_fg_threshold(
            v0_layer_param.det_fg_threshold());
      } else {
        is_fully_compatible = false;
      }
    }
    if (v0_layer_param.has_det_bg_threshold()) {
      if (type == "window_data") {
        layer_param->mutable_window_data_param()->set_bg_threshold(
            v0_layer_param.det_bg_threshold());
      } else {
        is_fully_compatible = false;
      }
    }
    if (v0_layer_param.has_det_fg_fraction()) {
      if (type == "window_data") {
        layer_param->mutable_window_data_param()->set_fg_fraction(
            v0_layer_param.det_fg_fraction());
      } else {
        is_fully_compatible = false;
      }
    }
    if (v0_layer_param.has_det_context_pad()) {
      if (type == "window_data") {
        layer_param->mutable_window_data_param()->set_context_pad(
            v0_layer_param.det_context_pad());
      } else {
        is_fully_compatible = false;
      }
    }
    if (v0_layer_param.has_det_crop_mode()) {
      if (type == "window_data") {
        layer_param->mutable_window_data_param()->set_crop_mode(
            v0_layer_param.det_crop_mode());
      } else {
        is_fully_compatible = false;
      }
    }
    if (v0_layer_param.has_hdf5_output_param()) {
      if (type == "hdf5_output") {
        layer_param->mutable_hdf5_output_param()->CopyFrom(
            v0_layer_param.hdf5_output_param());
      } else {
        is_fully_compatible = false;
      }
    }
  }
  return is_fully_compatible;
}

caffe::V1LayerParameter_LayerType UpgradeV0LayerType(const std::string& type) {
  if (type == "accuracy") {
    return caffe::V1LayerParameter_LayerType_ACCURACY;
  } else if (type == "bnll") {
    return caffe::V1LayerParameter_LayerType_BNLL;
  } else if (type == "concat") {
    return caffe::V1LayerParameter_LayerType_CONCAT;
  } else if (type == "conv") {
    return caffe::V1LayerParameter_LayerType_CONVOLUTION;
  } else if (type == "data") {
    return caffe::V1LayerParameter_LayerType_DATA;
  } else if (type == "dropout") {
    return caffe::V1LayerParameter_LayerType_DROPOUT;
  } else if (type == "euclidean_loss") {
    return caffe::V1LayerParameter_LayerType_EUCLIDEAN_LOSS;
  } else if (type == "flatten") {
    return caffe::V1LayerParameter_LayerType_FLATTEN;
  } else if (type == "hdf5_data") {
    return caffe::V1LayerParameter_LayerType_HDF5_DATA;
  } else if (type == "hdf5_output") {
    return caffe::V1LayerParameter_LayerType_HDF5_OUTPUT;
  } else if (type == "im2col") {
    return caffe::V1LayerParameter_LayerType_IM2COL;
  } else if (type == "images") {
    return caffe::V1LayerParameter_LayerType_IMAGE_DATA;
  } else if (type == "infogain_loss") {
    return caffe::V1LayerParameter_LayerType_INFOGAIN_LOSS;
  } else if (type == "innerproduct") {
    return caffe::V1LayerParameter_LayerType_INNER_PRODUCT;
  } else if (type == "lrn") {
    return caffe::V1LayerParameter_LayerType_LRN;
  } else if (type == "multinomial_logistic_loss") {
    return caffe::V1LayerParameter_LayerType_MULTINOMIAL_LOGISTIC_LOSS;
  } else if (type == "pool") {
    return caffe::V1LayerParameter_LayerType_POOLING;
  } else if (type == "relu") {
    return caffe::V1LayerParameter_LayerType_RELU;
  } else if (type == "sigmoid") {
    return caffe::V1LayerParameter_LayerType_SIGMOID;
  } else if (type == "softmax") {
    return caffe::V1LayerParameter_LayerType_SOFTMAX;
  } else if (type == "softmax_loss") {
    return caffe::V1LayerParameter_LayerType_SOFTMAX_LOSS;
  } else if (type == "split") {
    return caffe::V1LayerParameter_LayerType_SPLIT;
  } else if (type == "tanh") {
    return caffe::V1LayerParameter_LayerType_TANH;
  } else if (type == "window_data") {
    return caffe::V1LayerParameter_LayerType_WINDOW_DATA;
  } else {
    return caffe::V1LayerParameter_LayerType_NONE;
  }
}

bool NetNeedsDataUpgrade(const caffe::NetParameter& net_param) {
  for (int i = 0; i < net_param.layers_size(); ++i) {
    if (net_param.layers(i).type() == caffe::V1LayerParameter_LayerType_DATA) {
      caffe::DataParameter layer_param = net_param.layers(i).data_param();
      if (layer_param.has_scale()) { return true; }
      if (layer_param.has_mean_file()) { return true; }
      if (layer_param.has_crop_size()) { return true; }
      if (layer_param.has_mirror()) { return true; }
    }
    if (net_param.layers(i).type() == caffe::V1LayerParameter_LayerType_IMAGE_DATA) {
      caffe::ImageDataParameter layer_param = net_param.layers(i).image_data_param();
      if (layer_param.has_scale()) { return true; }
      if (layer_param.has_mean_file()) { return true; }
      if (layer_param.has_crop_size()) { return true; }
      if (layer_param.has_mirror()) { return true; }
    }
    if (net_param.layers(i).type() == caffe::V1LayerParameter_LayerType_WINDOW_DATA) {
      caffe::WindowDataParameter layer_param = net_param.layers(i).window_data_param();
      if (layer_param.has_scale()) { return true; }
      if (layer_param.has_mean_file()) { return true; }
      if (layer_param.has_crop_size()) { return true; }
      if (layer_param.has_mirror()) { return true; }
    }
  }
  return false;
}

#define CONVERT_LAYER_TRANSFORM_PARAM(TYPE, Name, param_name) \
  do { \
    if (net_param->layers(i).type() == caffe::V1LayerParameter_LayerType_##TYPE) { \
      Name##Parameter* layer_param = \
          net_param->mutable_layers(i)->mutable_##param_name##_param(); \
      caffe::TransformationParameter* transform_param =                 \
          net_param->mutable_layers(i)->mutable_transform_param(); \
      if (layer_param->has_scale()) { \
        transform_param->set_scale(layer_param->scale()); \
        layer_param->clear_scale(); \
      } \
      if (layer_param->has_mean_file()) { \
        transform_param->set_mean_file(layer_param->mean_file()); \
        layer_param->clear_mean_file(); \
      } \
      if (layer_param->has_crop_size()) { \
        transform_param->set_crop_size(layer_param->crop_size()); \
        layer_param->clear_crop_size(); \
      } \
      if (layer_param->has_mirror()) { \
        transform_param->set_mirror(layer_param->mirror()); \
        layer_param->clear_mirror(); \
      } \
    } \
  } while (0)

void UpgradeNetDataTransformation(caffe::NetParameter* net_param) {
  for (int i = 0; i < net_param->layers_size(); ++i) {
    CONVERT_LAYER_TRANSFORM_PARAM(DATA, caffe::Data, data);
    CONVERT_LAYER_TRANSFORM_PARAM(IMAGE_DATA, caffe::ImageData, image_data);
    CONVERT_LAYER_TRANSFORM_PARAM(WINDOW_DATA, caffe::WindowData, window_data);
  }
}

bool UpgradeV1Net(const caffe::NetParameter& v1_net_param, caffe::NetParameter* net_param) {
  if (v1_net_param.layer_size() > 0) {
    return false;
  }
  bool is_fully_compatible = true;
  net_param->CopyFrom(v1_net_param);
  net_param->clear_layers();
  net_param->clear_layer();
  for (int i = 0; i < v1_net_param.layers_size(); ++i) {
    if (!UpgradeV1LayerParameter(v1_net_param.layers(i),
                                 net_param->add_layer())) {
      is_fully_compatible = false;
    }
  }
  return is_fully_compatible;
}

bool UpgradeV1LayerParameter(const caffe::V1LayerParameter& v1_layer_param,
                             caffe::LayerParameter* layer_param) {
  layer_param->Clear();
  bool is_fully_compatible = true;
  for (int i = 0; i < v1_layer_param.bottom_size(); ++i) {
    layer_param->add_bottom(v1_layer_param.bottom(i));
  }
  for (int i = 0; i < v1_layer_param.top_size(); ++i) {
    layer_param->add_top(v1_layer_param.top(i));
  }
  if (v1_layer_param.has_name()) {
    layer_param->set_name(v1_layer_param.name());
  }
  for (int i = 0; i < v1_layer_param.include_size(); ++i) {
    layer_param->add_include()->CopyFrom(v1_layer_param.include(i));
  }
  for (int i = 0; i < v1_layer_param.exclude_size(); ++i) {
    layer_param->add_exclude()->CopyFrom(v1_layer_param.exclude(i));
  }
  if (v1_layer_param.has_type()) {
    layer_param->set_type(UpgradeV1LayerType(v1_layer_param.type()));
  }
  for (int i = 0; i < v1_layer_param.blobs_size(); ++i) {
    layer_param->add_blobs()->CopyFrom(v1_layer_param.blobs(i));
  }
  for (int i = 0; i < v1_layer_param.param_size(); ++i) {
    while (layer_param->param_size() <= i) { layer_param->add_param(); }
    layer_param->mutable_param(i)->set_name(v1_layer_param.param(i));
  }
  caffe::ParamSpec_DimCheckMode mode;
  for (int i = 0; i < v1_layer_param.blob_share_mode_size(); ++i) {
    while (layer_param->param_size() <= i) { layer_param->add_param(); }
    switch (v1_layer_param.blob_share_mode(i)) {
    case caffe::V1LayerParameter_DimCheckMode_STRICT:
      mode = caffe::ParamSpec_DimCheckMode_STRICT;
      break;
    case caffe::V1LayerParameter_DimCheckMode_PERMISSIVE:
      mode = caffe::ParamSpec_DimCheckMode_PERMISSIVE;
      break;
    default:
      break;
    }
    layer_param->mutable_param(i)->set_share_mode(mode);
  }
  for (int i = 0; i < v1_layer_param.blobs_lr_size(); ++i) {
    while (layer_param->param_size() <= i) { layer_param->add_param(); }
    layer_param->mutable_param(i)->set_lr_mult(v1_layer_param.blobs_lr(i));
  }
  for (int i = 0; i < v1_layer_param.weight_decay_size(); ++i) {
    while (layer_param->param_size() <= i) { layer_param->add_param(); }
    layer_param->mutable_param(i)->set_decay_mult(
        v1_layer_param.weight_decay(i));
  }
  for (int i = 0; i < v1_layer_param.loss_weight_size(); ++i) {
    layer_param->add_loss_weight(v1_layer_param.loss_weight(i));
  }
  if (v1_layer_param.has_accuracy_param()) {
    layer_param->mutable_accuracy_param()->CopyFrom(
        v1_layer_param.accuracy_param());
  }
  if (v1_layer_param.has_argmax_param()) {
    layer_param->mutable_argmax_param()->CopyFrom(
        v1_layer_param.argmax_param());
  }
  if (v1_layer_param.has_concat_param()) {
    layer_param->mutable_concat_param()->CopyFrom(
        v1_layer_param.concat_param());
  }
  if (v1_layer_param.has_contrastive_loss_param()) {
    layer_param->mutable_contrastive_loss_param()->CopyFrom(
        v1_layer_param.contrastive_loss_param());
  }
  if (v1_layer_param.has_convolution_param()) {
    layer_param->mutable_convolution_param()->CopyFrom(
        v1_layer_param.convolution_param());
  }
  if (v1_layer_param.has_data_param()) {
    layer_param->mutable_data_param()->CopyFrom(
        v1_layer_param.data_param());
  }
  if (v1_layer_param.has_dropout_param()) {
    layer_param->mutable_dropout_param()->CopyFrom(
        v1_layer_param.dropout_param());
  }
  if (v1_layer_param.has_dummy_data_param()) {
    layer_param->mutable_dummy_data_param()->CopyFrom(
        v1_layer_param.dummy_data_param());
  }
  if (v1_layer_param.has_eltwise_param()) {
    layer_param->mutable_eltwise_param()->CopyFrom(
        v1_layer_param.eltwise_param());
  }
  if (v1_layer_param.has_exp_param()) {
    layer_param->mutable_exp_param()->CopyFrom(
        v1_layer_param.exp_param());
  }
  if (v1_layer_param.has_hdf5_data_param()) {
    layer_param->mutable_hdf5_data_param()->CopyFrom(
        v1_layer_param.hdf5_data_param());
  }
  if (v1_layer_param.has_hdf5_output_param()) {
    layer_param->mutable_hdf5_output_param()->CopyFrom(
        v1_layer_param.hdf5_output_param());
  }
  if (v1_layer_param.has_hinge_loss_param()) {
    layer_param->mutable_hinge_loss_param()->CopyFrom(
        v1_layer_param.hinge_loss_param());
  }
  if (v1_layer_param.has_image_data_param()) {
    layer_param->mutable_image_data_param()->CopyFrom(
        v1_layer_param.image_data_param());
  }
  if (v1_layer_param.has_infogain_loss_param()) {
    layer_param->mutable_infogain_loss_param()->CopyFrom(
        v1_layer_param.infogain_loss_param());
  }
  if (v1_layer_param.has_inner_product_param()) {
    layer_param->mutable_inner_product_param()->CopyFrom(
        v1_layer_param.inner_product_param());
  }
  if (v1_layer_param.has_lrn_param()) {
    layer_param->mutable_lrn_param()->CopyFrom(
        v1_layer_param.lrn_param());
  }
  if (v1_layer_param.has_memory_data_param()) {
    layer_param->mutable_memory_data_param()->CopyFrom(
        v1_layer_param.memory_data_param());
  }
  if (v1_layer_param.has_mvn_param()) {
    layer_param->mutable_mvn_param()->CopyFrom(
        v1_layer_param.mvn_param());
  }
  if (v1_layer_param.has_pooling_param()) {
    layer_param->mutable_pooling_param()->CopyFrom(
        v1_layer_param.pooling_param());
  }
  if (v1_layer_param.has_power_param()) {
    layer_param->mutable_power_param()->CopyFrom(
        v1_layer_param.power_param());
  }
  if (v1_layer_param.has_relu_param()) {
    layer_param->mutable_relu_param()->CopyFrom(
        v1_layer_param.relu_param());
  }
  if (v1_layer_param.has_sigmoid_param()) {
    layer_param->mutable_sigmoid_param()->CopyFrom(
        v1_layer_param.sigmoid_param());
  }
  if (v1_layer_param.has_softmax_param()) {
    layer_param->mutable_softmax_param()->CopyFrom(
        v1_layer_param.softmax_param());
  }
  if (v1_layer_param.has_slice_param()) {
    layer_param->mutable_slice_param()->CopyFrom(
        v1_layer_param.slice_param());
  }
  if (v1_layer_param.has_tanh_param()) {
    layer_param->mutable_tanh_param()->CopyFrom(
        v1_layer_param.tanh_param());
  }
  if (v1_layer_param.has_threshold_param()) {
    layer_param->mutable_threshold_param()->CopyFrom(
        v1_layer_param.threshold_param());
  }
  if (v1_layer_param.has_window_data_param()) {
    layer_param->mutable_window_data_param()->CopyFrom(
        v1_layer_param.window_data_param());
  }
  if (v1_layer_param.has_transform_param()) {
    layer_param->mutable_transform_param()->CopyFrom(
        v1_layer_param.transform_param());
  }
  if (v1_layer_param.has_loss_param()) {
    layer_param->mutable_loss_param()->CopyFrom(
        v1_layer_param.loss_param());
  }
  if (v1_layer_param.has_layer()) {
    is_fully_compatible = false;
  }
  return is_fully_compatible;
}

const char* UpgradeV1LayerType(const caffe::V1LayerParameter_LayerType type) {
  switch (type) {
  case caffe::V1LayerParameter_LayerType_NONE:
    return "";
  case caffe::V1LayerParameter_LayerType_ABSVAL:
    return "AbsVal";
  case caffe::V1LayerParameter_LayerType_ACCURACY:
    return "Accuracy";
  case caffe::V1LayerParameter_LayerType_ARGMAX:
    return "ArgMax";
  case caffe::V1LayerParameter_LayerType_BNLL:
    return "BNLL";
  case caffe::V1LayerParameter_LayerType_CONCAT:
    return "Concat";
  case caffe::V1LayerParameter_LayerType_CONTRASTIVE_LOSS:
    return "ContrastiveLoss";
  case caffe::V1LayerParameter_LayerType_CONVOLUTION:
    return "Convolution";
  case caffe::V1LayerParameter_LayerType_DECONVOLUTION:
    return "Deconvolution";
  case caffe::V1LayerParameter_LayerType_DATA:
    return "Data";
  case caffe::V1LayerParameter_LayerType_DROPOUT:
    return "Dropout";
  case caffe::V1LayerParameter_LayerType_DUMMY_DATA:
    return "DummyData";
  case caffe::V1LayerParameter_LayerType_EUCLIDEAN_LOSS:
    return "EuclideanLoss";
  case caffe::V1LayerParameter_LayerType_ELTWISE:
    return "Eltwise";
  case caffe::V1LayerParameter_LayerType_EXP:
    return "Exp";
  case caffe::V1LayerParameter_LayerType_FLATTEN:
    return "Flatten";
  case caffe::V1LayerParameter_LayerType_HDF5_DATA:
    return "HDF5Data";
  case caffe::V1LayerParameter_LayerType_HDF5_OUTPUT:
    return "HDF5Output";
  case caffe::V1LayerParameter_LayerType_HINGE_LOSS:
    return "HingeLoss";
  case caffe::V1LayerParameter_LayerType_IM2COL:
    return "Im2col";
  case caffe::V1LayerParameter_LayerType_IMAGE_DATA:
    return "ImageData";
  case caffe::V1LayerParameter_LayerType_INFOGAIN_LOSS:
    return "InfogainLoss";
  case caffe::V1LayerParameter_LayerType_INNER_PRODUCT:
    return "InnerProduct";
  case caffe::V1LayerParameter_LayerType_LRN:
    return "LRN";
  case caffe::V1LayerParameter_LayerType_MEMORY_DATA:
    return "MemoryData";
  case caffe::V1LayerParameter_LayerType_MULTINOMIAL_LOGISTIC_LOSS:
    return "MultinomialLogisticLoss";
  case caffe::V1LayerParameter_LayerType_MVN:
    return "MVN";
  case caffe::V1LayerParameter_LayerType_POOLING:
    return "Pooling";
  case caffe::V1LayerParameter_LayerType_POWER:
    return "Power";
  case caffe::V1LayerParameter_LayerType_RELU:
    return "ReLU";
  case caffe::V1LayerParameter_LayerType_SIGMOID:
    return "Sigmoid";
  case caffe::V1LayerParameter_LayerType_SIGMOID_CROSS_ENTROPY_LOSS:
    return "SigmoidCrossEntropyLoss";
  case caffe::V1LayerParameter_LayerType_SILENCE:
    return "Silence";
  case caffe::V1LayerParameter_LayerType_SOFTMAX:
    return "Softmax";
  case caffe::V1LayerParameter_LayerType_SOFTMAX_LOSS:
    return "SoftmaxWithLoss";
  case caffe::V1LayerParameter_LayerType_SPLIT:
    return "Split";
  case caffe::V1LayerParameter_LayerType_SLICE:
    return "Slice";
  case caffe::V1LayerParameter_LayerType_TANH:
    return "TanH";
  case caffe::V1LayerParameter_LayerType_WINDOW_DATA:
    return "WindowData";
  case caffe::V1LayerParameter_LayerType_THRESHOLD:
    return "Threshold";
  default:
    return "";
  }
}

bool NetNeedsInputUpgrade(const caffe::NetParameter& net_param) {
  (void)net_param;
  return false; // beniz: deactivated, as breaking existing net -> net_param.input_size() > 0;
}

void UpgradeNetInput(caffe::NetParameter* net_param) {
  // Collect inputs and convert to Input layer definitions.
  // If the NetParameter holds an input alone, without shape/dim, then
  // it's a legacy caffemodel and simply stripping the input field is enough.
  bool has_shape = net_param->input_shape_size() > 0;
  bool has_dim = net_param->input_dim_size() > 0;
  if (has_shape || has_dim) {
    caffe::LayerParameter* layer_param = net_param->add_layer();
    layer_param->set_name("input");
    layer_param->set_type("Input");
    caffe::InputParameter* input_param = layer_param->mutable_input_param();
    // Convert input fields into a layer.
    for (int i = 0; i < net_param->input_size(); ++i) {
      layer_param->add_top(net_param->input(i));
      if (has_shape) {
        input_param->add_shape()->CopyFrom(net_param->input_shape(i));
      } else {
        // Turn legacy input dimensions into shape.
        caffe::BlobShape* shape = input_param->add_shape();
        int first_dim = i*4;
        int last_dim = first_dim + 4;
        for (int j = first_dim; j < last_dim; j++) {
          shape->add_dim(net_param->input_dim(j));
        }
      }
    }
    // Swap input layer to beginning of net to satisfy layer dependencies.
    for (int i = net_param->layer_size() - 1; i > 0; --i) {
      net_param->mutable_layer(i-1)->Swap(net_param->mutable_layer(i));
    }
  }
  // Clear inputs.
  net_param->clear_input();
  net_param->clear_input_shape();
  net_param->clear_input_dim();
}

bool NetNeedsBatchNormUpgrade(const caffe::NetParameter& net_param) {
  for (int i = 0; i < net_param.layer_size(); ++i) {
    // Check if BatchNorm layers declare three parameters, as required by
    // the previous BatchNorm layer definition.
    if (net_param.layer(i).type() == "BatchNorm"
        && net_param.layer(i).param_size() == 3) {
      return true;
    }
  }
  return false;
}

void UpgradeNetBatchNorm(caffe::NetParameter* net_param) {
  for (int i = 0; i < net_param->layer_size(); ++i) {
    // Check if BatchNorm layers declare three parameters, as required by
    // the previous BatchNorm layer definition.
    if (net_param->layer(i).type() == "BatchNorm"
        && net_param->layer(i).param_size() == 3) {
      net_param->mutable_layer(i)->clear_param();
    }
  }
}
