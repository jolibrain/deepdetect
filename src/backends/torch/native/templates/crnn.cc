/**
 * DeepDetect
 * Copyright (c) 2022 Jolibrain
 * Author:  Louis Jean <louis.jean@jolibrain.com>
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

#include "crnn.hpp"

#include "../../torchlib.h"

namespace dd
{
  // Basic Block

  void ResNetFeatImpl::BasicBlockImpl::init_block()
  {
    conv1 = register_module(
        "conv1",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(_inplanes, _planes, 3)
                              .stride(1)
                              .padding(1)
                              .bias(false)));
    bn1 = register_module("bn1", torch::nn::BatchNorm2d(_planes));
    conv2 = register_module(
        "conv2",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(_planes, _planes, 3)
                              .stride(1)
                              .padding(1)
                              .bias(false)));
    bn2 = register_module("bn2", torch::nn::BatchNorm2d(_planes));

    if (_downsample)
      register_module("downsample", _downsample);
  }

  torch::Tensor ResNetFeatImpl::BasicBlockImpl::forward(torch::Tensor x)
  {
    torch::Tensor residual = x;
    torch::Tensor out = conv1->forward(x);
    out = bn1->forward(out).relu_();
    out = conv2->forward(out);
    out = bn2->forward(out);

    if (_downsample)
      {
        residual = _downsample->forward(x);
      }

    out += residual;
    out = torch::relu(out);
    return out;
  }

  // Bottleneck

  void ResNetFeatImpl::BottleneckImpl::init_block()
  {
    int base_width = 64;
    int groups = 1;
    int expansion = 4;
    int width = _planes * (base_width / 64) * groups;

    conv1 = register_module(
        "conv1",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(_inplanes, width, 1)
                              .stride(1)
                              .padding(1)
                              .bias(false)));
    bn1 = register_module("bn1", torch::nn::BatchNorm2d(width));
    conv2 = register_module(
        "conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(width, width, 3)
                                       .stride(1)
                                       .padding(1)
                                       .bias(false)));
    bn2 = register_module("bn2", torch::nn::BatchNorm2d(width));
    conv3 = register_module(
        "conv3", torch::nn::Conv2d(
                     torch::nn::Conv2dOptions(width, _planes * expansion, 1)
                         .stride(1)
                         .padding(1)
                         .bias(false)));
    bn3 = register_module("bn3", torch::nn::BatchNorm2d(_planes));

    if (_downsample)
      register_module("downsample", _downsample);
  }

  torch::Tensor ResNetFeatImpl::BottleneckImpl::forward(torch::Tensor x)
  {
    torch::Tensor residual = x;
    torch::Tensor out = conv1->forward(x);
    out = bn1->forward(out).relu_();
    out = conv2->forward(out);
    out = bn2->forward(out).relu_();
    out = conv3->forward(out);
    out = bn3->forward(out);

    if (_downsample)
      residual = _downsample->forward(x);

    out += residual;
    out = torch::relu(out);
    return out;
  }

  // ResNet

  torch::nn::Sequential ResNetFeatImpl::make_layer(int &inplanes, int planes,
                                                   int layer_count)
  {
    torch::nn::Sequential layers;

    int expansion = _bottleneck ? 4 : 1;
    torch::nn::Sequential downsample = nullptr;
    // if stride != 1 ...
    if (inplanes != planes * expansion)
      {
        downsample = torch::nn::Sequential(
            torch::nn::Conv2d(
                torch::nn::Conv2dOptions(inplanes, planes * expansion, 1)
                    .stride(1)
                    .bias(false)),
            torch::nn::BatchNorm2d(planes * expansion));
      }

    // first block
    if (_bottleneck)
      layers->push_back(Bottleneck());
    else
      layers->push_back(BasicBlock(inplanes, planes, 1, downsample));

    inplanes = planes * expansion;

    for (int i = 1; i < layer_count; ++i)
      {
        if (_bottleneck)
          layers->push_back(Bottleneck());
        else
          layers->push_back(BasicBlock(inplanes, planes, 1));
      }

    layers->push_back(
        torch::nn::Conv2d(torch::nn::Conv2dOptions(planes, planes, 3)
                              .stride(1)
                              .padding(1)
                              .bias(false)));
    layers->push_back(torch::nn::BatchNorm2d(planes));

    return layers;
  }

  void ResNetFeatImpl::init_blocks()
  {
    std::vector<int> output_channel{ _output_channel / 4, _output_channel / 2,
                                     _output_channel, _output_channel };

    int inplanes = _output_channel / 8;
    conv0_1 = register_module(
        "conv0_1",
        torch::nn::Conv2d(
            torch::nn::Conv2dOptions(_input_channel, _output_channel / 16, 3)
                .stride(1)
                .padding(1)
                .bias(false)));
    bn0_1 = register_module("bn0_1",
                            torch::nn::BatchNorm2d(_output_channel / 16));
    conv0_2 = register_module(
        "conv0_2", torch::nn::Conv2d(torch::nn::Conv2dOptions(
                                         _output_channel / 16, inplanes, 3)
                                         .stride(1)
                                         .padding(1)
                                         .bias(false)));
    bn0_2 = register_module("bn0_2", torch::nn::BatchNorm2d(inplanes));

    _maxpool1 = torch::nn::MaxPool2d(
        torch::nn::MaxPool2dOptions(2).stride(2).padding(0));
    layer1 = make_layer(inplanes, output_channel[0], _layer_count[0]);
    register_module("layer1", layer1);
    setup_conv3x3(conv1, bn1, output_channel[0]);
    register_module("conv1", conv1);
    register_module("bn1", bn1);

    _maxpool2 = torch::nn::MaxPool2d(
        torch::nn::MaxPool2dOptions(2).stride(2).padding(0));
    layer2 = make_layer(inplanes, output_channel[1], _layer_count[1]);
    register_module("layer2", layer2);
    setup_conv3x3(conv2, bn2, output_channel[1]);
    register_module("conv2", conv2);
    register_module("bn2", bn2);

    _maxpool3 = torch::nn::MaxPool2d(
        torch::nn::MaxPool2dOptions(2).stride({ 2, 1 }).padding({ 0, 1 }));
    layer3 = make_layer(inplanes, output_channel[2], _layer_count[2]);
    register_module("layer3", layer3);
    setup_conv3x3(conv3, bn3, output_channel[2]);
    register_module("conv3", conv3);
    register_module("bn3", bn3);

    layer4 = make_layer(inplanes, output_channel[3], _layer_count[3]);
    register_module("layer4", layer4);
    conv4_1 = register_module(
        "conv4_1",
        torch::nn::Conv2d(
            torch::nn::Conv2dOptions(output_channel[3], output_channel[3], 2)
                .stride({ 2, 1 })
                .padding({ 0, 1 })
                .bias(false)));
    bn4_1
        = register_module("bn4_1", torch::nn::BatchNorm2d(output_channel[3]));
    conv4_2 = register_module(
        "conv4_2",
        torch::nn::Conv2d(
            torch::nn::Conv2dOptions(output_channel[3], output_channel[3], 2)
                .stride(1)
                .padding(0)
                .bias(false)));
    bn4_2
        = register_module("bn4_2", torch::nn::BatchNorm2d(output_channel[3]));
  }

  torch::Tensor ResNetFeatImpl::forward(torch::Tensor x)
  {
    x = conv0_1->forward(x);
    x = bn0_1->forward(x);
    x = conv0_2->forward(x);
    x = bn0_2->forward(x).relu_();

    x = _maxpool1->forward(x);
    x = layer1->forward(x);
    x = conv1->forward(x);
    x = bn1->forward(x).relu_();

    x = _maxpool2->forward(x);
    x = layer2->forward(x);
    x = conv2->forward(x);
    x = bn2->forward(x).relu_();

    x = _maxpool3->forward(x);
    x = layer3->forward(x);
    x = conv3->forward(x);
    x = bn3->forward(x).relu_();

    x = layer4->forward(x);
    x = conv4_1->forward(x);
    x = bn4_1->forward(x).relu_();
    x = conv4_2->forward(x);
    x = bn4_2->forward(x).relu_();

    return x;
  }

  // CRNN

  void CRNN::get_params(const ImgTorchInputFileConn &inputc,
                        const APIData &ad_params)
  {
    if (ad_params.has("timesteps"))
      _timesteps = ad_params.get("timesteps").get<int>();
    if (ad_params.has("hidden_size"))
      _hidden_size = ad_params.get("hidden_size").get<int>();
    if (ad_params.has("num_layers"))
      _num_layers = ad_params.get("num_layers").get<int>();
    if (ad_params.has("bidirectional"))
      _bidirectional = ad_params.get("bidirectional").get<bool>();
    if (ad_params.has("backbone"))
      _backbone_template = ad_params.get("backbone").get<std::string>();

    if (inputc._alphabet_size > 0)
      _output_size = inputc._alphabet_size;

    _img_width = inputc.width();
    _img_height = inputc.height();
    _input_channels = inputc._bw ? 1 : 3;
  }

  void CRNN::init()
  {
    // backbone
    if (_backbone)
      {
        unregister_module("backbone");
        _backbone = nullptr;
      }

    // if (_backbone_template == "resnet")
    std::vector<int> layer_count{ 1, 2, 5, 3 };
    int output_channel = 512;
    _backbone
        = register_module("backbone", ResNetFeat(layer_count, _input_channels,
                                                 output_channel, false));

    if (_timesteps > 0)
      {
        at::Tensor dummy_img = torch::zeros({ 1, 3, _img_height, _img_width });
        at::Tensor dummy = _backbone(dummy_img).reshape({ 1, -1, _timesteps });
        output_channel = dummy.size(1);
        // XXX should use logger
        std::cout << "LSTM input shape: " << dummy.sizes() << std::endl;
      }

    // LSTM
    if (_lstm)
      {
        unregister_module("lstm");
        _lstm = nullptr;
      }

    _lstm = register_module(
        "lstm",
        torch::nn::LSTM(torch::nn::LSTMOptions(output_channel, _hidden_size)
                            .num_layers(_num_layers)));

    // Projection
    if (_proj)
      {
        unregister_module("proj");
        _proj = nullptr;
      }

    int proj_size = _output_size;
    int d = _bidirectional ? 2 : 1;
    _proj = register_module("proj",
                            torch::nn::Linear(d * _hidden_size, proj_size));
  }

  void CRNN::set_output_size(int output_size)
  {
    if (_output_size != output_size)
      {
        _output_size = output_size;
        init();
      }
  }

  torch::Tensor CRNN::forward(torch::Tensor x)
  {
    torch::Tensor feats = _backbone->forward(x);

    // Input: feature map from resnet
    if (_timesteps > 0)
      {
        feats = feats.reshape({ feats.size(0), -1, _timesteps });
      }
    else
      {
        feats = feats.squeeze(2);
      }
    feats = feats.permute({ 2, 0, 1 });

    std::tuple<at::Tensor, std::tuple<at::Tensor, at::Tensor>> outputs
        = _lstm->forward(feats);
    auto out = _proj->forward(std::get<0>(outputs));

    return out;
  }

  torch::Tensor CRNN::loss(std::string loss, torch::Tensor input,
                           torch::Tensor output, torch::Tensor target)
  {
    (void)loss;
    (void)input;
    (void)output;
    (void)target;
    throw MLLibInternalException("CRNN::loss not implemented");
  }
}
