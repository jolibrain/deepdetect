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

#ifndef CRNN_H
#define CRNN_H

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "torch/torch.h"
#pragma GCC diagnostic pop
#include "../../torchinputconns.h"
#include "../native_net.h"

namespace dd
{
  // Resnet backbone from paper
  // Focusing Attention: Towards Accurate Text Recognition in Natural Images
  // https://arxiv.org/abs/1709.02054
  //
  // Implementation by ClovaAI
  // https://github.com/clovaai/deep-text-recognition-benchmark

  class ResNetFeatImpl : public torch::nn::Cloneable<ResNetFeatImpl>
  {
  public:
    class BasicBlockImpl : public torch::nn::Cloneable<BasicBlockImpl>
    {
    public:
      BasicBlockImpl(int inplanes, int planes, int stride,
                     torch::nn::Sequential downsample = nullptr)
          : _inplanes(inplanes), _planes(planes), _stride(stride),
            _downsample(downsample)
      {
        init_block();
      }

      void reset() override
      {
        init_block();
      }

      void init_block();

      torch::Tensor forward(torch::Tensor x);

    protected:
      int _inplanes = 0;
      int _planes = 0;
      // XXX Unused?
      int _stride = 1;

      torch::nn::Conv2d conv1{ nullptr }, conv2{ nullptr };
      torch::nn::BatchNorm2d bn1{ nullptr }, bn2{ nullptr };
      torch::nn::Sequential _downsample{ nullptr };
    };
    typedef torch::nn::ModuleHolder<BasicBlockImpl> BasicBlock;

    class BottleneckImpl : public torch::nn::Cloneable<BottleneckImpl>
    {
    public:
      void reset() override
      {
        init_block();
      }

      void init_block();

      torch::Tensor forward(torch::Tensor x);

    protected:
      int _inplanes = 0;
      int _planes = 0;
      int _stride = 1;

      torch::nn::Conv2d conv1{ nullptr }, conv2{ nullptr }, conv3{ nullptr };
      torch::nn::BatchNorm2d bn1{ nullptr }, bn2{ nullptr }, bn3{ nullptr };
      torch::nn::Sequential _downsample{ nullptr };
    };
    typedef torch::nn::ModuleHolder<BottleneckImpl> Bottleneck;

    ResNetFeatImpl(std::vector<int> layer_count, int input_channel,
                   int output_channel, bool bottleneck = false)
        : _layer_count(layer_count), _input_channel(input_channel),
          _output_channel(output_channel), _bottleneck(bottleneck)
    {
      init_blocks();
    }

    void reset() override
    {
      init_blocks();
    }

    void init_blocks();

    torch::Tensor forward(torch::Tensor x);

  protected:
    std::vector<int> _layer_count;

    int _input_channel;
    int _output_channel;
    bool _bottleneck;

    torch::nn::MaxPool2d _maxpool1{ nullptr }, _maxpool2{ nullptr },
        _maxpool3{ nullptr };
    torch::nn::Conv2d conv0_1{ nullptr }, conv0_2{ nullptr }, conv1{ nullptr },
        conv2{ nullptr }, conv3{ nullptr }, conv4_1{ nullptr },
        conv4_2{ nullptr };
    torch::nn::BatchNorm2d bn0_1{ nullptr }, bn0_2{ nullptr }, bn1{ nullptr },
        bn2{ nullptr }, bn3{ nullptr }, bn4_1{ nullptr }, bn4_2{ nullptr };
    torch::nn::Sequential layer1{ nullptr }, layer2{ nullptr },
        layer3{ nullptr }, layer4{ nullptr };

    torch::nn::Sequential make_layer(int &inplanes, int planes,
                                     int layer_count);

    void setup_conv3x3(torch::nn::Conv2d &conv, torch::nn::BatchNorm2d &bn,
                       int output_channel)
    {
      conv = torch::nn::Conv2d(
          torch::nn::Conv2dOptions(output_channel, output_channel, 3)
              .stride(1)
              .padding(1)
              .bias(false));
      bn = torch::nn::BatchNorm2d(output_channel);
    }
  };

  typedef torch::nn::ModuleHolder<ResNetFeatImpl> ResNetFeat;

  class CRNN : public NativeModuleImpl<CRNN>
  {
  public:
    std::string _backbone_template = "resnet18";
    int _num_layers = 3;
    int _hidden_size = 256;
    bool _bidirectional = false;
    int _timesteps = -1;

    // dataset dependent variables
    int _output_size;
    int _input_channels = 3;
    int _img_width, _img_height;

    ResNetFeat _backbone = nullptr;
    torch::nn::LSTM _lstm = nullptr;
    torch::nn::Linear _proj = nullptr;

  public:
    CRNN(const std::string &backbone = "resnet", int num_layers = 3,
         int hidden_size = 0, bool bidirectional = false, int timesteps = -1,
         int output_size = 2, int input_channels = 3, int img_width = 0,
         int img_height = 0)
        : _backbone_template(backbone), _num_layers(num_layers),
          _hidden_size(hidden_size), _bidirectional(bidirectional),
          _timesteps(timesteps), _output_size(output_size),
          _input_channels(input_channels), _img_width(img_width),
          _img_height(img_height)
    {
      init();
    }

    CRNN(const ImgTorchInputFileConn &inputc, const APIData &ad_params)
    {
      get_params(inputc, ad_params);
      init();
    }

    CRNN(const CRNN &other)
        : torch::nn::Module(other),
          _backbone_template(other._backbone_template),
          _num_layers(other._num_layers), _hidden_size(other._hidden_size),
          _bidirectional(other._bidirectional), _timesteps(other._timesteps),
          _output_size(other._output_size),
          _input_channels(other._input_channels), _img_width(other._img_width),
          _img_height(other._img_height)
    {
      init();
    }

    CRNN &operator=(const CRNN &other)
    {
      _backbone_template = other._backbone_template;
      _num_layers = other._num_layers;
      _hidden_size = other._hidden_size;
      _bidirectional = other._bidirectional;
      _timesteps = other._timesteps;
      _output_size = other._output_size;
      _input_channels = other._input_channels;
      _img_width = other._img_width;
      _img_height = other._img_height;
      init();
      return *this;
    }

    void init();

    void get_params(const ImgTorchInputFileConn &inputc,
                    const APIData &ad_params);

    void set_output_size(int output_size);

    torch::Tensor forward(torch::Tensor x) override;

    void reset() override
    {
      init();
    }

    torch::Tensor extract(torch::Tensor x, std::string extract_layer) override
    {
      (void)x;
      (void)extract_layer;
      return torch::Tensor();
    }

    bool extractable(std::string extract_layer) const override
    {
      (void)extract_layer;
      return false;
    }

    std::vector<std::string> extractable_layers() const override
    {
      return std::vector<std::string>();
    }

    torch::Tensor cleanup_output(torch::Tensor output) override
    {
      return output;
    }

    torch::Tensor loss(std::string loss, torch::Tensor input,
                       torch::Tensor output, torch::Tensor target) override;
  };
}

#endif // CRNN_H
