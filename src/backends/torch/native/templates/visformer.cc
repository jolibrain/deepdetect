/**
 * DeepDetect
 * Copyright (c) 2021 Jolibrain
 * Author:  Emmanuel Benazera <emmanuel.benazera@jolibrain.com>
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

#include "visformer.h"
#include <iostream>

namespace dd
{

  /*-- MLPImpl --*/
  void Visformer::MLPImpl::init_block()
  {
    if (!_output_dim)
      _output_dim = _input_dim;
    if (!_hidden_dim)
      _hidden_dim = _input_dim;

    if (_spatial_conv)
      {
        if (_group < 2)
          _hidden_dim = std::floor(_input_dim * 5 / 6);
        else
          _hidden_dim = _input_dim * 2;
      }

    _convc1 = register_module(
        "convc1",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(_input_dim, _hidden_dim, 1)
                              .stride(1)
                              .padding(0)
                              .bias(false)));
    if (_spatial_conv)
      _convc2 = register_module(
          "convc2", torch::nn::Conv2d(
                        torch::nn::Conv2dOptions(_hidden_dim, _hidden_dim, 3)
                            .stride(1)
                            .padding(1)
                            .groups(_group)
                            .bias(false)));
    _convc3 = register_module(
        "convc3",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(_hidden_dim, _output_dim, 1)
                              .stride(1)
                              .padding(0)
                              .bias(false)));
    _drop1 = register_module(
        "drop", torch::nn::Dropout(torch::nn::DropoutOptions(_drop)));
  }

  torch::Tensor Visformer::MLPImpl::forward(torch::Tensor x)
  {
    x = _convc1(x);
    x = torch::gelu(x);
    x = _drop1(x);
    if (_spatial_conv)
      {
        x = _convc2(x);
        x = torch::gelu(x);
      }
    x = _convc3(x);
    x = _drop1(x);
    return x;
  }

  /*-- AttentionImpl --*/
  void Visformer::AttentionImpl::init_block()
  {
    _head_dim = std::floor(_dim / _num_heads);
    if (_qk_scale > 0.0)
      _scale = _qk_scale;
    else
      _scale = std::pow(_head_dim, -0.5);

    _qkv = register_module(
        "qkv", torch::nn::Conv2d(torch::nn::Conv2dOptions(
                                     _dim, _head_dim * _num_heads * 3, 1)
                                     .stride(1)
                                     .padding(0)
                                     .bias(_qkv_bias)));
    _attn_drop = register_module(
        "attn_drop",
        torch::nn::Dropout(torch::nn::DropoutOptions(_attn_drop_val)));
    _proj = register_module(
        "proj", torch::nn::Conv2d(
                    torch::nn::Conv2dOptions(_head_dim * _num_heads, _dim, 1)
                        .stride(1)
                        .padding(0)
                        .bias(false)));
    _proj_drop = register_module(
        "proj_drop",
        torch::nn::Dropout(torch::nn::DropoutOptions(_proj_drop_val)));
  }

  torch::Tensor Visformer::AttentionImpl::forward(torch::Tensor x)
  {
    long int B = x.size(0);
    long int C = x.size(2);
    long int W = x.size(3);
    auto qkv = _qkv(x);
    qkv = qkv.reshape({ B, 3, _num_heads, _head_dim, C * W })
              .permute({ 1, 0, 2, 4, 3 });
    auto q = qkv[0];
    auto k = qkv[1];
    auto v = qkv[2];

    auto attn = q.matmul(k.transpose(-2, -1)) * _scale;

    attn = torch::softmax(attn, -1);
    attn = _attn_drop(attn);

    x = attn.matmul(v);
    x = x.permute({ 0, 1, 2, 3 }).reshape({ B, _num_heads * _head_dim, C, W });
    x = _proj(x);
    x = _proj_drop(x);

    return x;
  }

  /*-- BlockImpl --*/
  void Visformer::BlockImpl::init_block()
  {
    if (!_attn_disabled)
      {
        _norm1 = register_module(
            "norm1",
            torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions({ _dim })));
        _attn = register_module(
            "attn", Attention(_dim, _num_heads, _head_dim_ratio, _qkv_bias,
                              _qk_scale, _attn_drop_val, _drop_val));
      }
    _norm2 = register_module(
        "norm2",
        torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions({ _dim })));

    unsigned int mlp_hidden_dim = static_cast<int>(_dim * _mlp_ratio);
    _mlp = register_module(
        "mlp", MLP(_dim, mlp_hidden_dim, 0, _drop_val, _group, _spatial_conv));
  }

  torch::Tensor Visformer::BlockImpl::forward(torch::Tensor x)
  {
    if (!_attn_disabled)
      x = x + _attn(_norm1(x));
    x = x + _mlp(_norm2(x));
    return x;
  }

  /*-- PatchEmbedImpl --*/
  void Visformer::PatchEmbedImpl::init_block(const int &img_size,
                                             const int &patch_size)
  {
    _img_size = std::make_pair<unsigned int, unsigned int>(img_size, img_size);
    _patch_size
        = std::make_pair<unsigned int, unsigned int>(patch_size, patch_size);
    _num_patches = std::floor(_img_size.second / _patch_size.second)
                   * std::floor(_img_size.first / _patch_size.first);

    _proj = register_module(
        "proj", torch::nn::Conv2d(
                    torch::nn::Conv2dOptions(_in_chans, _embed_dim, patch_size)
                        .stride(patch_size)));
    _norm_embed
        = register_module("norm_embed", torch::nn::BatchNorm2d(_embed_dim));
  }

  torch::Tensor Visformer::PatchEmbedImpl::forward(torch::Tensor x)
  {
    x = _proj(x);
    x = _norm_embed(x);
    return x;
  }

  /*-- Visformer --*/
  void Visformer::init_block()
  {
    const unsigned int stage_num1 = 7;
    const unsigned int stage_num2 = 4;
    const unsigned int stage_num3 = 4;
    _depth = stage_num1 + stage_num2 + stage_num3;

    // stage 1
    _stem = register_module(
        "stem", torch::nn::Sequential(
                    torch::nn::Conv2d(torch::nn::Conv2dOptions(3, _in_chans, 7)
                                          .stride(2)
                                          .padding(3)
                                          .bias(false)),
                    torch::nn::BatchNorm2d(_in_chans), torch::nn::ReLU(true)));

    int n_img_size = std::floor(_img_size / 2);

    int p1_out_dim = std::floor(_embed_dim / 2);
    _patch_embed1 = register_module(
        "patch_embed1", PatchEmbed(n_img_size, 4, _in_chans, p1_out_dim));
    n_img_size = std::floor(n_img_size / 4);

    int half_embed_dim = std::floor(_embed_dim / 2);
    _pos_embed1 = register_parameter(
        "pos_embed1",
        torch::randn({ 1, half_embed_dim, n_img_size, n_img_size }));
    _pos_drop = register_module(
        "pos_drop", torch::nn::Dropout(torch::nn::DropoutOptions(_drop_rate)));

    for (unsigned int d = 0; d < stage_num1; ++d)
      {
        _stage1_blocks->push_back(Block(
            std::floor(_embed_dim / 2), _num_heads, 0.5 /* head_dim_ratio */,
            _mlp_ratio, _qkv_bias, _qk_scale, _drop_rate, _attn_drop_rate,
            _group, _attn_stage[0] == '0' /* attn_disabled */,
            _spatial_conv[0] == '1' /* spatial_conv */));
      }
    register_module("blocks1", _stage1_blocks);

    // stage 2
    _patch_embed2 = register_module(
        "patch_embed2",
        PatchEmbed(n_img_size, 2, std::floor(_embed_dim / 2), _embed_dim));
    n_img_size = std::floor(n_img_size / 2);
    _pos_embed2 = register_parameter(
        "pos_embed2", torch::randn({ 1, _embed_dim, n_img_size, n_img_size }));
    for (unsigned int d = stage_num1; d < stage_num1 + stage_num2; ++d)
      {
        _stage2_blocks->push_back(
            Block(_embed_dim, _num_heads, 1.0 /* head_dim_ratio */, _mlp_ratio,
                  _qkv_bias, _qk_scale, _drop_rate, _attn_drop_rate, _group,
                  _attn_stage[1] == '0', _spatial_conv[1] == '1'));
      }
    register_module("blocks2", _stage2_blocks);

    // stage 3
    int p3_out_dim = _embed_dim * 2;
    _patch_embed3 = register_module(
        "patch_embed3", PatchEmbed(n_img_size, 2, _embed_dim, p3_out_dim));
    n_img_size = std::floor(n_img_size / 2);
    _pos_embed3 = register_parameter(
        "pos_embed3", torch::randn({ 1, p3_out_dim, n_img_size, n_img_size }));
    for (unsigned int d = stage_num2; d < _depth; ++d)
      {
        _stage3_blocks->push_back(
            Block(p3_out_dim, _num_heads, 1.0 /* head_dim_ratio */, _mlp_ratio,
                  _qkv_bias, _qk_scale, _drop_rate, _attn_drop_rate, _group,
                  _attn_stage[2] == '0', _spatial_conv[2] == '1'));
      }
    register_module("blocks3", _stage3_blocks);

    // head
    _global_pooling = torch::nn::AdaptiveAvgPool2d(1);
    _norm = register_module(
        "norm", torch::nn::BatchNorm2d(
                    torch::nn::BatchNorm2dOptions({ _embed_dim * 2 })));
    _head
        = register_module("head", torch::nn::Linear(p3_out_dim, _num_classes));
  }

  torch::Tensor Visformer::forward(torch::Tensor x)
  {
    x = _stem->forward(x);

    // stage 1
    x = _patch_embed1(x);
    x = x + _pos_embed1;
    x = _pos_drop(x);

    for (const auto &blk : *_stage1_blocks)
      {
        x = blk->as<Block>()->forward(x);
      }

    // stage 2
    x = _patch_embed2(x);
    x = x + _pos_embed2;
    x = _pos_drop(x);
    for (const auto &blk : *_stage2_blocks)
      {
        x = blk->as<Block>()->forward(x);
      }

    // stage 3
    x = _patch_embed3(x);
    x = x + _pos_embed3;
    x = _pos_drop(x);
    for (const auto &blk : *_stage3_blocks)
      {
        x = blk->as<Block>()->forward(x);
      }

    // head
    x = _norm(x);
    x = _global_pooling(x);
    int view_size = x.size(0);
    x = _head(x.view({ view_size, -1 }));

    return x;
  }

  void
  Visformer::get_params_and_init_block(const ImgTorchInputFileConn &inputc,
                                       const APIData &ad_params)
  {
    _img_size = inputc.width();
    _patch_size = 16;
    _in_chans = 32;
    _embed_dim = 192;
    _num_heads = 12;
    _mlp_ratio = 4.0;
    _qkv_bias = false;
    _qk_scale = -1.0;
    _drop_rate = 0.0;
    _attn_drop_rate = 0.0;

    _num_classes = 2;
    if (ad_params.has("nclasses"))
      _num_classes = ad_params.get("nclasses").get<int>();

    if (ad_params.has("dropout"))
      _drop_rate = _attn_drop_rate = ad_params.get("dropout").get<double>();

    std::string visformer_flavor = "visformer_tiny";
    if (ad_params.has("visformer_tiny"))
      visformer_flavor = ad_params.get("visformer_tiny").get<std::string>();

    if (visformer_flavor == "visformer_tiny")
      {
        _embed_dim = 192;
        _in_chans = 16;
        //_depth = 16; // XXX: unused, 7, 4, 4 aka the three stages
        _num_heads = 3;
        _mlp_ratio = 4.0;
        _group = 8;
        _attn_stage = "011";
        _spatial_conv = "100";
      }
    else if (visformer_flavor == "visformer_small")
      {
        _embed_dim = 384;
        _num_heads = 6;
        _mlp_ratio = 4.0;
        _group = 8;
        _attn_stage = "011";
        _spatial_conv = "100";
      }
    else
      {
        throw MLLibBadParamException("unknown Visformer flavor "
                                     + visformer_flavor);
      }

    init_block();
  }
}
