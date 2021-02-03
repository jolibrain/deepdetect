/**
 * DeepDetect
 * Copyright (c) 2020 Jolibrain
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

#include "vit.h"
#include <iostream>

namespace dd
{
  /*-- MLPImpl --*/
  void ViT::MLPImpl::init_block()
  {
    if (!_output_dim)
      _output_dim = _input_dim;
    if (!_hidden_dim)
      _hidden_dim = _input_dim;

    _fc1 = register_module("fc1", torch::nn::Linear(_input_dim, _hidden_dim));
    _fc2 = register_module("fc2", torch::nn::Linear(_hidden_dim, _output_dim));
    _drop1 = register_module(
        "drop", torch::nn::Dropout(torch::nn::DropoutOptions(_drop)));
  }

  torch::Tensor ViT::MLPImpl::forward(torch::Tensor x)
  {
    x = _fc1(x);
    x = torch::gelu(x);
    x = _drop1(x);
    x = _fc2(x);
    x = _drop1(x);
    return x;
  }

  /*-- AttentionImpl --*/
  void ViT::AttentionImpl::init_block()
  {
    _head_dim = std::floor(_dim / _num_heads);
    if (_qk_scale > 0.0)
      _scale = _qk_scale;
    else
      _scale = std::pow(_head_dim, -0.5);

    _qkv = register_module(
        "qkv", torch::nn::Linear(
                   torch::nn::LinearOptions(_dim, _dim * 3).bias(_qkv_bias)));
    _attn_drop = register_module(
        "attn_drop",
        torch::nn::Dropout(torch::nn::DropoutOptions(_attn_drop_val)));
    _proj = register_module("proj", torch::nn::Linear(_dim, _dim));
    _proj_drop = register_module(
        "proj_drop",
        torch::nn::Dropout(torch::nn::DropoutOptions(_proj_drop_val)));
  }

  torch::Tensor ViT::AttentionImpl::residual_mha(torch::Tensor x,
                                                 torch::Tensor &prev)
  {
    if (prev.defined())
      x = x + prev;
    prev = x;
    return x;
  }

  torch::Tensor ViT::AttentionImpl::forward(torch::Tensor x,
                                            torch::Tensor &prev)
  {
    long int B = x.size(0);
    long int N = x.size(1);
    long int C = x.size(2);
    long int C2 = std::floor(C / _num_heads);
    auto qkv = _qkv(x)
                   .reshape({ B, N, 3, _num_heads, C2 })
                   .permute({ 2, 0, 3, 1, 4 });
    auto q = qkv[0];
    auto k = qkv[1];
    auto v = qkv[2];

    auto attn = q.matmul(k.transpose(-2, -1)) * _scale;

    // if realformer, residual
    if (_realformer)
      attn = residual_mha(attn, prev);

    attn = torch::softmax(attn, -1);
    attn = _attn_drop(attn);

    x = attn.matmul(v).transpose(1, 2).reshape({ B, N, C });
    x = _proj(x);
    x = _proj_drop(x);

    return x;
  }

  /*-- BlockImpl --*/
  void ViT::BlockImpl::init_block(const double &mlp_ratio,
                                  const bool &qkv_bias, const double &qk_scale,
                                  const double &drop_val,
                                  const double &attn_drop_val,
                                  const bool &realformer)
  {
    _norm1 = register_module(
        "norm1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({ _dim })));

    _attn = register_module("attn",
                            Attention(_dim, _num_heads, qkv_bias, qk_scale,
                                      attn_drop_val, drop_val, realformer));
    _norm2 = register_module(
        "norm2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({ _dim })));

    unsigned int mlp_hidden_dim = static_cast<int>(_dim * mlp_ratio);
    _mlp = register_module("mlp",
                           MLP(_dim, mlp_hidden_dim, 0, "gelu", drop_val));
  }

  torch::Tensor ViT::BlockImpl::forward(torch::Tensor x, torch::Tensor &prev)
  {
    x = x + _attn(_norm1(x), prev);
    x = x + _mlp(_norm2(x));
    return x;
  }

  /*-- PatchEmbedImpl --*/
  void ViT::PatchEmbedImpl::init_block(const int &img_size,
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
  }

  torch::Tensor ViT::PatchEmbedImpl::forward(torch::Tensor x)
  {
    x = _proj(x).flatten(2).transpose(1, 2);
    return x;
  }

  /*-- ViTImpl --*/
  void ViT::get_params_and_init_block(const ImgTorchInputFileConn &inputc,
                                      const APIData &ad_params)
  {
    _patch_size = 16;
    _in_chans = inputc._bw ? 1 : 3;
    _embed_dim = 768;
    _num_heads = 12;
    _mlp_ratio = 4.0;
    _qkv_bias = false;
    _qk_scale = -1.0;
    _drop_rate = 0.0;
    _attn_drop_rate = 0.0;

    _img_size = inputc.width();
    _num_classes = 2;
    if (ad_params.has("nclasses"))
      _num_classes = ad_params.get("nclasses").get<int>();
    _depth = 12;

    if (ad_params.has("dropout"))
      _drop_rate = _attn_drop_rate = ad_params.get("dropout").get<double>();

    if (ad_params.has("realformer"))
      _realformer = ad_params.get("realformer").get<bool>();

    std::string vit_flavor = "vit_base_patch16";
    if (ad_params.has("vit_flavor"))
      vit_flavor = ad_params.get("vit_flavor").get<std::string>();

    if (vit_flavor == "vit_base_patch16")
      {
        _qkv_bias = true;
      }
    else if (vit_flavor == "vit_base_patch32")
      {
        _patch_size = 32;
        _qkv_bias = true;
      }
    else if (vit_flavor == "vit_large_patch16")
      {
        _embed_dim = 1024;
        _depth = 24;
        _num_heads = 16;
        _qkv_bias = true;
      }
    else if (vit_flavor == "vit_large_patch32")
      {
        _patch_size = 32;
        _embed_dim = 1024;
        _depth = 24;
        _num_heads = 16;
        _qkv_bias = true;
      }
    else if (vit_flavor == "vit_huge_patch16")
      {
        _embed_dim = 1280;
        _depth = 32;
        _num_heads = 16;
      }
    else if (vit_flavor == "vit_huge_patch32")
      {
        _patch_size = 32;
        _embed_dim = 1280;
        _depth = 32;
        _num_heads = 16;
      }
    else if (vit_flavor == "vit_tiny_patch16")
      {
        _depth = 12;
        _num_heads = 3;
        _mlp_ratio = 4.0;
        _embed_dim = 192;
        _qkv_bias = true;
      }
    else if (vit_flavor == "vit_small_patch16")
      {
        _depth = 12;
        _num_heads = 6;
        _mlp_ratio = 4.0;
        _embed_dim = 384;
        _qkv_bias = true;
      }
    else
      {
        throw MLLibBadParamException("unknown ViT flavor " + vit_flavor);
      }
    init_block(_img_size, _patch_size, _in_chans, _embed_dim, _num_heads,
               _mlp_ratio, _qkv_bias, _qk_scale, _drop_rate, _attn_drop_rate,
               _realformer);
  }

  void ViT::init_block(const int &img_size, const int &patch_size,
                       const int &in_chans, const int &embed_dim,
                       const int &num_heads, const double &mlp_ratio,
                       const double &qkv_bias, const double &qk_scale,
                       const double &drop_rate, const double &attn_drop_rate,
                       const bool &realformer)
  {
    _num_features = embed_dim;
    _patch_embed = register_module(
        "patch_embed", PatchEmbed(img_size, patch_size, in_chans, embed_dim));
    unsigned int num_patches = _patch_embed->_num_patches;

    _cls_token
        = register_parameter("cls_token", torch::randn({ 1, 1, embed_dim }));
    _pos_embed = register_parameter(
        "pos_embed", torch::randn({ 1, num_patches + 1, embed_dim }));
    _pos_drop = register_module(
        "pos_drop", torch::nn::Dropout(torch::nn::DropoutOptions(drop_rate)));

    for (unsigned int d = 0; d < _depth; ++d)
      {
        _blocks->push_back(Block(embed_dim, num_heads, mlp_ratio, qkv_bias,
                                 qk_scale, drop_rate, attn_drop_rate,
                                 realformer));
      }
    register_module("blocks", _blocks);
    _norm = register_module(
        "norm",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({ embed_dim })));

    _head
        = register_module("head", torch::nn::Linear(embed_dim, _num_classes));
  }

  torch::Tensor ViT::forward_features(torch::Tensor x)
  {
    unsigned int B = x.size(0);
    x = _patch_embed(x);

    auto cls_tokens = _cls_token.expand({ B, -1, -1 });
    x = torch::cat({ cls_tokens, x }, 1);
    x = x + _pos_embed;
    x = _pos_drop(x);

    torch::Tensor prev_x;
    for (const auto &blk : *_blocks)
      {
        x = blk->as<Block>()->forward(x, prev_x);
      }

    x = _norm(x);
    x = torch::narrow(x, 1, 0, 1); // x[:,0]
    return x;
  }

  torch::Tensor ViT::forward(torch::Tensor x)
  {
    x = forward_features(x);
    x = _head(x);
    x = x.reshape({ x.size(0), _num_classes }); // custom
    return x;
  }
}
