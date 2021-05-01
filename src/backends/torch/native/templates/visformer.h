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

#ifndef VISFORMER_H
#define VISFORMER_H

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "torch/torch.h"
#pragma GCC diagnostic pop
#include "../../torchinputconns.h"
#include "mllibstrategy.h"
#include "../native_net.h"

namespace dd
{

  class Visformer : public NativeModuleImpl<Visformer>
  {

    class MLPImpl : public torch::nn::Cloneable<MLPImpl>
    {
    public:
      MLPImpl(const int &input_dim, const int &hidden_dim,
              const int &output_dim, const double &drop = 0.0,
              const int &group = 8, const bool &spatial_conv = false)
          : _input_dim(input_dim), _hidden_dim(hidden_dim),
            _output_dim(output_dim), _drop(drop), _group(group),
            _spatial_conv(spatial_conv)
      {
        init_block();
      }

      MLPImpl(const MLPImpl &m)
          : torch::nn::Module(m), _input_dim(m._input_dim),
            _hidden_dim(m._hidden_dim), _output_dim(m._output_dim),
            _drop(m._drop), _group(m._group), _spatial_conv(m._spatial_conv)
      {
      }

      MLPImpl &operator=(const MLPImpl &m)
      {
        torch::nn::Module::operator=(m);
        _input_dim = m._input_dim;
        _hidden_dim = m._hidden_dim;
        _output_dim = m._output_dim;
        _drop = m._drop;

        _convc1 = m._convc1;
        _convc2 = m._convc2;
        _convc3 = m._convc3;
        _drop1 = m._drop1;
        return *this;
      }

      void reset()
      {
        init_block();
      }

      torch::Tensor forward(torch::Tensor x);

    protected:
      void init_block();

      unsigned int _input_dim = 0;
      unsigned int _hidden_dim = 0;
      unsigned int _output_dim = 0;
      std::string _act = "gelu";
      double _drop = 0.0;
      int _group = 8;
      bool _spatial_conv = false;

      torch::nn::Conv2d _convc1{ nullptr };
      torch::nn::Conv2d _convc2{ nullptr };
      torch::nn::Conv2d _convc3{ nullptr };
      torch::nn::Dropout _drop1{ nullptr };
    };

    typedef torch::nn::ModuleHolder<MLPImpl> MLP;

    class AttentionImpl : public torch::nn::Cloneable<AttentionImpl>
    {
    public:
      AttentionImpl(const int &dim, const int &num_heads = 8,
                    const double &head_dim_ratio = 1.0,
                    const bool &qkv_bias = false,
                    const double &qk_scale = -1.0,
                    const double &attn_drop_val = 0.0,
                    const double &proj_drop_val = 0.0)
          : _dim(dim), _num_heads(num_heads), _head_dim_ratio(head_dim_ratio),
            _qkv_bias(qkv_bias), _qk_scale(qk_scale),
            _attn_drop_val(attn_drop_val), _proj_drop_val(proj_drop_val)
      {
        init_block();
      }

      AttentionImpl(const AttentionImpl &a)
          : torch::nn::Module(a), _dim(a._dim), _num_heads(a._num_heads),
            _head_dim_ratio(a._head_dim_ratio), _qkv_bias(a._qkv_bias),
            _qk_scale(a._qk_scale), _attn_drop_val(a._attn_drop_val),
            _proj_drop_val(a._proj_drop_val)
      {
      }

      AttentionImpl &operator=(const AttentionImpl &a)
      {
        torch::nn::Module::operator=(a);
        _dim = a._dim;
        _num_heads = a._num_heads;
        _head_dim_ratio = a._head_dim_ratio;
        _qkv_bias = a._qkv_bias;
        _qk_scale = a._qk_scale;
        _attn_drop_val = a._attn_drop_val;
        _proj_drop_val = a._proj_drop_val;

        _scale = a._scale;
        _head_dim = a._head_dim;
        _qkv = a._qkv;
        _attn_drop = a._attn_drop;
        _proj = a._proj;
        _proj_drop = a._proj_drop;
        return *this;
      }

      void reset()
      {
        init_block();
      }

      torch::Tensor forward(torch::Tensor x);

    protected:
      void init_block();

      unsigned int _dim;
      unsigned int _num_heads = 8;
      double _head_dim_ratio = 1.0;
      bool _qkv_bias = false;
      double _qk_scale = -1.0;
      double _attn_drop_val = 0.0;
      double _proj_drop_val = 0.0;

      double _scale = 1.0;
      unsigned int _head_dim = 0;

      torch::nn::Conv2d _qkv{ nullptr };
      torch::nn::Dropout _attn_drop{ nullptr };
      torch::nn::Conv2d _proj{ nullptr };
      torch::nn::Dropout _proj_drop{ nullptr };
    };

    typedef torch::nn::ModuleHolder<AttentionImpl> Attention;

    class BlockImpl : public torch::nn::Cloneable<BlockImpl>
    {
    public:
      BlockImpl(const int &dim, const int &num_heads,
                const int &head_dim_ratio = 1.0, const double &mlp_ratio = 4.0,
                const bool &qkv_bias = false, const double &qk_scale = -1.0,
                const double &drop_val = 0.0,
                const double &attn_drop_val = 0.0, const int &group = 8,
                const bool &attn_disabled = false,
                const bool &spatial_conv = false)
          : _dim(dim), _num_heads(num_heads), _head_dim_ratio(head_dim_ratio),
            _mlp_ratio(mlp_ratio), _qkv_bias(qkv_bias), _qk_scale(qk_scale),
            _drop_val(drop_val), _attn_drop_val(attn_drop_val), _group(group),
            _attn_disabled(attn_disabled), _spatial_conv(spatial_conv)
      {
        init_block();
      }

      BlockImpl(const BlockImpl &b)
          : torch::nn::Module(b), _dim(b._dim), _num_heads(b._num_heads),
            _head_dim_ratio(b._head_dim_ratio), _mlp_ratio(b._mlp_ratio),
            _qkv_bias(b._qkv_bias), _qk_scale(b._qk_scale),
            _drop_val(b._drop_val), _attn_drop_val(b._attn_drop_val),
            _group(b._group), _attn_disabled(b._attn_disabled),
            _spatial_conv(b._spatial_conv)
      {
      }

      BlockImpl &operator=(const BlockImpl &b)
      {
        torch::nn::Module::operator=(b);
        _dim = b._dim;
        _num_heads = b._num_heads;
        _head_dim_ratio = b._head_dim_ratio;
        _mlp_ratio = b._mlp_ratio;
        _qkv_bias = b._qkv_bias;
        _qk_scale = b._qk_scale;
        _drop_val = b._drop_val;
        _attn_drop_val = b._attn_drop_val;
        _group = b._group;
        _attn_disabled = b._attn_disabled;
        _spatial_conv = b._spatial_conv;

        _norm1 = b._norm1;
        _attn = b._attn;
        _norm2 = b._norm2;
        _mlp = b._mlp;
        return *this;
      }

      void reset()
      {
        init_block();
      }

      torch::Tensor forward(torch::Tensor x);

    protected:
      void init_block();

      unsigned int _dim = 0;
      unsigned int _num_heads = 0;
      double _head_dim_ratio = 1.0;

      double _mlp_ratio;
      double _qkv_bias;
      double _qk_scale;
      double _drop_val;
      double _attn_drop_val;
      int _group;
      bool _attn_disabled;
      bool _spatial_conv;

      torch::nn::BatchNorm2d _norm1{ nullptr };

    public:
      Attention _attn{ nullptr };

    protected:
      torch::nn::BatchNorm2d _norm2{ nullptr };
      MLP _mlp{ nullptr };
    };

    typedef torch::nn::ModuleHolder<BlockImpl> Block;

    class PatchEmbedImpl : public torch::nn::Cloneable<PatchEmbedImpl>
    {
    public:
      PatchEmbedImpl(const int &img_size = 224, const int &patch_size = 16,
                     const int &in_chans = 3, const int &embed_dim = 768)
          : _in_chans(in_chans), _embed_dim(embed_dim)
      {
        init_block(img_size, patch_size);
      }

      PatchEmbedImpl(const PatchEmbedImpl &p)
          : torch::nn::Module(p), _img_size(p._img_size),
            _patch_size(p._patch_size), _in_chans(p._in_chans),
            _embed_dim(p._embed_dim)
      {
      }

      PatchEmbedImpl &operator=(const PatchEmbedImpl &p)
      {
        torch::nn::Module::operator=(p);
        _img_size = p._img_size;
        _patch_size = p._patch_size;
        _in_chans = p._in_chans;
        _embed_dim = p._embed_dim;

        _proj = p._proj;
        return *this;
      }

      void reset()
      {
        init_block(_img_size.first, _patch_size.first);
      }

      torch::Tensor forward(torch::Tensor x);

      unsigned int _num_patches = 0;

    protected:
      void init_block(const int &img_size, const int &patch_size);

      std::pair<unsigned int, unsigned int> _img_size;
      std::pair<unsigned int, unsigned int> _patch_size;
      unsigned int _in_chans = 3;
      unsigned int _embed_dim = 768;

      torch::nn::Conv2d _proj{ nullptr };
      torch::nn::BatchNorm2d _norm_embed{ nullptr };
    };

    typedef torch::nn::ModuleHolder<PatchEmbedImpl> PatchEmbed;

  public:
    Visformer(const ImgTorchInputFileConn &inputc, const APIData &ad_params)
    {
      get_params_and_init_block(inputc, ad_params);
    }

    // TODO: copy constructors

    virtual ~Visformer() = default;

    void reset() override
    {
      init_block();
    }

    void get_params_and_init_block(const ImgTorchInputFileConn &inputc,
                                   const APIData &ad_params);

    torch::Tensor forward(torch::Tensor x);

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
                       torch::Tensor output, torch::Tensor target) override
    {
      (void)loss;
      (void)input;
      (void)output;
      (void)target;
      throw MLLibInternalException("Visformer::loss not implemented");
    }

  protected:
    void init_block();

    unsigned int _img_size = 224;
    unsigned int _patch_size = 16;
    unsigned int _in_chans = 3;
    unsigned int _num_classes;
    unsigned int _embed_dim;
    unsigned int _depth = 12;
    unsigned int _num_heads;
    double _mlp_ratio;
    double _qkv_bias;
    double _qk_scale;
    double _drop_rate;
    double _attn_drop_rate;
    unsigned int _num_features;
    int _group;
    std::string _attn_stage;
    std::string _spatial_conv;

    torch::nn::Sequential _stem{ nullptr };
    PatchEmbed _patch_embed1{ nullptr };
    PatchEmbed _patch_embed2{ nullptr };
    PatchEmbed _patch_embed3{ nullptr };
    torch::Tensor _cls_token;
    torch::Tensor _pos_embed1;
    torch::Tensor _pos_embed2;
    torch::Tensor _pos_embed3;
    torch::nn::Dropout _pos_drop{ nullptr };
    torch::nn::ModuleList _stage1_blocks;
    torch::nn::ModuleList _stage2_blocks;
    torch::nn::ModuleList _stage3_blocks;
    torch::nn::BatchNorm2d _norm{ nullptr };
    torch::nn::AdaptiveAvgPool2d _global_pooling{ nullptr };
    torch::nn::Linear _head{ nullptr };
  };

}

#endif
