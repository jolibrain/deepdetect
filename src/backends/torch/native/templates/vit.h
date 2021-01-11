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

#ifndef VIT_H
#define VIT_H

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "torch/torch.h"
#pragma GCC diagnostic pop
#include "../../torchinputconns.h"
#include "mllibstrategy.h"
#include "../native_net.h"

namespace dd
{

  class ViT : public NativeModuleImpl<ViT>
  {

    class MLPImpl : public torch::nn::Cloneable<MLPImpl>
    {
    public:
      MLPImpl(const int &input_dim, const int &hidden_dim,
              const int &output_dim, const std::string &act = "gelu",
              const double &drop = 0.0)
          : _input_dim(input_dim), _hidden_dim(hidden_dim),
            _output_dim(output_dim), _act(act), _drop(drop)
      {
        init_block();
      }

      MLPImpl(const MLPImpl &m)
          : torch::nn::Module(m), _input_dim(m._input_dim),
            _hidden_dim(m._hidden_dim), _output_dim(m._output_dim),
            _act(m._act), _drop(m._drop)
      {
      }

      ~MLPImpl()
      {
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

      torch::nn::Linear _fc1{ nullptr };
      torch::nn::Linear _fc2{ nullptr };
      torch::nn::Dropout _drop1{ nullptr };
    };

    typedef torch::nn::ModuleHolder<MLPImpl> MLP;

    class AttentionImpl : public torch::nn::Cloneable<AttentionImpl>
    {
    public:
      AttentionImpl(const int &dim, const int &num_heads = 8,
                    const bool &qkv_bias = false,
                    const double &qk_scale = -1.0,
                    const double &attn_drop_val = 0.0,
                    const double &proj_drop_val = 0.0,
                    const bool &realformer = false)
          : _dim(dim), _num_heads(num_heads), _qkv_bias(qkv_bias),
            _qk_scale(qk_scale), _attn_drop_val(attn_drop_val),
            _proj_drop_val(proj_drop_val), _realformer(realformer)
      {
        init_block();
      }

      AttentionImpl(const AttentionImpl &a)
          : torch::nn::Module(a), _dim(a._dim), _num_heads(a._num_heads),
            _qkv_bias(a._qkv_bias), _qk_scale(a._qk_scale),
            _attn_drop_val(a._attn_drop_val), _proj_drop_val(a._proj_drop_val),
            _realformer(a._realformer)
      {
      }

      ~AttentionImpl()
      {
      }

      void reset()
      {
        init_block();
      }

      torch::Tensor residual_mha(torch::Tensor x, torch::Tensor &prev);

      torch::Tensor forward(torch::Tensor x, torch::Tensor &prev);

    protected:
      void init_block();

      unsigned int _dim;
      unsigned int _num_heads = 8;
      bool _qkv_bias = false;
      double _qk_scale = -1.0;
      double _attn_drop_val = 0.0;
      double _proj_drop_val = 0.0;

      double _scale = 1.0;
      unsigned int _head_dim = 0;

      torch::nn::Linear _qkv{ nullptr };
      torch::nn::Dropout _attn_drop{ nullptr };
      torch::nn::Linear _proj{ nullptr };
      torch::nn::Dropout _proj_drop{ nullptr };

      bool _realformer = false;
    };

    typedef torch::nn::ModuleHolder<AttentionImpl> Attention;

    class BlockImpl : public torch::nn::Cloneable<BlockImpl>
    {
    public:
      BlockImpl(const int &dim, const int &num_heads,
                const double &mlp_ratio = 4.0, const bool &qkv_bias = false,
                const double &qk_scale = -1.0, const double &drop_val = 0.0,
                const double &attn_drop_val = 0.0,
                const bool &realformer = false)
          : _dim(dim), _num_heads(num_heads), _mlp_ratio(mlp_ratio),
            _qkv_bias(qkv_bias), _qk_scale(qk_scale), _drop_val(drop_val),
            _attn_drop_val(attn_drop_val), _realformer(realformer)
      {
        init_block(_mlp_ratio, _qkv_bias, _qk_scale, _drop_val, _attn_drop_val,
                   realformer);
      }

      BlockImpl(const BlockImpl &b)
          : torch::nn::Module(b), _dim(b._dim), _num_heads(b._num_heads),
            _mlp_ratio(b._mlp_ratio), _qkv_bias(b._qkv_bias),
            _qk_scale(b._qk_scale), _drop_val(b._drop_val),
            _attn_drop_val(b._attn_drop_val), _realformer(b._realformer)
      {
      }

      ~BlockImpl()
      {
      }

      void reset()
      {
        init_block(_mlp_ratio, _qkv_bias, _qk_scale, _drop_val, _attn_drop_val,
                   _realformer);
      }

      torch::Tensor forward(torch::Tensor x, torch::Tensor &prev);

    protected:
      void init_block(const double &mlp_ratio, const bool &qkv_bias,
                      const double &qk_scale, const double &drop,
                      const double &attn_drop, const bool &realformer);

      unsigned int _dim = 0;
      unsigned int _num_heads = 0;

      double _mlp_ratio;
      double _qkv_bias;
      double _qk_scale;
      double _drop_val;
      double _attn_drop_val;
      bool _realformer;

      torch::nn::LayerNorm _norm1{ nullptr };

    public:
      Attention _attn{ nullptr };

    protected:
      torch::nn::LayerNorm _norm2{ nullptr };
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

      ~PatchEmbedImpl()
      {
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
    };

    typedef torch::nn::ModuleHolder<PatchEmbedImpl> PatchEmbed;

    // TODO: Hybrid embed

  public:
    ViT(const int &img_size = 224, const int &patch_size = 16,
        const int &in_chans = 3, const int &num_classes = 2,
        const int &embed_dim = 768, const int &depth = 12,
        const int &num_heads = 12, const double &mlp_ratio = 4.0,
        const bool &qkv_bias = false, const double &qk_scale = -1.0,
        const double &drop_rate = 0.0, const double &attn_drop_rate = 0.0,
        const bool &realformer = false)
        : _img_size(img_size), _patch_size(patch_size), _in_chans(in_chans),
          _num_classes(num_classes), _embed_dim(embed_dim), _depth(depth),
          _num_heads(num_heads), _mlp_ratio(mlp_ratio), _qkv_bias(qkv_bias),
          _qk_scale(qk_scale), _drop_rate(drop_rate),
          _attn_drop_rate(attn_drop_rate), _realformer(realformer)
    {
      init_block(_img_size, _patch_size, _in_chans, _embed_dim, _num_heads,
                 _mlp_ratio, _qkv_bias, _qk_scale, _drop_rate, _attn_drop_rate,
                 _realformer);
    }

    ViT(const ImgTorchInputFileConn &inputc, const APIData &ad_params)
    {
      get_params_and_init_block(inputc, ad_params);
    }

    ViT(const ViT &v)
        : torch::nn::Module(v), _img_size(v._img_size),
          _patch_size(v._patch_size), _in_chans(v._in_chans),
          _num_classes(v._num_classes), _embed_dim(v._embed_dim),
          _depth(v._depth), _num_heads(v._num_heads), _mlp_ratio(v._mlp_ratio),
          _qkv_bias(v._qkv_bias), _qk_scale(v._qk_scale),
          _drop_rate(v._drop_rate), _attn_drop_rate(v._attn_drop_rate)
    {
    }

    virtual ~ViT()
    {
    }

    void reset() override
    {
      init_block(_img_size, _patch_size, _in_chans, _embed_dim, _num_heads,
                 _mlp_ratio, _qkv_bias, _qk_scale, _drop_rate, _attn_drop_rate,
                 _realformer);
    }

    void get_params_and_init_block(const ImgTorchInputFileConn &inputc,
                                   const APIData &ad_params);

    torch::Tensor forward(torch::Tensor x) override;

    torch::Tensor forward_features(torch::Tensor x);

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
      throw MLLibInternalException("ViT::loss not implemented");
    }

  protected:
    void init_block(const int &img_size, const int &patch_size,
                    const int &in_chans, const int &embed_dim,
                    const int &num_heads, const double &mlp_ratio,
                    const double &qkv_bias, const double &qk_scale,
                    const double &drop_rate, const double &attn_drop_rate,
                    const bool &realformer);

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
    bool _realformer = false;

    PatchEmbed _patch_embed{ nullptr };
    torch::Tensor _cls_token;
    torch::Tensor _pos_embed;
    torch::nn::Dropout _pos_drop{ nullptr };
    torch::nn::ModuleList _blocks;
    torch::nn::LayerNorm _norm{ nullptr };
    torch::nn::Linear _head{ nullptr };
  };

}

#endif
