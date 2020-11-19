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

  class ViT : public NativeModule
  {

    class MLPImpl : public torch::nn::Module
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

      ~MLPImpl()
      {
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

    class AttentionImpl : public torch::nn::Module
    {
    public:
      AttentionImpl(const int &dim, const int &num_heads = 8,
                    const bool &qkv_bias = false,
                    const double &qk_scale = -1.0,
                    const double &attn_drop_val = 0.0,
                    const double &proj_drop_val = 0.0)
          : _dim(dim), _num_heads(num_heads), _qkv_bias(qkv_bias),
            _qk_scale(qk_scale), _attn_drop_val(attn_drop_val),
            _proj_drop_val(proj_drop_val)
      {
        init_block();
      }

      ~AttentionImpl()
      {
      }

      torch::Tensor forward(torch::Tensor x);

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
    };

    typedef torch::nn::ModuleHolder<AttentionImpl> Attention;

    class BlockImpl : public torch::nn::Module
    {
    public:
      BlockImpl(const int &dim, const int &num_heads,
                const double &mlp_ratio = 4.0, const bool &qkv_bias = false,
                const double &qk_scale = -1.0, const double &drop_val = 0.0,
                const double &attn_drop_val = 0.0,
                const double &drop_path = 0.0, const std::string &act = "gelu")
          : _dim(dim), _num_heads(num_heads)
      {
        init_block(mlp_ratio, qkv_bias, qk_scale, drop_val, attn_drop_val,
                   drop_path, act);
      }

      ~BlockImpl()
      {
      }

      torch::Tensor forward(torch::Tensor x);

    protected:
      void init_block(const double &mlp_ratio, const bool &qkv_bias,
                      const double &qk_scale, const double &drop,
                      const double &attn_drop, const double &drop_path,
                      const std::string &act);

      unsigned int _dim = 0;
      unsigned int _num_heads = 0;

      torch::nn::LayerNorm _norm1{ nullptr };
      Attention _attn{ nullptr };
      torch::nn::LayerNorm _norm2{ nullptr };
      MLP _mlp{ nullptr };
    };

    typedef torch::nn::ModuleHolder<BlockImpl> Block;

    class PatchEmbedImpl : public torch::nn::Module
    {
    public:
      PatchEmbedImpl(const int &img_size = 224, const int &patch_size = 16,
                     const int &in_chans = 3, const int &embed_dim = 768)
          : _in_chans(in_chans), _embed_dim(embed_dim)
      {
        init_block(img_size, patch_size);
      }

      ~PatchEmbedImpl()
      {
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
        const double &drop_rate = 0.0, const double &attn_drop_rate = 0.0)
        : _img_size(img_size), _num_classes(num_classes), _depth(depth)
    {
      init_block(_img_size, patch_size, in_chans, embed_dim, num_heads,
                 mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate);
    }

    ViT(const ImgTorchInputFileConn &inputc, const APIData &ad_params)
    {
      get_params_and_init_block(inputc, ad_params);
    }

    virtual ~ViT()
    {
    }

    void get_params_and_init_block(const ImgTorchInputFileConn &inputc,
                                   const APIData &ad_params);

    virtual torch::Tensor forward(torch::Tensor x);

    torch::Tensor forward_features(torch::Tensor x);

    virtual torch::Tensor extract(torch::Tensor x, std::string extract_layer)
    {
      (void)x;
      (void)extract_layer;
      return torch::Tensor();
    }

    virtual bool extractable(std::string extract_layer) const
    {
      (void)extract_layer;
      return false;
    }

    virtual std::vector<std::string> extractable_layers() const
    {
      return std::vector<std::string>();
    }

    virtual torch::Tensor cleanup_output(torch::Tensor output)
    {
      return output;
    }

    virtual torch::Tensor loss(std::string loss, torch::Tensor input,
                               torch::Tensor output, torch::Tensor target)
    {
      (void)loss;
      (void)input;
      (void)output;
      (void)target;
      return torch::Tensor();
    }

    virtual void update_input_connector(TorchInputInterface &inputc)
    {
      (void)inputc;
    }

  protected:
    void init_block(const int &img_size, const int &patch_size,
                    const int &in_chans, const int &embed_dim,
                    const int &num_heads, const double &mlp_ratio,
                    const double &qkv_bias, const double &qk_scale,
                    const double &drop_rate, const double &attn_drop_rate);

    unsigned int _img_size = 224;
    unsigned int _num_classes;
    unsigned int _depth = 12;
    unsigned int _num_features;

    PatchEmbed _patch_embed{ nullptr };
    torch::Tensor _cls_token;
    torch::Tensor _pos_embed;
    torch::nn::Dropout _pos_drop{ nullptr };
    // torch::nn::ModuleList _blocks;
    std::vector<Block> _blocks;
    torch::nn::LayerNorm _norm{ nullptr };
    torch::nn::Linear _head{ nullptr };
  };

}

#endif
