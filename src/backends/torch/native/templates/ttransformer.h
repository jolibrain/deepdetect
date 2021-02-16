/**
 * DeepDetect
 * Copyright (c) 2020 Jolibrain
 * Author:  Guillaume Infantes <guillaume.infantes@jolibrain.com>
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

#ifndef TTRANSFOMER_H
#define TTRANSFOMER_H

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "torch/torch.h"
#pragma GCC diagnostic pop
#include "../../torchinputconns.h"
#include "mllibstrategy.h"
#include "../native_net.h"

#include "ttransformer/ttypes.h"
#include "ttransformer/tencoder.h"
#include "ttransformer/tembedder.h"
#include "ttransformer/positionalenc.h"
#include "ttransformer/tdecoder.h"

#define DEFAULT_EMBED_NLAYERS 3
#define DEFAULT_EMBED_DIM 32
#define DEFAULT_ENCODER_NHEADS 8
#define DEFAULT_ENCODER_NLAYERS 1
#define DEFAULT_ENCODER_ACTIVATION Activation::gelu
#define DEFAULT_DECODER_ACTIVATION Activation::gelu
#define DEFAULT_DECODER_NHEADS 8
#define DEFAULT_DECODER_NLAYERS 1
#define DEFAULT_EMBED_ACTIVATION Activation::relu
#define DEFAULT_EMBED_TYPE EmbedType::step
#define DEFAULT_DROPOUT 0.1
#define DEFAULT_POSITIONAL_ENCODING_TYPE PEType::sincos
#define DEFAULT_POSITIONAL_ENCODING_AGGREGATION PEAggregation::sum
#define DEFAULT_POSITIONAL_ENCODING_LEARN false

namespace dd
{

  class TTransformer : public NativeModuleImpl<TTransformer>
  {
  public:
    TTransformer(int input_len, int input_dim, int output_len, int output_dim,
                 PEType petype = PEType::sincos,
                 bool pelearn = DEFAULT_POSITIONAL_ENCODING_LEARN,
                 float pe_dropout = DEFAULT_DROPOUT,
                 int embed_nlayers = DEFAULT_EMBED_NLAYERS,
                 int embed_dim = DEFAULT_EMBED_DIM,
                 Activation embed_activation = DEFAULT_EMBED_ACTIVATION,
                 EmbedType embed_type = DEFAULT_EMBED_TYPE,
                 float embed_dropout = DEFAULT_DROPOUT,
                 int nEncoderHeads = DEFAULT_ENCODER_NHEADS,
                 int nEncoderLayers = DEFAULT_ENCODER_NLAYERS,
                 int encoder_hidden_dim = -1,
                 Activation encoder_activation = DEFAULT_ENCODER_ACTIVATION,
                 float encoder_dropout = DEFAULT_DROPOUT,
                 int nDecoderHeads = DEFAULT_DECODER_NHEADS,
                 int nDecoderLayers = DEFAULT_DECODER_NHEADS,
                 int decoder_hidden_dim = -1,
                 float decoder_dropout = DEFAULT_DROPOUT, bool autoreg = false)
        : _input_len(input_len), _input_dim(input_dim),
          _output_len(output_len), _output_dim(output_dim), _petype(petype),
          _pelearn(pelearn), _pe_dropout(pe_dropout),
          _embed_nlayers(embed_nlayers), _embed_dim(embed_dim),
          _embed_activation(embed_activation), _embed_type(embed_type),
          _embed_dropout(embed_dropout), _encoder_nheads(nEncoderHeads),
          _encoder_nlayers(nEncoderLayers),
          _encoder_hidden_dim(encoder_hidden_dim),
          _encoder_activation(encoder_activation),
          _encoder_dropout(encoder_dropout), _decoder_nheads(nDecoderHeads),
          _decoder_nlayers(nDecoderLayers),
          _decoder_hidden_dim(decoder_hidden_dim),
          _decoder_dropout(decoder_dropout), _autoreg(autoreg)
    {
      if (_encoder_hidden_dim == -1)
        _encoder_hidden_dim = _input_len * _embed_dim;
      if (_decoder_hidden_dim == -1)
        _decoder_hidden_dim = _output_len * _embed_dim;
      init();
    }

    TTransformer(const CSVTSTorchInputFileConn &inputc,
                 const APIData &template_params,
                 const std::shared_ptr<spdlog::logger> &logger)
        : _logger(logger)
    {
      if (inputc._forecast_timesteps > 0)
        {
          _output_len = inputc._forecast_timesteps;
          _input_len = inputc._backcast_timesteps;
        }
      else
        _input_len = _output_len = inputc._timesteps;
      if (inputc._label.size() != 0)
        {
          _input_dim = inputc._datadim - inputc._label.size();
          _output_dim = inputc._label.size();
        }
      else
        _input_dim = _output_dim = inputc._datadim;

      _embed_nlayers = DEFAULT_EMBED_NLAYERS;
      _embed_dim = DEFAULT_EMBED_DIM;
      _embed_type = DEFAULT_EMBED_TYPE;
      _embed_activation = DEFAULT_EMBED_ACTIVATION;
      _embed_dropout = DEFAULT_DROPOUT;

      _pelearn = DEFAULT_POSITIONAL_ENCODING_LEARN;
      _petype = DEFAULT_POSITIONAL_ENCODING_TYPE;
      _peagg = DEFAULT_POSITIONAL_ENCODING_AGGREGATION;
      _pe_dropout = DEFAULT_DROPOUT;

      _encoder_nlayers = DEFAULT_ENCODER_NLAYERS;
      _encoder_nheads = DEFAULT_ENCODER_NHEADS;
      _encoder_hidden_dim = _input_len * _embed_dim;
      _encoder_activation = DEFAULT_ENCODER_ACTIVATION;
      _encoder_dropout = DEFAULT_DROPOUT;

      _decoder_nheads = DEFAULT_DECODER_NHEADS;
      _decoder_nlayers = DEFAULT_DECODER_NLAYERS;
      _decoder_hidden_dim = _output_len * _embed_dim;
      _decoder_dropout = DEFAULT_DROPOUT;
      _decoder_activation = DEFAULT_DECODER_ACTIVATION;

      _autoreg = false;
      _simple_decoder = true;

      if (template_params.has("embed"))
        {
          APIData embed_ad = template_params.getobj("embed");
          if (embed_ad.has("layers"))
            _embed_nlayers = embed_ad.get("layers").get<int>();

          if (embed_ad.has("dim"))
            _embed_dim = embed_ad.get("dim").get<int>();

          if (embed_ad.has("type"))
            {
              std::string t = embed_ad.get("type").get<std::string>();
              if (t == "serie")
                _embed_type = EmbedType::serie;
              else if (t == "all")
                _embed_type = EmbedType::all;
              else if (t == "step")
                _embed_type = EmbedType::step;
              else
                throw MLLibBadParamException("unknown embed type: " + t);
            }

          if (embed_ad.has("activation"))
            {
              std::string ac = embed_ad.get("activation").get<std::string>();
              if (ac == "siren")
                _embed_activation = Activation::siren;
              else if (ac == "relu")
                _embed_activation = Activation::relu;
              else if (ac == "gelu")
                _embed_activation = Activation::gelu;
              else
                throw MLLibBadParamException("unknown embedder activation: "
                                             + ac);
            }

          if (embed_ad.has("dropout"))
            _embed_dropout = embed_ad.get("dropout").get<double>();
        }

      if (template_params.has("encoder"))
        {
          APIData encoder_ad = template_params.getobj("encoder");
          if (encoder_ad.has("heads"))
            _encoder_nheads = encoder_ad.get("heads").get<int>();

          if (encoder_ad.has("layers"))
            _encoder_nlayers = encoder_ad.get("layers").get<int>();
          if (encoder_ad.has("dropout"))
            _encoder_dropout = encoder_ad.get("dropout").get<double>();
          if (encoder_ad.has("hidden_dim"))
            _encoder_hidden_dim
                = encoder_ad.get("hidden_dim").get<int>() * _input_len;
          if (encoder_ad.has("activation"))
            {
              std::string ac = encoder_ad.get("activation").get<std::string>();
              if (ac == "relu")
                _encoder_activation = Activation::relu;
              else if (ac == "gelu")
                _encoder_activation = Activation::gelu;
              else
                throw MLLibBadParamException("unsupported encoder activation: "
                                             + ac);
            }
        }

      if (template_params.has("decoder"))
        {
          APIData decoder_ad = template_params.getobj("decoder");
          if (decoder_ad.has("type"))
            {
              std::string dt = decoder_ad.get("type").get<std::string>();
              if (dt == "simple")
                _simple_decoder = true;
              else if (dt == "transformer")
                _simple_decoder = false;
              else
                throw MLLibBadParamException("unknow decoder style " + dt);
            }
          if (decoder_ad.has("activation"))
            {
              std::string da = decoder_ad.get("activation").get<std::string>();
              if (da == "relu")
                _decoder_activation = Activation::relu;
              else if (da == "gelu")
                _decoder_activation = Activation::gelu;
              else
                throw MLLibBadParamException("unsupported decoder activation: "
                                             + da);
            }

          if (decoder_ad.has("heads"))
            {
              _decoder_nheads = decoder_ad.get("heads").get<int>();
              _simple_decoder = false;
            }

          if (decoder_ad.has("layers"))
            _decoder_nlayers = decoder_ad.get("layers").get<int>();
          if (decoder_ad.has("dropout"))
            _decoder_dropout = decoder_ad.get("dropout").get<double>();

          if (decoder_ad.has("hidden_dim"))
            _decoder_hidden_dim
                = decoder_ad.get("hidden_dim").get<int>() * _output_len;
        }

      if (template_params.has("autoreg"))
        _autoreg = template_params.get("autoreg").get<bool>();
      else
        _autoreg = false;

      if (_autoreg)
        _simple_decoder = false;

      if (_autoreg)
        throw MLLibBadParamException(
            "autoreg not yet implemented due to masks not implementated");

      if (template_params.has("positional_encoding"))
        {
          APIData pe_ad = template_params.getobj("positional_encoding");

          std::string type;
          if (pe_ad.has("type"))
            {
              type = pe_ad.get("type").get<std::string>();
              if (type == "naive")
                _petype = PEType::naive;
              else if (type == "sincos")
                _petype = PEType::sincos;
              else if (type == "none")
                _petype = PEType::none;
              else
                throw MLLibBadParamException(
                    "unknow positional encoding type: " + type);
            }
          if (pe_ad.has("learn"))
            _pelearn = pe_ad.get("learn").get<bool>();

          if (pe_ad.has("dropout"))
            _pe_dropout = pe_ad.get("dropout").get<double>();

          if (pe_ad.has("agg"))
            {
              std::string ada = pe_ad.get("agg").get<std::string>();
              if (ada == "sum")
                _peagg = PEAggregation::sum;
              else if (ada == "cat")
                _peagg = PEAggregation::cat;
              else
                throw MLLibBadParamException(
                    "unknow positional encoding aggregation: " + ada);
            }
        }
      init();
    }

    virtual ~TTransformer() override
    {
    }

    torch::Tensor forward(torch::Tensor x) override;
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

    torch::Tensor loss(std::string loss, torch::Tensor input,
                       torch::Tensor output, torch::Tensor target) override
    {
      // simple transformer w/o nbeats style ensembling/boosting
      (void)input;
      if (loss.empty() || loss == "L1" || loss == "l1")
        return torch::l1_loss(output, target);
      if (loss == "L2" || loss == "l2" || loss == "eucl")
        return torch::mse_loss(output, target);
      throw MLLibBadParamException("unknown loss " + loss);
    }
    torch::Tensor cleanup_output(torch::Tensor output) override
    {
      return output;
    }

    void reset() override
    {
      init();
    }

  protected:
    std::shared_ptr<spdlog::logger> _logger; /**< mllib logger. */

    void init();

    torch::Tensor generate_mask_none(int len);
    torch::Tensor generate_mask_past(int len);
    void set_encoder_mask(torch::Tensor mask);
    void set_decoder_mask(torch::Tensor mask);
    Embedder _embedder{ nullptr };
    torch::Tensor _encoder_mask;
    torch::Tensor _decoder_mask;
    bool _simple_decoder;
    PositionalEncoding _pe{ nullptr };
    TEncoder _encoder{ nullptr };
    TDecoder _decoder{ nullptr };
    int _input_len;
    int _input_dim;
    int _output_len;
    int _output_dim;
    PEType _petype;
    bool _pelearn;
    PEAggregation _peagg;
    float _pe_dropout;
    int _embed_nlayers;
    int _embed_dim;
    Activation _embed_activation;
    EmbedType _embed_type;
    float _embed_dropout;
    int _encoder_nheads;
    int _encoder_nlayers;
    int _encoder_hidden_dim;
    Activation _encoder_activation;
    float _encoder_dropout;
    int _decoder_nheads;
    int _decoder_nlayers;
    int _decoder_hidden_dim;
    float _decoder_dropout;
    Activation _decoder_activation;
    bool _autoreg;
  };
}
#endif
