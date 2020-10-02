#ifndef NBEATS_H
#define NBEATS_H

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "torch/torch.h"
#pragma GCC diagnostic pop
#include "../../torchinputconns.h"
#include "mllibstrategy.h"
#include "../native_net.h"

#define NBEATS_DEFAULT_STACK_TYPES                                            \
  {                                                                           \
    trend, seasonality, generic                                               \
  }
#define NBEATS_DEFAULT_NB_BLOCKS 3
#define NBEATS_DEFAULT_DATA_SIZE 1
#define NBEATS_DEFAULT_OUTPUT_SIZE 1
#define NBEATS_DEFAULT_BACKCAST_LENGTH 50
//#define NBEATS_DEFAULT_FORECAST_LENGTH 50
#define NBEATS_DEFAULT_THETAS                                                 \
  {                                                                           \
    2, 8, 3                                                                   \
  }
#define NBEATS_DEFAULT_SHARE_WEIGHTS false
#define NBEATS_DEFAULT_HIDDEN_LAYER_UNITS 10

namespace dd
{
  class NBeats : public NativeModule
  {

    enum BlockType
    {
      seasonality,
      trend,
      generic
    };

    class BlockImpl : public torch::nn::Module
    {
    public:
      BlockImpl(int units, int thetas_dim, int backcast_length,
                bool share_thetas, unsigned int data_size = 1)
          : _units(units), _thetas_dim(thetas_dim), _data_size(data_size),
            _backcast_length(backcast_length), _share_thetas(share_thetas)
      {
        init_block();
      }

      BlockImpl(BlockImpl &b)
          : torch::nn::Module(b), _units(b._units), _thetas_dim(b._thetas_dim),
            _data_size(b._data_size), _backcast_length(b._backcast_length),
            _share_thetas(b._share_thetas)
      {
        init_block();
      }

      torch::Tensor first_forward(torch::Tensor x);
      torch::Tensor first_extract(torch::Tensor x, std::string extract_layer);

    protected:
      void init_block();

      unsigned int _units;
      unsigned int _thetas_dim;
      unsigned int _data_size;
      unsigned int _backcast_length;
      bool _share_thetas;
      torch::nn::Linear _fc1{ nullptr };
      torch::nn::Linear _fc2{ nullptr };
      torch::nn::Linear _fc3{ nullptr };
      torch::nn::Linear _fc4{ nullptr };
      torch::nn::Linear _theta_b_fc{ nullptr };
      torch::nn::Linear _theta_f_fc{ nullptr };
    };

    typedef torch::nn::ModuleHolder<BlockImpl> Block;

    class SeasonalityBlockImpl : public BlockImpl
    {
    public:
      SeasonalityBlockImpl(int units, int thetas_dim, int backcast_length,
                           int data_size, torch::Tensor bS, torch::Tensor fS)
          : BlockImpl(units, thetas_dim, backcast_length, true, data_size),
            _bS(bS), _fS(fS)
      {
      }

      SeasonalityBlockImpl(SeasonalityBlockImpl &b)
          : BlockImpl(b), _bS(b._bS), _fS(b._fS)
      {
      }

      std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x);
      torch::Tensor extract(torch::Tensor x, std::string extract_layer);

    protected:
      torch::Tensor _bS, _fS;
    };

    typedef torch::nn::ModuleHolder<SeasonalityBlockImpl> SeasonalityBlock;

    class TrendBlockImpl : public BlockImpl
    {
    public:
      TrendBlockImpl(int units, int thetas_dim, int backcast_length,
                     int data_size, torch::Tensor bT, torch::Tensor fT)
          : BlockImpl(units, thetas_dim, backcast_length, true, data_size),
            _bT(bT), _fT(fT)
      {
      }

      TrendBlockImpl(TrendBlockImpl &b) : BlockImpl(b), _bT(b._bT), _fT(b._fT)
      {
      }

      std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x);
      torch::Tensor extract(torch::Tensor x, std::string extract_layer);

    protected:
      torch::Tensor _bT, _fT;
    };

    typedef torch::nn::ModuleHolder<TrendBlockImpl> TrendBlock;

    class GenericBlockImpl : public BlockImpl
    {
    public:
      GenericBlockImpl(int units, int thetas_dim, int backcast_length,
                       int data_size)
          : BlockImpl(units, thetas_dim, backcast_length, false, data_size)
      {
        _backcast_fc = register_module(
            "backcast_fc",
            torch::nn::Linear(_thetas_dim, _backcast_length * _data_size));
        _forecast_fc = register_module(
            "forecast_fc",
            torch::nn::Linear(_thetas_dim, _backcast_length * _data_size));
      }

      GenericBlockImpl(GenericBlockImpl &b) : BlockImpl(b)
      {
        _backcast_fc = b._backcast_fc;
        _forecast_fc = b._forecast_fc;
      }

      std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x);
      torch::Tensor extract(torch::Tensor x, std::string extract_layer);

    protected:
      torch::nn::Linear _backcast_fc{ nullptr };
      torch::nn::Linear _forecast_fc{ nullptr };
    };

    typedef torch::nn::ModuleHolder<GenericBlockImpl> GenericBlock;

    typedef std::vector<torch::nn::AnyModule> Stack;

  public:
    NBeats(std::vector<std::string> stackdef,
           std::vector<BlockType> stackTypes = { trend, seasonality, generic },
           int nb_blocks_per_stack = 3, int data_size = 1, int output_size = 1,
           int backcast_length = 50,
           std::vector<int> thetas_dims = { 2, 8, 3 },
           bool share_weights_in_stack = false, int hidden_layer_units = 10)
        : _data_size(data_size), _output_size(output_size),
          _backcast_length(backcast_length),
          _hidden_layer_units(hidden_layer_units),
          _nb_blocks_per_stack(nb_blocks_per_stack),
          _share_weights_in_stack(share_weights_in_stack),
          _stack_types(stackTypes), _thetas_dims(thetas_dims)
    {
      parse_stackdef(stackdef);
      create_nbeats();
    }

    NBeats(const CSVTSTorchInputFileConn &inputc,
           std::vector<std::string> stackdef,
           std::vector<BlockType> stackTypes = NBEATS_DEFAULT_STACK_TYPES,
           int nb_blocks_per_stack = NBEATS_DEFAULT_NB_BLOCKS,
           int data_size = NBEATS_DEFAULT_DATA_SIZE,
           int output_size = NBEATS_DEFAULT_OUTPUT_SIZE,
           int backcast_length = NBEATS_DEFAULT_BACKCAST_LENGTH,
           //           int forecast_length = NBEATS_DEFAULT_FORECAST_LENGTH,
           std::vector<int> thetas_dims = NBEATS_DEFAULT_THETAS,
           bool share_weights_in_stack = NBEATS_DEFAULT_SHARE_WEIGHTS,
           int hidden_layer_units = NBEATS_DEFAULT_HIDDEN_LAYER_UNITS)
        : _data_size(data_size), _output_size(output_size),
          _backcast_length(backcast_length),
          _hidden_layer_units(hidden_layer_units),
          _nb_blocks_per_stack(nb_blocks_per_stack),
          _share_weights_in_stack(share_weights_in_stack),
          _stack_types(stackTypes), _thetas_dims(thetas_dims)
    {
      parse_stackdef(stackdef);
      update_params(inputc);
      create_nbeats();
    }
    NBeats()
        : _data_size(1), _output_size(1),
          _backcast_length(NBEATS_DEFAULT_BACKCAST_LENGTH),
          //          _forecast_length(NBEATS_DEFAULT_FORECAST_LENGTH),
          _hidden_layer_units(NBEATS_DEFAULT_HIDDEN_LAYER_UNITS),
          _nb_blocks_per_stack(NBEATS_DEFAULT_NB_BLOCKS),
          _share_weights_in_stack(NBEATS_DEFAULT_SHARE_WEIGHTS),
          _stack_types(NBEATS_DEFAULT_STACK_TYPES),
          _thetas_dims(NBEATS_DEFAULT_THETAS)
    {
      create_nbeats();
    }

    NBeats(std::vector<BlockType> stackTypes, int nb_blocks_per_stack,
           int data_size, int output_size, int backcast_length,
           std::vector<int> thetas_dims, bool share_weights_in_stack,
           int hidden_layer_units)
        : _data_size(data_size), _output_size(output_size),
          _backcast_length(backcast_length),
          _hidden_layer_units(hidden_layer_units),
          _nb_blocks_per_stack(nb_blocks_per_stack),
          _share_weights_in_stack(share_weights_in_stack),
          _stack_types(stackTypes), _thetas_dims(thetas_dims)
    {
      create_nbeats();
    }

    void parse_stackdef(std::vector<std::string> stackdef)
    {
      if (stackdef.size() == 0)
        return;
      _stack_types.clear();
      _thetas_dims.clear();
      for (auto s : stackdef)
        {
          size_t pos = 0;
          if ((pos = s.find(_trend_str)) != std::string::npos)
            {
              _stack_types.push_back(trend);
              _thetas_dims.push_back(
                  std::stoi(s.substr(pos + _trend_str.size())));
            }
          else if ((pos = s.find(_season_str)) != std::string::npos)
            {
              _stack_types.push_back(seasonality);
              _thetas_dims.push_back(
                  std::stoi(s.substr(pos + _season_str.size())));
            }
          else if ((pos = s.find(_generic_str)) != std::string::npos)
            {
              _stack_types.push_back(generic);
              _thetas_dims.push_back(
                  std::stoi(s.substr(pos + _generic_str.size())));
            }
          else if ((pos = s.find(_nbblock_str)) != std::string::npos)
            {
              _nb_blocks_per_stack
                  = std::stoi(s.substr(pos + _nbblock_str.size()));
            }
          else if ((pos = s.find(_hsize_str)) != std::string::npos)
            {
              _hidden_layer_units
                  = std::stoi(s.substr(pos + _hsize_str.size()));
            }
          else
            {
              throw MLLibBadParamException(
                  "nbeats options is of the form ['t2', 's8', 'g3', 'b3'] for "
                  "1 trend stack of width 2 thetas, 1 seasonality stack of "
                  "width 8 thetas, 1 generic stack of width 3 thetas, 3 "
                  "blocks per stack ");
            }
        }
    }

    virtual torch::Tensor forward(torch::Tensor x);
    virtual torch::Tensor extract(torch::Tensor x, std::string extract_layer);

    virtual bool extractable(std::string extract_layer) const;
    virtual std::vector<std::string> extractable_layers() const;

    virtual torch::Tensor cleanup_output(torch::Tensor output)
    {
      return torch::chunk(output, 2, 0)[1].flatten(0, 1);
    }

    virtual torch::Tensor loss(std::string loss, torch::Tensor input,
                               torch::Tensor output, torch::Tensor target)
    {
      std::vector<torch::Tensor> chunks = torch::chunk(output, 2, 0);
      torch::Tensor x_pred = chunks[0].flatten(0, 1);
      torch::Tensor y_pred = chunks[1].flatten(0, 1);
      if (loss.empty() || loss == "L1" || loss == "l1")
        return torch::l1_loss(y_pred, target) + torch::l1_loss(x_pred, input);
      if (loss == "L2" || loss == "l2" || loss == "eucl")
        return torch::mse_loss(y_pred, target)
               + torch::mse_loss(x_pred, input);
      throw MLLibBadParamException("unknown loss " + loss);
    }

    virtual void update_input_connector(TorchInputInterface &inputc)
    {
      inputc._split_ts_for_predict = true;
    }

    virtual ~NBeats()
    {
    }

  protected:
    unsigned int _data_size = NBEATS_DEFAULT_DATA_SIZE;
    unsigned int _output_size = NBEATS_DEFAULT_OUTPUT_SIZE;
    unsigned int _backcast_length = NBEATS_DEFAULT_BACKCAST_LENGTH;
    //    unsigned int _forecast_length = NBEATS_DEFAULT_FORECAST_LENGTH;
    unsigned int _hidden_layer_units = NBEATS_DEFAULT_HIDDEN_LAYER_UNITS;
    unsigned int _nb_blocks_per_stack = NBEATS_DEFAULT_NB_BLOCKS;
    bool _share_weights_in_stack = NBEATS_DEFAULT_SHARE_WEIGHTS;
    std::vector<BlockType> _stack_types = NBEATS_DEFAULT_STACK_TYPES;
    std::vector<int> _thetas_dims = NBEATS_DEFAULT_THETAS;

    std::vector<Stack> _stacks;
    torch::nn::Linear _fcn{ nullptr };
    std::vector<float> _backcast_linspace;
    std::vector<float> _forecast_linspace;
    std::tuple<torch::Tensor, torch::Tensor> create_sin_basis(int thetas_dim);
    std::tuple<torch::Tensor, torch::Tensor> create_exp_basis(int thetas_dim);
    torch::Tensor _finit;
    void create_nbeats();
    void update_params(const CSVTSTorchInputFileConn &inputc);
    std::string _trend_str = "t";
    std::string _season_str = "s";
    std::string _generic_str = "g";
    std::string _nbblock_str = "b";
    std::string _hsize_str = "h";
  };
}
#endif
