#ifndef NBEATS_H
#define NBEATS_H

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "torch/torch.h"
#pragma GCC diagnostic pop
#include "../../torchinputconns.h"
#include "mllibstrategy.h"
#include "../native_net.h"

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
                int forecast_length, bool share_thetas,
                unsigned int data_size = 1)
          : _units(units), _thetas_dim(thetas_dim), _data_size(data_size),
            _backcast_length(backcast_length),
            _forecast_length(forecast_length), _share_thetas(share_thetas)
      {
        init_block();
      }

      BlockImpl(BlockImpl &b)
          : torch::nn::Module(b), _units(b._units), _thetas_dim(b._thetas_dim),
            _data_size(b._data_size), _backcast_length(b._backcast_length),
            _forecast_length(b._forecast_length),
            _share_thetas(b._share_thetas)
      {
        init_block();
      }

      torch::Tensor first_forward(torch::Tensor x);

      virtual void to(torch::Device device, bool non_blocking = false)
      {
        torch::nn::Module::to(device, non_blocking);
        _device = device;
      }

      /**
       * \brief see torch::module::to
       * @param dtype : torch::kFloat32 or torch::kFloat64
       * @param non_blocking
       */
      virtual void to(torch::Dtype dtype, bool non_blocking = false)
      {
        torch::nn::Module::to(dtype, non_blocking);
        _dtype = dtype;
      }

      /**
       * \brief see torch::module::to
       * @param device cpu / gpu
       * @param dtype : torch::kFloat32 or torch::kFloat64
       * @param non_blocking
       */
      virtual void to(torch::Device device, torch::Dtype dtype,
                      bool non_blocking = false)
      {
        torch::nn::Module::to(device, dtype, non_blocking);
        _device = device;
        _dtype = dtype;
      }

    protected:
      void init_block();

      unsigned int _units;
      unsigned int _thetas_dim;
      unsigned int _data_size;
      unsigned int _backcast_length;
      unsigned int _forecast_length;
      bool _share_thetas;
      torch::nn::Linear _fc1{ nullptr };
      torch::nn::Linear _fc2{ nullptr };
      torch::nn::Linear _fc3{ nullptr };
      torch::nn::Linear _fc4{ nullptr };
      torch::nn::Linear _theta_b_fc{ nullptr };
      torch::nn::Linear _theta_f_fc{ nullptr };
      torch::Dtype _dtype = torch::kFloat32;
      torch::Device _device = torch::Device("cpu");
    };

    typedef torch::nn::ModuleHolder<BlockImpl> Block;

    class SeasonalityBlockImpl : public BlockImpl
    {
    public:
      SeasonalityBlockImpl(int units, int thetas_dim, int backcast_length,
                           int forecast_length, int data_size,
                           torch::Tensor bS, torch::Tensor fS)
          : BlockImpl(units, thetas_dim, backcast_length, forecast_length,
                      true, data_size),
            _bS(bS), _fS(fS)
      {
      }

      SeasonalityBlockImpl(SeasonalityBlockImpl &b)
          : BlockImpl(b), _bS(b._bS), _fS(b._fS)
      {
      }

      std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x);

    protected:
      torch::Tensor _bS, _fS;
    };

    typedef torch::nn::ModuleHolder<SeasonalityBlockImpl> SeasonalityBlock;

    class TrendBlockImpl : public BlockImpl
    {
    public:
      TrendBlockImpl(int units, int thetas_dim, int backcast_length,
                     int forecast_length, int data_size, torch::Tensor bT,
                     torch::Tensor fT)
          : BlockImpl(units, thetas_dim, backcast_length, forecast_length,
                      true, data_size),
            _bT(bT), _fT(fT)
      {
      }

      TrendBlockImpl(TrendBlockImpl &b) : BlockImpl(b), _bT(b._bT), _fT(b._fT)
      {
      }

      std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x);

    protected:
      torch::Tensor _bT, _fT;
    };

    typedef torch::nn::ModuleHolder<TrendBlockImpl> TrendBlock;

    class GenericBlockImpl : public BlockImpl
    {
    public:
      GenericBlockImpl(int units, int thetas_dim, int backcast_length,
                       int forecast_length, int data_size)
          : BlockImpl(units, thetas_dim, backcast_length, forecast_length,
                      false, data_size)
      {
        _backcast_fc = register_module(
            "backcast_fc", torch::nn::Linear(_thetas_dim * _data_size,
                                             _backcast_length * _data_size));
        _forecast_fc = register_module(
            "forecast_fc", torch::nn::Linear(_thetas_dim * _data_size,
                                             _forecast_length * _data_size));
      }

      GenericBlockImpl(GenericBlockImpl &b) : BlockImpl(b)
      {
        _backcast_fc = b._backcast_fc;
        _forecast_fc = b._forecast_fc;
      }

      std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x);

    protected:
      torch::nn::Linear _backcast_fc{ nullptr };
      torch::nn::Linear _forecast_fc{ nullptr };
    };

    typedef torch::nn::ModuleHolder<GenericBlockImpl> GenericBlock;

    typedef std::vector<torch::nn::AnyModule> Stack;

  public:
    NBeats(const CSVTSTorchInputFileConn &inputc,
           std::vector<std::string> stackdef,
           std::vector<BlockType> stackTypes = { trend, seasonality, generic },
           int nb_blocks_per_stack = 3, int data_size = 1, int output_size = 1,
           int backcast_length = 50, int forecast_length = 10,
           std::vector<int> thetas_dims = { 2, 8, 3 },
           bool share_weights_in_stack = false, int hidden_layer_units = 10)
        : _data_size(data_size), _output_size(output_size),
          _backcast_length(backcast_length), _forecast_length(forecast_length),
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
        : _data_size(1), _output_size(1), _backcast_length(50),
          _forecast_length(10), _hidden_layer_units(1024),
          _nb_blocks_per_stack(3), _share_weights_in_stack(false),
          _stack_types({ trend, seasonality, generic }),
          _thetas_dims({ 2, 8, 3 })
    {
      create_nbeats();
    }

    NBeats(std::vector<BlockType> stackTypes, int nb_blocks_per_stack,
           int data_size, int output_size, int backcast_length,
           int forecast_length, std::vector<int> thetas_dims,
           bool share_weights_in_stack, int hidden_layer_units)
        : _data_size(data_size), _output_size(output_size),
          _backcast_length(backcast_length), _forecast_length(forecast_length),
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
                  = std::stoi(s.substr(pos + _generic_str.size()));
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

    virtual void to(torch::Device device, bool non_blocking = false)
    {
      torch::nn::Module::to(device, non_blocking);
      _device = device;
      for (unsigned int i = 0; i < _stack_types.size(); ++i)
        {
          Stack s = _stacks[i];
          switch (_stack_types[i])
            {
            case seasonality:
              for (auto b : s)
                b.get<SeasonalityBlock>()->to(device);
              break;
            case trend:
              for (auto b : s)
                b.get<TrendBlock>()->to(device);
              break;
            case generic:
              for (auto b : s)
                b.get<GenericBlock>()->to(device);
              break;
            default:
              break;
            }
        }
      _fcn->to(device);
    }

    /**
     * \brief see torch::module::to
     * @param dtype : torch::kFloat32 or torch::kFloat64
     * @param non_blocking
     */
    virtual void to(torch::Dtype dtype, bool non_blocking = false)
    {
      torch::nn::Module::to(dtype, non_blocking);
      _dtype = dtype;
      for (unsigned int i = 0; i < _stack_types.size(); ++i)
        {
          Stack s = _stacks[i];
          switch (_stack_types[i])
            {
            case seasonality:
              for (auto b : s)
                b.get<SeasonalityBlock>()->to(dtype);
              break;
            case trend:
              for (auto b : s)
                b.get<TrendBlock>()->to(dtype);
              break;
            case generic:
              for (auto b : s)
                b.get<GenericBlock>()->to(dtype);
              break;
            default:
              break;
            }
        }
      _fcn->to(dtype);
    }

    /**
     * \brief see torch::module::to
     * @param device cpu / gpu
     * @param dtype : torch::kFloat32 or torch::kFloat64
     * @param non_blocking
     */
    virtual void to(torch::Device device, torch::Dtype dtype,
                    bool non_blocking = false)
    {
      torch::nn::Module::to(device, dtype, non_blocking);
      _device = device;
      _dtype = dtype;
      for (unsigned int i = 0; i < _stack_types.size(); ++i)
        {
          Stack s = _stacks[i];
          switch (_stack_types[i])
            {
            case seasonality:
              for (auto b : s)
                b.get<SeasonalityBlock>()->to(device, dtype);
              break;
            case trend:
              for (auto b : s)
                b.get<TrendBlock>()->to(device, dtype);
              break;
            case generic:
              for (auto b : s)
                b.get<GenericBlock>()->to(device, dtype);
              break;
            default:
              break;
            }
        }
      _fcn->to(device, dtype);
    }

    virtual torch::Tensor forward(torch::Tensor x);

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
    unsigned int _data_size;
    unsigned int _output_size;
    unsigned int _backcast_length;
    unsigned int _forecast_length;
    unsigned int _hidden_layer_units;
    unsigned int _nb_blocks_per_stack;
    bool _share_weights_in_stack;
    std::vector<BlockType> _stack_types;
    std::vector<Stack> _stacks;
    std::vector<int> _thetas_dims;
    torch::nn::Linear _fcn{ nullptr };
    torch::Device _device = torch::Device("cpu");
    std::vector<float> _backcast_linspace;
    std::vector<float> _forecast_linspace;
    std::tuple<torch::Tensor, torch::Tensor> create_sin_basis(int thetas_dim);
    std::tuple<torch::Tensor, torch::Tensor> create_exp_basis(int thetas_dim);
    void create_nbeats();
    void update_params(const CSVTSTorchInputFileConn &inputc);
    std::string _trend_str = "t";
    std::string _season_str = "s";
    std::string _generic_str = "g";
    std::string _nbblock_str = "b";
  };
}
#endif
