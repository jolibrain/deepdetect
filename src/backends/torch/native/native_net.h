#ifndef NATIVE_NET_H
#define NATIVE_NET_H

#include "torch/torch.h"
#include "templates/nbeats.h"

namespace dd
{
  
  typedef mapbox::util::variant<
    //NativeModule,
    std::shared_ptr<NBeats>
    > native_variant_type;
  
  class visitor_to_device
  {
  public:
    visitor_to_device() {}
    ~visitor_to_device() {}

    template <typename T> void operator()(T &nativem)
      {
	nativem->to(_device);
      }
    torch::Device _device = torch::DeviceType::CPU;
  };

  class visitor_to_type
  {
  public:
    visitor_to_type() {}
    ~visitor_to_type() {}

    template <typename T> void operator()(T &nativem)
      {
	nativem->to(_dtype);
      }
    torch::Dtype _dtype = torch::kFloat32;
  };

  class visitor_to_device_type
  {
  public:
    visitor_to_device_type() {}
    ~visitor_to_device_type() {}

    template <typename T> void operator()(T &nativem)
      {
	nativem->to(_device,_dtype);
      }
    torch::Device _device = torch::DeviceType::CPU;
    torch::Dtype _dtype = torch::kFloat32;
  };

  class visitor_forward
  {
  public:
    visitor_forward() {}
    ~visitor_forward() {}

    template <typename T> void operator()(T &nativem)
      {
	_out = nativem->forward(_source);
      }

    torch::Tensor _source;
    torch::Tensor _out;
  };

  class visitor_parameters
  {
  public:
    visitor_parameters() {}
    ~visitor_parameters() {}

    template <typename T> void operator()(T &nativem)
      {
	_params = nativem->parameters();
      }

    std::vector<torch::Tensor> _params;
  };

  class visitor_native_load
  {
  public:
    visitor_native_load() {}
    ~visitor_native_load() {}

    template <typename T> void operator()(T &nativem)
      {	
	torch::load(nativem, _fname);
	/*std::shared_ptr<torch::nn::Module> m = std::make_shared<torch::nn::Module>(nativem);
	  torch::load(m, _fname);*/
      }
    std::string _fname;
  };

  class visitor_native_save
  {
  public:
    visitor_native_save() {}
    ~visitor_native_save() {}

    template <typename T> void operator()(T &nativem)
      {	
	torch::save(nativem, _fname);
	/*std::shared_ptr<torch::nn::Module> m = std::make_shared<torch::nn::Module>(nativem);
	  torch::save(m, _fname);*/
      }
    std::string _fname;
  };

  class visitor_native_load_device
  {
  public:
    visitor_native_load_device() {}
    ~visitor_native_load_device() {}

    template <typename T> void operator()(T &nativem)
      {
	torch::load(nativem, _fname, _device);
      }

    std::string _fname;
    torch::Device _device = torch::DeviceType::CPU;
  };
  
  class visitor_native_eval
  {
  public:
    visitor_native_eval() {}
    ~visitor_native_eval() {}

    template <typename T> void operator()(T &nativem)
      {
	nativem->eval();
      }
  };

  class visitor_native_train
  {
  public:
    visitor_native_train() {}
    ~visitor_native_train() {}

    template <typename T> void operator()(T &nativem)
      {
	nativem->train();
      }
  };

  class visitor_native_loss
  {
  public:
    visitor_native_loss() {}
    ~visitor_native_loss() {}

    template <typename T> void operator()(T &nativem)
      {
	_loutput = nativem->loss(_loss, _input, _output, _target);
      }

    std::string _loss;
    torch::Tensor _input;
    torch::Tensor _output;
    torch::Tensor _target;
    torch::Tensor _loutput;
  };

  template<class TInputConnectorStrategy>
  class visitor_native_input_conn
  {
  public:
    visitor_native_input_conn() {}
    ~visitor_native_input_conn() {}

    template <typename T> void operator()(T &nativem)
      {
	nativem->update_input_connector(_inputc);
      }

    TInputConnectorStrategy _inputc;
  };

  class visitor_native_output
  {
  public:
    visitor_native_output() {}
    ~visitor_native_output() {}

    template <typename T> void operator()(T &nativem)
      {
	_output = nativem->cleanup_output(_output);
      }

    torch::Tensor _output;
  };
}  

#endif
