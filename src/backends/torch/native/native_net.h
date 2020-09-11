#ifndef NATIVE_NET_H
#define NATIVE_NET_H

#include "torch/torch.h"

namespace dd
{

  class NativeModule : public torch::nn::Module
  {
  public:
    virtual torch::Tensor forward(torch::Tensor x) = 0;
    virtual ~NativeModule()
    {
    }
    /**
     * \brief see torch::module::to
     * @param device cpu / gpu
     * @param non_blocking
     */
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

    virtual torch::Tensor cleanup_output(torch::Tensor output)
    {
      return output;
    }

    virtual torch::Tensor loss(std::string loss, torch::Tensor input,
                               torch::Tensor output, torch::Tensor target)
        = 0;

    virtual void update_input_connector(TorchInputInterface &inputc)
    {
      (void)(inputc);
    }

  protected:
    torch::Dtype _dtype
        = torch::kFloat32; /**< type of data stored in tensors */
    torch::Device _device
        = torch::DeviceType::CPU; /**< device to compute on */
  };
}

#endif
