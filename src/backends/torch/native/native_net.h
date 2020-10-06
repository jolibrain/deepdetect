#ifndef NATIVE_NET_H
#define NATIVE_NET_H

#include "torch/torch.h"

namespace dd
{

  class NativeModule : public torch::nn::Module
  {
  public:
    /**
     * \brief forward pass over the
     * @param input tensor
     * @return value of output
     */
    virtual torch::Tensor forward(torch::Tensor x) = 0;

    /**
     * \brief extract layer from net
     * @param input
     * @param name of data to extract
     * @return extracted tensor
     */
    virtual torch::Tensor extract(torch::Tensor x, std::string extract_layer)
        = 0;

    /**
     * \brief check is string correspond to some layer in the net
     * @param the name of the data node
     * @return true if it exists in the net
     */
    virtual bool extractable(std::string extract_layer) const = 0;

    /**
     * \brief return all candidates for extraction, ie all data nodes of the
     * net
     */
    virtual std::vector<std::string> extractable_layers() const = 0;

    virtual ~NativeModule()
    {
    }

    virtual torch::Tensor cleanup_output(torch::Tensor output) = 0;

    virtual torch::Tensor loss(std::string loss, torch::Tensor input,
                               torch::Tensor output, torch::Tensor target)
        = 0;

    virtual void update_input_connector(TorchInputInterface &inputc) = 0;
  };
}

#endif
