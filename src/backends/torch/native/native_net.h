#ifndef NATIVE_NET_H
#define NATIVE_NET_H


#include "torch/torch.h"


namespace dd
{

  class NativeModule : public torch::nn::Module
  {
  public:
	virtual torch::Tensor forward(torch::Tensor x) = 0;
	virtual ~NativeModule() {}
  };
}




#endif
