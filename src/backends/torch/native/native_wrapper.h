/**
 * DeepDetect
 * Copyright (c) 2020 Jolibrain
 * Author:  Louis Jean <louis.jean@jolibrain.com>
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

#ifndef DD_NATIVE_WRAPPER_H
#define DD_NATIVE_WRAPPER_H

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <torch/torch.h>
#pragma GCC diagnostic pop

#include "native_net.h"

namespace dd
{
  template <typename TModule>
  class NativeModuleWrapper
      : public NativeModuleImpl<NativeModuleWrapper<TModule>>
  {
  public:
    TModule _module;

    template <typename... Args>
    NativeModuleWrapper(Args &&... args) : _module(args...)
    {
      this->register_module("wrapped", _module);
    }

    void reset() override
    {
      this->register_module("wrapped", _module);
    }

    torch::Tensor forward(torch::Tensor x) override
    {
      return _module->forward(x);
    }

    torch::Tensor extract(__attribute__((unused)) torch::Tensor x,
                          __attribute__((unused))
                          std::string extract_layer) override
    {
      throw std::runtime_error(
          "extract() not implemented for NativeModuleWrapper");
    }

    bool extractable(__attribute__((unused))
                     std::string extract_layer) const override
    {
      return false;
    }

    std::vector<std::string> extractable_layers() const override
    {
      return {};
    }

    torch::Tensor cleanup_output(torch::Tensor output) override
    {
      return output;
    }

    torch::Tensor loss(__attribute__((unused)) std::string loss,
                       __attribute__((unused)) torch::Tensor input,
                       __attribute__((unused)) torch::Tensor output,
                       __attribute__((unused)) torch::Tensor target) override
    {
      throw MLLibInternalException(
          "NativeModuleWrapper::loss not implemented");
    }
  };
}

#endif // DD_NATIVE_WRAPPER_H
