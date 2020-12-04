/**
 * DeepDetect
 * Copyright (c) 2019-2020 Jolibrain
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

#ifndef NATIVE_NET_H
#define NATIVE_NET_H

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <torch/torch.h>
#pragma GCC diagnostic pop

#include "../torchinputconns.h"

namespace dd
{

  class NativeModule : public virtual torch::nn::Module
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

    virtual ~NativeModule() = default;

    virtual torch::Tensor cleanup_output(torch::Tensor output) = 0;

    /**
     * \brief compute custom loss
     */
    virtual torch::Tensor loss(std::string loss, torch::Tensor input,
                               torch::Tensor output, torch::Tensor target)
        = 0;
  };

  template <typename T>
  class NativeModuleImpl : public NativeModule, public torch::nn::Cloneable<T>
  {
  };
}

#endif
