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

#ifndef TORCH_UTILS_H
#define TORCH_UTILS_H

#if !defined(CPU_ONLY)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <c10/cuda/CUDACachingAllocator.h>
#pragma GCC diagnostic pop
#endif

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <torch/torch.h>
#pragma GCC diagnostic pop
#include <torch/script.h>

namespace dd
{

  namespace torch_utils
  {
    inline void empty_cuda_cache()
    {
#if !defined(CPU_ONLY)
      c10::cuda::CUDACachingAllocator::emptyCache();
#endif
    }

    bool torch_write_proto_to_text_file(const google::protobuf::Message &proto,
                                        std::string filename);

    /// Convert IValue to Tensor and throw an exception if the IValue is not a
    /// Tensor.
    torch::Tensor to_tensor_safe(const c10::IValue &value);

    /// Convert id Tensor to one_hot Tensor
    void fill_one_hot(torch::Tensor &one_hot, torch::Tensor ids,
                      __attribute__((unused)) int nclasses);

    torch::Tensor to_one_hot(torch::Tensor ids, int nclasses);

    void add_parameters(std::shared_ptr<torch::jit::script::Module> module,
                        std::vector<torch::Tensor> &params,
                        bool requires_grad = true);
  }
}
#endif
