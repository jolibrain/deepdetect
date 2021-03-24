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

#include <google/protobuf/message.h>

namespace dd
{

  namespace torch_utils
  {

    /**
     * \brief empty cuda caching allocator, This should be called after every
     * job that allocates a model. Pytorch keeps cuda memory allocated even
     * after the deletion of tensors, to avoid reallocating later. Sometimes it
     * means reserving all the GPU ram even if training is over.
     */
    inline void empty_cuda_cache()
    {
#if !defined(CPU_ONLY)
      c10::cuda::CUDACachingAllocator::emptyCache();
#endif
    }

    /**
     * \brief torch version of writing a (caffe) protobu to file
     * @return true on sucess
     */
    bool torch_write_proto_to_text_file(const google::protobuf::Message &proto,
                                        std::string filename);

    /**
     * \brief Convert IValue to Tensor and throw an exception if the IValue is
     * not a Tensor.
     */
    torch::Tensor to_tensor_safe(const c10::IValue &value);

    /**
     * \brief convert vector of int to tensor
     */
    torch::Tensor toLongTensor(std::vector<int64_t> &values);

    /**
     * \brief  Convert id Tensor to one_hot Tensor (on already allocated
     * tensor)
     */
    void fill_one_hot(torch::Tensor &one_hot, torch::Tensor ids,
                      __attribute__((unused)) int nclasses);

    /**
     * \brief  Convert id Tensor to one_hot Tensor (allocates tensor)
     */
    torch::Tensor to_one_hot(torch::Tensor ids, int nclasses);

    /**
     * \brief recursively add module parameters to params
     * \param requires_grad if true, collect only parameters that
     * require grad (method Tensor::requires_grad())
     */
    void add_parameters(std::shared_ptr<torch::jit::script::Module> module,
                        std::vector<torch::Tensor> &params,
                        bool requires_grad = true);

    std::vector<c10::IValue> unwrap_c10_vector(const c10::IValue &output);

    /** Copy weights from a torchscript module to a native module.
     * \param strict if false, some weights are allowed to mismatch, or be
     * missing in either copy source or copy destination. If true, an exception
     * will be thrown.
     */
    void copy_weights(const torch::jit::script::Module &from,
                      torch::nn::Module &to, const torch::Device &device,
                      std::shared_ptr<spdlog::logger> logger = nullptr,
                      bool strict = false);

    /** Load weights from a model file to a native module.
     * \param strict if false, some weights are allowed to mismatch, or be
     * missing in either copy source or copy destination. If true, an exception
     * will be thrown.
     */
    void load_weights(torch::nn::Module &module, const std::string &filename,
                      const torch::Device &device,
                      std::shared_ptr<spdlog::logger> logger = nullptr,
                      bool strict = false);
  }
}
#endif
