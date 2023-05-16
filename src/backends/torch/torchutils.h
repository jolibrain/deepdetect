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

#if !defined(CPU_ONLY) && !defined(USE_MPS)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <c10/cuda/CUDACachingAllocator.h>
#pragma GCC diagnostic pop
#endif

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <torch/torch.h>
#include <torch/script.h>
#pragma GCC diagnostic pop

#include <google/protobuf/message.h>
#include "dd_spdlog.h"
#include <opencv2/opencv.hpp>

namespace dd
{

  namespace torch_utils
  {

    inline void cerr_tensor_shape(const std::string &tname,
                                  const torch::Tensor t)
    {
      std::cerr << tname << " shape=";
      for (auto d : t.sizes())
        std::cerr << d << " ";
      std::cerr << std::endl;
    }

    /**
     * \brief empty cuda caching allocator, This should be called after every
     * job that allocates a model. Pytorch keeps cuda memory allocated even
     * after the deletion of tensors, to avoid reallocating later. Sometimes it
     * means reserving all the GPU ram even if training is over.
     */
    inline void free_gpu_memory()
    {
#if !defined(CPU_ONLY) && !defined(USE_MPS)
      c10::cuda::CUDACachingAllocator::emptyCache();
#endif
    }

    inline bool is_gpu_available()
    {
#if defined(USE_MPS)
      return true;
#elif !defined(CPU_ONLY)
      return torch::cuda::is_available();
#else
      return false;
#endif
    }

    inline int get_gpu_count()
    {
#if defined(USE_MPS)
      return 1;
#elif !defined(CPU_ONLY)
      return torch::cuda::device_count();
#else
      return 0;
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

    /** Copy weights from a native module to another native module. This is
     * used in multigpu settings. */
    void copy_native_weights(const torch::nn::Module &from,
                             torch::nn::Module &to,
                             const torch::Device &device);

    /** Load weights from a model file to a native module.
     * \param strict if false, some weights are allowed to mismatch, or be
     * missing in either copy source or copy destination. If true, an exception
     * will be thrown.
     */
    void load_weights(torch::nn::Module &module, const std::string &filename,
                      const torch::Device &device,
                      std::shared_ptr<spdlog::logger> logger = nullptr,
                      bool strict = false);

    /** Converts a tensor to a CV image that can be saved on the disk.
     * XXX(louis) this function is currently debug only, and makes strong
     * assumptions on the input tensor format. */
    cv::Mat tensorToImage(torch::Tensor tensor);
  }
}
#endif
