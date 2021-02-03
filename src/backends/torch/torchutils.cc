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

#include "torchutils.h"
#include "mllibstrategy.h"

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <fcntl.h>
#include <unordered_set>

using google::protobuf::io::CodedInputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::ZeroCopyOutputStream;

namespace dd
{
  namespace torch_utils
  {
    bool torch_write_proto_to_text_file(const google::protobuf::Message &proto,
                                        std::string filename)
    {
      int fd = open(filename.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
      if (fd == -1)
        return false;
      FileOutputStream *output = new FileOutputStream(fd);
      bool success = google::protobuf::TextFormat::Print(proto, output);
      delete output;
      close(fd);
      return success;
    }

    torch::Tensor to_tensor_safe(const c10::IValue &value)
    {
      if (!value.isTensor())
        throw MLLibInternalException("Expected Tensor, found "
                                     + value.tagKind());
      return value.toTensor();
    }

    void fill_one_hot(torch::Tensor &one_hot, torch::Tensor ids,
                      __attribute__((unused)) int nclasses)
    {
      one_hot.zero_();
      for (int i = 0; i < ids.size(0); ++i)
        {
          one_hot[i][ids[i].item<int>()] = 1;
        }
    }

    torch::Tensor to_one_hot(torch::Tensor ids, int nclasses)
    {
      torch::Tensor one_hot
          = torch::zeros(torch::IntList{ ids.size(0), nclasses });
      for (int i = 0; i < ids.size(0); ++i)
        {
          one_hot[i][ids[i].item<int>()] = 1;
        }
      return one_hot;
    }

    void add_parameters(std::shared_ptr<torch::jit::script::Module> module,
                        std::vector<torch::Tensor> &params, bool requires_grad)
    {
      for (const auto &tensor : module->parameters())
        {
          if (tensor.requires_grad() || !requires_grad)
            params.push_back(tensor);
        }
      for (auto child : module->children())
        {
          add_parameters(std::make_shared<torch::jit::script::Module>(child),
                         params, requires_grad);
        }
    }

    torch::Tensor toLongTensor(std::vector<int64_t> &values)
    {
      int64_t val_size = values.size();
      return torch::from_blob(&values[0], at::IntList{ val_size }, at::kLong)
          .clone();
    }

    std::vector<c10::IValue> unwrap_c10_vector(const c10::IValue &output)
    {
      if (output.isTensorList())
        {
          auto elems = output.toTensorList();
          return std::vector<c10::IValue>(elems.begin(), elems.end());
        }
      else if (output.isTuple())
        {
          auto &elems = output.toTuple()->elements();
          return std::vector<c10::IValue>(elems.begin(), elems.end());
        }
      else
        {
          return { output };
        }
    }

    void copy_weights(const torch::jit::script::Module &from,
                      torch::nn::Module &to, const torch::Device &device,
                      std::shared_ptr<spdlog::logger> logger)
    {
      auto from_params = from.named_parameters();
      auto to_params = to.named_parameters();
      std::unordered_set<std::string> copied_params;

      for (const auto &from_item : from_params)
        {
          torch::Tensor *to_value_ptr = to_params.find(from_item.name);

          if (to_value_ptr == nullptr)
            {
              if (logger)
                {
                  logger->warn("skipped " + from_item.name
                               + ": not found in destination module");
                }
              continue;
            }
          torch::Tensor &to_value = *to_value_ptr;

          if (from_item.value.sizes() != to_value.sizes())
            {
              if (logger)
                {
                  std::stringstream sstream;
                  sstream << "skipped " << from_item.name
                          << ": cannot copy tensor of size "
                          << from_item.value.sizes() << " into tensor of size "
                          << to_value.sizes();
                  logger->warn(sstream.str());
                }
              continue;
            }

          to_value.set_data(from_item.value.to(device));
          copied_params.insert(from_item.name);
          if (logger)
            logger->info("copied " + from_item.name);
        }

      if (copied_params.empty())
        {
          throw MLLibBadParamException(
              "No weights were copied: models do not match.");
        }

      for (const auto &param_name : to_params.keys())
        {
          if (copied_params.find(param_name) == copied_params.end())
            {
              logger->warn(param_name + " was not found in source module.");
            }
        }
    }

    void load_weights(torch::nn::Module &module, const std::string &filename,
                      const torch::Device &device,
                      std::shared_ptr<spdlog::logger> logger)
    {
      auto jit_module = torch::jit::load(filename, device);
      torch_utils::copy_weights(jit_module, module, device, logger);
    }
  }
}
