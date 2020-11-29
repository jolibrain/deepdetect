#pragma once

#include <torch/cuda.h>
#include <torch/nn/module.h>
#include <torch/nn/pimpl.h>
#include <torch/types.h>

#include <torch/csrc/autograd/functions/comm.h>
#include <torch/csrc/autograd/functions/utils.h>
#include <ATen/core/functional.h>

#include <ATen/Device.h>
#include <ATen/Parallel.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Exception.h>

#include <cstddef>
#include <exception>
#include <memory>
#include <mutex>
#include <vector>

namespace dd
{
  namespace parallel
  {

    template <typename ModuleType>
    std::vector<ModuleType>
        _replicas; /**< holder of module replica pointers. */

    /**
     * \brief Customized torch data_parallel to avoid deep module replication
     * in two situations:
     *      - When the weights to be copied are already on the same device as
     * target.
     *      - When successive iterations do accumulate over the loss and no
     * weight update is required. This allows using `iter_size` to spend more
     * time on every GPU, accumulating the loss over large batches.
     */
    template <typename ModuleType>
    Tensor dd_data_parallel(ModuleType module, Tensor input,
                            optional<std::vector<Device>> devices = nullopt,
                            optional<Device> output_device = nullopt,
                            int64_t dim = 0, bool replicate = true)
    {
      if (!devices)
        {
          const auto device_count = torch::cuda::device_count();
          TORCH_CHECK(device_count > 0,
                      "Expected at least one CUDA device to be available");
          devices = std::vector<Device>();
          devices->reserve(device_count);
          for (size_t index = 0; index < device_count; ++index)
            {
              devices->emplace_back(kCUDA, index);
            }
        }
      if (!output_device)
        {
          output_device = devices->front();
        }

      if (devices->size() == 1)
        {
          module->to(devices->front());
          input = input.to(devices->front());
          return module->forward(std::move(input)).to(*output_device);
        }

      autograd::Scatter scatter(*devices, /*chunk_sizes=*/nullopt, dim);
      auto scattered_inputs
          = fmap<Tensor>(scatter.apply({ std::move(input) }));
      // Input tensor might not be big enough to scale across all available
      // devices
      if (scattered_inputs.size() < devices->size())
        {
          devices->resize(scattered_inputs.size(),
                          Device(DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES));
        }

      if (replicate)
        _replicas<ModuleType> = torch::nn::parallel::replicate(module,
                                                               *devices);

      auto outputs = torch::nn::parallel::parallel_apply(
          _replicas<ModuleType>, scattered_inputs, *devices);
      return autograd::Gather(*output_device, dim)
          .apply(fmap<autograd::Variable>(std::move(outputs)))
          .front();
    }

  } // namespace parallel
} // namespace dd
