/**
 * DeepDetect
 * Copyright (c) 2021 Jolibrain SASU
 * Author: Mehdi Abaakouk <mehdi.abaakouk@jolibrain.com>
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

#ifndef DTO_MLLIB_H
#define DTO_MLLIB_H

#include "dd_config.h"
#include "utils/utils.hpp"
#include "oatpp/core/Types.hpp"
#include "oatpp/core/macro/codegen.hpp"
#include "ddtypes.hpp"

namespace dd
{
  namespace DTO
  {
#include OATPP_CODEGEN_BEGIN(DTO) ///< Begin DTO codegen section

    class Net : public oatpp::DTO
    {
      DTO_INIT(Net, DTO)

      DTO_FIELD_INFO(test_batch_size)
      {
        info->description = "Testing batch size";
      }
      DTO_FIELD(Int32, test_batch_size) = 1;
    };

    class MLLib : public oatpp::DTO
    {
      DTO_INIT(MLLib, DTO /* extends */)

      // General Options
      DTO_FIELD_INFO(nclasses)
      {
        info->description = "number of output classes (`supervised` service "
                            "type), classification only";
      };
      DTO_FIELD(Int32, nclasses) = 0;

      DTO_FIELD_INFO(ntargets)
      {
        info->description
            = "number of regression targets (`supervised` service "
              "type), regression only";
      };
      DTO_FIELD(Int32, ntargets) = 0;

      DTO_FIELD_INFO(segmentation)
      {
        info->description = "whether the model type is segmentation";
      };
      DTO_FIELD(Boolean, segmentation) = false;

      DTO_FIELD_INFO(ctc)
      {
        info->description = "whether the model type is ctc";
      };
      DTO_FIELD(Boolean, ctc) = false;

      DTO_FIELD_INFO(from_repository)
      {
        info->description = "initialize model repository with checkpoint and "
                            "configuration from another repository";
      };
      DTO_FIELD(String, from_repository);

      DTO_FIELD_INFO(model_template)
      {
        info->description = "Model template";
      };
      DTO_FIELD(String, model_template, "template") = "";

      DTO_FIELD_INFO(gpu)
      {
        info->description = "Whether to use GPU";
      };
      DTO_FIELD(Boolean, gpu) = false;

      DTO_FIELD_INFO(gpuid)
      {
        info->description
            = "GPU id, use single int for single GPU, -1 for using all GPUs, "
              "and array e.g. [1,3] for selecting among multiple GPUs";
      };
      DTO_FIELD(GpuIds, gpuid) = { VGpuIds{} };

      DTO_FIELD_INFO(finetuning)
      {
        // XXX: torch does not require weights since the weights are in the
        // repo. But other mllib require weights (eg caffe).
        info->description = "Whether to prepare neural net template for "
                            "finetuning (requires `weights`)";
      };
      DTO_FIELD(Boolean, finetuning) = false;

      DTO_FIELD_INFO(datatype)
      {
        info->description
            = "Datatype used at prediction time. fp16 or fp32 or fp64 (torch)";
      };
      DTO_FIELD(String, datatype) = "fp32";

      DTO_FIELD_INFO(extract_layer)
      {
        info->description
            = "Returns tensor values from an intermediate layer. If set to "
              "'last', returns the values from last layer.";
      }
      DTO_FIELD(String, extract_layer) = "";

      DTO_FIELD(Object<Net>, net) = Net::createShared();

      // =====
      // Libtorch options
      DTO_FIELD_INFO(self_supervised)
      {
        info->description
            = "self-supervised mode: “mask” for masked language model";
      };
      DTO_FIELD(String, self_supervised) = "";

      DTO_FIELD_INFO(embedding_size)
      {
        info->description = "embedding size for NLP models";
      };
      DTO_FIELD(Int32, embedding_size) = 768;

      DTO_FIELD_INFO(freeze_traced)
      {
        info->description = "Freeze the traced part of the net during "
                            "finetuning (e.g. for classification)";
      };
      DTO_FIELD(Boolean, freeze_traced) = false;

      DTO_FIELD_INFO(loss)
      {
        // TODO add other losses.
        info->description
            = "Special network losses. (e.g. L1 & L2 for timeseries)";
      };
      DTO_FIELD(String, loss) = "";

      DTO_FIELD_INFO(timesteps)
      {
        info->description
            = "Number of output timesteps eg for ctc. -1 for auto mode.";
      }
      DTO_FIELD(Int32, timesteps) = -1;

      // TODO template parameters depends on the template, so the DTO must be
      // custom
      DTO_FIELD_INFO(template_params)
      {
        info->description = "Model parameter for templates. All parameters "
                            "are listed in the Model Templates section.";
      };
      DTO_FIELD(UnorderedFields<Any>, template_params);

      // Libtorch predict options
      DTO_FIELD_INFO(forward_method)
      {
        info->description
            = "Executes a custom function from within a traced/JIT model, "
              "instead of the standard forward()";
      }
      DTO_FIELD(String, forward_method) = "";

      // =====
      // NCNN Options
      DTO_FIELD_INFO(threads)
      {
        info->description = "number of threads";
      };
      DTO_FIELD(Int32, threads) = dd::dd_utils::my_hardware_concurrency();

      DTO_FIELD_INFO(lightmode)
      {
        info->description = "enable light mode";
      };
      DTO_FIELD(Boolean, lightmode) = true;

      DTO_FIELD_INFO(inputBlob)
      {
        info->description = "network input blob name";
      };
      DTO_FIELD(String, inputBlob) = "data";

      DTO_FIELD_INFO(outputBlob)
      {
        info->description = "network output blob name (default depends on "
                            "network type(ie prob or "
                            "rnn_pred or probs or detection_out)";
      };
      DTO_FIELD(String, outputBlob);
    };
#include OATPP_CODEGEN_END(DTO) ///< End DTO codegen section

  }
}
#endif
