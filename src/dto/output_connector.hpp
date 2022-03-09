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

#ifndef DTO_OUTPUT_CONNECTOR_H
#define DTO_OUTPUT_CONNECTOR_H

#include "oatpp/core/Types.hpp"
#include "oatpp/core/macro/codegen.hpp"
#include "dto/ddtypes.hpp"

namespace dd
{
  namespace DTO
  {
#include OATPP_CODEGEN_BEGIN(DTO) ///< Begin DTO codegen section

    class OutputConnector : public oatpp::DTO
    {
      DTO_INIT(OutputConnector, DTO /* extends */)

      /* output supervised init */
      DTO_FIELD(Boolean, store_config) = false;

      /* output supervised predict */
      DTO_FIELD(Boolean, bbox) = false;
      DTO_FIELD(Boolean, ctc) = false;
      DTO_FIELD(Float32, confidence_threshold) = 0.0;
      DTO_FIELD(Int32, best);
      DTO_FIELD(Int32, best_bbox) = -1;
      DTO_FIELD(Float32, nms_threshold) = 0.45;
      DTO_FIELD(Vector<String>, confidences);
      DTO_FIELD(Int32, top_k) = -1;

      DTO_FIELD_INFO(image)
      {
        info->description = "wether to convert result to a cv::Mat (e.g. for "
                            "GANs or segmentation)";
      };
      DTO_FIELD(Boolean, image) = false;

      /* output unsupervised predict */
      DTO_FIELD_INFO(binarized)
      {
        info->description
            = "Output values are binarized and set either to 0.0 or 1.0";
      }
      DTO_FIELD(Boolean, binarized) = false;

      DTO_FIELD_INFO(bool_binarized)
      {
        info->description = "Output values as a bool vector";
      }
      DTO_FIELD(Boolean, bool_binarized) = false;

      DTO_FIELD_INFO(string_binarized)
      {
        info->description = "Output values as a string, eg 0001101...";
      }
      DTO_FIELD(Boolean, string_binarized) = false;

      /* simsearch (unsupervised) */
      DTO_FIELD(Boolean, index) = false;
      DTO_FIELD(Boolean, build_index) = false;
      DTO_FIELD(Boolean, search) = false;
      DTO_FIELD(Int32, search_nn);

      // model parameters
      DTO_FIELD(Int32, nprobe);
      DTO_FIELD(String, index_type);
      DTO_FIELD(Int32, train_samples);
      DTO_FIELD(Boolean, ondisk);
      DTO_FIELD(Boolean, index_gpu) = false;
      DTO_FIELD(GpuIds, index_gpuid);

      /* ncnn */
      DTO_FIELD(Int32, blank_label) = -1;

      /* TRT */
      DTO_FIELD(Boolean, regression) = false;
    };

#include OATPP_CODEGEN_END(DTO) ///< End DTO codegen section
  }
}

#endif
