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

namespace dd
{
  namespace DTO
  {
#include OATPP_CODEGEN_BEGIN(DTO) ///< Begin DTO codegen section

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

      DTO_FIELD_INFO(datatype)
      {
        info->description = "fp16 or fp32";
      };

      DTO_FIELD(String, datatype) = "fp16";
    };
#include OATPP_CODEGEN_END(DTO) ///< End DTO codegen section

  }
}
#endif
