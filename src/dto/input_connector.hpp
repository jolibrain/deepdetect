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

#ifndef DTO_INPUT_CONNECTOR_HPP
#define DTO_INPUT_CONNECTOR_HPP

#include "dd_config.h"
#include "oatpp/core/Types.hpp"
#include "oatpp/core/macro/codegen.hpp"

namespace dd
{
  namespace DTO
  {
#include OATPP_CODEGEN_BEGIN(DTO)

    class InputConnector : public oatpp::DTO
    {
      DTO_INIT(InputConnector, DTO /* extends */)
      // Connector type
      DTO_FIELD(String, connector);

      // IMG Input Connector
      DTO_FIELD(Int32, width);
      DTO_FIELD(Int32, height);
      DTO_FIELD(Int32, crop_width);
      DTO_FIELD(Int32, crop_height);
      DTO_FIELD(Boolean, bw);
      DTO_FIELD(Boolean, rgb);
      DTO_FIELD(Boolean, histogram_equalization);
      DTO_FIELD(Boolean, unchanged_data);
      DTO_FIELD(Boolean, shuffle);
      DTO_FIELD(Int32, seed);
      DTO_FIELD(Float64, test_split);
      DTO_FIELD(Vector<Float32>, mean);
      DTO_FIELD(Vector<Float32>, std);
      DTO_FIELD(Any, scale); // bool for csv/csvts , float for img
      DTO_FIELD(Boolean, scaled);
      DTO_FIELD(Int32, scale_min);
      DTO_FIELD(Int32, scale_max);
      DTO_FIELD(Boolean, keep_orig);
      DTO_FIELD(String, interp);

      // image resizing on GPU
#ifdef USE_CUDA_CV
      DTO_FIELD(Boolean, cuda);
#endif
    };

#include OATPP_CODEGEN_END(DTO)
  }
}

#endif
