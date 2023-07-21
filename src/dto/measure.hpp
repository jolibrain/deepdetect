/**
 * DeepDetect
 * Copyright (c) 2023 Jolibrain SASU
 * Author: Louis Jean <louis.jean@jolibrain.com>
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

#ifndef DTO_MEASURE_HPP
#define DTO_MEASURE_HPP

#include "dd_config.h"
#include "oatpp/core/Types.hpp"
#include "oatpp/core/macro/codegen.hpp"

namespace dd
{
  namespace DTO
  {
#include OATPP_CODEGEN_BEGIN(DTO) ///< Begin DTO codegen section

    class Measure : public oatpp::DTO
    {
    public:
      DTO_INIT(Measure, DTO /* extends */)

      DTO_FIELD_INFO(measure)
      {
        info->description = "Measure from the latest iteration.";
      }
      DTO_FIELD(APIData, measure);

      DTO_FIELD_INFO(measure_hist)
      {
        info->description
            = "Measure history during training. Used to make loss graphs etc";
      }
      DTO_FIELD(APIData, measure_hist);
    };

#include OATPP_CODEGEN_END(DTO) ///< End DTO codegen section
  }
}

#endif // DTO_MEASURE_HPP
