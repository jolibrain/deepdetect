/**
 * DeepDetect
 * Copyright (c) 2021 Jolibrain SASU
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

#ifndef DTO_COMMON_HPP
#define DTO_COMMON_HPP

#include "oatpp/core/Types.hpp"
#include "oatpp/core/macro/codegen.hpp"

namespace dd
{
  namespace DTO
  {
#include OATPP_CODEGEN_BEGIN(DTO) ///< Begin DTO codegen section

    class Status : public oatpp::DTO
    {
      DTO_INIT(Status, DTO /* extends */)
      DTO_FIELD(Int32, code);
      DTO_FIELD(String, msg);
      DTO_FIELD(Int32, dd_code);
      DTO_FIELD(String, dd_msg);
    };

    class GenericResponse : public oatpp::DTO
    {
      DTO_INIT(GenericResponse, DTO)

      DTO_FIELD(Object<Status>, status) = Status::createShared();
    };

    class BBox : public oatpp::DTO
    {
      DTO_INIT(BBox, oatpp::DTO)

      DTO_FIELD(Float64, xmin);
      DTO_FIELD(Float64, ymin);
      DTO_FIELD(Float64, xmax);
      DTO_FIELD(Float64, ymax);
    };

    class Dimensions : public oatpp::DTO
    {
      DTO_INIT(Dimensions, oatpp::DTO)

      DTO_FIELD(UInt32, width);
      DTO_FIELD(UInt32, height);
    };

#include OATPP_CODEGEN_END(DTO) ///< End DTO codegen section
  }
}

#endif // DTO_COMMON_HPP
