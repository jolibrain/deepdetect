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

#include "ddtypes.hpp"

namespace dd
{
  namespace DTO
  {
    namespace __class
    {
      const oatpp::ClassId APIDataClass::CLASS_ID("APIData");

      const oatpp::ClassId GpuIdsClass::CLASS_ID("GpuIds");

      const oatpp::ClassId ImageClass::CLASS_ID("Image");

      template <>
      const oatpp::ClassId DTOVectorClass<double>::CLASS_ID("vector<double>");

      template <>
      const oatpp::ClassId
          DTOVectorClass<uint8_t>::CLASS_ID("vector<uint8_t>");

      template <>
      const oatpp::ClassId DTOVectorClass<bool>::CLASS_ID("vector<bool>");

      template class DTOVectorClass<double>;
      template class DTOVectorClass<bool>;
    }
  }
}
