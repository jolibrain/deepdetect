/**
 * DeepDetect
 * Copyright (c) 2021 Jolibrain SASU
 * Author: Emmanuel Benazera <emmanuel.benazera@jolibrain.com>
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

#ifndef HTTP_DTO_SVM_CONNECTOR_HPP
#define HTTP_DTO_SVM_CONNECTOR_HPP

#include "dd_config.h"
#include "oatpp/core/Types.hpp"
#include "oatpp/core/macro/codegen.hpp"

namespace dd
{
  namespace DTO
  {
#include OATPP_CODEGEN_BEGIN(DTO)

    class SVMInputConnectorParameters : public oatpp::DTO
    {
      DTO_INIT(SVMInputConnectorParameters, DTO /* extends */)

      DTO_FIELD(Float64, test_split);
      DTO_FIELD(Boolean, shuffle);
      DTO_FIELD(Int32, seed);
    };

#include OATPP_CODEGEN_END(DTO)
  }
}

#endif
