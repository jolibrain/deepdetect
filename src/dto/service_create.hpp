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

#ifndef DTO_SERVICE_CREATE_H
#define DTO_SERVICE_CREATE_H

#include "oatpp/core/Types.hpp"
#include "oatpp/core/macro/codegen.hpp"
#include "dto/model.hpp"
#include "dto/parameters.hpp"

namespace dd
{
  namespace DTO
  {
#include OATPP_CODEGEN_BEGIN(DTO) ///< Begin DTO codegen section

class ServiceCreate : public oatpp::DTO
{
  DTO_INIT(ServiceCreate, DTO /* extends */)

  DTO_FIELD(String, mllib);
  DTO_FIELD(String, description) = "";
  DTO_FIELD(String, type) = "supervised";
  DTO_FIELD(Object<Parameters>, parameters) = Parameters::createShared();
  DTO_FIELD(Object<Model>, model) = Model::createShared();
};

#include OATPP_CODEGEN_END(DTO) ///< End DTO codegen section
}
}

#endif
