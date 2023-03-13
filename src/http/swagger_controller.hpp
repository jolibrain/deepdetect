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

#include "oatpp-swagger/Controller.hpp"

namespace dd
{
  class DedeSwaggerController : public oatpp::swagger::Controller
  {
  public:
    DedeSwaggerController(
        const std::shared_ptr<oatpp::data::mapping::ObjectMapper>
            &objectMapper,
        const oatpp::Object<oatpp::swagger::oas3::Document> &document,
        const std::shared_ptr<oatpp::swagger::Resources> &resources,
        const oatpp::swagger::ControllerPaths &paths)
        : oatpp::swagger::Controller(objectMapper, document, resources, paths)
    {
    }

  public:
#include OATPP_CODEGEN_BEGIN(ApiController)

    ENDPOINT("GET", "api-docs/oas-3.0.0.json", api2)
    {
      return api();
    }

#include OATPP_CODEGEN_END(ApiController)
  };
}
