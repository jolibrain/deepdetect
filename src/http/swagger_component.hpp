/**
 * DeepDetect
 * Copyright (c) 2020 Jolibrain SASU
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

#ifndef HTTP_SWAGGERCOMPONENT_HPP
#define HTTP_SWAGGERCOMPONENT_HPP

#include <iostream>

#include "oatpp-swagger/Model.hpp"
#include "oatpp-swagger/Resources.hpp"
#include "oatpp/core/macro/component.hpp"

#include "dd_config.h"

class SwaggerComponent
{
public:
  OATPP_CREATE_COMPONENT(std::shared_ptr<oatpp::swagger::DocumentInfo>,
                         swaggerDocumentInfo)
  ([] {
    oatpp::swagger::DocumentInfo::Builder builder;

    std::ostringstream version;
    version << GIT_VERSION << " (" << BUILD_TYPE << ")";
    builder.setTitle("DeepDetect")
        .setDescription("DeepDetect REST API")
        .setVersion(version.str().c_str())
        .setContactName("Jolibrain")
        .setContactUrl("https://deepdetect.com/")

        .setLicenseName("GNU LESSER GENERAL PUBLIC LICENSE Version 3")
        .setLicenseUrl("https://www.gnu.org/licenses/lgpl-3.0.en.html");

    //.addServer("http://localhost:8000", "server on localhost");

    return builder.build();
  }());

  /**
   *  Swagger-Ui Resources (<oatpp-examples>/lib/oatpp-swagger/res)
   */
  OATPP_CREATE_COMPONENT(std::shared_ptr<oatpp::swagger::Resources>,
                         swaggerResources)
  ([] {
    // Make sure to specify correct full path to oatpp-swagger/res folder !!!
    return oatpp::swagger::Resources::loadResources(OATPP_SWAGGER_RES_PATH);
  }());
};

#endif /* HTTP_SWAGGERCOMPONENT_HPP */
