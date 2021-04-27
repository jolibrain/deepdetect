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

#ifndef HTTP_DTO_INFO_H
#define HTTP_DTO_INFO_H

#include "dd_config.h"
#include "oatpp/core/Types.hpp"
#include "oatpp/core/macro/codegen.hpp"

namespace dd
{
  namespace DTO
  {
#include OATPP_CODEGEN_BEGIN(DTO) ///< Begin DTO codegen section

    class Service : public oatpp::DTO
    {
      DTO_INIT(Service, DTO /* extends */)

      DTO_FIELD(String, name);
      DTO_FIELD(String, description);
      DTO_FIELD(String, mllib);
      DTO_FIELD(String, mltype);
      DTO_FIELD(Boolean, predict) = false;
      DTO_FIELD(Boolean, training) = false;
    };

    class InfoHead : public oatpp::DTO
    {
      DTO_INIT(InfoHead, DTO /* extends */)
      DTO_FIELD(String, method) = "/info";

      // Why this is not in body ?
      DTO_FIELD(String, build_type, "build-type") = BUILD_TYPE;
      DTO_FIELD(String, version) = GIT_VERSION;
      DTO_FIELD(String, branch) = GIT_BRANCH;
      DTO_FIELD(String, commit) = GIT_COMMIT_HASH;
      DTO_FIELD(String, compile_flags) = COMPILE_FLAGS;
      DTO_FIELD(String, deps_version) = DEPS_VERSION;
      DTO_FIELD(List<Object<Service>>, services);
    };

    class InfoBody : public oatpp::DTO
    {
      DTO_INIT(InfoBody, DTO /* extends */)
    };

    class Status : public oatpp::DTO
    {
      DTO_INIT(Status, DTO /* extends */)
      DTO_FIELD(Int32, code);
      DTO_FIELD(String, msg);
      DTO_FIELD(Int32, dd_code);
      DTO_FIELD(String, dd_msg);
    };

    class InfoResponse : public oatpp::DTO
    {
      DTO_INIT(InfoResponse, DTO /* extends */)
      DTO_FIELD(String, dd_msg);
      DTO_FIELD(Object<Status>, status);
      DTO_FIELD(Object<InfoHead>, head);
      DTO_FIELD(Object<InfoBody>, body);
    };

#include OATPP_CODEGEN_END(DTO) ///< End DTO codegen section
  }
}

#endif // HTTP_DTO_INFO_H
