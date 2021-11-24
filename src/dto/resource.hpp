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

#ifndef HTTP_DTO_RESOURCE_H
#define HTTP_DTO_RESOURCE_H

#include "dd_config.h"
#include "oatpp/core/Types.hpp"
#include "oatpp/core/macro/codegen.hpp"

#include "common.hpp"

namespace dd
{
  namespace DTO
  {
#include OATPP_CODEGEN_BEGIN(DTO) ///< Begin DTO codegen section

    class Resource : public oatpp::DTO
    {
      DTO_INIT(Resource, DTO);

      DTO_FIELD_INFO(type)
      {
        info->description = "Type of source. Available types: video.";
      }
      DTO_FIELD(String, type);

      DTO_FIELD_INFO(video_backend)
      {
        info->description
            = "For video sources, what backend is prefered for decoding";
      }
      DTO_FIELD(String, video_backend);

      DTO_FIELD_INFO(source)
      {
        info->description
            = "Source URI. Can be: stream url (youtube,...), video file, "
              "local device (camera...), custom gstreamer pipeline";
      }
      DTO_FIELD(String, source);
    };

    // OUTPUT

    class ResourceResponseHead : public oatpp::DTO
    {
      DTO_INIT(ResourceResponseHead, DTO)
    };

    class ResourceResponseBody : public oatpp::DTO
    {
      DTO_INIT(ResourceResponseBody, DTO)
    };

    class ResourceResponse : public oatpp::DTO
    {
      DTO_INIT(ResourceResponse, DTO)

      DTO_FIELD(String, dd_msg);
      DTO_FIELD(Object<Status>, status);
      DTO_FIELD(Object<ResourceResponseHead>, head);
      DTO_FIELD(Object<ResourceResponseBody>, body);
    };

#include OATPP_CODEGEN_END(DTO) ///< End DTO codegen section
  }
}

#endif // HTTP_DTO_RESOURCE_H
