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

#ifndef HTTP_DTO_STREAM_H
#define HTTP_DTO_STREAM_H

#include "dd_config.h"
#include "oatpp/core/Types.hpp"
#include "oatpp/core/macro/codegen.hpp"

#include "common.hpp"

namespace dd
{
  namespace DTO
  {
#include OATPP_CODEGEN_BEGIN(DTO) ///< Begin DTO codegen section

    class StreamOutput : public oatpp::DTO
    {
      DTO_INIT(StreamOutput, DTO);

      DTO_FIELD_INFO(type)
      {
        info->description = "Type of output.";
      }
      DTO_FIELD(String, type) = "video";

      DTO_FIELD_INFO(video_backend)
      {
        info->description = "(video only) Backend used to encode the video. "
                            "Leave empty for autodetection";
      }
      DTO_FIELD(String, video_backend) = "";

      DTO_FIELD_INFO(video_encoding)
      {
        info->description = "Encoding used to write the output video (four "
                            "letter code). Empty = same as input";
      }
      // upper case -> fourcc
      DTO_FIELD(String, video_encoding) = "H264";

      DTO_FIELD_INFO(video_out)
      {
        info->description = "Video output uri. Can be a file, stream url, "
                            "custom gstreamer pipeline.";
      }
      DTO_FIELD(String, video_out);
    };

    class Stream : public oatpp::DTO
    {
      DTO_INIT(Stream, DTO);

      DTO_FIELD_INFO(chain)
      {
        info->description
            = "Chain call that will be called on every stream element";
      }
      DTO_FIELD(Object<Chain>, chain);

      DTO_FIELD_INFO(predict)
      {
        info->description
            = "Predict call that will be called on every stream element";
      }
      DTO_FIELD(Object<ServicePredict>, predict);

      DTO_FIELD_INFO(output)
      {
        info->description = "Parameters for streaming out.";
      }
      DTO_FIELD(Object<StreamOutput>, output) = StreamOutput::createShared();
    };

    // OUTPUT

    class StreamResponseHead : public oatpp::DTO
    {
      DTO_INIT(StreamResponseHead, DTO)
    };

    class StreamResponseBody : public oatpp::DTO
    {
      DTO_INIT(StreamResponseBody, DTO)
    };

    class StreamResponse : public oatpp::DTO
    {
      DTO_INIT(StreamResponse, DTO)

      DTO_FIELD(String, dd_msg);
      DTO_FIELD(Object<Status>, status);
      DTO_FIELD(Object<StreamResponseHead>, head);
      DTO_FIELD(Object<StreamResponseBody>, body);
    };

#include OATPP_CODEGEN_END(DTO) ///< End DTO codegen section
  }
}

#endif // HTTP_DTO_STREAM_H
