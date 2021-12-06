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

    /** Information on video source */
    class VideoInfo : public oatpp::DTO
    {
      DTO_INIT(VideoInfo, DTO)

      DTO_FIELD(Int32, width) = -1;
      DTO_FIELD(Int32, height) = -1;
      DTO_FIELD(Float32, fps) = -1;

      DTO_FIELD_INFO(fourcc)
      {
        info->description
            = "Video codec, as a 4-letter code. eg YUYV, MJPEG, H264...";
      }
      DTO_FIELD(String, fourcc) = "";

      DTO_FIELD_INFO(frame_count)
      {
        info->description = "Video frame count";
      }
      DTO_FIELD(Int32, frame_count) = -1;

      DTO_FIELD_INFO(current_frame)
      {
        info->description = "Id of current frame";
      }
      DTO_FIELD(Int32, current_frame) = -1;
    };

    /** Video requirements for cameras */
    class VideoRequirements : public oatpp::DTO
    {
      DTO_INIT(VideoRequirements, DTO);

      DTO_FIELD_INFO(fps)
      {
        info->description = "Requested camera FPS";
      }
      DTO_FIELD(Int32, fps) = -1;

      DTO_FIELD_INFO(width)
      {
        info->description = "Requested video width";
      }
      DTO_FIELD(Int32, width) = -1;

      DTO_FIELD_INFO(height)
      {
        info->description = "Requested video height";
      }
      DTO_FIELD(Int32, height) = -1;

      DTO_FIELD_INFO(fourcc)
      {
        info->description
            = "Requested codec, as a 4-letter code. eg YUYV, MJPEG, H264...";
      }
      DTO_FIELD(String, fourcc) = "";
    };

    // INPUT

    class Resource : public oatpp::DTO
    {
      DTO_INIT(Resource, DTO);

      DTO_FIELD_INFO(type)
      {
        info->description = "Type of source. Available types: video.";
      }
      DTO_FIELD(String, type);

      // video
      DTO_FIELD_INFO(video_backend)
      {
        info->description
            = "For video sources, what backend is prefered for decoding";
      }
      DTO_FIELD(String, video_backend) = "any";

      DTO_FIELD_INFO(source)
      {
        info->description
            = "Source URI. Can be: stream url (youtube,...), video file, "
              "local device (camera...), custom gstreamer pipeline";
      }
      DTO_FIELD(String, source);

      DTO_FIELD_INFO(video_requirements)
      {
        info->description = "Requirements when multiple formats are available "
                            "for a source (e.g. for cameras)";
      }
      DTO_FIELD(Object<VideoRequirements>, video_requirements)
          = VideoRequirements::createShared();
    };

    // OUTPUT

    class ResourceResponseHead : public oatpp::DTO
    {
      DTO_INIT(ResourceResponseHead, DTO)
    };

    class ResourceResponseBody : public oatpp::DTO
    {
      DTO_INIT(ResourceResponseBody, DTO)

      DTO_FIELD_INFO(name)
      {
        info->description = "Name of the resource";
      }
      DTO_FIELD(String, name);

      DTO_FIELD_INFO(status)
      {
        info->description = "Resource status: open, ended, error";
      }
      DTO_FIELD(String, status);

      DTO_FIELD_INFO(message)
      {
        info->description = "Message that can go with an \"error\" status";
      }
      DTO_FIELD(String, message);

      DTO_FIELD(Object<VideoInfo>, video);
    };

    class ResourceResponse : public GenericResponse
    {
      DTO_INIT(ResourceResponse, GenericResponse)

      DTO_FIELD(Object<ResourceResponseHead>, head);
      DTO_FIELD(Object<ResourceResponseBody>, body);
    };

#include OATPP_CODEGEN_END(DTO) ///< End DTO codegen section
  }
}

#endif // HTTP_DTO_RESOURCE_H
