/**
 * DeepDetect
 * Copyright (c) 2021 Louis Jean
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

#ifndef RESOURCES_H
#define RESOURCES_H

#include <iostream>
#include <mapbox/variant.hpp>
#include <opencv2/opencv.hpp>
#include "dd_spdlog.h"

#include "dto/resource.hpp"
#include "inputconnectorstrategy.h"
#include "videoinputfileconn.h"

namespace dd
{

  // Exceptions

  /**
   * \brief ML library bad parameter exception
   */
  class ResourceBadParamException : public std::exception
  {
  public:
    ResourceBadParamException(const std::string &s) : _s(s)
    {
    }
    ~ResourceBadParamException()
    {
    }
    const char *what() const noexcept
    {
      return _s.c_str();
    }

  private:
    std::string _s;
  };

  /**
   * \brief ML library internal error exception
   */
  class ResourceInternalException : public std::exception
  {
  public:
    ResourceInternalException(const std::string &s) : _s(s)
    {
    }
    ~ResourceInternalException()
    {
    }
    const char *what() const noexcept
    {
      return _s.c_str();
    }

  private:
    std::string _s;
  };

  class ResourceForbiddenException : public std::exception
  {
  public:
    ResourceForbiddenException(const std::string &s) : _s(s)
    {
    }
    ~ResourceForbiddenException()
    {
    }
    const char *what() const noexcept
    {
      return _s.c_str();
    }

  private:
    std::string _s;
  };

  class ResourceNotFoundException : public std::exception
  {
  public:
    ResourceNotFoundException(const std::string &s) : _s(s)
    {
    }
    ~ResourceNotFoundException()
    {
    }
    const char *what() const noexcept
    {
      return _s.c_str();
    }

  private:
    std::string _s;
  };

  enum class ResourceStatus
  {
    OPEN,
    ENDED,
    ERROR
  };

  class Resource
  {
  public:
    static std::string to_str(ResourceStatus status);

  public:
    Resource(const std::string &name) : _name(name)
    {
      if (!name.empty())
        _logger = DD_SPDLOG_LOGGER(name);
    }

    Resource(Resource &&other)
        : _name(std::move(other._name)), _logger(std::move(other._logger))
    {
    }

    virtual ~Resource()
    {
      if (!_name.empty())
        spdlog::drop(_name);
    }

    /** Return resource status. */
    virtual ResourceStatus get_status() const = 0;

    void fill_info(oatpp::Object<DTO::ResourceResponseBody> &resource);

  public:
    std::string _name;
    std::shared_ptr<spdlog::logger> _logger;
  };

  class VideoResource : public Resource
  {
  public:
    /** Convert string to video backend. */
    static cv::VideoCaptureAPIs
    get_video_backend_by_name(const std::string &backend);

  public:
    VideoResource(const std::string &name = "");

    void init(const oatpp::Object<DTO::Resource> &res_data);

    cv::Mat get_image();

    ResourceStatus get_status() const override;

    void fill_info(oatpp::Object<DTO::ResourceResponseBody> &resource);

  public:
    cv::VideoCapture _capture;
    bool _stream_ended = false;
    bool _stream_error = false;
    int _frame_counter = 0;
  };

  typedef mapbox::util::variant<VideoResource> res_variant_type;

  namespace visitor_resources
  {
    class v_init
    {
    public:
      oatpp::Object<dd::DTO::Resource> _res_data;

      template <typename T> void operator()(T &resource)
      {
        resource.init(_res_data);
      }
    };

    template <typename T>
    static inline void init(T &resource,
                            const oatpp::Object<dd::DTO::Resource> &res_data)
    {
      visitor_resources::v_init v{ res_data };
      mapbox::util::apply_visitor(v, resource);
    }

    class v_apply
    {
    public:
      APIData &_ad_in;
      oatpp::Object<DTO::ResourceResponseBody> &_response;

      void operator()(VideoResource &resource)
      {
        std::vector<cv::Mat> raw_images{ resource.get_image() };
        if (_ad_in.has("dto"))
          {
            auto any = _ad_in.get("dto").get<oatpp::Any>();

            oatpp::Object<DTO::ServicePredict> predict_dto(
                std::static_pointer_cast<typename DTO::ServicePredict>(
                    any->ptr));
            predict_dto->_data_raw_img = raw_images;
          }
        else
          {
            _ad_in.add("data_raw_img", raw_images);
          }

        // Set returned infos
        _response->name = resource._name.c_str();
        _response->status = Resource::to_str(resource.get_status()).c_str();
      }
    };

    template <typename T>
    static inline void
    apply(T &resource, APIData &ad_in,
          oatpp::Object<dd::DTO::ResourceResponseBody> &response)
    {
      visitor_resources::v_apply v{ ad_in, response };
      mapbox::util::apply_visitor(v, resource);
    }

    class v_get_info
    {
    public:
      oatpp::Object<dd::DTO::ResourceResponseBody> &_response;

      template <typename T> void operator()(T &resource)
      {
        resource.fill_info(_response);
      }
    };

    template <typename T>
    static inline void
    get_info(T &resource,
             oatpp::Object<dd::DTO::ResourceResponseBody> &resource_info)
    {
      visitor_resources::v_get_info v{ resource_info };
      mapbox::util::apply_visitor(v, resource);
    }
  }

  class ResourceFactory
  {
  public:
    static res_variant_type create(const std::string &name,
                                   const oatpp::Object<DTO::Resource> &res);
  };
}

#endif // RESOURCES_H
