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

#include "resources.h"

#include <boost/algorithm/string/predicate.hpp>

#include "utils/cv_utils.hpp"

namespace dd
{
  std::string Resource::to_str(ResourceStatus status)
  {
    switch (status)
      {
      case ResourceStatus::ENDED:
        return "ended";
      case ResourceStatus::OPEN:
        return "open";
      case ResourceStatus::ERROR:
        return "error";
      }
    return "unknown";
  }

  cv::VideoCaptureAPIs
  VideoResource::get_video_backend_by_name(const std::string &backend)
  {
    if (backend == "gstreamer")
      {
        return cv::CAP_GSTREAMER;
      }
    else if (backend == "v4l2")
      {
        return cv::CAP_V4L2;
      }
    else if (backend == "ffmpeg")
      {
        return cv::CAP_FFMPEG;
      }
    else
      return cv::CAP_ANY;
  }

  VideoResource::VideoResource(const std::string &name) : Resource(name)
  {
  }

  void VideoResource::init(const oatpp::Object<DTO::Resource> &res_data)
  {
    std::string reader_backend_str = res_data->video_backend->std_str();
    std::string uri = res_data->source->std_str();
    this->_logger->info("Creating resource \"{}\" with backend \"{}\"", uri,
                        reader_backend_str);
    auto reader_backend = get_video_backend_by_name(reader_backend_str);
    std::vector<int> capture_params;
    int cam_id = -1;

    // camera device
    try
      {
        std::size_t pos = 0;
        int read_id = std::stoi(uri, &pos);

        if (pos == uri.size()) // input is a number
          {
            cam_id = read_id;
            this->_logger->info("Requested camera #{} as video source.",
                                cam_id);

            auto req = res_data->video_requirements;

            if (req->width > 0)
              {
                capture_params.push_back(cv::CAP_PROP_FRAME_WIDTH);
                capture_params.push_back(req->width);
                this->_logger->info("Requested width={}", req->width);
              }
            if (req->height > 0)
              {
                capture_params.push_back(cv::CAP_PROP_FRAME_HEIGHT);
                capture_params.push_back(req->height);
                this->_logger->info("Requested height={}", req->height);
              }
            if (req->fps > 0)
              {
                capture_params.push_back(cv::CAP_PROP_FPS);
                capture_params.push_back(req->fps);
                this->_logger->info("Requested fps={}", req->fps);
              }
            std::string req_fourcc = req->fourcc->std_str();
            if (req_fourcc.size() == 4)
              {
                capture_params.push_back(cv::CAP_PROP_FOURCC);
                capture_params.push_back(
                    cv::VideoWriter::fourcc(req_fourcc[0], req_fourcc[1],
                                            req_fourcc[2], req_fourcc[3]));
                this->_logger->info("Requested fourcc={}", req_fourcc);
              }
            else if (req_fourcc.size() != 0)
              {
                this->_logger->warn("Requested invalid fourcc: {}",
                                    req_fourcc);
              }
          }
      }
    catch (...)
      {
        // pass
      }

    bool in_is_gst_pipeline = boost::algorithm::ends_with(uri, "! appsink");

    // set params
    for (size_t i = 0; i < capture_params.size(); i += 2)
      {
        _capture.set(capture_params.at(i), capture_params.at(i + 1));
      }

    // open capture
    if (cam_id >= 0)
      {
        this->_logger->info("Opening VideoCapture on camera {}", cam_id);
        _capture.open(cam_id, reader_backend);
      }
    else if (in_is_gst_pipeline)
      {
        this->_logger->info("Opening VideoCapture with pipeline {}", uri);
        _capture.open(uri, cv::CAP_GSTREAMER);
      }
    else
      {
        this->_logger->info("Opening VideoCapture on video {}", uri);
        _capture.open(uri, reader_backend);
      }

    if (!_capture.isOpened())
      {
        this->_logger->error("VideoCapture could not be opened");
        throw ResourceBadParamException("Video with uri \"" + uri
                                        + "\" could not be opened");
      }

    int fps = _capture.get(cv::CAP_PROP_FPS);
    cv::Size dims((int)_capture.get(cv::CAP_PROP_FRAME_WIDTH),
                  (int)_capture.get(cv::CAP_PROP_FRAME_HEIGHT));
    auto frame_count = uint32_t(_capture.get(cv::CAP_PROP_FRAME_COUNT));
    int fourcc = static_cast<int>(_capture.get(cv::CAP_PROP_FOURCC));
    this->_logger->info("Video properties: {}x{} - {} fps, {} frames, enc={}",
                        dims.width, dims.height, fps, frame_count,
                        cv_utils::fourcc_to_string(fourcc));
  }

  cv::Mat VideoResource::get_image()
  {
    if (_stream_ended)
      throw ResourceForbiddenException("Resource is exhausted");

    cv::Mat frame;
    bool success = _capture.read(frame);

    if (!success)
      {
        if (_frame_counter == 0)
          {
            _stream_error = true;
            throw ResourceInternalException("Could not read frame");
          }
        else
          {
            // if this is the last
            _stream_ended = true;
          }
      }

    _frame_counter++;
    int frame_count = (int)_capture.get(cv::CAP_PROP_FRAME_COUNT);

    if (frame_count > 0 && _frame_counter == frame_count)
      {
        _stream_ended = true;
      }

    return frame;
  }

  ResourceStatus VideoResource::get_status() const
  {
    if (_stream_error)
      return ResourceStatus::ERROR;
    if (_stream_ended)
      return ResourceStatus::ENDED;
    else
      return ResourceStatus::OPEN;
  }

  void VideoResource::fill_info(oatpp::Object<DTO::ResourceResponseBody> &res)
  {
    res->name = _name.c_str();
    res->status = Resource::to_str(get_status()).c_str();

    res->video = DTO::VideoInfo::createShared();
    res->video->width = (int)_capture.get(cv::CAP_PROP_FRAME_WIDTH);
    res->video->height = (int)_capture.get(cv::CAP_PROP_FRAME_HEIGHT);
    res->video->fps = (float)_capture.get(cv::CAP_PROP_FPS);
    auto fourcc_str = cv_utils::fourcc_to_string(
        static_cast<int>(_capture.get(cv::CAP_PROP_FOURCC)));
    res->video->fourcc = fourcc_str.c_str();
    res->video->frame_count = (int)_capture.get(cv::CAP_PROP_FRAME_COUNT);
    res->video->current_frame = _frame_counter;
  }

  res_variant_type
  ResourceFactory::create(const std::string &name,
                          const oatpp::Object<DTO::Resource> &res_data)
  {
    if (res_data->type == "video")
      {
        return VideoResource(name);
      }
    else
      throw ResourceBadParamException("Unknown resource type: "
                                      + res_data->type->std_str());
  }
}
