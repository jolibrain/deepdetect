/**
 * DeepDetect
 * Copyright (c) 2014 Emmanuel Benazera
 * Author: Emmanuel Benazera <beniz@droidnik.fr>
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

#ifndef VIDINPUTCONN_H
#define VIDINPUTCONN_H

#include "inputconnectorstrategy.h"

#include "streamlibgstreamerdesktop.h"
#include "vnninputconnectorfile.h"
#include "vnnoutputconnectordummy.h"
#include "utils/apitools.h"
#include <opencv2/imgproc/imgproc.hpp>

namespace dd
{

  class DDVid
  {
    public:
      DDVid() {}
      ~DDVid() {}

      int read_mem(const std::string &content)
      {
        return 0;
      };

      int read_db(const std::string &fname)
      {
        return 0;
      }

      int read_dir(const std::string &dir)
      {
        return 0;
      }

      int read_file(const std::string &fname)
      {
        // TODO: TO IMPLEMENT
        return 1;
      }

      std::shared_ptr<spdlog::logger> _logger;
  };

  class VidInputConn : public InputConnectorStrategy
  {
    public:
      VidInputConn()
        :InputConnectorStrategy(){}
      VidInputConn(const VidInputConn &i)
        :InputConnectorStrategy(i) {}
      ~VidInputConn() {}

      void init(const APIData &ad);

      void fillup_parameters(const APIData &ad)
      {
        // TODO: Ask what is fill up parameters fo video
        if (ad.has("width"))
          _width = ad.get("width").get<int>();
      };

      int feature_size() const
      {
        // TODO: what for ? 
        return 30;
      };
       int test_batch_size() const
    {
      
      //return _test_images.size();
      return MAX_FRAMES;
    }
 
      void transform(const APIData &ad);

      // data
      unsigned int MAX_FRAMES = 30;
      int _width = 300;
      int _height = 300;
      std::vector<cv::Mat> _images;
      std::vector<cv::Mat> _images_size;
      unsigned long max_video_buffer = 300;
      std::string _current_uri;

      vnn::StreamLibGstreamerDesktop<vnn::VnnInputConnectorFile,
        vnn::VnnOutputConnectorDummy>  streamlib;

      /* TODO: ImgInput heritage: deal what to do with that */
      bool _has_mean_scalar = false; /**< whether scalar is set. */
      std::vector<float> _mean; /**< mean image pixels, to be subtracted from images. */

  };
}

#include "caffeinputconns.h"

#ifdef USE_TF
#include "backends/tf/tfinputconns.h"
#endif
#ifdef USE_DLIB
#include "backends/dlib/dlibinputconns.h"
#endif

#ifdef USE_CAFFE2
#include "backends/caffe2/caffe2inputconns.h"
#endif

#endif
