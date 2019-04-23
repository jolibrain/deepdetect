/**
 * DeepDetect
 * Copyright (c) 2018 Jolibrain
 * Author: Nicolas Bertrand <nicolas@davionbertrand.net>
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
//TODO: streamlibgstreamertx2.h
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
        // NOT USED
        _logger->info("not used:: read_mem : {}", content);
        return 1;
      };

      int read_file(const std::string &fname)
      {
        // TODO: TO IMPLEMENT
        // add streamlib handling in DDvid instead of VidInputConnector
        _logger->info(" read file : {}", fname);
        return 1;
      }

      int read_db(const std::string &fname)
      {

        // NOT USED
        _logger->info("not used:: read_db: {}", fname);
        return 1;
      }

      int read_dir(const std::string &dir)
      {
        // NOT USED
        _logger->info("not used:: read_dir: {}", dir);
        return 1;
      }

      std::shared_ptr<spdlog::logger> _logger;
  };

  class VidInputConn : public InputConnectorStrategy
  {
    public:
      VidInputConn()
        :InputConnectorStrategy()
      {
	   //std::cout << "Normal constructor : " << std::endl;
           _streamlib =  new vnn::StreamLibGstreamerDesktop
             <vnn::VnnInputConnectorFile, vnn::VnnOutputConnectorDummy>();
      }

      VidInputConn(const VidInputConn &i);

      ~VidInputConn()
	{
	  //delete _streamlib;
	}

      void init(const APIData &ad);

      void fillup_parameters(const APIData &ad)
      {
        if (ad.has("width"))
          _width = ad.get("width").get<int>();
        if (ad.has("height"))
          _height = ad.get("height").get<int>();
        if (ad.has("mean"))
        {
          apitools::get_floats(ad, "mean", _mean);
          _has_mean_scalar = true;
        }
      };

      int feature_size() const
      {
	if (_bw)
	  return _width * _height;
	else return _width * _height * 3; // RGB
      };

      int test_batch_size() const
      {
        return MAX_FRAMES;
      };

      void transform(const APIData &ad);

      // data
      unsigned int MAX_FRAMES = 30;
      int _width = 300; /**< default value. */
      int _height = 300;
      bool _bw = false; /**< whether to convert to black & white. */
      std::vector<cv::Mat> _images;
      std::vector<std::pair<int,int>> _images_size;
      vnn::StreamLibGstreamerDesktop<vnn::VnnInputConnectorFile,
        vnn::VnnOutputConnectorDummy> *_streamlib;
      //TODO: tx2 stream

      bool _has_mean_scalar = false; /**< whether scalar is set. */
      std::vector<float> _mean; /**< mean image pixels, to be subtracted from images. */
      unsigned long max_video_buffer = 300;
      std::string _current_uri;
      bool is_running = false;
      int _counter = 0 ;
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
