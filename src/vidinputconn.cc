/**
 * DeepDetect
 * Copyright (c) 2015 Emmanuel Benazera
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

#include "vidinputconn.h"


namespace dd
{
    void VidInputConn::init(const APIData &ad) 
    {
        std::string video_path = "/tmp/bbb_60.mkv";
        fillup_parameters(ad);
    };

    void VidInputConn::transform(const APIData &ad)
    {
      unsigned long video_buffer_size;
      cv::Mat rimg;
      DataEl <DDVid> dvid;
      this->_logger->info("VidInputConn::transform");

      get_data(ad);
      if (this->_uris[0].empty())
      {
        // TODO : stop acquisition
        return;
      }
      else
      {
        if ( this->_current_uri.compare(this->_uris[0]) != 0)
        {
          // New URI received start frame acquistion
          this->init(ad);
          if (dvid.read_element(this->_uris[0], this->_logger))
          {
            this->streamlib._input.set_filepath(this->_uris[0]);
            this->streamlib.init();
            this->_logger->info("run async");
            this->streamlib.run_async();
            this->streamlib.set_max_video_buffer(this->max_video_buffer);
          }

          this->_current_uri = this->_uris[0];
        }
        video_buffer_size = this->streamlib.get_video_buffer(rimg);
        if (video_buffer_size == 0){
          // TODO non an exception ? just waiting frames of frame count finished
          throw InputConnectorBadParamException("no video frame");
        }
        this->_images.push_back(rimg);
      };
      return;    
    };


}
