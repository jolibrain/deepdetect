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

#include "vidinputconn.h"


namespace dd
{

  VidInputConn::VidInputConn(const VidInputConn &i)
        :InputConnectorStrategy(i)
    {
      this->_streamlib = i._streamlib;
      this->_counter = i._counter;
      this->_counter++;
    };

    void VidInputConn::init(const APIData &ad) 
    {
        //std::string video_path = "/tmp/bbb_60.mkv";
        fillup_parameters(ad);
    };

    void VidInputConn::transform(const APIData &ad)
    {
      unsigned long video_buffer_size;
      cv::Mat rimg;
      DataEl <DDVid> dvid;
      long int  timestamp;
      //this->_logger->info("VidInputConn::transform");

      get_data(ad);
      if (this->_uris[0].empty())
      {
        // TODO : stop acquisition
        this->is_running = false;
        return;
      }
      else
      {

        //this->_logger->info(" is playing = {}", this->streamlib->is_playing());
        if ( !this->_streamlib->is_playing() )
        {
          // New URI received start frame acquistion
          this->init(ad);
          if (dvid.read_element(this->_uris[0], this->_logger))
          {
            this->_streamlib->_input.set_filepath(this->_uris[0]);
            this->_streamlib->set_scale_size(this->_width, this->_height);
	    this->_streamlib->_scalesink_sync = true;
            this->_streamlib->init();
            this->_logger->info("run async");
            this->_streamlib->run_async();
            this->_streamlib->set_max_video_buffer(this->max_video_buffer);
            //std::this_thread::sleep_for(std::chrono::seconds(1)); //beniz: deactivated
          }

          this->_current_uri = this->_uris[0];
          this->is_running = true;
          this->_logger->info("_current_uri = {} // {}", this->_current_uri, this->is_running);
        }

        video_buffer_size = this->_streamlib->get_video_buffer(rimg, timestamp);
        this->_uris[0] = std::to_string(timestamp);
        if (video_buffer_size == 0){
          // TODO non an exception ? just waiting frames of frame count finished
          throw InputConnectorBadParamException("no video frame");
        }
        this->_images.push_back(rimg);
        this->_images_size.push_back( std::pair<int,int>(
           this->_streamlib->get_original_height(),
            this->_streamlib->get_original_width()
            )
            );

      };
      return;
    };


}
