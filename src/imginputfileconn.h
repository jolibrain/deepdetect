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

#ifndef IMGINPUTFILECONN_H
#define IMGINPUTFILECONN_H

#include "inputconnectorstrategy.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace dd
{
  class ImgInputFileConn : public InputConnectorStrategy
  {
  public:
  ImgInputFileConn()
    :InputConnectorStrategy()
    {
    }
    
    ~ImgInputFileConn() {}

    int transform(const APIData &ad)
    {
      _uris = ad.get("data").get<std::vector<std::string>>();
      //_imgfname = ad.get(_imgfname).get<std::string>();
      //_imgfname = vdata.at(0);
      //std::cout << "opening image=" << _imgfname << std::endl;
      for (size_t i=0;i<_uris.size();i++)
	{
	  cv::Mat imaget = cv::imread(_uris.at(i),CV_LOAD_IMAGE_COLOR); //TODO: catch errors (size = 0 for file not found)
	  /*cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );
	    cv::imshow( "Display window", imaget);
	    cv::waitKey(0);*/
	  cv::Size size(227,227);
	  cv::Mat image;
	  cv::resize(imaget,image,size);
	  _images.push_back(image);
	}
      return 0;
    }

    std::vector<std::string> _uris;
    std::vector<cv::Mat> _images;
    //std::string _imgfname = "imgfname";
  };
}

#endif
