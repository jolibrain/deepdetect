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
    :InputConnectorStrategy(){}
    ImgInputFileConn(const ImgInputFileConn &i)
      :InputConnectorStrategy(),_width(i._width),_height(i._height),_bw(i._bw) {}
    ~ImgInputFileConn() {}

    void init(const APIData &ad)
    {
      fillup_parameters(ad,_width,_height);
    }

    void fillup_parameters(const APIData &ad,
			   int &width, int &height)
    {
      // optional parameters.
      if (ad.has("width"))
	width = ad.get("width").get<int>();
      if (ad.has("height"))
	height = ad.get("height").get<int>();
      if (ad.has("bw"))
	_bw = ad.get("bw").get<bool>();
    }
    
    size_t size() const
    {
      return _images.size();
    }
    
    int transform(const APIData &ad)
    {
      try
	{
	  _uris = ad.get("data").get<std::vector<std::string>>();
	}
      catch(...)
	{
	  throw InputConnectorBadParamException("missing data");
	}
      if (_uris.empty())
	{
	  throw InputConnectorBadParamException("missing data");
	}
    
      int width = _width;
      int height = _height;
      if (ad.has("parameters")) // hotplug of parameters, overriding the defaults
	{
	  APIData ad_param = ad.getobj("parameters");
	  if (ad_param.has("input"))
	    {
	      fillup_parameters(ad_param.getobj("input"),width,height);
	    }
	}
      for (size_t i=0;i<_uris.size();i++)
	{
	  cv::Mat imaget = cv::imread(_uris.at(i),_bw ? CV_LOAD_IMAGE_GRAYSCALE : CV_LOAD_IMAGE_COLOR); //TODO: catch errors (size = 0 for file not found)
	  /*cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );
	    cv::imshow( "Display window", imaget);
	    cv::waitKey(0);*/
	  cv::Size size(width,height);
	  cv::Mat image;
	  cv::resize(imaget,image,size);
	  _images.push_back(image);
	}
      return 0;
    }

    std::vector<std::string> _uris;
    std::vector<cv::Mat> _images;
    
    // resizing images
    int _width = 227;
    int _height = 227;
    bool _bw = false;
  };
}

#define CPU_ONLY // TODO: elsewhere
#include "caffeinputconns.h"

#endif
