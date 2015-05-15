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
  
  class DDImg
  {
  public:
    DDImg() {}
    ~DDImg() {}

    int read_file(const std::string &fname)
    {
      _img = cv::imread(fname,_bw ? CV_LOAD_IMAGE_GRAYSCALE : CV_LOAD_IMAGE_COLOR);
      if (_img.empty())
	return -1;
      return 0;
    }

    int read_mem(const std::string &content)
    {
      std::vector<unsigned char> vdat(content.begin(),content.end());
      cv::Mat timg(vdat,true);
      _img = cv::Mat(cv::imdecode(timg,_bw ? CV_LOAD_IMAGE_GRAYSCALE : CV_LOAD_IMAGE_COLOR));
      if (_img.empty())
	return -1;
      return 0;
    }
    
    cv::Mat _img;      
    bool _bw = false;
  };
  
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
      fillup_parameters(ad);
    }

    void fillup_parameters(const APIData &ad)
    {
      // optional parameters.
      if (ad.has("width"))
	_width = ad.get("width").get<int>();
      if (ad.has("height"))
	_height = ad.get("height").get<int>();
      if (ad.has("bw"))
	_bw = ad.get("bw").get<bool>();
      if (ad.has("shuffle"))
	_shuffle = ad.get("shuffle").get<bool>();
      if (ad.has("test_split"))
	_test_split = ad.get("test_split").get<double>();
    }
    
    int feature_size() const
    {
      if (_bw) return _width*_height;
      else return _width*_height*3; // RGB
    }

    int batch_size() const
    {
      return _images.size();
    }

    int test_batch_size() const
    {
      return _test_images.size();
    }
    
    void transform(const APIData &ad)
    {
      get_data(ad);
      
      if (ad.has("parameters")) // hotplug of parameters, overriding the defaults
	{
	  APIData ad_param = ad.getobj("parameters");
	  if (ad_param.has("input"))
	    {
	      fillup_parameters(ad_param.getobj("input"));
	    }
	}
      //TODO: could parallelize the reading then push
      for (size_t i=0;i<_uris.size();i++)
	{
	  //cv::Mat imaget = cv::imread(_uris.at(i),_bw ? CV_LOAD_IMAGE_GRAYSCALE : CV_LOAD_IMAGE_COLOR); //TODO: catch errors (size = 0 for file not found)
	  DataEl<DDImg> dimg;
	  dimg._ctype._bw = _bw;
	  if (dimg.read_element(_uris.at(i)))
	  //if (!imaget.data)
	    {
	      throw InputConnectorBadParamException("no data for image " + _uris.at(i));
	    }
	  /*cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );
	    cv::imshow( "Display window", imaget);
	    cv::waitKey(0);*/
	  cv::Size size(_width,_height);
	  cv::Mat image;
	  //cv::resize(imaget,image,size);
	  cv::resize(dimg._ctype._img,image,size);
	  _images.push_back(image);
	}
      // shuffle before possible split
      if (_shuffle)
	{
	  std::random_device rd;
	  std::mt19937 g(rd());
	  std::shuffle(_images.begin(),_images.end(),g);
	}
      // split as required
      if (_test_split > 0)
	{
	  int split_size = std::floor(_images.size() * (1.0-_test_split));
	  auto chit = _images.begin();
	  auto dchit = chit;
	  int cpos = 0;
	  while(chit!=_images.end())
	    {
	      if (cpos == split_size)
		{
		  if (dchit == _images.begin())
		    dchit = chit;
		  _test_images.push_back((*chit));
		}
	      else ++cpos;
	      ++chit;
	    }
	  _images.erase(dchit,_images.end());
	  std::cout << "data split test size=" << _test_images.size() << " / remaining data size=" << _images.size() << std::endl;
	}
    }

    std::vector<cv::Mat> _images;
    std::vector<cv::Mat> _test_images;
    
    // image parameters
    int _width = 227;
    int _height = 227;
    bool _bw = false; /**< whether to convert to black & white. */
    double _test_split = 0.0; /**< auto-split of the dataset. */
    bool _shuffle = false; /**< whether to shuffle the dataset, usually before splitting. */
  };
}

#include "caffeinputconns.h"

#endif
