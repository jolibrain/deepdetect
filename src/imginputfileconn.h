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
#include "ext/base64/base64.h"
#include <glog/logging.h>
#include <random>

namespace dd
{
  
  class DDImg
  {
  public:
    DDImg() {}
    ~DDImg() {}

    // base64 detection
    bool is_within_base64_range(char c) const
    {
      if ((c >= 'A' && c <= 'Z')
	  || (c >= 'a' && c <= 'z')
	  || (c >= '0' && c <= '9')
	  || (c == '+' || c=='/' || c=='='))
	return true;
      else return false;
    }

    bool possibly_base64(const std::string &s) const
    {
      bool ism = is_multiple_four(s);
      if (!ism)
	return false;
      for (char c: s)
	{
	  bool within_64 = is_within_base64_range(c);
	  if (!within_64)
	    return false;
	}
      return true;
    }
    
    bool is_multiple_four(const std::string &s) const
    {
      if (s.length() % 4 == 0)
	return true;
      else return false;
    }

    // decode image
    void decode(const std::string &str)
      {
	std::vector<unsigned char> vdat(str.begin(),str.end());
	cv::Mat img = cv::Mat(cv::imdecode(cv::Mat(vdat,true),_bw ? CV_LOAD_IMAGE_GRAYSCALE : CV_LOAD_IMAGE_COLOR));
	_imgs_size.push_back(std::pair<int,int>(img.rows,img.cols));
	cv::Size size(_width,_height);
	cv::Mat rimg;
	cv::resize(img,rimg,size,0,0,CV_INTER_CUBIC);
	_imgs.push_back(rimg);
      }
    
    // deserialize image, independent of format
    void deserialize(std::stringstream &input)
      {
	size_t size = 0;
	input.seekg(0,input.end);
	size = input.tellg();
	input.seekg(0,input.beg);
	char* data = new char[size];
	input.read(data, size);
	std::string str(data,data+size);
	delete[]data;
	decode(str);
      }
    
    // data acquisition
    int read_file(const std::string &fname)
    {
      cv::Mat img = cv::imread(fname,_bw ? CV_LOAD_IMAGE_GRAYSCALE : CV_LOAD_IMAGE_COLOR);
      if (img.empty())
	return -1;
      _imgs_size.push_back(std::pair<int,int>(img.rows,img.cols));
      cv::Size size(_width,_height);
      cv::Mat rimg;
      cv::resize(img,rimg,size,0,0,CV_INTER_CUBIC);
      _imgs.push_back(rimg);
      return 0;
    }

    int read_db(const std::string &fname)
    {
      _db_fname = fname;
      return 0;
    }
    
    int read_mem(const std::string &content)
    {
      cv::Mat timg;
      _b64 = possibly_base64(content);
      if (_b64)
	{
	  std::string ccontent;
	  Base64::Decode(content,&ccontent);
	  std::stringstream sstr;
	  sstr << ccontent;
	  deserialize(sstr);
	}
      else
	{
	  decode(content);
	}
      if (_imgs.at(0).empty())
	return -1;
      return 0;
    }

    int read_dir(const std::string &dir)
    {
      // list directories in dir
      std::unordered_set<std::string> subdirs;
      if (fileops::list_directory(dir,false,true,subdirs))
	throw InputConnectorBadParamException("failed reading text subdirectories in data directory " + dir);
      LOG(INFO) << "imginputfileconn: list subdirs size=" << subdirs.size();

      // list files and classes
      std::vector<std::pair<std::string,int>> lfiles; // labeled files
      std::unordered_map<int,std::string> hcorresp; // correspondence class number / class name
      if (!subdirs.empty())
	{
	  int cl = 0;
	  auto uit = subdirs.begin();
	  while(uit!=subdirs.end())
	    {
	      std::unordered_set<std::string> subdir_files;
	      if (fileops::list_directory((*uit),true,false,subdir_files))
		throw InputConnectorBadParamException("failed reading image data sub-directory " + (*uit));
	      auto fit = subdir_files.begin();
	      while(fit!=subdir_files.end()) // XXX: re-iterating the file is not optimal
		{
		  lfiles.push_back(std::pair<std::string,int>((*fit),cl));
		  ++fit;
		}
	      ++cl;
	      ++uit;
	    }
	}
      else
	{
	  std::unordered_set<std::string> test_files;
	  fileops::list_directory(dir,true,false,test_files);
	  auto fit = test_files.begin();
	  while(fit!=test_files.end())
	    {
	      lfiles.push_back(std::pair<std::string,int>((*fit),-1)); // -1 for no class
	      ++fit;
	    }
	}
      
      // read images
      cv::Size size(_width,_height);
      _imgs.reserve(lfiles.size());
      _img_files.reserve(lfiles.size());
      _labels.reserve(lfiles.size());
      for (std::pair<std::string,int> &p: lfiles)
	{
	  cv::Mat img = cv::imread(p.first,_bw ? CV_LOAD_IMAGE_GRAYSCALE : CV_LOAD_IMAGE_COLOR);
	  _imgs_size.push_back(std::pair<int,int>(img.rows,img.cols));
	  cv::Mat rimg;
	  cv::resize(img,rimg,size,0,0,CV_INTER_CUBIC);
	  _imgs.push_back(rimg);
	  _img_files.push_back(p.first);
	  if (p.second >= 0)
	    _labels.push_back(p.second);
	  if (_imgs.size() % 1000 == 0)
	    LOG(INFO) << "read " << _imgs.size() << " images\n";
	}
      return 0;
    }
    
    std::vector<cv::Mat> _imgs;
    std::vector<std::string> _img_files;
    std::vector<std::pair<int,int>> _imgs_size;
    bool _bw = false;
    bool _b64 = false;
    std::vector<int> _labels;
    int _width = 227;
    int _height = 227;
    std::string _db_fname;
  };
  
  class ImgInputFileConn : public InputConnectorStrategy
  {
  public:
  ImgInputFileConn()
    :InputConnectorStrategy(){}
    ImgInputFileConn(const ImgInputFileConn &i)
      :InputConnectorStrategy(i),_width(i._width),_height(i._height),_bw(i._bw),_mean(i._mean),_has_mean_scalar(i._has_mean_scalar) {}
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
      if (ad.has("seed"))
	_seed = ad.get("seed").get<int>();
      if (ad.has("test_split"))
	_test_split = ad.get("test_split").get<double>();
      if (ad.has("mean"))
	{
	  std::vector<int> vm = ad.get("mean").get<std::vector<int>>();
	  if (vm.size() == 3)
	    {
	      int r,g,b;
	      r = vm[0];
	      g = vm[1];
	      b = vm[2];
	      _mean = cv::Scalar(r,g,b);
	      _has_mean_scalar = true;
	    }
	  else if (vm.size() == 1) // bw
	    {
	      _mean = cv::Scalar(vm.at(0));
	      _has_mean_scalar = true;
	    }
	}
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
      int catch_read = 0;
      std::string catch_msg;
      std::vector<std::string> uris;
#pragma omp parallel for
      for (size_t i=0;i<_uris.size();i++)
	{
	  bool no_img = false;
	  std::string u = _uris.at(i);
	  DataEl<DDImg> dimg;
	  dimg._ctype._bw = _bw;
	  dimg._ctype._width = _width;
	  dimg._ctype._height = _height;
	  try
	    {
	      if (dimg.read_element(u))
		{
		  LOG(ERROR) << "no data for image " << u;
		  no_img = true;
		}
	      if (!dimg._ctype._db_fname.empty())
		_db_fname = dimg._ctype._db_fname;
	    }
	  catch(std::exception &e)
	    {
#pragma omp critical
	      {
		++catch_read;
		catch_msg = e.what();
		no_img = true;
	      }
	    }
	  if (no_img)
	    continue;
	  if (!_db_fname.empty())
	    continue;
	  
#pragma omp critical
	  {
	    _images.insert(_images.end(),
	      std::make_move_iterator(dimg._ctype._imgs.begin()),
	      std::make_move_iterator(dimg._ctype._imgs.end()));
	    _images_size.insert(_images_size.end(),
				std::make_move_iterator(dimg._ctype._imgs_size.begin()),
				std::make_move_iterator(dimg._ctype._imgs_size.end()));
	    if (!dimg._ctype._labels.empty())
	      _test_labels.insert(_test_labels.end(),
	      std::make_move_iterator(dimg._ctype._labels.begin()),
	      std::make_move_iterator(dimg._ctype._labels.end()));
	    if (!dimg._ctype._b64 && dimg._ctype._imgs.size() == 1)
	      uris.push_back(u);
	    else if (!dimg._ctype._img_files.empty())
	      uris.insert(uris.end(),
	      std::make_move_iterator(dimg._ctype._img_files.begin()),
	      std::make_move_iterator(dimg._ctype._img_files.end()));
	    else uris.push_back(std::to_string(i));
	  }
	}
      if (catch_read)
	throw InputConnectorBadParamException(catch_msg);
      _uris = uris;
      if (!_db_fname.empty())
	return; // db filename is passed to backend
      
      // shuffle before possible split
      if (_shuffle)
	{
	  std::mt19937 g;
	  if (_seed >= 0)
	    g = std::mt19937(_seed);
	  else
	    {
	      std::random_device rd;
	      g = std::mt19937(rd());
	    }
	  std::shuffle(_images.begin(),_images.end(),g); //XXX beware: labels are not shuffled, i.e. let's not shuffle while testing
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
	  LOG(INFO) << "data split test size=" << _test_images.size() << " / remaining data size=" << _images.size();
	}
      if (_images.empty())
	throw InputConnectorBadParamException("no image could be found");
    }

    // data
    std::vector<cv::Mat> _images;
    std::vector<cv::Mat> _test_images;
    std::vector<int> _test_labels;
    std::vector<std::pair<int,int>> _images_size;
    // image parameters
    int _width = 227;
    int _height = 227;
    bool _bw = false; /**< whether to convert to black & white. */
    double _test_split = 0.0; /**< auto-split of the dataset. */
    bool _shuffle = false; /**< whether to shuffle the dataset, usually before splitting. */
    int _seed = -1; /**< shuffling seed. */
    cv::Scalar _mean; /**< mean image pixels, to be subtracted from images. */
    bool _has_mean_scalar = false; /**< whether scalar is set. */
    std::string _db_fname;
  };
}

#include "caffeinputconns.h"

#ifdef USE_TF
#include "tfinputconns.h"
#endif

#endif
