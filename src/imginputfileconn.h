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
#include <opencv2/opencv.hpp>
#ifdef USE_CUDA_CV
#include <opencv2/cudaimgproc.hpp>
#endif
#if CV_VERSION_MAJOR >= 3
#define CV_LOAD_IMAGE_COLOR cv::IMREAD_COLOR
#define CV_LOAD_IMAGE_GRAYSCALE cv::IMREAD_GRAYSCALE
#define CV_LOAD_IMAGE_UNCHANGED cv::IMREAD_UNCHANGED
#define CV_BGR2RGB cv::COLOR_BGR2RGB
#define CV_BGR2GRAY cv::COLOR_BGR2GRAY
#define CV_INTER_CUBIC cv::INTER_CUBIC
#endif
#include "ext/base64/base64.h"
#include "utils/apitools.h"
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

    void resize(const cv::Mat &src, cv::Mat &dst,
		const cv::Size &cvsize,
		const double &fx, const double &fy) const
    {
#ifdef USE_CUDA_CV
      if (_cuda)
	{
	  cv::cuda::GpuMat d_src;
	  d_src.upload(src);
	  cv::cuda::GpuMat d_dst;
	  cv::cuda::resize(d_src, d_dst, cvsize, fx, fy, select_cv_interp());
	  if (_rgb)
	    cv::cuda::cvtColor(d_dst, d_dst, CV_BGR2RGB);
	  d_dst.download(dst);
	}
      else
#endif
	{
	  cv::resize(src, dst, cvsize, fx, fy, select_cv_interp());
	  if (_rgb)
	    cv::cvtColor(dst, dst, CV_BGR2RGB);
	}
    }
    
    void scale(const cv::Mat &src, cv::Mat &dst) const {
      float coef = std::min(static_cast<float>(_scale_max) / std::max(src.rows, src.cols),
			    static_cast<float>(_scale_min) / std::min(src.rows, src.cols));
      
      resize(src, dst, cv::Size(), coef, coef);
    }

    // decode image
    void decode(const std::string &str)
      {
	std::vector<unsigned char> vdat(str.begin(),str.end());
	cv::Mat img = cv::Mat(cv::imdecode(cv::Mat(vdat,false),
					   _unchanged_data ? CV_LOAD_IMAGE_UNCHANGED :
					   (_bw ? CV_LOAD_IMAGE_GRAYSCALE : CV_LOAD_IMAGE_COLOR)));
      	if (_keep_orig)
	  _orig_imgs.push_back(img);
	_imgs_size.push_back(std::pair<int,int>(img.rows,img.cols));
	cv::Mat rimg;
	if (_scaled)
	  scale(img, rimg);
	else if (_width == 0 || _height == 0) {
	  if (_width == 0 && _height == 0) {
	    // XXX - Do nothing and keep native resolution. May cause issues if batched images are different resolutions
            rimg = img;
	  } else {
	    // Resize so that the larger dimension is set to whichever (width or height) is non-zero, maintaining aspect ratio
	    // XXX - This may cause issues if batch images are different resolutions
	    size_t currMaxDim = std::max(img.rows, img.cols);
	    double scale = static_cast<double>(std::max(_width, _height)) / static_cast<double>(currMaxDim);
	    resize(img,rimg,cv::Size(),scale,scale);
	  }
	} else {
	  // Resize normally to the specified width and height
	  resize(img,rimg,cv::Size(_width,_height),0,0);
	}
	
	if (_crop_width != 0 && _crop_height != 0) {
	  int widthBorder = (_width - _crop_width)/2;
	  int heightBorder = (_height - _crop_height)/2;
	  rimg = rimg(cv::Rect(widthBorder, heightBorder, _crop_width, _crop_height));
	}

	_imgs.push_back(std::move(rimg));
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
      cv::Mat img = cv::imread(fname, _unchanged_data ? CV_LOAD_IMAGE_UNCHANGED :
			       (_bw ? CV_LOAD_IMAGE_GRAYSCALE : CV_LOAD_IMAGE_COLOR));
      if (_keep_orig)
	_orig_imgs.push_back(img);
      if (img.empty())
	{
	  _logger->error("empty image {}",fname);
	  return -1;
	}
      _imgs_size.push_back(std::pair<int,int>(img.rows,img.cols));
      cv::Mat rimg;
      try
	{
	  if (_scaled)
	    scale(img, rimg);
	  else if (_width == 0 || _height == 0) {
	    if (_width == 0 && _height == 0) {
	      // Do nothing and keep native resolution. May cause issues if batched images are different resolutions
	      rimg = img;
	    } else {
	      // Resize so that the larger dimension is set to whichever (width or height) is non-zero, maintaining aspect ratio
	      // XXX - This may cause issues if batch images are different resolutions
	      size_t currMaxDim = std::max(img.rows, img.cols);
	      double scale = static_cast<double>(std::max(_width, _height)) / static_cast<double>(currMaxDim);
	      resize(img,rimg,cv::Size(),scale,scale);
	    }
	  } else {
	    // Resize normally to the specified width and height
	    resize(img,rimg,cv::Size(_width,_height),0,0);
	  }
	}
      catch(...)
	{
	  throw InputConnectorBadParamException("failed resizing image " + fname);
	}
      if (_crop_width != 0 && _crop_height != 0) {
	int widthBorder = (_width - _crop_width)/2;
	int heightBorder = (_height - _crop_height)/2;
	try {
	  rimg = rimg(cv::Rect(widthBorder, heightBorder, _crop_width, _crop_height));
	} catch(...) {
	  throw InputConnectorBadParamException("failed cropping image " + fname);
	}
      }
      _imgs.push_back(std::move(rimg));
      return 0;
    }

    int read_db(const std::string &fname)
    {
      _db_fname = fname;
      return 0;
    }
    
    int read_mem(const std::string &content)
    {
      _in_mem = true;
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
      if (fileops::list_directory(dir,false,true,false,subdirs))
	throw InputConnectorBadParamException("failed reading text subdirectories in data directory " + dir);
      _logger->info("imginputfileconn: list subdirs size={}",subdirs.size());
      
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
	      if (fileops::list_directory((*uit),true,false,true,subdir_files))
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
	  fileops::list_directory(dir,true,false,false,test_files);
	  auto fit = test_files.begin();
	  while(fit!=test_files.end())
	    {
	      lfiles.push_back(std::pair<std::string,int>((*fit),-1)); // -1 for no class
	      ++fit;
	    }
	}
      
      // read images
      _imgs.reserve(lfiles.size());
      _img_files.reserve(lfiles.size());
      _labels.reserve(lfiles.size());
      for (std::pair<std::string,int> &p: lfiles)
	{
	  cv::Mat img = cv::imread(p.first, _unchanged_data ? CV_LOAD_IMAGE_UNCHANGED :
                             (_bw ? CV_LOAD_IMAGE_GRAYSCALE : CV_LOAD_IMAGE_COLOR));
	  _imgs_size.push_back(std::pair<int,int>(img.rows,img.cols));
	  if (_keep_orig)
	    _orig_imgs.push_back(img);
	  cv::Mat rimg;

	  try
	    {
	      if (_scaled)
		scale(img, rimg);
	      else if (_width == 0 || _height == 0) {
		if (_width == 0 && _height == 0) {
		  // Do nothing and keep native resolution. May cause issues if batched images are different resolutions
		  rimg = img;
		} else {
		  // Resize so that the larger dimension is set to whichever (width or height) is non-zero, maintaining aspect ratio
		  // XXX - This may cause issues if batch images are different resolutions
		  size_t currMaxDim = std::max(img.rows, img.cols);
		  double scale = static_cast<double>(std::max(_width, _height)) / static_cast<double>(currMaxDim);
		  resize(img,rimg,cv::Size(),scale,scale);
		}
	      } else {
		// Resize normally to the specified width and height
		resize(img,rimg,cv::Size(_width,_height),0,0);
	      }
	    }
	  catch(...)
	    {
	      throw InputConnectorBadParamException("failed resizing image " + p.first);
	    }
	  if (_crop_width != 0 && _crop_height != 0) {
	    int widthBorder = (_width - _crop_width)/2;
	    int heightBorder = (_height - _crop_height)/2;
	    try {
	      rimg = rimg(cv::Rect(widthBorder, heightBorder, _crop_width, _crop_height));
	    } catch(...) {
	      throw InputConnectorBadParamException("failed cropping image " + p.first);
	    }
	  }
	  _imgs.push_back(std::move(rimg));
	  _img_files.push_back(p.first);
	  if (p.second >= 0)
	    _labels.push_back(p.second);
	  if (_imgs.size() % 1000 == 0)
	    _logger->info("read {} images",_imgs.size());
	}
      return 0;
    }

    int select_cv_interp() const
    {
      if (_interp == "nearest")
	return cv::INTER_NEAREST;
      else if (_interp == "linear")
	return cv::INTER_LINEAR;
      else if (_interp == "area")
	return cv::INTER_AREA;
      else if (_interp == "lanczos4")
	return cv::INTER_LANCZOS4;
      else /* if (_interp == "cubic") */
	return cv::INTER_CUBIC; // default
    }
    
    std::vector<cv::Mat> _imgs;
    std::vector<cv::Mat> _orig_imgs;
    std::vector<std::string> _img_files;
    std::vector<std::pair<int,int>> _imgs_size;
    bool _bw = false;
    bool _rgb = false;
    bool _in_mem = false;
    bool _unchanged_data = false;
    std::vector<int> _labels;
    int _width = 224;
    int _height = 224;
    int _crop_width = 0;
    int _crop_height = 0;
    float _scale = 1.0;
    bool _scaled = false;
    int _scale_min = 600;
    int _scale_max = 1000;
    bool _keep_orig = false;
    bool _b64 = false;
    std::string _interp = "cubic";
#ifdef USE_CUDA_CV
    bool _cuda = false;
#endif
    std::string _db_fname;
    std::shared_ptr<spdlog::logger> _logger;
  };
  
  class ImgInputFileConn : public InputConnectorStrategy
  {
  public:
  ImgInputFileConn()
    :InputConnectorStrategy(){}
    ImgInputFileConn(const ImgInputFileConn &i)
      :InputConnectorStrategy(i),
      _width(i._width),_height(i._height),
      _crop_width(i._crop_width),_crop_height(i._crop_height),
      _bw(i._bw),_unchanged_data(i._unchanged_data),
      _mean(i._mean),_has_mean_scalar(i._has_mean_scalar),_scale(i._scale),
      _scaled(i._scaled), _scale_min(i._scale_min), _scale_max(i._scale_max),
      _keep_orig(i._keep_orig), _interp(i._interp)
#ifdef USE_CUDA_CV
      , _cuda(i._cuda)
#endif
      {}
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
      if (ad.has("crop_width")) {
		  _crop_width = ad.get("crop_width").get<int>();
		  if (_crop_width > _width) {
			  _logger->error("Crop width must be less than or equal to width");
			  throw InputConnectorBadParamException("Crop width must be less than or equal to width");
		  }
	  }
      if (ad.has("crop_height")) {
		  _crop_height = ad.get("crop_height").get<int>();
		  if (_crop_height > _height) {
			  _logger->error("Crop height must be less than or equal to height");
			  throw InputConnectorBadParamException("Crop height must be less than or equal to height");
		  }
	  }
      if (ad.has("bw"))
	_bw = ad.get("bw").get<bool>();
      if (ad.has("rgb"))
	_rgb = ad.get("rgb").get<bool>();
      if (ad.has("unchanged_data"))
        _unchanged_data = ad.get("unchanged_data").get<bool>();
      if (ad.has("shuffle"))
	_shuffle = ad.get("shuffle").get<bool>();
      if (ad.has("seed"))
	_seed = ad.get("seed").get<int>();
      if (ad.has("test_split"))
	_test_split = ad.get("test_split").get<double>();
      if (ad.has("mean"))
	{
	  apitools::get_floats(ad, "mean", _mean);
	  _has_mean_scalar = true;
	}
      if (ad.has("std"))
	{
	  apitools::get_floats(ad, "std", _std);
	}
      
      // Variable size
      if (ad.has("scale"))
	  _scale = ad.get("scale").get<double>();
      if (ad.has("scaled") || ad.has("scale_min") || ad.has("scale_max"))
	_scaled = true;
      if (ad.has("scale_min"))
	_scale_min = ad.get("scale_min").get<int>();
      if (ad.has("scale_max"))
	_scale_max = ad.get("scale_max").get<int>();

      // whether to keep original image (for chained ops, e.g. cropping)
      if (ad.has("keep_orig"))
	_keep_orig = ad.get("keep_orig").get<bool>();

      // image interpolation method
      if (ad.has("interp"))
	_interp = ad.get("interp").get<std::string>();

      // timeout
      this->set_timeout(ad);
      
#ifdef USE_CUDA_CV
      // image resizing on GPU
      if (ad.has("cuda"))
	_cuda = ad.get("cuda").get<bool>();
#endif
    }
    
    int feature_size() const
    {
      if (_bw || _unchanged_data) {
		  // XXX: only valid for single channels
      	if (_crop_width != 0 && _crop_height != 0) return _crop_width*_crop_height;
      	else return _width*_height;
      }
      else {
	  	// RGB
	  	if (_crop_width != 0 && _crop_height != 0) return _crop_width*_crop_height*3;
	  	else return _width*_height*3;
      }
    }

    int batch_size() const
    {
      return _images.size();
    }

    int test_batch_size() const
    {
      return _test_images.size();
    }

    void get_data(const APIData &ad)
    {
      // check for raw cv::Mat
      if (ad.has("data_raw_img"))
	{
	  if (ad.has("ids"))
	    _ids = ad.get("ids").get<std::vector<std::string>>();
	  if (ad.has("meta_uris"))
	    _meta_uris = ad.get("meta_uris").get<std::vector<std::string>>();
	  if (ad.has("index_uris"))
	    _index_uris = ad.get("index_uris").get<std::vector<std::string>>();
	  
	  _images = ad.get("data_raw_img").get<std::vector<cv::Mat>>();
	  std::vector<cv::Mat> rimgs;
	  std::vector<std::string> uris;
	  int i = 0;
	  for (auto img: _images)
	    {
	      cv::Mat rimg;
	      resize(img,rimg,cv::Size(_width,_height),0,0);
	      if (_bw && rimg.channels() > 1)
		{
		  cv::Mat bwimg;
		  cv::cvtColor(rimg, bwimg, CV_BGR2GRAY);
		  rimg = bwimg;
		}
	      _images_size.push_back(std::pair<int,int>(img.rows,img.cols));
	      if (_keep_orig)
		_orig_images.push_back(std::move(img));
	      if (!_ids.empty())
		uris.push_back(_ids.at(i));
	      else
		{
		  _ids.push_back(std::to_string(i));
		  uris.push_back(_ids.back());
		}
	      rimgs.push_back(std::move(rimg));
	      ++i;
	    }
	  _images = rimgs;
	  if (!uris.empty())
	    _uris = uris;
	}
      else InputConnectorStrategy::get_data(ad);
    }
    
    void transform(const APIData &ad)
    {
      if (ad.has("parameters")) // hotplug of parameters, overriding the defaults
	{
	  APIData ad_param = ad.getobj("parameters");
	  if (ad_param.has("input"))
	    {
	      fillup_parameters(ad_param.getobj("input"));
	    }
	}

      get_data(ad);
      if (!_images.empty()) // got ready raw images
	{
	  return;
	}
      
      int catch_read = 0;
      std::string catch_msg;
      std::vector<std::string> uris;
      std::vector<std::string> meta_uris;
      std::vector<std::string> index_uris;
      std::vector<std::string> failed_uris;
#pragma omp parallel for
      for (size_t i=0;i<_uris.size();i++)
	{
	  bool no_img = false;
	  std::string u = _uris.at(i);
	  DataEl<DDImg> dimg(this->_input_timeout);
	  dimg._ctype._bw = _bw;
          dimg._ctype._rgb = _rgb;
	  dimg._ctype._unchanged_data = _unchanged_data;
	  dimg._ctype._width = _width;
	  dimg._ctype._height = _height;
	  dimg._ctype._crop_width = _crop_width;
	  dimg._ctype._crop_height = _crop_height;
	  dimg._ctype._scale = _scale;
	  dimg._ctype._scaled = _scaled;
	  dimg._ctype._scale_min = _scale_min;
	  dimg._ctype._scale_max = _scale_max;
	  dimg._ctype._keep_orig = _keep_orig;
	  dimg._ctype._interp = _interp;
#ifdef USE_CUDA_CV
	  dimg._ctype._cuda = _cuda;
#endif
	  try
	    {
	      if (dimg.read_element(u,this->_logger))
		{
		  _logger->error("no data for image {}",u);
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
		failed_uris.push_back(u);
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
	    if (_keep_orig)
	      _orig_images.insert(_orig_images.end(),
				  std::make_move_iterator(dimg._ctype._orig_imgs.begin()),
				  std::make_move_iterator(dimg._ctype._orig_imgs.end()));
	    _images_size.insert(_images_size.end(),
				std::make_move_iterator(dimg._ctype._imgs_size.begin()),
				std::make_move_iterator(dimg._ctype._imgs_size.end()));
	    if (!dimg._ctype._labels.empty())
	      _test_labels.insert(_test_labels.end(),
	      std::make_move_iterator(dimg._ctype._labels.begin()),
	      std::make_move_iterator(dimg._ctype._labels.end()));
	    if (!_ids.empty())
	      uris.push_back(_ids.at(i));
	    else if (!dimg._ctype._b64 && dimg._ctype._imgs.size() == 1)
	      uris.push_back(u);
	    else if (!dimg._ctype._img_files.empty())
	      uris.insert(uris.end(),
	      std::make_move_iterator(dimg._ctype._img_files.begin()),
	      std::make_move_iterator(dimg._ctype._img_files.end()));
	    else uris.push_back(std::to_string(i));
	    if (!_meta_uris.empty())
	      meta_uris.push_back(_meta_uris.at(i));
	    if (!_index_uris.empty())
	      index_uris.push_back(_index_uris.at(i));
	  }
	}
      if (catch_read)
	{
	  for (auto s: failed_uris)
	    _logger->error("failed reading image {}",s);
	  throw InputConnectorBadParamException(catch_msg);
	}
      _uris = uris;
      _ids = _uris; // since uris may be in different order than before transform
      _meta_uris = meta_uris;
      _index_uris = index_uris;
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
	  _logger->info("data split test size={} / remaining data size={}",_test_images.size(),_images.size());
	}
      if (_images.empty())
	throw InputConnectorBadParamException("no image could be found");
    }

    // data
    std::vector<cv::Mat> _images;
    std::vector<cv::Mat> _orig_images; /**< stored upon request. */
    std::vector<cv::Mat> _test_images;
    std::vector<int> _test_labels;
    std::vector<std::pair<int,int>> _images_size;
    
    // image parameters
    int _width = 224;
    int _height = 224;
    int _crop_width = 0;
    int _crop_height = 0;
    bool _bw = false; /**< whether to convert to black & white. */
    bool _rgb = false; /**< whether to convert to rgb. */
    bool _unchanged_data = false; /**< IMREAD_UNCHANGED flag. */
    double _test_split = 0.0; /**< auto-split of the dataset. */
    int _seed = -1; /**< shuffling seed. */
    std::vector<float> _mean; /**< mean image pixels, to be subtracted from images. */
    std::vector<float> _std; /**< std, to divide image values. */
    bool _has_mean_scalar = false; /**< whether scalar is set. */
    std::string _db_fname;
    double _scale = 1.0;
    bool _scaled = false;
    int _scale_min = 600;
    int _scale_max = 1000;
    bool _keep_orig = false;
    std::string _interp = "cubic";
#ifdef USE_CUDA_CV
    bool _cuda = false;
#endif
  };
}

#ifdef USE_CAFFE
#include "caffeinputconns.h"
#endif

#ifdef USE_TF
#include "backends/tf/tfinputconns.h"
#endif

#ifdef USE_DLIB
#include "backends/dlib/dlibinputconns.h"
#endif

#ifdef USE_NCNN
#include "backends/ncnn/ncnninputconns.h"
#endif

#ifdef USE_CAFFE2
#include "backends/caffe2/caffe2inputconns.h"
#endif

#ifdef USE_TENSORRT
#include "backends/tensorrt/tensorrtinputconns.h"
#endif

#endif
