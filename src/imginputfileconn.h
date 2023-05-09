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
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#endif
#if CV_VERSION_MAJOR >= 3
#define CV_LOAD_IMAGE_COLOR cv::IMREAD_COLOR
#define CV_LOAD_IMAGE_GRAYSCALE cv::IMREAD_GRAYSCALE
#define CV_LOAD_IMAGE_UNCHANGED cv::IMREAD_UNCHANGED
#define CV_BGR2RGB cv::COLOR_BGR2RGB
#define CV_BGR2GRAY cv::COLOR_BGR2GRAY
#define CV_GRAY2RGB cv::COLOR_GRAY2RGB
#define CV_YCrCb2RGB cv::COLOR_YCrCb2RGB
#define CV_YCrCb2BGR cv::COLOR_YCrCb2BGR
#define CV_BGR2YCrCb cv::COLOR_BGR2YCrCb
#define CV_INTER_CUBIC cv::INTER_CUBIC
#endif
#include "ext/base64/base64.h"
#include "utils/apitools.h"
#include <random>

#include "dto/input_connector.hpp"

namespace dd
{

  class DDImg
  {
  public:
    DDImg()
    {
    }
    ~DDImg()
    {
    }

    // base64 detection
    bool is_within_base64_range(char c) const
    {
      if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')
          || (c >= '0' && c <= '9') || (c == '+' || c == '/' || c == '='))
        return true;
      else
        return false;
    }

    bool possibly_base64(const std::string &s) const
    {
      bool ism = is_multiple_four(s);
      if (!ism)
        return false;
      for (char c : s)
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
      else
        return false;
    }

    /** apply preprocessing to image */
    void prepare(const cv::Mat &src, cv::Mat &dst,
                 const std::string &img_name) const
    {
      try
        {
          if (_scaled)
            scale(src, dst);
          else if (_width < 0 || _height < 0)
            {
              if (_width < 0 && _height < 0)
                {
                  // Do nothing and keep native resolution. May cause issues if
                  // batched images are different resolutions
                  dst = src;
                }
              else
                {
                  // Resize so that the larger dimension is set to whichever
                  // (width or height) is non-zero, maintaining aspect ratio
                  // XXX - This may cause issues if batch images are different
                  // resolutions
                  size_t currMaxDim = std::max(src.rows, src.cols);
                  double scale = static_cast<double>(std::max(_width, _height))
                                 / static_cast<double>(currMaxDim);
                  cv::resize(src, dst, cv::Size(), scale, scale,
                             select_cv_interp());
                }
            }
          else
            {
              // Resize normally to the specified width and height
              cv::resize(src, dst, cv::Size(_width, _height), 0, 0,
                         select_cv_interp());
            }
        }
      catch (...)
        {
          throw InputConnectorBadParamException("failed resizing image "
                                                + img_name);
        }

      // cropping
      if (_crop_width != 0 && _crop_height != 0)
        {
          int widthBorder = (_width - _crop_width) / 2;
          int heightBorder = (_height - _crop_height) / 2;
          try
            {
              dst = dst(cv::Rect(widthBorder, heightBorder, _crop_width,
                                 _crop_height));
            }
          catch (...)
            {
              throw InputConnectorBadParamException("failed cropping image "
                                                    + img_name);
            }
        }

      // color adjustments
      if (_bw && dst.channels() > 1)
        {
          cv::cvtColor(dst, dst, CV_BGR2GRAY);
        }

      if (_histogram_equalization)
        {
          if (_bw)
            {
              cv::equalizeHist(dst, dst);
              if (_rgb)
                cv::cvtColor(dst, dst, CV_GRAY2RGB);
            }
          else
            {
              // We don't apply equalizeHist on each BGR channels to keep
              // the color balance of the image. equalizeHist(V) of HSV can
              // works too, the result is almost the same
              cv::cvtColor(dst, dst, CV_BGR2YCrCb);
              std::vector<cv::Mat> vec_channels;
              cv::split(dst, vec_channels);
              cv::equalizeHist(vec_channels[0], vec_channels[0]);
              cv::merge(vec_channels, dst);
              if (_rgb)
                cv::cvtColor(dst, dst, CV_YCrCb2RGB);
              else
                cv::cvtColor(dst, dst, CV_YCrCb2BGR);
            }
        }
      else if (_rgb)
        {
          if (_bw)
            cv::cvtColor(dst, dst, CV_GRAY2RGB);
          else
            cv::cvtColor(dst, dst, CV_BGR2RGB);
        }
    }

#ifdef USE_CUDA_CV
    /** apply preprocessing to cuda image */
    void prepare_cuda(const cv::cuda::GpuMat &src, cv::cuda::GpuMat &dst,
                      const std::string &img_name) const
    {
      try
        {
          if (_scaled)
            scale_cuda(src, dst);
          else if (_width < 0 || _height < 0)
            {
              if (_width < 0 && _height < 0)
                {
                  // Do nothing and keep native resolution. May cause issues if
                  // batched images are different resolutions
                  dst = src;
                }
              else
                {
                  // Resize so that the larger dimension is set to whichever
                  // (width or height) is non-zero, maintaining aspect ratio
                  // XXX - This may cause issues if batch images are different
                  // resolutions
                  size_t currMaxDim = std::max(src.rows, src.cols);
                  double scale = static_cast<double>(std::max(_width, _height))
                                 / static_cast<double>(currMaxDim);
                  cv::cuda::resize(src, dst, cv::Size(), scale, scale,
                                   select_cv_interp(), *_cuda_stream);
                }
            }
          else
            {
              // Resize normally to the specified width and height
              cv::cuda::resize(src, dst, cv::Size(_width, _height), 0, 0,
                               select_cv_interp(), *_cuda_stream);
            }
        }
      catch (...)
        {
          throw InputConnectorBadParamException("failed resizing image "
                                                + img_name);
        }

      // cropping
      if (_crop_width != 0 && _crop_height != 0)
        {
          int widthBorder = (_width - _crop_width) / 2;
          int heightBorder = (_height - _crop_height) / 2;
          try
            {
              // TODO cuda crop with stream
              dst = dst(cv::Rect(widthBorder, heightBorder, _crop_width,
                                 _crop_height));
            }
          catch (...)
            {
              throw InputConnectorBadParamException("failed cropping image "
                                                    + img_name);
            }
        }

      // color adjustment
      if (_bw && dst.channels() > 1)
        {
          cv::cuda::cvtColor(dst, dst, CV_BGR2GRAY, 0, *_cuda_stream);
        }

      if (_histogram_equalization)
        {
          if (_bw)
            {
              cv::cuda::equalizeHist(dst, dst, *_cuda_stream);
              if (_rgb)
                cv::cuda::cvtColor(dst, dst, CV_GRAY2RGB, 0, *_cuda_stream);
            }
          else
            {
              // We don't apply equalizeHist on each BGR channels to keep
              // the color balance of the image. equalizeHist(V) of HSV can
              // works too, the result is almost the same
              cv::cuda::cvtColor(dst, dst, CV_BGR2YCrCb, 0, *_cuda_stream);
              std::vector<cv::cuda::GpuMat> vec_channels;
              cv::cuda::split(dst, vec_channels, *_cuda_stream);
              cv::cuda::equalizeHist(vec_channels[0], vec_channels[0],
                                     *_cuda_stream);
              cv::cuda::merge(vec_channels, dst, *_cuda_stream);
              if (_rgb)
                cv::cuda::cvtColor(dst, dst, CV_YCrCb2RGB, 0, *_cuda_stream);
              else
                cv::cuda::cvtColor(dst, dst, CV_YCrCb2BGR, 0, *_cuda_stream);
            }
        }
      else if (_rgb)
        {
          if (_bw)
            cv::cuda::cvtColor(dst, dst, CV_GRAY2RGB, 0, *_cuda_stream);
          else
            cv::cuda::cvtColor(dst, dst, CV_BGR2RGB, 0, *_cuda_stream);
        }
    }
#endif

    void scale(const cv::Mat &src, cv::Mat &dst) const
    {
      float coef = std::min(
          static_cast<float>(_scale_max) / std::max(src.rows, src.cols),
          static_cast<float>(_scale_min) / std::min(src.rows, src.cols));

      cv::resize(src, dst, cv::Size(), coef, coef, select_cv_interp());
    }

#ifdef USE_CUDA_CV
    void scale_cuda(const cv::cuda::GpuMat &src, cv::cuda::GpuMat &dst) const
    {
      float coef = std::min(
          static_cast<float>(_scale_max) / std::max(src.rows, src.cols),
          static_cast<float>(_scale_min) / std::min(src.rows, src.cols));

      cv::cuda::resize(src, dst, cv::Size(), coef, coef, select_cv_interp(),
                       *_cuda_stream);
    }
#endif

    /// Apply preprocessing to image and add it to the list of images
    /// img_name: name of the image as displayed in error messages
    int add_image(const cv::Mat &img, const std::string &img_name)
    {
      if (img.empty())
        {
          _logger->error("empty image {}", img_name);
          return -1;
        }
      _imgs_size.push_back(std::pair<int, int>(img.rows, img.cols));

#ifdef USE_CUDA_CV
      if (_cuda)
        {
          cv::cuda::GpuMat d_src;
          d_src.upload(img);

          if (_keep_orig)
            _cuda_orig_imgs.push_back(d_src);

          cv::cuda::GpuMat d_dst;
          prepare_cuda(d_src, d_dst, img_name);

          _cuda_imgs.push_back(std::move(d_dst));
        }
      else
#endif
        {
          if (_keep_orig)
            _orig_imgs.push_back(img);

          cv::Mat rimg;
          prepare(img, rimg, img_name);
          _imgs.push_back(std::move(rimg));
        }
      return 0;
    }

#ifdef USE_CUDA_CV
    /// add_image but directly from a cv::cuda::GpuMat
    int add_image_cuda(const cv::cuda::GpuMat &d_src,
                       const std::string &img_name)
    {
      _imgs_size.push_back(std::pair<int, int>(d_src.rows, d_src.cols));
      if (_keep_orig)
        _cuda_orig_imgs.push_back(d_src);

      cv::cuda::GpuMat d_dst;
      prepare_cuda(d_src, d_dst, img_name);
      _cuda_imgs.push_back(std::move(d_dst));
      return 0;
    }
#endif

    // decode image
    void decode(const std::string &str)
    {
      std::vector<unsigned char> vdat(str.begin(), str.end());
      cv::Mat img = cv::Mat(cv::imdecode(
          cv::Mat(vdat, false),
          _unchanged_data
              ? CV_LOAD_IMAGE_UNCHANGED
              : (_bw ? CV_LOAD_IMAGE_GRAYSCALE : CV_LOAD_IMAGE_COLOR)));
      add_image(img, "base64 image");
    }

    // deserialize image, independent of format
    void deserialize(std::stringstream &input)
    {
      size_t size = 0;
      input.seekg(0, input.end);
      size = input.tellg();
      input.seekg(0, input.beg);
      char *data = new char[size];
      input.read(data, size);
      std::string str(data, data + size);
      delete[] data;
      decode(str);
    }

    // data acquisition
    int read_file(const std::string &fname, int test_id)
    {
      (void)test_id;
      cv::Mat img
          = cv::imread(fname, _unchanged_data ? CV_LOAD_IMAGE_UNCHANGED
                                              : (_bw ? CV_LOAD_IMAGE_GRAYSCALE
                                                     : CV_LOAD_IMAGE_COLOR));
      return add_image(img, fname);
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
          Base64::Decode(content, &ccontent);
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

    int read_dir(const std::string &dir, int test_id)
    {
      (void)test_id;
      // list directories in dir
      std::unordered_set<std::string> subdirs;
      if (fileops::list_directory(dir, false, true, false, subdirs))
        throw InputConnectorBadParamException(
            "failed reading text subdirectories in data directory " + dir);
      _logger->info("imginputfileconn: list subdirs size={}", subdirs.size());

      // list files and classes
      std::vector<std::pair<std::string, int>> lfiles; // labeled files
      std::unordered_map<int, std::string>
          hcorresp; // correspondence class number / class name
      if (!subdirs.empty())
        {
          int cl = 0;
          auto uit = subdirs.begin();
          while (uit != subdirs.end())
            {
              std::unordered_set<std::string> subdir_files;
              if (fileops::list_directory((*uit), true, false, true,
                                          subdir_files))
                throw InputConnectorBadParamException(
                    "failed reading image data sub-directory " + (*uit));
              auto fit = subdir_files.begin();
              while (fit != subdir_files.end()) // XXX: re-iterating the file
                                                // is not optimal
                {
                  lfiles.push_back(std::pair<std::string, int>((*fit), cl));
                  ++fit;
                }
              ++cl;
              ++uit;
            }
        }
      else
        {
          std::unordered_set<std::string> test_files;
          fileops::list_directory(dir, true, false, false, test_files);
          auto fit = test_files.begin();
          while (fit != test_files.end())
            {
              lfiles.push_back(
                  std::pair<std::string, int>((*fit), -1)); // -1 for no class
              ++fit;
            }
        }

      // read images
      _imgs.reserve(lfiles.size());
      _img_files.reserve(lfiles.size());
      _labels.reserve(lfiles.size());
      for (std::pair<std::string, int> &p : lfiles)
        {
          cv::Mat img = cv::imread(
              p.first, _unchanged_data ? CV_LOAD_IMAGE_UNCHANGED
                                       : (_bw ? CV_LOAD_IMAGE_GRAYSCALE
                                              : CV_LOAD_IMAGE_COLOR));
          add_image(img, p.first);
          _img_files.push_back(p.first);
          if (p.second >= 0)
            _labels.push_back(p.second);
          if (_imgs.size() % 1000 == 0)
            _logger->info("read {} images", _imgs.size());
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
      else                      /* if (_interp == "cubic") */
        return cv::INTER_CUBIC; // default
    }

    std::vector<cv::Mat> _imgs;
    std::vector<cv::Mat> _orig_imgs;
    std::vector<std::string> _img_files;
    std::vector<std::pair<int, int>> _imgs_size;
    bool _bw = false;
    bool _rgb = false;
    bool _histogram_equalization = false;
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
    std::vector<cv::cuda::GpuMat> _cuda_imgs;
    std::vector<cv::cuda::GpuMat> _cuda_orig_imgs;
    cv::cuda::Stream *_cuda_stream = nullptr;
#endif
    std::string _db_fname;
    std::shared_ptr<spdlog::logger> _logger;
  };

  class ImgInputFileConn : public InputConnectorStrategy
  {
  public:
    ImgInputFileConn() : InputConnectorStrategy()
    {
    }
    ImgInputFileConn(const ImgInputFileConn &i)
        : InputConnectorStrategy(i), _width(i._width), _height(i._height),
          _crop_width(i._crop_width), _crop_height(i._crop_height), _bw(i._bw),
          _rgb(i._rgb), _unchanged_data(i._unchanged_data),
          _test_split(i._test_split), _mean(i._mean),
          _has_mean_scalar(i._has_mean_scalar), _scale(i._scale),
          _scaled(i._scaled), _scale_min(i._scale_min),
          _scale_max(i._scale_max), _keep_orig(i._keep_orig),
          _interp(i._interp)
#ifdef USE_CUDA_CV
          ,
          _cuda(i._cuda)
#endif
    {
    }
    ~ImgInputFileConn()
    {
    }

    void init(const APIData &ad)
    {
      fillup_parameters(ad);
    }

    void fillup_parameters(const APIData &ad)
    {
      auto params = ad.createSharedDTO<dd::DTO::InputConnector>();
      fillup_parameters(params);
    }

    void fillup_parameters(oatpp::Object<DTO::InputConnector> params)
    {
      // optional parameters.
      if (params->width)
        _width = params->width;
      if (params->height)
        _height = params->height;
      if (params->crop_width)
        {
          if (params->crop_width > _width)
            {
              _logger->error("Crop width must be less than or equal to width");
              throw InputConnectorBadParamException(
                  "Crop width must be less than or equal to width");
            }
          _width = params->crop_width;
        }
      if (params->crop_height)
        {
          if (params->crop_height > _height)
            {
              _logger->error(
                  "Crop height must be less than or equal to height");
              throw InputConnectorBadParamException(
                  "Crop height must be less than or equal to height");
            }
          _height = params->crop_height;
        }

      if (params->bw != nullptr)
        _bw = params->bw;
      if (params->rgb != nullptr)
        _rgb = params->rgb;
      if (params->histogram_equalization != nullptr)
        _histogram_equalization = params->histogram_equalization;
      if (params->unchanged_data != nullptr)
        _unchanged_data = params->unchanged_data;
      if (params->shuffle != nullptr)
        _shuffle = params->shuffle;
      if (params->seed)
        _seed = params->seed;
      if (params->test_split)
        _test_split = params->test_split;
      if (params->mean)
        {
          // NOTE(sileht): if we have two much of this we can create
          // an oat++ type that directly handle std::vector<float> instead
          // of using the oatpp::Vector<oatpp::Float32>
          _mean = std::vector<float>();
          for (auto &v : *params->mean)
            _mean.push_back(v);
          _has_mean_scalar = true;
        }
      if (params->std)
        {
          _std = std::vector<float>();
          for (auto &v : *params->std)
            _std.push_back(v);
        }

      // Variable size
      _scaled |= params->scaled;
      if (params->scale)
        try
          {
            _scale = params->scale.retrieve<oatpp::Float64>();
          }
        catch (const std::runtime_error &error)
          {
            std::string msg
                = "could not read double value for scale input parameter";
            _logger->error(msg);
            throw InputConnectorBadParamException(msg);
          }
      if (params->scale_min)
        {
          _scaled = true;
          _scale_min = params->scale_min;
        }
      if (params->scale_max)
        {
          _scaled = true;
          _scale_max = params->scale_max;
        }

      // whether to keep original image (for chained ops, e.g. cropping)
      _keep_orig |= params->keep_orig;

      // image interpolation method
      if (params->interp)
        _interp = params->interp;

      // timeout
      this->set_timeout(params);

#ifdef USE_CUDA_CV
      // image resizing on GPU
      _cuda |= params->cuda;
#endif
    }

    void copy_parameters_to(DDImg &dimg) const
    {
      dimg._bw = _bw;
      dimg._rgb = _rgb;
      dimg._histogram_equalization = _histogram_equalization;
      dimg._unchanged_data = _unchanged_data;
      dimg._width = _width;
      dimg._height = _height;
      dimg._crop_width = _crop_width;
      dimg._crop_height = _crop_height;
      dimg._scale = _scale;
      dimg._scaled = _scaled;
      dimg._scale_min = _scale_min;
      dimg._scale_max = _scale_max;
      dimg._keep_orig = _keep_orig;
      dimg._interp = _interp;
#ifdef USE_CUDA_CV
      dimg._cuda = _cuda;
      dimg._cuda_stream = _cuda_stream;
#endif
      dimg._logger = _logger;
    }

    int feature_size() const
    {
      if (_bw || _unchanged_data)
        {
          // XXX: only valid for single channels
          if (_crop_width != 0 && _crop_height != 0)
            return _crop_width * _crop_height;
          else
            return _width * _height;
        }
      else
        {
          // RGB
          if (_crop_width != 0 && _crop_height != 0)
            return _crop_width * _crop_height * 3;
          else
            return _width * _height * 3;
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

    // add cuda raw images
    void add_raw_images(const std::vector<cv::Mat> &imgs
#ifdef USE_CUDA_CV
                        ,
                        const std::vector<cv::cuda::GpuMat> &cuda_imgs
#endif
    )
    {

      std::vector<std::string> uris;
      DataEl<DDImg> dimg(this->_input_timeout);
      copy_parameters_to(dimg._ctype);
      int i = 0;

      // preprocess
#ifdef USE_CUDA_CV
      for (auto cuda_img : cuda_imgs)
        {
          if (!_ids.empty())
            uris.push_back(_ids.at(i));
          else
            {
              _ids.push_back(std::to_string(i));
              uris.push_back(_ids.back());
            }

          dimg._ctype.add_image_cuda(cuda_img, _ids.back());
          ++i;
        }
#endif

      for (auto img : imgs)
        {
          if (!_ids.empty())
            uris.push_back(_ids.at(i));
          else
            {
              _ids.push_back(std::to_string(i));
              uris.push_back(_ids.back());
            }
          dimg._ctype.add_image(img, _ids.back());
          ++i;
        }

        // add preprocessed images
#ifdef USE_CUDA_CV
      if (_cuda)
        {
          if (_keep_orig)
            _cuda_orig_images.insert(_cuda_orig_images.end(),
                                     dimg._ctype._cuda_orig_imgs.begin(),
                                     dimg._ctype._cuda_orig_imgs.end());
          _cuda_images.insert(_cuda_images.end(),
                              dimg._ctype._cuda_imgs.begin(),
                              dimg._ctype._cuda_imgs.end());
        }
      else
#endif
        {
          if (_keep_orig)
            _orig_images = dimg._ctype._orig_imgs;
          _images = dimg._ctype._imgs;
        }
      _images_size.insert(_images_size.end(), dimg._ctype._imgs_size.begin(),
                          dimg._ctype._imgs_size.end());
      if (!uris.empty())
        _uris = uris;
    }

    void get_data(oatpp::Object<DTO::ServicePredict> pred_in)
    {
      if (!pred_in->_data_raw_img.empty()
#ifdef USE_CUDA_CV
          || !pred_in->_data_raw_img_cuda.empty()
#endif
      )
        {
          _ids = pred_in->_ids;
          _meta_uris = pred_in->_meta_uris;
          _index_uris = pred_in->_index_uris;

          add_raw_images(pred_in->_data_raw_img
#ifdef USE_CUDA_CV
                         ,
                         pred_in->_data_raw_img_cuda
#endif
          );
        }
      else
        InputConnectorStrategy::get_data(pred_in);
    }

    void get_data(const APIData &ad)
    {
      // check for raw cv::Mat
      if (ad.has("data_raw_img")
#ifdef USE_CUDA_CV
          || ad.has("data_raw_img_cuda")
#endif
      )
        {
          if (ad.has("ids"))
            _ids = ad.get("ids").get<std::vector<std::string>>();
          if (ad.has("meta_uris"))
            _meta_uris = ad.get("meta_uris").get<std::vector<std::string>>();
          if (ad.has("index_uris"))
            _index_uris = ad.get("index_uris").get<std::vector<std::string>>();

          std::vector<cv::Mat> imgs
              = ad.has("data_raw_img")
                    ? ad.get("data_raw_img").get<std::vector<cv::Mat>>()
                    : std::vector<cv::Mat>();
#ifdef USE_CUDA_CV
          std::vector<cv::cuda::GpuMat> cuda_imgs
              = ad.has("data_raw_img_cuda")
                    ? ad.get("data_raw_img_cuda")
                          .get<std::vector<cv::cuda::GpuMat>>()
                    : std::vector<cv::cuda::GpuMat>();
          add_raw_images(imgs, cuda_imgs);
#else
          add_raw_images(imgs);
#endif
        }
      else
        InputConnectorStrategy::get_data(ad);
    }

    void transform(const APIData &ad)
    {
      if (ad.has(
              "parameters")) // hotplug of parameters, overriding the defaults
        {
          APIData ad_param = ad.getobj("parameters");
          if (ad_param.has("input"))
            {
              fillup_parameters(ad_param.getobj("input"));
            }
        }

      get_data(ad);
      transform(nullptr);
    }

    void transform(oatpp::Object<DTO::ServicePredict> input_dto)
    {

      if (input_dto != nullptr) // [temporary] == nullptr if called from
                                // transform(APIData)
        {
          fillup_parameters(input_dto->parameters->input);
          get_data(input_dto);
        }

      if (!_images.empty() // got ready raw images
#ifdef USE_CUDA_CV
          || !_cuda_images.empty() // got ready cuda images
#endif
      )
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
      for (size_t i = 0; i < _uris.size(); i++)
        {
          bool no_img = false;
          std::string u = _uris.at(i);
          DataEl<DDImg> dimg(this->_input_timeout);
          copy_parameters_to(dimg._ctype);

          try
            {
              if (dimg.read_element(u, this->_logger))
                {
                  _logger->error("no data for image {}", u);
                  no_img = true;
                }
              if (!dimg._ctype._db_fname.empty())
                _db_fname = dimg._ctype._db_fname;
            }
          catch (std::exception &e)
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
#ifdef USE_CUDA_CV
            if (_cuda)
              {
                _cuda_images.insert(
                    _cuda_images.end(),
                    std::make_move_iterator(dimg._ctype._cuda_imgs.begin()),
                    std::make_move_iterator(dimg._ctype._cuda_imgs.end()));
                _cuda_orig_images.insert(
                    _cuda_orig_images.end(),
                    std::make_move_iterator(
                        dimg._ctype._cuda_orig_imgs.begin()),
                    std::make_move_iterator(
                        dimg._ctype._cuda_orig_imgs.end()));
              }
            else
#endif
              {
                _images.insert(
                    _images.end(),
                    std::make_move_iterator(dimg._ctype._imgs.begin()),
                    std::make_move_iterator(dimg._ctype._imgs.end()));
                if (_keep_orig)
                  _orig_images.insert(
                      _orig_images.end(),
                      std::make_move_iterator(dimg._ctype._orig_imgs.begin()),
                      std::make_move_iterator(dimg._ctype._orig_imgs.end()));
              }

            _images_size.insert(
                _images_size.end(),
                std::make_move_iterator(dimg._ctype._imgs_size.begin()),
                std::make_move_iterator(dimg._ctype._imgs_size.end()));
            if (!dimg._ctype._labels.empty())
              _test_labels.insert(
                  _test_labels.end(),
                  std::make_move_iterator(dimg._ctype._labels.begin()),
                  std::make_move_iterator(dimg._ctype._labels.end()));
            if (!_ids.empty())
              uris.push_back(_ids.at(i));
            else if (!dimg._ctype._b64 && dimg._ctype._imgs.size() == 1)
              uris.push_back(u);
            else if (!dimg._ctype._img_files.empty())
              uris.insert(
                  uris.end(),
                  std::make_move_iterator(dimg._ctype._img_files.begin()),
                  std::make_move_iterator(dimg._ctype._img_files.end()));
            else
              uris.push_back(std::to_string(i));
            if (!_meta_uris.empty())
              meta_uris.push_back(_meta_uris.at(i));
            if (!_index_uris.empty())
              index_uris.push_back(_index_uris.at(i));
          }
        }
      if (catch_read)
        {
          for (auto s : failed_uris)
            _logger->error("failed reading image {}", s);
          throw InputConnectorBadParamException(catch_msg);
        }
      _uris = uris;
      _ids = _uris; // since uris may be in different order than before
                    // transform
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
          std::shuffle(_images.begin(), _images.end(),
                       g); // XXX beware: labels are not shuffled, i.e. let's
                           // not shuffle while testing
        }
      // split as required
      if (_test_split > 0)
        {
          int split_size = std::floor(_images.size() * (1.0 - _test_split));
          auto chit = _images.begin();
          auto dchit = chit;
          int cpos = 0;
          while (chit != _images.end())
            {
              if (cpos == split_size)
                {
                  if (dchit == _images.begin())
                    dchit = chit;
                  _test_images.push_back((*chit));
                }
              else
                ++cpos;
              ++chit;
            }
          _images.erase(dchit, _images.end());
          _logger->info("data split test size={} / remaining data size={}",
                        _test_images.size(), _images.size());
        }
      if (_images.empty()
#ifdef USE_CUDA_CV
          && _cuda_images.empty()
#endif
      )
        throw InputConnectorBadParamException("no image could be found");
    }

    static std::vector<double>
    img_resize_vector(const std::vector<double> &vals, const int height_net,
                      const int width_net, const int height_dest,
                      const int width_dest, bool resize_nn)
    {
      cv::Mat segimg = cv::Mat(height_net, width_net, CV_64FC1);
      std::memcpy(segimg.data, vals.data(), vals.size() * sizeof(double));
      cv::Mat segimg_res;
      if (resize_nn)
        cv::resize(segimg, segimg_res, cv::Size(width_dest, height_dest), 0, 0,
                   cv::INTER_NEAREST);
      else
        cv::resize(segimg, segimg_res, cv::Size(width_dest, height_dest), 0, 0,
                   cv::INTER_LINEAR);
      return std::vector<double>((double *)segimg_res.data,
                                 (double *)segimg_res.data
                                     + segimg_res.rows * segimg_res.cols);
    }

    // data
    std::vector<cv::Mat> _images;
    std::vector<cv::Mat> _orig_images; /**< stored upon request. */
    std::vector<cv::Mat> _test_images;
    std::vector<int> _test_labels;
    std::vector<std::pair<int, int>> _images_size;
#ifdef USE_CUDA_CV
    std::vector<cv::cuda::GpuMat>
        _cuda_images; /**< cuda images for full-GPU processing. */
    std::vector<cv::cuda::GpuMat>
        _cuda_orig_images; /**< original images stored on GPU */
#endif

    // image parameters
    int _width = 224;
    int _height = 224;
    int _crop_width = 0;
    int _crop_height = 0;
    bool _bw = false;  /**< whether to convert to black & white. */
    bool _rgb = false; /**< whether to convert to rgb. */
    bool _histogram_equalization
        = false;                  /**< whether to apply histogram equalizer. */
    bool _unchanged_data = false; /**< IMREAD_UNCHANGED flag. */
    double _test_split = 0.0;     /**< auto-split of the dataset. */
    int _seed = -1;               /**< shuffling seed. */
    std::vector<float>
        _mean; /**< mean image pixels, to be subtracted from images. */
    std::vector<float> _std;       /**< std, to divide image values. */
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
    cv::cuda::Stream *_cuda_stream = &cv::cuda::Stream::Null();
#endif
  };
}

#endif
