/**
 * DeepDetect
 * Copyright (c) 2018 Jolibrain
 * Author: Corentin Barreau <corentin.barreau@epitech.eu>
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

#ifndef NCNNINPUTCONNS_H
#define NCNNINPUTCONNS_H

#include "imginputfileconn.h"
#include "csvtsinputfileconn.h"

// NCNN
#include "net.h"

namespace dd
{
  class NCNNInputInterface
  {
  public:
    NCNNInputInterface()
    {
    }
    NCNNInputInterface(const NCNNInputInterface &i)
    {
      _timeseries_lengths = i._timeseries_lengths;
      _continuation = i._continuation;
      _ntargets = i._ntargets;
    }

    ~NCNNInputInterface()
    {
    }

    double unscale_res(double res, int nout);

    std::vector<int> _timeseries_lengths;
    bool _continuation = false;
    int _ntargets;
    std::unordered_map<std::string, std::pair<int, int>>
        _imgs_size; /**< image sizes, used in detection. */
  };

  class ImgNCNNInputFileConn : public ImgInputFileConn,
                               public NCNNInputInterface
  {
  public:
    ImgNCNNInputFileConn() : ImgInputFileConn()
    {
    }
    ImgNCNNInputFileConn(const ImgNCNNInputFileConn &i)
        : ImgInputFileConn(i), NCNNInputInterface(i)
    {
    }
    ~ImgNCNNInputFileConn()
    {
    }

    // for API info only
    int width() const
    {
      return _width;
    }

    // for API info only
    int height() const
    {
      return _height;
    }

    void init(const APIData &ad)
    {
      ImgInputFileConn::init(ad);
    }

    void transform(const APIData &ad)
    {
      try
        {
          ImgInputFileConn::transform(ad);
        }
      catch (const std::exception &e)
        {
          throw;
        }

      for (size_t i = 0; i < _images.size(); ++i)
        {
          cv::Mat bgr = this->_images.at(i);
          _height = bgr.rows;
          _width = bgr.cols;

          _in.push_back(ncnn::Mat::from_pixels(bgr.data, ncnn::Mat::PIXEL_BGR,
                                               bgr.cols, bgr.rows));
          {
            if (_has_mean_scalar)
              {
                if (_std.empty())
                  {
                    _in.at(i).substract_mean_normalize(_mean.data(), 0);
                  }
                else
                  {
                    _in.at(i).substract_mean_normalize(_mean.data(),
                                                       _std.data());
                  }
              }
            else if (_scale != 1.0)
              {
                float norm = 1.0 / _scale;
                if (_bw)
                  {
                    std::vector<float> vscale = { norm };
                    _in.at(i).substract_mean_normalize(0, vscale.data());
                  }
                else
                  {
                    std::vector<float> vscale = { norm, norm, norm };
                    _in.at(i).substract_mean_normalize(0, vscale.data());
                  }
              }
            _ids.push_back(this->_uris.at(i));
            _imgs_size.insert(std::pair<std::string, std::pair<int, int>>(
                this->_ids.at(i), this->_images_size.at(i)));
          }
          _out = std::vector<ncnn::Mat>(_ids.size(), ncnn::Mat());
        }
    }

    double unscale_res(double res, int nout)
    {
      return 0 * res * nout;
    }

  public:
    std::vector<ncnn::Mat> _in;
    std::vector<ncnn::Mat> _out;
    std::vector<std::string> _ids; /**< input ids (e.g. image ids) */
  };

  class CSVTSNCNNInputFileConn : public CSVTSInputFileConn,
                                 public NCNNInputInterface
  {
  public:
    CSVTSNCNNInputFileConn() : CSVTSInputFileConn()
    {
    }
    CSVTSNCNNInputFileConn(const CSVTSNCNNInputFileConn &i)
        : CSVTSInputFileConn(i), NCNNInputInterface(i), _in(i._in),
          _out(i._out), _height(i._height), _width(i._width), _ids(i._ids)
    {
    }
    ~CSVTSNCNNInputFileConn()
    {
    }

    // for API info only
    int width() const
    {
      return _width;
    }

    // for API info only
    int height() const
    {
      return _height;
    }

    void init(const APIData &ad)
    {
      CSVTSInputFileConn::init(ad);
      if (ad.has("continuation"))
        _continuation = ad.get("continuation").get<bool>();
    }

    void transform(const APIData &ad)
    {
      try
        {
          CSVTSInputFileConn::transform(ad);
        }
      catch (const std::exception &e)
        {
          throw;
        }

      APIData ad_input = ad.getobj("parameters").getobj("input");
      if (ad_input.has("continuation"))
        _continuation = ad_input.get("continuation").get<bool>();

      //  data should be is in this->_csvtsdata
      int nseries = this->_csvtsdata.size();
      _height = 0;
      _width = this->_csvtsdata[0][0]._v.size() + 1;

      for (int i = 0; i < nseries; ++i)
        {
          int l = this->_csvtsdata[i].size();
          _timeseries_lengths.push_back(l);
          _height += l;
        }
      // Mat(w,h)
      _in.emplace_back(_width, _height);
      _out.emplace_back();

      int mati = 0;

      // only inputs are put into _in
      _ntargets = this->_label_pos.size();
      std::vector<int> input_pos;
      for (int i = 0; i < _width - 1; ++i)
        if (std::find(this->_label_pos.begin(), this->_label_pos.end(), i)
            == this->_label_pos.end())
          input_pos.push_back(i);

      for (unsigned int si = 0; si < this->_csvtsdata.size(); ++si)
        {
          if (_continuation)
            _in.at(0)[mati++] = 1.0;
          else
            _in.at(0)[mati++] = 0.0;
          for (int di = 0; di < _ntargets; ++di)
            _in.at(0)[mati++] = 0.0;
          for (unsigned int di : input_pos)
            _in.at(0)[mati++] = this->_csvtsdata[si][0]._v[di];
          for (unsigned int ti = 1; ti < this->_csvtsdata[si].size(); ++ti)
            {
              _in.at(0)[mati++] = 1.0;
              for (int di = 0; di < _ntargets; ++di)
                _in.at(0)[mati++] = 0.0;
              for (unsigned int di : input_pos)
                _in.at(0)[mati++] = this->_csvtsdata[si][ti]._v[di];
            }
        }
      _ids.push_back(this->_uris.at(0));
    }

    double unscale_res(double res, int k)
    {
      if (_dont_scale_labels)
        return res;
      if (_min_vals.empty() || _max_vals.empty())
        return res;
      double min = _min_vals[_label_pos[k]];
      if (_scale_between_minus1_and_1)
        return (res + 0.5) * (_max_vals[_label_pos[k]] - min) + min;
      else
        return res * (_max_vals[_label_pos[k]] - min) + min;
    }

  public:
    std::vector<ncnn::Mat> _in;
    std::vector<ncnn::Mat> _out;
    int _height;
    int _width;
    std::vector<std::string> _ids; /**< input ids (e.g. image ids) */
  };

}

#endif
