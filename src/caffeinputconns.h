/**
 * DeepDetect
 * Copyright (c) 2014-2015 Emmanuel Benazera
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

#ifndef CAFFEINPUTCONNS_H
#define CAFFEINPUTCONNS_H

#include "imginputfileconn.h"
#include "csvinputfileconn.h"
#include "caffe/caffe.hpp"

namespace dd
{
  class ImgCaffeInputFileConn : public ImgInputFileConn
  {
  public:
    ImgCaffeInputFileConn()
      :ImgInputFileConn() {}
    ImgCaffeInputFileConn(const ImgCaffeInputFileConn &i)
      :ImgInputFileConn(i),_dv(i._dv) {}
    ~ImgCaffeInputFileConn() {}

    void init(const APIData &ad)
    {
      ImgInputFileConn::init(ad);
    }

    int transform(const APIData &ad)
    {
      ImgInputFileConn::transform(ad);
      for (int i=0;i<this->size();i++)
      {      
	caffe::Datum datum;
	caffe::CVMatToDatum(this->_images.at(i),&datum);
	_dv.push_back(datum);
      }
      return 0;
    }
    
    std::vector<caffe::Datum> _dv;
  };

  class CSVCaffeInputFileConn : public CSVInputFileConn
  {
  public:
    CSVCaffeInputFileConn()
      :CSVInputFileConn() {}
    ~CSVCaffeInputFileConn() {}

    void init(const APIData &ad)
    {
      CSVInputFileConn::init(ad);
    }
    
    int transform(const APIData &ad)
    {
      CSVInputFileConn::transform(ad);
      
      // transform to datum by filling up float_data
      auto hit = _csvdata.cbegin();
      while(hit!=_csvdata.cend())
	{
	  _dv.push_back(to_datum((*hit).second));
	  ++hit;
	}
      
      return 0;
    }

    caffe::Datum to_datum(const std::vector<double> &vf)
    {
      caffe::Datum datum;
      datum.set_channels(vf.size());
      datum.set_height(1);
      datum.set_width(1);
      int datum_channels = datum.channels();
      for (int i=0;i<datum_channels;i++)
	datum.add_float_data(static_cast<float>(vf.at(i)));
      return datum;
    }

    std::vector<caffe::Datum> _dv;
  };

}

#endif
