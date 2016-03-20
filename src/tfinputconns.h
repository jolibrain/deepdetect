/**
 * DeepDetect
 * Copyright (c) 2016 Emmanuel Benazera
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

#ifndef TFINPUTCONNS_H
#define TFINPUTCONNS_H

#include "imginputfileconn.h"

namespace dd
{
  class TFInputInterface
  {
  public:
    TFInputInterface() {}
    TFInputInterface(const TFInputInterface &tii) {}
    ~TFInputInterface() {}

  public:
    //TODO: parameters common to all TF input connectors
  };

  class ImgTFInputFileConn : public ImgInputFileConn, public TFInputInterface
  {
  public:
    ImgTFInputFileConn()
      :ImgInputFileConn() {}
    ImgTFInputFileConn(const ImgTFInputFileConn &i)
      :ImgInputFileConn(i),TFInputInterface(i) {}
    ~ImgTFInputFileConn() {}

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
      catch(InputConnectorBadParamException &e)
	{
	  throw;
	}
      //TODO: convert acquired images to TF input format
    }

  public:
    //TODO: image connector parameters

  };
  
}

#endif
