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

#ifndef XGBINPUTCONNS_H
#define XGBINPUTCONNS_H

#include "csvinputfileconn.h"
#include "txtinputfileconn.h"

#define DMLC_THROW_EXCEPTION noexcept(false)
#define DMLC_NO_EXCEPTION  noexcept(true)

#include <dmlc/base.h>
#include <dmlc/io.h>
#include <dmlc/build_config.h>
#include <data/parser.h> // dmlc
#include <xgboost/data.h>

namespace dd
{
  class XGBInputInterface
  {
  public:
    XGBInputInterface() {}
    XGBInputInterface(const XGBInputInterface &xii)
      :_missing(xii._missing),_ids(xii._ids) {}
    ~XGBInputInterface()
      {
      }

  public:
    std::shared_ptr<xgboost::DMatrix>_m;
    std::shared_ptr<xgboost::DMatrix> _mtest;

    // for API info only
    int width() const
    {
      return -1;
    }

    // for API info only
    int height() const
    {
      return -1;
    }
    
    // parameters
    float _missing;// = std::NAN; /**< represents missing values. */
    std::vector<std::string> _ids; /**< input ids. */
  };
  
  class CSVXGBInputFileConn : public CSVInputFileConn, public XGBInputInterface
  {
  public:
    CSVXGBInputFileConn()
      :CSVInputFileConn() {}
    CSVXGBInputFileConn(const CSVXGBInputFileConn &i)
      :CSVInputFileConn(i),XGBInputInterface(i),_direct_csv(i._direct_csv) {}
    ~CSVXGBInputFileConn() {}
    
    void init(const APIData &ad)
    {
      if (ad.has("direct_csv") && ad.get("direct_csv").get<bool>())
	_direct_csv = true;
      CSVInputFileConn::init(ad);
    }

    void transform(const APIData &ad);

    xgboost::DMatrix* create_from_mat(const std::vector<CSVline> &csvl);

  public:
    bool _direct_csv = false; /**< whether to use the xgboost built-in CSV reader. */
  };

  class SVMXGBInputFileConn : public InputConnectorStrategy, public XGBInputInterface
  {
  public:
    SVMXGBInputFileConn()
      :InputConnectorStrategy() {}
    SVMXGBInputFileConn(const SVMXGBInputFileConn &i)
      :InputConnectorStrategy(i),XGBInputInterface(i) {}
    ~SVMXGBInputFileConn() {}
    
    void fillup_parameters(const APIData &ad_input)
    {
      if (ad_input.has("shuffle"))
	_shuffle = ad_input.get("shuffle").get<bool>();
      if (ad_input.has("seed"))
	_seed = ad_input.get("seed").get<int>();
      if (ad_input.has("test_split"))
	_test_split = ad_input.get("test_split").get<double>();
    }
    
    void init(const APIData &ad)
    {
      fillup_parameters(ad);
    }
    
    void transform(const APIData &ad);
    
  public:
    bool _shuffle = false;
    int _seed = -1;
    double _test_split = -1;
  };

  class TxtXGBInputFileConn : public TxtInputFileConn, public XGBInputInterface
  {
  public:
    TxtXGBInputFileConn()
      :TxtInputFileConn() {}
    TxtXGBInputFileConn(const TxtXGBInputFileConn &i)
      :TxtInputFileConn(i),XGBInputInterface(i) {}
    ~TxtXGBInputFileConn() {}

    void init(const APIData &ad)
    {
      TxtInputFileConn::init(ad);
    }

    void transform(const APIData &ad);

    xgboost::DMatrix* create_from_mat(const std::vector<TxtEntry<double>*> &txt);
    
  };
  
}

#endif
