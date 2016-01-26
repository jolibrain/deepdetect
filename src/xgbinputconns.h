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
#include <data/parser.h> // dmlc
#include <xgboost/data.h>

namespace dd
{
  class CSVXGBInputFileConn : public CSVInputFileConn
  {
  public:
    CSVXGBInputFileConn()
      :CSVInputFileConn() {}
    CSVXGBInputFileConn(const CSVXGBInputFileConn &i)
      :CSVInputFileConn(i),_missing(i._missing),_direct_csv(i._direct_csv),_ids(i._ids) {}
    ~CSVXGBInputFileConn() {}

    void init(const APIData &ad)
    {
      CSVInputFileConn::init(ad);
    }

    void transform(const APIData &ad);

    xgboost::DMatrix* create_from_mat(const std::vector<CSVline> &csvl);

  public:
    xgboost::DMatrix *_m = nullptr;
    xgboost::DMatrix *_mtest = nullptr; 

    // parameters
    float _missing;// = std::NAN; /**< represents missing values. */
    bool _direct_csv = false; /**< whether to use the xgboost built-in CSV reader. */

    std::vector<std::string> _ids; /**< input ids. */
  };
  
}

#endif
