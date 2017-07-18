/**
 * DeepDetect
 * Copyright (c) 2017 Emmanuel Benazera
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

#ifndef TSNEINPUTCONNS_H
#define TSNEINPUTCONNS_H

#include <Eigen/Dense>
#include "csvinputfileconn.h"
#include "txtinputfileconn.h"

namespace dd
{
  typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> dMatR;
  
  class TSNEInputInterface
  {
  public:
    TSNEInputInterface() {}
    TSNEInputInterface(const TSNEInputInterface &tii)
      :_X(tii._X),_D(tii._D),_N(tii._N) // XXX: avoid copying data ?
      {
      }
    ~TSNEInputInterface() {}
  public:
    dMatR _X; /**< data holder */
    int _D = -1; /**< problem dimensions */
    int _N = -1; /**< number of samples */

    //TODO: parameters, ids
  };

  class CSVTSNEInputFileConn : public CSVInputFileConn, public TSNEInputInterface
  {
  public:
    CSVTSNEInputFileConn()
      :CSVInputFileConn() {}
    CSVTSNEInputFileConn(const CSVTSNEInputFileConn &i)
      :CSVInputFileConn(i),TSNEInputInterface(i) {}
    ~CSVTSNEInputFileConn() {}

    void init(const APIData &ad)
    {
      CSVInputFileConn::init(ad);
    }

    void transform(const APIData &ad);
  };

  class TxtTSNEInputFileConn : public TxtInputFileConn, public TSNEInputInterface
  {
  public:
    TxtTSNEInputFileConn()
      :TxtInputFileConn() {}
    TxtTSNEInputFileConn(const TxtTSNEInputFileConn &i)
      :TxtInputFileConn(i),TSNEInputInterface(i) {}
    ~TxtTSNEInputFileConn() {}

    void init(const APIData &ad)
    {
      TxtInputFileConn::init(ad);
    }

    void transform(const APIData &ad);
  };
  
}

#endif
