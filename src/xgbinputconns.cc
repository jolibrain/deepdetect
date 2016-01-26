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

#include "xgbinputconns.h"
#include <dmlc/data.h>
#include <dmlc/registry.h>
#include "data/simple_csr_source.h"
#include "common/math.h"

namespace dd
{

  xgboost::DMatrix* CSVXGBInputFileConn::create_from_mat(const std::vector<CSVline> &csvl)
  {
    if (csvl.empty())
      return nullptr;
    std::unique_ptr<xgboost::data::SimpleCSRSource> source(new xgboost::data::SimpleCSRSource());
    xgboost::data::SimpleCSRSource& mat = *source;
    bool nan_missing = xgboost::common::CheckNAN(_missing);
    mat.info.num_row = csvl.size();
    mat.info.num_col = feature_size();
    auto hit = csvl.begin();
    while(hit!=csvl.end())
      {
	long nelem = 0;
	auto lit = _columns.begin();
	for (int i=0;i<(int)(*hit)._v.size();i++)
	  {
	    double v = (*hit)._v.at(i);
	    if (xgboost::common::CheckNAN(v) && !nan_missing)
	      throw InputConnectorBadParamException("NaN value in input data matrix, and missing != NaN");
	    if (!_label_pos.empty() && i == _label_pos[0]) //TODO: multilabel ?
	      {
		mat.info.labels.push_back(v+_label_offset[0]);
	      }
	    else if (i == _id_pos)
	      {
		++lit;
		continue;
	      }
	    else if (std::find(_label_pos.begin(),_label_pos.end(),i)==_label_pos.end())
	      {
		if (nan_missing || v != _missing)
		  {
		    mat.row_data_.push_back(xgboost::RowBatch::Entry(i,v));
		    ++nelem;
		  }
	      }
	  }
	mat.row_ptr_.push_back(mat.row_ptr_.back()+nelem);
	_ids.push_back((*hit)._str);
	++hit;
      }    
    mat.info.num_nonzero = mat.row_data_.size();
    xgboost::DMatrix *out = xgboost::DMatrix::Create(std::move(source));
    return out;
  }
  
  void CSVXGBInputFileConn::transform(const APIData &ad)
  {
    try
      {
	CSVInputFileConn::transform(ad);
      }
    catch (std::exception &e)
      {
	throw;
      }    

    if (!_direct_csv)
      {
	if (_m)
	  delete _m;
	_m = create_from_mat(_csvdata);
	_csvdata.clear();
	if (_mtest)
	  delete _mtest;
	_mtest = create_from_mat(_csvdata_test);
	_csvdata_test.clear();
      }
    else
      {
	//TODO
      }
  }
  
}
