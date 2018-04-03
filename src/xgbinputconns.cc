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

  // XXX: Adapted from XGBoost c_api.cc, pretty much sub-optimal memory wise,
  // and not very much adapted to large data...
  xgboost::DMatrix* XGDMatrixSliceDMatrix(xgboost::DMatrix* handle,
					  const int* idxset,
					  unsigned long len)
  {
    std::unique_ptr<xgboost::data::SimpleCSRSource> source(new xgboost::data::SimpleCSRSource());
    
    xgboost::data::SimpleCSRSource src;
    src.CopyFrom(handle);
    xgboost::data::SimpleCSRSource& ret = *source;
    
    CHECK_EQ(src.info.group_ptr.size(), 0)
      << "slice does not support group structure";
    
    ret.Clear();
    ret.info.num_row = len;
    ret.info.num_col = src.info.num_col;
    
    dmlc::DataIter<xgboost::RowBatch>* iter = &src;
    iter->BeforeFirst();
    CHECK(iter->Next());
    
    const xgboost::RowBatch& batch = iter->Value();
    for (unsigned long i = 0; i < len; ++i) {
      const int ridx = idxset[i];
      xgboost::RowBatch::Inst inst = batch[ridx];
      CHECK_LT(static_cast<unsigned long>(ridx), batch.size);
      ret.row_data_.resize(ret.row_data_.size() + inst.length);
      std::memcpy(dmlc::BeginPtr(ret.row_data_) + ret.row_ptr_.back(), inst.data,
		  sizeof(xgboost::RowBatch::Entry) * inst.length);
      ret.row_ptr_.push_back(ret.row_ptr_.back() + inst.length);
      ret.info.num_nonzero += inst.length;
      
      if (src.info.labels.size() != 0) {
	ret.info.labels.push_back(src.info.labels[ridx]);
      }
      if (src.info.weights.size() != 0) {
	ret.info.weights.push_back(src.info.weights[ridx]);
      }
      if (src.info.root_index.size() != 0) {
	ret.info.root_index.push_back(src.info.root_index[ridx]);
      }
    }
    xgboost::DMatrix *out = xgboost::DMatrix::Create(std::move(source));
    return out;
  }
  
  xgboost::DMatrix* CSVXGBInputFileConn::create_from_mat(const std::vector<CSVline> &csvl)
  {
    if (csvl.empty())
      return nullptr;
    std::unique_ptr<xgboost::data::SimpleCSRSource> source(new xgboost::data::SimpleCSRSource());
    xgboost::data::SimpleCSRSource& mat = *source;
    bool nan_missing = xgboost::common::CheckNAN(_missing);
    mat.info.num_row = csvl.size();
    mat.info.num_col = feature_size()+1; // XXX: +1 otherwise there's a mismatch in xgboost's simple_dmatrix.cc:151
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
	    std::vector<int>::iterator ipos;
	    if (!_label_pos.empty()
		&& (ipos = std::find(_label_pos.begin(),_label_pos.end(),i))!=_label_pos.end())
	      {
		int pos = std::distance(_label_pos.begin(),ipos);
		mat.info.labels.push_back(v+_label_offset[pos]);
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
	    ++lit;
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

    // feature map (useful for feature importance analysis)
    if (ad.has("model_repo"))
      {
	std::ofstream fmap(ad.get("model_repo").get<std::string>()+"/model.fmap",std::ios::binary);
	int nc = 0;
	for (auto c: this->_columns)
	  {
	    fmap << nc << "\t" << c << "\tq\n";
	    ++nc;
	  }
      }
      
    if (!_direct_csv)
      {
	_m = std::shared_ptr<xgboost::DMatrix>(create_from_mat(_csvdata));
	_csvdata.clear();
	if (_m->info().num_nonzero == 0)
	  throw InputConnectorBadParamException("no data could be found processing XGBoost CSV input");
	_mtest = std::shared_ptr<xgboost::DMatrix>(create_from_mat(_csvdata_test));
	_csvdata_test.clear();
      }
    else
      {
	// Not robust enough from within XGBoost
	/*bool silent = false;
	int dsplit = 2;
	_m = xgboost::DMatrix::Load(_csv_fname,silent,dsplit==2);
	if (!_csv_test_fname.empty())
	_mtest = xgboost::DMatrix::Load(_csv_test_fname,silent,dsplit==2);*/
      }
  }

  void SVMXGBInputFileConn::transform(const APIData &ad)
  {
    //- get data
    InputConnectorStrategy::get_data(ad);
    
    //- load lsvm file(s)
    bool silent = false;
    int dsplit = 2;
    if (_uris.size() == 1)
      {
	//- shuffle & split matrix as required (read parameters -> fillup_parameters or init ?)
	APIData ad_input = ad.getobj("parameters").getobj("input");
	fillup_parameters(ad_input);
	_logger->info("loading {}",_uris.at(0));
	_m = std::shared_ptr<xgboost::DMatrix>(xgboost::DMatrix::Load(_uris.at(0),silent,dsplit));
	size_t rsize = _m->info().num_row;
	_logger->info("successfully read {} rows",rsize);

	std::vector<int> rindex(rsize);
	std::iota(std::begin(rindex),std::end(rindex),0);
	if (_shuffle)
	  {
	    std::mt19937 g;
	    if (_seed != -1)
	      g = std::mt19937(_seed);
	    else
	      {
		std::random_device rd;
		g = std::mt19937(rd());
	      }
	    std::shuffle(rindex.begin(),rindex.end(),g);
	  }
	if (_test_split > 0.0)
	  {
	    // XXX: not optimal memory-wise, due to the XGDMatrixSlice op
	    _logger->info("splitting dataset");
	    int split_size = std::floor(rsize * (1.0-_test_split));
	    std::vector<int> train_rindex(rindex.begin(),rindex.begin()+split_size);
	    std::vector<int> test_rindex(rindex.begin()+split_size,rindex.end());
	    rindex.clear();
	    xgboost::DMatrix *mtrain = XGDMatrixSliceDMatrix(_m.get(),&train_rindex[0],train_rindex.size());
	    _mtest = std::shared_ptr<xgboost::DMatrix>(XGDMatrixSliceDMatrix(_m.get(),&test_rindex[0],test_rindex.size()));
	    _m = std::shared_ptr<xgboost::DMatrix>(mtrain);
	    _logger->info("dataset successfully splitted");
	  }
	else
	  {
	    xgboost::DMatrix *mtrain = XGDMatrixSliceDMatrix(_m.get(),&rindex[0],rindex.size());
	    _m = std::shared_ptr<xgboost::DMatrix>(mtrain);
	    std::vector<int> ids(_m->info().num_row);
	    std::iota(std::begin(ids),std::end(ids),0);
	    for (int i: ids)
	      _ids.push_back(std::to_string(i));
	  }
	
      }
    else if (_uris.size() == 2) // with test file
      {
	_logger->info("reading train and test matrices");
	_m = std::shared_ptr<xgboost::DMatrix>(xgboost::DMatrix::Load(_uris.at(0),silent,dsplit));
	_mtest = std::shared_ptr<xgboost::DMatrix>(xgboost::DMatrix::Load(_uris.at(1),silent,dsplit));
	_logger->info("successfully acquired data");
      }
  }

  void TxtXGBInputFileConn::transform(const APIData &ad)
  {
    try
      {
	TxtInputFileConn::transform(ad);
      }
    catch (std::exception &e)
      {
	throw;
      }

    _m = std::shared_ptr<xgboost::DMatrix>(create_from_mat(_txt));
    destroy_txt_entries(_txt);
    if (!_test_txt.empty())
      _mtest = std::shared_ptr<xgboost::DMatrix>(create_from_mat(_test_txt));
    destroy_txt_entries(_test_txt);
  }
  
  xgboost::DMatrix* TxtXGBInputFileConn::create_from_mat(const std::vector<TxtEntry<double>*> &txt)
  {
    if (txt.empty())
      return nullptr;
    std::unique_ptr<xgboost::data::SimpleCSRSource> source(new xgboost::data::SimpleCSRSource());
    xgboost::data::SimpleCSRSource& mat = *source;
    bool nan_missing = xgboost::common::CheckNAN(_missing);
    mat.info.num_row = txt.size();
    mat.info.num_col = feature_size()+1; // XXX: +1 otherwise there's a mismatch in xgnoost's simple_dmatrix.cc:151
    int nid = 0;
    auto hit = txt.begin();
    while(hit!=txt.end())
      {
	long nelem = 0;
	TxtBowEntry *tbe = static_cast<TxtBowEntry*>((*hit));
	mat.info.labels.push_back(tbe->_target);
	tbe->reset();
	while(tbe->has_elt())
	  {
	    std::string key;
	    double v;
	    tbe->get_next_elt(key,v);
	    if (xgboost::common::CheckNAN(v) && !nan_missing)
	      throw InputConnectorBadParamException("NaN value in input data matrix, and missing != NaN");
	    mat.row_data_.push_back(xgboost::RowBatch::Entry(_vocab[key]._pos,v));
	    ++nelem;
	  }
	mat.row_ptr_.push_back(mat.row_ptr_.back()+nelem);
	_ids.push_back(std::to_string(nid));
	++nid;
	++hit;
      }
    mat.info.num_nonzero = mat.row_data_.size();
    xgboost::DMatrix *out = xgboost::DMatrix::Create(std::move(source));
    return out;
  }
  
}
