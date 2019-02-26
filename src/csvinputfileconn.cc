/**
 * DeepDetect
 * Copyright (c) 2015 Emmanuel Benazera
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

#include "csvinputfileconn.h"

namespace dd
{

  /*- DDCsv -*/
  int DDCsv::read_file(const std::string &fname)
  {
    if (_cifc)
      {
        _cifc->read_csv(fname);
	return 0;
      }
    else return -1;
  }

  int DDCsv::read_db(const std::string &fname)
  {
    _cifc->_db_fname = fname;
    return 0;
  }
  
  int DDCsv::read_mem(const std::string &content)
  {
    if (!_cifc)
      return -1;
    std::stringstream sh(content);
    std::string line;
    int l = 0;
    while(std::getline(sh,line))
      {
	if (_cifc->_columns.empty() && _cifc->_train && l == 0)
	  {
	    _cifc->read_header(line);
	    ++l;
	    continue;
	  }
	
	std::vector<double> vals;
	std::string cid;
	int nlines = 0;
	_cifc->read_csv_line(line,_cifc->_delim,vals,cid,nlines);
	if (_cifc->_scale)
	  {
	    if (!_cifc->_train) // in prediction mode, on-the-fly scaling
	      {
		_cifc->scale_vals(vals);
	      }
	    else // in training mode, collect bounds, then scale in another pass over the data
	      {
		if (_cifc->_min_vals.empty() && _cifc->_max_vals.empty())
		  _cifc->_min_vals = _cifc->_max_vals = vals;
		for (size_t j=0;j<vals.size();j++)
		  {
		    _cifc->_min_vals.at(j) = std::min(vals.at(j),_cifc->_min_vals.at(j));
		    _cifc->_max_vals.at(j) = std::max(vals.at(j),_cifc->_max_vals.at(j));
		  }
	      }
	  }
	if (!cid.empty())
	  _cifc->add_train_csvline(cid,vals);
	else _cifc->add_train_csvline(std::to_string(_cifc->_csvdata.size()+1),vals);
	++l;
      }
    _cifc->update_columns();
    return 0;
  }

  /*- CSVInputFileConn -*/
  void CSVInputFileConn::update_category(const std::string &c,
					 const std::string &val)
  {
    std::unordered_map<std::string,CCategorical>::iterator hit;
    if ((hit=_categoricals.find(c))!=_categoricals.end())
      (*hit).second.add_cat(val);
  }

  void CSVInputFileConn::update_columns()
  {
    std::unordered_map<std::string,CCategorical>::iterator chit;
    auto lit = _columns.begin();
    std::list<std::string> ncolumns = _columns;
    auto nlit = ncolumns.begin();
    while(lit!=_columns.end())
      {
	if (is_category((*lit)))
	  {
	    chit = _categoricals.find((*lit));
	    auto hit = (*chit).second._vals.begin();
	    while(hit!=(*chit).second._vals.end())
	      {
		std::string ncolname = (*lit) + "_" + (*hit).first;
		auto nlit2 = nlit;
		nlit = ncolumns.insert(nlit,ncolname);
		if (hit == (*chit).second._vals.begin())
		  ncolumns.erase(nlit2);
		++hit;
	      }
	  }
	++lit;
	++nlit;
      }
    _columns = ncolumns;

    // update label_pos and id_pos
    int i = 0;
    lit = _columns.begin();
    while(lit!=_columns.end())
      {
	if ((*lit) == _id)
	  _id_pos = i;
  else
         for (unsigned int j=0; j< _label.size(); ++j)
           if ((*lit) == _label[j])
             _label_pos[j] = i;
	++i;
	++lit;
      }

    //debug
    /*std::cerr << "number of new columns=" << _columns.size() << std::endl;
    std::cerr << "new CSV columns:\n";
    std::copy(_columns.begin(),_columns.end(),
	      std::ostream_iterator<std::string>(std::cout," "));
	      std::cerr << std::endl;*/
    //debug
  }
  
  void CSVInputFileConn::read_csv_line(const std::string &hline,
				       const std::string &delim,
				       std::vector<double> &vals,
				       std::string &column_id,
				       int &nlines)
    {
      std::string col;
      std::unordered_set<int>::const_iterator hit;
      std::stringstream sh(hline);
      int c = -1;
      auto lit = _columns.begin();
      while(std::getline(sh,col,delim[0]))
	{
	  ++c;
	  std::string col_name;

	  // detect strings by looking for characters and for quotes
	  // convert to float unless it is string (ignore strings, aka categorical fields, for now)
	  if (!_columns.empty()) // in prediction mode, columns from header are not mandatory
	    {
	      if ((hit=_ignored_columns_pos.find(c))!=_ignored_columns_pos.end())
		{
		  continue;
		}
	      col_name = (*lit);
	      if (_id_pos == c)
		{
		  column_id = col;
		}
	    }
	  try
	    {
	      double val = 0.0;
	      if (!col.empty())
		{
		  // one-hot vector encoding as required
		  if (!_columns.empty() && is_category(col_name))
		    {
		      // - look up category
		      std::unordered_map<std::string,CCategorical>::const_iterator chit
			= _categoricals.find(col_name);
		      int cnum = (*chit).second.get_cat_num(col);
		      if (cnum < 0)
			{
			  throw InputConnectorBadParamException("unknown category " + col + " for variable " + col_name);
			}
		      
		      // - create one-hot vector
		      int csize = (*chit).second._vals.size();
		      std::vector<double> ohv = one_hot_vector(cnum,csize);
		      vals.insert(vals.end(),ohv.begin(),ohv.end());
		    }
		  else
		    {
		      val = std::stod(col);
		      vals.push_back(val);
		    }
		}
	    }
	  catch (std::invalid_argument &e)
	    {
	      // not a number, skip for now
	      if (column_id == col) // if id is string, replace with number / TODO: better scheme
		vals.push_back(c);
	      else
		{
		  _logger->error("line {}: skipping column {} / not a number",nlines,col_name);
		  _logger->error(hline);
		  throw InputConnectorBadParamException("column " + col_name + " is not a number, use categoricals or ignore parameters instead");
		}
	    }
	  ++lit;
	}
      ++nlines;
    }

  void CSVInputFileConn::read_header(std::string &hline)
  {
    hline.erase(std::remove(hline.begin(),hline.end(),'\r'),hline.end()); // remove ^M if any
    std::stringstream sg(hline);
    std::string col;
    
    // read header
    std::unordered_set<std::string>::const_iterator hit;
    std::unordered_map<std::string,int>::const_iterator hit2;
    int i = 0;
    bool has_id = false;
    while(std::getline(sg,col,_delim[0]))
      {
	if ((hit=_ignored_columns.find(col))!=_ignored_columns.end())
	  {
	    _ignored_columns_pos.insert(i);
	    ++i;
	    continue;
	  }
	else _columns.push_back(col);
	
	if ((hit2=_label_set.find(col))!=_label_set.end())
	  _label_pos.at((*hit2).second) = i;
	if (!has_id && !_id.empty() && col == _id)
	  {
	    _id_pos = i;
	    has_id = true;
	  }
	++i;
      }
    _detect_cols = i;
    for (size_t j=0;j<_label_pos.size();j++)
      if (_label_pos.at(j) < 0 && _train)
	throw InputConnectorBadParamException("cannot find label column " + _label[j]);
    if (!_id.empty() && !has_id)
      throw InputConnectorBadParamException("cannot find id column " + _id);
  }


  void CSVInputFileConn::find_min_max(std::ifstream &csv_file)
  {
    int nlines = 0;
    std::string hline;
    while(std::getline(csv_file,hline))
      {
        hline.erase(std::remove(hline.begin(),hline.end(),'\r'),hline.end());
        std::vector<double> vals;
        std::string cid;
        read_csv_line(hline,_delim,vals,cid,nlines);
        if (nlines == 1)
          _min_vals = _max_vals = vals;
        else
          {
            for (size_t j=0;j<vals.size();j++)
              {
                _min_vals.at(j) = std::min(vals.at(j),_min_vals.at(j));
                _max_vals.at(j) = std::max(vals.at(j),_max_vals.at(j));
              }
          }
      }
    csv_file.clear();
    csv_file.seekg(0,std::ios::beg);
    std::getline(csv_file,hline); // skip header line
  }
  
  void CSVInputFileConn::read_csv(const std::string &fname, const bool forbid_shuffle)
  {
      std::ifstream csv_file(fname,std::ios::binary);
      _logger->info("fname={} / open={}",fname,csv_file.is_open());
      if (!csv_file.is_open())
	throw InputConnectorBadParamException("cannot open file " + fname);
      std::string hline;
      std::getline(csv_file,hline);
      read_header(hline);
      
      //debug
      /*std::cerr << "found " << _detect_cols << " columns\n";
      std::cerr << "label size=" << _label.size() << " / label_pos size=" << _label_pos.size() << std::endl;
      std::cout << "CSV columns:\n";
      std::copy(_columns.begin(),_columns.end(),
		std::ostream_iterator<std::string>(std::cout," "));
		std::cout << std::endl;*/
      //debug

      // categorical variables
      if (_train && !_categoricals.empty())
	{
	  int l = 0;
	  while(std::getline(csv_file,hline))
	    {
	      hline.erase(std::remove(hline.begin(),hline.end(),'\r'),hline.end());
	      std::vector<double> vals;
	      std::string cid;
	      std::string col;
	      auto hit = _columns.begin();
	      std::unordered_set<int>::const_iterator igit;
	      std::stringstream sh(hline);
	      int cu = 0;
	      while(std::getline(sh,col,_delim[0]))
		{
		  if (cu >= _detect_cols)
		    {
		      _logger->error("line {} has more columns than headers",l);
		      _logger->error(hline);
		      throw InputConnectorBadParamException("line has more columns than headers");
		    }
		  if ((igit=_ignored_columns_pos.find(cu))!=_ignored_columns_pos.end())
		    {
		      ++cu;
		      continue;
		    }
		  update_category((*hit),col);
		  ++hit;
		  ++cu;
		}
	      ++l;
	    }
	  csv_file.clear();
	  csv_file.seekg(0,std::ios::beg);
	  std::getline(csv_file,hline); // skip header line
	}

      // scaling to [0,1]
      int nlines = 0;
      if (_scale && (_min_vals.empty() || _max_vals.empty()))
	{
         find_min_max(csv_file);
	}
      
      // read data
      while(std::getline(csv_file,hline))
	{
	  hline.erase(std::remove(hline.begin(),hline.end(),'\r'),hline.end());
	  std::vector<double> vals;
	  std::string cid;
	  read_csv_line(hline,_delim,vals,cid,nlines);
	  if (_scale)
	    {
	      scale_vals(vals);
	    }
	  if (!_id.empty())
	    {
	      add_train_csvline(cid,vals);
	    }
	  else add_train_csvline(std::to_string(nlines),vals); 
	  
	  //debug
	  /*std::cout << "csv data line #" << nlines << "= " << vals.size() << std::endl;
	    std::copy(vals.begin(),vals.end(),std::ostream_iterator<double>(std::cout," "));
	    std::cout << std::endl;*/
	  //debug
	}
      _logger->info("read {} lines from {}",nlines,fname);
      csv_file.close();
      
      // test file, if any.
      if (!_csv_test_fname.empty())
	{
	  nlines = 0;
	  std::ifstream csv_test_file(_csv_test_fname,std::ios::binary);
	  if (!csv_test_file.is_open())
	    throw InputConnectorBadParamException("cannot open test file " + fname);
	  std::getline(csv_test_file,hline); // skip header line
	  while(std::getline(csv_test_file,hline))
	    {
	      hline.erase(std::remove(hline.begin(),hline.end(),'\r'),hline.end());
	      std::vector<double> vals;
	      std::string cid;
	      read_csv_line(hline,_delim,vals,cid,nlines);
	      if (_scale)
		{
		  scale_vals(vals);
		}
	      if (!_id.empty())
		add_test_csvline(cid,vals);
	      //_csvdata_test.emplace_back(cid,vals);
	      else add_test_csvline(std::to_string(nlines),vals);
	      //_csvdata_test.emplace_back(std::to_string(nlines),vals);
	      
	      //debug
	      /*std::cout << "csv test data line=";
		std::copy(vals.begin(),vals.end(),std::ostream_iterator<double>(std::cout," "));
		std::cout << std::endl;*/
	      //debug
	    }
	  _logger->info("read {} lines from {}", nlines, _csv_test_fname);
	  csv_test_file.close();
	}

      // shuffle before possible test data selection.
      if (!forbid_shuffle)
        shuffle_data(_csvdata);
      
      if (_csv_test_fname.empty() && _test_split > 0)
	{
	  split_data(_csvdata, _csvdata_test);
	  _logger->info("data split test size={} / remaining data size={}",_csvdata_test.size(),_csvdata.size());
	}
      if (!_ignored_columns.empty() || !_categoricals.empty())
	update_columns();
  }
  
}
