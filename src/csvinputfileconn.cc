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
#include <random>
#include <algorithm>
#include <glog/logging.h>

namespace dd
{

  /*- DDCsv -*/
  int DDCsv::read_file(const std::string &fname)
  {
    if (_cifc)
      {
	_cifc->read_csv(_adconf,fname);
	return 0;
      }
    else return -1;
  }
  
  int DDCsv::read_mem(const std::string &content)
  {
    if (!_cifc)
      return -1;
    std::vector<double> vals;
    std::string cid;
    int nlines = 0;
    _cifc->read_csv_line(content,_cifc->_delim,vals,cid,nlines);
    if (_cifc->_scale)
      _cifc->scale_vals(vals);
    if (!cid.empty())
      _cifc->_csvdata.emplace_back(cid,vals);
    else _cifc->_csvdata.emplace_back(std::to_string(_cifc->_csvdata.size()+1),vals);
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
	else if ((*lit) == _label)
	  _label_pos = i;
	++i;
	++lit;
      }

    //debug
    /*std::cout << "number of new columns=" << _columns.size() << std::endl;
    std::cout << "new CSV columns:\n";
    std::copy(_columns.begin(),_columns.end(),
	      std::ostream_iterator<std::string>(std::cout," "));
	      std::cout << std::endl;*/
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
	  std::string col_name = (*lit);
	  
	  // detect strings by looking for characters and for quotes
	  // convert to float unless it is string (ignore strings, aka categorical fields, for now)
	  if (!_columns.empty()) // in prediction mode, columns from header are not mandatory
	    {
	      if ((hit=_ignored_columns_pos.find(c))!=_ignored_columns_pos.end())
		{
		  continue;
		}
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
	
	if (col == _label)
	  _label_pos = i;
	if (!has_id && !_id.empty() && col == _id)
	  {
	    _id_pos = i;
	    has_id = true;
	  }
	++i;
      }
    if (_label_pos < 0 && _train)
      throw InputConnectorBadParamException("cannot find label column " + _label);
    if (!_id.empty() && !has_id)
      throw InputConnectorBadParamException("cannot find id column " + _id);
  }
  
  void CSVInputFileConn::read_csv(const APIData &ad,
				  const std::string &fname)
  {
      std::ifstream csv_file(fname,std::ios::binary);
      LOG(INFO) << "fname=" << fname << " / open=" << csv_file.is_open() << std::endl;
      if (!csv_file.is_open())
	throw InputConnectorBadParamException("cannot open file " + fname);
      std::string hline;
      std::getline(csv_file,hline);
      read_header(hline);
      
      //debug
      std::cout << "label=" << _label << " / pos=" << _label_pos << std::endl;
      std::cout << "CSV columns:\n";
      std::copy(_columns.begin(),_columns.end(),
		std::ostream_iterator<std::string>(std::cout," "));
      std::cout << std::endl;
      //debug

      // categorical variables
      if (!_categoricals.empty())
	{
	  while(std::getline(csv_file,hline))
	    {
	      hline.erase(std::remove(hline.begin(),hline.end(),'\r'),hline.end());
	      std::vector<double> vals;
	      std::string cid;
	      std::string col;
	      auto hit = _columns.begin();
	      std::stringstream sh(hline);
	      while(std::getline(sh,col,_delim[0]))
		{
		  update_category((*hit),col);
		  ++hit;
		}
	    }
	  csv_file.clear();
	  csv_file.seekg(0,std::ios::beg);
	  std::getline(csv_file,hline); // skip header line
	}

      // scaling to [0,1]
      int nlines = 0;
      if (_scale && _min_vals.empty() && _max_vals.empty())
	{
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
	  
	  //debug
	  /*std::cout << "min/max scales:\n";
	  std::copy(_min_vals.begin(),_min_vals.end(),std::ostream_iterator<double>(std::cout," "));
	  std::cout << std::endl;
	  std::copy(_max_vals.begin(),_max_vals.end(),std::ostream_iterator<double>(std::cout," "));
	  std::cout << std::endl;*/
	  //debug
	  
	  csv_file.clear();
	  csv_file.seekg(0,std::ios::beg);
	  std::getline(csv_file,hline); // skip header line
	  nlines = 0;
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
	  //_csvdata.emplace_back(cid,std::move(vals));
	  else add_train_csvline(std::to_string(nlines),vals); 
	    //_csvdata.emplace_back(std::to_string(nlines),std::move(vals)); 
	  
	  //debug
	  /*std::cout << "csv data line #" << nlines << "=";
	  std::copy(vals.begin(),vals.end(),std::ostream_iterator<double>(std::cout," "));
	  std::cout << std::endl;*/
	  //debug
	}
      LOG(INFO) << "read " << nlines << " lines from " << fname << std::endl;
      csv_file.close();
      
      // test file, if any.
      std::cerr << "csv test fname=" << _csv_test_fname << std::endl;
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
	  LOG(INFO) << "read " << nlines << " lines from " << _csv_test_fname << std::endl;
	  csv_test_file.close();
	}

      // shuffle before possible test data selection.
      if (ad.has("shuffle") && ad.get("shuffle").get<bool>())
	{
	  std::random_device rd;
	  std::mt19937 g(rd());
	  std::shuffle(_csvdata.begin(),_csvdata.end(),g);
	}
      
      if (_csv_test_fname.empty() && _test_split > 0)
	{
	  if (_test_split > 0.0)
	    {
	      int split_size = std::floor(_csvdata.size() * (1.0-_test_split));
	      auto chit = _csvdata.begin();
	      auto dchit = chit;
	      int cpos = 0;
	      while(chit!=_csvdata.end())
		{
		  if (cpos == split_size)
		    {
		      if (dchit == _csvdata.begin())
			dchit = chit;
		      _csvdata_test.push_back((*chit));
		    }
		  else ++cpos;
		  ++chit;
		}
	      _csvdata.erase(dchit,_csvdata.end());
	      LOG(INFO) << "data split test size=" << _csvdata_test.size() << " / remaining data size=" << _csvdata.size() << std::endl;
	    }
	}
      if (!_ignored_columns.empty() || !_categoricals.empty())
	update_columns();
  }
  
}
