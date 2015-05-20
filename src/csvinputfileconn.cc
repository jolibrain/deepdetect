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
      {
	std::cerr << "scaling in mem vals\n";
	_cifc->scale_vals(vals);
      }
    if (!cid.empty())
      _cifc->_csvdata.emplace_back(cid,vals);
    else _cifc->_csvdata.emplace_back(std::to_string(vals.size()+1),vals);
    return 0;
  }

  /*- CSVInputFileConn -*/
  void CSVInputFileConn::read_csv_line(const std::string &hline,
				       const std::string &delim,
				       std::vector<double> &vals,
				       std::string &column_id,
				       int &nlines)
    {
      std::string col;
      std::unordered_set<std::string>::const_iterator hit;
      std::stringstream sh(hline);
      int c = -1;
      while(std::getline(sh,col,delim[0]))
	{
	  ++c;
	  
	  // detect strings by looking for characters and for quotes
	  // convert to float unless it is string (ignore strings, aka categorical fields, for now)
	  if (!_columns.empty()) // in prediction mode, columns from header are not mandatory
	    {
	      if ((hit=_ignored_columns.find(_columns.at(c)))!=_ignored_columns.end())
		{
		  LOG(INFO) << "ignoring column: " << col << std::endl;
		  continue;
		}
	      if (_columns.at(c) == _id)
		{
		  column_id = col;
		}
	    }
	  try
	    {
	      double val = 0.0;
	      if (!col.empty())
		val = std::stod(col);
	      vals.push_back(val);
	    }
	  catch (std::invalid_argument &e)
	    {
	      // not a number, skip for now
	      //std::cout << "not a number: " << col << std::endl;
	      if (column_id == col) // if id is string, replace with number / TODO: better scheme
		vals.push_back(c);
	    }
	}
      ++nlines;
    }

  void CSVInputFileConn::read_header(std::string &hline)
  {
    hline.erase(std::remove(hline.begin(),hline.end(),'\r'),hline.end()); // remove ^M if any
    std::stringstream sg(hline);
    std::string col;
    
    // read header
    int i = 0;
    //auto hit = _ignored_columns.begin();
    while(std::getline(sg,col,_delim[0]))
      {
	/*if ((hit=_ignored_columns.find(col))!=_ignored_columns.end())
	  continue;*/
	_columns.push_back(col);
	if (col == _label)
	  _label_pos = i;
	++i;
      }
    if (_label_pos < 0 && _train)
      throw InputConnectorBadParamException("cannot find label column " + _label);
  }
  
  void CSVInputFileConn::read_csv(const APIData &ad,
				  const std::string &fname)
  {
      std::ifstream csv_file(fname,std::ios::binary);
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
	  std::cout << "min/max scales:\n";
	  std::copy(_min_vals.begin(),_min_vals.end(),std::ostream_iterator<double>(std::cout," "));
	  std::cout << std::endl;
	  std::copy(_max_vals.begin(),_max_vals.end(),std::ostream_iterator<double>(std::cout," "));
	  std::cout << std::endl;
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
	      /*for (int j=0;j<(int)vals.size();j++)
		{
		  if ((!_columns.empty() && _columns.at(j) != _id) && j != _label_pos && _max_vals.at(j) != _min_vals.at(j))
		    vals.at(j) = (vals.at(j) - _min_vals.at(j)) / (_max_vals.at(j) - _min_vals.at(j));
		    }*/
	      scale_vals(vals);
	    }
	  if (!_id.empty())
	    _csvdata.emplace_back(cid,vals);
	  else _csvdata.emplace_back(std::to_string(nlines),vals); 
	  
	  //debug
	  std::cout << "csv data line #" << nlines << "=";
	  std::copy(vals.begin(),vals.end(),std::ostream_iterator<double>(std::cout," "));
	  std::cout << std::endl;
	  //debug
	}
      std::cerr << "read " << nlines << " lines from " << fname << std::endl;
      std::cerr << "id=" << _id << std::endl;
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
		  /*for (int j=0;j<(int)vals.size();j++)
		    {
		      if (_columns.at(j) != _id && j != _label_pos && _max_vals.at(j) != _min_vals.at(j))
			vals.at(j) = (vals.at(j) - _min_vals.at(j)) / (_max_vals.at(j) - _min_vals.at(j));
			}*/
		  scale_vals(vals);
		}
	      if (!_id.empty())
		_csvdata_test.emplace_back(cid,vals);
	      else _csvdata_test.emplace_back(std::to_string(nlines),vals);
	      
	      //debug
	      std::cout << "csv test data line=";
	      std::copy(vals.begin(),vals.end(),std::ostream_iterator<double>(std::cout," "));
	      std::cout << std::endl;
	      //debug
	    }
	  std::cerr << "read " << nlines << " lines from " << _csv_test_fname << std::endl;
	  csv_test_file.close();
	}

      // shuffle before possible test data selection.
      if (ad.has("shuffle") && ad.get("shuffle").get<bool>())
	{
	  std::random_device rd;
	  std::mt19937 g(rd());
	  std::shuffle(_csvdata.begin(),_csvdata.end(),g);
	}
      
      if (_csv_test_fname.empty() && ad.has("test_split"))
	{
	  double split = ad.get("test_split").get<double>();
	  if (split > 0.0)
	    {
	      int split_size = std::floor(_csvdata.size() * (1.0-split));
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
	      std::cout << "data split test size=" << _csvdata_test.size() << " / remaining data size=" << _csvdata.size() << std::endl;
	    }
	}
    }
  
}
