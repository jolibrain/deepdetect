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

#ifndef CSVINPUTFILECONN_H
#define CSVINPUTFILECONN_H

#include "inputconnectorstrategy.h"
//#include "ext/csv.h"
#include <fstream>
#include <unordered_set>

namespace dd
{
  class CSVInputFileConn : public InputConnectorStrategy
  {
  public:
    CSVInputFileConn()
      :InputConnectorStrategy() {}
    
    ~CSVInputFileConn() {}
  
    void init(const APIData &ad)
    {
      fillup_parameters(ad,_csv_fname);
    }
    
    void fillup_parameters(const APIData &ad,
			   std::string &fname)
    {
      //TODO: anything ?
    }
    
    int transform(const APIData &ad)
    {
      //TODO: iterate csv file and fill up the csvdata object.
      if (ad.has("filename"))
	_csv_fname = ad.get("filename").get<std::string>();
      else throw InputConnectorBadParamException("no CSV filename");
      if (ad.has("label"))
	_label = ad.get("label").get<std::string>();
      else throw InputConnectorBadParamException("missing label column parameter");
      if (ad.has("ignore"))
	{
	  std::vector<std::string> vignore = ad.get("ignore").get<std::vector<std::string>>();
	  for (std::string s: vignore)
	    _ignored_columns.insert(s);
	}
      if (ad.has("id"))
	_id = ad.get("id").get<std::string>();
      read_csv(ad,_csv_fname);
      
    }

    void read_csv(const APIData &ad,
		  const std::string &fname)
    {
      std::ifstream csv_file(fname);
      if (!csv_file.is_open())
	throw InputConnectorBadParamException("cannot open file " + fname);
      std::string hline;
      std::getline(csv_file,hline);
      std::stringstream sg(hline);
      std::string col;
      std::string delim = ",";
      if (ad.has("separator"))
	delim = ad.get("separator").get<std::string>();
      
      // read header
      int i = 0;
      auto hit = _ignored_columns.begin();
      while(std::getline(sg,col,delim[0]))
	{
	  if ((hit=_ignored_columns.find(col))!=_ignored_columns.end())
	    continue;
	  _columns.push_back(col);
	  if (col == _label)
	    _label_pos = i;
	  ++i;
	}
      if (_label_pos < 0)
	throw InputConnectorBadParamException("cannot find label column " + _label);
      
      //debug
      std::cout << "label=" << _label << " / pos=" << _label_pos << std::endl;
      std::cout << "CSV columns:\n";
      std::copy(_columns.begin(),_columns.end(),
		std::ostream_iterator<std::string>(std::cout," "));
      std::cout << std::endl;
      //debug

      // read data
      i = 0;
      while(std::getline(csv_file,hline))
	{
	  std::vector<double> vals;
	  std::stringstream sh(hline);
	  std::string cid;
	  int c = -1;
	  while(++c && std::getline(sh,col,delim[0]))
	    {
	      // detect strings by looking for characters and for quotes
	      // convert to float unless it is string (ignore strings, aka categorical fields, for now)
	      if ((hit=_ignored_columns.find(_columns.at(c)))!=_ignored_columns.end())
		continue;
	      if (_columns.at(c) == _id)
		{
		  cid = col;
		  continue;
		}
	      try
		{
		  double val = std::stod(col);
		  vals.push_back(val);
		}
	      catch (std::invalid_argument &e)
		{
		  // not a number, skip for now
		}
	    }
	  if (!_id.empty())
	    _csvdata.insert(std::pair<std::string,std::vector<double>>(cid,vals));
	  else _csvdata.insert(std::pair<std::string,std::vector<double>>(std::to_string(i),vals));
	  ++i;
	}
      csv_file.close();
    }

    int size() const
    {
      if (!_id.empty())
	return _columns.size() - 2; // minus label and id
      else return _columns.size() - 1; // minus label
    }

    std::string _csv_fname;
    std::vector<std::string> _columns; //TODO: unordered map to int as pos of the column
    std::string _label;
    int _label_pos = -1;
    std::unordered_set<std::string> _ignored_columns;
    std::string _id;
    std::unordered_map<std::string,std::vector<double>> _csvdata;
  };
}

#endif
