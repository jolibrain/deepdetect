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
#include <fstream>
#include <unordered_set>

namespace dd
{
  class CSVInputFileConn;
  
  class DDCsv
  {
  public:
    DDCsv() {}
    ~DDCsv() {}

    int read_file(const std::string &fname);
    int read_mem(const std::string &content);

    CSVInputFileConn *_cifc = nullptr;
    APIData _adconf;
  };
  
  class CSVline
  {
  public:
    CSVline(const std::string &str,
	    const std::vector<double> &v)
      :_str(str),_v(v) {}
    ~CSVline() {}
    std::string _str; /**< csv line id */
    std::vector<double> _v; /**< csv line data */
  };
  
  class CSVInputFileConn : public InputConnectorStrategy
  {
  public:
    CSVInputFileConn()
      :InputConnectorStrategy() {}
    
    ~CSVInputFileConn() {}
  
    void init(const APIData &ad)
    {
      fillup_parameters(ad);
    }

    // unused for now, receptacle for any generic option
    void fillup_parameters(const APIData &ad)
    {
      (void)ad;
    }

    void scale_vals(std::vector<double> &vals)
    {
      for (int j=0;j<(int)vals.size();j++)
	{
	  if ((!_columns.empty() && _columns.at(j) != _id) && (_label_pos >= 0 && j != _label_pos) && _max_vals.at(j) != _min_vals.at(j))
	    vals.at(j) = (vals.at(j) - _min_vals.at(j)) / (_max_vals.at(j) - _min_vals.at(j));
	}
    }
    
    void transform(const APIData &ad)
    {
      get_data(ad);

      APIData ad_input = ad.getobj("parameters").getobj("input");

      if (ad_input.has("id"))
	_id = ad_input.get("id").get<std::string>();
      if (ad_input.has("separator"))
	_delim = ad_input.get("separator").get<std::string>();

      if (ad_input.has("ignore"))
	{
	  std::vector<std::string> vignore = ad_input.get("ignore").get<std::vector<std::string>>();
	  for (std::string s: vignore)
	    _ignored_columns.insert(s);
	}
      
      // read scaling parameters, if any.
      if (ad_input.has("scale") && ad_input.get("scale").get<bool>())
	{
	  _scale = true;
	  if (ad.has("min_vals"))
	    _min_vals = ad.get("min_vals").get<std::vector<double>>();
	  if (ad.has("max_vals"))
	    _max_vals = ad.get("max_vals").get<std::vector<double>>();
	}
      
      if (_train)
	{
	  _csv_fname = _uris.at(0); // training only from file
	  if (_uris.size() > 1)
	    _csv_test_fname = _uris.at(1);
	  if (ad_input.has("label"))
	    _label = ad_input.get("label").get<std::string>();
	  else if (_train) throw InputConnectorBadParamException("missing label column parameter");
	  if (ad_input.has("label_offset"))
	    _label_offset = ad_input.get("label_offset").get<int>();
	  
	  DataEl<DDCsv> ddcsv;
	  ddcsv._ctype._cifc = this;
	  ddcsv._ctype._adconf = ad_input;
	  ddcsv.read_element(_csv_fname);
	}
      else // prediction mode
	{
	  for (size_t i=0;i<_uris.size();i++)
	    {
	      if (i == 0 && ad_input.size() && !_id.empty() && _uris.at(0).find(_delim)!=std::string::npos) // first line might be the header if we have some options to consider
		{
		  read_header(_uris.at(0));
		  continue;
		}
	      DataEl<DDCsv> ddcsv;
	      ddcsv._ctype._cifc = this;
	      ddcsv._ctype._adconf = ad_input;
	      ddcsv.read_element(_uris.at(i));
	    }
	}
      if (_csvdata.empty())
	throw InputConnectorBadParamException("no data could be found");
    }

    void read_header(std::string &hline);
      
    void read_csv_line(const std::string &hline,
		       const std::string &delim,
		       std::vector<double> &vals,
		       std::string &column_id,
		       int &nlines);
    
    void read_csv(const APIData &ad,
		  const std::string &fname);

    int batch_size() const
    {
      return _csvdata.size(); // XXX: what about test data size ?
    }

    int test_batch_size() const
    {
      return _csvdata_test.size();
    }

    int feature_size() const
    {
      if (!_id.empty())
	return _columns.size() - 2; // minus label and id
      else return _columns.size() - 1; // minus label
    }

    void response_params(APIData &out)
    {
      if (_scale)
	{
	  APIData adinput;
	  adinput.add("connector","csv");
	  adinput.add("min_vals",_min_vals);
	  adinput.add("max_vals",_max_vals);
	  std::vector<APIData> vip = { adinput };
	  if (!out.has("parameters"))
	    {
	      APIData adparams;
	      adparams.add("input",vip);
	      std::vector<APIData> vadp = { adparams };
	      out.add("parameters",vadp);
	    }
	  else
	    {
	      std::vector<APIData> vad = out.get("parameters").get<std::vector<APIData>>();
	      vad.at(0).add("input",vip);
	      out.add("parameters",vad);
	    }
	}
    }
    
    std::string _csv_fname;
    std::string _csv_test_fname;
    std::vector<std::string> _columns; //TODO: unordered map to int as pos of the column
    std::string _label;
    std::string _delim = ",";
    int _label_pos = -1;
    int _label_offset = 0; /**< negative offset so that labels range from 0 onward */
    std::unordered_set<std::string> _ignored_columns;
    std::string _id;
    bool _scale = false; /**< whether to scale all data between 0 and 1 */
    std::vector<double> _min_vals; /**< upper bound used for auto-scaling data */
    std::vector<double> _max_vals; /**< lower bound used for auto-scaling data */
    std::vector<CSVline> _csvdata;
    std::vector<CSVline> _csvdata_test;
  };
}

#endif
