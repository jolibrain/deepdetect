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

#ifndef SVMINPUTFILECONN_H
#define SVMINPUTFILECONN_H

#include "inputconnectorstrategy.h"
#include <random>
#include <algorithm>

namespace dd
{
  class SVMInputFileConn;

  class DDSvm
  {
  public:
    DDSvm() {}
    ~DDSvm() {}

    int read_file(const std::string &fname);
    int read_mem(const std::string &content);
    int read_dir(const std::string &dir)
    {
      throw InputConnectorBadParamException("uri " + dir + " is a directory, requires a file in libSVM format");
    }
    
    SVMInputFileConn *_cifc = nullptr;
    APIData _adconf;
  };
  
  class SVMline
  {
  public:
    SVMline(const int &label,
	    const std::unordered_map<int,double> &v)
      :_label(label),_v(v) {}
    ~SVMline() {}
    std::unordered_map<int,double> _v; /**< svm line data */
    int _label; /**< svm line label. */
  };

  class SVMInputFileConn : public InputConnectorStrategy
  {
  public:
    SVMInputFileConn()
      :InputConnectorStrategy() {}
    ~SVMInputFileConn() {}

    void init(const APIData &ad)
    {
      fillup_parameters(ad);
    }

    void fillup_parameters(const APIData &ad_input)
    {
      if (ad_input.has("test_split"))
	_test_split = ad_input.get("test_split").get<double>();
    }

    void shuffle_data(const APIData &ad)
    {
      if (ad.has("shuffle") && ad.get("shuffle").get<bool>())
	{
	  std::mt19937 g;
	  if (ad.has("seed") && ad.get("seed").get<int>() >= 0)
	    {
	      g = std::mt19937(ad.get("seed").get<int>());
	    }
	  else
	    {
	      std::random_device rd;
	      g = std::mt19937(rd());
	    }
	  std::shuffle(_svmdata.begin(),_svmdata.end(),g);
	}
    }

    void split_data()
    {
      if (_test_split > 0.0)
	{
	  int split_size = std::floor(_svmdata.size() * (1.0-_test_split));
	  auto chit = _svmdata.begin();
	  auto dchit = chit;
	  int cpos = 0;
	  while(chit!=_svmdata.end())
	    {
	      if (cpos == split_size)
		{
		  if (dchit == _svmdata.begin())
		    dchit = chit;
		  _svmdata_test.push_back((*chit));
		}
	      else ++cpos;
	      ++chit;
	    }
	  _svmdata.erase(dchit,_svmdata.end());
	}
    }

    void transform(const APIData &ad)
    {
      get_data(ad);
      APIData ad_input = ad.getobj("parameters").getobj("input");
      fillup_parameters(ad_input);

      // training from either file or memory.
      if (_train)
	{
	  if (fileops::file_exists(_uris.at(0))) // training from file
	    {
	      _svm_fname = _uris.at(0);
	      if (_uris.size() > 1)
		_svm_test_fname = _uris.at(1);
	    }
	  if (!_svm_fname.empty()) // when training from file
	    {
	      DataEl<DDSvm> ddsvm;
	      ddsvm._ctype._cifc = this;
	      ddsvm._ctype._adconf = ad_input;
	      ddsvm.read_element(_svm_fname);
	    }
	  else // training from posted data (in-memory)
	    {
	      for (size_t i=1;i<_uris.size();i++)
		{
		  DataEl<DDSvm> ddsvm;
		  ddsvm._ctype._cifc = this;
		  ddsvm._ctype._adconf = ad_input;
		  ddsvm.read_element(_uris.at(i));
		}
	      /*if (_scale)
		{
		  for (size_t j=0;j<_svmdata.size();j++)
		    {
		      scale_vals(_svmdata.at(j)._v);
		    }
		    }*/
	      shuffle_data(ad_input);
	      if (_test_split > 0.0)
		split_data();
	    }
	  serialize_vocab();
	}
      else // prediction mode
	{
	  if (_fids.empty())
	    deserialize_vocab();
	  for (size_t i=0;i<_uris.size();i++)
	    {
	      DataEl<DDSvm> ddsvm;
	      ddsvm._ctype._cifc = this;
	      ddsvm._ctype._adconf = ad_input;
	      ddsvm.read_element(_uris.at(i));
	    }
	}
      if (_svmdata.empty())
	throw InputConnectorBadParamException("no data could be found");
    }

    virtual void add_train_svmline(const int &label,
				   const std::unordered_map<int,double> &vals,
				   const int &count)
    {
      _svmdata.emplace_back(label,std::move(vals));
    }
    
    virtual void add_test_svmline(const int &label,
				  const std::unordered_map<int,double> &vals,
				  const int &count)
    {
      _svmdata_test.emplace_back(label,std::move(vals));
    }

    void read_svm(const APIData &ad,
		  const std::string &fname);
    void read_svm_line(const std::string &content,
		       std::unordered_map<int,double> &vals,
		       int &label);

    int batch_size() const
    {
      return _svmdata.size();
    }

    int test_batch_size() const
    {
      return _svmdata_test.size();
    }

    int feature_size() const
    {
      // total number of indices
      return _max_id;
    }

    // serialization of vocabulary
    void serialize_vocab();
    void deserialize_vocab(const bool &required=true);

    // options
    std::string _svm_fname;
    std::string _svm_test_fname;
    double _test_split = -1;

    // data
    std::vector<SVMline> _svmdata;
    std::vector<SVMline> _svmdata_test;
    std::unordered_set<int> _fids; /**< feature ids. */
    int _max_id = -1;
    std::string _vocabfname = "vocab.dat";
  };

}

#endif
