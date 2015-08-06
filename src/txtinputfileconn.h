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

#ifndef TXTINPUTFILECONN_H
#define TXTINPUTFILECONN_H

#include "inputconnectorstrategy.h"
#include <cmath>

namespace dd
{
  class TxtInputFileConn;
  
  class DDTxt
  {
  public:
    DDTxt() {}
    ~DDTxt() {}

    int read_file(const std::string &fname);
    int read_mem(const std::string &content);
    int read_dir(const std::string &dir);

    TxtInputFileConn *_ctfc = nullptr;
  };

  class Word
  {
  public:
    Word() {}
    Word(const int &pos)
      :_pos(pos) {}
    Word(const int &pos,
	 const int &tcount,
	 const int &tdocs)
      :_pos(pos),_total_count(tcount),_total_docs(tdocs) {}
    ~Word() {}
    
    int _pos = -1;
    int _total_count = 1;
    int _total_docs = 1;
  };
  
  class TxtBowEntry
  {
  public:
    TxtBowEntry() {}
    TxtBowEntry(const int &target):_target(target) {}
    ~TxtBowEntry() {}

    void add_word(const int &pos,
		  const double &v,
		  const bool &count)
    {
      std::unordered_map<int,double>::iterator hit;
      if ((hit=_v.find(pos))!=_v.end())
	{
	  if (count)
	    (*hit).second += v;
	}
      else _v.insert(std::pair<int,double>(pos,v));
    }

    bool has_word(const int &pos)
    {
      return _v.count(pos);
    }
    
    std::unordered_map<int,double> _v; /**< words as (<pos,val>). */
    float _target = -1; /**< class target in training mode. */
  };
  
  class TxtInputFileConn : public InputConnectorStrategy
  {
  public:
    TxtInputFileConn()
      :InputConnectorStrategy() 
      { 
	//_vocab = std::unordered_map<std::string,Word>(1e5);
      }

    ~TxtInputFileConn() {}

    void init(const APIData &ad)
    {
      fillup_parameters(ad);
    }

    void fillup_parameters(const APIData &ad_input)
    {
      if (ad_input.has("shuffle"))
	_shuffle = ad_input.get("shuffle").get<bool>();
      if (ad_input.has("test_split"))
	_test_split = ad_input.get("test_split").get<double>();
      if (ad_input.has("count"))
	_count = ad_input.get("count").get<bool>();
      if (ad_input.has("tfidf"))
	_tfidf = ad_input.get("tfidf").get<bool>();
    }

    int feature_size() const
    {
      // total number of words in training set for BOW
      return _vocab.size();
    }

    int batch_size() const
    {
      return _txt.size();
    }

    int test_batch_size() const
    {
      return _test_txt.size();
    }

    void transform(const APIData &ad)
    {
      get_data(ad);

      if (ad.has("parameters")) // hotplug of parameters, overriding the defaults
	{
	  APIData ad_param = ad.getobj("parameters");
	  if (ad_param.has("input"))
	    {
	      fillup_parameters(ad_param.getobj("input"));
	    }
	}

      if (ad.has("model_repo"))
	_model_repo = ad.get("model_repo").get<std::string>();
      
      std::cerr << "train=" << _train << " / vocab size=" << _vocab.size() << std::endl;
      if (!_train && _vocab.empty())
	deserialize_vocab();
      
      for (std::string u: _uris)
	{
	  DataEl<DDTxt> dtxt;
	  dtxt._ctype._ctfc = this;
	  if (dtxt.read_element(u))
	    {
	      throw InputConnectorBadParamException("no data for text in " + u);
	    }
	}
      
      if (_train)
	serialize_vocab();
      
      // split for test set
      if (_train && _test_split > 0)
	{
	  int split_size = std::floor(_txt.size() * (1.0-_test_split));
	  auto chit = _txt.begin();
	  auto dchit = chit;
	  int cpos = 0;
	  while(chit!=_txt.end())
	    {
	      if (cpos == split_size)
		{
		  if (dchit == _txt.begin())
		    dchit = chit;
		  _test_txt.push_back((*chit));
		}
	      else ++cpos;
	      ++chit;
	    }
	  _txt.erase(dchit,_txt.end());
	  std::cout << "data split test size=" << _test_txt.size() << " / remaining data size=" << _txt.size() << std::endl;
	}
      if (_txt.empty())
	throw InputConnectorBadParamException("no text could be found");
    }

    // text tokenization for BOW
    void parse_content(const std::string &content,
		       const int &target=-1);

    // serialization of vocabulary
    void serialize_vocab();
    void deserialize_vocab();

    // options
    std::string _iterator = "document";
    std::string _tokenizer = "bow";
    bool _shuffle = false;
    double _test_split = 0.0;
    bool _count = true; /**< whether to add up word counters */
    bool _tfidf = false; /**< whether to use TF/IDF */
    
    // internals
    std::unordered_map<std::string,Word> _vocab; /**< string to word stats, including pos */
    std::unordered_map<int,std::string> _rvocab; /**< pos to string */
    std::string _vocabfname = "vocab.dat";
    std::string _model_repo;
    std::string _correspname = "corresp.txt";
    
    // data
    std::vector<TxtBowEntry> _txt;
    std::vector<TxtBowEntry> _test_txt;
  };
  
}

#endif
