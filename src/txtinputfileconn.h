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
#include <algorithm>
#include <cmath>
#include "utf8.h"

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

  template<typename T> class TxtEntry
  {
  public:
    TxtEntry() {}
    TxtEntry(const float &target): _target(target) {}
    virtual ~TxtEntry() {}

    void reset() {}
    void get_elt(std::string &key, T &val) const { (void)key; (void)val; }
    bool has_elt() const { return false; }
    size_t size() const { return 0; }
    
    float _target = -1; /**< class target in training mode. */
    std::string _uri;
  };
  
  class TxtBowEntry: public TxtEntry<double>
  {
  public:
  TxtBowEntry():TxtEntry<double>() {};
  TxtBowEntry(const float &target):TxtEntry<double>(target) {}
    virtual ~TxtBowEntry() {}

    void add_word(const std::string &str,
		  const double &v,
		  const bool &count)
    {
      std::unordered_map<std::string,double>::iterator hit;
      if ((hit=_v.find(str))!=_v.end())
	{
	  if (count)
	    (*hit).second += v;
	}
      else _v.insert(std::pair<std::string,double>(str,v));
    }

    bool has_word(const std::string &str)
    {
      return _v.count(str);
    }

    void reset()
    {
      _vit = _v.begin();
    }

    void get_next_elt(std::string &key, double &val)
    {
      if (_vit!=_v.end())
	{
	  key = (*_vit).first;
	  val = (*_vit).second;
	  ++_vit;
	}
    }

    bool has_elt() const
    {
      return _vit != _v.end();
    }
    
    size_t size() const
    {
      return _v.size();
    }

    std::unordered_map<std::string,double> _v; /**< words as (<pos,val>). */
    std::unordered_map<std::string,double>::iterator _vit;
  };

  class TxtCharEntry: public TxtEntry<double>
  {
  public:
  TxtCharEntry():TxtEntry<double>() {}
  TxtCharEntry(const float &target):TxtEntry<double>(target) {}
    virtual ~TxtCharEntry() {}

    void add_char(const uint32_t &c)
    {
      _v.push_back(c);
    }

    void reset()
    {
      _vit = _v.begin();
    }

    void get_next_elt(std::string &key, double &val)
    {
      if (_vit!=_v.end())
	{
	  key = std::to_string((*_vit));
	  val = 1;
	  ++_vit;
	}
    }

    bool has_elt() const
    {
      return _vit != _v.end();
    }

    size_t size() const 
    {
      return _v.size();
    }
    
    std::vector<uint32_t> _v;
    std::vector<uint32_t>::iterator _vit;
  };
  
  class TxtInputFileConn : public InputConnectorStrategy
  {
  public:
    TxtInputFileConn()
      :InputConnectorStrategy() {}
    TxtInputFileConn(const TxtInputFileConn &i)
      :InputConnectorStrategy(i),_iterator(i._iterator),_tokenizer(i._tokenizer),_count(i._count),_tfidf(i._tfidf),_min_count(i._min_count),_min_word_length(i._min_word_length),_sentences(i._sentences),_characters(i._characters),_alphabet_str(i._alphabet_str),_alphabet(i._alphabet),_sequence(i._sequence),_seq_forward(i._seq_forward),_vocab(i._vocab) {}
    ~TxtInputFileConn()
      {
	destroy_txt_entries(_txt);
	destroy_txt_entries(_test_txt);
      }

    void init(const APIData &ad)
    {
      fillup_parameters(ad);
      if (!_characters && !_train)
	deserialize_vocab(false);
    }

    void fillup_parameters(const APIData &ad_input)
    {
      if (ad_input.has("shuffle"))
	_shuffle = ad_input.get("shuffle").get<bool>();
      if (ad_input.has("seed"))
	_seed = ad_input.get("seed").get<int>();
      if (ad_input.has("test_split"))
	_test_split = ad_input.get("test_split").get<double>();
      if (ad_input.has("count"))
	_count = ad_input.get("count").get<bool>();
      if (ad_input.has("tfidf"))
	_tfidf = ad_input.get("tfidf").get<bool>();
      if (ad_input.has("min_count"))
	_min_count = ad_input.get("min_count").get<int>();
      if (ad_input.has("min_word_length"))
	_min_word_length = ad_input.get("min_word_length").get<int>();
      if (ad_input.has("sentences"))
	_sentences = ad_input.get("sentences").get<bool>();
      if (ad_input.has("characters"))
	_characters = ad_input.get("characters").get<bool>();
      if (ad_input.has("lstm"))
  _lstm = ad_input.get("lstm").get<bool>();
      if (ad_input.has("alphabet"))
	_alphabet_str = ad_input.get("alphabet").get<std::string>();
      if (_characters)
	build_alphabet();
      if (ad_input.has("sequence"))
	_sequence = ad_input.get("sequence").get<int>();
      if (ad_input.has("read_forward"))
	_seq_forward = ad_input.get("read_forward").get<bool>();
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

      if (_alphabet.empty() && _characters)
	build_alphabet();
      
      if (!_characters && !_train && _vocab.empty())
	deserialize_vocab();
      
      for (std::string u: _uris)
	{
	  DataEl<DDTxt> dtxt;
	  dtxt._ctype._ctfc = this;
	  if (dtxt.read_element(u) || _txt.empty())
	    {
	      throw InputConnectorBadParamException("no data for text in " + u);
	    }
	  _txt.back()->_uri = u;
	}
      
      if (_train)
	serialize_vocab();

      // shuffle entries if requested
      if (_train && _shuffle)
	{
	  std::mt19937 g;
	  if (_seed >= 0)
	    g =std::mt19937(_seed);
	  else
	    {
	      std::random_device rd;
	      g = std::mt19937(rd());
	    }
	  std::shuffle(_txt.begin(),_txt.end(),g);
	}
      
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
	  std::cout << "vocab size=" << _vocab.size() << std::endl;
	}
      if (_txt.empty())
	throw InputConnectorBadParamException("no text could be found");
    }

    // text tokenization for BOW
    void parse_content(const std::string &content,
		       const float &target=-1);

    // serialization of vocabulary
    void serialize_vocab();
    void deserialize_vocab(const bool &required=true);

    // alphabet for character-level features
    void build_alphabet();

    // clearing up memory
    void destroy_txt_entries(std::vector<TxtEntry<double>*> &v);
    
    // options
    std::string _iterator = "document";
    std::string _tokenizer = "bow";
    bool _shuffle = false;
    int _seed = -1;
    double _test_split = 0.0;
    bool _count = true; /**< whether to add up word counters */
    bool _tfidf = false; /**< whether to use TF/IDF */
    int _min_count = 5; /**< min word occurence. */
    int _min_word_length = 5; /**< min word length. */
    bool _sentences = false; /**< whether to consider every sentence (\n separated) as a document. */
    bool _characters = false; /**< whether to use character-level input features. */
    std::string _alphabet_str = "abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}";
    std::unordered_map<uint32_t,int> _alphabet; /**< character-level alphabet. */
    int _sequence = 60; /**< sequence size when using character-level features. */
    bool _seq_forward = false; /**< whether to read character-based sequences forward. */
    bool _lstm = false;
    // internals
    std::unordered_map<std::string,Word> _vocab; /**< string to word stats, including word */
    std::string _vocabfname = "vocab.dat";
    std::string _correspname = "corresp.txt";
    
    // data
    std::vector<TxtEntry<double>*> _txt;
    std::vector<TxtEntry<double>*> _test_txt;
  };
  
}

#endif
