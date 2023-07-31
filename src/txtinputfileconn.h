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
#include <random>
#include "utf8.h"

namespace dd
{
  class TxtInputFileConn;

  class DDTxt
  {
  public:
    DDTxt()
    {
    }
    ~DDTxt()
    {
    }

    int read_file(const std::string &fname, int test_id);
    int read_db(const std::string &fname);
    int read_mem(const std::string &content);
    int read_dir(const std::string &dir, int test_id);

    TxtInputFileConn *_ctfc = nullptr;
    std::shared_ptr<spdlog::logger> _logger;
  };

  class Word
  {
  public:
    Word()
    {
    }
    Word(const int &pos) : _pos(pos)
    {
    }
    Word(const int &pos, const int &tcount, const int &tdocs)
        : _pos(pos), _total_count(tcount), _total_docs(tdocs)
    {
    }
    ~Word()
    {
    }

    int _pos = -1;
    int _total_count = 1;
    int _total_docs = 1;
  };

  template <typename T> class TxtEntry
  {
  public:
    TxtEntry()
    {
    }
    TxtEntry(const float &target) : _target(target)
    {
    }
    virtual ~TxtEntry()
    {
    }

    void reset()
    {
    }
    void get_elt(std::string &key, T &val) const
    {
      (void)key;
      (void)val;
    }
    bool has_elt() const
    {
      return false;
    }
    size_t size() const
    {
      return 0;
    }

    float _target = -1; /**< class target in training mode. */
    std::string _uri;
  };

  class TxtBowEntry : public TxtEntry<double>
  {
  public:
    TxtBowEntry() : TxtEntry<double>(){};
    TxtBowEntry(const float &target) : TxtEntry<double>(target)
    {
    }
    virtual ~TxtBowEntry()
    {
    }

    void add_word(const std::string &str, const double &v, const bool &count)
    {
      std::unordered_map<std::string, double>::iterator hit;
      if ((hit = _v.find(str)) != _v.end())
        {
          if (count)
            (*hit).second += v;
        }
      else
        _v.insert(std::pair<std::string, double>(str, v));
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
      if (_vit != _v.end())
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

    std::unordered_map<std::string, double> _v; /**< words as (<pos,val>). */
    std::unordered_map<std::string, double>::iterator _vit;
  };

  class TxtCharEntry : public TxtEntry<double>
  {
  public:
    TxtCharEntry() : TxtEntry<double>()
    {
    }
    TxtCharEntry(const float &target) : TxtEntry<double>(target)
    {
    }
    virtual ~TxtCharEntry()
    {
    }

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
      if (_vit != _v.end())
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

  class TxtOrderedWordsEntry : public TxtEntry<double>
  {
  public:
    TxtOrderedWordsEntry() : TxtEntry<double>()
    {
    }
    TxtOrderedWordsEntry(const float &target) : TxtEntry<double>(target)
    {
    }
    virtual ~TxtOrderedWordsEntry()
    {
    }

    void add_word(const std::string &word)
    {
      _v.push_back(word);
    }

    void reset()
    {
      _vit = _v.begin();
    }

    void get_next_elt(std::string &key, double &val)
    {
      if (_vit != _v.end())
        {
          key = *_vit;
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

    std::vector<std::string> _v;
    std::vector<std::string>::iterator _vit;
  };

  /** Tokenizer that uses greedy longest-match-first search to cut words
   * in pieces */
  class WordPieceTokenizer
  {
  public:
    std::vector<std::string> _tokens;
    TxtInputFileConn *_ctfc = nullptr;

    WordPieceTokenizer()
    {
    }

    void reset()
    {
      _tokens.clear();
    }

    void append_input(const std::string &word);

  public:
    bool in_vocab(const std::string &tok);

    std::string _suffix_start
        = "##"; /**< Suffix tokens in vocabulary are prefixed by this */
    std::string _word_start
        = ""; /**< Tokens corresponding to word or word beggining in the
                 vocabulary are prefixed by this */
    std::string _unk_token = "[UNK]";
  };

  class TxtInputFileConn : public InputConnectorStrategy
  {
  public:
    TxtInputFileConn() : InputConnectorStrategy()
    {
      _wordpiece_tokenizer._ctfc = this;
    }
    TxtInputFileConn(const TxtInputFileConn &i)
        : InputConnectorStrategy(i), _iterator(i._iterator),
          _test_split(i._test_split), _count(i._count), _tfidf(i._tfidf),
          _min_count(i._min_count), _min_word_length(i._min_word_length),
          _sentences(i._sentences), _characters(i._characters),
          _ordered_words(i._ordered_words), _lower_case(i._lower_case),
          _wordpiece_tokens(i._wordpiece_tokens),
          _punctuation_tokens(i._punctuation_tokens),
          _alphabet_str(i._alphabet_str), _alphabet(i._alphabet),
          _sequence(i._sequence), _seq_forward(i._seq_forward),
          _generate_vocab(i._generate_vocab), _vocab(i._vocab),
          _vocab_sep(i._vocab_sep),
          _wordpiece_tokenizer(i._wordpiece_tokenizer), _ndbed(i._ndbed)
    {
      _wordpiece_tokenizer._ctfc = this;
    }
    ~TxtInputFileConn()
    {
      destroy_txt_entries(_txt);
      for (auto te : _tests_txt)
        destroy_txt_entries(te);
      _tests_txt.clear();
    }

    void init(const APIData &ad)
    {
      fillup_parameters(ad);
      if (!_characters && !_train)
        deserialize_vocab(false);
    }

    void fillup_parameters(const APIData &ad_input)
    {
      auto params = ad_input.createSharedDTO<DTO::InputConnector>();
      fillup_parameters(params);
    }

    void fillup_parameters(oatpp::Object<DTO::InputConnector> params)
    {
      if (params->shuffle != nullptr)
        _shuffle = params->shuffle;
      if (params->seed != nullptr)
        _seed = params->seed;
      if (params->test_split != nullptr)
        _test_split = params->test_split;
      if (params->count != nullptr)
        _count = params->count;
      if (params->tfidf != nullptr)
        _tfidf = params->tfidf;
      if (params->min_count != nullptr)
        _min_count = params->min_count;
      if (params->min_word_length != nullptr)
        _min_word_length = params->min_word_length;
      if (params->sentences != nullptr)
        _sentences = params->sentences;
      if (params->characters != nullptr)
        _characters = params->characters;
      if (params->ordered_words != nullptr)
        _ordered_words = params->ordered_words;
      if (params->lower_case != nullptr)
        _lower_case = params->lower_case;
      if (params->wordpiece_tokens != nullptr)
        _wordpiece_tokens = params->wordpiece_tokens;
      if (params->word_start != nullptr)
        _wordpiece_tokenizer._word_start = params->word_start;
      if (params->suffix_start != nullptr)
        _wordpiece_tokenizer._suffix_start = params->suffix_start;
      if (params->punctuation_tokens != nullptr)
        _punctuation_tokens = params->punctuation_tokens;
      if (params->alphabet != nullptr)
        _alphabet_str = params->alphabet;
      if (_characters)
        build_alphabet();
      if (params->sequence != nullptr)
        _sequence = params->sequence;
      if (params->read_forward != nullptr)
        _seq_forward = params->read_forward;

      // timeout
      this->set_timeout(params);
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

    int test_batch_size(size_t test_id) const
    {
      return _tests_txt[test_id].size();
    }

    void transform(const APIData &ad)
    {
      get_data(ad);
      if (ad.has(
              "parameters")) // hotplug of parameters, overriding the defaults
        {
          APIData ad_param = ad.getobj("parameters");
          if (ad_param.has("input"))
            {
              fillup_parameters(ad_param.getobj("input"));
            }
        }

      transform(nullptr);
    }

    void transform(oatpp::Object<DTO::ServicePredict> input_dto)
    {
      if (input_dto != nullptr) // [temporary] == nullptr if called from
                                // transform(APIData)
        {
          get_data(input_dto);
          fillup_parameters(input_dto->parameters->input);
        }

      if (_alphabet.empty() && _characters)
        build_alphabet();

      if (!_characters && (!_train || _ordered_words) && _vocab.empty())
        deserialize_vocab();

      DataEl<DDTxt> dtxt(this->_input_timeout);
      dtxt._ctype._ctfc = this;
      if (dtxt.read_element(_uris[0], this->_logger, -1)
          || (_txt.empty() && _db_fname.empty() && _ndbed == 0))
        {
          throw InputConnectorBadParamException("no data for text in "
                                                + _uris[0]);
        }

      if (_ndbed == 0)
        {
          if (_db_fname.empty())
            _txt.back()->_uri = _uris[0];
          else
            return; // single db
        }

      for (size_t i = 1; i < _uris.size(); ++i)
        {
          DataEl<DDTxt> dtxt(this->_input_timeout);
          dtxt._ctype._ctfc = this;
          _tests_txt.resize(i);
          if (dtxt.read_element(_uris[i], this->_logger, i - 1)
              || (_txt.empty() && _db_fname.empty() && _ndbed == 0))
            {
              throw InputConnectorBadParamException("no data for text in "
                                                    + _uris[i]);
            }

          if (_ndbed == 0)
            {
              if (_db_fname.empty())
                _tests_txt[i - 1].back()->_uri = _uris[i];
              else
                return; // single db
            }
        }

      if (_train)
        {
          _shuffle = true;
          serialize_vocab();
        }

      // shuffle entries if requested
      if (_train && _shuffle && _ndbed == 0)
        {
          std::mt19937 g;
          if (_seed >= 0)
            g = std::mt19937(_seed);
          else
            {
              std::random_device rd;
              g = std::mt19937(rd());
            }
          std::shuffle(_txt.begin(), _txt.end(), g);
        }

      // split for test set
      if (_train && _test_split > 0 && _ndbed == 0 && _tests_txt.size() == 0)
        {
          _tests_txt.resize(1);
          int split_size = std::floor(_txt.size() * (1.0 - _test_split));
          auto chit = _txt.begin();
          auto dchit = chit;
          int cpos = 0;
          while (chit != _txt.end())
            {
              if (cpos == split_size)
                {
                  if (dchit == _txt.begin())
                    dchit = chit;
                  _tests_txt[0].push_back((*chit));
                }
              else
                ++cpos;
              ++chit;
            }
          _txt.erase(dchit, _txt.end());
          _logger->info(
              "data split test size=" + std::to_string(_tests_txt[0].size())
              + " / remaining data size=" + std::to_string(_txt.size()));
        }
      if (_txt.empty() && _ndbed == 0)
        // if _nbded != 0 _txt has be already db_ed and freed
        // and splitting / shuffling is done somewhere else
        throw InputConnectorBadParamException("no text could be found");
    }

    // text tokenization for BOW
    virtual void parse_content(const std::string &content,
                               const float &target = -1, int test_id = -1);
    // test -1 for train, 0 ,1  ... for test_id

    // serialization of vocabulary
    void serialize_vocab();
    void deserialize_vocab(const bool &required = true);

    // alphabet for character-level features
    void build_alphabet();

    // clearing up memory
    void destroy_txt_entries(std::vector<TxtEntry<double> *> &v);

    // options
    std::string _iterator = "document";
    bool _shuffle = false;
    int _seed = -1;
    double _test_split = 0.0;
    bool _count = true;       /**< whether to add up word counters */
    bool _tfidf = false;      /**< whether to use TF/IDF */
    int _min_count = 5;       /**< min word occurence. */
    int _min_word_length = 5; /**< min word length. */
    bool _sentences = false;  /**< whether to consider every sentence (\n
                                 separated) as a document. */
    bool _characters
        = false; /**< whether to use character-level input features. */
    bool _ordered_words = false; /**< whether to consider the position of each
                                    words in the sentence. */
    bool _lower_case = true;     /**< whether the input should be lower cased
                                    before processing */
    bool _wordpiece_tokens = false;   /**< whether to try to match word pieces
                                         from the vocabulary. */
    bool _punctuation_tokens = false; /**< accept punctuation tokens. */
    std::string _alphabet_str = "abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'"
                                "\"/\\|_@#$%^&*~`+-=<>()[]{}";
    std::unordered_map<uint32_t, int>
        _alphabet; /**< character-level alphabet. */
    int _sequence
        = 60; /**< sequence size when using character-level features. */
    bool _seq_forward
        = false; /**< whether to read character-based sequences forward. */

    // internals
    bool _generate_vocab = true;
    std::unordered_map<std::string, Word>
        _vocab; /**< string to word stats, including word */
    std::string _vocabfname = "vocab.dat";
    std::string _correspname = "corresp.txt";
    char _vocab_sep = ','; /**< vocabulary separator */
    int _dirs = 0;         /**< directories as input. */
    WordPieceTokenizer _wordpiece_tokenizer;

    // data
    std::vector<TxtEntry<double> *> _txt;
    std::vector<std::vector<TxtEntry<double> *>> _tests_txt;
    std::string _db_fname;

    int64_t _ndbed = 0;
  };

}

#endif
