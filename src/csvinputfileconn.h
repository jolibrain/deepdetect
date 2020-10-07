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
#include "utils/fileops.hpp"
#include <fstream>
#include <unordered_set>
#include <algorithm>
#include <random>

namespace dd
{
  class CSVInputFileConn;

  /**
   * \brief fetched data element for CSV inputs
   */
  class DDCsv
  {
  public:
    DDCsv()
    {
    }
    ~DDCsv()
    {
    }

    int read_file(const std::string &fname);
    int read_db(const std::string &fname);
    int read_mem(const std::string &content);
    int read_dir(const std::string &dir)
    {
      throw InputConnectorBadParamException(
          "uri " + dir + " is a directory, requires a CSV file");
    }

    CSVInputFileConn *_cifc = nullptr;
    APIData _adconf;
    std::shared_ptr<spdlog::logger> _logger;
  };

  /**
   * \brief In-memory CSV data line holder
   */
  class CSVline
  {
  public:
    CSVline(const std::string &str, const std::vector<double> &v)
        : _str(str), _v(v)
    {
    }
    ~CSVline()
    {
    }
    std::string _str;       /**< csv line id */
    std::vector<double> _v; /**< csv line data */
  };

  /**
   * \brief Categorical values mapper.
   *        Categorical values are discrete sets that are converted to int
   *        This class builds and holds the mapper from value to int
   */
  class CCategorical
  {
  public:
    CCategorical()
    {
    }
    ~CCategorical()
    {
    }

    /**
     * \brief adds a categorical value and its position
     * @param v is the categorical value
     * @param val is the categorical value position in the discrete set
     */
    void add_cat(const std::string &v, const int &val)
    {
      std::unordered_map<std::string, int>::iterator hit;
      if ((hit = _vals.find(v)) == _vals.end())
        _vals.insert(std::pair<std::string, int>(v, val));
    }

    /**
     * \brief adds categorical value at the last position in the discrete set
     * @param v is the categorical value
     */
    void add_cat(const std::string &v)
    {
      add_cat(v, _vals.size());
    }

    /**
     * \brief gets the discrete value for a categorical value
     * @param v is the categorical value
     * @return the discrete position value
     */
    int get_cat_num(const std::string &v) const
    {
      std::unordered_map<std::string, int>::const_iterator hit;
      if ((hit = _vals.find(v)) != _vals.end())
        return (*hit).second;
      return -1;
    }

    std::unordered_map<std::string, int>
        _vals; /**< categorical value mapping. */
  };

  /**
   * \brief Generic CSV data input connector
   */
  class CSVInputFileConn : public InputConnectorStrategy
  {
  public:
    CSVInputFileConn() : InputConnectorStrategy()
    {
    }
    ~CSVInputFileConn()
    {
    }

    void init(const APIData &ad)
    {
      fillup_parameters(ad);
    }

    void fillup_parameters(const APIData &ad_input)
    {

      if (ad_input.has("shuffle") && ad_input.get("shuffle").get<bool>())
        {
          _shuffle = true;
          if (ad_input.has("seed") && ad_input.get("seed").get<int>() >= 0)
            {
              _g = std::mt19937(ad_input.get("seed").get<int>());
            }
          else
            {
              std::random_device rd;
              _g = std::mt19937(rd());
            }
        }

      if (ad_input.has("id"))
        _id = ad_input.get("id").get<std::string>();
      if (ad_input.has("separator"))
        _delim = ad_input.get("separator").get<std::string>();

      if (ad_input.has("ignore"))
        {
          std::vector<std::string> vignore
              = ad_input.get("ignore").get<std::vector<std::string>>();
          for (std::string s : vignore)
            _ignored_columns.insert(s);
        }

      if (ad_input.has("test_split"))
        _test_split = ad_input.get("test_split").get<double>();

      // read categorical mapping, if any
      read_categoricals(ad_input);

      // read scaling parameters, if any
      read_scale_vals(ad_input);

      if (ad_input.has("label"))
        {
          try
            {
              std::string label = ad_input.get("label").get<std::string>();
              // weird stuff may happen if label is given as single string (not
              // vector) both at service creation and at train (mutiple calls
              // may be possible due to csvts fillup parameters everywhere it
              // can)
              bool already_in_labels = false;
              for (std::string l : _label)
                {
                  if (l == label)
                    {
                      already_in_labels = true;
                      break;
                    }
                }
              if (!already_in_labels)
                _label.push_back(label);
            }
          catch (std::exception &e)
            {
              try
                {
                  _label
                      = ad_input.get("label").get<std::vector<std::string>>();
                }
              catch (std::exception &e)
                {
                  throw InputConnectorBadParamException(
                      "wrong type for label parameter");
                }
            }
          _label_pos.clear();
          for (size_t l = 0; l < _label.size(); l++)
            {
              _label_pos.push_back(-1);
              _label_set.insert(std::pair<std::string, int>(_label.at(l), l));
            }
        }
      if (ad_input.has("label_offset"))
        {
          try
            {
              int label_offset = ad_input.get("label_offset").get<int>();
              _label_offset.push_back(label_offset);
            }
          catch (std::exception &e)
            {
              try
                {
                  _label_offset
                      = ad_input.get("label_offset").get<std::vector<int>>();
                }
              catch (std::exception &e)
                {
                  throw InputConnectorBadParamException(
                      "wrong type for label_offset parameter");
                }
            }
        }
      else
        _label_offset = std::vector<int>(_label.size(), 0);

      if (ad_input.has("categoricals"))
        {
          std::vector<std::string> vcats
              = ad_input.get("categoricals").get<std::vector<std::string>>();
          for (std::string v : vcats)
            _categoricals.emplace(std::make_pair(v, CCategorical()));
        }

      // timeout
      this->set_timeout(ad_input);
    }

    /**
     * \brief reads a categorical value mapping from inputs
     *        this most often applies when the mapping is provided at inference
     *        time.
     */
    void read_categoricals(const APIData &ad_input)
    {
      if (ad_input.has("categoricals_mapping"))
        {
          APIData ad_cats = ad_input.getobj("categoricals_mapping");
          std::vector<std::string> vcats = ad_cats.list_keys();
          for (std::string c : vcats)
            {
              APIData ad_cat = ad_cats.getobj(c);
              CCategorical cc;
              _categoricals.insert(std::make_pair(c, cc));
              auto chit = _categoricals.find(c);
              std::vector<std::string> vcvals = ad_cat.list_keys();
              for (std::string v : vcvals)
                {
                  (*chit).second.add_cat(v, ad_cat.get(v).get<int>());
                }
            }
        }
    }

    /**
     * \brief scales a vector of double based on min/max bounds
     * @param vals the vector with values to be scaled
     */
    void scale_vals(std::vector<double> &vals)
    {
      auto lit = _columns.begin();
      for (int j = 0; j < (int)vals.size(); j++)
        {
          bool j_is_id
              = (_columns.empty() || _id.empty()) ? false : (*lit) == _id;
          if (j_is_id)
            {
              ++lit;
              continue;
            }
          bool equal_bounds = (_max_vals.at(j) == _min_vals.at(j));
          if (equal_bounds)
            {
              ++lit;
              continue;
            }
          if (_dont_scale_labels)
            {
              bool j_is_label = false;
              if (!_columns.empty()
                  && std::find(_label_pos.begin(), _label_pos.end(), j)
                         != _label_pos.end())
                j_is_label = true;
              if (j_is_label)
                {
                  ++lit;
                  continue;
                }
            }
          vals.at(j) = (vals.at(j) - _min_vals.at(j))
                       / (_max_vals.at(j) - _min_vals.at(j));
          if (_scale_between_minus1_and_1)
            vals.at(j) = vals.at(j) - 0.5;
          ++lit;
        }
    }

    /**
     * \brief read min/max bounds for scaling input data
     *        sets _scale flag and _min_vals, _max_vals vectors
     * @param ad_input the APIData input object
     */
    void read_scale_vals(const APIData &ad_input)
    {
      if (ad_input.has("scale") && ad_input.get("scale").get<bool>())
        {
          _scale = true;
          if (ad_input.has("min_vals"))
            {
              try
                {
                  _min_vals
                      = ad_input.get("min_vals").get<std::vector<double>>();
                }
              catch (...)
                {
                  std::vector<int> vi
                      = ad_input.get("min_vals").get<std::vector<int>>();
                  _min_vals = std::vector<double>(vi.begin(), vi.end());
                }
            }
          if (ad_input.has("max_vals"))
            {
              try
                {
                  _max_vals
                      = ad_input.get("max_vals").get<std::vector<double>>();
                }
              catch (...)
                {
                  std::vector<int> vi
                      = ad_input.get("max_vals").get<std::vector<int>>();
                  _max_vals = std::vector<double>(vi.begin(), vi.end());
                }
            }

          // debug
          /*std::cout << "loaded min/max scales:\n";
          std::copy(_min_vals.begin(),_min_vals.end(),std::ostream_iterator<double>(std::cout,"
          ")); std::cout << std::endl;
          std::copy(_max_vals.begin(),_max_vals.end(),std::ostream_iterator<double>(std::cout,"
          ")); std::cout << std::endl;*/
          // debug

          if (!_train && (_max_vals.empty() || _min_vals.empty()))
            throw InputConnectorBadParamException(
                "predict: failed acquiring scaling min_vals or max_vals");
        }
    }

    /**
     * \brief shuffle CSV data vector if shuffle flag is true
     * @param csvdata CSV data line vector to be shuffled
     */
    void shuffle_data(std::vector<CSVline> &csvdata)
    {
      if (_shuffle)
        std::shuffle(csvdata.begin(), csvdata.end(), _g);
    }

    /**
     * \brief uses _test_split value to split the input dataset
     * @param csvdata is the full CSV dataset holder, in output reduced to size
     *        1-_test_split
     * @param csvdata_test is the test dataset sink, in otput of size
     *        _test_split
     */
    void split_data(std::vector<CSVline> &csvdata,
                    std::vector<CSVline> &csvdata_test)
    {
      if (_test_split > 0.0)
        {
          int split_size = std::floor(csvdata.size() * (1.0 - _test_split));
          auto chit = csvdata.begin();
          auto dchit = chit;
          int cpos = 0;
          while (chit != csvdata.end())
            {
              if (cpos == split_size)
                {
                  if (dchit == csvdata.begin())
                    dchit = chit;
                  csvdata_test.push_back((*chit));
                }
              else
                ++cpos;
              ++chit;
            }
          csvdata.erase(dchit, csvdata.end());
        }
    }

    /**
     * \brief adds a CSV data value line to the training set
     * @param id
     * @param vals
     */
    virtual void add_train_csvline(const std::string &id,
                                   std::vector<double> &vals)
    {
      _csvdata.emplace_back(id, std::move(vals));
    }

    /**
     * \brief adds a CSV data value line to the test set
     * @param id
     * @param vals
     */
    virtual void add_test_csvline(const std::string &id,
                                  std::vector<double> &vals)
    {
      _csvdata_test.emplace_back(id, std::move(vals));
    }

    /**
     * \brief input data transforms
     * @param ad APIData input object
     */
    void transform(const APIData &ad)
    {
      get_data(ad);
      APIData ad_input = ad.getobj("parameters").getobj("input");
      fillup_parameters(ad_input);

      /**
       * Training from either file or memory.
       */
      if (_train)
        {
          int uri_offset = 0;
          if (fileops::file_exists(_uris.at(0))) // training from file
            {
              _csv_fname = _uris.at(0);
              if (_uris.size() > 1)
                _csv_test_fname = _uris.at(1);
            }
          else // training from memory
            {
            }

          // check on common and required parameters
          bool autoencoder = ad_input.has("autoencoder")
                             && ad_input.get("autoencoder").get<bool>();
          if (!ad_input.has("label") && _train && _label.empty()
              && !autoencoder)
            throw InputConnectorBadParamException(
                "missing label column parameter");

          if (!_csv_fname.empty()) // when training from file
            {
              DataEl<DDCsv> ddcsv(this->_input_timeout);
              ddcsv._ctype._cifc = this;
              ddcsv._ctype._adconf = ad_input;
              ddcsv.read_element(_csv_fname, this->_logger);
            }
          else // training from posted data (in-memory)
            {
              for (size_t i = uri_offset; i < _uris.size(); i++)
                {
                  DataEl<DDCsv> ddcsv(this->_input_timeout);
                  ddcsv._ctype._cifc = this;
                  ddcsv._ctype._adconf = ad_input;
                  ddcsv.read_element(_uris.at(i), this->_logger);
                }
              if (_scale)
                {
                  for (size_t j = 0; j < _csvdata.size(); j++)
                    {
                      scale_vals(_csvdata.at(j)._v);
                    }
                }
              shuffle_data(_csvdata);
              if (_test_split > 0.0)
                split_data(_csvdata, _csvdata_test);
              // std::cerr << "data split test size=" << _csvdata_test.size()
              // << " / remaining data size=" << _csvdata.size() << std::endl;
              if (!_ignored_columns.empty() || !_categoricals.empty())
                update_columns();
            }
        }
      else // prediction mode
        {
          for (size_t i = 0; i < _uris.size(); i++)
            {
              if (i == 0 && !fileops::file_exists(_uris.at(0))
                  && (!_categoricals.empty()
                      || (ad_input.size() && !_id.empty()
                          && _uris.at(0).find(_delim)
                                 != std::string::
                                     npos))) // first line might be the header
                                             // if we have some options to
                                             // consider //TODO: prevents
                                             // reading from CSV file
                {
                  read_header(_uris.at(0));
                  continue;
                }
              /*else if (!_categoricals.empty())
                throw InputConnectorBadParamException("use of
                categoricals_mapping requires a CSV header");*/
              DataEl<DDCsv> ddcsv(this->_input_timeout);
              ddcsv._ctype._cifc = this;
              ddcsv._ctype._adconf = ad_input;
              ddcsv.read_element(_uris.at(i), this->_logger);
            }
        }
      if (_csvdata.empty() && _db_fname.empty())
        throw InputConnectorBadParamException("no data could be found");
    }

    /**
     * \brief parse CSV header and sets the reference CSV columns
     * @param hline header line as string
     */
    void read_header(std::string &hline);

    /**
     * \brief reads a full CSV dataset and builds the categorical variables and
     *        values mapper
     * @param csv_file input stream for the CSV data file
     */
    void fillup_categoricals(std::ifstream &csv_file);

    /**
     * \brief reads a CSV data line, fills up values and categorical variables
     * as one-hot-vectors
     * @param hline CSV data line
     * @param delim CSV column delimiter
     * @param vals vector to be filled up with CSV data values
     * @param column_id stores the column that holds the line id
     * @param nlines current line counter
     */
    void read_csv_line(const std::string &hline, const std::string &delim,
                       std::vector<double> &vals, std::string &column_id,
                       int &nlines);

    /**
     * \brief reads a full CSV data file, calls read_csv_line
     * @param fname the CSV file name
     * @param forbid_shuffle whether shuffle is forbidden
     */
    void read_csv(const std::string &fname,
                  const bool &forbid_shuffle = false);

    int batch_size() const
    {
      return _csvdata.size();
    }

    int test_batch_size() const
    {
      return _csvdata_test.size();
    }

    int feature_size() const
    {
      if (!_id.empty())
        return _columns.size() - 1 - _label.size(); // minus label and id
      else
        return _columns.size() - _label.size(); // minus label
    }

    /**
     * \brief fills out response params from input connector values
     * @param out APIData that holds the output values
     */
    void response_params(APIData &out)
    {
      APIData adparams;
      if (_scale || !_categoricals.empty())
        {
          if (out.has("parameters"))
            {
              adparams = out.getobj("parameters");
            }
          if (!adparams.has("input"))
            {
              APIData adinput;
              adinput.add("connector", std::string("csv"));
              adparams.add("input", adinput);
            }
        }
      APIData adinput = adparams.getobj("input");
      if (_scale)
        {
          adinput.add("min_vals", _min_vals);
          adinput.add("max_vals", _max_vals);
        }
      if (!_categoricals.empty())
        {
          APIData cats;
          auto hit = _categoricals.begin();
          while (hit != _categoricals.end())
            {
              APIData adcat;
              auto chit = (*hit).second._vals.begin();
              while (chit != (*hit).second._vals.end())
                {
                  adcat.add((*chit).first, (*chit).second);
                  ++chit;
                }
              cats.add((*hit).first, adcat);
              ++hit;
            }
          adinput.add("categoricals_mapping", cats);
        }
      adparams.add("input", adinput);
      out.add("parameters", adparams);
    }

    /**
     * \brief tests whether a CSV column holds a categorical variable
     * @param c the CSV column
     * @return true if category, false otherwise
     */
    bool is_category(const std::string &c)
    {
      std::unordered_map<std::string, CCategorical>::const_iterator hit;
      if ((hit = _categoricals.find(c)) != _categoricals.end())
        return true;
      return false;
    }

    /**
     * \brief adds a value to a categorical variable mapping, modifies
     *        _categoricals
     * @param c the variable name (column)
     * @param v the new categorical value to be added
     */
    void update_category(const std::string &c, const std::string &val);

    /**
     * \brief update data columns with one-hot columns introduced to translate
     * categorical variables
     */
    void update_columns();

    /**
     * \brief returns min/max variable values across a CSV dataset
     * @param fname CSV filename
     * @return pair of vectors for min/max values
     */
    std::pair<std::vector<double>, std::vector<double>>
    get_min_max_vals(std::string &fname)
    {
      clear_min_max();
      find_min_max(fname);
      return get_min_max_vals();
    }

    /**
     * \brief finds min/max variable values across a CSV dataset
     * @param fname CSV filename
     */
    void find_min_max(std::string &fname)
    {
      std::ifstream csv_file(fname, std::ios::binary);
      // discard header
      std::string hline;
      std::getline(csv_file, hline);
      if (_columns.empty())
        {
          read_header(hline);
          find_min_max(csv_file);
          update_columns();
        }
      else
        find_min_max(csv_file);
    }

    /**
     * \brief finds min/max variable values across a CSV dataset
     * @param csv_file CSV file stream
     */
    void find_min_max(std::ifstream &csv_file);

    /**
     * \brief removes min/max values for the CSV dataset variables
     */
    void clear_min_max()
    {
      _min_vals.clear();
      _max_vals.clear();
    }

    /**
     * \brief get pre-obtained min/max variable values
     * @return pair of vector of min/max variable values
     */
    std::pair<std::vector<double>, std::vector<double>> get_min_max_vals()
    {
      return std::pair<std::vector<double>, std::vector<double>>(_min_vals,
                                                                 _max_vals);
    }

    /**
     * \brief returns a one-hot-vector of a given size and index
     * @param cnum the index of the positive one-hot
     * @param size the size of the vector
     * @return the one hot vector of double
     */
    std::vector<double> one_hot_vector(const int &cnum, const int &size)
    {
      std::vector<double> v(size, 0.0);
      v.at(cnum) = 1.0;
      return v;
    }

    // options
    bool _shuffle = false;
    std::mt19937 _g;
    std::string _csv_fname;          /**< csv main filename. */
    std::string _csv_test_fname;     /**< csv test filename (optional). */
    std::list<std::string> _columns; /**< list of csv columns. */
    std::vector<std::string> _label; /**< list of label columns. */
    std::unordered_map<std::string, int> _label_set;
    std::string _delim = ",";
    int _id_pos = -1;
    std::vector<int> _label_pos;    /**< column positions of the labels. */
    std::vector<int> _label_offset; /**< negative offset so that labels range
                                       from 0 onward */
    std::unordered_set<std::string>
        _ignored_columns; /**< set of ignored columns. */
    std::unordered_set<int>
        _ignored_columns_pos; /**< ignored columns indexes. */
    std::string _id;
    bool _scale = false; /**< whether to scale all data between 0 and 1 */
    bool _dont_scale_labels
        = true; // original csv input conn does not scale labels, while it is
                // needed for csv timeseries
    bool _scale_between_minus1_and_1
        = false; /**< whether to scale within [-1,1]. */
    std::vector<double>
        _min_vals; /**< upper bound used for auto-scaling data */
    std::vector<double>
        _max_vals; /**< lower bound used for auto-scaling data */
    std::unordered_map<std::string, CCategorical>
        _categoricals;       /**< auto-converted categorical variables */
    double _test_split = -1; /**< dataset test split ratio (optional). */
    int _detect_cols = -1;   /**< number of detected csv columns. */

    // data
    std::vector<CSVline> _csvdata;
    std::vector<CSVline> _csvdata_test;
    std::string _db_fname;
  };
}

#ifdef USE_XGBOOST
#include "backends/xgb/xgbinputconns.h"
#endif

#ifdef USE_TSNE
#include "backends/tsne/tsneinputconns.h"
#endif

#endif
