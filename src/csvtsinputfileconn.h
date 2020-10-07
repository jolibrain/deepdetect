/**
 * DeepDetect
 * Copyright (c) 2018-2019 Emmanuel Benazera
 * Author: Guillaume Infantes <guillaume.infantes@jolibrain.com>
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

#ifndef CSVTSINPUTFILECONN_H
#define CSVTSINPUTFILECONN_H
#include "inputconnectorstrategy.h"
#include "csvinputfileconn.h"
#include "utils/fileops.hpp"
#include <fstream>
#include <unordered_set>
#include <algorithm>
#include <random>

namespace dd
{

  class CSVTSInputFileConn;

  /**
   * \brief fetched data element for timeseries
   */
  class DDCsvTS
  {
  public:
    DDCsvTS() : _ddcsv()
    {
    }
    ~DDCsvTS()
    {
    }

    int read_file(const std::string &fname, bool is_test_data = false);
    int read_db(const std::string &fname);
    int read_mem(const std::string &content);
    int read_dir(const std::string &dir);

    DDCsv _ddcsv;
    CSVTSInputFileConn *_cifc = nullptr;
    APIData _adconf;
    std::shared_ptr<spdlog::logger> _logger;
  };

  /**
   * \brief generic CSV-TimeSeries input connector
   */
  class CSVTSInputFileConn : public CSVInputFileConn
  {
  public:
    CSVTSInputFileConn() : CSVInputFileConn()
    {
      this->_dont_scale_labels = false;
      this->_scale_between_minus1_and_1 = true;
      this->_timeserie = true;
    }

    ~CSVTSInputFileConn()
    {
    }

    CSVTSInputFileConn(const CSVTSInputFileConn &i)
        : CSVInputFileConn(i), _csvtsdata(i._csvtsdata),
          _csvtsdata_test(i._csvtsdata_test)
    {
      this->_scale_between_minus1_and_1 = i._scale_between_minus1_and_1;
      this->_dont_scale_labels = i._dont_scale_labels;
      this->_min_vals = i._min_vals;
      this->_max_vals = i._max_vals;
      this->_timeserie = true;
    }

    void init(const APIData &ad)
    {
      fillup_parameters(ad);
    }

    void fillup_parameters(const APIData &ad_input)
    {
      if (ad_input.has("scale") && ad_input.get("scale").get<bool>())
        {
          _scale = true;
        }
      deserialize_bounds();
      CSVInputFileConn::fillup_parameters(ad_input);

      // timeout
      this->set_timeout(ad_input);
    }

    void shuffle_data(std::vector<std::vector<CSVline>> cvstsdata);

    /**
     * \brief shuffles _csvtsdata if shuffle is set to true
     */
    void shuffle_data_if_needed();

    /**
     * \brief uses _test_split value to split the input dataset
     * @param cvstsdata full dataset, in output reduced to 1-_test_split size
     * @param cvstsdata_test test dataset sink, in output of size _test_split
     */
    void split_data(std::vector<std::vector<CSVline>> &cvstsdata,
                    std::vector<std::vector<CSVline>> &cvstsdata_test);

    void transform(const APIData &ad);

    int batch_size() const
    {
      return 1;
    }

    int test_batch_size() const
    {
      return 1;
    }

    /**
     * \brief merge the local and argument categorical variable list and
     * values, both maps end up containing the merged variables.
     * @param categoricals list of categoricals to be merged
     */
    void merge_categoricals(
        std::unordered_map<std::string, CCategorical> &categoricals);

    /**
     * \brief merge the local min/max values into the argument vectors
     * @param min_vals min bounds to be updated
     * @param max_vals max bounds to be updated
     */
    void merge_min_max(std::vector<double> &min_vals,
                       std::vector<double> &max_vals);

    // read min max values, return false if not present

    /**
     * \brief read min/max per variable bounds from file
     * @param force to update the bounds even if they do already exist in
     *        memory
     * @return true if successful, false otherwise
     */
    bool deserialize_bounds(const bool &force = false);

    /**
     * \brief serializes per variable min/max bounds to file, throws on error
     */
    void serialize_bounds();

    /**
     * \brief fills out response params from input connector values
     * @param out APIData that holds the output values
     */
    void response_params(APIData &out);

    /**
     * \brief transfers data read by the CSV connector to _csvtsdata holder
     * @param is_test_data whether data is from test set
     */
    void push_csv_to_csvts(const bool &is_test_data = false);

    std::string _boundsfname
        = "bounds.dat"; /**< variables min/max bounds filename. */

    std::vector<std::vector<CSVline>> _csvtsdata;
    std::vector<std::vector<CSVline>> _csvtsdata_test;
    std::vector<std::string> _fnames;
    std::vector<std::string> _test_fnames;
  };
}

#ifdef USE_XGBOOST
#include "backends/xgb/xgbinputconns.h"
#endif

#ifdef USE_TSNE
#include "backends/tsne/tsneinputconns.h"
#endif

#endif
