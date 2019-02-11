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

  class DDCsvTS
  {
  public:
  DDCsvTS(): _ddcsv()  {}
    ~DDCsvTS() {}

    int read_file(const std::string &fname, bool is_test_data = false);
    int read_db(const std::string &fname);
    int read_mem(const std::string &content);
    int read_dir(const std::string &dir, bool is_test_data = false, bool allow_read_test = true);

    DDCsv _ddcsv;
    CSVTSInputFileConn *_cifc = nullptr;
    APIData _adconf;
    std::shared_ptr<spdlog::logger> _logger;
  };


  class CSVTSInputFileConn : public CSVInputFileConn
  {
  public:



  CSVTSInputFileConn()
    :CSVInputFileConn() {}

    ~CSVTSInputFileConn() {}

  CSVTSInputFileConn(const CSVTSInputFileConn &i)
    : CSVInputFileConn(i), _csvtsdata(i._csvtsdata),
      _csvtsdata_test(i._csvtsdata_test)
        {
          this->_dont_scale_labels = false;
        }



    void fillup_parameters(const APIData &ad_input)
    {
      if (ad_input.has("scale") && ad_input.get("scale").get<bool>())
        {
          _scale = true;
          deserialize_bounds();
        }
      CSVInputFileConn::fillup_parameters(ad_input);
    }

    void shuffle_data(std::vector<std::vector<CSVline>> cvstsdata);
    void shuffle_data_if_needed();


    void split_data(std::vector<std::vector<CSVline>> cvstsdata,
                    std::vector<std::vector<CSVline>> cvstsdata_test);

    void transform(const APIData &ad);

    int batch_size() const
    {
      return 1;
    }

    int test_batch_size() const
    {
      return 1;
    }

    // read min max values, return false if not present
    bool deserialize_bounds();
    void serialize_bounds();

    /*   std::string s = "Boost,\"C++ Libraries\""; */
    /*   boost::escaped_list_separator<char> els('\\',_delim,'\"\''); */
    /*   tokenizer<boost::escaped_list_separator<char> tok(s,els); */
    /*   for (const auto &t : tok) */
    /*     std::cout << t << '\n'; */
    /* } */

    void response_params(APIData &out);


    void push_csv_to_csvts(DDCsv &ddcsv);
    void push_csv_to_csvts(bool is_test_data=false);

    std::string _boundsfname = "bounds.dat";

    std::vector<std::vector<CSVline>> _csvtsdata;
    std::vector<std::vector<CSVline>> _csvtsdata_test;
    std::vector<std::string> _fnames;


  };
}

#ifdef USE_XGBOOST
#include "backends/xgb/xgbinputconns.h"
#endif

#ifdef USE_TSNE
#include "backends/tsne/tsneinputconns.h"
#endif

#endif
