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

#include "csvtsinputfileconn.h"
#include "utils/utils.hpp"
#include <iomanip>

namespace dd
{

  int DDCsvTS::read_file(const std::string &fname, bool is_test_data)
  {
    if (_cifc)
      {
        _cifc->_columns.clear();
        std::string testfname = _cifc->_csv_test_fname;
        _cifc->_csv_test_fname = "";
        _cifc->read_csv(fname, true);
        _cifc->_csv_test_fname = testfname;
        _cifc->push_csv_to_csvts(is_test_data);
        if (is_test_data)
          _cifc->_test_fnames.push_back(fname);
        else
          _cifc->_fnames.push_back(fname);
        return 0;
      }
    else
      return -1;
  }

  int DDCsvTS::read_db(const std::string &fname)
  {
    _cifc->_db_fname = fname;
    return 0;
  }

  int DDCsvTS::read_mem(const std::string &content)
  {
    if (_cifc)
      {
        // tokenize on END_OF_SEQ  markers
        std::string delim = "END_OF_SEQ";
        size_t start = 0;
        size_t next = 0;
        while ((next = content.find(delim, start)) != std::string::npos)
          {
            std::string csvfile = content.substr(start, next - start);
            _ddcsv._cifc = _cifc;
            _ddcsv.read_mem(csvfile);
            _cifc->push_csv_to_csvts();
            start = next + delim.length();
          }
        std::string csvfile = content.substr(start, next - start);
        _ddcsv._cifc = _cifc;
        _ddcsv.read_mem(csvfile);
        _cifc->push_csv_to_csvts();
        return 0;
      }
    else
      return -1;
  }

  int DDCsvTS::read_dir(const std::string &dir)
  {
    //- list all CSV files in directory
    std::unordered_set<std::string> trainfiles;
    int ret = fileops::list_directory(dir, true, false, true, trainfiles);
    if (ret != 0)
      return ret;
    // then simply read them
    if (!_cifc)
      return -1;

    //- pick one file up and read header once
    std::string fname = (*trainfiles.begin());
    std::ifstream csv_file(fname, std::ios::binary);
    if (!csv_file.is_open())
      throw InputConnectorBadParamException("cannot open file " + fname);
    std::string hline;
    std::getline(csv_file, hline);
    _cifc->read_header(hline);

    //- read all test files
    std::unordered_set<std::string> testfiles;
    if (!_cifc->_csv_test_fname.empty())
      fileops::list_directory(_cifc->_csv_test_fname, true, false, true,
                              testfiles);

    std::unordered_set<std::string> allfiles = trainfiles;

    //- aggregate all files = train + test
    allfiles.insert(testfiles.begin(), testfiles.end());

    //- read categoricals first if any as it affects the number of columns (and
    // thus bounds)
    if (!_cifc->_categoricals.empty())
      {
        std::unordered_map<std::string, CCategorical> categoricals;
        for (auto fname : allfiles)
          {
            csv_file = std::ifstream(fname, std::ios::binary);
            if (!csv_file.is_open())
              throw InputConnectorBadParamException("cannot open file "
                                                    + fname);
            std::string hline;
            std::getline(csv_file, hline); // skip header

            // read on categoricals
            _cifc->fillup_categoricals(csv_file);
            _cifc->merge_categoricals(categoricals);
          }
      }

    //- read bounds across all TS CSV files
    if (_cifc->_scale
        && (_cifc->_min_vals.empty() || _cifc->_max_vals.empty()))
      {
        std::vector<double> min_vals(_cifc->_min_vals);
        std::vector<double> max_vals(_cifc->_max_vals);
        for (auto fname : allfiles)
          {
            csv_file = std::ifstream(fname, std::ios::binary);
            if (!csv_file.is_open())
              throw InputConnectorBadParamException("cannot open file "
                                                    + fname);
            std::string hline;
            std::getline(csv_file, hline); // skip header

            //- read bounds min/max
            _cifc->_min_vals.clear();
            _cifc->_max_vals.clear();
            _cifc->find_min_max(csv_file);
            _cifc->merge_min_max(min_vals, max_vals);
          }

        //- update global bounds
        _cifc->_min_vals = min_vals;
        _cifc->_max_vals = max_vals;
        _cifc->serialize_bounds();
      }

    //- read TS CSV data train/test
    for (auto fname : trainfiles) // train data
      read_file(fname, false);
    _cifc->shuffle_data_if_needed();
    for (auto fname : testfiles) // test data
      read_file(fname, true);
    _cifc->update_columns();
    return 0;
  }

  void CSVTSInputFileConn::shuffle_data_if_needed()
  {
    if (_shuffle)
      shuffle_data(_csvtsdata);
  }

  void
  CSVTSInputFileConn::shuffle_data(std::vector<std::vector<CSVline>> csvtsdata)
  {
    std::shuffle(csvtsdata.begin(), csvtsdata.end(), _g);
  }

  void CSVTSInputFileConn::split_data(
      std::vector<std::vector<CSVline>> &csvtsdata,
      std::vector<std::vector<CSVline>> &csvtsdata_test)
  {
    if (_test_split > 0.0)
      {
        int split_size = std::floor(csvtsdata.size() * (1.0 - _test_split));
        auto chit = csvtsdata.begin();
        auto dchit = chit;
        int cpos = 0;
        while (chit != csvtsdata.end())
          {
            if (cpos == split_size)
              {
                if (dchit == csvtsdata.begin())
                  dchit = chit;
                csvtsdata_test.push_back((*chit));
              }
            else
              ++cpos;
            ++chit;
          }
        csvtsdata.erase(dchit, csvtsdata.end());
      }
  }

  void CSVTSInputFileConn::transform(const APIData &ad)
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
        if (fileops::file_exists(_uris.at(0))) // training from dir
          {
            _csv_fname = _uris.at(0);
            if (_uris.size() > 1)
              _csv_test_fname = _uris.at(1);
          }
        else // training from memory
          {
          }

        if (!_csv_fname.empty()) // when training from file
          {
            DataEl<DDCsvTS> ddcsvts(this->_input_timeout);
            ddcsvts._ctype._cifc = this;
            ddcsvts._ctype._adconf = ad_input;
            ddcsvts.read_element(_csv_fname, this->_logger);
            // this read element will call read csvts_dir and scale and shuffle
          }
        else
          {
            for (size_t i = uri_offset; i < _uris.size(); i++)
              {
                DataEl<DDCsvTS> ddcsvts(this->_input_timeout);
                ddcsvts._ctype._cifc = this;
                ddcsvts._ctype._adconf = ad_input;
                ddcsvts.read_element(_uris.at(i), this->_logger);
              }
            if (_scale)
              {
                for (size_t j = 0; j < _csvtsdata.size(); j++)
                  {
                    for (size_t k = 0; k < _csvtsdata.at(j).size(); k++)
                      scale_vals(_csvtsdata.at(j).at(k)._v);
                  }
              }
            shuffle_data(_csvtsdata);
            if (_test_split > 0.0)
              {
                split_data(_csvtsdata, _csvtsdata_test);
                std::cerr << "data split test size=" << _csvdata_test.size()
                          << " / remaining data size=" << _csvdata.size()
                          << std::endl;
              }
          }
      }
    else // prediction mode
      {

        for (size_t i = 0; i < _uris.size(); i++)
          {
            if (i == 0 && !fileops::file_exists(_uris.at(0))
                && (ad_input.size()
                    && _uris.at(0).find(_delim)
                           != std::string::
                               npos)) // first line might be the header if we
                                      // have some options to consider //TODO:
                                      // prevents reading from CSV file
              {
                read_header(_uris.at(0));
                continue;
              }
            /*else if (!_categoricals.empty())
              throw InputConnectorBadParamException("use of
              categoricals_mapping requires a CSV header");*/
            DataEl<DDCsvTS> ddcsvts(this->_input_timeout);
            ddcsvts._ctype._cifc = this;
            ddcsvts._ctype._adconf = ad_input;
            ddcsvts.read_element(_uris.at(i), this->_logger);
          }
      }
    if (_csvtsdata.empty() && _db_fname.empty())
      throw InputConnectorBadParamException("no data could be found");
  }

  void CSVTSInputFileConn::response_params(APIData &out)
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
            adinput.add("connector", std::string("csvts"));
            adparams.add("input", adinput);
          }
      }
    APIData adinput = adparams.getobj("input");
    if (_scale)
      {
        adinput.add("min_vals", _min_vals);
        adinput.add("max_vals", _max_vals);
      }
    adparams.add("input", adinput);
    out.add("parameters", adparams);
  }

  void CSVTSInputFileConn::push_csv_to_csvts(const bool &is_test_data)
  {
    if (_csvdata.size())
      {
        if (is_test_data)
          {
            _csvtsdata_test.push_back(_csvdata);
          }
        else
          {
            _csvtsdata.push_back(_csvdata);
          }
        _csvdata.clear();
      }
    if (_csvdata_test.size())
      {
        _csvtsdata_test.push_back(_csvdata_test);
        _csvdata_test.clear();
      }
  }

  void CSVTSInputFileConn::merge_categoricals(
      std::unordered_map<std::string, CCategorical> &categoricals)
  {
    std::unordered_map<std::string, CCategorical>::const_iterator chit
        = _categoricals.begin();
    while (chit != _categoricals.end())
      {
        std::unordered_map<std::string, CCategorical>::iterator dchit;
        if ((dchit = categoricals.find((*chit).first)) != categoricals.end())
          {
            std::unordered_map<std::string, int>::const_iterator vit
                = (*chit).second._vals.begin();
            while (vit != (*chit).second._vals.end())
              {
                (*dchit).second.add_cat((*vit).first, (*vit).second);
                ++vit;
              }
          }
        else
          {
            categoricals.insert(std::pair<std::string, CCategorical>(
                (*chit).first, (*chit).second));
          }
        ++chit;
      }
    _categoricals = categoricals;
    // debug
    /*std::cerr << "categoricals size=" << _categoricals.size() << std::endl;
      std::unordered_map<std::string,CCategorical>::const_iterator chit
      =_categoricals.begin();
      while(chit!=_cifc->_categoricals.end())
      {
      std::cerr << "cat=" << (*chit).first << " / cat size=" <<
      (*chit).second._vals.size() << std::endl;
      ++chit;
      }*/
    // debug
  }

  void CSVTSInputFileConn::merge_min_max(std::vector<double> &min_vals,
                                         std::vector<double> &max_vals)
  {
    if (min_vals.empty())
      {
        for (size_t j = 0; j < _min_vals.size(); j++)
          min_vals.push_back(_min_vals.at(j));
      }
    else
      {
        for (size_t j = 0; j < _min_vals.size(); j++)
          min_vals.at(j) = std::min(_min_vals.at(j), min_vals.at(j));
      }
    if (max_vals.empty())
      for (size_t j = 0; j < _max_vals.size(); j++)
        max_vals.push_back(_max_vals.at(j));
    else
      for (size_t j = 0; j < _max_vals.size(); j++)
        max_vals.at(j) = std::max(_max_vals.at(j), max_vals.at(j));
  }

  bool CSVTSInputFileConn::deserialize_bounds(const bool &force)
  {
    if (!force && !_min_vals.empty() && !_max_vals.empty())
      return true;
    std::string boundsfname = _model_repo + "/" + _boundsfname;
    if (!fileops::file_exists(boundsfname))
      {
        _logger->info("no bounds file {}", boundsfname);
        return false;
      }
    std::ifstream in;
    in.open(boundsfname);
    if (!in.is_open())
      {
        _logger->warn("bounds file {} detected but cannot be opened",
                      boundsfname);
        return false;
      }
    std::string line;
    std::vector<std::string> tokens;
    int ncols = -1;
    int nlabels = -1;

    while (getline(in, line))
      {
        tokens = dd_utils::split(line, ':');
        if (tokens.empty())
          continue;
        std::string key = tokens.at(0);

        if (key == "ncols")
          ncols = std::atoi(tokens.at(1).c_str());
        else if (key == "nlabels")
          nlabels = std::atoi(tokens.at(1).c_str());
        else if (key == "label_pos")
          {
            _label_pos.clear();
            for (int i = 0; i < nlabels; ++i)
              _label_pos.push_back(std::atoi(tokens.at(i + 1).c_str()));
          }
        else if (key == "min_vals")
          {
            _min_vals.clear();
            for (int i = 0; i < ncols; ++i)
              _min_vals.push_back(std::atof(tokens.at(i + 1).c_str()));
          }
        else if (key == "max_vals")
          {
            _max_vals.clear();
            for (int i = 0; i < ncols; ++i)
              _max_vals.push_back(std::atof(tokens.at(i + 1).c_str()));
          }
      }
    this->_logger->info("bounds loaded");
    in.close();
    return true;
  }

  void CSVTSInputFileConn::serialize_bounds()
  {
    static int boundsprecision = 15;
    std::string boundsfname = _model_repo + "/" + _boundsfname;
    std::string delim = ":";
    std::ofstream out;
    out.open(boundsfname);
    if (!out.is_open())
      throw InputConnectorBadParamException(
          "failed opening for writing bounds file " + boundsfname);

    out << "ncols: " << _min_vals.size() << std::endl;
    out << "nlabels: " << _label_pos.size() << std::endl;
    out << "label_pos: ";
    for (unsigned int i = 0; i < _label_pos.size() - 1; ++i)
      out << " " << _label_pos[i] << " " << delim;
    out << " " << _label_pos[_label_pos.size() - 1] << std::endl;
    out << "min_vals: ";
    for (unsigned int i = 0; i < _min_vals.size() - 1; ++i)
      out << " " << std::setprecision(boundsprecision) << _min_vals[i] << " "
          << delim;
    out << " " << std::setprecision(boundsprecision)
        << _min_vals[_min_vals.size() - 1] << std::endl;
    out << "max_vals: ";
    for (unsigned int i = 0; i < _max_vals.size() - 1; ++i)
      out << " " << std::setprecision(boundsprecision) << _max_vals[i] << " "
          << delim;
    out << " " << std::setprecision(boundsprecision)
        << _max_vals[_max_vals.size() - 1] << std::endl;

    out.close();
    deserialize_bounds(true);
  }
}
