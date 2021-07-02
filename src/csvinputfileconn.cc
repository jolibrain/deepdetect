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

#include "csvinputfileconn.h"
#include "utils/csv_parser.hpp"
#include "utils/utils.hpp"
#include <iomanip>

namespace dd
{

  /*- DDCsv -*/
  int DDCsv::read_file(const std::string &fname, int test_id)
  {
    (void)test_id;
    if (_cifc)
      {
        _cifc->read_csv(fname);
        return 0;
      }
    else
      return -1;
  }

  int DDCsv::read_db(const std::string &fname)
  {
    _cifc->_db_fname = fname;
    return 0;
  }

  int DDCsv::read_mem(const std::string &content)
  {
    if (!_cifc)
      return -1;

    std::stringstream sh(content);
    std::string line;
    int l = 0;
    while (std::getline(sh, line))
      {
        if (_cifc->_columns.empty() && _cifc->_train && l == 0)
          {
            _cifc->read_header(line);
            ++l;
            continue;
          }

        std::vector<double> vals;
        std::string cid;
        int nlines = 0;
        _cifc->read_csv_line(line, _cifc->_delim, vals, cid, nlines, false);
        // scale on the fly in predict mode
        // in train mode, will be scaled into transform() after everything is
        // read
        if (_cifc->_scale && !_cifc->_train)
          _cifc->scale_vals(vals);
        if (!cid.empty())
          _cifc->add_train_csvline(cid, vals);
        else
          _cifc->add_train_csvline(std::to_string(_cifc->_csvdata.size() + 1),
                                   vals);
        ++l;
      }
    _cifc->update_columns();
    return 0;
  }

  /*- CSVInputFileConn -*/
  void CSVInputFileConn::update_category(const std::string &c,
                                         const std::string &val)
  {
    std::unordered_map<std::string, CCategorical>::iterator hit;
    if ((hit = _categoricals.find(c)) != _categoricals.end())
      (*hit).second.add_cat(val);
  }

  void CSVInputFileConn::update_columns()
  {
    std::unordered_map<std::string, CCategorical>::iterator chit;
    auto lit = _columns.begin();
    std::list<std::string> ncolumns = _columns;
    auto nlit = ncolumns.begin();
    while (lit != _columns.end())
      {
        if (is_category((*lit)))
          {
            chit = _categoricals.find((*lit));
            auto hit = (*chit).second._vals.begin();
            while (hit != (*chit).second._vals.end())
              {
                std::string ncolname = (*lit) + "_" + (*hit).first;
                auto nlit2 = nlit;
                nlit = ncolumns.insert(nlit, ncolname);
                if (hit == (*chit).second._vals.begin())
                  ncolumns.erase(nlit2);
                ++hit;
              }
          }
        ++lit;
        ++nlit;
      }
    _columns = ncolumns;

    // update label_pos and id_pos
    int i = 0;
    lit = _columns.begin();
    while (lit != _columns.end())
      {
        if ((*lit) == _id)
          _id_pos = i;
        else
          for (unsigned int j = 0; j < _label.size(); ++j)
            if ((*lit) == _label[j])
              _label_pos[j] = i;
        ++i;
        ++lit;
      }

    // debug
    /*std::cerr << "number of new columns=" << _columns.size() << std::endl;
    std::cerr << "new CSV columns:\n";
    std::copy(_columns.begin(),_columns.end(),
              std::ostream_iterator<std::string>(std::cout," "));
              std::cerr << std::endl;*/
    // debug
  }

  void CSVInputFileConn::read_csv_line(const std::string &hline,
                                       const std::string &delim,
                                       std::vector<double> &vals,
                                       std::string &column_id, int &nlines,
                                       const bool &test)
  {
    std::string col;
    std::unordered_set<int>::const_iterator hit;
    std::stringstream sh(hline);
    int c = -1;
    auto lit = _columns.begin();
    aria::csv::CsvParser parser = aria::csv::CsvParser(sh)
                                      .delimiter(delim[0]) // default is ,
                                      .quote(_quote[0]);   // default is"
    for (auto &row : parser)
      {
        for (auto &col : row)
          {
            ++c;
            std::string col_name;

            // detect strings by looking for characters and for quotes
            // convert to float unless it is string (ignore strings, aka
            // categorical fields, for now)
            if (!_columns.empty()) // in prediction mode, columns from header
                                   // are not mandatory
              {
                if ((hit = _ignored_columns_pos.find(c))
                    != _ignored_columns_pos.end())
                  {
                    continue;
                  }
                col_name = (*lit);
                if (_id_pos == c)
                  {
                    column_id = col;
                  }
              }
            try
              {
                double val = 0.0;
                if (!col.empty())
                  {
                    // one-hot vector encoding as required
                    if (!_columns.empty() && is_category(col_name))
                      {
                        // - look up category
                        std::unordered_map<std::string,
                                           CCategorical>::const_iterator chit
                            = _categoricals.find(col_name);
                        int cnum = (*chit).second.get_cat_num(col);
                        if (cnum < 0)
                          {
                            throw InputConnectorBadParamException(
                                "unknown category " + col + " for variable "
                                + col_name);
                          }

                        // - create one-hot vector
                        int csize = (*chit).second._vals.size();
                        std::vector<double> ohv = one_hot_vector(cnum, csize);
                        vals.insert(vals.end(), ohv.begin(), ohv.end());
                      }
                    else
                      {
                        val = std::stod(col);
                        vals.push_back(val);
                      }
                  }
              }
            catch (std::invalid_argument &e)
              {
                // not a number, skip for now
                if (column_id == col) // if id is string, replace with number /
                  vals.push_back(c);
                else if (std::find(_label_pos.begin(), _label_pos.end(), c)
                         != _label_pos.end())
                  {
                    std::unordered_map<std::string, int>::iterator uit;
                    if ((uit = _hcorresp_r.find(col)) == _hcorresp_r.end())
                      {
                        if (test)
                          {
                            throw InputConnectorBadParamException(
                                "label " + col
                                + " found in test set but not in train set");
                          }
                        int clsn = _hcorresp_r.size();
                        vals.push_back(clsn);
                        _hcorresp_r.insert(
                            std::pair<std::string, int>(col, clsn));
                        _hcorresp.insert(
                            std::pair<int, std::string>(clsn, col));
                      }
                    else
                      {
                        vals.push_back((*uit).second);
                      }
                  }
                else
                  {
                    _logger->error(
                        "line {}: skipping column {} / not a number", nlines,
                        col_name);
                    _logger->error(hline);
                    throw InputConnectorBadParamException(
                        "column " + col_name
                        + " is not a number, use categoricals or ignore "
                          "parameters instead");
                  }
              }
            ++lit;
          }
      }
    ++nlines;
  }

  void CSVInputFileConn::read_header(std::string &hline)
  {
    hline.erase(std::remove(hline.begin(), hline.end(), '\r'),
                hline.end()); // remove ^M if any
    hline.erase(std::remove(hline.begin(), hline.end(), '\n'),
                hline.end()); // remove \n if any (in case of hline coming
                              // direclty from mem, and not parsed by line)
    std::stringstream sg(hline);
    std::string col;

    // read header
    std::unordered_set<std::string>::const_iterator hit;
    std::unordered_map<std::string, int>::const_iterator hit2;
    int i = 0;
    bool has_id = false;
    aria::csv::CsvParser parser = aria::csv::CsvParser(sg)
                                      .delimiter(_delim[0]) // default is ,
                                      .quote(_quote[0]);    // default is"
    for (auto &row : parser)
      {
        for (auto &col : row)
          {
            if ((hit = _ignored_columns.find(col)) != _ignored_columns.end())
              {
                _ignored_columns_pos.insert(i);
                ++i;
                continue;
              }
            else
              _columns.push_back(col);

            if ((hit2 = _label_set.find(col)) != _label_set.end())
              _label_pos.at((*hit2).second) = i;
            if (!has_id && !_id.empty() && col == _id)
              {
                _id_pos = i;
                has_id = true;
              }
            ++i;
          }
        _detect_cols = i;
      }
    for (size_t j = 0; j < _label_pos.size(); j++)
      if (_label_pos.at(j) < 0 && _train)
        throw InputConnectorBadParamException("cannot find label column "
                                              + _label[j]);
    if (!_id.empty() && !has_id)
      throw InputConnectorBadParamException("cannot find id column " + _id);
  }

  void CSVInputFileConn::fillup_categoricals(std::ifstream &csv_file)
  {
    int l = 0;
    std::string hline;
    while (std::getline(csv_file, hline))
      {
        hline.erase(std::remove(hline.begin(), hline.end(), '\r'),
                    hline.end());
        std::vector<double> vals;
        std::string cid;
        std::string col;
        auto hit = _columns.begin();
        std::unordered_set<int>::const_iterator igit;
        std::stringstream sh(hline);
        int cu = 0;
        // while (std::getline(sh, col, _delim[0]))
        aria::csv::CsvParser parser = aria::csv::CsvParser(sh)
                                          .delimiter(_delim[0]) // default is ,
                                          .quote(_quote[0]);    // default is"
        for (auto &row : parser)
          {
            for (auto &col : row)
              {
                if (cu >= _detect_cols)
                  {
                    _logger->error(
                        "line {} has more columns than headers / this "
                        "line: {} / header: {}",
                        l, cu, _detect_cols);
                    _logger->error(hline);
                    throw InputConnectorBadParamException(
                        "line has more columns than headers");
                  }
                if ((igit = _ignored_columns_pos.find(cu))
                    != _ignored_columns_pos.end())
                  {
                    ++cu;
                    continue;
                  }
                update_category((*hit), col);
                ++hit;
                ++cu;
              }
          }
        ++l;
      }
    csv_file.clear();
    csv_file.seekg(0, std::ios::beg);
    std::getline(csv_file, hline); // skip header line
  }

  int CSVInputFileConn::find_mean(std::istream &csv_file)
  {
    int nlines = 0;
    std::string hline;
    unsigned int nvals = 0;
    while (std::getline(csv_file, hline))
      {
        hline.erase(std::remove(hline.begin(), hline.end(), '\r'),
                    hline.end());
        std::vector<double> vals;
        std::string cid;
        read_csv_line(hline, _delim, vals, cid, nlines, false);
        nvals = vals.size();
        if (nlines == 1)
          _mean_vals = vals;
        else
          for (size_t j = 0; j < vals.size(); j++)
            _mean_vals.at(j) += vals.at(j);
      }
    for (size_t j = 0; j < nvals; j++)
      _mean_vals.at(j) /= nlines;

    csv_file.clear();
    csv_file.seekg(0, std::ios::beg);
    std::getline(csv_file, hline); // skip header line

    return nlines;
  }

  int CSVInputFileConn::find_variance(std::istream &csv_file,
                                      const std::vector<double> &mean)
  {
    int nlines = 0;
    std::string hline;
    unsigned int nvals = 0;

    _variance_vals.clear();
    _variance_vals.resize(mean.size(), 0.0);
    while (std::getline(csv_file, hline))
      {
        hline.erase(std::remove(hline.begin(), hline.end(), '\r'),
                    hline.end());
        std::vector<double> vals;
        std::string cid;
        read_csv_line(hline, _delim, vals, cid, nlines, false);

        for (size_t j = 0; j < vals.size(); j++)
          _variance_vals.at(j)
              += (vals.at(j) - mean.at(j)) * (vals.at(j) - mean.at(j));
      }
    for (size_t j = 0; j < nvals; j++)
      {
        _variance_vals.at(j) /= nlines;
      }

    csv_file.clear();
    csv_file.seekg(0, std::ios::beg);
    std::getline(csv_file, hline); // skip header line

    return nlines;
  }

  void CSVInputFileConn::find_min_max()
  {
    if (_min_vals.empty() && _max_vals.empty())
      if (_csvdata.size() > 0)
        _min_vals = _max_vals = _csvdata.at(0)._v;
    for (size_t i = 0; i < _csvdata.size(); ++i)
      for (size_t j = 0; j < _csvdata.at(i)._v.size(); j++)
        {
          _min_vals.at(j) = std::min(_csvdata.at(i)._v[j], _min_vals.at(j));
          _max_vals.at(j) = std::max(_csvdata.at(i)._v[j], _max_vals.at(j));
        }
  }

  void CSVInputFileConn::find_mean()
  {
    if (!_mean_vals.empty())
      return;
    if (_csvdata.size() < 1)
      return;
    _mean_vals = _csvdata.at(0)._v;
    for (size_t i = 1; i < _csvdata.size(); ++i)
      for (size_t j = 0; j < _csvdata.at(i)._v.size(); ++j)
        _mean_vals.at(j) += _csvdata.at(i)._v.at(j);
    for (size_t j = 0; j < _csvdata.at(0)._v.size(); ++j)
      _mean_vals.at(j) /= _csvdata.size();
  }

  void CSVInputFileConn::find_variance()
  {
    _variance_vals.clear();
    _variance_vals.resize(_mean_vals.size(), 0.0);
    for (size_t i = 0; i < _csvdata.size(); ++i)
      for (size_t j = 0; j < _csvdata.at(i)._v.size(); ++j)
        _variance_vals.at(j) += (_csvdata.at(i)._v.at(j) - _mean_vals.at(j))
                                * (_csvdata.at(i)._v.at(j) - _mean_vals.at(j));
    for (size_t j = 0; j < _csvdata.at(0)._v.size(); ++j)
      _variance_vals.at(j) /= _csvdata.size();
  }

  void CSVInputFileConn::find_min_max(std::istream &csv_file)
  {
    int nlines = 0;
    std::string hline;
    while (std::getline(csv_file, hline))
      {
        hline.erase(std::remove(hline.begin(), hline.end(), '\r'),
                    hline.end());
        std::vector<double> vals;
        std::string cid;
        read_csv_line(hline, _delim, vals, cid, nlines, false);
        if (nlines == 1)
          _min_vals = _max_vals = vals;
        else
          {
            for (size_t j = 0; j < vals.size(); j++)
              {
                _min_vals.at(j) = std::min(vals.at(j), _min_vals.at(j));
                _max_vals.at(j) = std::max(vals.at(j), _max_vals.at(j));
              }
          }
      }
    csv_file.clear();
    csv_file.seekg(0, std::ios::beg);
    std::getline(csv_file, hline); // skip header line
  }

  void CSVInputFileConn::read_csv(const std::string &fname,
                                  const bool &forbid_shuffle)
  {
    std::ifstream csv_file(fname, std::ios::binary);
    _logger->info("fname={} / open={}", fname, csv_file.is_open());
    if (!csv_file.is_open())
      throw InputConnectorBadParamException("cannot open file " + fname);
    std::string hline;
    std::getline(csv_file, hline);
    read_header(hline);

    // debug
    /*std::cerr << "found " << _detect_cols << " columns\n";
    std::cerr << "label size=" << _label.size() << " / label_pos size=" <<
    _label_pos.size() << std::endl; std::cout << "CSV columns:\n";
    std::copy(_columns.begin(),_columns.end(),
              std::ostream_iterator<std::string>(std::cout," "));
              std::cout << std::endl;*/
    // debug

    // categorical variables
    if (_train && !_categoricals.empty())
      {
        fillup_categoricals(csv_file);
      }

    // scaling to [0,1]
    int nlines = 0;
    if (_scale && _scale_type == MINMAX
        && (_min_vals.empty() || _max_vals.empty()))
      {
        find_min_max(csv_file);
      }
    if (_scale && _scale_type == ZNORM
        && (_mean_vals.empty() || _variance_vals.empty()))
      {
        find_mean(csv_file);
        find_variance(csv_file, _mean_vals);
      }

    // read data
    while (std::getline(csv_file, hline))
      {
        hline.erase(std::remove(hline.begin(), hline.end(), '\r'),
                    hline.end());
        std::vector<double> vals;
        std::string cid;
        read_csv_line(hline, _delim, vals, cid, nlines, false);
        if (_scale)
          {
            scale_vals(vals);
          }
        if (!_id.empty())
          {
            add_train_csvline(cid, vals);
          }
        else
          add_train_csvline(std::to_string(nlines), vals);

        // debug
        /*std::cout << "csv data line #" << nlines << "= " << vals.size() <<
          std::endl;
          std::copy(vals.begin(),vals.end(),std::ostream_iterator<double>(std::cout,""));
          std::cout << std::endl;*/
        // debug
      }
    _logger->info("read {} lines from {}", nlines, fname);
    csv_file.close();

    // test file, if any.
    if (!_csv_test_fnames.empty())
      {
        unsigned int test_set_id = 0;
        for (std::string csv_test_fname : _csv_test_fnames)
          {
            nlines = 0;
            std::ifstream csv_test_file(csv_test_fname, std::ios::binary);
            if (!csv_test_file.is_open())
              throw InputConnectorBadParamException("cannot open test file "
                                                    + fname);
            std::getline(csv_test_file, hline); // skip header line
            while (std::getline(csv_test_file, hline))
              {
                hline.erase(std::remove(hline.begin(), hline.end(), '\r'),
                            hline.end());
                std::vector<double> vals;
                std::string cid;
                read_csv_line(hline, _delim, vals, cid, nlines, true);
                if (_scale)
                  {
                    scale_vals(vals);
                  }
                if (!_id.empty())
                  add_test_csvline(test_set_id, cid, vals);
                else
                  add_test_csvline(test_set_id, std::to_string(nlines), vals);

                // debug
                /*std::cout << "csv test data line=";
                  std::copy(vals.begin(),vals.end(),std::ostream_iterator<double>(std::cout,"
                  ")); std::cout << std::endl;*/
                // debug
              }
            _logger->info("read {} lines from {}", nlines,
                          _csv_test_fnames[test_set_id]);
            csv_test_file.close();
            test_set_id++;
          }
      }

    // shuffle before possible test data selection.
    if (!forbid_shuffle)
      shuffle_data(_csvdata);

    if (_csv_test_fnames.empty() && _test_split > 0)
      {
        std::vector<CSVline> testdata;
        split_data(_csvdata, testdata);
        _csvdata_tests.push_back(testdata);
        _logger->info("data split test size={} / remaining data size={}",
                      _csvdata_tests[0].size(), _csvdata.size());
      }
    if (!_ignored_columns.empty() || !_categoricals.empty())
      update_columns();

    // write corresp file
    std::ofstream correspf(_model_repo + "/" + _correspname, std::ios::binary);
    auto hit = _hcorresp.begin();
    while (hit != _hcorresp.end())
      {
        correspf << (*hit).first << " " << (*hit).second << std::endl;
        ++hit;
      }
    correspf.close();
  }

  void CSVInputFileConn::serialize_bounds()
  {
    static int boundsprecision = 15;
    std::string boundsfname;
    if (_model_repo.empty())
      boundsfname = _boundsfname;
    else
      boundsfname = _model_repo + "/" + _boundsfname;
    std::string delim = ":";
    std::ofstream out;
    out.open(boundsfname);
    if (!out.is_open())
      throw InputConnectorBadParamException(
          "failed opening for writing bounds file " + boundsfname);

    if (this->_scale_type == MINMAX)
      out << "ncols: " << _min_vals.size() << std::endl;
    else
      out << "ncols: " << _mean_vals.size() << std::endl;
    out << "nlabels: " << _label_pos.size() << std::endl;
    if (_label_pos.size() > 0)
      {
        out << "label_pos: ";
        for (unsigned int i = 0; i < _label_pos.size() - 1; ++i)
          out << " " << _label_pos[i] << " " << delim;
        out << " " << _label_pos[_label_pos.size() - 1] << std::endl;
      }
    if (this->_scale_type == MINMAX)
      {
        if (_min_vals.size() > 0)
          {
            out << "min_vals: ";
            for (unsigned int i = 0; i < _min_vals.size() - 1; ++i)
              out << " " << std::setprecision(boundsprecision) << _min_vals[i]
                  << " " << delim;
            out << " " << std::setprecision(boundsprecision)
                << _min_vals[_min_vals.size() - 1] << std::endl;
          }
        else
          throw InputConnectorInternalException(
              "No min_val to write in bounds file!");

        if (_max_vals.size() > 0)
          {
            out << "max_vals: ";
            for (unsigned int i = 0; i < _max_vals.size() - 1; ++i)
              out << " " << std::setprecision(boundsprecision) << _max_vals[i]
                  << " " << delim;
            out << " " << std::setprecision(boundsprecision)
                << _max_vals[_max_vals.size() - 1] << std::endl;
          }
        else
          throw InputConnectorInternalException(
              "No max_val to write in bounds file!");
      }

    if (this->_scale_type == ZNORM)
      {
        if (_mean_vals.size() > 0)
          {
            out << "mean_vals: ";
            for (unsigned int i = 0; i < _mean_vals.size() - 1; ++i)
              out << " " << std::setprecision(boundsprecision) << _mean_vals[i]
                  << " " << delim;
            out << " " << std::setprecision(boundsprecision)
                << _mean_vals[_mean_vals.size() - 1] << std::endl;
          }
        else
          throw InputConnectorInternalException(
              "No mean_val to write in bounds file!");

        if (_variance_vals.size() > 0)
          {
            out << "variance_vals: ";
            for (unsigned int i = 0; i < _variance_vals.size() - 1; ++i)
              out << " " << std::setprecision(boundsprecision)
                  << _variance_vals[i] << " " << delim;
            out << " " << std::setprecision(boundsprecision)
                << _variance_vals[_variance_vals.size() - 1] << std::endl;
          }
        else
          throw InputConnectorInternalException(
              "No variance_val to write in bounds file!");
      }
    out.close();
    deserialize_bounds(true);
  }

  bool CSVInputFileConn::deserialize_bounds(const bool &force)
  {
    if (!force && _scale_type == MINMAX && !_min_vals.empty()
        && !_max_vals.empty())
      return true;
    if (!force && _scale_type == ZNORM && !_mean_vals.empty()
        && !_variance_vals.empty())
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
        else if (key == "mean_vals")
          {
            _mean_vals.clear();
            for (int i = 0; i < ncols; ++i)
              _mean_vals.push_back(std::atof(tokens.at(i + 1).c_str()));
          }
        else if (key == "variance_vals")
          {
            _variance_vals.clear();
            for (int i = 0; i < ncols; ++i)
              _variance_vals.push_back(std::atof(tokens.at(i + 1).c_str()));
          }
      }
    this->_logger->info("bounds loaded");
    in.close();
    return true;
  }

}
