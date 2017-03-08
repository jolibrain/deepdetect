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

#include "svminputfileconn.h"
#include "utils/utils.hpp"
#include <glog/logging.h>

namespace dd
{

  /*- DDSvm -*/
  int DDSvm::read_file(const std::string &fname)
  {
    if (_cifc)
      {
	_cifc->read_svm(_adconf,fname);
	return 0;
      }
    else return -1;
  }
  
  int DDSvm::read_mem(const std::string &content)
  {
    if (!_cifc)
      return -1;
    std::unordered_map<int,double> vals;
    int label = -1;
    //int nlines = 0;
    _cifc->read_svm_line(content,vals,label);
    /*if (_cifc->_scale)
      {
	if (!_cifc->_train) // in prediction mode, on-the-fly scaling
	  {
	    _cifc->scale_vals(vals);
	  }
	else // in training mode, collect bounds, then scale in another pass over the data
	  {
	    if (_cifc->_min_vals.empty() && _cifc->_max_vals.empty())
	      _cifc->_min_vals = _cifc->_max_vals = vals;
	    for (size_t j=0;j<vals.size();j++)
	      {
		_cifc->_min_vals.at(j) = std::min(vals.at(j),_cifc->_min_vals.at(j));
		_cifc->_max_vals.at(j) = std::max(vals.at(j),_cifc->_max_vals.at(j));
	      }
	  }
	  }*/
    _cifc->add_train_svmline(label,vals,0);
    return 0;
  }

  void SVMInputFileConn::read_svm_line(const std::string &content,
				       std::unordered_map<int,double> &vals,
				       int &label)
  {
    bool fpos = true;
    std::string col;
    std::stringstream sh(content);
    std::unordered_set<int>::const_iterator fit;
    while(std::getline(sh,col,' '))
      {
	try
	  {
	    if (fpos)
	      {
		label = std::stod(col);
		fpos = false;
	      }
	    std::vector<std::string> res = dd_utils::split(col,':');
	    if (res.size() == 2)
	      {
		int fid = std::stoi(res.at(0));
		if ((fit=_fids.find(fid))!=_fids.end())
		  vals.insert(std::pair<int,double>(fid,std::stod(res.at(1))));
	      }
	  }
	catch (std::invalid_argument &e)
	  {
	    throw InputConnectorBadParamException("invalid argument error reading SVM format line: " + content);
	  }
	catch (...)
	  {
	    throw InputConnectorBadParamException("error reading SVM format line: " + content);
	  }
      }
  }
  
  void SVMInputFileConn::read_svm(const APIData &ad,
				  const std::string &fname)
  {
    std::ifstream svm_file(fname,std::ios::binary);
    LOG(INFO) << "SVM fname=" << fname << " / open=" << svm_file.is_open() << std::endl;
    if (!svm_file.is_open())
      throw InputConnectorBadParamException("cannot open file " + fname);
    std::string hline;

    // first pass to get max index
    std::string col;
    while(std::getline(svm_file,hline))
      {
	bool fpos = true;
      	std::stringstream sh(hline);
	while(std::getline(sh,col,' '))
	  {
	    if (fpos)
	      {
		fpos = false;
		continue;
	      }
	    std::vector<std::string> res = dd_utils::split(col,':');
	    if (_train && res.size() == 2)
	      {
		int fid = std::stoi(res.at(0));
		if (fid > _max_id)
		  _max_id = fid;
		_fids.insert(fid);
	      }
	  }
      }
    svm_file.clear();
    svm_file.seekg(0,std::ios::beg);

    LOG(INFO) << "total number of dimensions=" << _fids.size() << std::endl;

    // read data
    int nlines = 0;
    while(std::getline(svm_file,hline))
      {
	std::unordered_map<int,double> vals;
	int label;
	read_svm_line(hline,vals,label);
	add_train_svmline(label,vals,nlines);
	++nlines;
      }
      svm_file.close();
      LOG(INFO) << "read " << nlines << " lines from SVM data file";

      if (!_svm_test_fname.empty())
	{
	  int tnlines = 0;
	  std::ifstream svm_test_file(_svm_test_fname,std::ios::binary);
	  if (!svm_test_file.is_open())
	    throw InputConnectorBadParamException("cannot open SVM test file " + fname);
	  while(std::getline(svm_test_file,hline))
	    {
	      hline.erase(std::remove(hline.begin(),hline.end(),'\r'),hline.end());
	      std::unordered_map<int,double> vals;
	      int label;
	      read_svm_line(hline,vals,label);
	      add_test_svmline(label,vals,tnlines);
	      ++tnlines;
	    }
	  svm_test_file.close();
	}
      
      // shuffle before test selection, if any
      shuffle_data(ad);
      
      if (_svm_test_fname.empty() && _test_split > 0)
	{
	  split_data();
	  LOG(INFO) << "data split test size=" << _svmdata_test.size() << " / remaining data size=" << _svmdata.size() << std::endl;
	}
  }

  void SVMInputFileConn::serialize_vocab()
  {
    std::string vocabfname = _model_repo + "/" + _vocabfname;
    std::string delim=",";
    std::ofstream out;
    out.open(vocabfname);
    if (!out.is_open())
      throw InputConnectorBadParamException("failed opening SVM vocabulary file " + vocabfname);
    auto fit = _fids.begin();
    while(fit!=_fids.end())
      {
	out << (*fit) << std::endl;
	++fit;
      }
    out.close();
  }

  void SVMInputFileConn::deserialize_vocab(const bool &required)
  {
    std::string vocabfname = _model_repo + "/" + _vocabfname;
    if (!fileops::file_exists(vocabfname))
      {
	if (required)
	  throw InputConnectorBadParamException("cannot find vocabulary file " + vocabfname);
	else return;
      }
    std::ifstream in;
    in.open(vocabfname);
    if (!in.is_open())
      throw InputConnectorBadParamException("failed opening vocabulary file " + vocabfname);
    std::string line;
    while(getline(in,line))
      {
	std::string cline = line.substr(0,line.size());
	int fid = std::atoi(cline.c_str());
	if (fid > _max_id)
	  _max_id = fid;
	_fids.insert(fid);
      }
    in.close();
    LOG(INFO) << "loaded SVM vocabulary of size=" << _fids.size() << std::endl;
  }
  
}
