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

#include "txtinputfileconn.h"
#include "utils/fileops.hpp"
#include "utils/utils.hpp"
#include <boost/tokenizer.hpp>
#include <glog/logging.h>
#include <iostream>

namespace dd
{

  /*- DDTxt -*/
  int DDTxt::read_file(const std::string &fname)
  {
    if (!_ctfc)
      return -1;
    std::ifstream txt_file(fname);
    if (!txt_file.is_open())
      throw InputConnectorBadParamException("cannot open file " + fname);
    std::stringstream buffer;
    buffer << txt_file.rdbuf();
    _ctfc->parse_content(buffer.str());
    return 0;
  }

  int DDTxt::read_mem(const std::string &content)
  {
    if (!_ctfc)
      return -1;
    _ctfc->parse_content(content); // no target
    return 0;
  }

  int DDTxt::read_dir(const std::string &dir)
  {
    if (!_ctfc)
      return -1;

    // list directories in dir
    std::unordered_set<std::string> subdirs;
    if (fileops::list_directory(dir,false,true,subdirs))
      throw InputConnectorBadParamException("failed reading text subdirectories in data directory " + dir);
    std::cerr << "list subdirs size=" << subdirs.size() << std::endl;
    
    // list files and classes
    int cl = 0;
    std::unordered_map<int,std::string> hcorresp; // correspondence class number / class name
    std::vector<std::pair<std::string,int>> lfiles; // labeled files
    auto uit = subdirs.begin();
    while(uit!=subdirs.end())
      {
	std::unordered_set<std::string> subdir_files;
	if (fileops::list_directory((*uit),true,false,subdir_files))
	  throw InputConnectorBadParamException("failed reading image data sub-directory " + (*uit));
	hcorresp.insert(std::pair<int,std::string>(cl,dd_utils::split((*uit),'/').back()));
	auto fit = subdir_files.begin();
	while(fit!=subdir_files.end()) // XXX: re-iterating the file is not optimal
	  {
	    lfiles.push_back(std::pair<std::string,int>((*fit),cl));
	    ++fit;
	  }
	++cl;
	++uit;
      }

    // shuffle files if requested
    if (_ctfc->_shuffle)
      {
	std::random_device rd;
	std::mt19937 g(rd());
	std::shuffle(lfiles.begin(),lfiles.end(),g);
      }

    // parse content
    // XXX: parallelize with openmp -> requires thread safe parse_content
    for (std::pair<std::string,int> &p: lfiles)
      {
	std::ifstream txt_file(p.first);
	if (!txt_file.is_open())
	  throw InputConnectorBadParamException("cannot open file " + p.first);
	std::stringstream buffer;
	buffer << txt_file.rdbuf();
	std::string ct = buffer.str();
	std::transform(ct.begin(),ct.end(),ct.begin(),::tolower);
	_ctfc->parse_content(ct,p.second);
      }

    // post-processing
    if (_ctfc->_tfidf)
      {
	//std::unordered_map<std::string,Word>::const_iterator vhit;
	//std::unordered_map<int,std::string>::const_iterator rvhit;
	for (TxtBowEntry &tbe: _ctfc->_txt)
	  {
	    auto hit = tbe._v.begin();
	    while(hit!=tbe._v.end())
	      {
		std::string ws = _ctfc->_rvocab[(*hit).first];
		Word w = _ctfc->_vocab[ws];
		//std::cerr << "txt size=" << _ctfc->_txt.size() << " / total_classes=" << w._total_classes << std::endl;
		(*hit).second = ((*hit).second / static_cast<double>(w._total_count)) * std::log(_ctfc->_txt.size() / static_cast<double>(w._total_classes));
		//std::cerr << "tfidf feature w=" << ws << " / val=" << (*hit).second << std::endl;
		++hit;
	      }
	  }
      }

    LOG(INFO) << "vocabulary size=" << _ctfc->_vocab.size() << std::endl;
    //_ctfc->_vocab.clear(); //TODO: serialize to disk / db
    
    return 0;
  }
  
  /*- TxtInputFileConn -*/
  void TxtInputFileConn::parse_content(const std::string &content,
				       const int &target)
  {
    // Coming up:
    // - sentence separator
    // - non bow parsing
    
    TxtBowEntry tbe(target);
    std::unordered_map<std::string,Word>::iterator vhit;
    boost::char_separator<char> sep("\n\t\f\r ,.;:`'!?)(-|><^·&\"\\/{}#$–");
    boost::tokenizer<boost::char_separator<char>> tokens(content,sep);
    for (std::string w : tokens)
      {
	//std::cout << w << std::endl;

	// check and fillup vocab.
	int pos = -1;
	if ((vhit=_vocab.find(w))==_vocab.end())
	  {
	    pos = _vocab.size();
	    _vocab.emplace(std::make_pair(w,Word(pos)));
	    if (_tfidf)
	      _rvocab.insert(std::pair<int,std::string>(pos,w));
	  }
	else
	  {
	    pos = (*vhit).second._pos;
	    (*vhit).second._total_count++;
	    if (!tbe.has_word(pos))
	      (*vhit).second._total_classes++;
	  }
	tbe.add_word(pos,1.0,_count);
      }
    _txt.push_back(tbe);
  }
  
}
