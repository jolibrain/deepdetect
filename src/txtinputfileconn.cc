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
      {
	return -1;
      }
    std::ifstream txt_file(fname);
    if (!txt_file.is_open())
      {
	std::cerr << "cannot open file\n";
	throw InputConnectorBadParamException("cannot open file " + fname);
      }
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
    std::vector<std::pair<std::string,int>> lfiles; // labeled files
    std::unordered_map<int,std::string> hcorresp; // correspondence class number / class name
    if (_ctfc->_train)
      {
	int cl = 0;
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
      }
    else
      {
	std::unordered_set<std::string> test_files;
	fileops::list_directory(dir,true,false,test_files);
	auto fit = test_files.begin();
	while(fit!=test_files.end())
	  {
	    lfiles.push_back(std::pair<std::string,int>((*fit),0)); // 0 but no class really
	    ++fit;
	  }
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
	_ctfc->parse_content(ct,p.second);
      }

    // post-processing
    size_t initial_vocab_size = _ctfc->_vocab.size();
    auto vhit = _ctfc->_vocab.begin();
    while(vhit!=_ctfc->_vocab.end())
      {
	if ((*vhit).second._total_count < _ctfc->_min_count)
	  vhit = _ctfc->_vocab.erase(vhit);
	else ++vhit;
      }
    if (initial_vocab_size != _ctfc->_vocab.size())
      {
	// update pos
	int pos = 0;
	vhit = _ctfc->_vocab.begin();
	while(vhit!=_ctfc->_vocab.end())
	  {
	    (*vhit).second._pos = pos;
	    ++pos;
	    ++vhit;
	  }
      }

    if (!_ctfc->_characters && (initial_vocab_size != _ctfc->_vocab.size() || _ctfc->_tfidf))
      {
	// clearing up the corpus + tfidf
	std::unordered_map<std::string,Word>::iterator whit;
	for (TxtEntry<double> *te: _ctfc->_txt)
	  {
	    TxtBowEntry *tbe = static_cast<TxtBowEntry*>(te);
	    auto hit = tbe->_v.begin();
	    while(hit!=tbe->_v.end())
	      {
		if ((whit=_ctfc->_vocab.find((*hit).first))!=_ctfc->_vocab.end())
		  {
		    if (_ctfc->_tfidf)
		      {
			Word w = (*whit).second;
			(*hit).second = (std::log(1.0+(*hit).second / static_cast<double>(w._total_count))) * std::log(_ctfc->_txt.size() / static_cast<double>(w._total_docs) + 1.0);
			//std::cerr << "tfidf feature w=" << (*hit).first << " / val=" << (*hit).second << std::endl;
		      }
		    ++hit;
		  }
		else 
		  {
		    //std::cerr << "removing ws=" << (*hit).first << std::endl;
		    hit = tbe->_v.erase(hit);
		  }
	      }
	  }
      }

    // write corresp file
    if (_ctfc->_train)
      {
	std::ofstream correspf(_ctfc->_model_repo + "/" + _ctfc->_correspname,std::ios::binary);
	auto hit = hcorresp.begin();
	while(hit!=hcorresp.end())
	  {
	    correspf << (*hit).first << " " << (*hit).second << std::endl;
	    ++hit;
	  }
	correspf.close();
      }    

    LOG(INFO) << "vocabulary size=" << _ctfc->_vocab.size() << std::endl;
        
    return 0;
  }
  
  /*- TxtInputFileConn -*/
  void TxtInputFileConn::parse_content(const std::string &content,
				       const float &target)
  {
    if (!_train && content.empty())
      throw InputConnectorBadParamException("no text data found");
    std::vector<std::string> cts;
    if (_sentences)
      {
	boost::char_separator<char> sep("\n");
	boost::tokenizer<boost::char_separator<char>> tokens(content,sep);
	for (std::string s: tokens)
	  cts.push_back(s);
      }
    else
      {
	cts.push_back(content);
      }
    for (std::string ct: cts)
      {
	std::transform(ct.begin(),ct.end(),ct.begin(),::tolower);
	if (!_characters)
	  {
	    TxtBowEntry *tbe = new TxtBowEntry(target);
	    std::unordered_map<std::string,Word>::iterator vhit;
	    boost::char_separator<char> sep("\n\t\f\r ,.;:`'!?)(-|><^·&\"\\/{}#$–=+");
	    boost::tokenizer<boost::char_separator<char>> tokens(ct,sep);
	    for (std::string w : tokens)
	      {
		if (static_cast<int>(w.length()) < _min_word_length)
		  continue;
		
		// check and fillup vocab.
		int pos = -1;
		if ((vhit=_vocab.find(w))==_vocab.end())
		  {
		    if (_train)
		      {
			pos = _vocab.size();
			_vocab.emplace(std::make_pair(w,Word(pos)));
		      }
		  }
		else
		  {
		    if (_train)
		      {
			(*vhit).second._total_count++;
			if (!tbe->has_word(w))
			  (*vhit).second._total_docs++;
		      }
		  }
		tbe->add_word(w,1.0,_count);
	      }
	    _txt.push_back(tbe);
	  }
	else // character-level features
	  {
	    if (_seq_forward)
	      std::reverse(ct.begin(),ct.end());
	    TxtCharEntry *tce = new TxtCharEntry(target);
	    std::unordered_map<uint32_t,int>::const_iterator whit;
	    boost::char_separator<char> sep("\n\t\f\r");
	    boost::tokenizer<boost::char_separator<char>> tokens(ct,sep);
	    int seq = 0;
	    bool prev_space = false;
	    for (std::string w: tokens)
	      {
		char *str = (char*)w.c_str();
		char *str_i = str;
		char *end = str+strlen(str)+1;
		do
		{
		  uint32_t c = 0;
		  try
		    {
		      c = utf8::next(str_i,end);
		    }
		  catch(...)
		    {
		      LOG(ERROR) << "Invalid UTF-8 character in " << w << std::endl;
		      c = 0;
		      ++str_i;
		    }
		  if (c == 0)
		    continue;
		  if ((whit=_alphabet.find(c))==_alphabet.end())
		    {
		      if (!prev_space)
			{
			  tce->add_char(' ');
			  seq++;
			  prev_space = true;
			}
		    }
		  else 
		    {
		      tce->add_char(c);
		      seq++;
		      prev_space = false;
		    }
		}
		while(str_i<end && seq < _sequence);
	      }
	    _txt.push_back(tce);
	    std::cerr << "\rloaded text samples=" << _txt.size();
	  }

      }
  }

  void TxtInputFileConn::serialize_vocab()
  {
    std::string vocabfname = _model_repo + "/" + _vocabfname;
    std::string delim=",";
    std::ofstream out;
    out.open(vocabfname);
    if (!out.is_open())
      throw InputConnectorBadParamException("failed opening vocabulary file " + vocabfname);
    for (auto const &p: _vocab)
      {
	out << p.first << delim << p.second._pos << std::endl;
      }
  }

  void TxtInputFileConn::deserialize_vocab()
  {
    std::string vocabfname = _model_repo + "/" + _vocabfname;
    if (!fileops::file_exists(vocabfname))
      throw InputConnectorBadParamException("cannot find vocabulary file " + vocabfname);
    std::ifstream in;
    in.open(vocabfname);
    if (!in.is_open())
      throw InputConnectorBadParamException("failed opening vocabulary file " + vocabfname);
    std::string line;
    while(getline(in,line))
      {
	std::vector<std::string> tokens = dd_utils::split(line,',');
	std::string key = tokens.at(0);
	int pos = atoi(tokens.at(1).c_str());
	_vocab.emplace(std::make_pair(key,Word(pos)));
      }
    std::cerr << "loaded vocabulary of size=" << _vocab.size() << std::endl;
  }

  void TxtInputFileConn::build_alphabet()
  {
    _alphabet.clear();
    auto hit = _alphabet.begin();
    int pos = 0;
    char *str = (char*)_alphabet_str.c_str();
    char *str_i = str;
    char *end = str+strlen(str)+1;
    do
      {
	uint32_t c = utf8::next(str_i,end);
	if ((hit=_alphabet.find(c))==_alphabet.end())
	  {
	    _alphabet.insert(std::pair<uint32_t,int>(c,pos));
	    ++pos;
	  }
      }
    while(str_i<end);
  }

  void TxtInputFileConn::destroy_txt_entries(std::vector<TxtEntry<double>*> &v)
  {
    for (auto e: v)
      delete e;
    v.clear();
  }
  
}
