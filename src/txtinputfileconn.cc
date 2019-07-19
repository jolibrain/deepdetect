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
	_logger->error("cannot open file={}",fname);
	throw InputConnectorBadParamException("cannot open file " + fname);
      }
    std::stringstream buffer;
    buffer << txt_file.rdbuf();
    _ctfc->parse_content(buffer.str());
    return 0;
  }

  int DDTxt::read_db(const std::string &fname)
    {
      _ctfc->_db_fname = fname;
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
    if (fileops::list_directory(dir,false,true,false,subdirs))
      throw InputConnectorBadParamException("failed reading text subdirectories in data directory " + dir);
    _logger->info("txtinputfileconn: list subdirs size={}",subdirs.size());
    
    // list files and classes
    bool test_dir = false;
    std::vector<std::pair<std::string,int>> lfiles; // labeled files
    std::unordered_map<int,std::string> hcorresp; // correspondence class number / class name
    std::unordered_map<std::string,int> hcorresp_r; // reverse correspondence for test set.
    if (!subdirs.empty())
      {
	++_ctfc->_dirs; // this is a directory containing classes info.
	if (_ctfc->_dirs >= 2)
	  {
	    test_dir = true;
	    std::ifstream correspf(_ctfc->_model_repo + "/" + _ctfc->_correspname,std::ios::binary);
	    if (!correspf.is_open())
	      {
		std::string err_msg = "Failed opening corresp file before reading txt test data directory";;
		_logger->error(err_msg);
		throw InputConnectorInternalException(err_msg);
	      }
	    std::string line;
	    while(std::getline(correspf,line))
	      {
		std::vector<std::string> vstr = dd_utils::split(line,' ');
		hcorresp_r.insert(std::pair<std::string,int>(vstr.at(1),std::atoi(vstr.at(0).c_str())));
	      }
	  }
	int cl = 0;
	auto uit = subdirs.begin();
	while(uit!=subdirs.end())
	  {
	    std::unordered_set<std::string> subdir_files;
	    if (fileops::list_directory((*uit),true,false,true,subdir_files))
	      throw InputConnectorBadParamException("failed reading text data sub-directory " + (*uit));
	    std::string cls = dd_utils::split((*uit),'/').back();
	    if (!test_dir)
	      {
		if (_ctfc->_train)
		  {
		    hcorresp.insert(std::pair<int,std::string>(cl,cls));
		    hcorresp_r.insert(std::pair<std::string,int>(cls,cl));
		  }
	      }
	    else
	      {
		std::unordered_map<std::string,int>::const_iterator hcit;
		if ((hcit=hcorresp_r.find(cls))==hcorresp_r.end())
		  {
		    _logger->error("class {} appears in testing set but not in training set, skipping",cls);
		    ++uit;
		    continue;
		  }
		cl = (*hcit).second;
	      }
	    auto fit = subdir_files.begin();
	    while(fit!=subdir_files.end()) // XXX: re-iterating the file is not optimal
	      {
		lfiles.push_back(std::pair<std::string,int>((*fit),cl));
		++fit;
	      }
	    if (!test_dir)
	      ++cl;
	    ++uit;
	  }
      }
    else
      {
	std::unordered_set<std::string> test_files;
	fileops::list_directory(dir,true,false,false,test_files);
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
	_ctfc->parse_content(ct,p.second,test_dir);
      }

    // post-processing
    size_t initial_vocab_size = _ctfc->_vocab.size();
    if (_ctfc->_train && !test_dir)
      {
	auto vhit = _ctfc->_vocab.begin();
	while(vhit!=_ctfc->_vocab.end())
	  {
	    if ((*vhit).second._total_count < _ctfc->_min_count)
	      vhit = _ctfc->_vocab.erase(vhit);
	    else ++vhit;
	  }
      }
    if (_ctfc->_train && !test_dir && initial_vocab_size != _ctfc->_vocab.size())
      {
	// update pos
	int pos = 0;
	auto vhit = _ctfc->_vocab.begin();
	while(vhit!=_ctfc->_vocab.end())
	  {
	    (*vhit).second._pos = pos;
	    ++pos;
	    ++vhit;
	  }
      }

    if (!_ctfc->_characters && !test_dir && (initial_vocab_size != _ctfc->_vocab.size() || _ctfc->_tfidf))
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
    if (_ctfc->_train && !test_dir)
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

    _logger->info("vocabulary size={}",_ctfc->_vocab.size());
    
    return 0;
  }
  
  /*- TxtInputFileConn -*/
  void TxtInputFileConn::parse_content(const std::string &content,
				       const float &target,
				       const bool &test)
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
	    std::unordered_map<std::string,Word>::iterator vhit;
	    boost::char_separator<char> sep("\n\t\f\r ,.;:`'!?)(-|><^·&\"\\/{}#$–=+");
	    boost::tokenizer<boost::char_separator<char>> tokens(ct,sep);

            if (_ordered_words) 
              {
                TxtOrderedWordsEntry *towe = new TxtOrderedWordsEntry(target);
                std::unordered_map<std::string,Word>::iterator vhit;

                for (std::string w : tokens)
                  {
                    towe->add_word(w, 0);
                  }
                
                if (!test)
                  _txt.push_back(towe);
                else _test_txt.push_back(towe);
              }
            else
              {
                TxtBowEntry *tbe = new TxtBowEntry(target);
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
                if (!test)
                  _txt.push_back(tbe);
                else _test_txt.push_back(tbe);
              }
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
		      _logger->error("Invalid UTF-8 character in {}",w);
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
	    if (!test)
	      _txt.push_back(tce);
	    else _test_txt.push_back(tce);
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
    out.close();
  }

  void TxtInputFileConn::deserialize_vocab(const bool &required)
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
	std::vector<std::string> tokens = dd_utils::split(line,',');
	std::string key = tokens.at(0);
	int pos = std::atoi(tokens.at(1).c_str());
	_vocab.emplace(std::make_pair(key,Word(pos)));
      }
    _logger->info("loaded vocabulary of size={}",_vocab.size());
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
