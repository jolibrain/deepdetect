/**
 * DeepDetect
 * Copyright (c) 2014-2015 Emmanuel Benazera
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

#ifndef MODEL_H
#define MODEL_H

#include "simsearch.h"
#include <string>
#include <fstream>
#include <iostream>
#include <unordered_map>

namespace dd
{
  class MLModel
  {
  public:
    MLModel() {};
    MLModel(const std::string &repo)
      :_repo(repo) {
    }
    ~MLModel() {
      delete _se;
    }

    void read_corresp_file()
    {
      if (!_corresp.empty())
      {
	std::ifstream ff(_corresp);
	if (!ff.is_open())
	  std::cerr << "cannot open model corresp file=" << _corresp << std::endl;
	else{
	  std::string line;
	  while(!ff.eof())
	    {
	      std::getline(ff,line);
	      std::string key = line.substr(0,line.find(' '));
	      if (!key.empty())
		{
		  std::string value = line.substr(line.find(' ')+1);
		  _hcorresp.insert(std::pair<int,std::string>(std::stoi(key),value));
		}
	    }
	}
      }
    }

    inline std::string get_hcorresp(const int &i)
      {
	if (_hcorresp.empty())
	  return std::to_string(i);
	else return _hcorresp[i];
      }

    /**
     * \brief create similarity search engine
     */
    void create_sim_search(const int &dim)
    {
      if (!_se)
	{
	  _se = new SearchEngine<AnnoySE>(dim,_repo);
	  _se->create_index();
	}
    }

    /**
     * \brief create similarity search index
     */
    void create_index()
    {
      if (_se)
	_se->create_index();
    }

    /**
     * \brief build similarity search index
     */
    void build_index()
    {
      if (_se)
	_se->update_index();
    }
    
    /**
     * \brief remove similarity search index
     */
    void remove_index()
    {
      if (_se)
	_se->remove_index();
    }
    
    std::string _repo; /**< model repository. */
    std::string _mlmodel_template_repo = "templates/";
    std::unordered_map<int,std::string> _hcorresp; /**< table of class correspondences. */
    std::string _corresp; /**< file name of the class correspondences (e.g. house / 23) */

    SearchEngine<AnnoySE> *_se = nullptr;
  };
}

#endif
