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

#include "tfmodel.h"
#include <utils/fileops.hpp>
#include <glog/logging.h>
#include <string>
namespace dd
{
  TFModel::TFModel(const APIData &ad)
  {
   if (ad.has("repository"))
      {
	read_from_repository(ad.get("repository").get<std::string>(),spdlog::get("api")); // XXX: beware, error not caught
	this-> _modelRepo = ad.get("repository").get<std::string>();
      }
   read_corresp_file(spdlog::get("api"));
  }

  int TFModel::read_from_repository(const std::string &repo,
				    const std::shared_ptr<spdlog::logger> &logger)
  {
    std::string graphName = ".pb";
    std::string labelFile = ".txt";
    this->_repo = repo;
    std::unordered_set<std::string> lfiles;
  	int e = fileops::list_directory(repo,true,false,lfiles);
    if (e != 0)
      {
	logger->error("error reading or listing caffe models in repository {}",repo);
	return 1;
      }
      
  	auto hit = lfiles.begin();
  	std::string graphf,labelf;
  	long int state_t=-1;
  	while(hit!=lfiles.end())
      {
	if ((*hit).find(graphName)!=std::string::npos)
	  {
	    // stat file to pick the latest one
	    long int st = fileops::file_last_modif((*hit));
	    if (st > state_t)
	      {
			graphf = (*hit);
			state_t = st;
	      }
	  }
	else if ((*hit).find(labelFile)!=std::string::npos)
	  labelf = (*hit);

	++hit;
	}
	_graphName = graphf;
	_corresp = labelf;

	return 0;
  }

  int TFModel::read_corresp_file(const std::shared_ptr<spdlog::logger> &logger)
  {
    if (!_corresp.empty())
      {
	std::ifstream ff(_corresp);
	if (!ff.is_open())
	  logger->error("cannot open TF model corresp file={}",_corresp);
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
    return 0;
  }
}
