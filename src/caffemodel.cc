/**
 * DeepDetect
 * Copyright (c) 2014 Emmanuel Benazera
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

#include "caffemodel.h"
#include "utils/fileops.hpp"
#include <glog/logging.h>
#include <fstream>
#include <iostream>

namespace dd
{
  CaffeModel::CaffeModel(const APIData &ad)
    :MLModel()
  {
    if (ad.has("repository"))
      {
	read_from_repository(ad.get("repository").get<std::string>());
      }
    else
      {
	_def = ad.get("def").get<std::string>();
	_weights = ad.get("weights").get<std::string>();
	_corresp = ad.get("corresp").get<std::string>();
	_solver = ad.get("solver").get<std::string>();
      }
    
    if (!_corresp.empty()) //TODO: test for supervised.
      {
	std::ifstream ff(_corresp);
	if (!ff.is_open())
	  LOG(ERROR) << "cannot open Caffe model corresp file=" << _corresp << std::endl;
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
    else // TODO: training, requires solver.
      {
      }
  }
  
  int CaffeModel::read_from_repository(const std::string &repo)
  {
    static std::string deploy = "deploy.prototxt";
    static std::string weights = ".caffemodel";
    static std::string corresp = "corresp.txt";
    static std::string solver = "solver";
    _repo = repo;
    std::unordered_set<std::string> lfiles;
    int e = list_directory_files(repo,lfiles);
    if (e != 0)
      {
	LOG(ERROR) << "error reading caffe model repository\n";
      }
    std::string deployf,weightsf,correspf,solverf;
    auto hit = lfiles.begin();
    while(hit!=lfiles.end())
      {
	if ((*hit).find(deploy)!=std::string::npos)
	  deployf = (*hit);
	else if ((*hit).find(weights)!=std::string::npos)
	  weightsf = (*hit);
	else if ((*hit).find(corresp)!=std::string::npos)
	  correspf = (*hit);
	else if ((*hit).find(solver)!=std::string::npos)
	  solverf = (*hit);
	++hit;
      }
    _def = deployf;
    _weights = weightsf;
    _corresp = correspf;
    _solver = solverf;
    
    /*    if (deployf.empty() || weightsf.empty())
      {
	LOG(ERROR) << "missing caffe model file(s) in repository\n";
	return CaffeModel();
	}*/
    return 0;
  }
}
