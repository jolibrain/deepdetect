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
#include "mllibstrategy.h"
#include "utils/fileops.hpp"
#include <glog/logging.h>
#include <exception>
#include <fstream>
#include <iostream>

namespace dd
{
  CaffeModel::CaffeModel(const APIData &ad)
    :MLModel()
  {
    if (ad.has("templates"))
      this->_mlmodel_template_repo = ad.get("templates").get<std::string>();
    else this->_mlmodel_template_repo += "caffe/"; // default
    if (ad.has("repository"))
      {
       	if (read_from_repository(ad.get("repository").get<std::string>()))
	  throw MLLibBadParamException("error reading or listing Caffe models in repository " + _repo);
      }
    else
      {
	_def = ad.get("def").get<std::string>();
	_trainf = ad.get("trainf").get<std::string>();
	_weights = ad.get("weights").get<std::string>();
	_corresp = ad.get("corresp").get<std::string>();
	_solver = ad.get("solver").get<std::string>();
      }

    read_corresp_file();
  }
  
  int CaffeModel::read_from_repository(const std::string &repo)
  {
    static std::string deploy = "deploy.prototxt";
    static std::string train = ".prototxt";
    static std::string weights = ".caffemodel";
    static std::string sstate = ".solverstate";
    static std::string corresp = "corresp";
    static std::string solver = "_solver.prototxt";
    static std::string meanf = "mean.binaryproto";
    this->_repo = repo;
    std::unordered_set<std::string> lfiles;
    int e = fileops::list_directory(repo,true,false,lfiles);
    if (e != 0)
      {
	LOG(ERROR) << "error reading or listing caffe models in repository " << repo << std::endl;
	return 1;
      }
    std::string deployf,trainf,weightsf,correspf,solverf,sstatef;
    long int state_t=-1, weight_t=-1;
    auto hit = lfiles.begin();
    while(hit!=lfiles.end())
      {
	if ((*hit) == meanf)
	  {
	    _has_mean_file = true;
	  }
	else if ((*hit).find(sstate)!=std::string::npos)
	  {
	    // stat file to pick the latest one
	    long int st = fileops::file_last_modif((*hit));
	    if (st > state_t)
	      {
		sstatef = (*hit);
		state_t = st;
	      }
	  }
	else if ((*hit).find(weights)!=std::string::npos)
	  {
	    // stat file to pick the latest one
	    long int wt = fileops::file_last_modif((*hit));
	    if (wt > weight_t)
	      {
		weightsf = (*hit);
		weight_t = wt;
	      }
	  }
	else if ((*hit).find(corresp)!=std::string::npos)
	  correspf = (*hit);
	else if ((*hit).find("~")!=std::string::npos 
	    || (*hit).find(".prototxt")==std::string::npos)
	  {
	    ++hit;
	    continue;
	  }
	else if ((*hit).find(deploy)!=std::string::npos)
	  deployf = (*hit);
	else if ((*hit).find(solver)!=std::string::npos)
	  solverf = (*hit);
	else if ((*hit).find(train)!=std::string::npos)
	  trainf = (*hit);
	++hit;
      }
    _def = deployf;
    _trainf = trainf;
    _weights = weightsf;
    _corresp = correspf;
    _solver = solverf;
    _sstate = sstatef;
    
    /*    if (deployf.empty() || weightsf.empty())
      {
	LOG(ERROR) << "missing caffe model file(s) in repository\n";
	return CaffeModel();
	}*/
    return 0;
  }
}
