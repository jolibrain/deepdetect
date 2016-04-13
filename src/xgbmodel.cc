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

#include "xgbmodel.h"
#include <utils/fileops.hpp>
#include <glog/logging.h>

namespace dd
{

  XGBModel::XGBModel(const APIData &ad)
    :MLModel()
  {
    if (ad.has("repository"))
      this->_repo = ad.get("repository").get<std::string>();
    read_from_repository();
    read_corresp_file();
  }

  int XGBModel::read_from_repository()
  {
    static std::string weights = ".model";
    static std::string corresp = "corresp";
    std::unordered_set<std::string> lfiles;
    int e = fileops::list_directory(_repo,true,false,lfiles);
    if (e != 0)
      {
	LOG(ERROR) << "error reading or listing XGBoost models in repository " << _repo << std::endl;
	return 1;
      }
    std::string weightsf,correspf;
    int weight_t=-1;
    auto hit = lfiles.begin();
    while(hit!=lfiles.end())
      {
	if ((*hit).find(weights)!=std::string::npos)
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
	++hit;
      }
    _weights = weightsf;
    _corresp = correspf;
    return 0;
  }

  std::string XGBModel::lookup_objective(const std::string &modelfile)
  {
    static std::string objective_softprob = "multi:softprob";
    static std::string objective_binary = "binary:logistic";
    std::ifstream ff(modelfile);
    if (!ff.is_open())
      {
	LOG(ERROR) << "cannot open xgb model file " << modelfile << " for looking objective up";
	return "";
      }
    std::string line;
    while(!ff.eof())
      {
	std::getline(ff,line);
	if (line.find(objective_softprob,0)!=std::string::npos)
	  return objective_softprob;
	else if (line.find(objective_binary,0)!=std::string::npos)
	  return objective_binary;
      }
    return "";
  }
  
}
