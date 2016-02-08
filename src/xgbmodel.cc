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

  XGBModel::XGBModel::XGBModel(const APIData &ad)
    :MLModel()
  {
    if (ad.has("repository"))
      this->_repo = ad.get("repository").get<std::string>();
  }

  int XGBModel::read_from_repository()
  {
    static std::string weights = ".model";
    std::unordered_set<std::string> lfiles;
    int e = fileops::list_directory(_repo,true,false,lfiles);
    if (e != 0)
      {
	LOG(ERROR) << "error reading or listing XGBoost models in repository " << _repo << std::endl;
	return 1;
      }
    std::string weightsf;
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
	++hit;
      }
    _weights = weightsf;
    return 0;
  }
  
}
