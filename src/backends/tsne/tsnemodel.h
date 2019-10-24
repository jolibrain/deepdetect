/**
 * DeepDetect
 * Copyright (c) 2017 Emmanuel Benazera
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

#ifndef TSNEMODEL_H
#define TSNEMODEL_H

#include "mlmodel.h"
#include "apidata.h"
#include <string>
#include <unordered_map>

namespace dd
{
  class TSNEModel : public MLModel
  {
  public:
    TSNEModel():MLModel() {}
  TSNEModel(const APIData &ad,APIData &adg,
	    const std::shared_ptr<spdlog::logger> &logger)
    :MLModel(ad,adg,logger)
      {
	if (ad.has("repository"))
	  this->_repo = ad.get("repository").get<std::string>();
	read_from_repository();	
      }
    TSNEModel(const std::string &repo)
      :MLModel(repo) {}
    ~TSNEModel() {}

    //TODO: load and save
    int read_from_repository() { return 0; };
  };
}

#endif
