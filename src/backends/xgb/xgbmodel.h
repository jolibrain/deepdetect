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

#ifndef XGBMODEL_H
#define XGBMODEL_H

#include "mlmodel.h"
#include "apidata.h"
#include <spdlog/spdlog.h>
#include <string>
#include <unordered_map>

namespace dd
{
  class XGBModel : public MLModel
  {
  public:
    XGBModel():MLModel() {}
    XGBModel(const APIData &ad, APIData &adg,
	     const std::shared_ptr<spdlog::logger> &logger);
    XGBModel(const std::string &repo)
      :MLModel(repo) {}
    ~XGBModel() {}

    int read_from_repository(const std::shared_ptr<spdlog::logger> &logger);

    std::string lookup_objective(const std::string &modelfile,
				 const std::shared_ptr<spdlog::logger> &logger);
    
    //TODO
    std::string _weights; /**< file with model weights. */
  };
  
}

#endif
