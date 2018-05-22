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

#ifndef TFMODEL_H
#define TFMODEL_H

#include "mlmodel.h"
#include "apidata.h"
#include <spdlog/spdlog.h>
#include <string>

namespace dd
{
  class TFModel : public MLModel
  {
  public:
    TFModel():MLModel() {}
    TFModel(const APIData &ad);
    TFModel(const std::string &repo)
      :MLModel(repo) {}
    ~TFModel() {}
    
    int read_from_repository(const std:: string &repo,
			     const std::shared_ptr<spdlog::logger> &logger);

    int read_corresp_file(const std::shared_ptr<spdlog::logger> &logger);

    inline std::string get_hcorresp(const int &i)
    {
      if (_hcorresp.empty())
	return std::to_string(i);
      else return _hcorresp[i];
    }
    
    std::string _graphName; // Name of the graph 
    std::string _modelRepo;
    std::string _corresp; /**< file name of the class correspondences (e.g. house / 23) */
    std::unordered_map<int,std::string> _hcorresp; /**< table of class correspondences. */
  };

}

#endif
