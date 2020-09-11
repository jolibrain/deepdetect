/**
 * DeepDetect
 * Copyright (c) 2018 Pixel Forensics, Inc.
 * Author: Cheni Chadowitz <cchadowitz@pixelforensics.com>
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

#include "dlibmodel.h"
#include <utils/fileops.hpp>
#include <string>

namespace dd
{
  DlibModel::DlibModel(const APIData &ad, APIData &adg,
                       const std::shared_ptr<spdlog::logger> &logger)
      : MLModel(ad, adg, logger)
  {
    if (ad.has("repository"))
      {
        read_from_repository(
            ad.get("repository").get<std::string>(),
            spdlog::get("api")); // XXX: beware, error not caught
        this->_modelRepo = ad.get("repository").get<std::string>();
      }
  }

  int DlibModel::read_from_repository(
      const std::string &repo, const std::shared_ptr<spdlog::logger> &logger)
  {
    std::string modelName = ".dat";
    std::string shapePredictorName = ".shapepredictor";
    this->_repo = repo;
    std::unordered_set<std::string> lfiles;
    int e = fileops::list_directory(repo, true, false, false, lfiles);
    if (e != 0)
      {
        logger->error("error reading or listing dlib models in repository {}",
                      repo);
        throw MLLibBadParamException(
            "error reading or listing dlib models in repository " + _repo);
      }

    auto hit = lfiles.begin();
    std::string modelf;
    long int state_t = -1, state_t_sp = -1;
    while (hit != lfiles.end())
      {
        if ((*hit).find(modelName) != std::string::npos)
          {
            // stat file to pick the latest one
            long int st = fileops::file_last_modif((*hit));
            if (st > state_t)
              {
                modelf = (*hit);
                state_t = st;
              }
          }
        if (_hasShapePredictor
            && (*hit).find(shapePredictorName) != std::string::npos)
          {
            long int st = fileops::file_last_modif((*hit));
            if (st > state_t_sp)
              {
                _shapePredictorName = (*hit);
                state_t_sp = st;
              }
          }
        ++hit;
      }
    _modelName = modelf;

    return 0;
  }
}
