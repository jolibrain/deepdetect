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

#include "backends/caffe2/caffe2model.h"
#include "mllibstrategy.h"
#include "utils/fileops.hpp"

namespace dd
{
  Caffe2Model::Caffe2Model(const APIData &ad)
    :MLModel()
  {
    if (ad.has("templates"))
      this->_mlmodel_template_repo = ad.get("templates").get<std::string>();
    else this->_mlmodel_template_repo += "caffe2"; // default
    if (ad.has("corresp"))
      _corresp = ad.get("corresp").get<std::string>();
    if (ad.has("repository"))
      {
	if (read_from_repository(ad.get("repository").get<std::string>(),spdlog::get("api")))
	  throw MLLibBadParamException("error reading or listing Caffe2 models in repository " + _repo);
      }
    read_corresp_file();
  }

  int Caffe2Model::read_from_repository(const std::string &repo,
					const std::shared_ptr<spdlog::logger> &logger)
  {
    static const std::string &corresp = "corresp";
    this->_repo = repo;
    std::unordered_set<std::string> lfiles;
    int e = fileops::list_directory(repo, true, false, lfiles);
    if (e) {
      logger->error("error reading or listing caffe2 models in repository {}",repo);
      return 1;
    }
    std::string correspf;
    for (auto &file : lfiles) {
      if (file.find(corresp) != std::string::npos) {
	correspf = file;
      }
    }
    if (_corresp.empty())
      _corresp = correspf;
    return 0;
  }
}
