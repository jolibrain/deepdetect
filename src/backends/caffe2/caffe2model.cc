/**
 * DeepDetect
 * Copyright (c) 2018 Jolibrain
 * Author: Julien Chicha
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
    static const std::map<std::string, std::string *> names =
      {
	{ "predictf", &_predict },
	{ "initf", &_init },
	{ "corresp", &_corresp}
      };

    for (auto &it : names) {
      if (ad.has(it.first)) {
	*it.second = ad.get(it.first).get<std::string>();
      }
    }

    if (ad.has("repository"))
      {
	if (read_from_repository(ad.get("repository").get<std::string>(),spdlog::get("api")))
	  throw MLLibBadParamException("error reading or listing Caffe2 models in repository " +
				       _repo);
      }
    read_corresp_file();
  }

  int Caffe2Model::read_from_repository(const std::string &repo,
					const std::shared_ptr<spdlog::logger> &logger)
  {
    std::map<std::string, std::string *> names =
      {
	{ "predict_net.pb", &_predict },
	{ "init_net.pb", &_init },
	{ "corresp.txt", &_corresp}
      };
    this->_repo = repo;
    std::unordered_set<std::string> lfiles;
    int e = fileops::list_directory(repo, true, false, lfiles);
    if (e) {
      logger->error("error reading or listing caffe2 models in repository {}",repo);
      return 1;
    }

    for (const std::string &file : lfiles) {
      for (auto &it : names) {
	if (file.find(it.first) != std::string::npos) {
	  if (it.second->empty()) {
	    *it.second = file;
	  }
	  break;
	}
      }
    }
    return 0;
  }
}
