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

namespace dd {

  Caffe2Model::Caffe2Model(const APIData &ad)
    :MLModel()
  {
    std::map<std::string, std::string *> names =
      {
	{ "predictf", &_predict },
	{ "initf", &_init },
	{ "corresp", &_corresp },
	{ "weights", &_weights }
      };

    // Update from API
    for (auto &it : names) {
      if (ad.has(it.first)) {
	*it.second = ad.get(it.first).get<std::string>();
      }
    }

    const std::shared_ptr<spdlog::logger> &logger = spdlog::get("api");

    // Register repositories
    this->_repo = ad.get("repository").get<std::string>();
    this->_mlmodel_template_repo = ad.has("templates") ?
      ad.get("templates").get<std::string>() : "caffe2"; // Default

    // List the extensions' nets
    if (ad.has("extensions")) {
      std::vector<APIData> extensions = ad.getv("extensions");
      for (const APIData &extension : extensions) {

	const std::string &ext_type(extension.get("type").get<std::string>());
	std::string ext_repo;
	if (extension.has("repository")) {
	  ext_repo = extension.get("repository").get<std::string>();
	} else {
	  ext_repo = this->_repo + "/" + ext_type;
	}

	// Check if the extension is a repository
	bool is_dir;
	if (!fileops::file_exists(ext_repo, is_dir) || !is_dir) {
	  std::string msg("'" + ext_repo + "' is not a directory");
	  logger->error(msg);
	  throw MLLibBadParamException(msg);
	}

	_extensions.emplace_back();
	Extension &ext(_extensions.back());
	ext._init = ext_repo + "/init_net.pb";
	ext._predict = ext_repo + "/predict_net.pb";
	ext._type = ext_type;

	// Check if the nets exist
	if (!fileops::file_exists(ext._predict)) {
	  std::string msg("'" + ext._predict + "' does not exists");
	  logger->error(msg);
	  throw MLLibBadParamException(msg);
	}
	if (!fileops::file_exists(ext._init)) {
	  logger->warn("No initialization net found in '" + ext_repo + "'");
	  ext._init = "";
	}
      }
    }

    update_from_repository(logger);
  }

  void Caffe2Model::update_from_repository(const std::shared_ptr<spdlog::logger> &logger) {
    std::map<std::string, std::string *> names =
      {
	{ "predict_net.pb", &_predict },
	{ "init_net.pb", &_init },
	{ "corresp.txt", &_corresp },
	{ "mean.pb", &_meanfile },
	{ "init_state.pb", &_init_state },
	{ "dbreader_state.pb", &_dbreader_state },
	{ "dbreader_train_state.pb", &_dbreader_train_state },
	{ "iter_state.pb", &_iter_state },
	{ "lr_state.pb", &_lr_state },
      };

    // List available files
    std::unordered_set<std::string> lfiles;
    if (fileops::list_directory(_repo, true, false, false, lfiles)) {
      std::string msg("error reading or listing Caffe2 models in repository " + _repo);
      logger->error(msg);
      throw MLLibBadParamException(msg);
    }

    for (const std::string &file : lfiles) {
      for (auto &it : names) {
	// if the file name contains a string from the map
	if (file.find(it.first) != std::string::npos) {
	  // And if the corresponding variable is still uninitialized
	  if (it.second->empty()) {
	    *it.second = file;
	  }
	  break;
	}
      }
    }

    read_corresp_file();
  }

  void Caffe2Model::get_hcorresp(std::vector<std::string> &clnames) {
    int i = 0;
    for (std::string &name : clnames) {
      name = get_hcorresp(i++);
    }
  }

  void Caffe2Model::write_state(const google::protobuf::Message &init,
				const std::map<std::string, std::string> &blobs) {
    for (auto it : blobs) {
      std::ofstream(_repo + "/" + it.first + "_state.pb") << it.second;
    }
    std::ofstream f(_repo + "/init_state.pb");
    init.SerializeToOstream(&f);
  }

  void Caffe2Model::list_template_files(const std::string &name,
					std::map<std::string, std::string> &files,
					bool external_weights) {

    // Path manipulation
    std::string source = this->_mlmodel_template_repo + '/' + name;
    auto set_path = [&](const std::string &net, const std::string &remote="") {
      files[remote.empty() ? source + '/' + net + ".pbtxt" : remote] = _repo + '/' + net + ".pb";
    };

    // Choose the files
    set_path("predict_net");
    if (!external_weights) {
      set_path("init_net");
    } else if (_weights.empty()) {
      throw MLLibBadParamException("No external weights specified");
    } else {
      set_path("init_net", _weights);
    }
  }
}
