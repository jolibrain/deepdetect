/**
 * DeepDetect
 * Copyright (c) 2019-2020 Jolibrain
 * Authors: Louis Jean <ljean@etud.insa-toulouse.fr>
 *           Guillaume Infantes <guillaume.infantes@jolibrain.com>
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

#ifndef TORCHMODEL_H
#define TORCHMODEL_H

#include "mlmodel.h"
#include "apidata.h"

namespace dd
{
  class TorchModel : public MLModel
  {
  public:
    /**
     * empty constructor
     */
    TorchModel() : MLModel()
    {
    }

    /**
     * constructor that inits file names from repo
     */
    TorchModel(const APIData &ad, APIData &adg,
               const std::shared_ptr<spdlog::logger> &logger)
        : MLModel(ad, adg, logger)
    {

      read_from_repository(spdlog::get("api"));
      read_corresp_file();
    }

    TorchModel(const std::string &repo) : MLModel(repo)
    {
    }

    ~TorchModel()
    {
    }

    /**
     * find files in repo
     */
    int read_from_repository(const std::shared_ptr<spdlog::logger> &logger);

    /** Copy source repository to target repository. */
    int copy_to_target(const std::string &source_repo,
                       const std::string &target_repo,
                       const std::shared_ptr<spdlog::logger> &logger);

    void update_config_json_parameters(
        const std::string &target_repo,
        const std::shared_ptr<spdlog::logger> &logger);

  public:
    std::string _traced;  /**< path of the traced part of the net. */
    std::string _weights; /**< path of the weights of the net. */
    std::string _sstate;  /**< current solver state to resume training */
    std::string _proto;   /**< prototxt file generated or read as graph */
    std::string _native;  /**< native torch net */
  };
}

#endif // TORCHMODEL_H
