/**
 * DeepDetect
 * Copyright (c) 2018 Jolibrain
 * Author: Corentin Barreau <corentin.barreau@epitech.eu>
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

#ifndef NCNNMODEL_H
#define NCNNMODEL_H

#include "dd_spdlog.h"
#include "mlmodel.h"
#include "apidata.h"
#include "dto/model.hpp"
#include "dto/service_create.hpp"

namespace dd
{
  class NCNNModel : public MLModel
  {
  public:
    NCNNModel() : MLModel()
    {
    }
    NCNNModel(const oatpp::Object<DTO::Model> &model_dto,
            const oatpp::Object<DTO::ServiceCreate> &service_dto,
              const std::shared_ptr<spdlog::logger> &logger)
        : MLModel(model_dto, service_dto, logger)
    {
      if (model_dto->repository)
        this->_repo = model_dto->repository->std_str();
      read_from_repository(spdlog::get("api"));
      read_corresp_file();
    }
    NCNNModel(const std::string &repo) : MLModel(repo)
    {
    }
    ~NCNNModel()
    {
    }

    int read_from_repository(const std::shared_ptr<spdlog::logger> &logger);

  public:
    std::string _weights;
    std::string _params;
  };
}

#endif
