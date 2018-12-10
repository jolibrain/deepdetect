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

#ifndef DLIBMODEL_H
#define DLIBMODEL_H

#include "mlmodel.h"
#include "apidata.h"
#include <spdlog/spdlog.h>
#include <string>

namespace dd {
    class DlibModel : public MLModel {
    public:
        DlibModel() : MLModel() {}

        DlibModel(const APIData &ad, APIData &adg,
		  const std::shared_ptr<spdlog::logger> &logger);

        DlibModel(const std::string &repo)
                : MLModel(repo) {}

        ~DlibModel() {}

        int read_from_repository(const std::string &repo,
                                 const std::shared_ptr<spdlog::logger> &logger);

        std::string _modelName; // Name of the graph
        std::string _modelRepo;
    };

}

#endif
