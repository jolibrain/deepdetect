/**
* DeepDetect
 * Copyright (c) 2026 Pixel Forensics, Inc.
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

#ifndef ONNXRUNTIMEMODEL_H
#define ONNXRUNTIMEMODEL_H

#include "dd_spdlog.h"
#include "mlmodel.h"
#include "apidata.h"

namespace dd
{
  class ONNXRuntimeModel : public MLModel
  {
  public:
    ONNXRuntimeModel() : MLModel()
    {
    }

    ONNXRuntimeModel(const APIData &ad, APIData &adg,
                     const std::shared_ptr<spdlog::logger> &logger)
        : MLModel(ad, adg, logger)
    {
      if (ad.has("repository"))
        {
          this->_repo = ad.get("repository").get<std::string>();
          read_from_repository(logger);
          read_corresp_file();
        }
    }

    ONNXRuntimeModel(const std::string &repo) : MLModel(repo)
    {
    }

    ~ONNXRuntimeModel()
    {
    }

    int read_from_repository(const std::shared_ptr<spdlog::logger> &logger);

    std::string _model; // Path to the resolved .onnx weights file
  };
}


#endif // ONNXRUNTIMEMODEL_H
