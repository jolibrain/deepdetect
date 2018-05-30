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

#ifndef CAFFE2MODEL_H
#define CAFFE2MODEL_H

#include "mlmodel.h"
#include "apidata.h"
#include <spdlog/spdlog.h>
#include <string>

namespace dd
{
  class Caffe2Model : public MLModel
  {
  public:
    Caffe2Model():MLModel() {}
    Caffe2Model(const APIData &ad);
    Caffe2Model(const std::string &repo)
      :MLModel(repo) {}
    ~Caffe2Model() {};

    int read_from_repository(const std::string &repo,
			     const std::shared_ptr<spdlog::logger> &logger);

    std::string _predict; /**< file name of the predict net. */
    std::string _init; /**< file name of the predict net. */
  };
}

#endif
