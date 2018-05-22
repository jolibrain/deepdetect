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
#include <google/protobuf/message.h>

namespace dd {

  class Caffe2Model : public MLModel {
  public:
    Caffe2Model():MLModel() {}
    Caffe2Model(const APIData &ad);
    Caffe2Model(const std::string &repo)
      :MLModel(repo) {}
    ~Caffe2Model() {};

    /**
     * \brief check if the repository contains new files
     */
    void update_from_repository(const std::shared_ptr<spdlog::logger> &logger);

    /**
     * \brief dump informations in the repository
     */
    void write_state(const google::protobuf::Message &init_net,
		     const std::map<std::string, std::string> &blobs);

    /* state of blobs, useful for resuming training */
    std::string _init_state;
    std::string _dbreader_state;
    std::string _dbreader_test_state;
    std::string _iter_state;
    std::string _lr_state;

    std::string _model_template; /* model template name, if any. */
    std::string _predict; /* file name of the predict net. */
    std::string _init; /* file name of the predict net. */
    std::string _meanfile; /* path to a mean.pb file, if available (for image models only). */
  };
}

#endif
