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

#ifndef CAFFEMODEL_H
#define CAFFEMODEL_H

#include "mlmodel.h"
#include "apidata.h"
#include <spdlog/spdlog.h>
#include <string>

namespace dd
{
  class CaffeModel : public MLModel
  {
  public:
  CaffeModel(): MLModel() {}
    CaffeModel(const APIData &ad);
    CaffeModel(const APIData &ad, APIData &adg,
	       const std::shared_ptr<spdlog::logger> &logger);
  CaffeModel(const APIData &ad, const std::string &repo)
    :MLModel(ad, repo) {}
    ~CaffeModel() {};

    int read_from_repository(const std::string &repo,
			     const std::shared_ptr<spdlog::logger> &logger);

    int copy_to_target(const std::string &source_repo,
		       const std::string &target_repo,
		       const std::shared_ptr<spdlog::logger> &logger);
    
    std::string _def; /**< file name of the model definition in the form of a protocol buffer message description. */
    std::string _trainf; /**< file name of the training model definition. */
    std::string _weights; /**< file name of the network's weights. */
    std::string _solver; /**< solver description file, included here as part of the model, very specific to Caffe. */
    std::string _sstate; /**< current solver state, useful for resuming training. */
    std::string _model_template; /**< model template name, if any. */
    bool _has_mean_file = false; /**< whether a mean.binaryproto file is available, for image models only. */
};
  
}

#endif
