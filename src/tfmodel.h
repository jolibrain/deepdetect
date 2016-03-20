/**
 * DeepDetect
 * Copyright (c) 2016 Emmanuel Benazera
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

#ifndef TFMODEL_H
#define TFMODEL_H

#include "mlmodel.h"
#include "apidata.h"

namespace dd
{
  class TFModel : public MLModel
  {
  public:
    TFModel():MLModel() {}
    TFModel(const APIData &ad);
    TFModel(const std::string &repo)
      :MLModel(repo) {}
    ~TFModel() {}

    int read_from_repository();
    
    //TODO: model files
    
  };
}

#endif
