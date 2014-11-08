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
#include "caffemodel.h"
#include <string>

namespace dd
{
  class CaffeModel : public MLModel
  {
  public:
    CaffeModel(const std::string &def,
	       const std::string &weights)
      :MLModel(),_def(def),_weights(weights)
    {}
    ~CaffeModel() {};

    std::string _def; /**< file name of the model definition in the form of a protocol buffer message description. */
    std::string _weights; /**< file name of the network's weights. */
    std::string _mean; /**< file name of the mean of images, if needed. */
  };
  
}

#endif
