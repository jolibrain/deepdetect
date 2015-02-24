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
#include <string>
#include <unordered_map>

namespace dd
{
  class CaffeModel : public MLModel
  {
  public:
    CaffeModel():MLModel() {}
    CaffeModel(const APIData &ad);
    CaffeModel(const std::string &repo);
    ~CaffeModel() {};

    int read_from_repository(const std::string &repo);
    
    int read_corresp_file();

    inline std::string get_hcorresp(const int &i)
      {
	if (_hcorresp.empty())
	  return std::to_string(i);
	else return _hcorresp[i];
      }

    std::string _repo; /** model repository. */
    std::string _def; /**< file name of the model definition in the form of a protocol buffer message description. */
    std::string _weights; /**< file name of the network's weights. */
    std::string _corresp; /**< file name of the class correspondences (e.g. house / 23) */
    std::unordered_map<int,std::string> _hcorresp; /**< table of class correspondences. */
    std::string _solver; /**< solver description file, included here as part of the model, very specific to Caffe. */
    std::string _sstate; /**< current solver state, useful for resuming training. */
    std::string _model_template; /**< model template name, if any. */
  };
  
}

#endif
