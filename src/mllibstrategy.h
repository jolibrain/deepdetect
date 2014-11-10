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

#ifndef MLLIBSTRATEGY_H
#define MLLIBSTRATEGY_H

#include "apidata.h"

namespace dd
{

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
    class MLLib
  {
  public:
    MLLib(const TMLModel &mlmodel)
      :_mlmodel(mlmodel) {}
    
    MLLib(MLLib &&mll) noexcept
      :_mlmodel(mll._mlmodel)
      {}
    
    ~MLLib() {}

    int train(const APIData &ad);
    int predict(const APIData &ad, std::string &output);
    int status() const;
    
    TInputConnectorStrategy _inputc;
    TOutputConnectorStrategy _outputc;

    bool _has_train = false; /**< whether training is available. */
    bool _has_predict = true; /**< whether prediction is available. */

    TMLModel _mlmodel;
  };  
  
}

#endif
