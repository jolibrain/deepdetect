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

#ifndef CAFFELIB_H
#define CAFFELIB_H

#define CPU_ONLY // TODO: in configure, and add gpu flag

#include "mllibstrategy.h"
#include "caffemodel.h"
#include "caffe/caffe.hpp"

using caffe::Blob;

namespace dd
{
  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel=CaffeModel>
    class CaffeLib : public MLLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>
    {
    public:
      CaffeLib(const CaffeModel &cmodel);
      CaffeLib(CaffeLib &&cl) noexcept;
      ~CaffeLib();
    
      int train(const APIData &ad, std::string &output);
      int predict(const APIData &ad, std::string &output);

      caffe::Net<float> *_net = nullptr;
      bool _gpu = false; /**< whether to use GPU. */
      int _gpuid = 1; /**< GPU id. */
    };
  
}

#endif
