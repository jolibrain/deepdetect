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

#ifndef MLSERVICE_H
#define MLSERVICE_H

#include "mllibstrategy.h"
#include "mlmodel.h"
#include <string>
#include <iostream>

namespace dd
{
  template<template <class U,class V,class W> class TMLLib, class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
    class MLService : public TMLLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>
  {
  public:
    MLService(const TMLModel &mlmodel)
      :TMLLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>(mlmodel)
      {}
    MLService(MLService &&mls) noexcept
      :TMLLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>(std::move(mls)),_sid(std::move(mls._sid)),_sname(std::move(mls._sname))
      {}
    ~MLService() {}
    
    int _sid; /**< service id, hash from name. */
    std::string _sname; /**< service name. */
  };
}

#endif
