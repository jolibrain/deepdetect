/**
 * DeepDetect
 * Copyright (c) 2014-2016 Emmanuel Benazera
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

#ifndef NET_GENERATOR_H
#define NET_GENERATOR_H

#include "apidata.h"

namespace dd
{

  template <class TNetInput, class TNetLayers, class TNetLoss>
  class NetGenerator
  {
  public:
    NetGenerator()
    {
    }
    ~NetGenerator()
    {
    }
  };

  template <class TInputConnectorStrategy> class NetInput
  {
  public:
    NetInput()
    {
    }
    ~NetInput()
    {
    }

    void configure_inputs(const APIData &ad_input,
                          const TInputConnectorStrategy &inputc)
    {
    }
  };

  class NetLayers
  {
  public:
    NetLayers()
    {
    }
    ~NetLayers()
    {
    }

    // void add_basic_block();
    void configure_net(const APIData &ad_mllib);

    // TODO: basic layers, e.g. conv, etc ?
  };

  class NetLoss
  {
  public:
    NetLoss()
    {
    }
    ~NetLoss()
    {
    }

    // TODO
  };

}

#endif
