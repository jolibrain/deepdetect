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

#include "backends/caffe2/nettools.h"

namespace dd
{
  namespace Caffe2NetTools
  {

    /*
     *  NetGroup
     */

    void NetGroup::swap(NetGroup &nets)
    {
      _init.Swap(&nets._init);
      _train.Swap(&nets._train);
      _predict.Swap(&nets._predict);
    }

    void NetGroup::rename(const std::string &name)
    {
      std::string prefix("(" + _type + ")" + name);
      _init.set_name(prefix + "_init");
      _train.set_name(prefix + "_train");
      _predict.set_name(prefix + "_predict");
    }

    void NetGroup::import(const std::string &init, const std::string &predict,
                          const std::string &train)
    {
      if (init.size())
        import_net(_init, init);
      if (predict.size())
        import_net(_predict, predict);
      if (train.size())
        import_net(_train, train);
    }

    NetGroup::NetGroup(const std::string &type, const std::string &init,
                       const std::string &predict, const std::string &train)
        : _type(type)
    {
      import(init, predict, train);
    }

  }
}
