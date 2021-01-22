/**
 * DeepDetect
 * Copyright (c) 2019-2020 Jolibrain
 * Author:  Guillaume Infantes <guillaume.infantes@jolibrain.com>
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

#ifndef NATIVE_FACTORY_H
#define NATIVE_FACTORY_H

#include "native_net.h"
#include "./templates/nbeats.h"
#include "./templates/vit.h"
#include "./templates/ttransformer.h"
#include "../torchinputconns.h"
#include "apidata.h"
#include "templates/vision_models.h"

namespace dd
{
  class NativeFactory
  {
  public:
    template <class TInputConnectorStrategy>
    static NativeModule *
    from_template(const std::string tdef, const APIData &template_params,
                  const TInputConnectorStrategy &inputc,
                  const std::shared_ptr<spdlog::logger> &logger);

    static bool valid_template_def(std::string tdef)
    {
      if (tdef.find("nbeats") != std::string::npos
          || tdef.find("vit") != std::string::npos
          || tdef.find("ttransformer") != std::string::npos)
        return true;
      else if (VisionModelsFactory::is_vision_template(tdef))
        return true;
      return false;
    }

    static bool is_timeserie(std::string tdef)
    {
      if (tdef.find("nbeats") != std::string::npos
          || tdef.find("ttransformer") != std::string::npos)
        return true;
      return false;
    }
  };
}
#endif
