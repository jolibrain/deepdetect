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

#include "native_factory.h"

namespace dd
{

  template <class TInputConnectorStrategy>
  NativeModule *
  NativeFactory::from_template(const std::string tdef,
                               const APIData template_params,
                               const TInputConnectorStrategy &inputc)
  {
    (void)(tdef);
    (void)(template_params);
    (void)(inputc);
    return nullptr;
  }

  template <>
  NativeModule *NativeFactory::from_template<CSVTSTorchInputFileConn>(
      const std::string tdef, const APIData template_params,
      const CSVTSTorchInputFileConn &inputc)
  {
    if (tdef.find("nbeats") != std::string::npos)
      {
        std::vector<std::string> p;
        if (template_params.has("template_params"))
          p = template_params.get("template_params")
                  .get<std::vector<std::string>>();
        return new NBeats(inputc, p);
      }
    else
      return nullptr;
  }

  template NativeModule *
  NativeFactory::from_template(const std::string tdef,
                               const APIData template_params,
                               const TxtTorchInputFileConn &inputc);

  template NativeModule *
  NativeFactory::from_template(const std::string tdef,
                               const APIData template_params,
                               const ImgTorchInputFileConn &inputc);
}
