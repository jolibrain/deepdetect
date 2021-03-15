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

#include "native_wrapper.h"

namespace dd
{

  template <class TInputConnectorStrategy>
  NativeModule *
  NativeFactory::from_template(const std::string tdef,
                               const APIData &template_params,
                               const TInputConnectorStrategy &inputc,
                               const std::shared_ptr<spdlog::logger> &logger)
  {
    (void)(tdef);
    (void)(template_params);
    (void)(inputc);
    (void)(logger);
    return nullptr;
  }

  template <>
  NativeModule *NativeFactory::from_template<CSVTSTorchInputFileConn>(
      const std::string tdef, const APIData &template_params,
      const CSVTSTorchInputFileConn &inputc,
      const std::shared_ptr<spdlog::logger> &logger)
  {
    if (tdef.find("nbeats") != std::string::npos)
      {
        std::vector<std::string> p;
        double bc_loss_coef = NBEATS_DEFAULT_BACKCAST_LOSS_COEFF;
        if (template_params.has("stackdef"))
          {
            p = template_params.get("stackdef")
                    .get<std::vector<std::string>>();
          }
        if (template_params.has("backcast_loss_coef"))
          {
            bc_loss_coef
                = template_params.get("backcast_loss_coef").get<double>();
          }
        return new NBeats(inputc, p, bc_loss_coef);
      }
    else if (tdef.find("ttransformer") != std::string::npos)
      return new TTransformer(inputc, template_params, logger);
    else
      return nullptr;
  }

  template <>
  NativeModule *NativeFactory::from_template<ImgTorchInputFileConn>(
      const std::string tdef, const APIData &template_params,
      const ImgTorchInputFileConn &inputc,
      const std::shared_ptr<spdlog::logger> &logger)
  {
    (void)inputc;
    (void)logger;

    if (tdef.find("vit") != std::string::npos)
      {

        return new ViT(inputc, template_params);
      }
    else if (VisionModelsFactory::is_vision_template(tdef))
      {
        return VisionModelsFactory::from_template(tdef, template_params,
                                                  inputc);
      }
    return nullptr;
  }

  template NativeModule *
  NativeFactory::from_template(const std::string tdef,
                               const APIData &template_params,
                               const TxtTorchInputFileConn &inputc,
                               const std::shared_ptr<spdlog::logger> &logger);
}
