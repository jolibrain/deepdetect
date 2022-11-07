/**
 * DeepDetect
 * Copyright (c) 2022 Jolibrain
 * Author: Louis Jean <louis.jean@jolibrain.com>
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

#ifndef DD_VISION_FACTORY_H
#define DD_VISION_FACTORY_H

#include <functional>

#include "torchvision/resnet.h"
#include "../native_wrapper.h"

namespace dd
{
  class VisionModelsFactory
  {
  public:
    template <typename TModule>
    static NativeModule *create_wrapper(const APIData &template_params)
    {
      if (template_params.has("nclasses"))
        {
          return new NativeModuleWrapper<TModule>(
              template_params.get("nclasses").get<int>());
        }
      else
        {
          return new NativeModuleWrapper<TModule>();
        }
    }

    static bool is_vision_template(const std::string tdef)
    {
      auto &ctor_map = get_constructor_map();
      return ctor_map.find(tdef) != ctor_map.end();
    }

    template <class TInputConnectorStrategy>
    static NativeModule *from_template(const std::string tdef,
                                       const APIData &template_params,
                                       const TInputConnectorStrategy &inputc)
    {
      (void)(inputc);
      auto &ctor_map = get_constructor_map();
      auto it = ctor_map.find(tdef);

      if (it != ctor_map.end())
        {
          return it->second(template_params);
        }
      else
        return nullptr;
    }

  private:
    static std::map<std::string,
                    std::function<NativeModule *(const APIData &)>> &
    get_constructor_map()
    {
      static std::map<std::string,
                      std::function<NativeModule *(const APIData &)>>
          ctor_map{
            { "resnet18", create_wrapper<vision::models::ResNet18> },
            { "resnet34", create_wrapper<vision::models::ResNet34> },
            { "resnet50", create_wrapper<vision::models::ResNet50> },
            { "resnet101", create_wrapper<vision::models::ResNet101> },
            { "resnet152", create_wrapper<vision::models::ResNet152> },
            { "resnext50_32x4d",
              create_wrapper<vision::models::ResNext50_32x4d> },
            { "resnext101_32x8d",
              create_wrapper<vision::models::ResNext101_32x8d> },
            { "wideresnet50_2",
              create_wrapper<vision::models::WideResNet50_2> },
            { "wideresnet101_2",
              create_wrapper<vision::models::WideResNet101_2> },
          };
      return ctor_map;
    }
  };
}

#endif // DD_VISION_FACTORY_H
