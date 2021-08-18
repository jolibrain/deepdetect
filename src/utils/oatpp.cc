/**
 * DeepDetect
 * Copyright (c) 2021 Jolibrain SASU
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

#include "oatpp.hpp"

#include "dto/ddtypes.hpp"

namespace dd
{
  namespace oatpp_utils
  {
    std::shared_ptr<oatpp::parser::json::mapping::ObjectMapper>
    createDDMapper()
    {
      std::shared_ptr<oatpp::parser::json::mapping::ObjectMapper> object_mapper
          = oatpp::parser::json::mapping::ObjectMapper::createShared();
      auto deser = object_mapper->getDeserializer();
      deser->setDeserializerMethod(DTO::GpuIds::Class::CLASS_ID,
                                   DTO::gpuIdsDeserialize);
      deser->setDeserializerMethod(DTO::DTOVector<double>::Class::CLASS_ID,
                                   DTO::vectorDeserialize<double>);
      deser->setDeserializerMethod(DTO::DTOVector<bool>::Class::CLASS_ID,
                                   DTO::vectorDeserialize<bool>);
      auto ser = object_mapper->getSerializer();
      ser->setSerializerMethod(DTO::GpuIds::Class::CLASS_ID,
                               DTO::gpuIdsSerialize);
      ser->setSerializerMethod(DTO::DTOVector<double>::Class::CLASS_ID,
                               DTO::vectorSerialize<double>);
      ser->setSerializerMethod(DTO::DTOVector<bool>::Class::CLASS_ID,
                               DTO::vectorSerialize<bool>);
      return object_mapper;
    }
  }
}
