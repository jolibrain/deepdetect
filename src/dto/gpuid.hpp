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

#ifndef DD_DTO_GPUID_HPP
#define DD_DTO_GPUID_HPP

#include "oatpp/core/Types.hpp"
#include "oatpp/parser/json/mapping/ObjectMapper.hpp"

namespace dd
{
  namespace DTO
  {
    struct VGpuIds
    {
      std::vector<int> _ids;

      VGpuIds() : _ids({ 0 })
      {
      }
      VGpuIds(const oatpp::Vector<oatpp::Int32> &vec)
          : _ids(vec->begin(), vec->end())
      {
      }
      VGpuIds(oatpp::Int32 i) : _ids({ i })
      {
      }
    };

    namespace __class
    {
      class GpuIdsClass;
    }

    typedef oatpp::data::mapping::type::Primitive<VGpuIds,
                                                  __class::GpuIdsClass>
        GpuIds;

    namespace __class
    {
      class GpuIdsClass
      {
      public:
        static const oatpp::ClassId CLASS_ID;

        static oatpp::Type *getType()
        {
          static oatpp::Type type(CLASS_ID, nullptr, nullptr);
          return &type;
        }
      };
    }

    static inline oatpp::Void
    gpuIdsDeserialize(oatpp::parser::json::mapping::Deserializer *deserializer,
                      oatpp::parser::Caret &caret,
                      const oatpp::Type *const type)
    {
      (void)type;
      if (caret.isAtChar('['))
        {
          return GpuIds(VGpuIds{
              deserializer
                  ->deserialize(caret,
                                oatpp::Vector<oatpp::Int32>::Class::getType())
                  .staticCast<oatpp::Vector<oatpp::Int32>>() });
        }
      else
        {
          return GpuIds(VGpuIds{
              deserializer->deserialize(caret, oatpp::Int32::Class::getType())
                  .staticCast<oatpp::Int32>() });
        }
    }
  }
}

#endif // DD_DTO_GPUID_HPP
