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

#ifndef DD_DTO_TYPES_HPP
#define DD_DTO_TYPES_HPP

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
      template <typename T> class DTOVectorClass;
    }

    typedef oatpp::data::mapping::type::Primitive<VGpuIds,
                                                  __class::GpuIdsClass>
        GpuIds;
    template <typename T>
    using DTOVector
        = oatpp::data::mapping::type::Primitive<std::vector<T>,
                                                __class::DTOVectorClass<T>>;

    namespace __class
    {
      class GpuIdsClass
      {
      public:
        static const oatpp::ClassId CLASS_ID;

        static oatpp::Type *getType()
        {
          static oatpp::Type type(CLASS_ID);
          return &type;
        }
      };

      template <typename T> class DTOVectorClass
      {
      public:
        static const oatpp::ClassId CLASS_ID;

        static oatpp::Type *getType()
        {
          static oatpp::Type type(CLASS_ID);
          return &type;
        }
      };

      template class DTOVectorClass<double>;
      template class DTOVectorClass<bool>;
    }

    // ==== Serialization methods

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
                  .cast<oatpp::Vector<oatpp::Int32>>() });
        }
      else
        {
          return GpuIds(VGpuIds{
              deserializer->deserialize(caret, oatpp::Int32::Class::getType())
                  .cast<oatpp::Int32>() });
        }
    }

    static inline void
    gpuIdsSerialize(oatpp::parser::json::mapping::Serializer *serializer,
                    oatpp::data::stream::ConsistentOutputStream *stream,
                    const oatpp::Void &obj)
    {
      auto gpuid = obj.cast<GpuIds>();
      if (gpuid->_ids.size() == 1)
        {
          oatpp::Int32 id = gpuid->_ids[0];
          serializer->serializeToStream(stream, id);
        }
      else
        {
          oatpp::Vector<oatpp::Int32> ids;
          for (auto i : gpuid->_ids)
            {
              ids->push_back(i);
            }
          serializer->serializeToStream(stream, ids);
        }
    }

    // Inspired by oatpp json deserializer
    template <typename T> inline T readVecElement(oatpp::parser::Caret &caret);

    template <>
    inline double readVecElement<double>(oatpp::parser::Caret &caret)
    {
      return caret.parseFloat64();
    }

    template <>
    inline uint8_t readVecElement<uint8_t>(oatpp::parser::Caret &caret)
    {
      return static_cast<uint8_t>(caret.parseUnsignedInt());
    }

    template <> inline bool readVecElement<bool>(oatpp::parser::Caret &caret)
    {
      if (caret.isAtText("true"))
        {
          return true;
        }
      else if (caret.isAtText("false"))
        {
          return false;
        }
      else
        {
          caret.setError("[readVecElement<bool>] expected 'true' or 'false'",
                         oatpp::parser::json::mapping::Deserializer::
                             ERROR_CODE_VALUE_BOOLEAN);
          return false;
        }
    }

    template <typename T>
    static inline oatpp::Void
    vectorDeserialize(oatpp::parser::json::mapping::Deserializer *deserializer,
                      oatpp::parser::Caret &caret,
                      const oatpp::Type *const type)
    {
      (void)deserializer;
      (void)type;
      if (caret.isAtText("null", true))
        {
          return oatpp::Void(type);
        }

      if (caret.canContinueAtChar('[', 1))
        {
          std::vector<T> vec;

          while (!caret.isAtChar(']') && caret.canContinue())
            {
              caret.skipBlankChars();
              vec.push_back(caret.parseFloat64());
              if (caret.hasError())
                {
                  return nullptr;
                }
              caret.skipBlankChars();
              caret.canContinueAtChar(',', 1);
            }

          if (!caret.canContinueAtChar(']', 1))
            {
              if (!caret.hasError())
                {
                  caret.setError(
                      "[oatpp::parser::json::mapping::Deserializer::"
                      "deserializeList()]: Error. ']' - expected",
                      oatpp::parser::json::mapping::Deserializer::
                          ERROR_CODE_ARRAY_SCOPE_CLOSE);
                }
              return nullptr;
            };

          return DTOVector<T>(std::move(vec));
        }
      else
        {
          caret.setError("[oatpp::parser::json::mapping::Deserializer::"
                         "deserializeList()]: Error. '[' - expected",
                         oatpp::parser::json::mapping::Deserializer::
                             ERROR_CODE_ARRAY_SCOPE_OPEN);
          return nullptr;
        }
    }

    template <typename T>
    static inline void
    vectorSerialize(oatpp::parser::json::mapping::Serializer *serializer,
                    oatpp::data::stream::ConsistentOutputStream *stream,
                    const oatpp::Void &obj)
    {
      (void)serializer;
      auto vec = obj.cast<DTOVector<T>>();
      stream->writeCharSimple('[');
      bool first = true;

      for (auto val : *vec)
        {
          if (first)
            first = false;
          else
            stream->writeCharSimple(',');
          stream->writeAsString(val);
        }

      stream->writeCharSimple(']');
    }
  }
}

#endif // DD_DTO_TYPES_HPP
