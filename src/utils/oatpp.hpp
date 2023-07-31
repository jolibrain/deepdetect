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

#ifndef DD_OATPP_UTILS
#define DD_OATPP_UTILS

#include "oatpp/parser/json/mapping/ObjectMapper.hpp"

#include "dd_types.h"

namespace dd
{
  namespace oatpp_utils
  {
    /** Create Oat++ ObjectMapper with serialization / deserialization
     * methods for custom types. */
    std::shared_ptr<oatpp::parser::json::mapping::ObjectMapper>
    createDDMapper();

    /** Convert a DTO into dynamic structure */
    oatpp::UnorderedFields<oatpp::Any>
    dtoToUFields(const oatpp::Void &polymorph);

    template <typename Wrapper> inline Wrapper staticCast(oatpp::Void val)
    {
      return Wrapper(
          std::static_pointer_cast<typename Wrapper::ObjectType>(val.getPtr()),
          val.getValueType());
    }

    template <class S, class D>
    inline std::vector<D> dtoVecToVec(oatpp::Vector<S> dto_vec)
    {
      std::vector<D> vec;
      for (auto v : *dto_vec)
        vec.push_back(v);
      return vec;
    }

    template <>
    std::vector<std::string> inline dtoVecToVec<oatpp::String>(
        oatpp::Vector<oatpp::String> dto_vec)
    {
      std::vector<std::string> vec;
      for (auto v : *dto_vec)
        vec.push_back(v);
      return vec;
    }

    inline oatpp::Void getAnyContent(const oatpp::Any &any)
    {
      auto anyHandle
          = static_cast<oatpp::data::mapping::type::AnyHandle *>(any.get());
      return oatpp::Void(anyHandle->ptr, anyHandle->type);
    }

    template <class S, class D> inline std::vector<D> anyToVec(oatpp::Any any)
    {
      auto polymorph = getAnyContent(any);
      auto poly_dispatch
          = static_cast<const oatpp::data::mapping::type::__class::Collection::
                            PolymorphicDispatcher *>(
              polymorph.getValueType()->polymorphicDispatcher);
      std::vector<D> vec;
      for (auto it = poly_dispatch->beginIteration(polymorph); !it->finished();
           it->next())
        {
          auto val = it->get().cast<oatpp::Any>().retrieve<S>();
          vec.push_back(val);
        }
      return vec;
    }

    inline bool isList(const oatpp::Void &polymorph)
    {
      return polymorph.getValueType()->classId.id
                 == oatpp::data::mapping::type::__class::AbstractVector::
                        CLASS_ID.id
             || polymorph.getValueType()->classId.id
                    == oatpp::data::mapping::type::__class::AbstractList::
                           CLASS_ID.id;
    }

    void dtoToJDoc(const oatpp::Void &polymorph, JDoc &jdoc,
                   bool ignore_null = true);
    void dtoToJVal(const oatpp::Void &polymorph, JDoc &jdoc, JVal &jval,
                   bool ignore_null = true);

    std::string dtoToJSONString(const oatpp::Void &polymorph,
                                bool ignore_null = true);
  }
}

#endif // DD_OATPP_UTILS
