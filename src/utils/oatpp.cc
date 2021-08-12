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

    oatpp::UnorderedFields<oatpp::Any>
    dtoToUFields(const oatpp::Void &polymorph)
    {
      if (polymorph.valueType->classId.id
          != oatpp::data::mapping::type::__class::AbstractObject::CLASS_ID.id)
        {
          return nullptr;
        }

      auto dispatcher
          = static_cast<const oatpp::data::mapping::type::__class::
                            AbstractObject::PolymorphicDispatcher *>(
              polymorph.valueType->polymorphicDispatcher);
      auto fields = dispatcher->getProperties()->getList();
      auto object = static_cast<oatpp::BaseObject *>(polymorph.get());

      oatpp::UnorderedFields<oatpp::Any> result({});

      for (auto const &field : fields)
        {
          result->emplace(field->name, field->get(object));
        }

      return result;
    }

    void dtoToJDoc(const oatpp::Void &polymorph, JDoc &jdoc, bool ignore_null)
    {
      dtoToJVal(polymorph, jdoc, jdoc, ignore_null);
    }

    void dtoToJVal(const oatpp::Void &polymorph, JDoc &jdoc, JVal &jval,
                   bool ignore_null)
    {
      if (polymorph == nullptr)
        {
          return;
        }
      else if (polymorph.valueType == oatpp::Any::Class::getType())
        {
          auto anyHandle
              = static_cast<oatpp::data::mapping::type::AnyHandle *>(
                  polymorph.get());
          dtoToJVal(oatpp::Void(anyHandle->ptr, anyHandle->type), jdoc, jval,
                    ignore_null);
        }
      else if (polymorph.valueType == oatpp::String::Class::getType())
        {
          auto str = polymorph.staticCast<oatpp::String>();
          jval.SetString(str->c_str(), jdoc.GetAllocator());
        }
      else if (polymorph.valueType == oatpp::Int32::Class::getType())
        {
          int32_t i = polymorph.staticCast<oatpp::Int32>();
          jval.SetInt(i);
        }
      else if (polymorph.valueType == oatpp::UInt32::Class::getType())
        {
          uint32_t i = polymorph.staticCast<oatpp::UInt32>();
          jval.SetUint(i);
        }
      else if (polymorph.valueType == oatpp::Int64::Class::getType())
        {
          int64_t i = polymorph.staticCast<oatpp::Int64>();
          jval.SetInt64(i);
        }
      else if (polymorph.valueType == oatpp::UInt64::Class::getType())
        {
          uint64_t i = polymorph.staticCast<oatpp::UInt64>();
          jval.SetUint64(i);
        }
      else if (polymorph.valueType == oatpp::Float32::Class::getType())
        {
          float f = polymorph.staticCast<oatpp::Float32>();
          jval.SetFloat(f);
        }
      else if (polymorph.valueType == oatpp::Float64::Class::getType())
        {
          double f = polymorph.staticCast<oatpp::Float64>();
          jval.SetDouble(f);
        }
      else if (polymorph.valueType == oatpp::Boolean::Class::getType())
        {
          bool b = polymorph.staticCast<oatpp::Boolean>();
          jval = JVal(b);
        }
      else if (polymorph.valueType == DTO::DTOVector<double>::Class::getType())
        {
          auto vec = polymorph.staticCast<DTO::DTOVector<double>>();
          jval = JVal(rapidjson::kArrayType);
          for (size_t i = 0; i < vec->size(); ++i)
            {
              jval.PushBack(vec->at(i), jdoc.GetAllocator());
            }
        }
      else if (polymorph.valueType == DTO::DTOVector<bool>::Class::getType())
        {
          auto vec = polymorph.staticCast<DTO::DTOVector<bool>>();
          jval = JVal(rapidjson::kArrayType);
          for (size_t i = 0; i < vec->size(); ++i)
            {
              jval.PushBack(JVal(bool(vec->at(i))), jdoc.GetAllocator());
            }
        }
      else if (polymorph.valueType->classId.id
               == oatpp::data::mapping::type::__class::AbstractVector::CLASS_ID
                      .id)
        {
          auto vec = polymorph.staticCast<oatpp::AbstractVector>();
          jval = JVal(rapidjson::kArrayType);
          for (size_t i = 0; i < vec->size(); ++i)
            {
              JVal elemJVal;
              dtoToJVal(vec->at(i), jdoc, elemJVal, ignore_null);
              jval.PushBack(elemJVal, jdoc.GetAllocator());
            }
        }
      else if (polymorph.valueType->classId.id
               == oatpp::data::mapping::type::__class::AbstractPairList::
                      CLASS_ID.id)
        {
          jval = JVal(rapidjson::kObjectType);
          auto fields = polymorph.staticCast<oatpp::AbstractFields>();
          for (auto const &field : *fields)
            {
              JVal childJVal;
              dtoToJVal(field.second, jdoc, childJVal, ignore_null);

              if (childJVal.IsNull() && ignore_null)
                continue;

              jval.AddMember(
                  JVal().SetString(field.first->c_str(), jdoc.GetAllocator()),
                  childJVal, jdoc.GetAllocator());
            }
        }
      else if (polymorph.valueType->classId.id
               == oatpp::data::mapping::type::__class::AbstractUnorderedMap::
                      CLASS_ID.id)
        {
          jval = JVal(rapidjson::kObjectType);
          auto fields = polymorph.staticCast<oatpp::AbstractUnorderedFields>();

          for (auto const &field : *fields)
            {
              JVal childJVal;
              dtoToJVal(field.second, jdoc, childJVal, ignore_null);

              if (childJVal.IsNull() && ignore_null)
                continue;

              jval.AddMember(
                  JVal().SetString(field.first->c_str(), jdoc.GetAllocator()),
                  childJVal, jdoc.GetAllocator());
            }
        }
      else if (polymorph.valueType->classId.id
               == oatpp::data::mapping::type::__class::AbstractObject::CLASS_ID
                      .id)
        {
          jval = JVal(rapidjson::kObjectType);
          auto dispatcher
              = static_cast<const oatpp::data::mapping::type::__class::
                                AbstractObject::PolymorphicDispatcher *>(
                  polymorph.valueType->polymorphicDispatcher);
          auto fields = dispatcher->getProperties()->getList();
          auto object = static_cast<oatpp::BaseObject *>(polymorph.get());

          for (auto const &field : fields)
            {
              auto val = field->get(object);
              JVal childJVal;
              dtoToJVal(val, jdoc, childJVal, ignore_null);

              if (childJVal.IsNull() && ignore_null)
                continue;

              jval.AddMember(
                  JVal().SetString(field->name, jdoc.GetAllocator()),
                  childJVal, jdoc.GetAllocator());
            }
        }
      else
        {
          throw std::runtime_error("dtoToJVal: Type not recognised");
        }
    }
  }
}
