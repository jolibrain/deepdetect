/**
 * DeepDetect
 * Copyright (c) 2020 Jolibrain SASU
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

#include <iostream>
#include <gtest/gtest.h>

#include "oatpp/core/Types.hpp"
#include "oatpp/core/macro/codegen.hpp"

#include "dto/ddtypes.hpp"
#include "utils/oatpp.hpp"

#include OATPP_CODEGEN_BEGIN(DTO) ///< Begin DTO codegen section

class VectorDTOTest : public oatpp::DTO
{
  DTO_INIT(VectorDTOTest, DTO)

  DTO_FIELD(dd::DTO::DTOVector<double>, dto_vec) = std::vector<double>();
};

class CompleteDTOTest : public oatpp::DTO
{
  DTO_INIT(CompleteDTOTest, DTO);
  DTO_FIELD(Boolean, b) = true;
  DTO_FIELD(Int32, i32) = 12;
  DTO_FIELD(Int64, i64) = 0xfffffffffl;
  DTO_FIELD(Float32, f32) = 0.12f;
  DTO_FIELD(Float64, f64) = 3.14159265358979323846;
  DTO_FIELD(String, s) = "string";
  DTO_FIELD(oatpp::Vector<oatpp::Any>, v)
      = oatpp::Vector<oatpp::Any>::createShared();
  DTO_FIELD(dd::DTO::DTOVector<bool>, dto_vec) = std::vector<bool>();
  DTO_FIELD(oatpp::Object<VectorDTOTest>, child)
      = VectorDTOTest::createShared();
  DTO_FIELD(oatpp::UnorderedFields<Any>, ufields)
      = UnorderedFields<Any>::createShared();
  DTO_FIELD(oatpp::Fields<Any>, fields) = Fields<Any>::createShared();
};

#include OATPP_CODEGEN_END(DTO) ///< End DTO codegen section

TEST(dto, vector_dto)
{
  auto vec_test = VectorDTOTest::createShared();
  ASSERT_TRUE(vec_test->dto_vec != nullptr);

  int size = 1000;
  for (int i = 0; i < size; ++i)
    {
      vec_test->dto_vec->push_back(i);
    }
  ASSERT_EQ(vec_test->dto_vec->size(), size);
  ASSERT_EQ(vec_test->dto_vec->at(567), 567.0);
  ASSERT_EQ(vec_test->dto_vec->at(size - 1), size - 1.0);

  auto mapper = dd::oatpp_utils::createDDMapper();
  oatpp::String saved = mapper->writeToString(vec_test);
  auto vec_test2 = mapper->readFromString<oatpp::Object<VectorDTOTest>>(saved);

  ASSERT_EQ(vec_test2->dto_vec->size(), size);
  ASSERT_EQ(vec_test2->dto_vec->at(567), 567.0);
  ASSERT_EQ(vec_test2->dto_vec->at(size - 1), size - 1.0);
}

TEST(dto, dto_to_json)
{
  auto dto = CompleteDTOTest::createShared();
  dto->dto_vec->push_back(true);
  dto->child->dto_vec->push_back(2.3);
  dto->v->push_back(oatpp::Int32(5));
  dto->v->push_back(VectorDTOTest::createShared());
  dto->ufields->emplace("a", oatpp::Boolean(false));
  dto->fields->push_back({ "b", oatpp::Boolean(true) });

  // Debug help
  auto mapper = dd::oatpp_utils::createDDMapper();
  oatpp::String saved = mapper->writeToString(dto);
  std::cout << saved->std_str() << std::endl;

  JDoc jdoc;
  dd::oatpp_utils::dtoToJDoc(dto, jdoc);

  ASSERT_EQ(jdoc["b"].GetBool(), true);
  ASSERT_EQ(jdoc["i32"].GetInt(), 12);
  ASSERT_EQ(jdoc["i64"].GetInt64(), 0xfffffffffl);
  ASSERT_EQ(jdoc["f32"].GetFloat(), 0.12f);
  ASSERT_EQ(jdoc["f64"].GetDouble(), 3.14159265358979323846);
  ASSERT_EQ(jdoc["s"].GetString(), std::string("string"));
  ASSERT_EQ(jdoc["v"].Size(), 2);
  ASSERT_EQ(jdoc["v"][0].GetInt(), 5);
  ASSERT_EQ(jdoc["v"][1]["dto_vec"].Size(), 0);
  ASSERT_EQ(jdoc["dto_vec"].Size(), 1);
  ASSERT_EQ(jdoc["dto_vec"][0].GetBool(), true);
  ASSERT_EQ(jdoc["child"]["dto_vec"].Size(), 1);
  ASSERT_EQ(jdoc["child"]["dto_vec"][0].GetDouble(), 2.3);
  ASSERT_EQ(jdoc["ufields"]["a"].GetBool(), false);
  ASSERT_EQ(jdoc["fields"]["b"].GetBool(), true);
}
