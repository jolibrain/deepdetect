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

#include OATPP_CODEGEN_END(DTO) ///< End DTO codegen section

TEST(dto, vector_dto)
{
  auto vec_test = VectorDTOTest::createShared();
  ASSERT_TRUE(vec_test->dto_vec != nullptr);

  int size = 1000;
  for (int i = 0; i < size; ++i)
    {
      vec_test->dto_vec->_vec.push_back(i);
    }
  ASSERT_EQ(vec_test->dto_vec->_vec.size(), size);
  ASSERT_EQ(vec_test->dto_vec->_vec[567], 567.0);
  ASSERT_EQ(vec_test->dto_vec->_vec[size - 1], size - 1.0);

  auto mapper = dd::oatpp_utils::createDDMapper();
  oatpp::String saved = mapper->writeToString(vec_test);
  auto vec_test2 = mapper->readFromString<oatpp::Object<VectorDTOTest>>(saved);

  ASSERT_EQ(vec_test2->dto_vec->_vec.size(), size);
  ASSERT_EQ(vec_test2->dto_vec->_vec[567], 567.0);
  ASSERT_EQ(vec_test2->dto_vec->_vec[size - 1], size - 1.0);
}
