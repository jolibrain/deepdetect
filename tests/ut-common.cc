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

#include "utils/utils.hpp"

using namespace dd;

TEST(common, trim_spaces)
{
  ASSERT_EQ("test_name", dd_utils::trim_spaces("\n test_name"));
  ASSERT_EQ("test_name", dd_utils::trim_spaces("test_name\t"));
  ASSERT_EQ("test_name", dd_utils::trim_spaces("   test_name "));
  ASSERT_EQ("test_name", dd_utils::trim_spaces("test_name"));
  ASSERT_EQ("test_name test_name",
            dd_utils::trim_spaces("  test_name test_name\t"));
  ASSERT_EQ("", dd_utils::trim_spaces("   \n  "));
}
