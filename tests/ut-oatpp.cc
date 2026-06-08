/**
 * DeepDetect
 * Copyright (c) 2020 Jolibrain SASU
 * Author: Mehdi Abaakouk <mabaakouk@jolibrain.com>
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

#include "oatpp-test/UnitTest.hpp"

#include "ut-oatpp.h"

const std::string serv
    = "very_long_label_service_name_with_😀_inside_and_some_MAJ";
const std::string serv_lower
    = "very_long_label_service_name_with_😀_inside_and_some_maj";
const std::string serv2 = "myserv2";
#if defined(CPU_ONLY)
static std::string iterations_mnist = "10";
#else
static std::string iterations_mnist = "10000";
#endif

void test_info(std::shared_ptr<DedeApiTestClient> client)
{
  auto response = client->get_info();
  ASSERT_EQ(response->getStatusCode(), 200);
  auto message = response->readBodyToString();
  ASSERT_TRUE(message != nullptr);
  std::cout << "jstr=" << *message << std::endl;

  rapidjson::Document d;
  d.Parse<rapidjson::kParseNanAndInfFlag>(message->c_str());
  ASSERT_FALSE(d.HasParseError());
  ASSERT_FALSE(d.HasMember("status"));
  ASSERT_TRUE(d.HasMember("head"));
  ASSERT_TRUE(d["head"].HasMember("services"));
  ASSERT_EQ(0, d["head"]["services"].Size());
}

void test_services(std::shared_ptr<DedeApiTestClient> client)
{
  std::string serv_put
      = "{\"mllib\":\"torch\",\"description\":\"image "
        "classification\",\"type\":"
        "\"supervised\",\"model\":{\"repository\":\"./examples/"
        "oatpp_test_service\",\"create_repository\":"
        "true},\"parameters\":{\"input\":{"
        "\"connector\":\"image\",\"width\":224,\"height\":224,\"db\":true},"
        "\"mllib\":{\"nclasses\":2,\"gpu\":true,\"template\":\"resnet18\"}}"
        "}";

  auto response = client->put_services(serv.c_str(), serv_put.c_str());
  auto message = response->readBodyToString();
  ASSERT_TRUE(message != nullptr);
  std::cout << "jstr=" << *message << std::endl;
  ASSERT_EQ(response->getStatusCode(), 201);

  rapidjson::Document d;
  d.Parse<rapidjson::kParseNanAndInfFlag>(message->c_str());
  ASSERT_FALSE(d.HasParseError());
  ASSERT_TRUE(d.HasMember("status"));
  ASSERT_EQ(201, d["status"]["code"].GetInt());

  // service info
  response = client->get_service_with_labels(serv.c_str(), "true");
  message = response->readBodyToString();
  ASSERT_TRUE(message != nullptr);
  std::cout << "jstr=" << *message << std::endl;
  ASSERT_EQ(response->getStatusCode(), 200);
  d = rapidjson::Document();
  d.Parse<rapidjson::kParseNanAndInfFlag>(message->c_str());
  ASSERT_TRUE(d.HasMember("status"));
  ASSERT_TRUE(d.HasMember("body"));
  ASSERT_STREQ("torch", d["body"]["mllib"].GetString());
  ASSERT_STREQ(serv_lower.c_str(), d["body"]["name"].GetString());
  ASSERT_TRUE(d["body"].HasMember("jobs"));
  ASSERT_TRUE(d["body"].HasMember("model_stats"));
  ASSERT_TRUE(d["body"]["model_stats"]["params"].GetInt() == 11177538);
  ASSERT_TRUE(d["body"].HasMember("parameters"));
  ASSERT_TRUE(d["body"]["parameters"].HasMember("input"));
  ASSERT_TRUE(d["body"]["parameters"].HasMember("mllib"));
  ASSERT_TRUE(d["body"]["parameters"].HasMember("output"));
  ASSERT_EQ(d["body"]["parameters"]["input"]["connector"].GetString(),
            std::string("image"));
  ASSERT_TRUE(d["body"].HasMember("labels"));
  ASSERT_EQ(d["body"]["labels"].Size(), 0);

  // test other boolean values
  response = client->get_service_with_labels(serv.c_str(), "1");
  ASSERT_EQ(response->getStatusCode(), 200);
  response = client->get_service_with_labels(serv.c_str(), "false");
  ASSERT_EQ(response->getStatusCode(), 200);
  response = client->get_service_with_labels(serv.c_str(), "0");
  ASSERT_EQ(response->getStatusCode(), 200);
  response = client->get_service_with_labels(serv.c_str(), "flase");
  ASSERT_EQ(response->getStatusCode(), 400);

  // info call
  response = client->get_info();
  message = response->readBodyToString();
  ASSERT_TRUE(message != nullptr);
  std::cout << "jstr=" << *message << std::endl;
  ASSERT_EQ(response->getStatusCode(), 200);
  d = rapidjson::Document();
  d.Parse<rapidjson::kParseNanAndInfFlag>(message->c_str());
  ASSERT_TRUE(d["head"].HasMember("services"));
  ASSERT_EQ(1, d["head"]["services"].Size());

  // delete call
  response = client->delete_services(serv.c_str(), nullptr);
  ASSERT_EQ(response->getStatusCode(), 200);

  // info call
  response = client->get_info();
  message = response->readBodyToString();
  ASSERT_TRUE(message != nullptr);
  std::cout << "jstr=" << *message << std::endl;
  ASSERT_EQ(response->getStatusCode(), 200);
  d = rapidjson::Document();
  d.Parse<rapidjson::kParseNanAndInfFlag>(message->c_str());
  ASSERT_TRUE(d["head"].HasMember("services"));
  ASSERT_EQ(0, d["head"]["services"].Size());

  // service info
  response = client->get_services(serv.c_str());
  message = response->readBodyToString();
  ASSERT_TRUE(message != nullptr);
  std::cout << "jstr=" << *message << std::endl;
  ASSERT_EQ(response->getStatusCode(), 404);
  d = rapidjson::Document();
  d.Parse<rapidjson::kParseNanAndInfFlag>(message->c_str());
}

#define OATPP_DEDE_TEST(FUNC)                                                 \
  TEST(oatpp_jsonapi, FUNC)                                                   \
  {                                                                           \
    oatpp::base::Environment::init();                                         \
    DedeControllerTest *test = new DedeControllerTest(#FUNC, FUNC);           \
    test->run(1);                                                             \
    delete test;                                                              \
    oatpp::base::Environment::destroy();                                      \
  }

OATPP_DEDE_TEST(test_info);

#ifdef USE_TORCH

OATPP_DEDE_TEST(test_services);

#endif
