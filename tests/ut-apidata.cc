/**
 * DeepDetect
 * Copyright (c) 2014 Emmanuel Benazera
 * Author: Emmanuel Benazera <beniz@droidnik.fr>
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

#include "apidata.h"
#include "jsonapi.h"
#include "supervisedoutputconnector.h"
#include <gtest/gtest.h>
#include <iostream>

using namespace dd;

TEST(apidata, visitor_vad)
{
  double loss = 1.17;
  double prob1 = 0.67;
  double prob2 = 0.29;

  APIData ad;
  std::vector<APIData> vad;
  APIData ivad1;
  ivad1.add("cat", std::string("car"));
  ivad1.add("prob", prob1);
  vad.push_back(ivad1);
  APIData ivad2;
  ivad2.add("cat", std::string("wolf"));
  ivad2.add("prob", prob2);
  vad.push_back(ivad2);
  ad.add("classes", vad);
  ad.add("loss", loss);
  APIData tad;
  tad.add("test", 1);
  ad.add("tad", tad);

  std::vector<APIData> ad_cl = ad.getv("classes");
  std::cout << "prob=" << ad_cl.at(0).get("prob").get<double>() << std::endl;
  ASSERT_EQ(prob1, ad_cl.at(0).get("prob").get<double>());
  ASSERT_EQ(1, ad.getobj("tad").get("test").get<int>());
}

TEST(apidata, to_from_json)
{
  double prob1 = 0.67;
  double prob2 = 0.29;
  JDoc jd;
  jd.SetObject();

  // to JSON
  APIData ad;
  ad.add("string", std::string("string"));
  ad.add("double", 2.3);
  ad.add("int", 3);
  ad.add("bool", true);
  std::vector<double> vd = { 1.1, 2.2, 3.3 };
  ad.add("vdouble", vd);
  std::vector<std::string> vs = { "one", "two", "three" };
  ad.add("vstring", vs);
  std::vector<APIData> vad;
  APIData ivad1;
  ivad1.add("cat", std::string("car"));
  ivad1.add("prob", prob1);
  vad.push_back(ivad1);
  APIData ivad2;
  ivad2.add("cat", std::string("wolf"));
  ivad2.add("prob", prob2);
  vad.push_back(ivad2);
  ad.add("classes", vad);
  APIData tad;
  tad.add("test", 1);
  ad.add("tad", tad);
  ad.toJDoc(jd);
  JsonAPI japi;
  std::string jrstr = japi.jrender(jd);
  std::cout << jrstr << std::endl;
  ASSERT_TRUE(jd["string"].GetString() == std::string("string"));
  ASSERT_EQ(2.3, jd["double"].GetDouble());
  ASSERT_EQ(3, jd["int"].GetInt());
  ASSERT_EQ(true, jd["bool"].GetBool());
  ASSERT_TRUE(jd["vdouble"].IsArray());
  ASSERT_EQ(1.1, jd["vdouble"][0].GetDouble());
  ASSERT_TRUE(jd["vstring"].IsArray());
  ASSERT_TRUE(jd["vstring"][1].GetString() == std::string("two"));
  ASSERT_TRUE(jd["classes"].IsArray());
  ASSERT_TRUE(jd["classes"][0]["cat"].GetString() == std::string("car"));
  ASSERT_EQ(prob1, jd["classes"][0]["prob"].GetDouble());

  // to APIData
  APIData nad;
  nad.fromRapidJson(jd);
  ASSERT_EQ("string", nad.get("string").get<std::string>());
  ASSERT_EQ(2.3, nad.get("double").get<double>());
  ASSERT_EQ(true, nad.get("bool").get<bool>());
  ASSERT_EQ(3, nad.get("vdouble").get<std::vector<double>>().size());
  ASSERT_EQ(2.2, nad.get("vdouble").get<std::vector<double>>().at(1));
  ASSERT_EQ(3, nad.get("vstring").get<std::vector<std::string>>().size());
  ASSERT_EQ("two", nad.get("vstring").get<std::vector<std::string>>().at(1));

  // and back to JSON for comparison
  JDoc njd;
  njd.SetObject();
  nad.toJDoc(njd);
  ASSERT_TRUE(njd["string"].GetString() == std::string("string"));
  ASSERT_EQ(2.3, njd["double"].GetDouble());
  ASSERT_EQ(3, njd["int"].GetInt());
  ASSERT_EQ(true, njd["bool"].GetBool());
  ASSERT_TRUE(njd["vdouble"].IsArray());
  ASSERT_EQ(1.1, njd["vdouble"][0].GetDouble());
  ASSERT_TRUE(njd["vstring"].IsArray());
  ASSERT_TRUE(njd["vstring"][1].GetString() == std::string("two"));
  ASSERT_TRUE(njd["classes"].IsArray());
  ASSERT_TRUE(njd["classes"][0]["cat"].GetString() == std::string("car"));
  ASSERT_EQ(prob1, njd["classes"][0]["prob"].GetDouble());
}

TEST(apidata, supervised_output_keypoints_to_dto)
{
  SupervisedOutput output;

  APIData point0;
  point0.add("x", 1.0);
  point0.add("y", 2.0);
  point0.add("prob", 0.8);
  point0.add("valid", true);
  APIData point1;
  point1.add("x", -1.0);
  point1.add("y", -1.0);
  point1.add("prob", 0.0);
  point1.add("valid", false);
  std::vector<APIData> points = { point0, point1 };
  APIData keypoints;
  keypoints.add("points", points);

  APIData result;
  result.add("uri", std::string("image.jpg"));
  result.add("loss", 0.0);
  result.add("probs", std::vector<double>{ 0.9 });
  result.add("cats", std::vector<std::string>{ "pose" });
  result.add("keypoints", std::vector<APIData>{ keypoints });
  output.add_results(std::vector<APIData>{ result });

  auto output_params = DTO::OutputConnector::createShared();
  output_params->keypoints = true;
  OutputConnectorConfig config;
  config._nclasses = 1;
  config._has_keypoints = true;
  auto dto = output.finalize(output_params, config, nullptr);

  ASSERT_EQ(1U, dto->predictions->size());
  ASSERT_EQ(1U, dto->predictions->at(0)->classes->size());
  auto cls = dto->predictions->at(0)->classes->at(0);
  ASSERT_EQ(std::string("pose"), cls->cat.getValue(""));
  ASSERT_TRUE(cls->keypoints != nullptr);
  ASSERT_EQ(2U, cls->keypoints->size());
  EXPECT_DOUBLE_EQ(1.0, cls->keypoints->at(0)->get("x").get<double>());
  EXPECT_DOUBLE_EQ(2.0, cls->keypoints->at(0)->get("y").get<double>());
  EXPECT_TRUE(cls->keypoints->at(0)->get("valid").get<bool>());
  EXPECT_DOUBLE_EQ(-1.0, cls->keypoints->at(1)->get("x").get<double>());
  EXPECT_FALSE(cls->keypoints->at(1)->get("valid").get<bool>());
}
