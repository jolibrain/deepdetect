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
#include <gtest/gtest.h>
#include <iostream>

using namespace dd;

TEST(apidata,visitor_vad)
{
  double loss = 1.17;
  double prob1 = 0.67;
  double prob2 = 0.29;
  
  APIData ad;
  std::vector<APIData> vad;
  APIData ivad1;
  ivad1.add("cat","car");
  ivad1.add("prob",prob1);
  vad.push_back(ivad1);
  APIData ivad2;
  ivad2.add("cat","wolf");
  ivad2.add("prob",prob2);
  vad.push_back(ivad2);
  ad.add("classes",vad);
  ad.add("loss",loss);

  std::vector<APIData> ad_cl = ad.getv("classes");
  std::cout << "prob=" << ad_cl.at(0).get("prob").get<double>() << std::endl;
  ASSERT_EQ(prob1,ad_cl.at(0).get("prob").get<double>());
}

TEST(apidata,to_from_json)
{
  double prob1 = 0.67;
  double prob2 = 0.29;
  JDoc jd;
  jd.SetObject();

  // to JSON
  APIData ad;
  ad.add("string","string");
  ad.add("double",2.3);
  ad.add("bool",true);
  std::vector<double> vd = {1.1,2.2,3.3};
  ad.add("vdouble",vd);
  std::vector<std::string> vs = {"one","two","three"};
  ad.add("vstring",vs);
  std::vector<APIData> vad;
  APIData ivad1;
  ivad1.add("cat","car");
  ivad1.add("prob",prob1);
  vad.push_back(ivad1);
  APIData ivad2;
  ivad2.add("cat","wolf");
  ivad2.add("prob",prob2);
  vad.push_back(ivad2);
  ad.add("classes",vad);
  ad.toJDoc(jd);
  JsonAPI japi;
  std::string jrstr = japi.jrender(jd);
  std::cout << jrstr << std::endl;
  ASSERT_TRUE(jd["string"].GetString()==std::string("string"));
  ASSERT_EQ(2.3,jd["double"]);
  ASSERT_EQ(true,jd["bool"]);
  ASSERT_TRUE(jd["vdouble"].IsArray());
  ASSERT_EQ(1.1,jd["vdouble"][0]);
  ASSERT_TRUE(jd["vstring"].IsArray());
  ASSERT_TRUE(jd["vstring"][1].GetString()==std::string("two"));
  ASSERT_TRUE(jd["classes"].IsArray());
  ASSERT_TRUE(jd["classes"][0]["cat"].GetString()==std::string("car"));
  ASSERT_EQ(prob1,jd["classes"][0]["prob"].GetDouble());

  // to APIData
  APIData nad(jd);
  ASSERT_EQ("string",nad.get("string").get<std::string>());
  ASSERT_EQ(2.3,nad.get("double").get<double>());
  ASSERT_EQ(true,nad.get("bool").get<bool>());
  ASSERT_EQ(3,nad.get("vdouble").get<std::vector<double>>().size());
  ASSERT_EQ(2.2,nad.get("vdouble").get<std::vector<double>>().at(1));
  ASSERT_EQ(3,nad.get("vstring").get<std::vector<std::string>>().size());
  ASSERT_EQ("two",nad.get("vstring").get<std::vector<std::string>>().at(1));
  
  // and back to JSON for comparison
  JDoc njd;
  njd.SetObject();
  nad.toJDoc(njd);
  ASSERT_TRUE(njd["string"].GetString()==std::string("string"));
  ASSERT_EQ(2.3,njd["double"]);
  ASSERT_EQ(true,njd["bool"]);
  ASSERT_TRUE(njd["vdouble"].IsArray());
  ASSERT_EQ(1.1,njd["vdouble"][0]);
  ASSERT_TRUE(njd["vstring"].IsArray());
  ASSERT_TRUE(njd["vstring"][1].GetString()==std::string("two"));
  ASSERT_TRUE(njd["classes"].IsArray());
  ASSERT_TRUE(njd["classes"][0]["cat"].GetString()==std::string("car"));
  ASSERT_EQ(prob1,njd["classes"][0]["prob"].GetDouble());
}


