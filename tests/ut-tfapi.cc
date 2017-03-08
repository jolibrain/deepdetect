/**
 * DeepDetect
 * Copyright (c) 2016 Emmanuel Benazera
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

#include "deepdetect.h"
#include "jsonapi.h"
#include <gtest/gtest.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <iostream>

using namespace dd;

static std::string ok_str = "{\"status\":{\"code\":200,\"msg\":\"OK\"}}";
static std::string created_str = "{\"status\":{\"code\":201,\"msg\":\"Created\"}}";
static std::string bad_param_str = "{\"status\":{\"code\":400,\"msg\":\"BadRequest\"}}";
static std::string not_found_str = "{\"status\":{\"code\":404,\"msg\":\"NotFound\"}}";

static std::string incept_repo = "../examples/tf/inception/";

TEST(tfapi,service_predict)
{
  // create service
  JsonAPI japi;
  std::string sname = "imgserv";
  std::string jstr = "{\"mllib\":\"tensorflow\",\"description\":\"my classifier\",\"type\":\"supervised\",\"model\":{\"repository\":\"" +  incept_repo + "\"},\"parameters\":{\"input\":{\"connector\":\"image\",\"height\":224,\"width\":224,\"inputlayer\":\"InputImage\"},\"mllib\":{\"nclasses\":1001}}}";
  std::string joutstr = japi.jrender(japi.service_create(sname,jstr));
  ASSERT_EQ(created_str,joutstr);

  // predict
  std::string jpredictstr = "{\"service\":\"imgserv\",\"parameters\":{\"output\":{\"best\":3}},\"data\":[\"" + incept_repo + "grace_hopper.jpg\"]}";
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  JDoc jd;
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200,jd["status"]["code"]);
  ASSERT_TRUE(jd["body"]["predictions"].IsArray());
  std::string cl1 = jd["body"]["predictions"][0]["classes"][0]["cat"].GetString();
  ASSERT_TRUE(cl1 == "n03763968 military uniform");
  ASSERT_TRUE(jd["body"]["predictions"][0]["classes"][0]["prob"].GetDouble() > 0.4);

  // predict batch
  jpredictstr = "{\"service\":\"imgserv\",\"parameters\":{\"output\":{\"best\":3}},\"data\":[\"" + incept_repo + "grace_hopper.jpg\",\"" + incept_repo + "cat.jpg\"]}";
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200,jd["status"]["code"]);
  ASSERT_TRUE(jd["body"]["predictions"].IsArray());
  ASSERT_EQ(2,jd["body"]["predictions"].Size());
  cl1 = jd["body"]["predictions"][0]["classes"][0]["cat"].GetString();
  std::string cl2 = jd["body"]["predictions"][1]["classes"][0]["cat"].GetString();
  ASSERT_TRUE((cl1 == "n03763968 military uniform" && cl2 == "n02123045 tabby, tabby cat")
	      || (cl1 == "n02123045 tabby, tabby cat" && cl2 == "n03763968 military uniform"));
  ASSERT_TRUE(jd["body"]["predictions"][0]["classes"][0]["prob"].GetDouble() > 0.4);
  ASSERT_TRUE(jd["body"]["predictions"][1]["classes"][0]["prob"].GetDouble() > 0.4);
}

TEST(tfapi,service_predict_unsup)
{
  // create service
  JsonAPI japi;
  std::string sname = "imgserv";
  std::string jstr = "{\"mllib\":\"tensorflow\",\"description\":\"my classifier\",\"type\":\"unsupervised\",\"model\":{\"repository\":\"" +  incept_repo + "\"},\"parameters\":{\"input\":{\"connector\":\"image\",\"height\":224,\"width\":224,\"inputlayer\":\"InputImage\"},\"mllib\":{\"nclasses\":1001}}}";
  std::string joutstr = japi.jrender(japi.service_create(sname,jstr));
  ASSERT_EQ(created_str,joutstr);

  // predict
  std::string jpredictstr = "{\"service\":\"imgserv\",\"parameters\":{\"mllib\":{\"extract_layer\":\"InceptionV1/InceptionV1/Mixed_5c/concat\"}},\"data\":[\"" + incept_repo + "grace_hopper.jpg\"]}";
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  JDoc jd;
  //std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200,jd["status"]["code"]);
  ASSERT_EQ(50176,jd["body"]["predictions"][0]["vals"].Size());
}
