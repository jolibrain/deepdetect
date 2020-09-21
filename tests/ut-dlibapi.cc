/**
 * DeepDetect
 * Copyright (c) 2018 Pixel Forensics, Inc.
 * Author: Cheni Chadowitz <cchadowitz@pixelforensics.com>
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
static std::string created_str
    = "{\"status\":{\"code\":201,\"msg\":\"Created\"}}";
static std::string bad_param_str
    = "{\"status\":{\"code\":400,\"msg\":\"BadRequest\"}}";
static std::string not_found_str
    = "{\"status\":{\"code\":404,\"msg\":\"NotFound\"}}";

static std::string face_repo = "../examples/dlib/face/";
static std::string obj_repo = "../examples/dlib/obj/";

TEST(dlibapi, service_predict_face)
{
  // create service
  JsonAPI japi;
  std::string sname = "imgserv";
  std::string jstr
      = "{\"mllib\":\"dlib\",\"description\":\"my face "
        "classifier\",\"type\":\"supervised\",\"model\":{\"repository\":\""
        + face_repo
        + "\"},\"parameters\":{\"input\":{\"connector\":\"image\", \"width\": "
          "512, \"height\": "
          "600},\"mllib\":{\"model_type\":\"face_detector\"}}}";
  std::string joutstr = japi.jrender(japi.service_create(sname, jstr));
  ASSERT_EQ(created_str, joutstr);

  // predict
  std::string jpredictstr = "{\"service\":\"imgserv\",\"parameters\":{"
                            "\"output\":{\"bbox\":true}},\"data\":[\""
                            + face_repo + "grace_hopper.jpg\"]}";
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  JDoc jd;
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200, jd["status"]["code"]);
  ASSERT_TRUE(jd["body"]["predictions"].IsArray());
  std::string cl1
      = jd["body"]["predictions"][0]["classes"][0]["cat"].GetString();
  ASSERT_EQ("1", cl1); // default label is "1" when no label is provided
  ASSERT_TRUE(jd["body"]["predictions"][0]["classes"][0]["prob"].GetDouble()
              > 0.4);
  ASSERT_TRUE(jd["body"]["predictions"][0]["classes"][0].HasMember("bbox"));
  ASSERT_TRUE(
      jd["body"]["predictions"][0]["classes"][0]["bbox"].HasMember("xmin"));

  // predict batch
  jpredictstr = "{\"service\":\"imgserv\",\"parameters\":{\"output\":{"
                "\"bbox\":true}},\"data\":[\""
                + face_repo + "grace_hopper.jpg\",\"" + face_repo
                + "cat.jpg\"]}";
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200, jd["status"]["code"]);
  ASSERT_TRUE(jd["body"]["predictions"].IsArray());
  ASSERT_EQ(2,
            jd["body"]["predictions"]
                .Size()); // Ensure we got a predictions obj back for each file

  int grace_hopper_idx = 0, cat_idx = 1;
  if (jd["body"]["predictions"][0]["classes"].Size() == 0)
    { // If the first result happens to be cat.jpg, check the other one
      grace_hopper_idx = 1;
      cat_idx = 0;
    }
  ASSERT_TRUE(jd["body"]["predictions"][grace_hopper_idx]["classes"].Size()
              > 0);
  cl1 = jd["body"]["predictions"][grace_hopper_idx]["classes"][0]["cat"]
            .GetString();
  ASSERT_EQ("1", cl1); // default label is "1" when no label is provided
  ASSERT_TRUE(jd["body"]["predictions"][grace_hopper_idx]["classes"][0]["prob"]
                  .GetDouble()
              > 0.4);
  ASSERT_TRUE(
      jd["body"]["predictions"][grace_hopper_idx]["classes"][0].HasMember(
          "bbox"));
  ASSERT_TRUE(jd["body"]["predictions"][grace_hopper_idx]["classes"][0]["bbox"]
                  .HasMember("xmin"));
  // There's no face in cat.jpg, so no result for it
  ASSERT_TRUE(jd["body"]["predictions"][cat_idx]["classes"].IsArray());
  ASSERT_EQ(0, jd["body"]["predictions"][cat_idx]["classes"].Size());
}

TEST(dlibapi, service_predict_obj)
{
  // create service
  JsonAPI japi;
  std::string sname = "imgserv";
  std::string jstr
      = "{\"mllib\":\"dlib\",\"description\":\"my obj "
        "classifier\",\"type\":\"supervised\",\"model\":{\"repository\":\""
        + obj_repo
        + "\"},\"parameters\":{\"input\":{\"connector\":\"image\", \"width\": "
          "1024, \"height\": "
          "254},\"mllib\":{\"model_type\":\"obj_detector\"}}}";
  std::string joutstr = japi.jrender(japi.service_create(sname, jstr));
  ASSERT_EQ(created_str, joutstr);

  // predict
  std::string jpredictstr = "{\"service\":\"imgserv\",\"parameters\":{"
                            "\"output\":{\"bbox\":true}},\"data\":[\""
                            + obj_repo + "mmod_cars_test_image2.jpg\"]}";
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  JDoc jd;
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200, jd["status"]["code"]);
  ASSERT_TRUE(jd["body"]["predictions"].IsArray());
  std::string cl1
      = jd["body"]["predictions"][0]["classes"][0]["cat"].GetString();
  ASSERT_TRUE(cl1 == "rear");
  ASSERT_TRUE(jd["body"]["predictions"][0]["classes"][0]["prob"].GetDouble()
              > 0.4);
  ASSERT_TRUE(jd["body"]["predictions"][0]["classes"][0].HasMember("bbox"));
  ASSERT_TRUE(
      jd["body"]["predictions"][0]["classes"][0]["bbox"].HasMember("xmin"));
  ASSERT_EQ(5, jd["body"]["predictions"][0]["classes"].Size());

  // predict batch
  jpredictstr = "{\"service\":\"imgserv\",\"parameters\":{\"output\":{"
                "\"bbox\":true}},\"data\":[\""
                + obj_repo + "mmod_cars_test_image2.jpg\",\"" + face_repo
                + "cat.jpg\"]}";
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200, jd["status"]["code"]);
  ASSERT_TRUE(jd["body"]["predictions"].IsArray());
  ASSERT_EQ(2,
            jd["body"]["predictions"]
                .Size()); // Ensure we got a predictions obj back for each file
  int vehicle_idx = 0, cat_idx = 1;
  if (jd["body"]["predictions"][0]["classes"].Size() == 0)
    { // If the first result happens to be cat.jpg, check the other one
      vehicle_idx = 1;
      cat_idx = 0;
    }
  // There's no front or rear vehicle in cat.jpg, so no result for it
  ASSERT_TRUE(jd["body"]["predictions"][cat_idx]["classes"].IsArray());
  ASSERT_EQ(0, jd["body"]["predictions"][cat_idx]["classes"].Size());

  ASSERT_TRUE(jd["body"]["predictions"][vehicle_idx]["classes"].Size() > 0);
  cl1 = jd["body"]["predictions"][vehicle_idx]["classes"][0]["cat"]
            .GetString();
  ASSERT_TRUE(cl1 == "rear");
  ASSERT_TRUE(
      jd["body"]["predictions"][vehicle_idx]["classes"][0]["prob"].GetDouble()
      > 0.4);
  ASSERT_TRUE(
      jd["body"]["predictions"][vehicle_idx]["classes"][0].HasMember("bbox"));
  ASSERT_TRUE(
      jd["body"]["predictions"][vehicle_idx]["classes"][0]["bbox"].HasMember(
          "xmin"));
}
