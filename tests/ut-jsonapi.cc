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

#include "deepdetect.h"
#include "jsonapi.h"
#include <gtest/gtest.h>
#include <iostream>

using namespace dd;

static std::string ok_str = "{\"status\":{\"code\":200,\"msg\":\"OK\"}}";
static std::string created_str = "{\"status\":{\"code\":201,\"msg\":\"Created\"}}";
static std::string bad_param_str = "{\"status\":{\"code\":400,\"msg\":\"BadRequest\"}}";
static std::string not_found_str = "{\"status\":{\"code\":404,\"msg\":\"NotFound\"}}";

TEST(jsonapi,service_create)
{
  JsonAPI japi;
  std::string sname = "my_service";
  std::string jstr = "{\"mllib\":\"caffe\",\"description\":\"my classifier\",\"type\":\"supervised\",\"model\":{\"repository\":\"here/\"},\"input\":\"image\"}";
  std::string joutstr = japi.service_create(sname,jstr);
  ASSERT_EQ(created_str,joutstr);

  JDoc jd;
  jd.Parse(jstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  
  // service
  joutstr = japi.service_create("",jstr);
  ASSERT_EQ(not_found_str,joutstr);
  jd.Parse(jstr.c_str());
  std::string jstrt = japi.jrender(jd);

  // mllib
  jd.RemoveMember("mllib");
  jstrt = japi.jrender(jd);
  joutstr = japi.service_create(sname,jstrt);
  ASSERT_EQ(bad_param_str,joutstr);
  jd.Parse(jstr.c_str());

  // description
  jd.RemoveMember("description");
  jstrt = japi.jrender(jd);
  joutstr = japi.service_create(sname,jstrt);
  ASSERT_EQ(created_str,joutstr);
  jd.Parse(jstr.c_str());

  // type
  jd.RemoveMember("type");
  jstrt = japi.jrender(jd);
  joutstr = japi.service_create(sname,jstrt);
  ASSERT_EQ(created_str,joutstr);
  jd.Parse(jstr.c_str());

  // for Caffe
  // model
  
  // input
  jd.RemoveMember("input");
  jstrt = japi.jrender(jd);
  joutstr = japi.service_create(sname,jstrt);
  ASSERT_EQ(bad_param_str,joutstr);
  jd.Parse(jstr.c_str());  
}

TEST(jsonapi,info)
{
  // create service
  JsonAPI japi;
  std::string sname = "my_service";
  std::string jstr = "{\"mllib\":\"caffe\",\"description\":\"my classifier\",\"type\":\"supervised\",\"model\":{\"repository\":\"here/\"},\"input\":\"image\"}";
  std::string joutstr = japi.service_create(sname,jstr);
  ASSERT_EQ(created_str,joutstr);

  // info
  std::string jinfostr = japi.info();
  //std::cout << "jinfostr=" << jinfostr << std::endl;
  JDoc jd;
  jd.Parse(jinfostr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_TRUE(jd.HasMember("status"));
  ASSERT_EQ(200,jd["status"]["code"]);
  ASSERT_EQ("OK",jd["status"]["msg"]);
  ASSERT_TRUE(jd.HasMember("head"));  
  ASSERT_TRUE(jd["head"].HasMember("version"));
  ASSERT_TRUE(jd["head"].HasMember("commit"));
  ASSERT_TRUE(jd["head"].HasMember("services"));
  ASSERT_EQ(1,jd["head"]["services"].Size());
  ASSERT_EQ("caffe",jd["head"]["services"][0]["mllib"]);
  ASSERT_EQ("my classifier",jd["head"]["services"][0]["description"]);
  ASSERT_EQ("my_service",jd["head"]["services"][0]["name"]);
}

TEST(jsonapi,service_delete)
{
  // create service.
  JsonAPI japi;
  std::string sname = "my_service";
  std::string jstr = "{\"mllib\":\"caffe\",\"description\":\"my classifier\",\"type\":\"supervised\",\"model\":{\"repository\":\"here/\"},\"input\":\"image\"}";
  std::string joutstr = japi.service_create(sname,jstr);
  ASSERT_EQ(created_str,joutstr);
  
  // delete service.
  std::string jdelstr = japi.service_delete(sname);
  ASSERT_EQ(ok_str,jdelstr);
  std::string jinfostr = japi.info();
  JDoc jd;
  jd.Parse(jinfostr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_TRUE(jd.HasMember("status"));
  ASSERT_EQ(200,jd["status"]["code"]);
  ASSERT_EQ("OK",jd["status"]["msg"]);
  ASSERT_EQ(0,jd["head"]["services"].Size());
}

TEST(jsonapi,service_status)
{
  // create service.
  JsonAPI japi;
  std::string sname = "my_service";
  std::string jstr = "{\"mllib\":\"caffe\",\"description\":\"my classifier\",\"type\":\"supervised\",\"model\":{\"repository\":\"here/\"},\"input\":\"image\"}";
  std::string joutstr = japi.service_create(sname,jstr);
  ASSERT_EQ(created_str,joutstr);

  // service status.
  std::string jstatstr = japi.service_status(sname);
  //std::cout << "jstatstr=" << jstatstr << std::endl;
  JDoc jd;
  jd.Parse(jstatstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_TRUE(jd.HasMember("status"));
  ASSERT_EQ(200,jd["status"]["code"]);
  ASSERT_EQ("OK",jd["status"]["msg"]);
  ASSERT_TRUE(jd.HasMember("body"));
  ASSERT_TRUE(jd["body"].HasMember("description"));
}
