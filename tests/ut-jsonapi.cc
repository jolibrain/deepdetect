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
#include <sys/stat.h>
#include <sys/types.h>
#include <iostream>

using namespace dd;

static std::string ok_str = "{\"status\":{\"code\":200,\"msg\":\"OK\"}}";
static std::string created_str = "{\"status\":{\"code\":201,\"msg\":\"Created\"}}";
static std::string bad_param_str = "{\"status\":{\"code\":400,\"msg\":\"BadRequest\"}}";
static std::string bad_request_str = "{\"status\":{\"code\":400,\"msg\":\"BadRequest\",\"dd_code\":1006,\"dd_msg\":\"Service Bad Request Error\"}}";
static std::string not_found_str = "{\"status\":{\"code\":404,\"msg\":\"NotFound\"}}";

TEST(jsonapi,service_delete)
{
  // fake model repository
  //  std::string here = "here";
  //  mkdir(here.c_str(),0777);
  
  // create service.
  JsonAPI japi;
  std::string sname = "my_service";
  std::string jstr = "{\"mllib\":\"caffe\",\"description\":\"my classifier\",\"type\":\"supervised\",\"model\":{\"repository\":\"here/\",\"create_repository\":true},\"parameters\":{\"input\":{\"connector\":\"image\"},\"mllib\":{\"nclasses\":2}}}";
  std::string joutstr = japi.jrender(japi.service_create(sname,jstr));
  ASSERT_EQ(created_str,joutstr);
  bool isdir = false;
  bool exists = fileops::file_exists("here", isdir);

  ASSERT_EQ(isdir, true);
  ASSERT_EQ(exists, true);
  //delete service.
  jstr = "{\"clear\":\"mem\"}";
  std::string jdelstr = japi.jrender(japi.service_delete(sname,jstr));
  ASSERT_EQ(ok_str,jdelstr);
  std::string jinfostr = japi.jrender(japi.info(""));
  JDoc jd;
  jd.Parse(jinfostr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_TRUE(jd.HasMember("status"));
  ASSERT_EQ(200,jd["status"]["code"]);
  ASSERT_EQ("OK",jd["status"]["msg"]);
  ASSERT_EQ(0,jd["head"]["services"].Size());
}

TEST(jsonapi,service_create)
{
  JsonAPI japi;
  std::string sname = "my_service";
  std::string jstr = "{\"mllib\":\"caffe\",\"description\":\"my classifier\",\"type\":\"supervised\",\"model\":{\"repository\":\"here\"},\"parameters\":{\"input\":{\"connector\":\"image\"},\"mllib\":{\"nclasses\":2}}}";
  std::string joutstr = japi.jrender(japi.service_create(sname,jstr));
  ASSERT_EQ(created_str,joutstr);
  std::string deljstr = "{\"clear\":\"mem\"}";
  std::string jdelstr = japi.jrender(japi.service_delete(sname,deljstr));
  ASSERT_EQ(ok_str,jdelstr);

  JDoc jd;
  jd.Parse(jstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  
  // service
  joutstr = japi.jrender(japi.service_create("",jstr));
  ASSERT_EQ(not_found_str,joutstr);
  jd.Parse(jstr.c_str());
  std::string jstrt = japi.jrender(jd);

  // mllib
  jd.RemoveMember("mllib");
  jstrt = japi.jrender(jd);
  joutstr = japi.jrender(japi.service_create(sname,jstrt));
  ASSERT_EQ(bad_param_str,joutstr);
  jd.Parse(jstr.c_str());

  // description
  jd.RemoveMember("description");
  jstrt = japi.jrender(jd);
  joutstr = japi.jrender(japi.service_create(sname,jstrt));
  ASSERT_EQ(created_str,joutstr);
  jd.Parse(jstr.c_str());
  deljstr = "{\"clear\":\"mem\"}";
  jdelstr = japi.jrender(japi.service_delete(sname,deljstr));
  ASSERT_EQ(ok_str,jdelstr);

  // for Caffe
  // model
  
  // input
  jd["parameters"].RemoveMember("input");
  jstrt = japi.jrender(jd);
  joutstr = japi.jrender(japi.service_create(sname,jstrt));
  ASSERT_EQ(bad_param_str,joutstr);
}

TEST(jsonapi,info)
{
  // create service
  JsonAPI japi;
  std::string sname = "my_service";
  std::string jstr = "{\"mllib\":\"caffe\",\"description\":\"my classifier\",\"type\":\"supervised\",\"model\":{\"repository\":\"here\"},\"parameters\":{\"input\":{\"connector\":\"image\"},\"mllib\":{\"nclasses\":2}}}";
  std::string joutstr = japi.jrender(japi.service_create(sname,jstr));
  ASSERT_EQ(created_str,joutstr);

  // info
  std::string jinfostr = japi.jrender(japi.info(""));
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

TEST(jsonapi,service_status)
{
  // create service.
  JsonAPI japi;
  std::string sname = "my_service";
  std::string jstr = "{\"mllib\":\"caffe\",\"description\":\"my classifier\",\"type\":\"supervised\",\"model\":{\"repository\":\"here\"},\"parameters\":{\"input\":{\"connector\":\"image\"},\"mllib\":{\"nclasses\":2}}}";
  std::string joutstr = japi.jrender(japi.service_create(sname,jstr));
  ASSERT_EQ(created_str,joutstr);

  // service status.
  std::string jstatstr = japi.jrender(japi.service_status(sname));
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

TEST(jsonapi, service_purge)
{
  JsonAPI japi;
  std::string sname = "my_service";
  std::string jstr = "{\"mllib\":\"caffe\",\"description\":\"my classifier\",\"type\":\"supervised\",\"model\":{\"repository\":\"here\"},\"parameters\":{\"input\":{\"connector\":\"image\"},\"mllib\":{\"nclasses\":2}}}";
  std::string joutstr = japi.jrender(japi.service_create(sname,jstr));
  ASSERT_EQ(created_str,joutstr);


  jstr = "{\"clear\":\"dir\"}";
  std::string jdelstr = japi.jrender(japi.service_delete(sname,jstr));
  ASSERT_EQ(ok_str,jdelstr);
  std::string jinfostr = japi.jrender(japi.info(""));
  JDoc jd;
  jd.Parse(jinfostr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_TRUE(jd.HasMember("status"));
  ASSERT_EQ(200,jd["status"]["code"]);
  ASSERT_EQ("OK",jd["status"]["msg"]);
  ASSERT_EQ(0,jd["head"]["services"].Size());
  bool isdir;
  bool exists = fileops::file_exists("here", isdir);
  ASSERT_EQ(exists, false);
}
