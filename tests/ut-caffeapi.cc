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
#include <stdio.h>
#include <unistd.h>
#include <iostream>

using namespace dd;

static std::string ok_str = "{\"status\":{\"code\":200,\"msg\":\"OK\"}}";
static std::string created_str = "{\"status\":{\"code\":201,\"msg\":\"Created\"}}";
static std::string bad_param_str = "{\"status\":{\"code\":400,\"msg\":\"BadRequest\"}}";
static std::string not_found_str = "{\"status\":{\"code\":404,\"msg\":\"NotFound\"}}";

static std::string mnist_repo = "../examples/caffe/mnist/";

TEST(caffeapi,service_train)
{
::google::InitGoogleLogging("ut_caffeapi");
// create service
  JsonAPI japi;
  std::string sname = "my_service";
  std::string jstr = "{\"mllib\":\"caffe\",\"description\":\"my classifier\",\"type\":\"supervised\",\"model\":{\"repository\":\"" +  mnist_repo + "\"},\"parameters\":{\"input\":{\"connector\":\"image\"}}}";
  std::string joutstr = japi.service_create(sname,jstr);
  ASSERT_EQ(created_str,joutstr);

  // train
  std::string jtrainstr = "{\"service\":\"" + sname + "\",\"async\":false,\"parameters\":{\"mllib\":{\"solver\":{\"iterations\":100}}}}";
  joutstr = japi.service_train(jtrainstr);
  std::cout << "joutstr=" << joutstr << std::endl;
  JDoc jd;
  jd.Parse(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_TRUE(jd.HasMember("status"));
  ASSERT_EQ(201,jd["status"]["code"].GetInt());
  ASSERT_EQ("Created",jd["status"]["msg"]);
  ASSERT_TRUE(jd.HasMember("head"));
  ASSERT_EQ("/train",jd["head"]["method"]);
  ASSERT_TRUE(jd["head"]["time"].GetDouble() > 0);
  ASSERT_TRUE(jd.HasMember("body"));
  ASSERT_TRUE(jd["body"]["measure"].HasMember("train_loss"));
  ASSERT_TRUE(jd["body"]["measure"]["train_loss"].GetDouble() > 0);
}

TEST(caffeapi,service_train_async_status_delete)
{
  // create service
  JsonAPI japi;
  std::string sname = "my_service";
  std::string jstr = "{\"mllib\":\"caffe\",\"description\":\"my classifier\",\"type\":\"supervised\",\"model\":{\"repository\":\"" +  mnist_repo + "\"},\"parameters\":{\"input\":{\"connector\":\"image\"}}}";
  std::string joutstr = japi.service_create(sname,jstr);
  ASSERT_EQ(created_str,joutstr);

  // train
  std::string jtrainstr = "{\"service\":\"" + sname + "\",\"async\":true,\"parameters\":{\"mllib\":{\"solver\":{\"iterations\":10000}}}}";
  joutstr = japi.service_train(jtrainstr);
  std::cout << "joutstr=" << joutstr << std::endl;
  JDoc jd;
  jd.Parse(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_TRUE(jd.HasMember("status"));
  ASSERT_EQ(201,jd["status"]["code"]);
  ASSERT_EQ("Created",jd["status"]["msg"]);
  ASSERT_TRUE(jd.HasMember("head"));
  ASSERT_EQ("/train",jd["head"]["method"]);
  ASSERT_EQ(1,jd["head"]["job"].GetInt());
  ASSERT_EQ("running",jd["head"]["status"]);
  
  // status.
  std::string jstatusstr = "{\"service\":\"" + sname + "\",\"job\":1,\"timeout\":5}";
  joutstr = japi.service_train_status(jstatusstr);
  std::cout << "status joutstr=" << joutstr << std::endl;
  JDoc jd2;
  jd2.Parse(joutstr.c_str());
  ASSERT_TRUE(!jd2.HasParseError());
  ASSERT_TRUE(jd2.HasMember("status"));
  ASSERT_EQ(200,jd2["status"]["code"]);
  ASSERT_EQ("OK",jd2["status"]["msg"]);
  ASSERT_TRUE(jd2.HasMember("head"));
  ASSERT_EQ("/train",jd2["head"]["method"]);
  ASSERT_EQ(5.0,jd2["head"]["time"].GetDouble());
  ASSERT_EQ("running",jd2["head"]["status"]);
  ASSERT_EQ(1,jd2["head"]["job"]);
  ASSERT_TRUE(jd2.HasMember("body"));
  ASSERT_TRUE(jd2["body"]["measure"].HasMember("train_loss"));
  ASSERT_TRUE(jd2["body"]["measure"]["train_loss"].GetDouble() >= 0);

  // delete job.
  std::string jdelstr = "{\"service\":\"" + sname + "\",\"job\":1}";
  joutstr = japi.service_train_delete(jdelstr);
  std::cout << "joutstr=" << joutstr << std::endl;
  JDoc jd3;
  jd3.Parse(joutstr.c_str());
  ASSERT_TRUE(!jd3.HasParseError());
  ASSERT_TRUE(jd3.HasMember("status"));
  ASSERT_EQ(200,jd3["status"]["code"]);
  ASSERT_EQ("OK",jd3["status"]["msg"]);
  ASSERT_TRUE(jd3.HasMember("head"));
  ASSERT_EQ("/train",jd3["head"]["method"]);
  ASSERT_TRUE(jd3["head"]["time"].GetDouble() > 0);
  ASSERT_EQ("terminated",jd3["head"]["status"]);
  ASSERT_EQ(1,jd3["head"]["job"].GetInt());
}

TEST(caffeapi,service_train_async_final_status)
{
  // create service
  JsonAPI japi;
  std::string sname = "my_service";
  std::string jstr = "{\"mllib\":\"caffe\",\"description\":\"my classifier\",\"type\":\"supervised\",\"model\":{\"repository\":\"" +  mnist_repo + "\"},\"parameters\":{\"input\":{\"connector\":\"image\"}}}";
  std::string joutstr = japi.service_create(sname,jstr);
  ASSERT_EQ(created_str,joutstr);

  // train
  std::string jtrainstr = "{\"service\":\"" + sname + "\",\"async\":true,\"parameters\":{\"mllib\":{\"solver\":{\"iterations\":250}}}}";
  joutstr = japi.service_train(jtrainstr);
  std::cout << "joutstr=" << joutstr << std::endl;
  JDoc jd;
  jd.Parse(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_TRUE(jd.HasMember("status"));
  ASSERT_EQ(201,jd["status"]["code"]);
  ASSERT_EQ("Created",jd["status"]["msg"]);
  ASSERT_TRUE(jd.HasMember("head"));
  ASSERT_EQ("/train",jd["head"]["method"]);
  ASSERT_EQ(1,jd["head"]["job"].GetInt());
  ASSERT_EQ("running",jd["head"]["status"]);
  
  // status.
  bool running = true;
  while(running)
    {
      //sleep(1);
      std::string jstatusstr = "{\"service\":\"" + sname + "\",\"job\":1,\"timeout\":1}";
      joutstr = japi.service_train_status(jstatusstr);
      std::cout << "joutstr=" << joutstr << std::endl;
      running = joutstr.find("running") != std::string::npos;
      if (!running)
	{
	  JDoc jd2;
	  jd2.Parse(joutstr.c_str());
	  ASSERT_TRUE(!jd2.HasParseError());
	  ASSERT_TRUE(jd2.HasMember("status"));
	  ASSERT_EQ(200,jd2["status"]["code"]);
	  ASSERT_EQ("OK",jd2["status"]["msg"]);
	  ASSERT_TRUE(jd2.HasMember("head"));
	  ASSERT_EQ("/train",jd2["head"]["method"]);
	  ASSERT_TRUE(jd2["head"]["time"].GetDouble() > 0);
	  ASSERT_EQ("finished",jd2["head"]["status"]);
	  ASSERT_EQ(1,jd2["head"]["job"]);
	  ASSERT_TRUE(jd2.HasMember("body"));
	  ASSERT_TRUE(jd2["body"]["measure"].HasMember("train_loss"));
	  ASSERT_TRUE(jd2["body"]["measure"]["train_loss"].GetDouble() > 0);
	}
    }
}

TEST(caffeapi,service_predict)
{
  // create service
  JsonAPI japi;
  std::string sname = "my_service";
  std::string jstr = "{\"mllib\":\"caffe\",\"description\":\"my classifier\",\"type\":\"supervised\",\"model\":{\"repository\":\"" +  mnist_repo + "\"},\"parameters\":{\"input\":{\"connector\":\"image\"}}}";
  std::string joutstr = japi.service_create(sname,jstr);
  ASSERT_EQ(created_str,joutstr);

  // train
  std::string jtrainstr = "{\"service\":\"" + sname + "\",\"async\":false,\"parameters\":{\"mllib\":{\"solver\":{\"iterations\":200,\"snapshot\":200,\"snapshot_prefix\":\"" + mnist_repo + "/mylenet\"}},\"output\":{\"measure_hist\":true}}}";
  joutstr = japi.service_train(jtrainstr);
  std::cout << "joutstr=" << joutstr << std::endl;
  JDoc jd;
  jd.Parse(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_TRUE(jd.HasMember("status"));
  ASSERT_EQ(201,jd["status"]["code"].GetInt());
  ASSERT_EQ("Created",jd["status"]["msg"]);
  ASSERT_TRUE(jd.HasMember("head"));
  ASSERT_EQ("/train",jd["head"]["method"]);
  ASSERT_TRUE(jd["head"]["time"].GetDouble() > 0);
  ASSERT_TRUE(jd.HasMember("body"));
  ASSERT_TRUE(jd["body"].HasMember("measure"));
  ASSERT_TRUE(jd["body"]["measure"]["train_loss"].GetDouble() > 0);
  ASSERT_TRUE(jd["body"]["measure_hist"]["train_loss_hist"].Size() > 0);

  // predict
  std::string jpredictstr = "{\"service\":\""+ sname + "\",\"parameters\":{\"input\":{\"bw\":true}},\"data\":[\"" + mnist_repo + "/sample_digit.png\"]}";
  joutstr = japi.service_predict(jpredictstr);
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(500,jd["status"]["code"]);
  ASSERT_EQ(1007,jd["status"]["dd_code"]);
  
  // predict with image size (could be set at service creation)
  jpredictstr = "{\"service\":\""+ sname + "\",\"parameters\":{\"input\":{\"bw\":true,\"width\":28,\"height\":28},\"output\":{\"best\":3}},\"data\":[\"" + mnist_repo + "/sample_digit.png\",\"" + mnist_repo + "/sample_digit2.png\"]}";
  joutstr = japi.service_predict(jpredictstr);
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200,jd["status"]["code"]);
  ASSERT_TRUE(jd["body"]["predictions"][0]["classes"][0]["prob"].GetDouble() > 0);
  ASSERT_TRUE(jd["body"]["predictions"][1]["classes"][0]["prob"].GetDouble() > 0);
}
