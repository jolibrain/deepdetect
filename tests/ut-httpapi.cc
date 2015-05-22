/**
 * DeepDetect
 * Copyright (c) 2014-2015 Emmanuel Benazera
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
#include "httpjsonapi.h"
#include "utils/httpclient.hpp"
#include <gtest/gtest.h>
#include <iostream>

using namespace dd;

std::string host = "127.0.0.1";
int port = 8080;
int nthreads = 10;
std::string serv = "myserv";
std::string serv_put = "{\"mllib\":\"caffe\",\"description\":\"example classification service\",\"type\":\"supervised\",\"parameters\":{\"input\":{\"connector\":\"csv\"},\"mllib\":{\"template\":\"mlp\",\"nclasses\":2,\"layers\":[50,50,50],\"activation\":\"PReLU\"}},\"model\":{\"templates\":\"../templates/caffe/\",\"repository\":\".\"}}";

TEST(httpjsonapi,uri_query_to_json)
{
  std::string q = "service=myserv&job=1";
  std::string q1 = q + "&test=true";
  std::string p = uri_query_to_json(q1);
  ASSERT_EQ("{\"service\":\"myserv\",\"job\":1,\"test\":true}",p);
  std::string q1b = q + "&test=false";
  p = uri_query_to_json(q1b);
  ASSERT_EQ("{\"service\":\"myserv\",\"job\":1,\"test\":false}",p);
  std::string q2 = q + "&parameters.output.measure_hist=true";
  p = uri_query_to_json(q2);
  ASSERT_EQ("{\"service\":\"myserv\",\"job\":1,\"parameters\":{\"output\":{\"measure_hist\":true}}}",p);
  std::string q3 = q + "&parameters.output.measure_hist=false";
  p = uri_query_to_json(q3);
  ASSERT_EQ("{\"service\":\"myserv\",\"job\":1,\"parameters\":{\"output\":{\"measure_hist\":false}}}",p);
}

TEST(httpjsonapi,info)
{
  ::google::InitGoogleLogging("ut_httpapi");
  HttpJsonAPI hja;
  hja.start_server_daemon(host,std::to_string(port),nthreads);
  std::string luri = "http://" + host + ":" + std::to_string(port);
  sleep(2);
  
  int code = -1;
  std::string jstr;
  httpclient::get_call(luri+"/info","GET",code,jstr);
  
  ASSERT_EQ(200,code);
  rapidjson::Document d;
  d.Parse(jstr.c_str());
  ASSERT_FALSE(d.HasParseError());
  ASSERT_TRUE(d.HasMember("status"));
  ASSERT_TRUE(d.HasMember("head"));
  ASSERT_TRUE(d["head"].HasMember("services"));
  ASSERT_EQ(0,d["head"]["services"].Size());
  
  hja.stop_server();
}

TEST(httpjsonapi,services)
{
  
  HttpJsonAPI hja;
  hja.start_server_daemon(host,std::to_string(++port),nthreads);
  std::string luri = "http://" + host + ":" + std::to_string(port);
  sleep(2);

  // service creation
  int code = 1;
  std::string jstr;
  httpclient::post_call(luri+"/services/"+serv,serv_put,"PUT",
	    code,jstr);  
  ASSERT_EQ(201,code);
  rapidjson::Document d;
  d.Parse(jstr.c_str());
  ASSERT_FALSE(d.HasParseError());
  ASSERT_TRUE(d.HasMember("status"));
  ASSERT_EQ(201,d["status"]["code"].GetInt());

  // service info
  httpclient::get_call(luri+"/services/"+serv,"GET",code,jstr);
  d = rapidjson::Document();
  d.Parse(jstr.c_str());
  ASSERT_EQ(200,code);
  ASSERT_TRUE(d.HasMember("status"));
  ASSERT_TRUE(d.HasMember("body"));
  ASSERT_EQ("caffe",d["body"]["mllib"]);
  ASSERT_EQ("myserv",d["body"]["name"]);
  ASSERT_TRUE(d["body"].HasMember("jobs"));
  
  // info call
  httpclient::get_call(luri+"/info","GET",code,jstr);
  d = rapidjson::Document();
  d.Parse(jstr.c_str());
  ASSERT_EQ(200,code);
  ASSERT_TRUE(d["head"].HasMember("services"));
  ASSERT_EQ(1,d["head"]["services"].Size());

  // delete call
  httpclient::get_call(luri+"/services/"+serv,"DELETE",code,jstr);
  ASSERT_EQ(200,code);

  // info call
  httpclient::get_call(luri+"/info","GET",code,jstr);
  d = rapidjson::Document();
  d.Parse(jstr.c_str());
  ASSERT_EQ(200,code);
  ASSERT_TRUE(d["head"].HasMember("services"));
  ASSERT_EQ(0,d["head"]["services"].Size());

  // service info
  httpclient::get_call(luri+"/services/"+serv,"GET",code,jstr);
  d = rapidjson::Document();
  d.Parse(jstr.c_str());
  ASSERT_EQ(404,code);
  
  hja.stop_server();
}

TEST(httpjsonapi,train)
{
  HttpJsonAPI hja;
  hja.start_server_daemon(host,std::to_string(++port),nthreads);
  std::string luri = "http://" + host + ":" + std::to_string(port);
  sleep(2);

  // service definition
  std::string mnist_repo = "../examples/caffe/mnist/";
  std::string serv_put2 = "{\"mllib\":\"caffe\",\"description\":\"my classifier\",\"type\":\"supervised\",\"model\":{\"repository\":\"" +  mnist_repo + "\"},\"parameters\":{\"input\":{\"connector\":\"image\"},\"mllib\":{\"nclasses\":10}}}";
  
  // service creation
  int code = 1;
  std::string jstr;
  httpclient::post_call(luri+"/services/"+serv,serv_put2,"PUT",
			 code,jstr);  
  ASSERT_EQ(201,code);
  rapidjson::Document d;
  d.Parse(jstr.c_str());
  ASSERT_FALSE(d.HasParseError());
  ASSERT_TRUE(d.HasMember("status"));
  ASSERT_EQ(201,d["status"]["code"].GetInt());

  //train blocking
  std::string train_post = "{\"service\":\"" + serv + "\",\"async\":false,\"parameters\":{\"mllib\":{\"gpu\":true,\"solver\":{\"iterations\":100}}}}";
  httpclient::post_call(luri+"/train",train_post,"POST",code,jstr);
  ASSERT_EQ(201,code);
  d.Parse(jstr.c_str());
  ASSERT_TRUE(d.HasMember("body"));
  ASSERT_TRUE(d["body"].HasMember("measure"));
  ASSERT_EQ(99,d["body"]["measure"]["iteration"].GetDouble());
  ASSERT_TRUE(d["body"]["measure"]["train_loss"].GetDouble()>0.0);
  
  // remove service and trained model files
  httpclient::get_call(luri+"/services/"+serv+"?clear=lib","DELETE",code,jstr);
  ASSERT_EQ(200,code);

  // service creation
  httpclient::post_call(luri+"/services/"+serv,serv_put2,"PUT",
			 code,jstr);  
  ASSERT_EQ(201,code);
  d.Parse(jstr.c_str());
  ASSERT_FALSE(d.HasParseError());
  ASSERT_TRUE(d.HasMember("status"));
  ASSERT_EQ(201,d["status"]["code"].GetInt());
  
  //train async
  train_post = "{\"service\":\"" + serv + "\",\"async\":true,\"parameters\":{\"mllib\":{\"gpu\":true,\"solver\":{\"iterations\":100}}}}";
  httpclient::post_call(luri+"/train",train_post,"POST",code,jstr);
  ASSERT_EQ(201,code);
  d.Parse(jstr.c_str());
  ASSERT_FALSE(d.HasMember("body"));

  sleep(1);

  // service info
  httpclient::get_call(luri+"/services/"+serv,"GET",code,jstr);
  d = rapidjson::Document();
  d.Parse(jstr.c_str());
  ASSERT_EQ(200,code);
  ASSERT_TRUE(d.HasMember("status"));
  ASSERT_TRUE(d.HasMember("body"));
  ASSERT_EQ("caffe",d["body"]["mllib"]);
  ASSERT_EQ("myserv",d["body"]["name"]);
  ASSERT_TRUE(d["body"].HasMember("jobs"));
  ASSERT_EQ("running",d["body"]["jobs"]["status"]);
  
  // get info on training job
  bool running = true;
  while(running)
    {
      httpclient::get_call(luri+"/train?service="+serv+"&job=1&timeout=1&parameters.output.measure_hist=true","GET",code,jstr);
      running = jstr.find("running") != std::string::npos;
      if (running)
	{
	  std::cerr << "jstr=" << jstr << std::endl;
	  JDoc jd2;
	  jd2.Parse(jstr.c_str());
	  ASSERT_TRUE(!jd2.HasParseError());
	  ASSERT_TRUE(jd2.HasMember("status"));
	  ASSERT_EQ(200,jd2["status"]["code"]);
	  ASSERT_EQ("OK",jd2["status"]["msg"]);
	  ASSERT_TRUE(jd2.HasMember("head"));
	  ASSERT_EQ("/train",jd2["head"]["method"]);
	  ASSERT_TRUE(jd2["head"]["time"].GetDouble() > 0);
	  ASSERT_EQ("running",jd2["head"]["status"]);
	  ASSERT_EQ(1,jd2["head"]["job"]);
	  ASSERT_TRUE(jd2.HasMember("body"));
	  ASSERT_TRUE(jd2["body"]["measure"].HasMember("train_loss"));
	  ASSERT_TRUE(jd2["body"]["measure"]["train_loss"].GetDouble() > 0);
	  ASSERT_TRUE(jd2["body"]["measure"].HasMember("iteration"));
	  ASSERT_TRUE(jd2["body"]["measure"]["iteration"].GetDouble() >= 0);
	  ASSERT_TRUE(jd2["body"].HasMember("measure_hist"));
	}
      else ASSERT_TRUE(jstr.find("finished")!=std::string::npos);
    }
  
  // remove service and trained model files
  httpclient::get_call(luri+"/services/"+serv+"?clear=lib","DELETE",code,jstr);
  ASSERT_EQ(200,code);
  
  hja.stop_server();
}

TEST(httpjsonapi,multiservices)
{
  HttpJsonAPI hja;
  hja.start_server_daemon(host,std::to_string(++port),nthreads);
  std::string luri = "http://" + host + ":" + std::to_string(port);
  sleep(2);

  // service definition
  std::string mnist_repo = "../examples/caffe/mnist/";
  std::string serv_put2 = "{\"mllib\":\"caffe\",\"description\":\"my classifier\",\"type\":\"supervised\",\"model\":{\"repository\":\"" +  mnist_repo + "\"},\"parameters\":{\"input\":{\"connector\":\"image\"},\"mllib\":{\"nclasses\":10}}}";
  
  // service creation
  int code = 1;
  std::string jstr;
  httpclient::post_call(luri+"/services/"+serv,serv_put2,"PUT",
			 code,jstr);  
  ASSERT_EQ(201,code);
  rapidjson::Document d;
  d.Parse(jstr.c_str());
  ASSERT_FALSE(d.HasParseError());
  ASSERT_TRUE(d.HasMember("status"));
  ASSERT_EQ(201,d["status"]["code"].GetInt());

  // service creation
  std::string serv2 = "myserv2";
  httpclient::post_call(luri+"/services/"+serv2,serv_put2,"PUT",
			 code,jstr);  
  ASSERT_EQ(201,code);
  d.Parse(jstr.c_str());
  ASSERT_FALSE(d.HasParseError());
  ASSERT_TRUE(d.HasMember("status"));
  ASSERT_EQ(201,d["status"]["code"].GetInt());

  // info call
  httpclient::get_call(luri+"/info","GET",code,jstr);
  d = rapidjson::Document();
  d.Parse(jstr.c_str());
  ASSERT_EQ(200,code);
  ASSERT_TRUE(d["head"].HasMember("services"));
  ASSERT_EQ(2,d["head"]["services"].Size());
  ASSERT_TRUE(jstr.find("\"name\":\"myserv\"")!=std::string::npos);
  ASSERT_TRUE(jstr.find("\"name\":\"myserv2\"")!=std::string::npos);
  
  // remove services and trained model files
  httpclient::get_call(luri+"/services/"+serv+"?clear=lib","DELETE",code,jstr);
  ASSERT_EQ(200,code);
  httpclient::get_call(luri+"/services/"+serv2+"?clear=lib","DELETE",code,jstr);
  ASSERT_EQ(200,code);
  
  hja.stop_server();
}

TEST(httpjsonapi,concurrency)
{
  HttpJsonAPI hja;
  hja.start_server_daemon(host,std::to_string(++port),nthreads);
  std::string luri = "http://" + host + ":" + std::to_string(port);
  sleep(2);

  // service definition
  std::string mnist_repo = "../examples/caffe/mnist/";
  std::string serv_put2 = "{\"mllib\":\"caffe\",\"description\":\"my classifier\",\"type\":\"supervised\",\"model\":{\"repository\":\"" +  mnist_repo + "\"},\"parameters\":{\"input\":{\"connector\":\"image\"},\"mllib\":{\"nclasses\":10}}}";
  
  // service creation
  int code = 1;
  std::string jstr;
  httpclient::post_call(luri+"/services/"+serv,serv_put2,"PUT",
			 code,jstr);  
  ASSERT_EQ(201,code);
  rapidjson::Document d;
  d.Parse(jstr.c_str());
  ASSERT_FALSE(d.HasParseError());
  ASSERT_TRUE(d.HasMember("status"));
  ASSERT_EQ(201,d["status"]["code"].GetInt());

  //train async
  std::string train_post = "{\"service\":\"" + serv + "\",\"async\":true,\"parameters\":{\"mllib\":{\"gpu\":true,\"solver\":{\"iterations\":1000}}}}";
  httpclient::post_call(luri+"/train",train_post,"POST",code,jstr);
  ASSERT_EQ(201,code);
  d.Parse(jstr.c_str());
  ASSERT_FALSE(d.HasMember("body"));
  
  // service creation
  std::string serv2 = "myserv2";
  httpclient::post_call(luri+"/services/"+serv2,serv_put2,"PUT",
			 code,jstr);  
  ASSERT_EQ(201,code);
  d.Parse(jstr.c_str());
  ASSERT_FALSE(d.HasParseError());
  ASSERT_TRUE(d.HasMember("status"));
  ASSERT_EQ(201,d["status"]["code"].GetInt());

  // info call
  httpclient::get_call(luri+"/info","GET",code,jstr);
  d = rapidjson::Document();
  d.Parse(jstr.c_str());
  ASSERT_EQ(200,code);
  ASSERT_TRUE(d["head"].HasMember("services"));
  ASSERT_EQ(2,d["head"]["services"].Size());
  ASSERT_TRUE(jstr.find("\"name\":\"myserv\"")!=std::string::npos);
  ASSERT_TRUE(jstr.find("\"name\":\"myserv2\"")!=std::string::npos);

  //train async second job
  train_post = "{\"service\":\"" + serv2 + "\",\"async\":true,\"parameters\":{\"mllib\":{\"gpu\":true,\"solver\":{\"iterations\":1000}}}}";
  httpclient::post_call(luri+"/train",train_post,"POST",code,jstr);
  ASSERT_EQ(201,code);
  d.Parse(jstr.c_str());
  ASSERT_FALSE(d.HasMember("body"));

  sleep(1);

  // get info on job2
  httpclient::get_call(luri+"/train?service="+serv2+"&job=1","GET",code,jstr);
  ASSERT_EQ(200,code);
  
  // get info on first training job
  int tmax = 10, t = 0;
  bool running = true;
  while(running)
    {
      httpclient::get_call(luri+"/train?service="+serv+"&job=1&timeout=1","GET",code,jstr);
      running = jstr.find("running") != std::string::npos;
      if (!running)
	{
	  std::cerr << "jstr=" << jstr << std::endl;
	  JDoc jd2;
	  jd2.Parse(jstr.c_str());
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
	  ASSERT_TRUE(jd2["body"]["measure"].HasMember("iteration"));
	  ASSERT_TRUE(jd2["body"]["measure"]["iteration"].GetDouble() > 0);
	}
      ++t;
      if (t > tmax)
	break;
    }

  // delete job1
  httpclient::get_call(luri+"/train?service="+serv+"&job=1","DELETE",code,jstr);
  ASSERT_EQ(200,code);
  d.Parse(jstr.c_str());
  ASSERT_TRUE(d.HasMember("head"));
  ASSERT_TRUE(d["head"].HasMember("status"));
  ASSERT_EQ("terminated",d["head"]["status"]);

  sleep(1);
  
  // get info on job1
  httpclient::get_call(luri+"/train?service="+serv+"&job=1","GET",code,jstr);
  ASSERT_EQ(404,code);

  // delete job2
  httpclient::get_call(luri+"/train?service="+serv2+"&job=1","DELETE",code,jstr);
  ASSERT_EQ(200,code);
  d.Parse(jstr.c_str());
  ASSERT_TRUE(d.HasMember("head"));
  ASSERT_TRUE(d["head"].HasMember("status"));
  ASSERT_EQ("terminated",d["head"]["status"]);

  sleep(1);
  
  // get info on job2
  httpclient::get_call(luri+"/train?service="+serv2+"&job=1","GET",code,jstr);
  ASSERT_EQ(404,code);
  
  // remove services and trained model files
  httpclient::get_call(luri+"/services/"+serv+"?clear=lib","DELETE",code,jstr);
  ASSERT_EQ(200,code);
  httpclient::get_call(luri+"/services/"+serv2+"?clear=lib","DELETE",code,jstr);
  ASSERT_EQ(200,code);
  
  hja.stop_server();
}

TEST(httpjsonapi,predict)
{
  HttpJsonAPI hja;
  hja.start_server_daemon(host,std::to_string(++port),nthreads);
  std::string luri = "http://" + host + ":" + std::to_string(port);
  sleep(2);

  std::string mnist_repo = "../examples/caffe/mnist/";
  std::string serv_put2 = "{\"mllib\":\"caffe\",\"description\":\"my classifier\",\"type\":\"supervised\",\"model\":{\"repository\":\"" +  mnist_repo + "\"},\"parameters\":{\"input\":{\"connector\":\"image\"},\"mllib\":{\"nclasses\":10}}}";
  
  // service creation
  int code = 1;
  std::string jstr;
  httpclient::post_call(luri+"/services/"+serv,serv_put2,"PUT",
			 code,jstr);  
  ASSERT_EQ(201,code);
  rapidjson::Document d;
  d.Parse(jstr.c_str());
  ASSERT_FALSE(d.HasParseError());
  ASSERT_TRUE(d.HasMember("status"));
  ASSERT_EQ(201,d["status"]["code"].GetInt());

  // train sync
  std::string train_post = "{\"service\":\"" + serv + "\",\"async\":false,\"parameters\":{\"mllib\":{\"gpu\":true,\"solver\":{\"iterations\":200,\"snapshot_prefix\":\""+mnist_repo+"/mylenet\"}},\"output\":{\"measure_hist\":true}}}";
  httpclient::post_call(luri+"/train",train_post,"POST",code,jstr);
  std::cerr << "jstr=" << jstr << std::endl;
  ASSERT_EQ(201,code);
  JDoc jd;
  jd.Parse(jstr.c_str());
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
  std::string predict_post = "{\"service\":\""+ serv + "\",\"parameters\":{\"input\":{\"bw\":true,\"width\":28,\"height\":28},\"output\":{\"best\":3}},\"data\":[\"" + mnist_repo + "/sample_digit.png\",\"" + mnist_repo + "/sample_digit2.png\"]}";
  httpclient::post_call(luri+"/predict",predict_post,"POST",code,jstr);
  std::cerr << "code=" << code << std::endl;
  ASSERT_EQ(200,code);
  std::cerr << "jstr=" << jstr << std::endl;
  jd.Parse(jstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200,jd["status"]["code"]);
  ASSERT_TRUE(jd["body"]["predictions"][0]["classes"][0]["prob"].GetDouble() > 0);
  ASSERT_TRUE(jd["body"]["predictions"][1]["classes"][0]["prob"].GetDouble() > 0);

  // remove services and trained model files
  httpclient::get_call(luri+"/services/"+serv+"?clear=lib","DELETE",code,jstr);
  ASSERT_EQ(200,code);
  
  hja.stop_server();
}
