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
#include <gtest/gtest.h>
#include <curlpp/cURLpp.hpp>
#include <curlpp/Easy.hpp>
#include <curlpp/Options.hpp>
#include <curlpp/Infos.hpp>
#include <iostream>

using namespace dd;

std::string host = "127.0.0.1";
int port = 8080;
int nthreads = 10;
std::string serv = "myserv";
std::string serv_put = "{\"mllib\":\"caffe\",\"description\":\"example classification service\",\"type\":\"supervised\",\"parameters\":{\"input\":{\"connector\":\"csv\"},\"mllib\":{\"template\":\"mlp\",\"nclasses\":2,\"layers\":[50,50,50],\"activation\":\"PReLU\"}},\"model\":{\"templates\":\"../templates/caffe/\",\"repository\":\".\"}}";

void get_call(const std::string &url,
	      const std::string &http_method,
	      int &outcode,
	      std::string &outstr)
{
  curlpp::Cleanup cl;
  std::ostringstream os;
  curlpp::Easy request;
  curlpp::options::WriteStream ws(&os);
  curlpp::options::CustomRequest pr(http_method);
  request.setOpt(curlpp::options::Url(url));
  request.setOpt(ws);
  request.setOpt(pr);
  request.perform();
  outstr = os.str();
  std::cout << "outstr=" << outstr << std::endl;
  outcode = curlpp::infos::ResponseCode::get(request);
}

void post_call(const std::string &url,
	       const std::string &jcontent,
	       const std::string &http_method,
	       int &outcode,
	       std::string &outstr)
{
  curlpp::Cleanup cl;
  std::ostringstream os;
  curlpp::Easy request_put;
  curlpp::options::WriteStream ws(&os);
  curlpp::options::CustomRequest pr(http_method);
  request_put.setOpt(curlpp::options::Url(url));
  request_put.setOpt(ws);
  request_put.setOpt(pr);
  std::list<std::string> header;
  header.push_back("Content-Type: application/json");
  request_put.setOpt(curlpp::options::HttpHeader(header));
  request_put.setOpt(curlpp::options::PostFields(jcontent));
  request_put.setOpt(curlpp::options::PostFieldSize(jcontent.length()));
  request_put.perform();
  outstr = os.str();
  std::cout << "outstr=" << outstr << std::endl;
  outcode = curlpp::infos::ResponseCode::get(request_put);
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
  get_call(luri+"/info","GET",code,jstr);
  
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
  post_call(luri+"/services/"+serv,serv_put,"PUT",
	    code,jstr);  
  ASSERT_EQ(201,code);
  rapidjson::Document d;
  d.Parse(jstr.c_str());
  ASSERT_FALSE(d.HasParseError());
  ASSERT_TRUE(d.HasMember("status"));
  ASSERT_EQ(201,d["status"]["code"].GetInt());

  // service info
  get_call(luri+"/services/"+serv,"GET",code,jstr);
  d = rapidjson::Document();
  d.Parse(jstr.c_str());
  ASSERT_EQ(200,code);
  ASSERT_TRUE(d.HasMember("status"));
  ASSERT_TRUE(d.HasMember("body"));
  ASSERT_EQ("caffe",d["body"]["mllib"]);
  ASSERT_EQ("myserv",d["body"]["name"]);
  
  // info call
  get_call(luri+"/info","GET",code,jstr);
  d = rapidjson::Document();
  d.Parse(jstr.c_str());
  ASSERT_EQ(200,code);
  ASSERT_TRUE(d["head"].HasMember("services"));
  ASSERT_EQ(1,d["head"]["services"].Size());

  // delete call
  get_call(luri+"/services/"+serv,"DELETE",code,jstr);
  //d.Parse(jstr.c_str());
  ASSERT_EQ(200,code);

  // info call
  get_call(luri+"/info","GET",code,jstr);
  d = rapidjson::Document();
  d.Parse(jstr.c_str());
  ASSERT_EQ(200,code);
  ASSERT_TRUE(d["head"].HasMember("services"));
  ASSERT_EQ(0,d["head"]["services"].Size());

  // service info
  get_call(luri+"/services/"+serv,"GET",code,jstr);
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
  std::string serv_put2 = "{\"mllib\":\"caffe\",\"description\":\"my classifier\",\"type\":\"supervised\",\"model\":{\"repository\":\"" +  mnist_repo + "\"},\"parameters\":{\"input\":{\"connector\":\"image\"}}}";
  
  // service creation
  int code = 1;
  std::string jstr;
  post_call(luri+"/services/"+serv,serv_put2,"PUT",
	    code,jstr);  
  ASSERT_EQ(201,code);
  rapidjson::Document d;
  d.Parse(jstr.c_str());
  ASSERT_FALSE(d.HasParseError());
  ASSERT_TRUE(d.HasMember("status"));
  ASSERT_EQ(201,d["status"]["code"].GetInt());

  //train blocking
  std::string train_post = "{\"service\":\"" + serv + "\",\"async\":false,\"parameters\":{\"mllib\":{\"gpu\":true,\"solver\":{\"iterations\":100}}}}";
  post_call(luri+"/train",train_post,"POST",code,jstr);
  ASSERT_EQ(201,code);
  d.Parse(jstr.c_str());
  ASSERT_TRUE(d.HasMember("body"));
  ASSERT_TRUE(d["body"].HasMember("measure"));
  ASSERT_EQ(99,d["body"]["measure"]["iteration"].GetDouble());
  ASSERT_TRUE(d["body"]["measure"]["train_loss"].GetDouble()>0.0);
  
  // remove service and trained model files
  get_call(luri+"/services/"+serv+"?clear=lib","DELETE",code,jstr);
  ASSERT_EQ(200,code);

  // service creation
  post_call(luri+"/services/"+serv,serv_put2,"PUT",
	    code,jstr);  
  ASSERT_EQ(201,code);
  d.Parse(jstr.c_str());
  ASSERT_FALSE(d.HasParseError());
  ASSERT_TRUE(d.HasMember("status"));
  ASSERT_EQ(201,d["status"]["code"].GetInt());
  
  //train async
  train_post = "{\"service\":\"" + serv + "\",\"async\":true,\"parameters\":{\"mllib\":{\"gpu\":true,\"solver\":{\"iterations\":100}}}}";
  post_call(luri+"/train",train_post,"POST",code,jstr);
  ASSERT_EQ(201,code);
  d.Parse(jstr.c_str());
  ASSERT_FALSE(d.HasMember("body"));
  
  // get info on training job
  bool running = true;
  while(running)
    {
      get_call(luri+"/train?service="+serv+"&job=1&timeout=1","GET",code,jstr);
      running = jstr.find("running") != std::string::npos;
      if (!running)
	{
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
    }
  
  // remove service and trained model files
  get_call(luri+"/services/"+serv+"?clear=lib","DELETE",code,jstr);
  ASSERT_EQ(200,code);
  
  hja.stop_server();
}
