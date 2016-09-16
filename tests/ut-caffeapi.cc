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

static std::string mnist_repo = "../examples/caffe/mnist/";
static std::string forest_repo = "../examples/all/forest_type/";
static std::string plank_repo = "../examples/caffe/plankton/";
static std::string n20_repo = "../examples/all/n20/";
static std::string sflare_repo = "../examples/all/sflare/";
static std::string model_templates_repo = "../templates/caffe/";

#ifndef CPU_ONLY
static std::string iterations_mnist = "250";
static std::string iterations_plank = "2000";
static std::string iterations_forest = "3000";
static std::string iterations_n20 = "2000";
static std::string iterations_n20_char = "1000";
static std::string iterations_sflare = "5000";
#else
static std::string iterations_mnist = "10";
static std::string iterations_plank = "10";
static std::string iterations_forest = "500";
static std::string iterations_n20 = "1000";
static std::string iterations_n20_char = "10";
static std::string iterations_sflare = "2000";
#endif

TEST(caffeapi,service_train)
{
::google::InitGoogleLogging("ut_caffeapi");
// create service
  JsonAPI japi;
  std::string sname = "my_service";
  std::string jstr = "{\"mllib\":\"caffe\",\"description\":\"my classifier\",\"type\":\"supervised\",\"model\":{\"repository\":\"" +  mnist_repo + "\"},\"parameters\":{\"input\":{\"connector\":\"image\"},\"mllib\":{\"nclasses\":10}}}";
  std::string joutstr = japi.jrender(japi.service_create(sname,jstr));
  ASSERT_EQ(created_str,joutstr);

  // train
  std::string jtrainstr = "{\"service\":\"" + sname + "\",\"async\":false,\"parameters\":{\"mllib\":{\"gpu\":true,\"solver\":{\"iterations\":" + iterations_mnist + "}}}}";
  joutstr = japi.jrender(japi.service_train(jtrainstr));
  std::cout << "joutstr=" << joutstr << std::endl;
  JDoc jd;
  jd.Parse(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_TRUE(jd.HasMember("status"));
  ASSERT_EQ(201,jd["status"]["code"].GetInt());
  ASSERT_EQ("Created",jd["status"]["msg"]);
  ASSERT_TRUE(jd.HasMember("head"));
  ASSERT_EQ("/train",jd["head"]["method"]);
  ASSERT_TRUE(jd["head"]["time"].GetDouble() >= 0);
  ASSERT_TRUE(jd.HasMember("body"));
  ASSERT_TRUE(jd["body"]["measure"].HasMember("train_loss"));
  ASSERT_TRUE(fabs(jd["body"]["measure"]["train_loss"].GetDouble()) > 0);

  // remove service
  jstr = "{\"clear\":\"lib\"}";
  joutstr = japi.jrender(japi.service_delete(sname,jstr));
  ASSERT_EQ(ok_str,joutstr);
  ASSERT_TRUE(!fileops::file_exists(mnist_repo + "mylenet_iter_101.caffemodel"));
  ASSERT_TRUE(!fileops::file_exists(mnist_repo + "mylenet_iter_101.solverstate"));
}

TEST(caffeapi,service_train_async_status_delete)
{
  // create service
  JsonAPI japi;
  std::string sname = "my_service";
  std::string jstr = "{\"mllib\":\"caffe\",\"description\":\"my classifier\",\"type\":\"supervised\",\"model\":{\"repository\":\"" +  mnist_repo + "\"},\"parameters\":{\"input\":{\"connector\":\"image\"},\"mllib\":{\"nclasses\":10}}}";
  std::string joutstr = japi.jrender(japi.service_create(sname,jstr));
  ASSERT_EQ(created_str,joutstr);

  // train
  std::string jtrainstr = "{\"service\":\"" + sname + "\",\"async\":true,\"parameters\":{\"mllib\":{\"gpu\":true,\"solver\":{\"iterations\":10000}}}}";
  joutstr = japi.jrender(japi.service_train(jtrainstr));
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
  joutstr = japi.jrender(japi.service_train_status(jstatusstr));
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
  ASSERT_TRUE(fabs(jd2["body"]["measure"]["train_loss"].GetDouble()) > 0);

  // delete job.
  std::string jdelstr = "{\"service\":\"" + sname + "\",\"job\":1}";
  joutstr = japi.jrender(japi.service_train_delete(jdelstr));
  std::cout << "joutstr=" << joutstr << std::endl;
  JDoc jd3;
  jd3.Parse(joutstr.c_str());
  ASSERT_TRUE(!jd3.HasParseError());
  ASSERT_TRUE(jd3.HasMember("status"));
  ASSERT_EQ(200,jd3["status"]["code"]);
  ASSERT_EQ("OK",jd3["status"]["msg"]);
  ASSERT_TRUE(jd3.HasMember("head"));
  ASSERT_EQ("/train",jd3["head"]["method"]);
  ASSERT_TRUE(jd3["head"]["time"].GetDouble() >= 0);
  ASSERT_EQ("terminated",jd3["head"]["status"]);
  ASSERT_EQ(1,jd3["head"]["job"].GetInt());

  // remove service
  jstr = "{\"clear\":\"lib\"}";
  joutstr = japi.jrender(japi.service_delete(sname,jstr));
  ASSERT_EQ(ok_str,joutstr);
}

TEST(caffeapi,service_train_async_final_status)
{
  // create service
  JsonAPI japi;
  std::string sname = "my_service";
  std::string jstr = "{\"mllib\":\"caffe\",\"description\":\"my classifier\",\"type\":\"supervised\",\"model\":{\"repository\":\"" +  mnist_repo + "\"},\"parameters\":{\"input\":{\"connector\":\"image\"},\"mllib\":{\"nclasses\":10}}}";
  std::string joutstr = japi.jrender(japi.service_create(sname,jstr));
  ASSERT_EQ(created_str,joutstr);

  // train
  std::string jtrainstr = "{\"service\":\"" + sname + "\",\"async\":true,\"parameters\":{\"mllib\":{\"gpu\":true,\"solver\":{\"iterations\":" + iterations_mnist + "}}}}";
  joutstr = japi.jrender(japi.service_train(jtrainstr));
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
      joutstr = japi.jrender(japi.service_train_status(jstatusstr));
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
	  ASSERT_TRUE(jd2["head"]["time"].GetDouble() >= 0);
	  ASSERT_EQ("finished",jd2["head"]["status"]);
	  ASSERT_EQ(1,jd2["head"]["job"]);
	  ASSERT_TRUE(jd2.HasMember("body"));
	  ASSERT_TRUE(jd2["body"]["measure"].HasMember("train_loss"));
	  ASSERT_TRUE(fabs(jd2["body"]["measure"]["train_loss"].GetDouble()) > 0);
	  ASSERT_TRUE(jd2["body"]["measure"].HasMember("iteration"));
	  ASSERT_TRUE(jd2["body"]["measure"]["iteration"].GetDouble() > 0);
	}
    }

   // remove service
  jstr = "{\"clear\":\"lib\"}";
  joutstr = japi.jrender(japi.service_delete(sname,jstr));
  ASSERT_EQ(ok_str,joutstr);
}

// predict while training
TEST(caffeapi,service_train_async_and_predict)
{
  // create service
  JsonAPI japi;
  std::string sname = "my_service";
  std::string jstr = "{\"mllib\":\"caffe\",\"description\":\"my classifier\",\"type\":\"supervised\",\"model\":{\"repository\":\"" +  mnist_repo + "\"},\"parameters\":{\"input\":{\"connector\":\"image\"},\"mllib\":{\"nclasses\":10}}}";
  std::string joutstr = japi.jrender(japi.service_create(sname,jstr));
  ASSERT_EQ(created_str,joutstr);

  // train
  std::string iterations = iterations_mnist;
#ifndef CPU_ONLY
  iterations = "1500";
#endif
  std::string jtrainstr = "{\"service\":\"" + sname + "\",\"async\":true,\"parameters\":{\"mllib\":{\"gpu\":true,\"solver\":{\"iterations\":" + iterations + "}}}}";
  joutstr = japi.jrender(japi.service_train(jtrainstr));
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
  
  // status
  std::string jstatusstr = "{\"service\":\"" + sname + "\",\"job\":1,\"timeout\":2}";
  joutstr = japi.jrender(japi.service_train_status(jstatusstr));
  std::cout << "joutstr=" << joutstr << std::endl;
    
  // predict call
  std::string jpredictstr = "{\"service\":\""+ sname + "\",\"parameters\":{\"input\":{\"bw\":true,\"width\":28,\"height\":28}},\"data\":[\"" + mnist_repo + "/sample_digit.png\"]}";
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(409,jd["status"]["code"]);
  ASSERT_EQ(1008,jd["status"]["dd_code"]);
  
  // remove service
  jstr = "{\"clear\":\"lib\"}";
  joutstr = japi.jrender(japi.service_delete(sname,jstr));
  ASSERT_EQ(ok_str,joutstr);
}

TEST(caffeapi,service_predict)
{
  // create service
  JsonAPI japi;
  std::string sname = "my_service";
  std::string jstr = "{\"mllib\":\"caffe\",\"description\":\"my classifier\",\"type\":\"supervised\",\"model\":{\"repository\":\"" +  mnist_repo + "\"},\"parameters\":{\"input\":{\"connector\":\"image\"},\"mllib\":{\"nclasses\":10}}}";
  std::string joutstr = japi.jrender(japi.service_create(sname,jstr));
  ASSERT_EQ(created_str,joutstr);

  // train
  std::string jtrainstr = "{\"service\":\"" + sname + "\",\"async\":false,\"parameters\":{\"mllib\":{\"gpu\":true,\"solver\":{\"iterations\":" + iterations_mnist + ",\"snapshot\":200,\"snapshot_prefix\":\"" + mnist_repo + "/mylenet\",\"test_interval\":2}},\"output\":{\"measure_hist\":true,\"measure\":[\"f1\"]}}}";
  joutstr = japi.jrender(japi.service_train(jtrainstr));
  std::cout << "joutstr=" << joutstr << std::endl;
  JDoc jd;
  jd.Parse(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_TRUE(jd.HasMember("status"));
  ASSERT_EQ(201,jd["status"]["code"].GetInt());
  ASSERT_EQ("Created",jd["status"]["msg"]);
  ASSERT_TRUE(jd.HasMember("head"));
  ASSERT_EQ("/train",jd["head"]["method"]);
  ASSERT_TRUE(jd["head"]["time"].GetDouble() >= 0);
  ASSERT_TRUE(jd.HasMember("body"));
  ASSERT_TRUE(jd["body"].HasMember("measure"));
  ASSERT_TRUE(fabs(jd["body"]["measure"]["train_loss"].GetDouble()) > 0);
  ASSERT_EQ(jd["body"]["measure_hist"]["iteration_hist"].Size(),jd["body"]["measure_hist"]["train_loss_hist"].Size());

  // predict
  std::string jpredictstr = "{\"service\":\""+ sname + "\",\"parameters\":{\"input\":{\"bw\":true}},\"data\":[\"" + mnist_repo + "/sample_digit.png\"]}";
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  std::cout << "joutstr predict=" << joutstr << std::endl;
  jd.Parse(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(500,jd["status"]["code"]);
  ASSERT_EQ(1007,jd["status"]["dd_code"]);
  
  // predict with image size (could be set at service creation)
  jpredictstr = "{\"service\":\""+ sname + "\",\"parameters\":{\"input\":{\"bw\":true,\"width\":28,\"height\":28},\"output\":{\"best\":3}},\"data\":[\"" + mnist_repo + "/sample_digit.png\",\"" + mnist_repo + "/sample_digit2.png\"]}";
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200,jd["status"]["code"]);
  ASSERT_TRUE(jd["body"]["predictions"].IsArray());
  ASSERT_TRUE((mnist_repo + "/sample_digit.png"==jd["body"]["predictions"][0]["uri"].GetString())
	      ||(mnist_repo + "/sample_digit2.png"==jd["body"]["predictions"][0]["uri"].GetString()));
  ASSERT_TRUE(jd["body"]["predictions"][0]["classes"][0]["prob"].GetDouble() > 0);
  ASSERT_TRUE(jd["body"]["predictions"][1]["classes"][0]["prob"].GetDouble() > 0);

  // base64 predict
  std::string img_str;
  std::fstream fimg(mnist_repo + "/sample_digit.png");
  std::stringstream buffer;
  buffer << fimg.rdbuf();
  img_str = buffer.str();
  std::string b64_str;
  Base64::Encode(img_str,&b64_str);
  jpredictstr = "{\"service\":\""+ sname + "\",\"parameters\":{\"input\":{\"bw\":true,\"width\":28,\"height\":28},\"output\":{\"best\":3}},\"data\":[\"" + b64_str + "\"]}";
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200,jd["status"]["code"]);
  std::string uri = jd["body"]["predictions"][0]["uri"].GetString();
  std::cerr << "uri=" << uri << std::endl;
  ASSERT_EQ("0",uri);

  // predict non existing image
  jpredictstr = "{\"service\":\""+ sname + "\",\"parameters\":{\"input\":{\"bw\":true,\"width\":28,\"height\":28},\"output\":{\"best\":3}},\"data\":[\"http://example.com/my_image.png\"]}";
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(400,jd["status"]["code"]);
    
  // remove service
  jstr = "{\"clear\":\"lib\"}";
  joutstr = japi.jrender(japi.service_delete(sname,jstr));
  ASSERT_EQ(ok_str,joutstr);
}

TEST(caffeapi,service_train_csv)
{
  // create service
  JsonAPI japi;
  std::string sname = "my_service";
  std::string jstr = "{\"mllib\":\"caffe\",\"description\":\"my classifier\",\"type\":\"supervised\",\"model\":{\"repository\":\"" +  forest_repo + "\",\"templates\":\"" + model_templates_repo  + "\"},\"parameters\":{\"input\":{\"connector\":\"csv\"},\"mllib\":{\"template\":\"mlp\",\"nclasses\":7,\"activation\":\"prelu\"}}}";
  std::string joutstr = japi.jrender(japi.service_create(sname,jstr));
  ASSERT_EQ(created_str,joutstr);

  // assert json blob file
  ASSERT_TRUE(fileops::file_exists(forest_repo + "/" + JsonAPI::_json_blob_fname));
  
  // train
  std::string jtrainstr = "{\"service\":\"" + sname + "\",\"async\":false,\"parameters\":{\"input\":{\"label\":\"Cover_Type\",\"id\":\"Id\",\"scale\":true,\"test_split\":0.1,\"label_offset\":-1,\"shuffle\":true},\"mllib\":{\"gpu\":true,\"solver\":{\"iterations\":" + iterations_forest + ",\"base_lr\":0.05},\"net\":{\"batch_size\":512}},\"output\":{\"measure\":[\"acc\",\"mcll\",\"f1\",\"cmdiag\",\"cmfull\"]}},\"data\":[\"" + forest_repo + "train.csv\"]}";
  joutstr = japi.jrender(japi.service_train(jtrainstr));
  std::cout << "joutstr=" << joutstr << std::endl;
  JDoc jd;
  jd.Parse(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_TRUE(jd.HasMember("status"));
  ASSERT_EQ(201,jd["status"]["code"].GetInt());
  ASSERT_EQ("Created",jd["status"]["msg"]);
  ASSERT_TRUE(jd.HasMember("head"));
  ASSERT_EQ("/train",jd["head"]["method"]);
  ASSERT_TRUE(jd["head"]["time"].GetDouble() >= 0);
  ASSERT_TRUE(jd.HasMember("body"));
  ASSERT_TRUE(jd["body"]["measure"].HasMember("train_loss"));
  ASSERT_TRUE(fabs(jd["body"]["measure"]["train_loss"].GetDouble()) > 0.0);
  ASSERT_TRUE(jd["body"]["measure"].HasMember("f1"));
#ifndef CPU_ONLY
  ASSERT_TRUE(jd["body"]["measure"]["f1"].GetDouble() > 0.7);
#else
  ASSERT_TRUE(jd["body"]["measure"]["f1"].GetDouble() > 0.5);
#endif
  ASSERT_EQ(jd["body"]["measure"]["accp"].GetDouble(),jd["body"]["measure"]["acc"].GetDouble());
  ASSERT_TRUE(jd["body"].HasMember("parameters"));
  ASSERT_TRUE(jd["body"]["parameters"].HasMember("input"));
  ASSERT_TRUE(jd["body"]["parameters"]["input"].HasMember("min_vals"));
  ASSERT_TRUE(jd["body"]["parameters"]["input"].HasMember("max_vals"));
  ASSERT_TRUE(jd["body"]["measure"].HasMember("cmdiag"));
  ASSERT_EQ(7,jd["body"]["measure"]["cmdiag"].Size());
  ASSERT_TRUE(jd["body"]["measure"]["cmdiag"][0].GetDouble() >= 0);
  ASSERT_TRUE(jd["body"]["measure"]["cmfull"].Size()); //TODO: update
  ASSERT_EQ(56,jd["body"]["parameters"]["input"]["min_vals"].Size());
  ASSERT_EQ(56,jd["body"]["parameters"]["input"]["max_vals"].Size());
  ASSERT_EQ(504,jd["body"]["parameters"]["mllib"]["batch_size"].GetInt());

  std::string str_min_vals = japi.jrender(jd["body"]["parameters"]["input"]["min_vals"]);
  std::string str_max_vals = japi.jrender(jd["body"]["parameters"]["input"]["max_vals"]);
  std::cerr << "str_min_vals=" << str_min_vals << std::endl;
  std::cerr << "str_max_vals=" << str_max_vals << std::endl;
  
  // predict from data, with header and id
  std::string mem_data_head = "Id,Elevation,Aspect,Slope,Horizontal_Distance_To_Hydrology,Vertical_Distance_To_Hydrology,Horizontal_Distance_To_Roadways,Hillshade_9am,Hillshade_Noon,Hillshade_3pm,Horizontal_Distance_To_Fire_Points,Wilderness_Area1,Wilderness_Area2,Wilderness_Area3,Wilderness_Area4,Soil_Type1,Soil_Type2,Soil_Type3,Soil_Type4,Soil_Type5,Soil_Type6,Soil_Type7,Soil_Type8,Soil_Type9,Soil_Type10,Soil_Type11,Soil_Type12,Soil_Type13,Soil_Type14,Soil_Type15,Soil_Type16,Soil_Type17,Soil_Type18,Soil_Type19,Soil_Type20,Soil_Type21,Soil_Type22,Soil_Type23,Soil_Type24,Soil_Type25,Soil_Type26,Soil_Type27,Soil_Type28,Soil_Type29,Soil_Type30,Soil_Type31,Soil_Type32,Soil_Type33,Soil_Type34,Soil_Type35,Soil_Type36,Soil_Type37,Soil_Type38,Soil_Type39,Soil_Type40";
  std::string mem_data = "0,2499,326,7,300,88,480,202,232,169,1676,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0";
  std::string jpredictstr = "{\"service\":\""+ sname + "\",\"parameters\":{\"input\":{\"connector\":\"csv\",\"id\":\"Id\",\"scale\":true,\"min_vals\":" + str_min_vals + ",\"max_vals\":" + str_max_vals + "},\"output\":{\"best\":3}},\"data\":[\"" + mem_data_head + "\",\"" + mem_data + "\"]}";
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_TRUE(jd.HasMember("status"));
  ASSERT_EQ(200,jd["status"]["code"].GetInt());
  std::string cat0 = jd["body"]["predictions"][0]["classes"][0]["cat"].GetString(); // XXX: true cat is 3, which is 2 here with the label offset
  std::string cat1 = jd["body"]["predictions"][0]["classes"][1]["cat"].GetString();
  ASSERT_TRUE("2"==cat0||"2"==cat1);
  
  // predict from data, omitting header and sample id
  std::string mem_data2 = "2499,326,7,300,88,480,202,232,169,1676,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0";
  std::string str_min_vals2="[1863.0,0.0,0.0,0.0,-146.0,0.0,0.0,99.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]";
  std::string str_max_vals2="[3849.0,360.0,52.0,1343.0,554.0,6890.0,254.0,254.0,248.0,6993.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,7.0]";
  jpredictstr = "{\"service\":\""+ sname + "\",\"parameters\":{\"input\":{\"connector\":\"csv\",\"scale\":true,\"min_vals\":" + str_min_vals2 + ",\"max_vals\":" + str_max_vals2 + "},\"output\":{\"best\":3}},\"data\":[\"" + mem_data2 + "\"]}";
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_TRUE(jd.HasMember("status"));
  ASSERT_EQ(200,jd["status"]["code"].GetInt());
  cat0 = jd["body"]["predictions"][0]["classes"][0]["cat"].GetString(); // XXX: true cat is 3, which is 2 here with the label offset
  cat1 = jd["body"]["predictions"][0]["classes"][1]["cat"].GetString();
  ASSERT_TRUE("2"==cat0||"2"==cat1);
  
  // remove service
  jstr = "{\"clear\":\"lib\"}";
  joutstr = japi.jrender(japi.service_delete(sname,jstr));
  ASSERT_EQ(ok_str,joutstr);

  // assert json blob file is still there (or gone if clear=full)
  ASSERT_TRUE(!fileops::file_exists(forest_repo + "/" + JsonAPI::_json_blob_fname));
}

TEST(caffeapi,service_train_csv_in_memory)
{
  // create service
  JsonAPI japi;
  std::string sname = "my_service2";
  std::string jstr = "{\"mllib\":\"caffe\",\"description\":\"my classifier\",\"type\":\"supervised\",\"model\":{\"repository\":\"" +  forest_repo + "\",\"templates\":\"" + model_templates_repo  + "\"},\"parameters\":{\"input\":{\"connector\":\"csv\"},\"mllib\":{\"template\":\"mlp\",\"nclasses\":7}}}";
  std::string joutstr = japi.jrender(japi.service_create(sname,jstr));
  ASSERT_EQ(created_str,joutstr);

  // assert json blob file
  ASSERT_TRUE(fileops::file_exists(forest_repo + "/" + JsonAPI::_json_blob_fname));

  // read CSV file
  std::string mem_data_str;
  std::ifstream inf(forest_repo + "train.csv");
  std::string line;
  int nlines = 0;
  while(std::getline(inf,line))
    {
      if (nlines > 0)
	mem_data_str += ",";
      mem_data_str += "\"" + line + "\"";
      ++nlines;
    }
  
  // train
  std::string jtrainstr = "{\"service\":\"" + sname + "\",\"async\":false,\"parameters\":{\"input\":{\"label\":\"Cover_Type\",\"id\":\"Id\",\"scale\":true,\"test_split\":0.1,\"label_offset\":-1,\"shuffle\":true},\"mllib\":{\"gpu\":true,\"solver\":{\"iterations\":" + iterations_forest + ",\"base_lr\":0.05},\"net\":{\"batch_size\":512}},\"output\":{\"measure\":[\"acc\",\"mcll\",\"f1\",\"cmdiag\"]}},\"data\":[" + mem_data_str + "]}";
  joutstr = japi.jrender(japi.service_train(jtrainstr));
  std::cout << "joutstr=" << joutstr << std::endl;
  JDoc jd;
  jd.Parse(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_TRUE(jd.HasMember("status"));
  ASSERT_EQ(201,jd["status"]["code"].GetInt());
  ASSERT_EQ("Created",jd["status"]["msg"]);
  ASSERT_TRUE(jd.HasMember("head"));
  ASSERT_EQ("/train",jd["head"]["method"]);
  ASSERT_TRUE(jd["head"]["time"].GetDouble() >= 0);
  ASSERT_TRUE(jd.HasMember("body"));
  ASSERT_TRUE(jd["body"]["measure"].HasMember("train_loss"));
  ASSERT_TRUE(fabs(jd["body"]["measure"]["train_loss"].GetDouble()) > 0.0);
  ASSERT_TRUE(jd["body"]["measure"].HasMember("f1"));
#ifndef CPU_ONLY
  ASSERT_TRUE(jd["body"]["measure"]["f1"].GetDouble() > 0.7);
#else
  ASSERT_TRUE(jd["body"]["measure"]["f1"].GetDouble() > 0.5);
#endif
  ASSERT_EQ(jd["body"]["measure"]["accp"].GetDouble(),jd["body"]["measure"]["acc"].GetDouble());
  ASSERT_TRUE(jd["body"].HasMember("parameters"));
  ASSERT_TRUE(jd["body"]["parameters"].HasMember("input"));
  ASSERT_TRUE(jd["body"]["parameters"]["input"].HasMember("min_vals"));
  ASSERT_TRUE(jd["body"]["parameters"]["input"].HasMember("max_vals"));
  ASSERT_TRUE(jd["body"]["measure"].HasMember("cmdiag"));
  ASSERT_EQ(7,jd["body"]["measure"]["cmdiag"].Size());
  ASSERT_TRUE(jd["body"]["measure"]["cmdiag"][0].GetDouble() >= 0);
  ASSERT_EQ(56,jd["body"]["parameters"]["input"]["min_vals"].Size());
  ASSERT_EQ(56,jd["body"]["parameters"]["input"]["max_vals"].Size());
  ASSERT_EQ(504,jd["body"]["parameters"]["mllib"]["batch_size"].GetInt());
  
  // remove service
  jstr = "{\"clear\":\"lib\"}";
  joutstr = japi.jrender(japi.service_delete(sname,jstr));
  ASSERT_EQ(ok_str,joutstr);

  // assert json blob file is still there (or gone if clear=full)
  ASSERT_TRUE(!fileops::file_exists(forest_repo + "/" + JsonAPI::_json_blob_fname));
}

TEST(caffeapi,service_train_images)
{
  // create service
  JsonAPI japi;
  std::string plank_repo_loc = "plank";
  mkdir(plank_repo_loc.c_str(),0777);
  std::string sname = "my_service";
  std::string jstr = "{\"mllib\":\"caffe\",\"description\":\"my classifier\",\"type\":\"supervised\",\"model\":{\"repository\":\"" +  plank_repo_loc + "\",\"templates\":\"" + model_templates_repo  + "\"},\"parameters\":{\"input\":{\"connector\":\"image\"},\"mllib\":{\"template\":\"cifar\",\"nclasses\":121}}}";
  std::string joutstr = japi.jrender(japi.service_create(sname,jstr));
  ASSERT_EQ(created_str,joutstr);

  // train
  std::string jtrainstr = "{\"service\":\"" + sname + "\",\"async\":false,\"parameters\":{\"input\":{\"width\":32,\"height\":32,\"test_split\":0.001,\"shuffle\":true,\"bw\":false},\"mllib\":{\"gpu\":true,\"solver\":{\"iterations\":" + iterations_plank + ",\"test_interval\":500,\"base_lr\":0.0001,\"snapshot\":2000,\"test_initialization\":false},\"net\":{\"batch_size\":100}},\"output\":{\"measure\":[\"acc\",\"acc-5\",\"mcll\",\"f1\"]}},\"data\":[\"" + plank_repo + "train\"]}";
  joutstr = japi.jrender(japi.service_train(jtrainstr));
  std::cout << "joutstr=" << joutstr << std::endl;
  JDoc jd;
  jd.Parse(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_TRUE(jd.HasMember("status"));
  ASSERT_EQ(201,jd["status"]["code"].GetInt());
  ASSERT_EQ("Created",jd["status"]["msg"]);
  ASSERT_TRUE(jd.HasMember("head"));
  ASSERT_EQ("/train",jd["head"]["method"]);
  ASSERT_TRUE(jd["head"]["time"].GetDouble() >= 0);
  ASSERT_TRUE(jd.HasMember("body"));
  ASSERT_TRUE(jd["body"]["measure"].HasMember("train_loss"));
  ASSERT_TRUE(fabs(jd["body"]["measure"]["train_loss"].GetDouble()) > 0);
  ASSERT_TRUE(jd["body"]["measure"].HasMember("f1"));
  ASSERT_TRUE(jd["body"]["measure"]["acc"].GetDouble() >= 0.0);
  ASSERT_TRUE(jd["body"]["measure"]["acc-5"].GetDouble() >= 0.0);
  ASSERT_EQ(jd["body"]["measure"]["accp"].GetDouble(),jd["body"]["measure"]["acc"].GetDouble());

  // remove service
  jstr = "{\"clear\":\"full\"}";
  joutstr = japi.jrender(japi.service_delete(sname,jstr));
  ASSERT_EQ(ok_str,joutstr);
  rmdir(plank_repo_loc.c_str());
}

TEST(caffeapi,service_train_txt)
{
  // create service
  JsonAPI japi;
  std::string n20_repo_loc = "n20";
  mkdir(n20_repo_loc.c_str(),0777);
  std::string sname = "my_service";
  std::string jstr = "{\"mllib\":\"caffe\",\"description\":\"my classifier\",\"type\":\"supervised\",\"model\":{\"repository\":\"" +  n20_repo_loc + "\",\"templates\":\"" + model_templates_repo  + "\"},\"parameters\":{\"input\":{\"connector\":\"txt\"},\"mllib\":{\"template\":\"mlp\",\"nclasses\":20}}}";
  std::string joutstr = japi.jrender(japi.service_create(sname,jstr));
  ASSERT_EQ(created_str,joutstr);

  // train
  std::string jtrainstr = "{\"service\":\"" + sname + "\",\"async\":false,\"parameters\":{\"input\":{\"test_split\":0.2,\"shuffle\":true,\"min_count\":10,\"min_word_length\":3,\"count\":false},\"mllib\":{\"gpu\":true,\"solver\":{\"iterations\":" + iterations_n20 + ",\"test_interval\":200,\"base_lr\":0.05,\"snapshot\":2000,\"test_initialization\":true},\"net\":{\"batch_size\":100}},\"output\":{\"measure\":[\"acc\",\"mcll\",\"f1\"]}},\"data\":[\"" + n20_repo + "news20\"]}";
  joutstr = japi.jrender(japi.service_train(jtrainstr));
  std::cout << "joutstr=" << joutstr << std::endl;
  JDoc jd;
  jd.Parse(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_TRUE(jd.HasMember("status"));
  ASSERT_EQ(201,jd["status"]["code"].GetInt());
  ASSERT_EQ("Created",jd["status"]["msg"]);
  ASSERT_TRUE(jd.HasMember("head"));
  ASSERT_EQ("/train",jd["head"]["method"]);
  ASSERT_TRUE(jd["head"]["time"].GetDouble() >= 0);
  ASSERT_TRUE(jd.HasMember("body"));
  ASSERT_TRUE(jd["body"]["measure"].HasMember("train_loss"));
  ASSERT_TRUE(fabs(jd["body"]["measure"]["train_loss"].GetDouble()) > 0);
  ASSERT_TRUE(jd["body"]["measure"].HasMember("f1"));
  ASSERT_TRUE(jd["body"]["measure"]["acc"].GetDouble() >= 0.5);
  ASSERT_EQ(jd["body"]["measure"]["accp"].GetDouble(),jd["body"]["measure"]["acc"].GetDouble());

  // predict with measure
  std::string jpredictstr = "{\"service\":\"" + sname + "\",\"parameters\":{\"mllib\":{\"gpu\":true,\"net\":{\"test_batch_size\":10}},\"output\":{\"measure\":[\"f1\"]}},\"data\":[\"" + n20_repo +"news20\"]}";
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_TRUE(jd.HasMember("status"));
  ASSERT_EQ(200,jd["status"]["code"].GetInt());
  ASSERT_TRUE(jd.HasMember("body"));
  ASSERT_TRUE(jd["body"].HasMember("measure"));
  ASSERT_TRUE(jd["body"]["measure"]["f1"].GetDouble() >= 0.6);
  
  // remove service
  jstr = "{\"clear\":\"full\"}";
  joutstr = japi.jrender(japi.service_delete(sname,jstr));
  ASSERT_EQ(ok_str,joutstr);
  rmdir(n20_repo_loc.c_str());
  }

TEST(caffeapi,service_train_txt_sparse)
{
  // create service
  JsonAPI japi;
  std::string n20_repo_loc = "n20";
  mkdir(n20_repo_loc.c_str(),0777);
  std::string sname = "my_service";
  std::string jstr = "{\"mllib\":\"caffe\",\"description\":\"my classifier\",\"type\":\"supervised\",\"model\":{\"repository\":\"" +  n20_repo_loc + "\",\"templates\":\"" + model_templates_repo  + "\"},\"parameters\":{\"input\":{\"connector\":\"txt\",\"sparse\":true},\"mllib\":{\"template\":\"mlp\",\"nclasses\":20,\"db\":false}}}";
  std::string joutstr = japi.jrender(japi.service_create(sname,jstr));
  ASSERT_EQ(created_str,joutstr);

  // train
  std::string jtrainstr = "{\"service\":\"" + sname + "\",\"async\":false,\"parameters\":{\"input\":{\"test_split\":0.2,\"shuffle\":true,\"min_count\":10,\"min_word_length\":3,\"count\":false,\"db\":false},\"mllib\":{\"gpu\":false,\"solver\":{\"iterations\":" + iterations_n20 + ",\"test_interval\":200,\"base_lr\":0.01,\"snapshot\":2000,\"test_initialization\":true},\"net\":{\"batch_size\":100}},\"output\":{\"measure\":[\"acc\",\"mcll\",\"f1\",\"cmdiag\"]}},\"data\":[\"" + n20_repo + "news20\"]}";
  joutstr = japi.jrender(japi.service_train(jtrainstr));
  std::cout << "joutstr=" << joutstr << std::endl;
  JDoc jd;
  jd.Parse(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_TRUE(jd.HasMember("status"));
  ASSERT_EQ(201,jd["status"]["code"].GetInt());
  ASSERT_EQ("Created",jd["status"]["msg"]);
  ASSERT_TRUE(jd.HasMember("head"));
  ASSERT_EQ("/train",jd["head"]["method"]);
  ASSERT_TRUE(jd["head"]["time"].GetDouble() >= 0);
  ASSERT_TRUE(jd.HasMember("body"));
  ASSERT_TRUE(jd["body"]["measure"].HasMember("train_loss"));
  ASSERT_TRUE(fabs(jd["body"]["measure"]["train_loss"].GetDouble()) > 0);
  ASSERT_TRUE(jd["body"]["measure"].HasMember("f1"));
  ASSERT_TRUE(jd["body"]["measure"]["acc"].GetDouble() >= 0.5);
  ASSERT_EQ(jd["body"]["measure"]["accp"].GetDouble(),jd["body"]["measure"]["acc"].GetDouble());

  // predict with measure
  std::string jpredictstr = "{\"service\":\"" + sname + "\",\"parameters\":{\"mllib\":{\"gpu\":false,\"net\":{\"test_batch_size\":10}},\"output\":{\"measure\":[\"f1\"]}},\"data\":[\"" + n20_repo +"news20\"]}";
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_TRUE(jd.HasMember("status"));
  ASSERT_EQ(200,jd["status"]["code"].GetInt());
  ASSERT_TRUE(jd.HasMember("body"));
  ASSERT_TRUE(jd["body"].HasMember("measure"));
  ASSERT_TRUE(jd["body"]["measure"]["f1"].GetDouble() >= 0.6);
  ASSERT_TRUE(jd["body"]["measure"]["f1"].GetDouble() <= 1.0);

  // remove service
  jstr = "{\"clear\":\"full\"}";
  joutstr = japi.jrender(japi.service_delete(sname,jstr));
  ASSERT_EQ(ok_str,joutstr);
  rmdir(n20_repo_loc.c_str());
}

TEST(caffeapi,service_train_txt_sparse_lr)
{
  // create service
  JsonAPI japi;
  std::string n20_repo_loc = "n20";
  mkdir(n20_repo_loc.c_str(),0777);
  std::string sname = "my_service";
  std::string jstr = "{\"mllib\":\"caffe\",\"description\":\"my classifier\",\"type\":\"supervised\",\"model\":{\"repository\":\"" +  n20_repo_loc + "\",\"templates\":\"" + model_templates_repo  + "\"},\"parameters\":{\"input\":{\"connector\":\"txt\",\"sparse\":true},\"mllib\":{\"template\":\"lregression\",\"nclasses\":20,\"db\":true}}}";
  std::string joutstr = japi.jrender(japi.service_create(sname,jstr));
  ASSERT_EQ(created_str,joutstr);

  // train
  std::string jtrainstr = "{\"service\":\"" + sname + "\",\"async\":false,\"parameters\":{\"input\":{\"test_split\":0.2,\"shuffle\":true,\"min_count\":10,\"min_word_length\":3,\"count\":false,\"db\":true},\"mllib\":{\"gpu\":true,\"solver\":{\"iterations\":" + iterations_n20 + ",\"test_interval\":200,\"base_lr\":0.01,\"snapshot\":2000,\"test_initialization\":true},\"net\":{\"batch_size\":100}},\"output\":{\"measure\":[\"acc\",\"mcll\",\"f1\",\"cmdiag\"]}},\"data\":[\"" + n20_repo + "news20\"]}";
  joutstr = japi.jrender(japi.service_train(jtrainstr));
  std::cout << "joutstr=" << joutstr << std::endl;
  JDoc jd;
  jd.Parse(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_TRUE(jd.HasMember("status"));
  ASSERT_EQ(201,jd["status"]["code"].GetInt());
  ASSERT_EQ("Created",jd["status"]["msg"]);
  ASSERT_TRUE(jd.HasMember("head"));
  ASSERT_EQ("/train",jd["head"]["method"]);
  ASSERT_TRUE(jd["head"]["time"].GetDouble() >= 0);
  ASSERT_TRUE(jd.HasMember("body"));
  ASSERT_TRUE(jd["body"]["measure"].HasMember("train_loss"));
  ASSERT_TRUE(fabs(jd["body"]["measure"]["train_loss"].GetDouble()) > 0);
  ASSERT_TRUE(jd["body"]["measure"].HasMember("f1"));
  ASSERT_TRUE(jd["body"]["measure"]["acc"].GetDouble() >= 0.5);
  ASSERT_EQ(jd["body"]["measure"]["accp"].GetDouble(),jd["body"]["measure"]["acc"].GetDouble());

  // predict with measure
  std::string jpredictstr = "{\"service\":\"" + sname + "\",\"parameters\":{\"mllib\":{\"gpu\":false,\"net\":{\"test_batch_size\":10}},\"output\":{\"measure\":[\"f1\"]}},\"data\":[\"" + n20_repo +"news20\"]}";
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_TRUE(jd.HasMember("status"));
  ASSERT_EQ(200,jd["status"]["code"].GetInt());
  ASSERT_TRUE(jd.HasMember("body"));
  ASSERT_TRUE(jd["body"].HasMember("measure"));
  ASSERT_TRUE(jd["body"]["measure"]["f1"].GetDouble() >= 0.6);
  ASSERT_TRUE(jd["body"]["measure"]["f1"].GetDouble() <= 1.0);

  // remove service
  jstr = "{\"clear\":\"full\"}";
  joutstr = japi.jrender(japi.service_delete(sname,jstr));
  ASSERT_EQ(ok_str,joutstr);
  rmdir(n20_repo_loc.c_str());
}

TEST(caffeapi,service_train_txt_char)
{
  // create service
  JsonAPI japi;
  std::string n20_repo_loc = "n20";
  mkdir(n20_repo_loc.c_str(),0777);
  std::string sname = "my_service";
  std::string jstr = "{\"mllib\":\"caffe\",\"description\":\"my classifier\",\"type\":\"supervised\",\"model\":{\"repository\":\"" +  n20_repo_loc + "\",\"templates\":\"" + model_templates_repo  + "\"},\"parameters\":{\"input\":{\"connector\":\"txt\",\"characters\":true,\"sequence\":50},\"mllib\":{\"template\":\"convnet\",\"layers\":[\"1CR16\",\"1CR16\",\"1CR16\",\"512\",\"512\"],\"nclasses\":20,\"db\":true}}}";
  std::string joutstr = japi.jrender(japi.service_create(sname,jstr));
  ASSERT_EQ(created_str,joutstr);

  // train
  std::string jtrainstr = "{\"service\":\"" + sname + "\",\"async\":false,\"parameters\":{\"input\":{\"test_split\":0.2,\"shuffle\":true,\"characters\":true,\"sequence\":50,\"db\":true},\"mllib\":{\"gpu\":true,\"solver\":{\"iterations\":" + iterations_n20_char + ",\"test_interval\":30,\"base_lr\":0.01,\"snapshot\":2000,\"test_initialization\":true},\"net\":{\"batch_size\":100}},\"output\":{\"measure\":[\"acc\",\"mcll\",\"f1\"]}},\"data\":[\"" + n20_repo + "news20\"]}";
  joutstr = japi.jrender(japi.service_train(jtrainstr));
  std::cout << "joutstr=" << joutstr << std::endl;
  JDoc jd;
  jd.Parse(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_TRUE(jd.HasMember("status"));
  ASSERT_EQ(201,jd["status"]["code"].GetInt());
  ASSERT_EQ("Created",jd["status"]["msg"]);
  ASSERT_TRUE(jd.HasMember("head"));
  ASSERT_EQ("/train",jd["head"]["method"]);
  ASSERT_TRUE(jd["head"]["time"].GetDouble() >= 0);
  ASSERT_TRUE(jd.HasMember("body"));
  ASSERT_TRUE(jd["body"]["measure"].HasMember("train_loss"));
  ASSERT_TRUE(fabs(jd["body"]["measure"]["train_loss"].GetDouble()) > 0);
  ASSERT_TRUE(jd["body"]["measure"].HasMember("f1"));
  ASSERT_TRUE(jd["body"]["measure"]["acc"].GetDouble() > 0.0);
  ASSERT_EQ(jd["body"]["measure"]["accp"].GetDouble(),jd["body"]["measure"]["acc"].GetDouble());

  // remove service
  jstr = "{\"clear\":\"full\"}";
  joutstr = japi.jrender(japi.service_delete(sname,jstr));
  ASSERT_EQ(ok_str,joutstr);
  rmdir(n20_repo_loc.c_str());
}

TEST(caffeapi,service_train_csv_mt_regression)
{
  // create service
  JsonAPI japi;
  std::string sflare_repo_loc = "sflare";
  mkdir(sflare_repo_loc.c_str(),0777);
  std::string sname = "my_service";
  std::string jstr = "{\"mllib\":\"caffe\",\"description\":\"my classifier\",\"type\":\"supervised\",\"model\":{\"repository\":\"" +  sflare_repo_loc + "\",\"templates\":\"" + model_templates_repo  + "\"},\"parameters\":{\"input\":{\"connector\":\"csv\"},\"mllib\":{\"template\":\"mlp\",\"regression\":true,\"ntargets\":3,\"layers\":[150,150]}}}";
  std::string joutstr = japi.jrender(japi.service_create(sname,jstr));
  ASSERT_EQ(created_str,joutstr);

  // train
  std::string jtrainstr = "{\"service\":\"" + sname + "\",\"async\":false,\"parameters\":{\"input\":{\"test_split\":0.1,\"shuffle\":true,\"label\":[\"c_class\",\"m_class\",\"x_class\"],\"separator\":\",\",\"scale\":true,\"categoricals\":[\"class_code\",\"code_spot\",\"code_spot_distr\"]},\"mllib\":{\"gpu\":true,\"solver\":{\"iterations\":" + iterations_sflare + ",\"test_interval\":200,\"base_lr\":0.001,\"snapshot\":2000,\"test_initialization\":true},\"net\":{\"batch_size\":100}},\"output\":{\"measure\":[\"eucll\"]}},\"data\":[\"" + sflare_repo + "flare.csv\"]}";
  std::cerr << "jtrainstr=" << jtrainstr << std::endl;
  joutstr = japi.jrender(japi.service_train(jtrainstr));
  std::cout << "joutstr=" << joutstr << std::endl;
  JDoc jd;
  jd.Parse(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_TRUE(jd.HasMember("status"));
  ASSERT_EQ(201,jd["status"]["code"].GetInt());
  ASSERT_EQ("Created",jd["status"]["msg"]);
  ASSERT_TRUE(jd.HasMember("head"));
  ASSERT_EQ("/train",jd["head"]["method"]);
  ASSERT_TRUE(jd["head"]["time"].GetDouble() >= 0);
  ASSERT_TRUE(jd.HasMember("body"));
  ASSERT_TRUE(jd["body"]["measure"].HasMember("train_loss"));
  ASSERT_TRUE(fabs(jd["body"]["measure"]["train_loss"].GetDouble()) > 0);
  ASSERT_TRUE(jd["body"]["measure"].HasMember("eucll"));
  ASSERT_TRUE(jd["body"]["measure"]["eucll"].GetDouble() > 0.0);
  ASSERT_TRUE(jd["body"]["parameters"]["input"].HasMember("max_vals"));
  ASSERT_TRUE(jd["body"]["parameters"]["input"].HasMember("min_vals"));

  std::string str_min_vals = japi.jrender(jd["body"]["parameters"]["input"]["min_vals"]);
  std::string str_max_vals = japi.jrender(jd["body"]["parameters"]["input"]["max_vals"]);
  std::string str_categoricals = japi.jrender(jd["body"]["parameters"]["input"]["categoricals_mapping"]);
  std::cerr << "categoricals=" << str_categoricals << std::endl;
  
  // predict
  std::string sflare_data_head = "class_code,code_spot,code_spot_distr,act,evo,prev_act,hist,reg,area,larg_area,x,y,z";
  std::string sflare_data = "B,X,O,1,2,1,1,2,1,1,0,0,0";
  std::string jpredictstr = "{\"service\":\""+ sname + "\",\"parameters\":{\"input\":{\"connector\":\"csv\",\"scale\":true,\"min_vals\":" + str_min_vals + ",\"max_vals\":" + str_max_vals + ",\"categoricals_mapping\":" + str_categoricals + "},\"output\":{}},\"data\":[\"" + sflare_data_head + "\",\"" + sflare_data + "\"]}";
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200,jd["status"]["code"]);
  std::string uri = jd["body"]["predictions"][0]["uri"].GetString();
  ASSERT_EQ("1",uri);
  ASSERT_TRUE(jd["body"]["predictions"][0]["vector"].IsArray());
  ASSERT_TRUE(jd["body"]["predictions"][0]["vector"][0]["val"].GetDouble() > 0.0);
  
  // remove service
  jstr = "{\"clear\":\"full\"}";
  joutstr = japi.jrender(japi.service_delete(sname,jstr));
  ASSERT_EQ(ok_str,joutstr);
  rmdir(sflare_repo_loc.c_str());
}
