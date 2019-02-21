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
static std::string farm_repo = "../examples/all/farm_ads/";
static std::string plank_repo = "../examples/caffe/plankton/";
static std::string voc_repo = "../examples/caffe/voc/voc/";
static std::string n20_repo = "../examples/all/n20/";
static std::string sinus = "../examples/all/sinus/";
static std::string sflare_repo = "../examples/all/sflare/";
static std::string camvid_repo = "../examples/caffe/camvid/CamVid_square/";
static std::string model_templates_repo = "../templates/caffe/";

#ifndef CPU_ONLY
static std::string iterations_mnist = "250";
static std::string iterations_plank = "2000";
static std::string iterations_forest = "3000";
static std::string iterations_farm = "1000";
static std::string iterations_n20 = "2000";
static std::string iterations_n20_char = "1000";
static std::string iterations_sflare = "5000";
static std::string iterations_camvid = "600";
static std::string iterations_lstm = "200";
#else
static std::string iterations_mnist = "10";
static std::string iterations_plank = "10";
static std::string iterations_forest = "500";
static std::string iterations_farm = "500";
static std::string iterations_n20 = "1000";
static std::string iterations_n20_char = "10";
static std::string iterations_sflare = "2000";
static std::string iterations_camvid = "2";
static std::string iterations_lstm = "20";
#endif

static std::string gpuid = "0"; // change as needed

TEST(caffeapi,service_train)
{
// create service
  JsonAPI japi;
  std::string sname = "my_service";
  std::string jstr = "{\"mllib\":\"caffe\",\"description\":\"my classifier\",\"type\":\"supervised\",\"model\":{\"repository\":\"" +  mnist_repo + "\"},\"parameters\":{\"input\":{\"connector\":\"image\"},\"mllib\":{\"nclasses\":10}}}";
  std::string joutstr = japi.jrender(japi.service_create(sname,jstr));
  ASSERT_EQ(created_str,joutstr);

  // train
  std::string jtrainstr = "{\"service\":\"" + sname + "\",\"async\":false,\"parameters\":{\"mllib\":{\"gpu\":true,\"gpuid\":"+gpuid+",\"solver\":{\"iterations\":" + iterations_mnist + "}}}}";
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
  std::string jtrainstr = "{\"service\":\"" + sname + "\",\"async\":true,\"parameters\":{\"mllib\":{\"gpu\":true,\"gpuid\":"+gpuid+",\"solver\":{\"iterations\":10000}}}}";
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
  std::string jtrainstr = "{\"service\":\"" + sname + "\",\"async\":true,\"parameters\":{\"mllib\":{\"gpu\":true,\"gpuid\":"+gpuid+",\"solver\":{\"iterations\":" + iterations_mnist + "}}}}";
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
  std::string jtrainstr = "{\"service\":\"" + sname + "\",\"async\":true,\"parameters\":{\"mllib\":{\"gpu\":true,\"gpuid\":"+gpuid+",\"solver\":{\"iterations\":" + iterations + "}}}}";
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
  std::string jtrainstr = "{\"service\":\"" + sname + "\",\"async\":false,\"parameters\":{\"mllib\":{\"gpu\":true,\"gpuid\":"+gpuid+",\"solver\":{\"iterations\":" + iterations_mnist + ",\"snapshot\":200,\"snapshot_prefix\":\"" + mnist_repo + "/mylenet\",\"test_interval\":2}},\"output\":{\"measure_hist\":true,\"measure\":[\"f1\"]}}}";
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
  ASSERT_EQ(jd["body"]["measure_hist"]["iteration_hist"].Size(),jd["body"]["measure_hist"]["f1_hist"].Size());

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

  // lmdb predict
  jpredictstr = "{\"service\":\""+ sname + "\",\"parameters\":{\"input\":{\"bw\":true,\"width\":28,\"height\":28},\"output\":{\"measure\":[\"f1\"]},\"mllib\":{\"net\":{\"test_batch_size\":100}}},\"data\":[\"" + mnist_repo + "/test.lmdb\"]}";
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200,jd["status"]["code"]);
  ASSERT_TRUE(jd["body"]["measure"]["f1"].GetDouble() > 0.0);

  jpredictstr = "{\"service\":\""+ sname + "\",\"parameters\":{\"input\":{\"bw\":true,\"width\":28,\"height\":28},\"output\":{},\"mllib\":{\"net\":{\"test_batch_size\":100}}},\"data\":[\"" + mnist_repo + "/test.lmdb\"]}";
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200,jd["status"]["code"]);
  uri = jd["body"]["predictions"][0]["uri"].GetString();
  ASSERT_EQ(uri,"00000000");

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
  std::string jtrainstr = "{\"service\":\"" + sname + "\",\"async\":false,\"parameters\":{\"input\":{\"label\":\"Cover_Type\",\"id\":\"Id\",\"scale\":true,\"test_split\":0.1,\"label_offset\":-1,\"shuffle\":true},\"mllib\":{\"gpu\":true,\"gpuid\":"+gpuid+",\"solver\":{\"iterations\":" + iterations_forest + ",\"base_lr\":0.05},\"net\":{\"batch_size\":512}},\"output\":{\"measure\":[\"acc\",\"mcll\",\"f1\",\"cmdiag\",\"cmfull\"]}},\"data\":[\"" + forest_repo + "train.csv\"]}";
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
  ASSERT_TRUE(!fileops::remove_directory_files(forest_repo,{".prototxt"}));
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
  std::string jtrainstr = "{\"service\":\"" + sname + "\",\"async\":false,\"parameters\":{\"input\":{\"label\":\"Cover_Type\",\"id\":\"Id\",\"scale\":true,\"test_split\":0.1,\"label_offset\":-1,\"shuffle\":true},\"mllib\":{\"gpu\":true,\"gpuid\":"+gpuid+",\"solver\":{\"iterations\":" + iterations_forest + ",\"base_lr\":0.05},\"net\":{\"batch_size\":512}},\"output\":{\"measure\":[\"acc\",\"mcll\",\"f1\",\"cmdiag\"]}},\"data\":[" + mem_data_str + "]}";
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
  ASSERT_TRUE(!fileops::remove_directory_files(forest_repo,{".prototxt"}));
}

TEST(caffeapi,service_train_csv_resnet)
{
  // create service
  JsonAPI japi;
  std::string sname = "my_service";
  std::string jstr = "{\"mllib\":\"caffe\",\"description\":\"my classifier\",\"type\":\"supervised\",\"model\":{\"repository\":\"" +  forest_repo + "\",\"templates\":\"" + model_templates_repo  + "\"},\"parameters\":{\"input\":{\"connector\":\"csv\"},\"mllib\":{\"template\":\"resnet\",\"nclasses\":7,\"activation\":\"relu\",\"layers\":[300,100,50],\"bn\":true,\"gpu\":false}}}";
  std::string joutstr = japi.jrender(japi.service_create(sname,jstr));
  ASSERT_EQ(created_str,joutstr);

  // assert json blob file
  ASSERT_TRUE(fileops::file_exists(forest_repo + "/" + JsonAPI::_json_blob_fname));
  
  // train
  std::string jtrainstr = "{\"service\":\"" + sname + "\",\"async\":false,\"parameters\":{\"input\":{\"label\":\"Cover_Type\",\"id\":\"Id\",\"scale\":true,\"test_split\":0.1,\"label_offset\":-1,\"shuffle\":true},\"mllib\":{\"gpu\":true,\"gpuid\":"+gpuid+",\"solver\":{\"iterations\":" + iterations_forest + ",\"base_lr\":0.001},\"net\":{\"batch_size\":512}},\"output\":{\"measure\":[\"acc\",\"mcll\",\"f1\",\"cmdiag\",\"cmfull\"]}},\"data\":[\"" + forest_repo + "train.csv\"]}";
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
  ASSERT_TRUE(!fileops::remove_directory_files(forest_repo,{".prototxt"}));
}

TEST(caffeapi,service_train_svm)
{
  // create service
  JsonAPI japi;
  std::string sname = "my_service";
  std::string jstr = "{\"mllib\":\"caffe\",\"description\":\"my classifier\",\"type\":\"supervised\",\"model\":{\"repository\":\"" +  farm_repo + "\",\"templates\":\"" + model_templates_repo  + "\"},\"parameters\":{\"input\":{\"connector\":\"svm\"},\"mllib\":{\"template\":\"mlp\",\"nclasses\":2,\"activation\":\"prelu\"}}}";
  std::string joutstr = japi.jrender(japi.service_create(sname,jstr));
  ASSERT_EQ(created_str,joutstr);

  // assert json blob file
  ASSERT_TRUE(fileops::file_exists(farm_repo + "/" + JsonAPI::_json_blob_fname));
  
  // train
  std::string jtrainstr = "{\"service\":\"" + sname + "\",\"async\":false,\"parameters\":{\"input\":{\"test_split\":0.1,\"shuffle\":true},\"mllib\":{\"gpu\":true,\"gpuid\":"+gpuid+",\"solver\":{\"iterations\":" + iterations_farm + ",\"base_lr\":0.01},\"net\":{\"batch_size\":100}},\"output\":{\"measure\":[\"acc\",\"mcll\",\"f1\",\"cmdiag\",\"cmfull\"]}},\"data\":[\"" + farm_repo + "farm-ads.svm\"]}";
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
  ASSERT_TRUE(jd["body"]["measure"]["f1"].GetDouble() > 0.8);
#else
  ASSERT_TRUE(jd["body"]["measure"]["f1"].GetDouble() > 0.7);
#endif
  ASSERT_EQ(jd["body"]["measure"]["accp"].GetDouble(),jd["body"]["measure"]["acc"].GetDouble());
  ASSERT_TRUE(jd["body"].HasMember("parameters"));
  ASSERT_TRUE(jd["body"]["measure"].HasMember("cmdiag"));
  ASSERT_EQ(2,jd["body"]["measure"]["cmdiag"].Size());
  ASSERT_TRUE(jd["body"]["measure"]["cmdiag"][0].GetDouble() >= 0);
  ASSERT_TRUE(jd["body"]["measure"]["cmfull"].Size());
  ASSERT_EQ(16,jd["body"]["parameters"]["mllib"]["batch_size"].GetInt());

  std::string mem_data = "8:1 9:1 23:1 31:1 32:1 34:1 45:1 46:1 49:1 50:1 52:1 54:1 57:1 60:1 61:1 64:1 70:1 80:1 82:1 84:1 87:1 91:1 94:1 95:1 101:1 104:1 106:1 107:1 113:1 115:1 125:1 126:1 127:1 128:1 129:1 131:1 132:1 134:1 152:1 161:1 166:1 167:1 179:1 229:1 235:1 248:1 251:1 255:1 260:1 262:1 267:1 268:1 270:1 272:1 273:1 277:1 280:1 281:1 282:1 285:1 299:1 305:1 310:1 311:1 324:1 342:1 372:1 379:1 390:1 391:1 408:1 409:1 415:1 472:1 483:1 486:1 498:1 525:1 538:1 548:1 583:1 614:1 616:1 620:1 621:1 651:1 678:1 700:1 712:1 715:1 794:1 796:1 800:1 804:1 805:1 811:1 819:1 847:1 848:1 853:1 873:1 885:1 899:1 949:1 963:1 964:1 965:1 966:1 967:1 968:1 969:1 970:1 971:1 972:1 973:1 974:1 975:1 976:1 977:1 978:1 979:1 980:1 981:1 982:1 983:1 984:1 985:1 986:1 987:1 988:1 989:1 990:1 991:1 992:1 993:1 994:1 995:1 996:1 997:1 998:1 999:1 1000:1 1001:1 1002:1 1003:1 1004:1 1005:1 1006:1 1007:1 1008:1 1009:1 1010:1 1011:1 1012:1 1013:1 1014:1 1015:1 1016:1 1017:1 1018:1 1019:1 1020:1 1021:1 1022:1 1023:1 1024:1 1025:1 1026:1 1027:1 1028:1 1029:1 1030:1 1031:1 1032:1 1033:1 1034:1 1035:1 1036:1 1037:1 1038:1 1039:1 1040:1 1041:1 1042:1 1043:1 1044:1 1045:1 1046:1 1047:1 1048:1 1049:1 1051:1 1052:1 1053:1 1054:1 1055:1 1056:1 1057:1 1058:1 1059:1 1060:1 1061:1 1062:1 1063:1 1064:1 1065:1 1066:1 1067:1 1068:1 1069:1 1070:1 1071:1 1072:1 1073:1 1074:1 1075:1 1076:1 1077:1 1078:1 1079:1 1080:1 1081:1 1082:1 1083:1 1084:1 1085:1 1086:1 1087:1 1088:1 1089:1 1090:1 1091:1 1092:1 1093:1 1094:1 1095:1 1096:1 1097:1 1098:1 1099:1 1100:1 1101:1 1102:1 1103:1 1104:1 1105:1 1106:1 1107:1 1108:1 1109:1 1110:1 1111:1 1112:1 1113:1 1114:1 1115:1 1116:1 1117:1 1118:1 1119:1 1120:1 1121:1 1122:1 1123:1 1124:1 1125:1 1126:1 1127:1 1128:1 1129:1 1130:1 1131:1 1132:1 1133:1 1134:1 1135:1 1136:1 1137:1 1138:1 1139:1 1140:1 1141:1 1142:1 1143:1 1144:1 1145:1 1146:1 1147:1 1148:1 1149:1 1150:1 1151:1 1152:1 1153:1 1154:1 1155:1 1156:1 1157:1 1158:1 1159:1 1160:1 1161:1 1162:1 1163:1 1164:1 1165:1 1166:1 1167:1 1168:1 1169:1 1170:1 1171:1 1172:1 1173:1 1174:1 1175:1 1176:1 1177:1 1178:1 1179:1 1180:1 1181:1 1182:1 1183:1 1184:1 1185:1 1186:1 1187:1 1188:1 1189:1 1190:1 1191:1 1192:1 1193:1 1194:1 1195:1 1196:1 1197:1 1198:1 1199:1 1200:1 1201:1 1202:1 1203:1 1204:1 1205:1 1206:1 1207:1 1208:1 1209:1 1210:1 1211:1 1212:1 1213:1 1214:1 1215:1 1216:1 1217:1 1218:1 1219:1 1220:1 1221:1 1222:1 1223:1 1224:1 1225:1 1226:1 1227:1 1228:1 1229:1 1230:1 1231:1 1232:1 1233:1 1234:1 1235:1 1236:1 1237:1 1238:1 1239:1 1240:1 1241:1 1242:1 1243:1 1244:1 1245:1 1246:1 1247:1 1248:1 1249:1 1250:1 1251:1 1252:1 1253:1 1254:1 1255:1 1256:1 1257:1 1258:1 1262:1 1268:1 1269:1 1270:1 1271:1 2679:1 3031:1 3204:1 4065:1 4795:1 4960:1 4961:1 4962:1 4963:1 4964:1 4965:1 4966:1 4967:11 8:1 9:1 23:1 31:1 32:1 34:1 45:1 46:1 49:1 50:1 52:1 54:1 57:1 60:1 61:1 64:1 70:1 80:1 82:1 84:1 87:1 91:1 94:1 95:1 101:1 104:1 106:1 107:1 113:1 115:1 125:1 126:1 127:1 128:1 129:1 131:1 132:1 134:1 152:1 161:1 166:1 167:1 179:1 229:1 235:1 248:1 251:1 255:1 260:1 262:1 267:1 268:1 270:1 272:1 273:1 277:1 280:1 281:1 282:1 285:1 299:1 305:1 310:1 311:1 324:1 342:1 372:1 379:1 390:1 391:1 408:1 409:1 415:1 472:1 483:1 486:1 498:1 525:1 538:1 548:1 583:1 614:1 616:1 620:1 621:1 651:1 678:1 700:1 712:1 715:1 794:1 796:1 800:1 804:1 805:1 811:1 819:1 847:1 848:1 853:1 873:1 885:1 899:1 949:1 963:1 964:1 965:1 966:1 967:1 968:1 969:1 970:1 971:1 972:1 973:1 974:1 975:1 976:1 977:1 978:1 979:1 980:1 981:1 982:1 983:1 984:1 985:1 986:1 987:1 988:1 989:1 990:1 991:1 992:1 993:1 994:1 995:1 996:1 997:1 998:1 999:1 1000:1 1001:1 1002:1 1003:1 1004:1 1005:1 1006:1 1007:1 1008:1 1009:1 1010:1 1011:1 1012:1 1013:1 1014:1 1015:1 1016:1 1017:1 1018:1 1019:1 1020:1 1021:1 1022:1 1023:1 1024:1 1025:1 1026:1 1027:1 1028:1 1029:1 1030:1 1031:1 1032:1 1033:1 1034:1 1035:1 1036:1 1037:1 1038:1 1039:1 1040:1 1041:1 1042:1 1043:1 1044:1 1045:1 1046:1 1047:1 1048:1 1049:1 1051:1 1052:1 1053:1 1054:1 1055:1 1056:1 1057:1 1058:1 1059:1 1060:1 1061:1 1062:1 1063:1 1064:1 1065:1 1066:1 1067:1 1068:1 1069:1 1070:1 1071:1 1072:1 1073:1 1074:1 1075:1 1076:1 1077:1 1078:1 1079:1 1080:1 1081:1 1082:1 1083:1 1084:1 1085:1 1086:1 1087:1 1088:1 1089:1 1090:1 1091:1 1092:1 1093:1 1094:1 1095:1 1096:1 1097:1 1098:1 1099:1 1100:1 1101:1 1102:1 1103:1 1104:1 1105:1 1106:1 1107:1 1108:1 1109:1 1110:1 1111:1 1112:1 1113:1 1114:1 1115:1 1116:1 1117:1 1118:1 1119:1 1120:1 1121:1 1122:1 1123:1 1124:1 1125:1 1126:1 1127:1 1128:1 1129:1 1130:1 1131:1 1132:1 1133:1 1134:1 1135:1 1136:1 1137:1 1138:1 1139:1 1140:1 1141:1 1142:1 1143:1 1144:1 1145:1 1146:1 1147:1 1148:1 1149:1 1150:1 1151:1 1152:1 1153:1 1154:1 1155:1 1156:1 1157:1 1158:1 1159:1 1160:1 1161:1 1162:1 1163:1 1164:1 1165:1 1166:1 1167:1 1168:1 1169:1 1170:1 1171:1 1172:1 1173:1 1174:1 1175:1 1176:1 1177:1 1178:1 1179:1 1180:1 1181:1 1182:1 1183:1 1184:1 1185:1 1186:1 1187:1 1188:1 1189:1 1190:1 1191:1 1192:1 1193:1 1194:1 1195:1 1196:1 1197:1 1198:1 1199:1 1200:1 1201:1 1202:1 1203:1 1204:1 1205:1 1206:1 1207:1 1208:1 1209:1 1210:1 1211:1 1212:1 1213:1 1214:1 1215:1 1216:1 1217:1 1218:1 1219:1 1220:1 1221:1 1222:1 1223:1 1224:1 1225:1 1226:1 1227:1 1228:1 1229:1 1230:1 1231:1 1232:1 1233:1 1234:1 1235:1 1236:1 1237:1 1238:1 1239:1 1240:1 1241:1 1242:1 1243:1 1244:1 1245:1 1246:1 1247:1 1248:1 1249:1 1250:1 1251:1 1252:1 1253:1 1254:1 1255:1 1256:1 1257:1 1258:1 1262:1 1268:1 1269:1 1270:1 1271:1 2679:1 3031:1 3204:1 4065:1 4795:1 4960:1 4961:1 4962:1 4963:1 4964:1 4965:1 4966:1 4967:1";
  std::string jpredictstr = "{\"service\":\""+ sname + "\",\"parameters\":{\"input\":{\"connector\":\"svm\"},\"output\":{\"best\":3}},\"data\":[\"" + mem_data + "\"]}";
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_TRUE(jd.HasMember("status"));
  ASSERT_EQ(200,jd["status"]["code"].GetInt());
  std::string cat0 = jd["body"]["predictions"][0]["classes"][0]["cat"].GetString();
  ASSERT_TRUE("1"==cat0);
  
  // remove service
  jstr = "{\"clear\":\"lib\"}";
  joutstr = japi.jrender(japi.service_delete(sname,jstr));
  ASSERT_EQ(ok_str,joutstr);

  // assert json blob file is still there (or gone if clear=full)
  ASSERT_TRUE(!fileops::file_exists(farm_repo + "/" + JsonAPI::_json_blob_fname));
  ASSERT_TRUE(!fileops::remove_directory_files(farm_repo,{".prototxt"}));
}

TEST(caffeapi,service_train_svm_resnet)
{
  // create service
  JsonAPI japi;
  std::string sname = "my_service";
  std::string jstr = "{\"mllib\":\"caffe\",\"description\":\"my classifier\",\"type\":\"supervised\",\"model\":{\"repository\":\"" +  farm_repo + "\",\"templates\":\"" + model_templates_repo  + "\"},\"parameters\":{\"input\":{\"connector\":\"svm\"},\"mllib\":{\"template\":\"resnet\",\"nclasses\":2,\"activation\":\"prelu\",\"layers\":[30,25,15]}}}";
  std::string joutstr = japi.jrender(japi.service_create(sname,jstr));
  ASSERT_EQ(created_str,joutstr);

  // assert json blob file
  ASSERT_TRUE(fileops::file_exists(farm_repo + "/" + JsonAPI::_json_blob_fname));
  
  // train
  std::string jtrainstr = "{\"service\":\"" + sname + "\",\"async\":false,\"parameters\":{\"input\":{\"test_split\":0.1,\"shuffle\":true},\"mllib\":{\"gpu\":true,\"gpuid\":"+gpuid+",\"solver\":{\"iterations\":" + iterations_farm + ",\"base_lr\":0.01},\"net\":{\"batch_size\":16}},\"output\":{\"measure\":[\"acc\",\"mcll\",\"f1\",\"cmdiag\",\"cmfull\"]}},\"data\":[\"" + farm_repo + "farm-ads.svm\"]}";
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
  ASSERT_TRUE(jd["body"]["measure"]["f1"].GetDouble() > 0.8);
#else
  ASSERT_TRUE(jd["body"]["measure"]["f1"].GetDouble() > 0.7);
#endif
  ASSERT_EQ(jd["body"]["measure"]["accp"].GetDouble(),jd["body"]["measure"]["acc"].GetDouble());
    ASSERT_TRUE(jd["body"]["measure"].HasMember("cmdiag"));
  ASSERT_EQ(2,jd["body"]["measure"]["cmdiag"].Size());
  ASSERT_TRUE(jd["body"]["measure"]["cmdiag"][0].GetDouble() >= 0);
  ASSERT_TRUE(jd["body"]["measure"]["cmfull"].Size());
  
  std::string mem_data = "8:1 9:1 23:1 31:1 32:1 34:1 45:1 46:1 49:1 50:1 52:1 54:1 57:1 60:1 61:1 64:1 70:1 80:1 82:1 84:1 87:1 91:1 94:1 95:1 101:1 104:1 106:1 107:1 113:1 115:1 125:1 126:1 127:1 128:1 129:1 131:1 132:1 134:1 152:1 161:1 166:1 167:1 179:1 229:1 235:1 248:1 251:1 255:1 260:1 262:1 267:1 268:1 270:1 272:1 273:1 277:1 280:1 281:1 282:1 285:1 299:1 305:1 310:1 311:1 324:1 342:1 372:1 379:1 390:1 391:1 408:1 409:1 415:1 472:1 483:1 486:1 498:1 525:1 538:1 548:1 583:1 614:1 616:1 620:1 621:1 651:1 678:1 700:1 712:1 715:1 794:1 796:1 800:1 804:1 805:1 811:1 819:1 847:1 848:1 853:1 873:1 885:1 899:1 949:1 963:1 964:1 965:1 966:1 967:1 968:1 969:1 970:1 971:1 972:1 973:1 974:1 975:1 976:1 977:1 978:1 979:1 980:1 981:1 982:1 983:1 984:1 985:1 986:1 987:1 988:1 989:1 990:1 991:1 992:1 993:1 994:1 995:1 996:1 997:1 998:1 999:1 1000:1 1001:1 1002:1 1003:1 1004:1 1005:1 1006:1 1007:1 1008:1 1009:1 1010:1 1011:1 1012:1 1013:1 1014:1 1015:1 1016:1 1017:1 1018:1 1019:1 1020:1 1021:1 1022:1 1023:1 1024:1 1025:1 1026:1 1027:1 1028:1 1029:1 1030:1 1031:1 1032:1 1033:1 1034:1 1035:1 1036:1 1037:1 1038:1 1039:1 1040:1 1041:1 1042:1 1043:1 1044:1 1045:1 1046:1 1047:1 1048:1 1049:1 1051:1 1052:1 1053:1 1054:1 1055:1 1056:1 1057:1 1058:1 1059:1 1060:1 1061:1 1062:1 1063:1 1064:1 1065:1 1066:1 1067:1 1068:1 1069:1 1070:1 1071:1 1072:1 1073:1 1074:1 1075:1 1076:1 1077:1 1078:1 1079:1 1080:1 1081:1 1082:1 1083:1 1084:1 1085:1 1086:1 1087:1 1088:1 1089:1 1090:1 1091:1 1092:1 1093:1 1094:1 1095:1 1096:1 1097:1 1098:1 1099:1 1100:1 1101:1 1102:1 1103:1 1104:1 1105:1 1106:1 1107:1 1108:1 1109:1 1110:1 1111:1 1112:1 1113:1 1114:1 1115:1 1116:1 1117:1 1118:1 1119:1 1120:1 1121:1 1122:1 1123:1 1124:1 1125:1 1126:1 1127:1 1128:1 1129:1 1130:1 1131:1 1132:1 1133:1 1134:1 1135:1 1136:1 1137:1 1138:1 1139:1 1140:1 1141:1 1142:1 1143:1 1144:1 1145:1 1146:1 1147:1 1148:1 1149:1 1150:1 1151:1 1152:1 1153:1 1154:1 1155:1 1156:1 1157:1 1158:1 1159:1 1160:1 1161:1 1162:1 1163:1 1164:1 1165:1 1166:1 1167:1 1168:1 1169:1 1170:1 1171:1 1172:1 1173:1 1174:1 1175:1 1176:1 1177:1 1178:1 1179:1 1180:1 1181:1 1182:1 1183:1 1184:1 1185:1 1186:1 1187:1 1188:1 1189:1 1190:1 1191:1 1192:1 1193:1 1194:1 1195:1 1196:1 1197:1 1198:1 1199:1 1200:1 1201:1 1202:1 1203:1 1204:1 1205:1 1206:1 1207:1 1208:1 1209:1 1210:1 1211:1 1212:1 1213:1 1214:1 1215:1 1216:1 1217:1 1218:1 1219:1 1220:1 1221:1 1222:1 1223:1 1224:1 1225:1 1226:1 1227:1 1228:1 1229:1 1230:1 1231:1 1232:1 1233:1 1234:1 1235:1 1236:1 1237:1 1238:1 1239:1 1240:1 1241:1 1242:1 1243:1 1244:1 1245:1 1246:1 1247:1 1248:1 1249:1 1250:1 1251:1 1252:1 1253:1 1254:1 1255:1 1256:1 1257:1 1258:1 1262:1 1268:1 1269:1 1270:1 1271:1 2679:1 3031:1 3204:1 4065:1 4795:1 4960:1 4961:1 4962:1 4963:1 4964:1 4965:1 4966:1 4967:11 8:1 9:1 23:1 31:1 32:1 34:1 45:1 46:1 49:1 50:1 52:1 54:1 57:1 60:1 61:1 64:1 70:1 80:1 82:1 84:1 87:1 91:1 94:1 95:1 101:1 104:1 106:1 107:1 113:1 115:1 125:1 126:1 127:1 128:1 129:1 131:1 132:1 134:1 152:1 161:1 166:1 167:1 179:1 229:1 235:1 248:1 251:1 255:1 260:1 262:1 267:1 268:1 270:1 272:1 273:1 277:1 280:1 281:1 282:1 285:1 299:1 305:1 310:1 311:1 324:1 342:1 372:1 379:1 390:1 391:1 408:1 409:1 415:1 472:1 483:1 486:1 498:1 525:1 538:1 548:1 583:1 614:1 616:1 620:1 621:1 651:1 678:1 700:1 712:1 715:1 794:1 796:1 800:1 804:1 805:1 811:1 819:1 847:1 848:1 853:1 873:1 885:1 899:1 949:1 963:1 964:1 965:1 966:1 967:1 968:1 969:1 970:1 971:1 972:1 973:1 974:1 975:1 976:1 977:1 978:1 979:1 980:1 981:1 982:1 983:1 984:1 985:1 986:1 987:1 988:1 989:1 990:1 991:1 992:1 993:1 994:1 995:1 996:1 997:1 998:1 999:1 1000:1 1001:1 1002:1 1003:1 1004:1 1005:1 1006:1 1007:1 1008:1 1009:1 1010:1 1011:1 1012:1 1013:1 1014:1 1015:1 1016:1 1017:1 1018:1 1019:1 1020:1 1021:1 1022:1 1023:1 1024:1 1025:1 1026:1 1027:1 1028:1 1029:1 1030:1 1031:1 1032:1 1033:1 1034:1 1035:1 1036:1 1037:1 1038:1 1039:1 1040:1 1041:1 1042:1 1043:1 1044:1 1045:1 1046:1 1047:1 1048:1 1049:1 1051:1 1052:1 1053:1 1054:1 1055:1 1056:1 1057:1 1058:1 1059:1 1060:1 1061:1 1062:1 1063:1 1064:1 1065:1 1066:1 1067:1 1068:1 1069:1 1070:1 1071:1 1072:1 1073:1 1074:1 1075:1 1076:1 1077:1 1078:1 1079:1 1080:1 1081:1 1082:1 1083:1 1084:1 1085:1 1086:1 1087:1 1088:1 1089:1 1090:1 1091:1 1092:1 1093:1 1094:1 1095:1 1096:1 1097:1 1098:1 1099:1 1100:1 1101:1 1102:1 1103:1 1104:1 1105:1 1106:1 1107:1 1108:1 1109:1 1110:1 1111:1 1112:1 1113:1 1114:1 1115:1 1116:1 1117:1 1118:1 1119:1 1120:1 1121:1 1122:1 1123:1 1124:1 1125:1 1126:1 1127:1 1128:1 1129:1 1130:1 1131:1 1132:1 1133:1 1134:1 1135:1 1136:1 1137:1 1138:1 1139:1 1140:1 1141:1 1142:1 1143:1 1144:1 1145:1 1146:1 1147:1 1148:1 1149:1 1150:1 1151:1 1152:1 1153:1 1154:1 1155:1 1156:1 1157:1 1158:1 1159:1 1160:1 1161:1 1162:1 1163:1 1164:1 1165:1 1166:1 1167:1 1168:1 1169:1 1170:1 1171:1 1172:1 1173:1 1174:1 1175:1 1176:1 1177:1 1178:1 1179:1 1180:1 1181:1 1182:1 1183:1 1184:1 1185:1 1186:1 1187:1 1188:1 1189:1 1190:1 1191:1 1192:1 1193:1 1194:1 1195:1 1196:1 1197:1 1198:1 1199:1 1200:1 1201:1 1202:1 1203:1 1204:1 1205:1 1206:1 1207:1 1208:1 1209:1 1210:1 1211:1 1212:1 1213:1 1214:1 1215:1 1216:1 1217:1 1218:1 1219:1 1220:1 1221:1 1222:1 1223:1 1224:1 1225:1 1226:1 1227:1 1228:1 1229:1 1230:1 1231:1 1232:1 1233:1 1234:1 1235:1 1236:1 1237:1 1238:1 1239:1 1240:1 1241:1 1242:1 1243:1 1244:1 1245:1 1246:1 1247:1 1248:1 1249:1 1250:1 1251:1 1252:1 1253:1 1254:1 1255:1 1256:1 1257:1 1258:1 1262:1 1268:1 1269:1 1270:1 1271:1 2679:1 3031:1 3204:1 4065:1 4795:1 4960:1 4961:1 4962:1 4963:1 4964:1 4965:1 4966:1 4967:1";
  std::string jpredictstr = "{\"service\":\""+ sname + "\",\"parameters\":{\"input\":{\"connector\":\"svm\"},\"output\":{\"best\":3}},\"data\":[\"" + mem_data + "\"]}";
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_TRUE(jd.HasMember("status"));
  ASSERT_EQ(200,jd["status"]["code"].GetInt());
  std::string cat0 = jd["body"]["predictions"][0]["classes"][0]["cat"].GetString();
  ASSERT_TRUE("1"==cat0);
  
  // remove service
  jstr = "{\"clear\":\"lib\"}";
  joutstr = japi.jrender(japi.service_delete(sname,jstr));
  ASSERT_EQ(ok_str,joutstr);

  // assert json blob file is still there (or gone if clear=full)
  ASSERT_TRUE(!fileops::file_exists(farm_repo + "/" + JsonAPI::_json_blob_fname));
  ASSERT_TRUE(!fileops::remove_directory_files(farm_repo,{".prototxt"}));
}

TEST(caffeapi,service_train_images)
{
  // create service
  JsonAPI japi;
  std::string plank_repo_loc = "plank";
  std::string plank_repo_loc2 = "plank2";
  mkdir(plank_repo_loc.c_str(),0777);
  std::string sname = "my_service";
  std::string jstr = "{\"mllib\":\"caffe\",\"description\":\"my classifier\",\"type\":\"supervised\",\"model\":{\"repository\":\"" +  plank_repo_loc + "\",\"templates\":\"" + model_templates_repo  + "\"},\"parameters\":{\"input\":{\"connector\":\"image\"},\"mllib\":{\"db\":true,\"template\":\"cifar\",\"nclasses\":121}}}";
  std::string joutstr = japi.jrender(japi.service_create(sname,jstr));
  ASSERT_EQ(created_str,joutstr);

  // train
  std::string jtrainstr = "{\"service\":\"" + sname + "\",\"async\":false,\"parameters\":{\"input\":{\"db\":true,\"width\":32,\"height\":32,\"test_split\":0.001,\"shuffle\":true,\"bw\":false},\"mllib\":{\"gpu\":true,\"gpuid\":"+gpuid+",\"solver\":{\"iterations\":" + iterations_plank + ",\"test_interval\":500,\"base_lr\":0.0001,\"snapshot\":2000,\"test_initialization\":true},\"net\":{\"batch_size\":100}},\"output\":{\"measure\":[\"acc\",\"acc-5\",\"mcll\",\"f1\"],\"target_repository\":\"" + plank_repo_loc2 + "\"}},\"data\":[\"" + plank_repo + "train\"]}";
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
  ASSERT_TRUE(fileops::file_exists(plank_repo_loc + "/best_model.txt"));
  ASSERT_TRUE(fileops::file_exists(plank_repo_loc2 + "/deploy.prototxt"));
  
  // remove service
  jstr = "{\"clear\":\"full\"}";
  joutstr = japi.jrender(japi.service_delete(sname,jstr));
  ASSERT_EQ(ok_str,joutstr);
  rmdir(plank_repo_loc.c_str());
  rmdir(plank_repo_loc2.c_str()); // XXX: fails due to creation with 0755
}


TEST(caffeapi,service_train_images_imagedatalayer_1label)
{
  // create service
  JsonAPI japi;
  std::string plank_repo_loc = "plank";
  mkdir(plank_repo_loc.c_str(),0777);
  std::string sname = "my_service";
  std::string jstr = "{\"mllib\":\"caffe\",\"description\":\"my classifier\",\"type\":\"supervised\",\"model\":{\"repository\":\"" +  plank_repo_loc + "\",\"templates\":\"" + model_templates_repo  + "\"},\"parameters\":{\"input\":{\"db\":false,\"connector\":\"image\"},\"mllib\":{\"template\":\"cifar\",\"nclasses\":121}}}";
  std::string joutstr = japi.jrender(japi.service_create(sname,jstr));
  ASSERT_EQ(created_str,joutstr);

  // train
  std::string jtrainstr = "{\"service\":\"" + sname + "\",\"async\":false,\"parameters\":{\"input\":{\"db\":false,\"width\":32,\"height\":32,\"test_split\":0.001,\"shuffle\":true,\"bw\":false,\"root_folder\":\"" + plank_repo + "\"},\"mllib\":{\"gpu\":true,\"gpuid\":"+gpuid+",\"solver\":{\"iterations\":" + iterations_plank + ",\"test_interval\":3000,\"base_lr\":0.0001,\"snapshot\":2000,\"test_initialization\":false},\"net\":{\"batch_size\":2}},\"output\":{\"measure\":[\"acc\",\"acc-5\",\"mcll\",\"f1\"]}},\"data\":[\"" + plank_repo + "file-lst.txt\",\"" + plank_repo + "file-lst-test.txt\"]}";
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

TEST(caffeapi,service_train_images_imagedatalayer_multilabel)
{
  // create service
  JsonAPI japi;
  std::string plank_repo_loc = "plank";
  mkdir(plank_repo_loc.c_str(),0777);
  std::string sname = "my_service";
  std::string jstr = "{\"mllib\":\"caffe\",\"description\":\"my classifier\",\"type\":\"supervised\",\"model\":{\"repository\":\"" +  plank_repo_loc + "\",\"templates\":\"" + model_templates_repo  + "\"},\"parameters\":{\"input\":{\"db\":false,\"connector\":\"image\",\"multi_label\":true},\"mllib\":{\"template\":\"cifar\",\"nclasses\":3}}}";
  std::string joutstr = japi.jrender(japi.service_create(sname,jstr));
  ASSERT_EQ(created_str,joutstr);

  // train
  std::string jtrainstr = "{\"service\":\"" + sname + "\",\"async\":false,\"parameters\":{\"input\":{\"db\":false,\"width\":32,\"height\":32,\"test_split\":0.001,\"shuffle\":true,\"bw\":false,\"root_folder\":\"" + plank_repo + "\"},\"mllib\":{\"gpu\":true,\"gpuid\":"+gpuid+",\"solver\":{\"iterations\":" + iterations_plank + ",\"test_interval\":3000,\"base_lr\":0.0001,\"snapshot\":2000,\"test_initialization\":false},\"net\":{\"batch_size\":2}},\"output\":{\"measure\":[\"acc\"]}},\"data\":[\"" + plank_repo + "file-lst-ml.txt\",\"" + plank_repo + "file-lst-test-ml.txt\"]}";
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
  ASSERT_TRUE(jd["body"]["measure"]["f1"].GetDouble() >= 0.0);

  // remove service
  jstr = "{\"clear\":\"full\"}";
  joutstr = japi.jrender(japi.service_delete(sname,jstr));
  ASSERT_EQ(ok_str,joutstr);
  rmdir(plank_repo_loc.c_str());
}

TEST(caffeapi,service_train_images_imagedatalayer_multilabel_softprob)
{
  // create service
  JsonAPI japi;
  std::string plank_repo_loc = "plank";
  mkdir(plank_repo_loc.c_str(),0777);
  std::string sname = "my_service";
  std::string jstr = "{\"mllib\":\"caffe\",\"description\":\"my classifier\",\"type\":\"supervised\",\"model\":{\"repository\":\"" +  plank_repo_loc + "\",\"templates\":\"" + model_templates_repo  + "\"},\"parameters\":{\"input\":{\"db\":false,\"connector\":\"image\",\"multi_label\":true},\"mllib\":{\"template\":\"cifar\",\"nclasses\":3,\"regression\":true}}}";
  std::string joutstr = japi.jrender(japi.service_create(sname,jstr));
  ASSERT_EQ(created_str,joutstr);

  // train
  std::string jtrainstr = "{\"service\":\"" + sname + "\",\"async\":false,\"parameters\":{\"input\":{\"db\":false,\"width\":32,\"height\":32,\"test_split\":0.001,\"shuffle\":true,\"bw\":false,\"root_folder\":\"" + plank_repo + "\"},\"mllib\":{\"gpu\":true,\"gpuid\":"+gpuid+",\"solver\":{\"iterations\":" + iterations_plank + ",\"test_interval\":3000,\"base_lr\":0.0001,\"snapshot\":2000,\"test_initialization\":false},\"net\":{\"batch_size\":2}},\"output\":{\"measure\":[\"acc\"]}},\"data\":[\"" + plank_repo + "file-lst-ml.txt\",\"" + plank_repo + "file-lst-test-ml.txt\"]}";
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
  ASSERT_TRUE(jd["body"]["measure"].HasMember("kl_divergence"));
  ASSERT_TRUE(jd["body"]["measure"]["kl_divergence"].GetDouble() >= 0.0);
  ASSERT_TRUE(jd["body"]["measure"].HasMember("kolmogorov_smirnov"));
  ASSERT_TRUE(jd["body"]["measure"]["kolmogorov_smirnov"].GetDouble() >= 0.0);
  ASSERT_TRUE(jd["body"]["measure"].HasMember("distance_correlation"));
  ASSERT_TRUE(jd["body"]["measure"]["distance_correlation"].GetDouble() >= 0.0);
  ASSERT_TRUE(jd["body"]["measure"].HasMember("r2"));
  ASSERT_TRUE(jd["body"]["measure"]["r2"].GetDouble() <= 1.0);
  ASSERT_TRUE(jd["body"]["measure"].HasMember("delta_score_0.05"));
  ASSERT_TRUE(jd["body"]["measure"]["delta_score_0.05"].GetDouble() >= 0.0);
  ASSERT_TRUE(jd["body"]["measure"].HasMember("delta_score_0.1"));
  ASSERT_TRUE(jd["body"]["measure"]["delta_score_0.1"].GetDouble() >= 0.0);
  ASSERT_TRUE(jd["body"]["measure"].HasMember("delta_score_0.2"));
  ASSERT_TRUE(jd["body"]["measure"]["delta_score_0.2"].GetDouble() >= 0.0);
  ASSERT_TRUE(jd["body"]["measure"].HasMember("delta_score_0.5"));
  ASSERT_TRUE(jd["body"]["measure"]["delta_score_0.5"].GetDouble() >= 0.0);

  // remove service
  jstr = "{\"clear\":\"full\"}";
  joutstr = japi.jrender(japi.service_delete(sname,jstr));
  ASSERT_EQ(ok_str,joutstr);
  rmdir(plank_repo_loc.c_str());
}

TEST(caffeapi,service_train_images_convnet)
{
  // create service
  JsonAPI japi;
  std::string plank_repo_loc = "plank";
  mkdir(plank_repo_loc.c_str(),0777);
  std::string sname = "my_service";
  std::string jstr = "{\"mllib\":\"caffe\",\"description\":\"my classifier\",\"type\":\"supervised\",\"model\":{\"repository\":\"" +  plank_repo_loc + "\",\"templates\":\"" + model_templates_repo  + "\"},\"parameters\":{\"input\":{\"connector\":\"image\"},\"mllib\":{\"db\":true,\"template\":\"convnet\",\"layers\":[\"1CR32\",\"1CR64\",\"1CR128\",\"1024\"],\"nclasses\":121}}}";
  std::string joutstr = japi.jrender(japi.service_create(sname,jstr));
  ASSERT_EQ(created_str,joutstr);

  // train
  std::string jtrainstr = "{\"service\":\"" + sname + "\",\"async\":false,\"parameters\":{\"input\":{\"db\":true,\"width\":32,\"height\":32,\"test_split\":0.001,\"shuffle\":true,\"bw\":false},\"mllib\":{\"gpu\":true,\"gpuid\":"+gpuid+",\"solver\":{\"iterations\":" + iterations_plank + ",\"test_interval\":500,\"base_lr\":0.0001,\"snapshot\":2000,\"test_initialization\":false},\"net\":{\"batch_size\":100}},\"output\":{\"measure\":[\"acc\",\"acc-5\",\"mcll\",\"f1\"]}},\"data\":[\"" + plank_repo + "train\"]}";
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

TEST(caffeapi,service_train_images_resnet)
{
  // create service
  JsonAPI japi;
  std::string plank_repo_loc = "plank";
  mkdir(plank_repo_loc.c_str(),0777);
  std::string sname = "my_service";
  std::string jstr = "{\"mllib\":\"caffe\",\"description\":\"my classifier\",\"type\":\"supervised\",\"model\":{\"repository\":\"" +  plank_repo_loc + "\",\"templates\":\"" + model_templates_repo  + "\"},\"parameters\":{\"input\":{\"connector\":\"image\",\"width\":32,\"height\":32},\"mllib\":{\"db\":true,\"template\":\"resnet\",\"layers\":[\"Res10\"],\"nclasses\":121}}}";
  std::string joutstr = japi.jrender(japi.service_create(sname,jstr));
  ASSERT_EQ(created_str,joutstr);

  // train
  std::string jtrainstr = "{\"service\":\"" + sname + "\",\"async\":false,\"parameters\":{\"input\":{\"db\":true,\"width\":32,\"height\":32,\"test_split\":0.001,\"shuffle\":true,\"bw\":false},\"mllib\":{\"gpu\":true,\"gpuid\":"+gpuid+",\"solver\":{\"iterations\":" + iterations_plank + ",\"test_interval\":500,\"base_lr\":0.001,\"snapshot\":2000,\"test_initialization\":false},\"net\":{\"batch_size\":100}},\"output\":{\"measure\":[\"acc\",\"acc-5\",\"mcll\",\"f1\"]}},\"data\":[\"" + plank_repo + "train\"]}";
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

TEST(caffeapi,service_train_images_autoenc)
{
 // create service
  JsonAPI japi;
  std::string plank_repo_loc = "plank";
  mkdir(plank_repo_loc.c_str(),0777);
  std::string sname = "my_service";
  std::string jstr = "{\"mllib\":\"caffe\",\"description\":\"my classifier\",\"type\":\"supervised\",\"model\":{\"repository\":\"" +  plank_repo_loc + "\",\"templates\":\"" + model_templates_repo  + "\"},\"parameters\":{\"input\":{\"connector\":\"image\",\"width\":224,\"height\":224},\"mllib\":{\"db\":true,\"template\":\"convnet\",\"layers\":[\"1CR32\",\"1CR64\",\"1CR128\",\"DR128\",\"1CR128\",\"DR64\",\"1CR64\",\"DR32\",\"1CR32\"],\"activation\":\"relu\",\"autoencoder\":true,\"scale\":0.0039}}}";
  std::string joutstr = japi.jrender(japi.service_create(sname,jstr));
  ASSERT_EQ(created_str,joutstr);
  
  // train
  std::string jtrainstr = "{\"service\":\"" + sname + "\",\"async\":false,\"parameters\":{\"input\":{\"db\":true,\"test_split\":0.001,\"shuffle\":true,\"bw\":false},\"mllib\":{\"gpu\":true,\"gpuid\":"+gpuid+",\"resume\":false,\"solver\":{\"iterations\":" + iterations_plank + ",\"test_interval\":500,\"base_lr\":0.0001,\"snapshot\":2000,\"test_initialization\":false},\"net\":{\"batch_size\":10}},\"output\":{\"measure\":[\"eucll\"]}},\"data\":[\"" + plank_repo + "train\"]}";
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
  ASSERT_TRUE(jd["body"]["measure"]["eucll"].GetDouble() <= 1000000);

  // remove service
  jstr = "{\"clear\":\"full\"}";
  joutstr = japi.jrender(japi.service_delete(sname,jstr));
  ASSERT_EQ(ok_str,joutstr);
  rmdir(plank_repo_loc.c_str());
}

TEST(caffeapi,service_train_images_autoenc_geometry)
{
  // create service
  JsonAPI japi;
  std::string plank_repo_loc = "plank";
  mkdir(plank_repo_loc.c_str(),0777);
  std::string sname = "my_service";
  std::string jstr = "{\"mllib\":\"caffe\",\"description\":\"my classifier\",\"type\":\"supervised\",\"model\":{\"repository\":\"" +  plank_repo_loc + "\",\"templates\":\"" + model_templates_repo  + "\"},\"parameters\":{\"input\":{\"connector\":\"image\",\"width\":224,\"height\":224},\"mllib\":{\"db\":true,\"template\":\"convnet\",\"layers\":[\"1CR32\",\"1CR64\",\"1CR128\",\"DR128\",\"1CR128\",\"DR64\",\"1CR64\",\"DR32\",\"1CR32\"],\"activation\":\"relu\",\"autoencoder\":true,\"scale\":0.0039,\"geometry\":{\"all_effects\":false,\"persp_horizontal\":true,\"persp_vertical\":false,\"zoom_in\":true,\"zoom_out\":true,\"pad_mode\":\"mirrored\",\"prob\":0.1}}}}";
  std::string joutstr = japi.jrender(japi.service_create(sname,jstr));
  ASSERT_EQ(created_str,joutstr);


  caffe::NetParameter net_param;
  std::string prototxt = plank_repo_loc + "/convnet.prototxt";;
  bool succ = caffe::ReadProtoFromTextFile(prototxt,&net_param);
  ASSERT_TRUE(succ);


  caffe::LayerParameter *lparam = net_param.mutable_layer(0);
  caffe::GeometryParameter gparam = lparam->transform_param().geometry_param();
  ASSERT_FLOAT_EQ(gparam.prob(), 0.1);
  ASSERT_EQ(gparam.persp_horizontal(),true);
  ASSERT_EQ(gparam.persp_vertical(),false);
  ASSERT_EQ(gparam.zoom_out(),true);
  ASSERT_EQ(gparam.zoom_in(),true);
  ASSERT_EQ(gparam.pad_mode(),caffe::GeometryParameter_Pad_mode_MIRRORED);

  // // train
  std::string jtrainstr = "{\"service\":\"" + sname + "\",\"async\":false,\"parameters\":{\"input\":{\"db\":true,\"test_split\":0.001,\"shuffle\":true,\"bw\":false},\"mllib\":{\"gpu\":true,\"gpuid\":"+gpuid+",\"resume\":false,\"solver\":{\"iterations\":" + iterations_plank + ",\"test_interval\":500,\"base_lr\":0.0001,\"snapshot\":2000,\"test_initialization\":false},\"net\":{\"batch_size\":10}},\"output\":{\"measure\":[\"eucll\"]}},\"data\":[\"" + plank_repo + "train\"]}";
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
  ASSERT_TRUE(jd["body"]["measure"]["eucll"].GetDouble() <= 1000000);

  // remove service
  jstr = "{\"clear\":\"full\"}";
  joutstr = japi.jrender(japi.service_delete(sname,jstr));
  ASSERT_EQ(ok_str,joutstr);
  rmdir(plank_repo_loc.c_str());
}

TEST(caffeapi,service_train_images_seg)
{
  // create service
  JsonAPI japi;
  std::string camvid_repo_loc = "camvid";
  mkdir(camvid_repo_loc.c_str(),0777);
  std::string sname = "my_service";
  std::string jstr = "{\"mllib\":\"caffe\",\"description\":\"my classifier\",\"type\":\"supervised\",\"model\":{\"repository\":\"" +  camvid_repo_loc + "\",\"templates\":\"" + model_templates_repo  + "\"},\"parameters\":{\"input\":{\"connector\":\"image\",\"segmentation\":true,\"width\":480,\"height\":480},\"mllib\":{\"template\":\"unet\",\"nclasses\":12}}}";
  std::string joutstr = japi.jrender(japi.service_create(sname,jstr));
  ASSERT_EQ(created_str,joutstr);

  // train
  std::string jtrainstr = "{\"service\":\"" + sname + "\",\"async\":false,\"parameters\":{\"input\":{\"segmentation\":true},\"mllib\":{\"gpu\":true,\"gpuid\":"+gpuid+",\"class_weights\":[0.2595,0.1826,4.5640,0.1417,0.9051,0.3826,9.6446,1.8418,0.6823,6.2478,7.3614,0.5],\"ignore_label\":11,\"solver\":{\"iterations\":" + iterations_camvid + ",\"test_interval\":200,\"base_lr\":0.0001,\"test_initialization\":false,\"mirror\":true,\"solver_type\":\"SGD\"},\"net\":{\"batch_size\":1,\"test_batch_size\":1}},\"output\":{\"measure\":[\"acc\"]}},\"data\":[\"" + camvid_repo + "train.txt\",\"" + camvid_repo + "test2.txt\"]}";
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
  ASSERT_TRUE(jd["body"]["measure"]["acc"].GetDouble() >= 0.0);

  //predict + conf map
  std::string jpredictstr = "{\"service\":\"" + sname + "\",\"async\":false,\"parameters\":{\"input\":{\"segmentation\":true},\"mllib\":{\"gpu\":true,\"gpuid\":"+gpuid+",\"net\":{\"batch_size\":1,\"test_batch_size\":1}},\"output\":{\"confidences\":[\"best\",\"2\"]}},\"data\":[\"" + camvid_repo + "test/0001TP_008550.png\"]}";
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  //std::cout << "joutstr predict=" << joutstr << std::endl;
  jd.Parse(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200,jd["status"]["code"]);
  ASSERT_TRUE(jd["body"]["predictions"][0].HasMember("vals"));
  ASSERT_TRUE(jd["body"]["predictions"][0]["vals"].IsArray());
  ASSERT_TRUE(jd["body"]["predictions"][0]["vals"].Size() == (480*480));
  ASSERT_TRUE(jd["body"]["predictions"][0].HasMember("confidences"));
  ASSERT_TRUE(jd["body"]["predictions"][0]["confidences"].HasMember("best"));
  ASSERT_TRUE(jd["body"]["predictions"][0]["confidences"]["best"].IsArray());
  ASSERT_TRUE(jd["body"]["predictions"][0]["confidences"]["best"].Size() == 480*480);
  ASSERT_TRUE(jd["body"]["predictions"][0]["confidences"].HasMember("2"));
  ASSERT_TRUE(jd["body"]["predictions"][0]["confidences"]["2"].IsArray());
  ASSERT_TRUE(jd["body"]["predictions"][0]["confidences"]["2"].Size() == 480*480);


  // remove service
  jstr = "{\"clear\":\"full\"}";
  joutstr = japi.jrender(japi.service_delete(sname,jstr));
  ASSERT_EQ(ok_str,joutstr);
  rmdir(camvid_repo_loc.c_str());
}

TEST(caffeapi,service_test_bbox)
{
    // create service
  JsonAPI japi;
  std::string sname = "my_service";
  std::string jstr = "{\"mllib\":\"caffe\",\"description\":\"my classifier\",\"type\":\"supervised\",\"model\":{\"repository\":\"" +  voc_repo + "\"},\"parameters\":{\"input\":{\"connector\":\"image\",\"width\":300,\"height\":300},\"mllib\":{\"nclasses\":21}}}";
  std::cerr << "jstr=" << jstr << std::endl;
  std::string joutstr = japi.jrender(japi.service_create(sname,jstr));
  ASSERT_EQ(created_str,joutstr);
  JDoc jd;

  // predict
  std::string jpredictstr = "{\"service\":\""+ sname + "\",\"parameters\":{\"input\":{},\"mllib\":{},\"output\":{\"bbox\":true,\"confidence_threshold\":0.1}},\"data\":[\"" + voc_repo + "/test_img.jpg\"]}";
  std::cerr << "jpredictstr=" << jpredictstr << std::endl;
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  std::cout << "joutstr predict=" << joutstr << std::endl;
  jd.Parse(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200,jd["status"]["code"]);
  ASSERT_TRUE(jd["body"]["predictions"][0].HasMember("classes"));
  ASSERT_TRUE(jd["body"]["predictions"][0]["classes"][0].HasMember("bbox"));
  ASSERT_TRUE(jd["body"]["predictions"][0]["classes"][0]["bbox"].HasMember("xmin"));
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
  std::string jtrainstr = "{\"service\":\"" + sname + "\",\"async\":false,\"parameters\":{\"input\":{\"test_split\":0.2,\"shuffle\":true,\"min_count\":10,\"min_word_length\":3,\"count\":false},\"mllib\":{\"gpu\":true,\"gpuid\":"+gpuid+",\"solver\":{\"iterations\":" + iterations_n20 + ",\"test_interval\":200,\"base_lr\":0.05,\"snapshot\":2000,\"test_initialization\":true},\"net\":{\"batch_size\":100}},\"output\":{\"measure\":[\"acc\",\"mcll\",\"f1\"]}},\"data\":[\"" + n20_repo + "news20\"]}";
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
  std::string jpredictstr = "{\"service\":\"" + sname + "\",\"parameters\":{\"mllib\":{\"gpu\":true,\"gpuid\":"+gpuid+",\"net\":{\"test_batch_size\":10}},\"output\":{\"measure\":[\"f1\"]}},\"data\":[\"" + n20_repo +"news20\"]}";
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
  std::string jstr = "{\"mllib\":\"caffe\",\"description\":\"my classifier\",\"type\":\"supervised\",\"model\":{\"repository\":\"" +  n20_repo_loc + "\",\"templates\":\"" + model_templates_repo  + "\"},\"parameters\":{\"input\":{\"connector\":\"txt\",\"sparse\":true},\"mllib\":{\"gpu\":true,\"gpuid\":"+gpuid+",\"template\":\"mlp\",\"nclasses\":20,\"db\":false}}}";
  std::string joutstr = japi.jrender(japi.service_create(sname,jstr));
  ASSERT_EQ(created_str,joutstr);

  // train
  std::string jtrainstr = "{\"service\":\"" + sname + "\",\"async\":false,\"parameters\":{\"input\":{\"test_split\":0.2,\"shuffle\":true,\"min_count\":10,\"min_word_length\":3,\"count\":false,\"db\":false},\"mllib\":{\"gpu\":true,\"gpuid\":"+gpuid+",\"solver\":{\"iterations\":" + iterations_n20 + ",\"test_interval\":200,\"base_lr\":0.01,\"snapshot\":2000,\"test_initialization\":true},\"net\":{\"batch_size\":100}},\"output\":{\"measure\":[\"acc\",\"mcll\",\"f1\",\"cmdiag\"]}},\"data\":[\"" + n20_repo + "news20\"]}";
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
  std::string jpredictstr = "{\"service\":\"" + sname + "\",\"parameters\":{\"mllib\":{\"gpu\":true,\"gpuid\":"+gpuid+",\"net\":{\"test_batch_size\":10}},\"output\":{\"measure\":[\"f1\"]}},\"data\":[\"" + n20_repo +"news20\"]}";
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
  std::string jtrainstr = "{\"service\":\"" + sname + "\",\"async\":false,\"parameters\":{\"input\":{\"test_split\":0.2,\"shuffle\":true,\"min_count\":10,\"min_word_length\":3,\"count\":false,\"db\":true},\"mllib\":{\"gpu\":true,\"gpuid\":"+gpuid+",\"solver\":{\"iterations\":" + iterations_n20 + ",\"test_interval\":200,\"base_lr\":0.01,\"snapshot\":2000,\"test_initialization\":true},\"net\":{\"batch_size\":100}},\"output\":{\"measure\":[\"acc\",\"mcll\",\"f1\",\"cmdiag\"]}},\"data\":[\"" + n20_repo + "news20\"]}";
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
  std::string jpredictstr = "{\"service\":\"" + sname + "\",\"parameters\":{\"mllib\":{\"gpu\":true,\"gpuid\":"+gpuid+",\"net\":{\"test_batch_size\":10}},\"output\":{\"measure\":[\"f1\"]}},\"data\":[\"" + n20_repo +"news20\"]}";
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
  std::string jtrainstr = "{\"service\":\"" + sname + "\",\"async\":false,\"parameters\":{\"input\":{\"test_split\":0.2,\"shuffle\":true,\"characters\":true,\"sequence\":50,\"db\":true},\"mllib\":{\"gpu\":true,\"gpuid\":"+gpuid+",\"solver\":{\"iterations\":" + iterations_n20_char + ",\"test_interval\":30,\"base_lr\":0.01,\"snapshot\":2000,\"test_initialization\":true},\"net\":{\"batch_size\":100}},\"output\":{\"measure\":[\"acc\",\"mcll\",\"f1\"]}},\"data\":[\"" + n20_repo + "news20\"]}";
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

TEST(caffeapi,service_train_txt_char_resnet)
{
  // create service
  JsonAPI japi;
  std::string n20_repo_loc = "n20";
  mkdir(n20_repo_loc.c_str(),0777);
  std::string sname = "my_service";
  std::string jstr = "{\"mllib\":\"caffe\",\"description\":\"my classifier\",\"type\":\"supervised\",\"model\":{\"repository\":\"" +  n20_repo_loc + "\",\"templates\":\"" + model_templates_repo  + "\"},\"parameters\":{\"input\":{\"connector\":\"txt\",\"characters\":true,\"sequence\":50},\"mllib\":{\"template\":\"resnet\",\"layers\":[\"1CR16\",\"1CR16\",\"1CR16\",\"512\",\"512\"],\"nclasses\":20,\"db\":true}}}";
  std::string joutstr = japi.jrender(japi.service_create(sname,jstr));
  ASSERT_EQ(created_str,joutstr);

  // train
  std::string jtrainstr = "{\"service\":\"" + sname + "\",\"async\":false,\"parameters\":{\"input\":{\"test_split\":0.2,\"shuffle\":true,\"characters\":true,\"sequence\":50,\"db\":true},\"mllib\":{\"gpu\":true,\"gpuid\":"+gpuid+",\"solver\":{\"iterations\":" + iterations_n20_char + ",\"test_interval\":30,\"base_lr\":0.001,\"snapshot\":2000,\"test_initialization\":true},\"net\":{\"batch_size\":100}},\"output\":{\"measure\":[\"acc\",\"mcll\",\"f1\"]}},\"data\":[\"" + n20_repo + "news20\"]}";
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
  std::string jtrainstr = "{\"service\":\"" + sname + "\",\"async\":false,\"parameters\":{\"input\":{\"test_split\":0.1,\"shuffle\":true,\"label\":[\"c_class\",\"m_class\",\"x_class\"],\"separator\":\",\",\"scale\":true,\"categoricals\":[\"class_code\",\"code_spot\",\"code_spot_distr\"]},\"mllib\":{\"gpu\":true,\"gpuid\":"+gpuid+",\"solver\":{\"iterations\":" + iterations_sflare + ",\"test_interval\":200,\"base_lr\":0.001,\"snapshot\":2000,\"test_initialization\":true},\"net\":{\"batch_size\":100}},\"output\":{\"measure\":[\"eucll\"]}},\"data\":[\"" + sflare_repo + "flare.csv\"]}";
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


TEST(caffeapi,service_train_csvts_lstm)
{
  // create service
  JsonAPI japi;
  std::string csvts_data = sinus + "train";
  std::string csvts_test = sinus +"test";
  std::string csvts_predict = sinus +"predict";
  std::string csvts_repo = "csvts";
  mkdir(csvts_repo.c_str(),0777);
  std::string sname = "my_service_csvts";
  std::string jstr = "{\"mllib\":\"caffe\",\"description\":\"my ts regressor\",\"type\":\"supervised\",\"model\":{\"repository\":\"" +  csvts_repo+"\",\"templates\":\"" + model_templates_repo + "\"},\"parameters\":{\"input\":{\"connector\":\"csvts\",\"label\":[\"output\"]},\"mllib\":{\"template\":\"recurrent\",\"layers\":[\"L10\",\"L10\"],\"dropout\":[0.0,0.0,0.0],\"regression\":true,\"sl1sigma\":100.0,\"loss\":\"L1\"}}}";
  std::string joutstr = japi.jrender(japi.service_create(sname,jstr));
  ASSERT_EQ(created_str,joutstr);

  // train
  std::string jtrainstr = "{\"service\":\"" + sname + "\",\"async\":false,\"parameters\":{\"input\":{\"shuffle\":true,\"separator\":\",\",\"scale\":true},\"mllib\":{\"gpu\":true,\"gpuid\":"+gpuid+",\"timesteps\":20,\"solver\":{\"iterations\":" + iterations_lstm + ",\"test_interval\":500,\"base_lr\":0.001,\"snapshot\":500,\"test_initialization\":false},\"net\":{\"batch_size\":100}},\"output\":{\"measure\":[\"L1\",\"L2\"]}},\"data\":[\"" + csvts_data+"\",\""+csvts_test+"\"]}";
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
  ASSERT_TRUE(jd["body"]["measure"].HasMember("L1_mean_error"));
  ASSERT_TRUE(jd["body"]["measure"]["L1_max_error_0"].GetDouble() > 0.0);
  ASSERT_TRUE(jd["body"]["parameters"]["input"].HasMember("max_vals"));
  ASSERT_TRUE(jd["body"]["parameters"]["input"].HasMember("min_vals"));

  std::string str_min_vals = japi.jrender(jd["body"]["parameters"]["input"]["min_vals"]);
  std::string str_max_vals = japi.jrender(jd["body"]["parameters"]["input"]["max_vals"]);

  //  predict
  std::string jpredictstr = "{\"service\":\""+ sname + "\",\"parameters\":{\"input\":{\"timesteps\":20,\"connector\":\"csvts\",\"scale\":true,\"min_vals\":" + str_min_vals + ",\"max_vals\":" + str_max_vals + "},\"output\":{}},\"data\":[\"" + csvts_predict + "\"]}";
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200,jd["status"]["code"]);
  std::string uri = jd["body"]["predictions"][0]["uri"].GetString();
  ASSERT_EQ("../examples/all/sinus/predict/seq_2.csv",uri);
  ASSERT_TRUE(jd["body"]["predictions"][0]["series"].IsArray());
  ASSERT_TRUE(jd["body"]["predictions"][0]["series"][0]["out"][0].GetDouble() >= -1.0);

  //  remove service
  jstr = "{\"clear\":\"full\"}";
  joutstr = japi.jrender(japi.service_delete(sname,jstr));
  ASSERT_EQ(ok_str,joutstr);
  rmdir(csvts_repo.c_str());
}


TEST(caffeapi,service_train_csvts_db_lstm)
{
  // create service
  JsonAPI japi;
  std::string csvts_data = sinus + "train";
  std::string csvts_test = sinus +"test";
  std::string csvts_predict = sinus +"predict";
  std::string csvts_repo = "csvts_db";
  mkdir(csvts_repo.c_str(),0777);
  std::string sname = "my_service_csvts";
  std::string jstr = "{\"mllib\":\"caffe\",\"description\":\"my ts regressor\",\"type\":\"supervised\",\"model\":{\"repository\":\"" +  csvts_repo+"\",\"templates\":\"" + model_templates_repo+  "\"},\"parameters\":{\"input\":{\"connector\":\"csvts\",\"db\":true,\"label\":[\"output\"]},\"mllib\":{\"template\":\"recurrent\",\"layers\":[\"L10\",\"L10\"],\"dropout\":[0.0,0.0],\"regression\":true,\"loss\":\"L1\",\"db\":true}}}";
  std::string joutstr = japi.jrender(japi.service_create(sname,jstr));
  ASSERT_EQ(created_str,joutstr);

  // train
  std::string jtrainstr = "{\"service\":\"" + sname + "\",\"async\":false,\"parameters\":{\"input\":{\"shuffle\":true,\"separator\":\",\",\"scale\":false,\"db\":true},\"mllib\":{\"db\":true,\"gpu\":true,\"gpuid\":"+gpuid+",\"timesteps\":20,\"solver\":{\"iterations\":" + iterations_lstm + ",\"test_interval\":500,\"base_lr\":0.001,\"snapshot\":500,\"test_initialization\":false},\"net\":{\"batch_size\":10}},\"output\":{\"measure\":[\"L1\",\"L2\"]}},\"data\":[\"" + csvts_data+"\",\""+csvts_test+"\"]}";
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
  ASSERT_TRUE(jd["body"]["measure"].HasMember("L1_max_error"));
  ASSERT_TRUE(jd["body"]["measure"]["L1_mean_error"].GetDouble() > 0.0);


  // predict
  std::string jpredictstr = "{\"service\":\""+ sname + "\",\"parameters\":{\"input\":{\"connector\":\"csvts\",\"scale\":false,\"timesteps\":20},\"output\":{}},\"data\":[\"" + csvts_predict + "\"]}";
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200,jd["status"]["code"]);
  std::string uri = jd["body"]["predictions"][0]["uri"].GetString();
  ASSERT_EQ("../examples/all/sinus/predict/seq_2.csv",uri);
  ASSERT_TRUE(jd["body"]["predictions"][0]["series"].IsArray());
  ASSERT_TRUE(jd["body"]["predictions"][0]["series"][0]["out"][0].GetDouble() >= -1.0);

  //  remove service
  jstr = "{\"clear\":\"full\"}";
  joutstr = japi.jrender(japi.service_delete(sname,jstr));
  ASSERT_EQ(ok_str,joutstr);
  rmdir(csvts_repo.c_str());
}
