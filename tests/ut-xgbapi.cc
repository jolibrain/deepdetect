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

static std::string forest_repo = "../examples/all/forest_type/";
static std::string n20_repo = "../examples/all/n20/";
//static std::string sflare_repo = "../examples/all/sflare/";

static std::string iterations_forest = "10";
static std::string iterations_n20 = "10";
//static std::string iterations_sflare = "5000";

TEST(xgbapi,service_train_csv)
{
::google::InitGoogleLogging("ut_xgbapi");
  JsonAPI japi;
  std::string sname = "my_service";
  std::string jstr = "{\"mllib\":\"xgboost\",\"description\":\"my classifier\",\"type\":\"supervised\",\"model\":{\"repository\":\"" +  forest_repo + "\"},\"parameters\":{\"input\":{\"connector\":\"csv\"},\"mllib\":{\"nclasses\":7}}}";
  std::string joutstr = japi.jrender(japi.service_create(sname,jstr));
  ASSERT_EQ(created_str,joutstr);

  // assert json blob file
  std::cerr << forest_repo + "/" + JsonAPI::_json_blob_fname << std::endl;
  ASSERT_TRUE(fileops::file_exists(forest_repo + "/" + JsonAPI::_json_blob_fname));
  
  // train
  std::string jtrainstr = "{\"service\":\"" + sname + "\",\"async\":false,\"parameters\":{\"input\":{\"label\":\"Cover_Type\",\"id\":\"Id\",\"test_split\":0.1,\"label_offset\":-1,\"shuffle\":true},\"mllib\":{\"iterations\":" + iterations_forest + ",\"objective\":\"multi:softprob\"},\"output\":{\"measure\":[\"acc\",\"mcll\",\"f1\",\"cmdiag\",\"cmfull\"]}},\"data\":[\"" + forest_repo + "train.csv\"]}";
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
  /*ASSERT_TRUE(jd["body"]["measure"].HasMember("train_loss"));
    ASSERT_TRUE(fabs(jd["body"]["measure"]["train_loss"].GetDouble()) > 0.0);*/
  ASSERT_TRUE(jd["body"]["measure"].HasMember("f1"));
  ASSERT_TRUE(jd["body"]["measure"]["f1"].GetDouble() > 0.7);
  ASSERT_EQ(jd["body"]["measure"]["accp"].GetDouble(),jd["body"]["measure"]["acc"].GetDouble());
  ASSERT_TRUE(jd["body"]["measure"].HasMember("cmdiag"));
  ASSERT_EQ(7,jd["body"]["measure"]["cmdiag"].Size());
  ASSERT_TRUE(jd["body"]["measure"]["cmdiag"][0].GetDouble() >= 0);
  ASSERT_TRUE(jd["body"]["measure"]["cmfull"]["1"].Size());

  // predict from data, with header and id
  std::string mem_data_head = "Id,Elevation,Aspect,Slope,Horizontal_Distance_To_Hydrology,Vertical_Distance_To_Hydrology,Horizontal_Distance_To_Roadways,Hillshade_9am,Hillshade_Noon,Hillshade_3pm,Horizontal_Distance_To_Fire_Points,Wilderness_Area1,Wilderness_Area2,Wilderness_Area3,Wilderness_Area4,Soil_Type1,Soil_Type2,Soil_Type3,Soil_Type4,Soil_Type5,Soil_Type6,Soil_Type7,Soil_Type8,Soil_Type9,Soil_Type10,Soil_Type11,Soil_Type12,Soil_Type13,Soil_Type14,Soil_Type15,Soil_Type16,Soil_Type17,Soil_Type18,Soil_Type19,Soil_Type20,Soil_Type21,Soil_Type22,Soil_Type23,Soil_Type24,Soil_Type25,Soil_Type26,Soil_Type27,Soil_Type28,Soil_Type29,Soil_Type30,Soil_Type31,Soil_Type32,Soil_Type33,Soil_Type34,Soil_Type35,Soil_Type36,Soil_Type37,Soil_Type38,Soil_Type39,Soil_Type40";
  std::string mem_data = "0,2499,326,7,300,88,480,202,232,169,1676,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0";
  std::string jpredictstr = "{\"service\":\""+ sname + "\",\"parameters\":{\"input\":{\"connector\":\"csv\",\"id\":\"Id\",\"scale\":false},\"output\":{\"best\":3}},\"data\":[\"" + mem_data_head + "\",\"" + mem_data + "\"]}";
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_TRUE(jd.HasMember("status"));
  ASSERT_EQ(200,jd["status"]["code"].GetInt());
  std::string cat0 = jd["body"]["predictions"]["classes"][0]["cat"].GetString(); // XXX: true cat is 3, which is 2 here with the label offset
  std::string cat1 = jd["body"]["predictions"]["classes"][1]["cat"].GetString();
  ASSERT_TRUE("2"==cat0||"2"==cat1);
  
  // predict from data, omitting header and sample id
  std::string mem_data2 = "2499,326,7,300,88,480,202,232,169,1676,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0";
  jpredictstr = "{\"service\":\""+ sname + "\",\"parameters\":{\"input\":{\"connector\":\"csv\",\"scale\":false},\"output\":{\"best\":3}},\"data\":[\"" + mem_data2 + "\"]}";
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_TRUE(jd.HasMember("status"));
  ASSERT_EQ(200,jd["status"]["code"].GetInt());
  cat0 = jd["body"]["predictions"]["classes"][0]["cat"].GetString(); // XXX: true cat is 3, which is 2 here with the label offset
  cat1 = jd["body"]["predictions"]["classes"][1]["cat"].GetString();
  ASSERT_TRUE("2"==cat0||"2"==cat1);
  
  // remove service
  jstr = "{\"clear\":\"lib\"}";
  joutstr = japi.jrender(japi.service_delete(sname,jstr));
  ASSERT_EQ(ok_str,joutstr);

  // assert json blob file is still there (or gone if clear=full)
  ASSERT_TRUE(fileops::file_exists(forest_repo + "/" + JsonAPI::_json_blob_fname));
}

TEST(xgbapi,service_train_txt)
{
  // create service
  JsonAPI japi;
  std::string n20_repo_loc = "n20";
  mkdir(n20_repo_loc.c_str(),0777);
  std::string sname = "my_service";
  std::string jstr = "{\"mllib\":\"xgboost\",\"description\":\"my classifier\",\"type\":\"supervised\",\"model\":{\"repository\":\"" +  n20_repo_loc + "\"},\"parameters\":{\"input\":{\"connector\":\"txt\"},\"mllib\":{\"nclasses\":20}}}";
  std::string joutstr = japi.jrender(japi.service_create(sname,jstr));
  ASSERT_EQ(created_str,joutstr);

  // train
  std::string jtrainstr = "{\"service\":\"" + sname + "\",\"async\":false,\"parameters\":{\"input\":{\"test_split\":0.2,\"shuffle\":true,\"min_count\":10,\"min_word_length\":3,\"count\":false},\"mllib\":{\"iterations\":" + iterations_n20 + ",\"objective\":\"multi:softprob\"},\"output\":{\"measure\":[\"acc\",\"mcll\",\"f1\"]}},\"data\":[\"" + n20_repo + "news20\"]}";
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
  //ASSERT_TRUE(jd["body"]["measure"].HasMember("train_loss"));
  //ASSERT_TRUE(fabs(jd["body"]["measure"]["train_loss"].GetDouble()) > 0);
  ASSERT_TRUE(jd["body"]["measure"].HasMember("f1"));
  ASSERT_TRUE(jd["body"]["measure"]["acc"].GetDouble() >= 0.7);
  ASSERT_EQ(jd["body"]["measure"]["accp"].GetDouble(),jd["body"]["measure"]["acc"].GetDouble());

  // remove service
  jstr = "{\"clear\":\"full\"}";
  joutstr = japi.jrender(japi.service_delete(sname,jstr));
  ASSERT_EQ(ok_str,joutstr);
  rmdir(n20_repo_loc.c_str());
}
