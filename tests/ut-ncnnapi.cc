/**
 * DeepDetect
 * Copyright (c) 2018 Jolibrain
 * Author: Corentin Barreau <corentin.barreau@epitech.eu>
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
#include <iostream>

using namespace dd;

static std::string ok_str = "{\"status\":{\"code\":200,\"msg\":\"OK\"}}";
static std::string created_str
    = "{\"status\":{\"code\":201,\"msg\":\"Created\"}}";
static std::string bad_param_str
    = "{\"status\":{\"code\":400,\"msg\":\"BadRequest\"}}";
static std::string not_found_str
    = "{\"status\":{\"code\":404,\"msg\":\"NotFound\"}}";

static std::string squeezenet_ssd_repo
    = "../examples/ncnn/squeezenet_ssd_ncnn/";
static std::string squeezenet_repo = "../examples/ncnn/squeezenet_ncnn/";
static std::string incept_repo = "../examples/ncnn/squeezenet_ssd_ncnn/";
static std::string ocr_repo = "../examples/ncnn/ocr/";
static std::string sinus = "../examples/all/sinus/";
static std::string model_templates_repo = "../templates/caffe/";
static std::string gpuid = "0"; // change as needed

#ifndef CPU_ONLY
static std::string iterations_lstm = "200";
#else
static std::string iterations_lstm = "20";
#endif

TEST(ncnnapi, service_predict_bbox)
{
  // create service
  JsonAPI japi;
  std::string sname = "imgserv";
  std::string jstr
      = "{\"mllib\":\"ncnn\",\"description\":\"squeezenet-ssd\",\"type\":"
        "\"supervised\",\"model\":{\"repository\":\""
        + squeezenet_ssd_repo
        + "\"},\"parameters\":{\"input\":{\"connector\":\"image\",\"height\":"
          "300,\"width\":300},"
          "\"mllib\":{\"nclasses\":21,\"gpu\":false}}}";
  std::string joutstr = japi.jrender(japi.service_create(sname, jstr));
  ASSERT_EQ(created_str, joutstr);

  // predict
  std::string jpredictstr
      = "{\"service\":\"imgserv\",\"parameters\":{\"input\":{\"height\":300,"
        "\"width\":300},\"output\":{\"bbox\":true,\"confidence_threshold\":0."
        "25}},\"data\":[\""
        + squeezenet_ssd_repo + "face.jpg\"]}";
  joutstr = japi.jrender(japi.service_predict(jpredictstr));

  JDoc jd;
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200, jd["status"]["code"]);
  ASSERT_TRUE(jd["body"]["predictions"].IsArray());
  std::string cl1
      = jd["body"]["predictions"][0]["classes"][0]["cat"].GetString();
  ASSERT_TRUE(cl1 == "15");
  ASSERT_TRUE(jd["body"]["predictions"][0]["classes"][0]["prob"].GetDouble()
              > 0.4);
  for (size_t cl = 0; cl < jd["body"]["predictions"][0]["classes"].Size();
       ++cl)
    {
      ASSERT_TRUE(jd["body"]["predictions"][0]["classes"][cl]["bbox"]["xmin"]
                      .GetDouble()
                  > 0.0);
    }

  // predict with mean and std, wrong values, for testing only
  /*jpredictstr
      = "{\"service\":\"imgserv\",\"parameters\":{\"input\":{\"height\":300,"
        "\"width\":300,\"mean\":[128,128,128],\"std\":[255,255,255]},"
        "\"output\":{\"bbox\":true,\"confidence_threshold\":0.25}},\"data\":["
        "\""
        + squeezenet_ssd_repo + "face.jpg\"]}";
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200, jd["status"]["code"]);
  ASSERT_TRUE(jd["body"]["predictions"].IsArray());
  cl1 = jd["body"]["predictions"][0]["classes"][0]["cat"].GetString();
  ASSERT_TRUE(jd["body"]["predictions"][0]["classes"][0]["prob"].GetDouble()
  > 0.4);*/

  // predict with scale, wrong value, for testing only
  /*jpredictstr
      = "{\"service\":\"imgserv\",\"parameters\":{\"input\":{\"height\":300,"
        "\"width\":300,\"scale\":0.0039},"
        "\"output\":{\"bbox\":true,\"confidence_threshold\":0.25}},\"data\":["
        "\""
        + squeezenet_ssd_repo + "face.jpg\"]}";
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200, jd["status"]["code"]);
  ASSERT_TRUE(jd["body"]["predictions"].IsArray());
  cl1 = jd["body"]["predictions"][0]["classes"][0]["cat"].GetString();
  ASSERT_TRUE(jd["body"]["predictions"][0]["classes"][0]["prob"].GetDouble()
  > 0.4);*/
}

#if !defined(CPU_ONLY)

TEST(ncnnapi, service_predict_gpu)
{
  // create service
  JsonAPI japi;
  std::string sname = "imgserv";
  std::string jstr
      = "{\"mllib\":\"ncnn\",\"description\":\"squeezenet-ssd\",\"type\":"
        "\"supervised\",\"model\":{\"repository\":\""
        + squeezenet_ssd_repo
        + "\"},\"parameters\":{\"input\":{\"connector\":\"image\",\"height\":"
          "300,\"width\":300},"
          "\"mllib\":{\"nclasses\":21,\"gpu\":true,\"datatype\":\"fp32\"}}}";
  std::string joutstr = japi.jrender(japi.service_create(sname, jstr));
  ASSERT_EQ(created_str, joutstr);

  // predict
  std::string jpredictstr = "";
  int i = 0;
  while (i < 20)
    {
      jpredictstr = "{\"service\":\"imgserv\",\"parameters\":{\"input\":{"
                    "\"height\":300,"
                    "\"width\":300},\"output\":{\"bbox\":true,\"confidence_"
                    "threshold\":0.25}},\"data\":[\""
                    + squeezenet_ssd_repo + "face.jpg\"]}";
      joutstr = japi.jrender(japi.service_predict(jpredictstr));
      ++i;
    }
  JDoc jd;
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200, jd["status"]["code"]);
  ASSERT_TRUE(jd["body"]["predictions"].IsArray());
  std::string cl1
      = jd["body"]["predictions"][0]["classes"][0]["cat"].GetString();
  ASSERT_TRUE(cl1 == "15");
  ASSERT_TRUE(jd["body"]["predictions"][0]["classes"][0]["prob"].GetDouble()
              > 0.4);

  // predict with batch_size > 1
  jpredictstr
      = "{\"service\":\"imgserv\",\"parameters\":{\"input\":{\"height\":300,"
        "\"width\":300},\"output\":{\"bbox\":true,\"confidence_threshold\":0."
        "25}},\"data\":[\""
        + squeezenet_ssd_repo + "face.jpg\",\"" + squeezenet_ssd_repo
        + "cat.jpg\"]}";
  // std::cerr << "predict=" << jpredictstr << std::endl;
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200, jd["status"]["code"]);
  ASSERT_TRUE(jd["body"]["predictions"].IsArray());
  cl1 = jd["body"]["predictions"][0]["classes"][0]["cat"].GetString();
  ASSERT_TRUE(jd["body"]["predictions"][0]["classes"][0]["prob"].GetDouble()
              > 0.4);
  ASSERT_TRUE(jd["body"]["predictions"][1]["classes"][0]["prob"].GetDouble()
              > 0.4);
}
#endif

TEST(ncnnapi, service_predict_classification)
{
  // create service
  JsonAPI japi;
  std::string sname = "imgserv";
  std::string jstr
      = "{\"mllib\":\"ncnn\",\"description\":\"squeezenet\",\"type\":"
        "\"supervised\",\"model\":{\"repository\":\""
        + squeezenet_repo
        + "\"},\"parameters\":{\"input\":{\"connector\":\"image\",\"height\":"
          "224,\"width\":224,\"mean\":[128,128,128]},"
          "\"mllib\":{\"nclasses\":1000}}}";
  std::string joutstr = japi.jrender(japi.service_create(sname, jstr));
  ASSERT_EQ(created_str, joutstr);

  // predict
  std::string jpredictstr
      = "{\"service\":\"imgserv\",\"parameters\":{\"input\":{\"height\":300,"
        "\"width\":300},\"output\":{\"best\":-1}},\"data\":[\""
        + squeezenet_ssd_repo + "face.jpg\"]}";
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  JDoc jd;
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200, jd["status"]["code"]);
  ASSERT_TRUE(jd["body"]["predictions"].IsArray());
  ASSERT_TRUE(jd["body"]["predictions"].Size() == 1);
  ASSERT_TRUE(jd["body"]["predictions"][0]["classes"].Size() == 1000);
}

#ifdef USE_CAFFE
TEST(ncnnapi, service_lstm)
{
  // create service
  JsonAPI japi;
  std::string csvts_data = sinus + "train";
  std::string csvts_test = sinus + "test";
  std::string csvts_predict = sinus + "predict";
  std::string csvts_repo = "csvts";
  mkdir(csvts_repo.c_str(), 0777);
  std::string sname = "my_service_csvts";
  std::string jstr
      = "{\"mllib\":\"ncnn\",\"description\":\"my ts "
        "regressor\",\"type\":\"supervised\",\"model\":{\"repository\":\""
        + csvts_repo + "\",\"templates\":\"" + model_templates_repo
        + "\"},\"parameters\":{\"input\":{\"connector\":\"csvts\",\"label\":["
          "\"output\"]},\"mllib\":{\"template\":\"recurrent\",\"layers\":["
          "\"L10\",\"L10\"],\"dropout\":[0.0,0.0,0.0],\"regression\":true,"
          "\"sl1sigma\":100.0,\"loss\":\"L1\"}}}";
  std::string joutstr = japi.jrender(japi.service_create(sname, jstr));
  ASSERT_EQ(created_str, joutstr);

  // train
  std::string jtrainstr
      = "{\"service\":\"" + sname
        + "\",\"async\":false,\"parameters\":{\"input\":{\"shuffle\":true,"
          "\"separator\":\",\",\"scale\":true},\"mllib\":{\"gpu\":true,"
          "\"gpuid\":"
        + gpuid
        + ",\"timesteps\":20,\"solver\":{\"iterations\":" + iterations_lstm
        + ",\"test_interval\":500,\"base_lr\":0.001,\"snapshot\":500,\"test_"
          "initialization\":false},\"net\":{\"batch_size\":100}},\"output\":{"
          "\"measure\":[\"L1\",\"L2\"]}},\"data\":[\""
        + csvts_data + "\",\"" + csvts_test + "\"]}";
  std::cerr << "jtrainstr=" << jtrainstr << std::endl;
  joutstr = japi.jrender(japi.service_train(jtrainstr));
  // std::cout << "joutstr=" << joutstr << std::endl;
  JDoc jd;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_TRUE(jd.HasMember("status"));
  ASSERT_EQ(201, jd["status"]["code"].GetInt());
  ASSERT_EQ("Created", jd["status"]["msg"]);
  ASSERT_TRUE(jd.HasMember("head"));
  ASSERT_EQ("/train", jd["head"]["method"]);
  ASSERT_TRUE(jd["head"]["time"].GetDouble() >= 0);
  ASSERT_TRUE(jd.HasMember("body"));
  ASSERT_TRUE(jd["body"]["measure"].HasMember("train_loss"));
  ASSERT_TRUE(fabs(jd["body"]["measure"]["train_loss"].GetDouble()) > 0);
  ASSERT_TRUE(jd["body"]["measure"].HasMember("L1_mean_error"));
  ASSERT_TRUE(jd["body"]["measure"]["L1_max_error_0"].GetDouble() > 0.0);
  ASSERT_TRUE(jd["body"]["parameters"]["input"].HasMember("max_vals"));
  ASSERT_TRUE(jd["body"]["parameters"]["input"].HasMember("min_vals"));

  std::string str_min_vals
      = japi.jrender(jd["body"]["parameters"]["input"]["min_vals"]);
  std::string str_max_vals
      = japi.jrender(jd["body"]["parameters"]["input"]["max_vals"]);

  //  predict
  std::string jpredictstr
      = "{\"service\":\"" + sname
        + "\",\"parameters\":{\"input\":{\"timesteps\":999,\"connector\":"
          "\"caffe\",\"scale\":true,\"min_vals\":"
        + str_min_vals + ",\"max_vals\":" + str_max_vals
        + "},\"output\":{}},\"data\":[\"" + csvts_predict + "\"]}";
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  // std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200, jd["status"]["code"]);
  std::string uri = jd["body"]["predictions"][0]["uri"].GetString();
  ASSERT_EQ("../examples/all/sinus/predict/seq_2.csv", uri);
  ASSERT_TRUE(jd["body"]["predictions"][0]["series"].IsArray());
  ASSERT_TRUE(jd["body"]["predictions"][0]["series"][0]["out"][0].GetDouble()
              >= -1.0);

  std::vector<double> pred_caffe;
  for (size_t i = 0; i < jd["body"]["predictions"][0]["series"].Size(); ++i)
    pred_caffe.push_back(
        jd["body"]["predictions"][0]["series"][i]["out"][0].GetDouble());

  //  remove service
  jstr = "{\"clear\":\"mem\"}";
  joutstr = japi.jrender(japi.service_delete(sname, jstr));
  ASSERT_EQ(ok_str, joutstr);
  //  rmdir(csvts_repo.c_str());

  jstr = "{\"mllib\":\"ncnn\",\"description\":\"lstm-predict\",\"type\":"
         "\"supervised\",\"model\":{\"repository\":\""
         + csvts_repo
         + "\"},\"parameters\":{\"input\":{\"timesteps\":999,\"connector\":"
           "\"csvts\",\"label\":["
           "\"output\"]}"
           "}}";
  joutstr = japi.jrender(japi.service_create(sname, jstr));
  ASSERT_EQ(created_str, joutstr);

  jpredictstr
      = "{\"service\":\"" + sname
        + "\",\"parameters\":{\"input\":{\"timesteps\":999,\"connector\":"
          "\"csvts\",\"scale\":true,\"min_vals\":"
        + str_min_vals + ",\"max_vals\":" + str_max_vals
        + "},\"output\":{}},\"data\":[\"" + csvts_predict + "\"]}";
  std::string joutstr2 = japi.jrender(japi.service_predict(jpredictstr));

  // std::cout << "joutstr2=" << joutstr2 << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr2.c_str());

  std::vector<double> pred_ncnn;
  for (size_t i = 0; i < jd["body"]["predictions"][0]["series"].Size(); ++i)
    pred_ncnn.push_back(
        jd["body"]["predictions"][0]["series"][i]["out"][0].GetDouble());
  for (size_t i = 0; i < pred_ncnn.size(); ++i)
    ASSERT_NEAR(pred_ncnn[i], pred_caffe[i], 1E-6);

  //  remove service
  jstr = "{\"clear\":\"full\"}";
  joutstr = japi.jrender(japi.service_delete(sname, jstr));
  ASSERT_EQ(ok_str, joutstr);
  rmdir(csvts_repo.c_str());
}
#endif

TEST(ncnnapi, ocr)
{
  // create service
  JsonAPI japi;
  std::string sname = "ocr";
  std::string jstr
      = "{\"mllib\":\"ncnn\",\"description\":\"ocr\",\"type\":"
        "\"supervised\",\"model\":{\"repository\":\""
        + ocr_repo
        + "\"},\"parameters\":{\"input\":{\"connector\":\"image\",\"ctc\":"
          "true, \"height\":136,\"width\":220},\"mllib\":{\"nclasses\":69}}}";
  std::string joutstr = japi.jrender(japi.service_create(sname, jstr));
  ASSERT_EQ(created_str, joutstr);

  // predict
  std::string jpredictstr = "{\"service\":\"ocr\",\"parameters\":{\"input\":{}"
                            ",\"output\":{\"confidence_threshold\":0,\"ctc\":"
                            "true,\"blank_label\":0}},\"data\":[\""
                            + ocr_repo + "word_ocr.jpg\"]}";
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  JDoc jd;
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200, jd["status"]["code"]);
  ASSERT_TRUE(jd["body"]["predictions"].IsArray());
  ASSERT_TRUE(jd["body"]["predictions"].Size() == 1);
  ASSERT_TRUE(jd["body"]["predictions"][0]["classes"][0]["cat"] == "beleved");
}
