/**
 * DeepDetect
 * Copyright (c) 2026 Pixel Forensics, Inc.
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

static std::string resnet_onnx_repo = "../examples/trt/resnet_onnx_trt/";
static std::string yolox_onnx_repo = "../examples/trt/yolox_onnx_trt_nowrap/";
static std::string cyclegan_onnx_repo
    = "../examples/trt/cyclegan_resnet_attn_onnx_trt/";

TEST(onnxruntimeapi, service_predict_classification)
{
  // create service
  JsonAPI japi;
  std::string sname = "ort_cls";
  std::string jstr
      = "{\"mllib\":\"onnxruntime\",\"description\":\"Test onnxruntime "
        "classification\",\"type\":\"supervised\",\"model\":{\"repository\":\""
        + resnet_onnx_repo
        + "\"},\"parameters\":{\"input\":{\"connector\":\"image\",\"height\":"
          "224,\"width\":224,\"rgb\":true,\"scale\":0.0039},\"mllib\":{"
          "\"gpu\":false,\"max_batch_size\":1,\"nclasses\":1000}}}";
  std::string joutstr = japi.jrender(japi.service_create(sname, jstr));
  ASSERT_EQ(created_str, joutstr);

  // predict
  std::string jpredictstr
      = "{\"service\":\"" + sname
        + "\",\"parameters\":{\"input\":{\"height\":224,"
          "\"width\":224},\"output\":{\"best\":1}},\"data\":[\""
        + resnet_onnx_repo + "cat.jpg\"]}";
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  JDoc jd;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());

  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200, jd["status"]["code"].GetInt());
  ASSERT_TRUE(jd["body"]["predictions"].IsArray());

  std::string cl1
      = jd["body"]["predictions"][0]["classes"][0]["cat"].GetString();
  ASSERT_EQ(cl1, "n02123159 tiger cat");
  ASSERT_TRUE(jd["body"]["predictions"][0]["classes"][0]["prob"].GetDouble()
              > 0.3);

  // delete
  jstr = "{\"clear\":\"lib\"}";
  joutstr = japi.jrender(japi.service_delete(sname, jstr));
  ASSERT_EQ(ok_str, joutstr);
}

TEST(onnxruntimeapi, service_predict_bbox)
{
  // create service
  JsonAPI japi;
  std::string sname = "ort_det";
  std::string jstr
      = "{\"mllib\":\"onnxruntime\",\"description\":\"Test onnxruntime "
        "detection\",\"type\":\"supervised\",\"model\":{\"repository\":\""
        + yolox_onnx_repo
        + "\"},\"parameters\":{\"input\":{\"connector\":\"image\",\"height\":"
          "640,\"width\":640,\"rgb\":true},\"mllib\":{\"gpu\":false,"
          "\"bbox\":true,\"max_batch_size\":2,\"nclasses\":81}}}";
  std::string joutstr = japi.jrender(japi.service_create(sname, jstr));
  ASSERT_EQ(created_str, joutstr);

  // predict
  std::string jpredictstr
      = "{\"service\":\"" + sname
        + "\",\"parameters\":{\"input\":{},\"output\":{\"bbox\":true,"
          "\"confidence_threshold\":0.8}},\"data\":[\""
        + resnet_onnx_repo + "cat.jpg\",\"" + yolox_onnx_repo + "dog.jpg\"]}";
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  JDoc jd;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());

  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200, jd["status"]["code"].GetInt());
  ASSERT_TRUE(jd["body"]["predictions"].IsArray());
  ASSERT_EQ(jd["body"]["predictions"].Size(), 2);

  uint32_t cat_id = jd["body"]["predictions"][0]["uri"].GetString()
                            == (resnet_onnx_repo + "cat.jpg")
                        ? 0
                        : 1;
  uint32_t dog_id = 1 - cat_id;

  // Check first prediction (cat)
  auto &preds = jd["body"]["predictions"][cat_id]["classes"];
  ASSERT_EQ(preds.Size(), 1);
  std::string cl1 = preds[0]["cat"].GetString();
  ASSERT_EQ(cl1, "16");
  ASSERT_TRUE(preds[0]["prob"].GetDouble() > 0.8);

  auto &bbox = preds[0]["bbox"];
  ASSERT_TRUE(bbox["xmin"].GetDouble() < 50 && bbox["xmax"].GetDouble() > 200
              && bbox["ymin"].GetDouble() < 50
              && bbox["ymax"].GetDouble() > 200);

  // Check second prediction (dog)
  auto &preds2 = jd["body"]["predictions"][dog_id]["classes"];
  ASSERT_EQ(preds2.Size(), 1);
  std::string cl2 = preds2[0]["cat"].GetString();
  ASSERT_EQ(cl2, "17");
  ASSERT_TRUE(preds2[0]["prob"].GetDouble() > 0.8);

  // delete
  jstr = "{\"clear\":\"lib\"}";
  joutstr = japi.jrender(japi.service_delete(sname, jstr));
  ASSERT_EQ(ok_str, joutstr);
}

TEST(onnxruntimeapi, service_predict_extract_layer)
{
  // create service
  JsonAPI japi;
  std::string sname = "ort_extract";
  std::string jstr
      = "{\"mllib\":\"onnxruntime\",\"description\":\"Test onnxruntime "
        "feature extraction\",\"type\":\"supervised\",\"model\":{"
        "\"repository\":\""
        + cyclegan_onnx_repo
        + "\"},\"parameters\":{\"input\":{\"connector\":\"image\",\"height\":"
          "360,\"width\":360},\"mllib\":{\"gpu\":false,"
          "\"max_batch_size\":1}}}";
  std::string joutstr = japi.jrender(japi.service_create(sname, jstr));
  ASSERT_EQ(created_str, joutstr);

  // predict extracting raw tensor values
  std::string jpredictstr
      = "{\"service\":\"" + sname
        + "\",\"parameters\":{\"input\":{\"height\":360,"
          "\"width\":360,\"rgb\":true,\"scale\":0.00392,\"mean\":[0.5,0.5,0.5]"
          ",\"std\":[0.5,0.5,0.5]},\"output\":{},\"mllib\":{\"extract_layer\":"
          "\"last\"}},\"data\":[\""
        + cyclegan_onnx_repo + "horse.jpg\"]}";
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  JDoc jd;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());

  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200, jd["status"]["code"].GetInt());
  ASSERT_TRUE(jd["body"]["predictions"].IsArray());
  ASSERT_TRUE(jd["body"]["predictions"][0]["vals"].IsArray());

  // 360 x 360 x 3 = 388800 expected float values in the output tensor
  ASSERT_EQ(jd["body"]["predictions"][0]["vals"].Size(), 360 * 360 * 3);

  // delete
  jstr = "{\"clear\":\"lib\"}";
  joutstr = japi.jrender(japi.service_delete(sname, jstr));
  ASSERT_EQ(ok_str, joutstr);
}

#if !defined(APPLE) && !defined(USE_ONNXRUNTIME_CPU_ONLY)
TEST(onnxruntimeapi, service_predict_gpu)
{
  // create GPU service
  JsonAPI japi;
  std::string sname = "ort_gpu_det";
  std::string jstr
      = "{\"mllib\":\"onnxruntime\",\"description\":\"Test onnxruntime GPU "
        "execution\",\"type\":\"supervised\",\"model\":{\"repository\":\""
        + yolox_onnx_repo
        + "\"},\"parameters\":{\"input\":{\"connector\":\"image\",\"height\":"
          "640,\"width\":640,\"rgb\":true},\"mllib\":{\"gpu\":true,"
          "\"gpuid\":0,\"bbox\":true,\"max_batch_size\":2,\"nclasses\":81}}}";
  std::string joutstr = japi.jrender(japi.service_create(sname, jstr));
  ASSERT_EQ(created_str, joutstr);

  // predict with CPU input preprocessing + CUDA Execution Provider
  std::string jpredictstr
      = "{\"service\":\"" + sname
        + "\",\"parameters\":{\"input\":{},\"output\":{\"bbox\":true,"
          "\"confidence_threshold\":0.8}},\"data\":[\""
        + resnet_onnx_repo + "cat.jpg\",\"" + yolox_onnx_repo + "dog.jpg\"]}";
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  JDoc jd;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());

  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200, jd["status"]["code"].GetInt());
  ASSERT_TRUE(jd["body"]["predictions"].IsArray());
  ASSERT_EQ(jd["body"]["predictions"].Size(), 2);

  uint32_t cat_id = jd["body"]["predictions"][0]["uri"].GetString()
                            == (resnet_onnx_repo + "cat.jpg")
                        ? 0
                        : 1;
  auto &preds = jd["body"]["predictions"][cat_id]["classes"];
  ASSERT_EQ(preds.Size(), 1);
  ASSERT_EQ(std::string(preds[0]["cat"].GetString()), "16");
  ASSERT_TRUE(preds[0]["prob"].GetDouble() > 0.8);

#ifdef USE_CUDA_CV
  // predict with GPU input preprocessing (GpuMatToORTBuffer) + zero-copy CUDA tensor
  jpredictstr
      = "{\"service\":\"" + sname
        + "\",\"parameters\":{\"input\":{\"cuda\":true},\"output\":{"
          "\"bbox\":true,\"confidence_threshold\":0.8}},\"data\":[\""
        + resnet_onnx_repo + "cat.jpg\"]}";
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  jd = JDoc();
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());

  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200, jd["status"]["code"].GetInt());
  ASSERT_TRUE(jd["body"]["predictions"].IsArray());
  ASSERT_EQ(jd["body"]["predictions"].Size(), 1);

  auto &preds_cat = jd["body"]["predictions"][0]["classes"];
  ASSERT_EQ(preds_cat.Size(), 1);
  ASSERT_EQ(std::string(preds_cat[0]["cat"].GetString()), "16");
  ASSERT_TRUE(preds_cat[0]["prob"].GetDouble() > 0.8);
  auto &bbox_cat = preds_cat[0]["bbox"];
  ASSERT_TRUE(bbox_cat["xmin"].GetDouble() < 50
              && bbox_cat["xmax"].GetDouble() > 200
              && bbox_cat["ymin"].GetDouble() < 50
              && bbox_cat["ymax"].GetDouble() > 200);
#endif

  // delete
  jstr = "{\"clear\":\"lib\"}";
  joutstr = japi.jrender(japi.service_delete(sname, jstr));
  ASSERT_EQ(ok_str, joutstr);
}
#endif