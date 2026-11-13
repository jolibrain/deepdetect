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
#include <fstream>
#include <sys/stat.h>

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
  // create service with template="yolox" to decode [batch, 8400, 85] + NMS
  JsonAPI japi;
  std::string sname = "ort_det";
  std::string jstr
      = "{\"mllib\":\"onnxruntime\",\"description\":\"Test onnxruntime "
        "detection\",\"type\":\"supervised\",\"model\":{\"repository\":\""
        + yolox_onnx_repo
        + "\"},\"parameters\":{\"input\":{\"connector\":\"image\",\"height\":"
          "640,\"width\":640,\"rgb\":true},\"mllib\":{\"template\":\"yolox\","
          "\"gpu\":false,\"max_batch_size\":2,\"nclasses\":81}}}";
  std::string joutstr = japi.jrender(japi.service_create(sname, jstr));
  ASSERT_EQ(created_str, joutstr);

  // multi-batch predict
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

  auto &bbox2 = preds2[0]["bbox"];
  ASSERT_TRUE(bbox2["xmin"].GetDouble() >= 0.0
              && bbox2["xmax"].GetDouble() > bbox2["xmin"].GetDouble()
              && bbox2["ymin"].GetDouble() >= 0.0
              && bbox2["ymax"].GetDouble() > bbox2["ymin"].GetDouble());

  // delete
  jstr = "{\"clear\":\"lib\"}";
  joutstr = japi.jrender(japi.service_delete(sname, jstr));
  ASSERT_EQ(ok_str, joutstr);
}

TEST(onnxruntimeapi, service_predict_bbox_invalid_shape)
{
  // create service without template="yolox" so backend expects [batch,
  // max_dets, 7]
  JsonAPI japi;
  std::string sname = "ort_det_invalid";
  std::string jstr
      = "{\"mllib\":\"onnxruntime\",\"description\":\"Test invalid bbox "
        "shape\",\"type\":\"supervised\",\"model\":{\"repository\":\""
        + yolox_onnx_repo
        + "\"},\"parameters\":{\"input\":{\"connector\":\"image\",\"height\":"
          "640,\"width\":640,\"rgb\":true},\"mllib\":{\"gpu\":false,"
          "\"bbox\":true,\"max_batch_size\":2,\"nclasses\":81}}}";
  std::string joutstr = japi.jrender(japi.service_create(sname, jstr));
  ASSERT_EQ(created_str, joutstr);

  std::string jpredictstr
      = "{\"service\":\"" + sname
        + "\",\"parameters\":{\"input\":{},\"output\":{\"bbox\":true}},"
          "\"data\":[\""
        + resnet_onnx_repo + "cat.jpg\",\"" + yolox_onnx_repo + "dog.jpg\"]}";
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  JDoc jd;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());

  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(400, jd["status"]["code"].GetInt());

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
          "640,\"width\":640,\"rgb\":true},\"mllib\":{\"template\":\"yolox\","
          "\"gpu\":true,\"gpuid\":0,\"max_batch_size\":2,\"nclasses\":81}}}";
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
  auto &bbox = preds[0]["bbox"];
  ASSERT_TRUE(bbox["xmin"].GetDouble() < 50 && bbox["xmax"].GetDouble() > 200
              && bbox["ymin"].GetDouble() < 50
              && bbox["ymax"].GetDouble() > 200);

#ifdef USE_CUDA_CV
  // predict with GPU input preprocessing (GpuMatToORTBuffer) + zero-copy CUDA
  // tensor
  jpredictstr = "{\"service\":\"" + sname
                + "\",\"parameters\":{\"input\":{\"cuda\":true},\"output\":{"
                  "\"bbox\":true,\"confidence_threshold\":0.8}},\"data\":[\""
                + resnet_onnx_repo + "cat.jpg\",\"" + yolox_onnx_repo
                + "dog.jpg\"]}";
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  jd = JDoc();
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());

  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200, jd["status"]["code"].GetInt());
  ASSERT_TRUE(jd["body"]["predictions"].IsArray());
  ASSERT_EQ(jd["body"]["predictions"].Size(), 2);

  cat_id = jd["body"]["predictions"][0]["uri"].GetString()
                   == (resnet_onnx_repo + "cat.jpg")
               ? 0
               : 1;
  uint32_t dog_id2 = 1 - cat_id;

  auto &preds_cat = jd["body"]["predictions"][cat_id]["classes"];
  ASSERT_EQ(preds_cat.Size(), 1);
  ASSERT_EQ(std::string(preds_cat[0]["cat"].GetString()), "16");
  ASSERT_TRUE(preds_cat[0]["prob"].GetDouble() > 0.8);
  auto &bbox_cat = preds_cat[0]["bbox"];
  ASSERT_TRUE(bbox_cat["xmin"].GetDouble() < 50
              && bbox_cat["xmax"].GetDouble() > 200
              && bbox_cat["ymin"].GetDouble() < 50
              && bbox_cat["ymax"].GetDouble() > 200);

  auto &preds_dog = jd["body"]["predictions"][dog_id2]["classes"];
  ASSERT_GE(preds_dog.Size(), 1);
  ASSERT_EQ(std::string(preds_dog[0]["cat"].GetString()), "17");
  ASSERT_TRUE(preds_dog[0]["prob"].GetDouble() > 0.8);
#endif

  // delete
  jstr = "{\"clear\":\"lib\"}";
  joutstr = japi.jrender(japi.service_delete(sname, jstr));
  ASSERT_EQ(ok_str, joutstr);
}
#endif

TEST(onnxruntimeapi, service_predict_bbox_7col)
{
  // Create a minimal [1, 3, 7] detection ONNX model + corresp.txt in /tmp
  std::string repo = "/tmp/dd_ort_det_7col/";
  mkdir(repo.c_str(), 0755);

  // Constant node outputting float[1, 3, 7]:
  //   row 0: [0.0, 1.0, 0.95, 0.1, 0.2, 0.5, 0.6] (valid detection)
  //   row 1: [0.0, 2.0, 0.20, 0.0, 0.0, 1.0, 1.0] (below conf threshold 0.5)
  //   row 2: [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] (padding sentinel)
  static const uint8_t kDetOnnxModel[] = {
    0x08, 0x07, 0x42, 0x04, 0x0a, 0x00, 0x10, 0x0b, 0x3a, 0xe3, 0x01, 0x0a,
    0x90, 0x01, 0x12, 0x0d, 0x64, 0x65, 0x74, 0x65, 0x63, 0x74, 0x69, 0x6f,
    0x6e, 0x5f, 0x6f, 0x75, 0x74, 0x22, 0x08, 0x43, 0x6f, 0x6e, 0x73, 0x74,
    0x61, 0x6e, 0x74, 0x2a, 0x75, 0x0a, 0x05, 0x76, 0x61, 0x6c, 0x75, 0x65,
    0x2a, 0x69, 0x08, 0x01, 0x08, 0x03, 0x08, 0x07, 0x10, 0x01, 0x42, 0x09,
    0x63, 0x6f, 0x6e, 0x73, 0x74, 0x5f, 0x6f, 0x75, 0x74, 0x4a, 0x54, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x3f, 0x33, 0x33, 0x73, 0x3f, 0xcd,
    0xcc, 0xcc, 0x3d, 0xcd, 0xcc, 0x4c, 0x3e, 0x00, 0x00, 0x00, 0x3f, 0x9a,
    0x99, 0x19, 0x3f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0xcd,
    0xcc, 0x4c, 0x3e, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x80, 0x3f, 0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x80, 0xbf, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xa0,
    0x01, 0x04, 0x12, 0x08, 0x74, 0x65, 0x73, 0x74, 0x5f, 0x64, 0x65, 0x74,
    0x5a, 0x1f, 0x0a, 0x05, 0x69, 0x6e, 0x70, 0x75, 0x74, 0x12, 0x16, 0x0a,
    0x14, 0x08, 0x01, 0x12, 0x10, 0x0a, 0x02, 0x08, 0x01, 0x0a, 0x02, 0x08,
    0x03, 0x0a, 0x02, 0x08, 0x08, 0x0a, 0x02, 0x08, 0x08, 0x62, 0x23, 0x0a,
    0x0d, 0x64, 0x65, 0x74, 0x65, 0x63, 0x74, 0x69, 0x6f, 0x6e, 0x5f, 0x6f,
    0x75, 0x74, 0x12, 0x12, 0x0a, 0x10, 0x08, 0x01, 0x12, 0x0c, 0x0a, 0x02,
    0x08, 0x01, 0x0a, 0x02, 0x08, 0x03, 0x0a, 0x02, 0x08, 0x07
  };

  {
    std::ofstream mf(repo + "model.onnx", std::ios::binary);
    mf.write(reinterpret_cast<const char *>(kDetOnnxModel),
             sizeof(kDetOnnxModel));
    std::ofstream cf(repo + "corresp.txt");
    cf << "0 background\n1 cat\n2 dog\n";
  }

  JsonAPI japi;
  std::string sname = "ort_det_7col";
  std::string jstr
      = "{\"mllib\":\"onnxruntime\",\"description\":\"Test 7-col bbox "
        "output\",\"type\":\"supervised\",\"model\":{\"repository\":\""
        + repo
        + "\"},\"parameters\":{\"input\":{\"connector\":\"image\",\"height\":"
          "8,\"width\":8,\"rgb\":true},\"mllib\":{\"gpu\":false,"
          "\"bbox\":true,\"max_batch_size\":1,\"nclasses\":3}}}";
  std::string joutstr = japi.jrender(japi.service_create(sname, jstr));
  ASSERT_EQ(created_str, joutstr);

  std::string jpredictstr
      = "{\"service\":\"" + sname
        + "\",\"parameters\":{\"input\":{},\"output\":{\"bbox\":true,"
          "\"confidence_threshold\":0.5}},\"data\":[\""
        + resnet_onnx_repo + "cat.jpg\"]}";
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  JDoc jd;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());

  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200, jd["status"]["code"].GetInt());
  auto &preds = jd["body"]["predictions"][0]["classes"];
  ASSERT_EQ(preds.Size(), 1);
  ASSERT_EQ(std::string(preds[0]["cat"].GetString()), "cat");
  ASSERT_NEAR(preds[0]["prob"].GetDouble(), 0.95, 1e-4);

  auto &bbox = preds[0]["bbox"];
  ASSERT_TRUE(bbox["xmin"].GetDouble() > 0.0
              && bbox["xmax"].GetDouble() > bbox["xmin"].GetDouble()
              && bbox["ymin"].GetDouble() > 0.0
              && bbox["ymax"].GetDouble() > bbox["ymin"].GetDouble());

  joutstr = japi.jrender(japi.service_delete(sname, "{\"clear\":\"lib\"}"));
  ASSERT_EQ(ok_str, joutstr);
  fileops::remove_dir(repo);
}
