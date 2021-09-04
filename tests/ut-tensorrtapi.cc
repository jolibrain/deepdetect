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
#include <cuda_runtime_api.h>

using namespace dd;

static std::string ok_str = "{\"status\":{\"code\":200,\"msg\":\"OK\"}}";
static std::string created_str
    = "{\"status\":{\"code\":201,\"msg\":\"Created\"}}";
static std::string bad_param_str
    = "{\"status\":{\"code\":400,\"msg\":\"BadRequest\"}}";
static std::string not_found_str
    = "{\"status\":{\"code\":404,\"msg\":\"NotFound\"}}";

static std::string squeez_repo = "../examples/trt/squeezenet_ssd_trt/";
static std::string refinedet_repo = "../examples/trt/faces_512/";
static std::string squeezv1_repo = "../examples/trt/squeezenet_v1/";
static std::string resnet_onnx_repo = "../examples/trt/resnet_onnx_trt/";
static std::string cyclegan_onnx_repo
    = "../examples/trt/cyclegan_resnet_attn_onnx_trt/";

inline std::string get_trt_archi()
{
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  return std::to_string(prop.major) + std::to_string(prop.minor);
}

TEST(tensorrtapi, service_predict)
{
  // create service
  JsonAPI japi;
  std::string sname = "imgserv";
  std::string jstr
      = "{\"mllib\":\"tensorrt\",\"description\":\"squeezenet-ssd\",\"type\":"
        "\"supervised\",\"model\":{\"repository\":\""
        + squeez_repo
        + "\"},\"parameters\":{\"input\":{\"connector\":\"image\",\"height\":"
          "300,\"width\":300},\"mllib\":{\"nclasses\":21}}}";
  std::string joutstr = japi.jrender(japi.service_create(sname, jstr));
  ASSERT_EQ(created_str, joutstr);

  // predict
  std::string jpredictstr
      = "{\"service\":\"imgserv\",\"parameters\":{\"input\":{\"height\":300,"
        "\"width\":300},\"output\":{\"bbox\":true}},\"data\":[\""
        + squeez_repo + "face.jpg\"]}";
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
  ASSERT_TRUE(fileops::file_exists(squeez_repo + "TRTengine_arch"
                                   + get_trt_archi() + "_bs48"));
  // ASSERT_TRUE(!fileops::remove_file(squeez_repo, "net_tensorRT.proto"));
  // ASSERT_TRUE(!fileops::remove_file(squeez_repo, "TRTengine_archXX_bs48"));
  jstr = "{\"clear\":\"lib\"}";
  joutstr = japi.jrender(japi.service_delete(sname, jstr));
  ASSERT_EQ(ok_str, joutstr);
  ASSERT_TRUE(!fileops::file_exists(squeez_repo + "net_tensorRT.proto"));
  ASSERT_TRUE(!fileops::file_exists(squeez_repo + "TRTengine_arch"
                                    + get_trt_archi() + "_bs48"));
}

TEST(tensorrtapi, service_predict_best)
{
  // create service
  JsonAPI japi;
  std::string sname = "imagenet";
  std::string jstr
      = "{\"mllib\":\"tensorrt\",\"description\":\"age_classif\",\"type\":"
        "\"supervised\",\"model\":{\"repository\":\""
        + squeezv1_repo
        + "\"},\"parameters\":{\"input\":{\"connector\":\"image\",\"height\":"
          "227,\"width\":227},\"mllib\":{\"datatype\":\"fp32\","
          "\"maxBatchSize\":1,\"maxWorkspaceSize\":256,"
          "\"tensorRTEngineFile\":\"TRTengine\",\"gpuid\":0}}}";
  std::string joutstr = japi.jrender(japi.service_create(sname, jstr));
  ASSERT_EQ(created_str, joutstr);

  // predict
  std::string jpredictstr
      = "{\"service\":\"imagenet\",\"parameters\":{\"input\":{\"height\":227,"
        "\"width\":227},\"output\":{\"best\":2}},\"data\":[\""
        + squeez_repo + "face.jpg\"]}";
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  JDoc jd;
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200, jd["status"]["code"]);
  ASSERT_TRUE(jd["body"]["predictions"].IsArray());
  ASSERT_EQ(2, jd["body"]["predictions"][0]["classes"].Size());
  std::string cls
      = jd["body"]["predictions"][0]["classes"][0]["cat"].GetString();
  ASSERT_TRUE(cls == "n04357314 sunscreen, sunblock, sun blocker");
  std::cout << "looking for " << squeezv1_repo << "TRTengine_arch"
            << get_trt_archi() << "_bs1" << std::endl;
  ASSERT_TRUE(fileops::file_exists(squeezv1_repo + "TRTengine_arch"
                                   + get_trt_archi() + "_bs1"));
  jstr = "{\"clear\":\"lib\"}";
  joutstr = japi.jrender(japi.service_delete(sname, jstr));
  ASSERT_EQ(ok_str, joutstr);
  ASSERT_TRUE(!fileops::file_exists(squeezv1_repo + "net_tensorRT.proto"));
  ASSERT_TRUE(!fileops::file_exists(squeezv1_repo + "TRTengine_arch"
                                    + get_trt_archi() + "_bs1"));
}

TEST(tensorrtapi, service_predict_refinedet)
{
  // create service
  JsonAPI japi;
  std::string sname = "imgserv";
  std::string jstr
      = "{\"mllib\":\"tensorrt\",\"description\":\"refinedet\",\"type\":"
        "\"supervised\",\"model\":{\"repository\":\""
        + refinedet_repo
        + "\"},\"parameters\":{\"input\":{\"connector\":\"image\",\"height\":"
          "512,\"width\":512},\"mllib\":{\"nclasses\":2,\"datatype\":\"fp16\","
          "\"maxBatchSize\":1,\"maxWorkSpaceSize\":256}}}";
  std::string joutstr = japi.jrender(japi.service_create(sname, jstr));
  ASSERT_EQ(created_str, joutstr);

  // predict
  std::string jpredictstr
      = "{\"service\":\"imgserv\",\"parameters\":{\"input\":{\"height\":512,"
        "\"width\":512},\"output\":{\"bbox\":true}},\"data\":[\""
        + squeez_repo + "face.jpg\"]}";
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  JDoc jd;
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200, jd["status"]["code"]);
  ASSERT_TRUE(jd["body"]["predictions"].IsArray());

  // predict with wrong input size
  jpredictstr
      = "{\"service\":\"imgserv\",\"parameters\":{\"input\":{\"height\":300,"
        "\"width\":300},\"output\":{\"bbox\":true}},\"data\":[\""
        + squeez_repo + "face.jpg\"]}";
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(400, jd["status"]["code"]);

  ASSERT_TRUE(fileops::file_exists(refinedet_repo + "TRTengine_arch"
                                   + get_trt_archi() + "_bs1"));
  jstr = "{\"clear\":\"lib\"}";
  joutstr = japi.jrender(japi.service_delete(sname, jstr));
  ASSERT_EQ(ok_str, joutstr);
  ASSERT_TRUE(!fileops::file_exists(refinedet_repo + "net_tensorRT.proto"));
  ASSERT_TRUE(!fileops::file_exists(refinedet_repo + "TRTengine_arch"
                                    + get_trt_archi() + "_bs1"));
}

TEST(tensorrtapi, service_predict_onnx)
{
  // create service
  JsonAPI japi;
  std::string sname = "onnx";
  std::string jstr
      = "{\"mllib\":\"tensorrt\",\"description\":\"Test onnx "
        "import\",\"type\":\"supervised\",\"model\":{\"repository\":\""
        + resnet_onnx_repo
        + "\"},\"parameters\":{\"input\":{\"connector\":\"image\",\"height\":"
          "224,\"width\":224,\"rgb\":true,\"scale\":0.0039},\"mllib\":{"
          "\"maxBatchSize\":1,\"maxWorkspaceSize\":256,\"gpuid\":0,"
          "\"nclasses\":1000}}}";
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
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200, jd["status"]["code"]);
  ASSERT_TRUE(jd["body"]["predictions"].IsArray());
  std::string cl1
      = jd["body"]["predictions"][0]["classes"][0]["cat"].GetString();
  ASSERT_TRUE(cl1 == "n02123159 tiger cat");
  std::cerr << "[WARNING] Confidence is > 1 because the model is traced "
               "directly from torchvision and does not perform softmax."
            << std::endl;
  ASSERT_TRUE(jd["body"]["predictions"][0]["classes"][0]["prob"].GetDouble()
              > 0.3);
}

TEST(tensorrtapi, service_predict_gan_onnx)
{
  // create service
  JsonAPI japi;
  std::string sname = "onnx";
  std::string jstr
      = "{\"mllib\":\"tensorrt\",\"description\":\"Test gan onnx "
        "import\",\"type\":\"supervised\",\"model\":{\"repository\":\""
        + cyclegan_onnx_repo
        + "\"},\"parameters\":{\"input\":{\"connector\":\"image\",\"height\":"
          "360,\"width\":360},\"mllib\":{"
          "\"maxBatchSize\":1,\"maxWorkspaceSize\":256,\"gpuid\":0,"
          "\"datatype\":\"fp16\"}}}";
  std::string joutstr = japi.jrender(japi.service_create(sname, jstr));
  ASSERT_EQ(created_str, joutstr);

  // predict
  std::string jpredictstr
      = "{\"service\":\"" + sname
        + "\",\"parameters\":{\"input\":{\"height\":360,"
          "\"width\":360,\"rgb\":true,\"scale\":0.00392,\"mean\":[0.5,0.5,0.5]"
          ",\"std\":[0.5,0.5,0.5]},\"output\":{},\"mllib\":{\"extract_layer\":"
          "\"last\"}},\"data\":[\""
        + cyclegan_onnx_repo + "horse.jpg\"]}";
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  JDoc jd;
  // std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200, jd["status"]["code"]);
  ASSERT_TRUE(jd["body"]["predictions"].IsArray());
  ASSERT_TRUE(jd["body"]["predictions"][0]["vals"].IsArray());
  ASSERT_EQ(jd["body"]["predictions"][0]["vals"].Size(), 360 * 360 * 3);

  ASSERT_TRUE(fileops::file_exists(cyclegan_onnx_repo + "TRTengine_arch"
                                   + get_trt_archi() + "_bs1"));

  jstr = "{\"clear\":\"lib\"}";
  joutstr = japi.jrender(japi.service_delete(sname, jstr));
  ASSERT_EQ(ok_str, joutstr);
  ASSERT_TRUE(!fileops::file_exists(cyclegan_onnx_repo + "TRTengine_arch"
                                    + get_trt_archi() + "_bs1"));
}
