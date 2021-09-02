/**
 * DeepDetect
 * Copyright (c) 2021 Jolibrain
 * Author: Louis Jean <louis.jean@jolibrain.com>
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

#ifdef USE_TENSORRT
#include <cuda_runtime_api.h>
#endif

using namespace dd;

static std::string ok_str = "{\"status\":{\"code\":200,\"msg\":\"OK\"}}";
static std::string created_str
    = "{\"status\":{\"code\":201,\"msg\":\"Created\"}}";
static std::string bad_param_str
    = "{\"status\":{\"code\":400,\"msg\":\"BadRequest\"}}";
static std::string not_found_str
    = "{\"status\":{\"code\":404,\"msg\":\"NotFound\"}}";

static std::string gpuid = "0"; // change as needed

static std::string torch_detect_repo = "../examples/torch/fasterrcnn_torch";
static std::string torch_classif_repo = "../examples/torch/resnet50_torch";

static std::string caffe_word_detect_repo = "../examples/caffe/word_detect_v2";
static std::string caffe_ocr_repo = "../examples/caffe/multiword_ocr";

static std::string trt_detect_repo = "../examples/trt/squeezenet_ssd_trt";
static std::string trt_gan_repo
    = "../examples/trt/cyclegan_resnet_attn_onnx_trt";

#ifdef USE_TORCH

TEST(chain, chain_torch_detection_classification)
{
  // create service
  JsonAPI japi;
  std::string detect_sname = "detect";
  std::string jstr
      = "{\"mllib\":\"torch\",\"description\":\"fasterrcnn\",\"type\":"
        "\"supervised\",\"model\":{\"repository\":\""
        + torch_detect_repo
        + "\"},\"parameters\":{\"input\":{\"connector\":\"image\",\"height\":"
          "224,\"width\":224,\"rgb\":true,\"scale\":0.0039},"
          "\"mllib\":{\"nclasses\":91,\"template\":\"fasterrcnn\"}}}";
  std::string joutstr = japi.jrender(japi.service_create(detect_sname, jstr));
  ASSERT_EQ(created_str, joutstr);

  std::string classif_sname = "classif";
  jstr = "{\"mllib\":\"torch\",\"description\":\"squeezenet\",\"type\":"
         "\"supervised\",\"model\":{\"repository\":\""
         + torch_classif_repo
         + "\"},\"parameters\":{\"input\":{\"connector\":\"image\",\"height\":"
           "224,\"width\":224,\"rgb\":true,\"scale\":0.0039},"
           "\"mllib\":{\"nclasses\":1000}}}";
  joutstr = japi.jrender(japi.service_create(classif_sname, jstr));
  ASSERT_EQ(created_str, joutstr);

  // chain predict
  std::string jchainstr
      = "{\"chain\":{\"name\":\"chain\",\"calls\":["
        "{\"service\":\""
        + detect_sname
        + "\",\"parameters\":{\"input\":{\"keep_orig\":true},\"output\":{"
          "\"bbox\":true,\"confidence_threshold\":0.2}},\"data\":[\""
        + torch_classif_repo
        + "/cat.jpg\"]},"
          "{\"id\":\"crop\",\"action\":{\"type\":\"crop\",\"parameters\":{"
          "\"padding_ratio\":0.05}}},"
          "{\"service\":\""
        + classif_sname
        + "\",\"parent_id\":\"crop\",\"parameters\":{\"output\":{\"best\":1}}}"
          "]}}";
  joutstr = japi.jrender(japi.service_chain("chain", jchainstr));
  JDoc jd;
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200, jd["status"]["code"]);
  ASSERT_TRUE(jd["body"]["predictions"].IsArray());
  ASSERT_EQ(jd["body"]["predictions"].Size(), 1);
  ASSERT_TRUE(jd["body"]["predictions"][0]["classes"].IsArray());
  ASSERT_EQ(jd["body"]["predictions"][0]["classes"].Size(), 2);
  ASSERT_TRUE(jd["body"]["predictions"][0]["classes"][0][classif_sname.c_str()]
                  .IsObject());
  ASSERT_TRUE(jd["body"]["predictions"][0]["classes"][0][classif_sname.c_str()]
                ["classes"]
                    .IsArray());

  // multiple models (tree)
  std::string classif2_sname = "classif2";
  jstr = "{\"mllib\":\"torch\",\"description\":\"squeezenet\",\"type\":"
         "\"supervised\",\"model\":{\"repository\":\""
         + torch_classif_repo
         + "\"},\"parameters\":{\"input\":{\"connector\":\"image\",\"height\":"
           "224,\"width\":224,\"rgb\":true,\"scale\":0.0039},"
           "\"mllib\":{\"nclasses\":1000}}}";
  joutstr = japi.jrender(japi.service_create(classif2_sname, jstr));
  ASSERT_EQ(created_str, joutstr);

  jchainstr
      = "{\"chain\":{\"name\":\"chain\",\"calls\":["
        "{\"service\":\""
        + detect_sname
        + "\",\"parameters\":{\"input\":{\"keep_orig\":true},\"output\":{"
          "\"bbox\":true,\"confidence_threshold\":0.2}},\"data\":[\""
        + torch_classif_repo
        + "/cat.jpg\"]},"
          "{\"id\":\"crop\",\"action\":{\"type\":\"crop\",\"parameters\":{"
          "\"padding_ratio\":0.05}}},"
          "{\"service\":\""
        + classif_sname
        + "\",\"parent_id\":\"crop\",\"parameters\":{\"output\":{\"best\":1}}}"
          ","
          "{\"service\":\""
        + classif2_sname
        + "\",\"parent_id\":\"crop\",\"parameters\":{\"output\":{\"best\":2}}}"
          "]}}";
  joutstr = japi.jrender(japi.service_chain("chain", jchainstr));
  jd = JDoc();
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200, jd["status"]["code"]);
  ASSERT_TRUE(jd["body"]["predictions"].IsArray());
  ASSERT_EQ(jd["body"]["predictions"].Size(), 1);
  ASSERT_TRUE(jd["body"]["predictions"][0]["classes"].IsArray());
  ASSERT_EQ(jd["body"]["predictions"][0]["classes"].Size(), 2);
  ASSERT_TRUE(jd["body"]["predictions"][0]["classes"][0][classif_sname.c_str()]
                ["classes"]
                    .IsArray());
  ASSERT_EQ(jd["body"]["predictions"][0]["classes"][0][classif_sname.c_str()]
              ["classes"]
                  .Size(),
            1);
  ASSERT_EQ(jd["body"]["predictions"][0]["classes"][0][classif2_sname.c_str()]
              ["classes"]
                  .Size(),
            2);

  // cleanup
  fileops::remove_file(torch_detect_repo, "model.json");
}

#endif

#ifdef USE_CAFFE

TEST(chain, chain_caffe_detection_ocr)
{
  // create service
  JsonAPI japi;
  std::string detect_sname = "detect";
  std::string jstr
      = "{\"mllib\":\"caffe\",\"description\":\"detection model\",\"type\":"
        "\"supervised\",\"model\":{\"repository\":\""
        + caffe_word_detect_repo
        + "\"},\"parameters\":{\"input\":{\"connector\":\"image\",\"width\":"
          "512,\"height\":512},\"mllib\":{\"nclasses\":2}}}";
  std::string joutstr = japi.jrender(japi.service_create(detect_sname, jstr));
  ASSERT_EQ(created_str, joutstr);

  std::string ocr_sname = "ocr";
  jstr = "{\"mllib\":\"caffe\",\"description\":\"ocr model\",\"type\":"
         "\"supervised\",\"model\":{\"repository\":\""
         + caffe_ocr_repo
         + "\"},\"parameters\":{\"input\":{\"connector\":\"image\",\"width\":"
           "220,\"height\":136},\"mllib\":{\"nclasses\":69}}}";
  joutstr = japi.jrender(japi.service_create(ocr_sname, jstr));
  ASSERT_EQ(created_str, joutstr);

  // chain predict
  std::string jchainstr
      = "{\"chain\":{\"name\":\"chain\",\"calls\":["
        "{\"service\":\""
        + detect_sname
        + "\",\"parameters\":{\"input\":{\"keep_orig\":true},\"output\":{"
          "\"bbox\":true,\"confidence_threshold\":0.2}},\"data\":[\""
        + caffe_word_detect_repo
        + "/word.png\"]},"
          "{\"id\":\"crop\",\"action\":{\"type\":\"crop\",\"parameters\":{"
          "\"padding_ratio\":0.2}}},"
          "{\"service\":\""
        + ocr_sname
        + "\",\"parent_id\":\"crop\",\"parameters\":{\"output\":{\"ctc\":true,"
          "\"blank_label\":0,\"confidence_threshold\":0}}}"
          "]}}";
  joutstr = japi.jrender(japi.service_chain("chain", jchainstr));
  JDoc jd;
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200, jd["status"]["code"]);
  ASSERT_TRUE(jd["body"]["predictions"].IsArray());
  ASSERT_EQ(jd["body"]["predictions"].Size(), 1);
  ASSERT_TRUE(jd["body"]["predictions"][0]["classes"].IsArray());
  ASSERT_EQ(jd["body"]["predictions"][0]["classes"].Size(), 1);
  ASSERT_TRUE(jd["body"]["predictions"][0]["classes"][0][ocr_sname.c_str()]
                  .IsObject());
  auto &ocr_pred
      = jd["body"]["predictions"][0]["classes"][0][ocr_sname.c_str()];
  ASSERT_TRUE(ocr_pred["classes"].IsArray());
  ASSERT_EQ(ocr_pred["classes"][0]["cat"].GetString(), std::string("word"));

  // cleanup
  fileops::remove_file(caffe_word_detect_repo, "model.json");
  fileops::remove_file(caffe_ocr_repo, "model.json");
}
#endif

#ifdef USE_TENSORRT

inline std::string get_trt_archi()
{
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  return std::to_string(prop.major) + std::to_string(prop.minor);
}

TEST(chain, chain_trt_detection_gan)
{
  // create service
  JsonAPI japi;
  std::string detect_sname = "detect";
  std::string jstr
      = "{\"mllib\":\"tensorrt\",\"description\":\"fasterrcnn\",\"type\":"
        "\"supervised\",\"model\":{\"repository\":\""
        + trt_detect_repo
        + "\"},\"parameters\":{\"input\":{\"connector\":\"image\",\"height\":"
          "300,\"width\":300},\"mllib\":{\"maxBatchSize\":1,"
          "\"maxWorkspaceSize\":256,\"gpuid\":0}}}"; //,\"datatype\":\"fp16\"
  std::string joutstr = japi.jrender(japi.service_create(detect_sname, jstr));
  ASSERT_EQ(created_str, joutstr);

  std::string gan_sname = "gan";
  jstr = "{\"mllib\":\"tensorrt\",\"description\":\"squeezenet\",\"type\":"
         "\"supervised\",\"model\":{\"repository\":\""
         + trt_gan_repo
         + "\"},\"parameters\":{\"input\":{\"connector\":\"image\",\"height\":"
           "360,\"width\":360,\"rgb\":true,\"scale\":0.0039,\"mean\":[0.5, "
           "0.5,0.5],\"std\":[0.5,0.5,0.5]},\"mllib\":{\"maxBatchSize\":1,"
           "\"maxWorkspaceSize\":256,\"gpuid\":0,\"datatype\":\"fp16\"}}}";
  joutstr = japi.jrender(japi.service_create(gan_sname, jstr));
  ASSERT_EQ(created_str, joutstr);

  // chain predict
  std::string jchainstr
      = "{\"chain\":{\"name\":\"chain\",\"calls\":["
        "{\"service\":\""
        + detect_sname
        + "\",\"parameters\":{\"input\":{\"keep_orig\":true},\"output\":{"
          "\"bbox\":true,\"confidence_threshold\":0.2}},\"data\":[\""
        + trt_gan_repo
        + "/horse.jpg\"]},"
          "{\"id\":\"crop\",\"action\":{\"type\":\"crop\",\"parameters\":{"
          "\"fixed_size\":360}}},"
          "{\"service\":\""
        + gan_sname
        + "\",\"parent_id\":\"crop\",\"parameters\":{\"mllib\":{\"extract_"
          "layer\":\"last\"},\"output\":{}}}"
          "]}}";
  joutstr = japi.jrender(japi.service_chain("chain", jchainstr));
  JDoc jd;
  // very long outstr is truncated
  std::cout << "joutstr=" << joutstr.substr(0, 500) << " ... "
            << joutstr.substr(joutstr.size() - 500) << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200, jd["status"]["code"]);
  ASSERT_TRUE(jd["body"]["predictions"].IsArray());
  ASSERT_EQ(jd["body"]["predictions"].Size(), 1);
  ASSERT_TRUE(jd["body"]["predictions"][0]["classes"].IsArray());
  ASSERT_EQ(jd["body"]["predictions"][0]["classes"].Size(), 1);
  ASSERT_TRUE(jd["body"]["predictions"][0]["classes"][0][gan_sname.c_str()]
                  .IsObject());

  auto &gan_pred
      = jd["body"]["predictions"][0]["classes"][0][gan_sname.c_str()];
  ASSERT_TRUE(gan_pred["vals"].IsArray());
  ASSERT_EQ(gan_pred["vals"].Size(), 360 * 360 * 3);

  jstr = "{\"clear\":\"lib\"}";
  joutstr = japi.jrender(japi.service_delete(detect_sname, jstr));
  ASSERT_EQ(ok_str, joutstr);
  joutstr = japi.jrender(japi.service_delete(gan_sname, jstr));
  ASSERT_EQ(ok_str, joutstr);
  ASSERT_TRUE(!fileops::file_exists(trt_detect_repo + "/TRTengine_arch"
                                    + get_trt_archi() + "_bs1"));
  ASSERT_TRUE(!fileops::file_exists(trt_gan_repo + "/TRTengine_arch"
                                    + get_trt_archi() + "_bs1"));
}

#endif
