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
static std::string caffe_faces_detect_repo = "../examples/caffe/faces_512";
static std::string caffe_age_repo = "../examples/caffe/age_real";

static std::string trt_detect_repo = "../examples/trt/yolox_onnx_trt_nowrap/";
static std::string trt_gan_repo
    = "../examples/trt/cyclegan_resnet_attn_onnx_trt";

static std::string test_img_folder = "../examples/all/images";

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
  std::string uri1 = torch_classif_repo + "/cat.jpg";
  std::string uri2 = torch_classif_repo + "/dog.jpg";
  std::string jchainstr
      = "{\"chain\":{\"name\":\"chain\",\"calls\":["
        "{\"service\":\""
        + detect_sname
        + "\",\"parameters\":{\"input\":{\"keep_orig\":true},\"output\":{"
          "\"bbox\":true,\"confidence_threshold\":0.2}},\"data\":[\""
        + uri1 + "\",\"" + uri2
        + "\"]},"
          "{\"id\":\"crop\",\"action\":{\"type\":\"crop\",\"parameters\":{"
          "\"padding_ratio\":0.05}}},{\"service\":\""
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
  ASSERT_EQ(jd["body"]["predictions"].Size(), 2);

  auto &pred1 = jd["body"]["predictions"][0]["uri"].GetString() == uri1
                    ? jd["body"]["predictions"][0]
                    : jd["body"]["predictions"][1];
  auto &pred2 = jd["body"]["predictions"][0]["uri"].GetString() == uri2
                    ? jd["body"]["predictions"][0]
                    : jd["body"]["predictions"][1];
  ASSERT_TRUE(jd["body"]["predictions"][0]["classes"].IsArray());
  ASSERT_EQ(pred1["classes"].Size(), 2);
  ASSERT_EQ(pred2["classes"].Size(), 4);
  ASSERT_TRUE(pred1["classes"][0][classif_sname.c_str()].IsObject());
  ASSERT_TRUE(pred1["classes"][0][classif_sname.c_str()]["classes"].IsArray());

  ASSERT_EQ(
      pred1["classes"][0][classif_sname.c_str()]["classes"][0]["cat"]
          .GetString(),
      std::string("n02120505 grey fox, gray fox, Urocyon cinereoargenteus"));
  ASSERT_EQ(pred2["classes"][0][classif_sname.c_str()]["classes"][0]["cat"]
                .GetString(),
            std::string("n02086079 Pekinese, Pekingese, Peke"));

  // chain predict without detection
  jchainstr
      = "{\"chain\":{\"name\":\"chain\",\"calls\":["
        "{\"service\":\""
        + detect_sname
        + "\",\"parameters\":{\"input\":{\"keep_orig\":true},\"output\":{"
          "\"bbox\":true,\"confidence_threshold\":0.9999}},\"data\":[\""
        + uri1
        + "\"]},"
          "{\"id\":\"crop\",\"action\":{\"type\":\"crop\",\"parameters\":{"
          "\"padding_ratio\":0.05}}},{\"service\":\""
        + classif_sname
        + "\",\"parent_id\":\"crop\",\"parameters\":{\"output\":{\"best\":1}}}"
          "]}}";
  joutstr = japi.jrender(japi.service_chain("chain", jchainstr));
  jd = JDoc();
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200, jd["status"]["code"]);
  ASSERT_TRUE(jd["body"]["predictions"].IsArray());
  ASSERT_EQ(jd["body"]["predictions"].Size(), 1);
  ASSERT_EQ(jd["body"]["predictions"][0]["classes"].Size(), 0);

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

TEST(chain, chain_caffe_faces_classification)
{
  JsonAPI japi;
  std::string detect_sname = "detect";
  std::string jstr
      = "{\"mllib\":\"caffe\",\"description\":\"face detection "
        "model\",\"type\":\"supervised\",\"model\":{\"repository\":\""
        + caffe_faces_detect_repo
        + "\"},\"parameters\":{\"input\":{\"connector\":\"image\",\"width\":"
          "512,\"height\":512},\"mllib\":{\"nclasses\":1,\"best\":-1,"
          "\"gpu\":true,\"gpuid\":0,\"net\":{\"test_batch_size\":1}}}}";
  std::string joutstr = japi.jrender(japi.service_create(detect_sname, jstr));
  ASSERT_EQ(created_str, joutstr);

  std::string age_sname = "age";
  jstr = "{\"mllib\":\"caffe\",\"description\":\"face detection "
         "model\",\"type\":\"supervised\",\"model\":{\"repository\":\""
         + caffe_age_repo
         + "\"},\"parameters\":{\"input\":{\"connector\":\"image\",\"width\":"
           "224,\"height\":224},\"mllib\":{\"nclasses\": "
           "101,\"gpu\":true,\"gpuid\": 0}}}";
  joutstr = japi.jrender(japi.service_create(age_sname, jstr));
  ASSERT_EQ(created_str, joutstr);

  std::string uri1 = test_img_folder + "/face.jpg";
  std::string uri2 = test_img_folder + "/600.jpg";

  std::string jchainstr
      = "{\"chain\": {\"calls\": [{\"parameters\": {\"input\": "
        "{\"keep_orig\": true,\"connector\": \"image\"},\"output\": "
        "{\"confidence_threshold\": 0.5,\"bbox\": true},\"mllib\": {\"net\": "
        "{\"test_batch_size\": 2}}},\"service\":\""
        + detect_sname + "\",\"data\":[\"" + uri1 + "\",\"" + uri2
        + "\"]},{\"id\": "
          "\"face_detection_crop\",\"action\": {\"parameters\": "
          "{\"padding_ratio\": 0.0},\"type\": \"crop\"}},{\"parent_id\": "
          "\"face_detection_crop\",\"parameters\": {\"input\": "
          "{\"keep_orig\":true,\"connector\": \"image\"},\"output\": "
          "{\"best\":1},\"mllib\": "
          "{\"net\": {\"test_batch_size\": 2}}},\"service\": \""
        + age_sname + "\"}]}}";

  joutstr = japi.jrender(japi.service_chain("chain", jchainstr));
  JDoc jd;
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200, jd["status"]["code"]);

  ASSERT_TRUE(jd["body"]["predictions"].IsArray());
  ASSERT_EQ(jd["body"]["predictions"].Size(), 2);

  auto &pred1 = jd["body"]["predictions"][0]["uri"].GetString() == uri1
                    ? jd["body"]["predictions"][0]
                    : jd["body"]["predictions"][1];
  auto &pred2 = jd["body"]["predictions"][0]["uri"].GetString() == uri2
                    ? jd["body"]["predictions"][0]
                    : jd["body"]["predictions"][1];
  ASSERT_EQ(pred1["classes"].Size(), 1);
  ASSERT_EQ(pred2["classes"].Size(), 0);
  ASSERT_EQ(pred1["classes"][0]["age"]["classes"][0]["cat"].GetString(),
            std::string("52"));
}

TEST(chain, chain_caffe_detect_draw_bboxes)
{
  JsonAPI japi;
  std::string detect_sname = "detect";
  std::string jstr
      = "{\"mllib\":\"caffe\",\"description\":\"face detection "
        "model\",\"type\":\"supervised\",\"model\":{\"repository\":\""
        + caffe_faces_detect_repo
        + "\"},\"parameters\":{\"input\":{\"connector\":\"image\",\"width\":"
          "512,\"height\":512},\"mllib\":{\"nclasses\":1,\"best\":-1,"
          "\"gpu\":true,\"gpuid\":0,\"net\":{\"test_batch_size\":1}}}}";
  std::string joutstr = japi.jrender(japi.service_create(detect_sname, jstr));
  ASSERT_EQ(created_str, joutstr);

  std::string uri1 = test_img_folder + "/face.jpg";

  std::string jchainstr
      = "{\"chain\": {\"calls\": [{\"parameters\": {\"input\": "
        "{\"keep_orig\": true,\"connector\": \"image\"},\"output\": "
        "{\"confidence_threshold\": 0.5,\"bbox\": true},\"mllib\": {\"net\": "
        "{\"test_batch_size\": 2}}},\"service\":\""
        + detect_sname + "\",\"data\":[\"" + uri1
        + "\"]},{\"id\":\"face_detection_bbox\",\"action\":{\"parameters\": "
          "{\"padding_ratio\":0.0,\"output_images\":true,\"write_prob\":true}"
          ",\"type\":\"draw_bbox\"}}]}}";
  joutstr = japi.jrender(japi.service_chain("chain", jchainstr));
  JDoc jd;
  // very long outstr is truncated
  std::cout << "joutstr=" << joutstr.substr(0, 500)
            << (joutstr.size() > 500
                    ? " ... " + joutstr.substr(joutstr.size() - 500)
                    : "")
            << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200, jd["status"]["code"]);

  ASSERT_TRUE(jd["body"]["predictions"].IsArray());
  ASSERT_TRUE(jd["body"]["predictions"][0]["classes"].IsArray());
  ASSERT_TRUE(jd["body"]["predictions"][0]["classes"].Size() == 1);
  auto &bbox_images = jd["body"]["predictions"][0]["face_detection_bbox"];
  ASSERT_TRUE(bbox_images["vals"].IsArray());
  ASSERT_EQ(bbox_images["vals"].Size(), 3 * 1000 * 562);
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
      = "{\"mllib\":\"tensorrt\",\"description\":\"yolox\","
        "\"type\":\"supervised\",\"model\":{\"repository\":\""
        + trt_detect_repo
        + "\"},\"parameters\":{\"input\":{\"connector\":"
          "\"image\",\"height\":640,\"width\":640},\"mllib\":{"
          "\"maxWorkspaceSize\":256,\"gpuid\":0,"
          "\"template\":\"yolox\",\"nclasses\":81,\"datatype\":\"fp16\"}}}";
  std::string joutstr = japi.jrender(japi.service_create(detect_sname, jstr));
  ASSERT_EQ(created_str, joutstr);

  std::string gan_sname = "gan";
  jstr = "{\"mllib\":\"tensorrt\",\"description\":\"gan\",\"type\":"
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
          "\"bbox\":true,\"best_bbox\":1}},\"data\":[\""
        + trt_gan_repo
        + "/horse_1024.jpg\"]},"
          "{\"id\":\"crop\",\"action\":{\"type\":\"crop\",\"parameters\":{"
          "\"fixed_width\":360,\"fixed_height\":360}}},{\"service\":\""
        + gan_sname
        + "\",\"parent_id\":\"crop\",\"parameters\":{\"mllib\":{\"extract_"
          "layer\":\"last\"},\"output\":{}}}"
          "]}}";
  joutstr = japi.jrender(japi.service_chain("chain", jchainstr));
  JDoc jd;
  // very long outstr is truncated
  std::cout << "joutstr=" << joutstr.substr(0, 500)
            << (joutstr.size() > 500
                    ? " ... " + joutstr.substr(joutstr.size() - 500)
                    : "")
            << std::endl;
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

  // image recompose
  // XXX: keep_orig = false doesn't work on CUDA images!
  jchainstr
      = "{\"chain\":{\"name\":\"chain\",\"calls\":[{\"service\":\""
        + detect_sname
        + "\",\"parameters\":{\"input\":{\"keep_orig\":true,"
#ifdef USE_CUDA_CV
          "\"cuda\":true"
#else
          "\"cuda\":false"
#endif
          "},\"output\":{\"bbox\":true,\"best_bbox\":2}},\"data\":[\""
        + trt_gan_repo
        + "/horse_1024.jpg\"]},"
          "{\"id\":\"crop\",\"action\":{\"type\":\"crop\",\"parameters\":{"
          "\"fixed_width\":360,\"fixed_height\":360}}},{\"service\":\""
        + gan_sname
        + "\",\"parent_id\":\"crop\",\"parameters\":{\"input\":{"
#ifdef USE_CUDA_CV
          "\"cuda\":true"
#else
          "\"cuda\":false"
#endif
          "},\"mllib\":{\"extract_layer\":\"last\"},\"output\":{\"image\":"
          "true}}},{\"id\":\"recompose\",\"action\":{\"type\":\"recompose\","
          "\"parameters\":{\"save_img\":true,\"save_path\":\".\"}}}]}}";
  joutstr = japi.jrender(japi.service_chain("chain", jchainstr));
  std::cout << "joutstr=" << joutstr.substr(0, 500)
            << (joutstr.size() > 500
                    ? " ... " + joutstr.substr(joutstr.size() - 500)
                    : "")
            << std::endl;
  jd = JDoc();
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200, jd["status"]["code"]);
  ASSERT_TRUE(jd["body"]["predictions"].IsArray());
  ASSERT_TRUE(jd["body"]["predictions"][0]["classes"].IsArray());
  ASSERT_EQ(2, jd["body"]["predictions"][0]["classes"].Size());
  ASSERT_TRUE(jd["body"]["predictions"][0]["recompose"].IsObject());

  auto &recompose_pred = jd["body"]["predictions"][0]["recompose"];
  ASSERT_TRUE(recompose_pred["images"].IsArray());

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

// Test internal call without json
TEST(chain, chain_trt_dto)
{
  JsonAPI japi;
  std::string detect_sname = "detect";
  std::string jstr
      = "{\"mllib\":\"tensorrt\",\"description\":\"yolox\","
        "\"type\":\"supervised\",\"model\":{\"repository\":\""
        + trt_detect_repo
        + "\"},\"parameters\":{\"input\":{\"connector\":"
          "\"image\",\"height\":640,\"width\":640},\"mllib\":{"
          "\"maxWorkspaceSize\":256,\"gpuid\":0,"
          "\"template\":\"yolox\",\"nclasses\":81,\"datatype\":\"fp16\"}}}";
  std::string joutstr = japi.jrender(japi.service_create(detect_sname, jstr));
  ASSERT_EQ(created_str, joutstr);

  std::string gan_sname = "gan";
  jstr = "{\"mllib\":\"tensorrt\",\"description\":\"gan\",\"type\":"
         "\"supervised\",\"model\":{\"repository\":\""
         + trt_gan_repo
         + "\"},\"parameters\":{\"input\":{\"connector\":\"image\",\"height\":"
           "360,\"width\":360,\"rgb\":true,\"scale\":0.0039,\"mean\":[0.5, "
           "0.5,0.5],\"std\":[0.5,0.5,0.5]},\"mllib\":{\"maxBatchSize\":1,"
           "\"maxWorkspaceSize\":256,\"gpuid\":0,\"datatype\":\"fp16\"}}}";
  joutstr = japi.jrender(japi.service_create(gan_sname, jstr));
  ASSERT_EQ(created_str, joutstr);

  // chain call with no predictions
  std::string uri1 = trt_gan_repo + "/horse_1024.jpg";
  auto input_dto = oatpp::Object<DTO::ServiceChain>::createShared();
  input_dto->chain = oatpp::Object<DTO::Chain>::createShared();

  auto call1 = oatpp::Object<DTO::ChainCall>::createShared();
  call1->service = detect_sname;
  call1->parameters->input->keep_orig = true;
  call1->parameters->output->bbox = true;
  call1->parameters->output->confidence_threshold = 0.9999;
  call1->data->push_back(uri1);
  input_dto->chain->calls->push_back(call1);

  auto call2 = oatpp::Object<DTO::ChainCall>::createShared();
  call2->id = "crop";
  call2->action = oatpp::Object<DTO::ChainAction>::createShared();
  call2->action->type = "crop";
  call2->action->parameters->padding_ratio = 0.05;
  input_dto->chain->calls->push_back(call2);

  auto call3 = oatpp::Object<DTO::ChainCall>::createShared();
  call3->service = gan_sname;
  call3->parent_id = "crop";
  call3->parameters->mllib->extract_layer = "last";
  call3->parameters->output->image = true;
  input_dto->chain->calls->push_back(call3);

  auto chain_out = japi.chain(input_dto, "chain");
  JDoc jdoc;
  oatpp_utils::dtoToJDoc(chain_out, jdoc);
  std::cout << dd_utils::jrender(jdoc) << std::endl;

  ASSERT_EQ(chain_out->predictions->size(), 1);
  // PredictClass -> only one model, so the Predict DTO is returned without
  // modifications
  ASSERT_EQ((*chain_out->predictions->at(0))["classes"]
                .retrieve<oatpp::Vector<oatpp::Object<DTO::PredictClass>>>()
                ->size(),
            0);

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
