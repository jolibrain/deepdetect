/**
 * DeepDetect
 * Copyright (c) 2019-2021 Jolibrain
 * Author: Louis Jean <louis.jean@jolibrain.com>
 *         Guillaume Infantes <guillaume.infantes@jolibrain.com>
 *         Emmanuel Benazera <emmanuel.benazera@jolibrain.com>
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
#include "txtinputfileconn.h"
#include <gtest/gtest.h>
#include <stdio.h>
#include <iostream>
#include <numeric>
#include "backends/torch/native/templates/nbeats.h"
#include <torch/torch.h>
#include <rapidjson/istreamwrapper.h>

using namespace dd;

static std::string ok_str = "{\"status\":{\"code\":200,\"msg\":\"OK\"}}";
static std::string created_str
    = "{\"status\":{\"code\":201,\"msg\":\"Created\"}}";
static std::string bad_param_str
    = "{\"status\":{\"code\":400,\"msg\":\"BadRequest\"}}";
static std::string not_found_str
    = "{\"status\":{\"code\":404,\"msg\":\"NotFound\"}}";

static std::string incept_repo = "../examples/torch/resnet50_torch/";
static std::string detect_repo = "../examples/torch/fasterrcnn_torch/";
static std::string seg_repo = "../examples/torch/deeplabv3_torch/";
static std::string detect_train_repo_fasterrcnn
    = "../examples/torch/fasterrcnn_train_torch";
static std::string detect_train_repo_yolox
    = "../examples/torch/yolox_train_torch";
static std::string resnet50_train_repo
    = "../examples/torch/resnet50_training_torch_small/";
static std::string resnet50_train_data
    = "../examples/torch/resnet50_training_torch_small/train/";
static std::string resnet50_train_data_reg
    = "../examples/torch/resnet50_training_torch_small/"
      "list_all_shuf_rel.txt";
static std::string resnet50_train_data_classif
    = "../examples/torch/resnet50_training_torch_small/"
      "list_all_shuf_rel_classif.txt";
static std::string resnet50_test_data
    = "../examples/torch/resnet50_training_torch_small/test/";
static std::string resnet50_test_cats_data
    = "../examples/torch/resnet50_training_torch_small/test_cats/";
static std::string resnet50_test_image
    = "../examples/torch/resnet50_training_torch_small/train/cats/"
      "cat.10097.jpg";
static std::string deeplabv3_train_repo
    = "../examples/torch/deeplabv3_training_torch/";
static std::string deeplabv3_train_data
    = "../examples/torch/deeplabv3_training_torch/CamVid_square/train.txt";
static std::string deeplabv3_test_data
    = "../examples/torch/deeplabv3_training_torch/CamVid_square/test50.txt";
static std::string deeplabv3_test_image
    = "../examples/torch/deeplabv3_training_torch/CamVid_square/test/"
      "Seq05VD_f00330.png";
static std::string segformer_train_repo
    = "../examples/torch/segformer_training_torch/";

static std::string resnet50_native_weights
    = "../examples/torch/resnet50_native_torch/resnet50.npt";

static std::string vit_train_repo = "../examples/torch/vit_training_torch/";

static std::string bert_classif_repo
    = "../examples/torch/bert_inference_torch/";
static std::string bert_train_repo
    = "../examples/torch/bert_training_torch_140_transformers_251/";
static std::string bert_train_data
    = "../examples/torch/bert_training_torch_140_transformers_251/data/";

static std::string fasterrcnn_train_data
    = "../examples/torch/fasterrcnn_train_torch/train.txt";
static std::string fasterrcnn_test_data
    = "../examples/torch/fasterrcnn_train_torch/test.txt";

static std::string sinus = "../examples/all/sinus/";

static std::string iterations_nbeats_cpu = "100";
static std::string iterations_nbeats_gpu = "1000";
static std::string iterations_ttransformer_cpu = "100";
static std::string iterations_ttransformer_gpu = "1000";

static std::string iterations_resnet50 = "200";
static std::string iterations_vit = "200";
static std::string iterations_detection = "200";
static std::string iterations_deeplabv3 = "200";

static int torch_seed = 1235;
static std::string torch_lr = "1e-5";

TEST(torchapi, service_predict)
{
  // create service
  JsonAPI japi;
  std::string sname = "imgserv";
  std::string jstr
      = "{\"mllib\":\"torch\",\"description\":\"resnet-50\",\"type\":"
        "\"supervised\",\"model\":{\"repository\":\""
        + incept_repo
        + "\"},\"parameters\":{\"input\":{\"connector\":\"image\",\"height\":"
          "224,\"width\":224,\"rgb\":true,\"scale\":0.0039},\"mllib\":{"
          "\"nclasses\":1000}}}";
  std::string joutstr = japi.jrender(japi.service_create(sname, jstr));
  ASSERT_EQ(created_str, joutstr);

  // predict
  std::string jpredictstr
      = "{\"service\":\"imgserv\",\"parameters\":{\"input\":{\"height\":224,"
        "\"width\":224},\"output\":{\"best\":1}},\"data\":[\""
        + incept_repo + "cat.jpg\"]}";
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  JDoc jd;
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200, jd["status"]["code"]);
  ASSERT_TRUE(jd["body"]["predictions"].IsArray());
  ASSERT_EQ(jd["body"]["predictions"][0]["classes"].Size(), 1);
  std::string cl1
      = jd["body"]["predictions"][0]["classes"][0]["cat"].GetString();
  ASSERT_TRUE(cl1 == "n02123045 tabby, tabby cat");
  ASSERT_TRUE(jd["body"]["predictions"][0]["classes"][0]["prob"].GetDouble()
              > 0.3);

  // confidence threshold
  jpredictstr
      = "{\"service\":\"imgserv\",\"parameters\":{\"input\":{\"height\":224,"
        "\"width\":224},\"output\":{\"confidence_threshold\":0.01}},\"data\":"
        "[\""
        + incept_repo + "cat.jpg\"]}";

  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  jd = JDoc();
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200, jd["status"]["code"]);
  ASSERT_TRUE(jd["body"]["predictions"].IsArray());
  ASSERT_EQ(jd["body"]["predictions"][0]["classes"].Size(), 8);

  // batch size == 2
  jpredictstr
      = "{\"service\":\"imgserv\",\"parameters\":{\"input\":{\"height\":224,"
        "\"width\":224},\"mllib\":{\"net\":{\"test_batch_size\":2}},"
        "\"output\":{\"best\":1}},\"data\":[\""
        + incept_repo + "cat.jpg\",\"" + incept_repo + "dog.jpg\"]}";
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  jd = JDoc();
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200, jd["status"]["code"]);
  ASSERT_TRUE(jd["body"]["predictions"].IsArray());
  ASSERT_EQ(jd["body"]["predictions"].Size(), 2);
  cl1 = jd["body"]["predictions"][0]["classes"][0]["cat"].GetString();
  std::string cl2
      = jd["body"]["predictions"][1]["classes"][0]["cat"].GetString();
  std::string cl_cat = jd["body"]["predictions"][0]["uri"].GetString()
                               == incept_repo + "cat.jpg"
                           ? cl1
                           : cl2;
  std::string cl_dog = jd["body"]["predictions"][1]["uri"].GetString()
                               == incept_repo + "dog.jpg"
                           ? cl2
                           : cl1;
  ASSERT_EQ(cl_cat, "n02123045 tabby, tabby cat");
  ASSERT_EQ(cl_dog, "n02096051 Airedale, Airedale terrier");
}

TEST(torchapi, service_predict_native_bw)
{
  // Predict greyscale image with native model should work
  std::string native_resnet_repo = "native_resnet";

  JsonAPI japi;
  std::string sname = "imgserv";
  std::string jstr
      = "{\"mllib\":\"torch\",\"description\":\"image\",\"type\":"
        "\"supervised\",\"model\":{\"repository\":\""
        + native_resnet_repo
        + "\",\"create_repository\":true},\"parameters\":{\"input\":{"
          "\"connector\":\"image\",\"bw\":true,"
          "\"width\":224,\"height\":224,\"db\":true},"
          "\"mllib\":{\"nclasses\":2,\"gpu\":true,\"template\":\"resnet50\"}}"
          "}";
  std::string joutstr = japi.jrender(japi.service_create(sname, jstr));

  std::string jpredictstr
      = "{\"service\":\"imgserv\",\"parameters\":{\"input\":{\"height\":224,"
        "\"width\":224},\"output\":{\"best\":1}},\"data\":[\""
        + incept_repo + "cat.jpg\"]}";
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  JDoc jd;
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200, jd["status"]["code"]);

  // clear directory
  jstr = "{\"clear\":\"full\"}";
  joutstr = japi.jrender(japi.service_delete(sname, jstr));
  ASSERT_EQ(ok_str, joutstr);
  fileops::remove_dir(native_resnet_repo);
}

#if !defined(CPU_ONLY)
TEST(torchapi, service_predict_fp16)
{
  // create service
  JsonAPI japi;
  std::string sname = "imgserv";
  std::string jstr
      = "{\"mllib\":\"torch\",\"description\":\"resnet-50\",\"type\":"
        "\"supervised\",\"model\":{\"repository\":\""
        + incept_repo
        + "\"},\"parameters\":{\"input\":{\"connector\":\"image\",\"height\":"
          "224,\"width\":224,\"rgb\":true,\"scale\":0.0039},\"mllib\":{"
          "\"gpu\":true}}}";
  std::string joutstr = japi.jrender(japi.service_create(sname, jstr));
  ASSERT_EQ(created_str, joutstr);

  // predict fp32
  std::string jpredictstr
      = "{\"service\":\"imgserv\",\"parameters\":{\"input\":{\"height\":224,"
        "\"width\":224},\"output\":{\"best\":1}},\"data\":[\""
        + incept_repo + "cat.jpg\"]}";
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  JDoc jd;
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200, jd["status"]["code"]);
  ASSERT_TRUE(jd["body"]["predictions"].IsArray());
  std::string cl1
      = jd["body"]["predictions"][0]["classes"][0]["cat"].GetString();
  ASSERT_TRUE(cl1 == "n02123045 tabby, tabby cat");
  double v32 = jd["body"]["predictions"][0]["classes"][0]["prob"].GetDouble();
  ASSERT_TRUE(v32 > 0.3);

  // predict fp16
  jpredictstr
      = "{\"service\":\"imgserv\",\"parameters\":{\"input\":{\"height\":224,"
        "\"width\":224},\"output\":{\"best\":1},\"mllib\":{\"datatype\":"
        "\"fp16\"}},\"data\":[\""
        + incept_repo + "cat.jpg\"]}";
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200, jd["status"]["code"]);
  ASSERT_TRUE(jd["body"]["predictions"].IsArray());
  std::string cl2
      = jd["body"]["predictions"][0]["classes"][0]["cat"].GetString();
  ASSERT_TRUE(cl2 == "n02123045 tabby, tabby cat");
  double v16 = jd["body"]["predictions"][0]["classes"][0]["prob"].GetDouble();
  ASSERT_TRUE(v16 > 0.3);
  ASSERT_TRUE(fabs(v32 - v16) < 0.01);
}
#endif

TEST(torchapi, service_predict_object_detection)
{
  JsonAPI japi;
  std::string sname = "detectserv";
  std::string jstr
      = "{\"mllib\":\"torch\",\"description\":\"fasterrcnn\",\"type\":"
        "\"supervised\",\"model\":{\"repository\":\""
        + detect_repo
        + "\"},\"parameters\":{\"input\":{\"connector\":\"image\",\"height\":"
          "224,\"width\":224,\"rgb\":true,\"scale\":0.0039},\"mllib\":{"
          "\"template\":\"fasterrcnn\"}}}";

  std::string joutstr = japi.jrender(japi.service_create(sname, jstr));
  ASSERT_EQ(created_str, joutstr);
  std::string jpredictstr = "{\"service\":\"detectserv\",\"parameters\":{"
                            "\"input\":{\"height\":224,"
                            "\"width\":224},\"output\":{\"bbox\":true, "
                            "\"confidence_threshold\":0.8}},\"data\":[\""
                            + detect_repo + "cat.jpg\"]}";

  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  JDoc jd;
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200, jd["status"]["code"]);
  ASSERT_TRUE(jd["body"]["predictions"].IsArray());

  auto &preds = jd["body"]["predictions"][0]["classes"];
  std::string cl1 = preds[0]["cat"].GetString();
  ASSERT_TRUE(cl1 == "cat");
  ASSERT_TRUE(preds[0]["prob"].GetDouble() > 0.9);
  auto &bbox = preds[0]["bbox"];
  // cat is approximately in bottom left corner of the image.
  ASSERT_TRUE(bbox["xmin"].GetDouble() < 100 && bbox["xmax"].GetDouble() > 300
              && bbox["ymin"].GetDouble() < 100
              && bbox["ymax"].GetDouble() > 300);
  // Check confidence threshold
  ASSERT_TRUE(preds[preds.Size() - 1]["prob"].GetDouble() >= 0.8);

  // best
  jpredictstr = "{\"service\":\"detectserv\",\"parameters\":{"
                "\"input\":{\"height\":224,"
                "\"width\":224},\"output\":{\"bbox\":true, "
                "\"best_bbox\":3}},\"data\":[\""
                + detect_repo + "cat.jpg\"]}";
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());

  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200, jd["status"]["code"]);
  ASSERT_TRUE(jd["body"]["predictions"].IsArray());

  auto &preds_best = jd["body"]["predictions"][0]["classes"];
  ASSERT_EQ(preds_best.Size(), 3);
}

TEST(torchapi, service_predict_segmentation)
{
  JsonAPI japi;
  std::string sname = "segserv";
  std::string jstr
      = "{\"mllib\":\"torch\",\"description\":\"deeplabv3\",\"type\":"
        "\"supervised\",\"model\":{\"repository\":\""
        + seg_repo
        + "\"},\"parameters\":{\"input\":{\"connector\":\"image\",\"height\":"
          "224,\"width\":224,\"rgb\":true,\"scale\":0.0039},\"mllib\":{"
          "\"segmentation\":true,\"nclasses\":21}}}";

  std::string joutstr = japi.jrender(japi.service_create(sname, jstr));
  ASSERT_EQ(created_str, joutstr);
  std::string jpredictstr
      = "{\"service\":\"segserv\",\"parameters\":{"
        "\"input\":{\"height\":224,"
        "\"width\":224},\"output\":{\"segmentation\":true, "
        "\"confidences\":[\"best\"]}},\"data\":[\""
        + seg_repo + "cat.jpg\"]}";

  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  JDoc jd;
  // std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200, jd["status"]["code"]);
  ASSERT_TRUE(jd["body"]["predictions"].IsArray());

  auto &preds = jd["body"]["predictions"][0]["vals"];
  auto &confs = jd["body"]["predictions"][0]["confidences"]["best"];
  ASSERT_TRUE(preds.IsArray());
  ASSERT_TRUE(confs.IsArray());
  ASSERT_TRUE(preds.Size() == 500 * 374);
  ASSERT_TRUE(confs.Size() == 500 * 374);
}

TEST(torchapi, service_predict_txt_classification)
{
  // create service
  JsonAPI japi;
  std::string sname = "txtserv";
  std::string jstr = "{\"mllib\":\"torch\",\"description\":\"bert\",\"type\":"
                     "\"supervised\",\"model\":{\"repository\":\""
                     + bert_classif_repo
                     + "\"},\"parameters\":{\"input\":{\"connector\":\"txt\","
                       "\"ordered_words\":true,"
                       "\"wordpiece_tokens\":true,\"punctuation_tokens\":true,"
                       "\"sequence\":512}}}";
  std::string joutstr = japi.jrender(japi.service_create(sname, jstr));
  ASSERT_EQ(created_str, joutstr);

  // predict
  std::string jpredictstr
      = "{\"service\":\"txtserv\",\"parameters\":{\"output\":{\"best\":1}},"
        "\"data\":["
        "\"Get the official USA poly ringtone or colour flag on your mobile "
        "for tonights game! Text TONE or FLAG to 84199. Optout txt ENG STOP "
        "Box39822 W111WX £1.50\"]}";
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  JDoc jd;
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200, jd["status"]["code"]);
  ASSERT_TRUE(jd["body"]["predictions"].IsArray());
  std::string cl1
      = jd["body"]["predictions"][0]["classes"][0]["cat"].GetString();
  ASSERT_TRUE(cl1 == "spam");
  ASSERT_TRUE(jd["body"]["predictions"][0]["classes"][0]["prob"].GetDouble()
              > 0.7);
}

TEST(inputconn, txt_tokenize_ordered_words)
{
  std::string str = "everything runs fine, right?";
  TxtInputFileConn tifc;
  tifc._ordered_words = true;
  tifc._wordpiece_tokens = true;
  tifc._punctuation_tokens = true;

  tifc._vocab["every"] = Word();
  tifc._vocab["##ing"] = Word();
  tifc._vocab["##thing"] = Word();
  tifc._vocab["fine"] = Word();
  tifc._vocab[","] = Word();
  tifc._vocab["?"] = Word();
  tifc._vocab["right"] = Word();

  tifc.parse_content(str, 1);
  TxtOrderedWordsEntry &towe
      = *dynamic_cast<TxtOrderedWordsEntry *>(tifc._txt.at(0));
  std::vector<std::string> tokens{ "every", "##thing", "[UNK]", "fine",
                                   ",",     "right",   "?" };
  ASSERT_EQ(tokens, towe._v);
}

TEST(torchapi, load_weights_native_model)
{
  APIData template_params;
  template_params.add("nclasses", 2);

  auto logger = DD_SPDLOG_LOGGER("test");
  ImgTorchInputFileConn inputc;
  TorchModel mlmodel;
  mlmodel._repo = ".";

  // =====
  // Fail if no weights are corresponding

  TorchModule module;
  module._logger = logger;
  module.create_native_template<ImgTorchInputFileConn>(
      "resnet50", template_params, inputc, mlmodel, torch::Device("cpu"));

  mlmodel._native = "bad_weights.npt";

  torch::nn::Sequential bad_weights{ torch::nn::Linear(3, 2) };
  torch::save(bad_weights, mlmodel._native);

  ASSERT_THROW(module.load(mlmodel), MLLibBadParamException);

  fileops::remove_file(".", mlmodel._native);

  // =====
  // Succeed to load for finetuning (but exclude the two last weights)
  // Check if the output tensor is of the right dimensions
  mlmodel._native = resnet50_native_weights;

  // I don't have the weights to finetune, need to make a new archive
  module._finetuning = true;
  module.load(mlmodel);
  auto jit_weights = torch::jit::load(resnet50_native_weights);

  std::string test_param_name = "wrapped.layer1.0.conv1.weight";
  torch::Tensor target_value;
  bool param_found = false;

  for (const auto &item : jit_weights.named_parameters())
    {
      if (item.name == test_param_name)
        {
          target_value = item.value;
          param_found = true;
          break;
        }
    }
  ASSERT_TRUE(param_found) << "Parameter not found in "
                                  + resnet50_native_weights;

  // <!> this line segfaults if test_param_name is not in named_parameters
  torch::Tensor tested_value
      = *module._native->named_parameters().find(test_param_name);

  ASSERT_TRUE(torch::allclose(target_value, tested_value));
  auto output_size = module.forward({ torch::zeros({ 1, 3, 224, 224 }) })
                         .toTensor()
                         .sizes();
  std::vector<long int> target_sizes = { 1, 2 };
  ASSERT_EQ(output_size, target_sizes);

  // =====
  // Check if we can reload a checkpoint from native module without any
  // problem.

  module._finetuning = false;
  test_param_name = "wrapped.fc.bias";
  // <!> segfault if test_param_name is not in named_parameters
  tested_value = *module._native->named_parameters().find(test_param_name);
  torch::Tensor before_val = tested_value.clone();

  module.save_checkpoint(mlmodel, "0");
  mlmodel._native = "checkpoint-0.npt";
  module.load(mlmodel);

  // <!> segfault if test_param_name is not in named_parameters
  tested_value = *module._native->named_parameters().find(test_param_name);
  torch::Tensor after_val = tested_value.clone();
  ASSERT_TRUE(torch::allclose(before_val, after_val));

  fileops::remove_file(".", mlmodel._native);
}

TEST(torchapi, compute_bbox_stats)
{
  TorchModel torchmodel;
  TorchLib<ImgTorchInputFileConn, SupervisedOutput, TorchModel> torchlib(
      torchmodel);
  torchlib._nclasses = 2;
  // img dims are e.g. 1000, 1000
  float targ_bboxes_data[] = {
    10,  10,  100, 100, // matching
    500, 500, 600, 600, // 2 preds for 1 targets
    10,  500, 100, 600, // not matching
    900, 900, 950, 950, // overlapping but iou < 0.5
  };
  at::Tensor targ_bboxes = torch::from_blob(targ_bboxes_data, { 4, 4 });

  // TODO test with multiple labels
  int64_t targ_labels_data[] = { 1, 1, 1, 1 };
  at::Tensor targ_labels = torch::from_blob(targ_labels_data, 4, torch::kLong);

  float bboxes_data[] = {
    11,  11,  101, 101, // matching
    900, 10,  950, 100, // false positive
    510, 510, 610, 610, // 2 preds for 1 targets
    490, 490, 590, 590, // --
    940, 940, 990, 990, // overlapping but iou < 0.5 -> false positive
  };
  at::Tensor bboxes_tensor = torch::from_blob(bboxes_data, { 5, 4 });

  int64_t labels_data[] = { 1, 1, 1, 1, 1 };
  at::Tensor labels_tensor = torch::from_blob(labels_data, 5, torch::kLong);

  float score_data[] = { 0.9, 0.8, 0.7, 0.6, 0.5 };
  at::Tensor score_tensor = torch::from_blob(score_data, 5);

  auto vbad = torchlib.get_bbox_stats(targ_bboxes, targ_labels, bboxes_tensor,
                                      labels_tensor, score_tensor);

  auto lbad = vbad.at(0);
  auto tp_i = lbad.get("tp_i").get<std::vector<int>>();
  auto tp_d = lbad.get("tp_d").get<std::vector<double>>();
  auto fp_i = lbad.get("fp_i").get<std::vector<int>>();
  auto fp_d = lbad.get("fp_d").get<std::vector<double>>();
  ASSERT_EQ(std::accumulate(tp_i.begin(), tp_i.end(), 0), 2);
  ASSERT_EQ(std::accumulate(fp_i.begin(), fp_i.end(), 0), 3);
  ASSERT_TRUE(tp_i[2]);
  ASSERT_FALSE(tp_i[3]);
  for (int i = 0; i < 5; ++i)
    {
      ASSERT_TRUE(tp_i[i] != fp_i[i]) << std::to_string(i);
      ASSERT_NEAR(tp_d[i], score_data[i],
                  std::numeric_limits<float>::epsilon())
          << std::to_string(i);
      ASSERT_NEAR(fp_d[i], score_data[i],
                  std::numeric_limits<float>::epsilon())
          << std::to_string(i);
    }
  ASSERT_EQ(lbad.get("num_pos").get<int>(), 4);
  ASSERT_EQ(lbad.get("label").get<int>(), 1);

  // Get MAP
  APIData ad_res;
  ad_res.add("clnames", std::vector<std::string>{ "0", "1" });
  ad_res.add("nclasses", static_cast<int>(torchlib._nclasses));
  ad_res.add("bbox", true);
  ad_res.add("pos_count", 1);
  APIData ad_bbox;
  ad_bbox.add("0", vbad);
  ad_res.add("0", ad_bbox);
  ad_res.add("batch_size", 1);
  APIData ad_out;
  ad_out.add("measure", std::vector<std::string>{ "map" });
  APIData out;
  SupervisedOutput::measure(ad_res, ad_out, out, 0, "test");
  ASSERT_NEAR(out.getobj("measure").get("map").get<double>(), 5. / 12.,
              std::numeric_limits<float>::epsilon());
}

// Training tests

#if !defined(CPU_ONLY)

TEST(torchapi, service_train_images_split)
{
  setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8", true);
  torch::manual_seed(torch_seed);
  at::globalContext().setDeterministicCuDNN(true);

  // Create service
  JsonAPI japi;
  std::string sname = "imgserv";
  std::string jstr
      = "{\"mllib\":\"torch\",\"description\":\"image\",\"type\":"
        "\"supervised\",\"model\":{\"repository\":\""
        + resnet50_train_repo
        + "\"},\"parameters\":{\"input\":{\"connector\":\"image\","
          "\"width\":224,\"height\":224,\"db\":true},\"mllib\":{\"nclasses\":"
          "2,\"finetuning\":true,\"gpu\":true}}}";
  std::string joutstr = japi.jrender(japi.service_create(sname, jstr));
  ASSERT_EQ(created_str, joutstr);

  // Train
  std::string jtrainstr
      = "{\"service\":\"imgserv\",\"async\":false,\"parameters\":{"
        "\"mllib\":{\"solver\":{\"iterations\":"
        + iterations_resnet50 + ",\"base_lr\":" + torch_lr
        + ",\"iter_size\":4,\"solver_type\":\"ADAM\",\"test_"
          "interval\":200},\"net\":{\"batch_size\":4},\"nclasses\":2,"
          "\"resume\":false},"
          "\"input\":{\"seed\":12345,\"db\":true,\"shuffle\":true,\"test_"
          "split\":0.1},"
          "\"output\":{\"measure\":[\"f1\",\"acc\"]}},\"data\":[\""
        + resnet50_train_data + "\"]}";
  joutstr = japi.jrender(japi.service_train(jtrainstr));
  JDoc jd;
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(201, jd["status"]["code"]);

  ASSERT_TRUE(jd["body"]["measure"]["iteration"] == 200) << "iterations";
  ASSERT_TRUE(jd["body"]["measure"]["train_loss"].GetDouble() <= 3.0)
      << "loss";

  ASSERT_TRUE(fileops::file_exists(resnet50_train_repo + "config.json"));
  ASSERT_TRUE(fileops::file_exists(resnet50_train_repo + "model.json"));

  std::unordered_set<std::string> lfiles;
  fileops::list_directory(resnet50_train_repo, true, false, false, lfiles);
  for (std::string ff : lfiles)
    {
      if (ff.find("checkpoint") != std::string::npos
          || ff.find("solver") != std::string::npos)
        remove(ff.c_str());
    }
  ASSERT_TRUE(!fileops::file_exists(resnet50_train_repo + "checkpoint-"
                                    + iterations_resnet50 + ".ptw"));
  ASSERT_TRUE(!fileops::file_exists(resnet50_train_repo + "checkpoint-"
                                    + iterations_resnet50 + ".pt"));
  fileops::clear_directory(resnet50_train_repo + "train.lmdb");
  fileops::clear_directory(resnet50_train_repo + "test_0.lmdb");
  fileops::remove_dir(resnet50_train_repo + "train.lmdb");
  fileops::remove_dir(resnet50_train_repo + "test_0.lmdb");
}

TEST(torchapi, service_train_images)
{
  setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8", true);
  torch::manual_seed(torch_seed);
  at::globalContext().setDeterministicCuDNN(true);

  // Create service
  JsonAPI japi;
  std::string sname = "imgserv";
  std::string jstr
      = "{\"mllib\":\"torch\",\"description\":\"image\",\"type\":"
        "\"supervised\",\"model\":{\"repository\":\""
        + resnet50_train_repo
        + "\"},\"parameters\":{\"input\":{\"connector\":\"image\","
          "\"width\":256,\"height\":256,\"db\":true},\"mllib\":{\"nclasses\":"
          "2,\"finetuning\":true,\"gpu\":true}}}";
  std::string joutstr = japi.jrender(japi.service_create(sname, jstr));
  ASSERT_EQ(created_str, joutstr);

  // Train
  std::string jtrainstr
      = "{\"service\":\"imgserv\",\"async\":false,\"parameters\":{"
        "\"mllib\":{\"solver\":{\"iterations\":"
        + iterations_resnet50 + ",\"base_lr\":" + torch_lr
        + ",\"iter_size\":4,\"solver_type\":\"ADAM\",\"test_"
          "interval\":200},\"net\":{\"batch_size\":4},"
          "\"resume\":false,\"mirror\":true,\"rotate\":true,\"crop_size\":224,"
          "\"cutout\":0.5,\"geometry\":{\"prob\":0.1,\"persp_horizontal\":"
          "true,\"persp_vertical\":true,\"zoom_in\":true,\"zoom_out\":true,"
          "\"pad_mode\":\"constant\"},\"noise\":{\"prob\":0.01},\"distort\":{"
          "\"prob\":0."
          "01},\"dataloader_threads\":4}"
          ","
          "\"input\":{\"seed\":12345,\"db\":true,\"shuffle\":true},"
          "\"output\":{\"measure\":[\"f1\",\"acc\"]}},\"data\":[\""
        + resnet50_train_data + "\",\"" + resnet50_test_data + "\"]}";
  joutstr = japi.jrender(japi.service_train(jtrainstr));
  JDoc jd;
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(201, jd["status"]["code"]);

  ASSERT_TRUE(jd["body"]["measure"]["acc"].GetDouble() <= 1) << "accuracy";
  ASSERT_TRUE(jd["body"]["measure"]["acc"].GetDouble() >= 0.49)
      << "accuracy good";
  ASSERT_TRUE(jd["body"]["measure"]["f1"].GetDouble() <= 1) << "f1";

  // Predict
  std::string jpredictstr
      = "{\"service\":\"imgserv\",\"parameters\":{\"output\":{\"best\":1}},"
        "\"data\":[\""
        + resnet50_test_image + "\"]}";

  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  jd = JDoc();
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200, jd["status"]["code"]);
  ASSERT_TRUE(jd["body"]["predictions"].IsArray());
  std::string cl1
      = jd["body"]["predictions"][0]["classes"][0]["cat"].GetString();
  // Not training for long enough to be 100% sure a cat will be detected
  // ASSERT_TRUE(cl1 == "cats");
  ASSERT_TRUE(jd["body"]["predictions"][0]["classes"][0]["prob"].GetDouble()
              > 0.0);

  std::unordered_set<std::string> lfiles;
  fileops::list_directory(resnet50_train_repo, true, false, false, lfiles);
  for (std::string ff : lfiles)
    {
      if (ff.find("checkpoint") != std::string::npos
          || ff.find("solver") != std::string::npos)
        remove(ff.c_str());
    }
  ASSERT_TRUE(!fileops::file_exists(resnet50_train_repo + "checkpoint-"
                                    + iterations_resnet50 + ".ptw"));
  ASSERT_TRUE(!fileops::file_exists(resnet50_train_repo + "checkpoint-"
                                    + iterations_resnet50 + ".pt"));

  fileops::clear_directory(resnet50_train_repo + "train.lmdb");
  fileops::clear_directory(resnet50_train_repo + "test_0.lmdb");
  fileops::remove_dir(resnet50_train_repo + "train.lmdb");
  fileops::remove_dir(resnet50_train_repo + "test_0.lmdb");
}

TEST(torchapi, service_train_image_segmentation_deeplabv3)
{
  setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8", true);
  torch::manual_seed(torch_seed);
  at::globalContext().setDeterministicCuDNN(true);

  // Create service
  JsonAPI japi;
  std::string sname = "imgserv";
  std::string jstr
      = "{\"mllib\":\"torch\",\"description\":\"image\",\"type\":"
        "\"supervised\",\"model\":{\"repository\":\""
        + deeplabv3_train_repo
        + "\"},\"parameters\":{\"input\":{\"connector\":\"image\","
          "\"width\":480,\"height\":480,\"db\":true,\"segmentation\":true},"
          "\"mllib\":{\"nclasses\":"
          "13,\"gpu\":true,\"segmentation\":true}}}";
  std::string joutstr = japi.jrender(japi.service_create(sname, jstr));
  ASSERT_EQ(created_str, joutstr);

  // Train
  std::string jtrainstr
      = "{\"service\":\"imgserv\",\"async\":false,\"parameters\":{"
        "\"mllib\":{\"solver\":{\"iterations\":"
        + iterations_deeplabv3 + ",\"base_lr\":" + torch_lr
        + ",\"iter_size\":1,\"solver_type\":\"ADAM\",\"test_"
          "interval\":100},\"net\":{\"batch_size\":4},"
          "\"resume\":false,\"mirror\":true,\"rotate\":true,\"crop_size\":224,"
          "\"cutout\":0.5,\"geometry\":{\"prob\":0.1,\"persp_horizontal\":"
          "true,\"persp_vertical\":true,\"zoom_in\":true,\"zoom_out\":true,"
          "\"pad_mode\":\"constant\"},\"noise\":{\"prob\":0.01},\"distort\":{"
          "\"prob\":0."
          "01}},"
          "\"input\":{\"seed\":12345,\"db\":true,\"shuffle\":true,"
          "\"segmentation\":true,\"scale\":0.0039,\"mean\":[0.485,0.456,0.406]"
          ",\"std\":[0.229,0.224,0.225]},"
          "\"output\":{\"measure\":[\"meaniou\",\"acc\"]}},\"data\":[\""
        + deeplabv3_train_data + "\",\"" + deeplabv3_test_data + "\"]}";
  joutstr = japi.jrender(japi.service_train(jtrainstr));
  JDoc jd;
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(201, jd["status"]["code"]);

  ASSERT_TRUE(jd["body"]["measure"]["meanacc"].GetDouble() <= 1) << "accuracy";
  ASSERT_TRUE(jd["body"]["measure"]["meanacc"].GetDouble() >= 0.007)
      << "accuracy good";
  ASSERT_TRUE(jd["body"]["measure"]["meaniou"].GetDouble() <= 1) << "meaniou";

  std::string jpredictstr
      = "{\"service\":\"imgserv\",\"parameters\":{"
        "\"input\":{\"height\":480,"
        "\"width\":480,\"scale\":0.0039,\"mean\":[0.485,0.456,0.406],\"std\":["
        "0.229,0.224,0.225]},\"output\":{\"segmentation\":true, "
        "\"confidences\":[\"best\"]}},\"data\":[\""
        + deeplabv3_test_image + "\"]}";

  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200, jd["status"]["code"]);
  ASSERT_TRUE(jd["body"]["predictions"].IsArray());

  std::unordered_set<std::string> lfiles;
  fileops::list_directory(deeplabv3_train_repo, true, false, false, lfiles);
  for (std::string ff : lfiles)
    {
      if (ff.find("checkpoint") != std::string::npos
          || ff.find("solver") != std::string::npos)
        remove(ff.c_str());
    }
  ASSERT_TRUE(!fileops::file_exists(deeplabv3_train_repo + "checkpoint-"
                                    + iterations_deeplabv3 + ".ptw"));
  ASSERT_TRUE(!fileops::file_exists(deeplabv3_train_repo + "checkpoint-"
                                    + iterations_deeplabv3 + ".pt"));

  fileops::clear_directory(deeplabv3_train_repo + "train.lmdb");
  fileops::clear_directory(deeplabv3_train_repo + "test_0.lmdb");
  fileops::remove_dir(deeplabv3_train_repo + "train.lmdb");
  fileops::remove_dir(deeplabv3_train_repo + "test_0.lmdb");
}

TEST(torchapi, service_train_image_segmentation_deeplabv3_dice)
{
  setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8", true);
  torch::manual_seed(torch_seed);
  at::globalContext().setDeterministicCuDNN(true);

  // Create service
  JsonAPI japi;
  std::string sname = "imgserv";
  std::string jstr
      = "{\"mllib\":\"torch\",\"description\":\"image\",\"type\":"
        "\"supervised\",\"model\":{\"repository\":\""
        + deeplabv3_train_repo
        + "\"},\"parameters\":{\"input\":{\"connector\":\"image\","
          "\"width\":480,\"height\":480,\"db\":true,\"segmentation\":true},"
          "\"mllib\":{\"nclasses\":"
          "13,\"gpu\":true,\"segmentation\":true,\"loss\":\"dice_weighted_"
          "all\"}}"
          "}";
  std::string joutstr = japi.jrender(japi.service_create(sname, jstr));
  ASSERT_EQ(created_str, joutstr);

  // Train
  std::string jtrainstr
      = "{\"service\":\"imgserv\",\"async\":false,\"parameters\":{"
        "\"mllib\":{\"solver\":{\"iterations\":"
        + iterations_deeplabv3 + ",\"base_lr\":" + torch_lr
        + ",\"iter_size\":1,\"solver_type\":\"ADAM\",\"test_"
          "interval\":100},\"net\":{\"batch_size\":4},"
          "\"resume\":false,\"mirror\":true,\"rotate\":true,\"crop_size\":224,"
          "\"cutout\":0.5,\"geometry\":{\"prob\":0.1,\"persp_horizontal\":"
          "true,\"persp_vertical\":true,\"zoom_in\":true,\"zoom_out\":true,"
          "\"pad_mode\":\"constant\"},\"noise\":{\"prob\":0.01},\"distort\":{"
          "\"prob\":0."
          "01}},"
          "\"input\":{\"seed\":12345,\"db\":true,\"shuffle\":true,"
          "\"segmentation\":true,\"scale\":0.0039,\"mean\":[0.485,0.456,0.406]"
          ",\"std\":[0.229,0.224,0.225]},"
          "\"output\":{\"measure\":[\"meaniou\",\"acc\"]}},\"data\":[\""
        + deeplabv3_train_data + "\",\"" + deeplabv3_test_data + "\"]}";
  joutstr = japi.jrender(japi.service_train(jtrainstr));
  JDoc jd;
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(201, jd["status"]["code"]);

  ASSERT_TRUE(jd["body"]["measure"]["meanacc"].GetDouble() <= 1) << "accuracy";
  ASSERT_TRUE(jd["body"]["measure"]["meanacc"].GetDouble() >= 0.007)
      << "accuracy good";
  ASSERT_TRUE(jd["body"]["measure"]["meaniou"].GetDouble() <= 1) << "meaniou";

  std::string jpredictstr
      = "{\"service\":\"imgserv\",\"parameters\":{"
        "\"input\":{\"height\":480,"
        "\"width\":480,\"scale\":0.0039,\"mean\":[0.485,0.456,0.406],\"std\":["
        "0.229,0.224,0.225]},\"output\":{\"segmentation\":true, "
        "\"confidences\":[\"best\"]}},\"data\":[\""
        + deeplabv3_test_image + "\"]}";

  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200, jd["status"]["code"]);
  ASSERT_TRUE(jd["body"]["predictions"].IsArray());

  std::unordered_set<std::string> lfiles;
  fileops::list_directory(deeplabv3_train_repo, true, false, false, lfiles);
  for (std::string ff : lfiles)
    {
      if (ff.find("checkpoint") != std::string::npos
          || ff.find("solver") != std::string::npos)
        remove(ff.c_str());
    }
  ASSERT_TRUE(!fileops::file_exists(deeplabv3_train_repo + "checkpoint-"
                                    + iterations_deeplabv3 + ".ptw"));
  ASSERT_TRUE(!fileops::file_exists(deeplabv3_train_repo + "checkpoint-"
                                    + iterations_deeplabv3 + ".pt"));

  fileops::clear_directory(deeplabv3_train_repo + "train.lmdb");
  fileops::clear_directory(deeplabv3_train_repo + "test_0.lmdb");
  fileops::remove_dir(deeplabv3_train_repo + "train.lmdb");
  fileops::remove_dir(deeplabv3_train_repo + "test_0.lmdb");
}

TEST(torchapi, service_train_image_segmentation_segformer)
{
  setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8", true);
  torch::manual_seed(torch_seed);
  at::globalContext().setDeterministicCuDNN(true);

  // Create service
  JsonAPI japi;
  std::string sname = "imgserv";
  std::string jstr
      = "{\"mllib\":\"torch\",\"description\":\"image\",\"type\":"
        "\"supervised\",\"model\":{\"repository\":\""
        + segformer_train_repo
        + "\"},\"parameters\":{\"input\":{\"connector\":\"image\","
          "\"width\":480,\"height\":480,\"db\":true,\"segmentation\":true},"
          "\"mllib\":{\"nclasses\":"
          "13,\"gpu\":true,\"segmentation\":true}}}";
  std::string joutstr = japi.jrender(japi.service_create(sname, jstr));
  ASSERT_EQ(created_str, joutstr);

  // Train
  std::string jtrainstr
      = "{\"service\":\"imgserv\",\"async\":false,\"parameters\":{"
        "\"mllib\":{\"solver\":{\"iterations\":"
        + iterations_deeplabv3 + ",\"base_lr\":" + torch_lr
        + ",\"iter_size\":1,\"solver_type\":\"ADAM\",\"test_"
          "interval\":100},\"net\":{\"batch_size\":4},"
          "\"resume\":false,\"mirror\":true,\"rotate\":true,\"crop_size\":224,"
          "\"cutout\":0.5,\"geometry\":{\"prob\":0.1,\"persp_horizontal\":"
          "true,\"persp_vertical\":true,\"zoom_in\":true,\"zoom_out\":true,"
          "\"pad_mode\":\"constant\"},\"noise\":{\"prob\":0.01},\"distort\":{"
          "\"prob\":0."
          "01}},"
          "\"input\":{\"seed\":12345,\"db\":true,\"shuffle\":true,"
          "\"segmentation\":true,\"scale\":0.0039,\"mean\":[0.485,0.456,0.406]"
          ",\"std\":[0.229,0.224,0.225]},"
          "\"output\":{\"measure\":[\"meaniou\",\"acc\"]}},\"data\":[\""
        + deeplabv3_train_data + "\",\"" + deeplabv3_test_data + "\"]}";
  joutstr = japi.jrender(japi.service_train(jtrainstr));
  JDoc jd;
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(201, jd["status"]["code"]);

  ASSERT_TRUE(jd["body"]["measure"]["meanacc"].GetDouble() <= 1) << "accuracy";
  ASSERT_TRUE(jd["body"]["measure"]["meanacc"].GetDouble() >= 0.003)
      << "accuracy good";
  ASSERT_TRUE(jd["body"]["measure"]["meaniou"].GetDouble() <= 1) << "meaniou";

  std::string jpredictstr
      = "{\"service\":\"imgserv\",\"parameters\":{"
        "\"input\":{\"height\":480,"
        "\"width\":480,\"scale\":0.0039,\"mean\":[0.485,0.456,0.406],\"std\":["
        "0.229,0.224,0.225]},\"output\":{\"segmentation\":true, "
        "\"confidences\":[\"best\"]}},\"data\":[\""
        + deeplabv3_test_image + "\"]}";

  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200, jd["status"]["code"]);
  ASSERT_TRUE(jd["body"]["predictions"].IsArray());

  std::unordered_set<std::string> lfiles;
  fileops::list_directory(segformer_train_repo, true, false, false, lfiles);
  for (std::string ff : lfiles)
    {
      if (ff.find("checkpoint") != std::string::npos
          || ff.find("solver") != std::string::npos)
        remove(ff.c_str());
    }
  ASSERT_TRUE(!fileops::file_exists(segformer_train_repo + "checkpoint-"
                                    + iterations_deeplabv3 + ".ptw"));
  ASSERT_TRUE(!fileops::file_exists(segformer_train_repo + "checkpoint-"
                                    + iterations_deeplabv3 + ".pt"));

  fileops::clear_directory(segformer_train_repo + "train.lmdb");
  fileops::clear_directory(segformer_train_repo + "test_0.lmdb");
  fileops::remove_dir(segformer_train_repo + "train.lmdb");
  fileops::remove_dir(segformer_train_repo + "test_0.lmdb");
}

TEST(torchapi, service_publish_trained_model)
{
  setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8", true);
  torch::manual_seed(torch_seed);
  at::globalContext().setDeterministicCuDNN(true);

  // Create service
  JsonAPI japi;
  std::string sname = "imgserv";
  std::string jstr
      = "{\"mllib\":\"torch\",\"description\":\"image\",\"type\":"
        "\"supervised\",\"model\":{\"repository\":\""
        + resnet50_train_repo
        + "\"},\"parameters\":{\"input\":{\"connector\":\"image\","
          "\"width\":256,\"height\":256,\"db\":true},\"mllib\":{\"nclasses\":"
          "2,\"finetuning\":true,\"gpu\":true}}}";
  std::string joutstr = japi.jrender(japi.service_create(sname, jstr));
  ASSERT_EQ(created_str, joutstr);

  // Train for 1 iteration
  std::string jtrainstr
      = "{\"service\":\"imgserv\",\"async\":false,\"parameters\":{"
        "\"mllib\":{\"solver\":{\"iterations\":1,\"base_lr\":"
        + torch_lr
        + ",\"iter_size\":4,\"solver_type\":\"ADAM\",\"test_"
          "interval\":1},\"net\":{\"batch_size\":4},"
          "\"resume\":false,\"mirror\":true,\"rotate\":true,\"crop_size\":224,"
          "\"cutout\":0.5},\"input\":{\"seed\":12345,\"db\":true,\"shuffle\":"
          "true},\"output\":{\"measure\":[\"f1\",\"acc\"]}},\"data\":[\""
        + resnet50_train_data + "\",\"" + resnet50_test_data + "\"]}";
  joutstr = japi.jrender(japi.service_train(jtrainstr));
  JDoc jd;
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(201, jd["status"]["code"]);

  // Delete service
  japi.service_delete(sname, "");

  // Publish service somewhere
  std::string published_repo = "published_resnet50";
  jstr = "{\"mllib\":\"torch\",\"description\":\"image\",\"type\":"
         "\"supervised\",\"model\":{\"repository\":\""
         + published_repo
         + "\",\"create_repository\":true},\"parameters\":{\"input\":{"
           "\"connector\":\"image\","
           "\"width\":256,\"height\":256,\"db\":true},\"mllib\":{\"nclasses\":"
           "2,\"gpu\":true,\"from_repository\":\""
         + resnet50_train_repo + "\"}}}";
  joutstr = japi.jrender(japi.service_create(sname, jstr));
  ASSERT_EQ(created_str, joutstr);

  ASSERT_TRUE(fileops::file_exists(published_repo + "/checkpoint-1.ptw"));
  ASSERT_TRUE(fileops::file_exists(published_repo + "/checkpoint-1.pt"));
  ASSERT_TRUE(fileops::file_exists(published_repo + "/best_model.txt"));
  ASSERT_TRUE(fileops::file_exists(published_repo + "/model.json"));
  ASSERT_TRUE(fileops::file_exists(published_repo + "/config.json"));
  ASSERT_FALSE(fileops::file_exists(published_repo + "/resnet50.pt"));

  // Check on published model configuration
  std::string config_path = published_repo + "/config.json";
  std::ifstream ifs_config(config_path.c_str(), std::ios::binary);
  ASSERT_TRUE(ifs_config.is_open());
  std::stringstream config_sstr;
  config_sstr << ifs_config.rdbuf();
  ifs_config.close();
  rapidjson::Document d_config;
  d_config.Parse<rapidjson::kParseNanAndInfFlag>(config_sstr.str().c_str());
  auto d_config_input = d_config["parameters"]["input"].GetObject();
  ASSERT_TRUE(d_config_input.HasMember("width"));
  ASSERT_TRUE(d_config_input["width"].GetInt() == 224);
  ASSERT_TRUE(d_config_input.HasMember("height"));
  ASSERT_TRUE(d_config_input["height"].GetInt() == 224);

  // Clean up train repo
  std::unordered_set<std::string> lfiles;
  fileops::list_directory(resnet50_train_repo, true, false, false, lfiles);
  for (std::string ff : lfiles)
    {
      if (ff.find("checkpoint") != std::string::npos
          || ff.find("solver") != std::string::npos
          || ff.find("best_model") != std::string::npos)
        remove(ff.c_str());
    }
  ASSERT_TRUE(!fileops::file_exists(resnet50_train_repo + "checkpoint-1.ptw"));
  ASSERT_TRUE(!fileops::file_exists(resnet50_train_repo + "checkpoint-1.pt"));

  fileops::clear_directory(resnet50_train_repo + "train.lmdb");
  fileops::clear_directory(resnet50_train_repo + "test_0.lmdb");
  fileops::remove_dir(resnet50_train_repo + "train.lmdb");
  fileops::remove_dir(resnet50_train_repo + "test_0.lmdb");

  // Clean up published repo
  fileops::clear_directory(published_repo);
  fileops::remove_dir(published_repo);
}

TEST(torchapi, service_create_multiple_models_fails)
{
  // check that creating native model in a repo with a torchscript raise an
  // exception
  JsonAPI japi;
  std::string sname = "imgserv";
  std::string jstr
      = "{\"mllib\":\"torch\",\"description\":\"image\",\"type\":"
        "\"supervised\",\"model\":{\"repository\":\""
        + resnet50_train_repo
        + "\"},\"parameters\":{\"input\":{"
          "\"connector\":\"image\",\"width\":224,\"height\":224,\"db\":true},"
          "\"mllib\":{\"nclasses\":2,\"gpu\":true,\"template\":\"resnet50\"}}"
          "}";
  std::string joutstr = japi.jrender(japi.service_create(sname, jstr));
  JDoc jd;
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_EQ(jd["status"]["code"], 500);
  ASSERT_TRUE(std::string(jd["status"]["dd_msg"].GetString())
                  .find("Only one of these must be provided: traced net, "
                        "protofile or native template")
              != std::string::npos);
}

TEST(torchapi, service_train_images_sam)
{
  setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8", true);
  torch::manual_seed(torch_seed);
  at::globalContext().setDeterministicCuDNN(true);

  // Create service
  JsonAPI japi;
  std::string sname = "imgserv";
  std::string jstr
      = "{\"mllib\":\"torch\",\"description\":\"image\",\"type\":"
        "\"supervised\",\"model\":{\"repository\":\""
        + resnet50_train_repo
        + "\"},\"parameters\":{\"input\":{\"connector\":\"image\","
          "\"width\":256,\"height\":256,\"db\":true},\"mllib\":{\"nclasses\":"
          "2,\"finetuning\":true,\"gpu\":true}}}";
  std::string joutstr = japi.jrender(japi.service_create(sname, jstr));
  ASSERT_EQ(created_str, joutstr);

  // Train
  std::string jtrainstr
      = "{\"service\":\"imgserv\",\"async\":false,\"parameters\":{"
        "\"mllib\":{\"solver\":{\"iterations\":"
        + iterations_resnet50 + ",\"base_lr\":" + torch_lr
        + ",\"iter_size\":4,\"solver_type\":\"ADAM\",\"test_"
          "interval\":200,\"sam\":true},\"net\":{\"batch_size\":4},"
          "\"resume\":false,\"mirror\":true,\"rotate\":true,\"crop_size\":224,"
          "\"cutout\":0.5}"
          ","
          "\"input\":{\"seed\":12345,\"db\":true,\"shuffle\":true},"
          "\"output\":{\"measure\":[\"f1\",\"acc\"]}},\"data\":[\""
        + resnet50_train_data + "\",\"" + resnet50_test_data + "\"]}";
  joutstr = japi.jrender(japi.service_train(jtrainstr));
  JDoc jd;
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(201, jd["status"]["code"]);

  ASSERT_TRUE(jd["body"]["measure"]["acc"].GetDouble() <= 1) << "accuracy";
  ASSERT_TRUE(jd["body"]["measure"]["acc"].GetDouble() >= 0.45)
      << "accuracy good";
  ASSERT_TRUE(jd["body"]["measure"]["f1"].GetDouble() <= 1) << "f1";

  // Predict
  std::string jpredictstr
      = "{\"service\":\"imgserv\",\"parameters\":{\"output\":{\"best\":1}},"
        "\"data\":[\""
        + resnet50_test_image + "\"]}";

  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  jd = JDoc();
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200, jd["status"]["code"]);
  ASSERT_TRUE(jd["body"]["predictions"].IsArray());
  std::string cl1
      = jd["body"]["predictions"][0]["classes"][0]["cat"].GetString();
  // Not training for long enough to be 100% sure a cat will be detected
  // ASSERT_TRUE(cl1 == "cats");
  ASSERT_TRUE(jd["body"]["predictions"][0]["classes"][0]["prob"].GetDouble()
              > 0.0);

  std::unordered_set<std::string> lfiles;
  fileops::list_directory(resnet50_train_repo, true, false, false, lfiles);
  for (std::string ff : lfiles)
    {
      if (ff.find("checkpoint") != std::string::npos
          || ff.find("solver") != std::string::npos)
        remove(ff.c_str());
    }
  ASSERT_TRUE(!fileops::file_exists(resnet50_train_repo + "checkpoint-"
                                    + iterations_resnet50 + ".ptw"));
  ASSERT_TRUE(!fileops::file_exists(resnet50_train_repo + "checkpoint-"
                                    + iterations_resnet50 + ".pt"));

  fileops::clear_directory(resnet50_train_repo + "train.lmdb");
  fileops::clear_directory(resnet50_train_repo + "test_0.lmdb");
  fileops::remove_dir(resnet50_train_repo + "train.lmdb");
  fileops::remove_dir(resnet50_train_repo + "test_0.lmdb");
}

TEST(torchapi, service_train_images_multiple_testsets)
{
  torch::manual_seed(torch_seed);
  at::globalContext().setDeterministicCuDNN(true);

  mkdir(resnet50_test_cats_data.c_str(), 0775);
  mkdir((resnet50_test_cats_data + "dogs").c_str(), 0775);

  int sym
      = symlink("../test/cats/", (resnet50_test_cats_data + "cats").c_str());
  (void)sym;

  // Create service
  JsonAPI japi;
  std::string sname = "imgserv";
  std::string jstr
      = "{\"mllib\":\"torch\",\"description\":\"image\",\"type\":"
        "\"supervised\",\"model\":{\"repository\":\""
        + resnet50_train_repo
        + "\"},\"parameters\":{\"input\":{\"connector\":\"image\","
          "\"width\":224,\"height\":224,\"db\":false},\"mllib\":{\"nclasses\":"
          "2,\"finetuning\":true,\"gpu\":true}}}";
  std::string joutstr = japi.jrender(japi.service_create(sname, jstr));
  ASSERT_EQ(created_str, joutstr);

  // Train
  std::string jtrainstr
      = "{\"service\":\"imgserv\",\"async\":false,\"parameters\":{"
        "\"mllib\":{\"solver\":{\"iterations\":5,\"base_lr\":"
        + torch_lr
        + ",\"iter_size\":4,\"solver_type\":\"ADAM\",\"test_"
          "interval\":2},\"net\":{\"batch_size\":4},"
          "\"resume\":false},"
          "\"input\":{\"seed\":12345,\"db\":false,\"shuffle\":true},"
          "\"output\":{\"measure\":[\"f1\",\"acc\"]}},\"data\":[\""
        + resnet50_train_data + "\",\"" + resnet50_test_data + "\",\""
        + resnet50_test_cats_data + "\"]}";
  joutstr = japi.jrender(japi.service_train(jtrainstr));
  JDoc jd;
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(201, jd["status"]["code"]);

  ASSERT_TRUE(jd["body"].HasMember("measure"));

  ASSERT_TRUE(jd["body"]["measures"].IsArray());
  ASSERT_EQ(jd["body"]["measures"].Size(), 2);

  ASSERT_TRUE(
      fileops::file_exists(resnet50_train_repo + "best_model_test_0.txt"));
  remove((resnet50_train_repo + "best_model_test_0.txt").c_str());
  ASSERT_TRUE(
      fileops::file_exists(resnet50_train_repo + "best_model_test_1.txt"));
  remove((resnet50_train_repo + "best_model_test_1.txt").c_str());
  ASSERT_TRUE(fileops::file_exists(resnet50_train_repo + "best_model.txt"));
  remove((resnet50_train_repo + "best_model.txt").c_str());

  std::unordered_set<std::string> lfiles;
  fileops::list_directory(resnet50_train_repo, true, false, false, lfiles);
  for (std::string ff : lfiles)
    {
      if (ff.find("checkpoint") != std::string::npos
          || ff.find("solver") != std::string::npos)
        remove(ff.c_str());
    }
  ASSERT_TRUE(!fileops::file_exists(resnet50_train_repo + "checkpoint-"
                                    + iterations_resnet50 + ".ptw"));
  ASSERT_TRUE(!fileops::file_exists(resnet50_train_repo + "checkpoint-"
                                    + iterations_resnet50 + ".pt"));

  fileops::clear_directory(resnet50_train_repo + "train.lmdb");
  fileops::clear_directory(resnet50_train_repo + "train.lmdb");
  remove((resnet50_test_cats_data + "cats").c_str());
  remove((resnet50_test_cats_data + "dogs").c_str());
  fileops::remove_dir(resnet50_test_cats_data);
}

TEST(torchapi, service_train_images_multiple_testsets_db)
{
  torch::manual_seed(torch_seed);
  at::globalContext().setDeterministicCuDNN(true);

  mkdir(resnet50_test_cats_data.c_str(), 0775);
  mkdir((resnet50_test_cats_data + "dogs").c_str(), 0775);

  ASSERT_EQ(
      symlink("../test/cats/", (resnet50_test_cats_data + "cats").c_str()), 0);

  // Create service
  JsonAPI japi;
  std::string sname = "imgserv";
  std::string jstr
      = "{\"mllib\":\"torch\",\"description\":\"image\",\"type\":"
        "\"supervised\",\"model\":{\"repository\":\""
        + resnet50_train_repo
        + "\"},\"parameters\":{\"input\":{\"connector\":\"image\","
          "\"width\":224,\"height\":224,\"db\":true},\"mllib\":{\"nclasses\":"
          "2,\"finetuning\":true,\"gpu\":true}}}";
  std::string joutstr = japi.jrender(japi.service_create(sname, jstr));
  ASSERT_EQ(created_str, joutstr);

  // Train
  std::string jtrainstr
      = "{\"service\":\"imgserv\",\"async\":false,\"parameters\":{"
        "\"mllib\":{\"solver\":{\"iterations\":5,\"base_lr\":"
        + torch_lr
        + ",\"iter_size\":4,\"solver_type\":\"ADAM\",\"test_"
          "interval\":2},\"net\":{\"batch_size\":4},"
          "\"resume\":false},"
          "\"input\":{\"seed\":12345,\"db\":true,\"shuffle\":true},"
          "\"output\":{\"measure\":[\"f1\",\"acc\"]}},\"data\":[\""
        + resnet50_train_data + "\",\"" + resnet50_test_data + "\",\""
        + resnet50_test_cats_data + "\"]}";
  joutstr = japi.jrender(japi.service_train(jtrainstr));
  JDoc jd;
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(201, jd["status"]["code"]);

  ASSERT_TRUE(jd["body"].HasMember("measure"));

  ASSERT_TRUE(jd["body"]["measures"].IsArray());
  ASSERT_EQ(jd["body"]["measures"].Size(), 2);

  ASSERT_TRUE(
      fileops::file_exists(resnet50_train_repo + "best_model_test_0.txt"));
  remove((resnet50_train_repo + "best_model_test_0.txt").c_str());
  ASSERT_TRUE(
      fileops::file_exists(resnet50_train_repo + "best_model_test_1.txt"));
  remove((resnet50_train_repo + "best_model_test_1.txt").c_str());
  ASSERT_TRUE(fileops::file_exists(resnet50_train_repo + "best_model.txt"));
  remove((resnet50_train_repo + "best_model.txt").c_str());

  std::unordered_set<std::string> lfiles;
  fileops::list_directory(resnet50_train_repo, true, false, false, lfiles);
  for (std::string ff : lfiles)
    {
      if (ff.find("checkpoint") != std::string::npos
          || ff.find("solver") != std::string::npos)
        remove(ff.c_str());
    }
  ASSERT_TRUE(!fileops::file_exists(resnet50_train_repo + "checkpoint-"
                                    + iterations_resnet50 + ".ptw"));
  ASSERT_TRUE(!fileops::file_exists(resnet50_train_repo + "checkpoint-"
                                    + iterations_resnet50 + ".pt"));

  fileops::clear_directory(resnet50_train_repo + "train.lmdb");
  fileops::clear_directory(resnet50_train_repo + "test_0.lmdb");
  fileops::clear_directory(resnet50_train_repo + "test_1.lmdb");
  remove((resnet50_test_cats_data + "cats").c_str());
  remove((resnet50_test_cats_data + "dogs").c_str());
  fileops::remove_dir(resnet50_test_cats_data);
  fileops::remove_dir(resnet50_train_repo + "train.lmdb");
  fileops::remove_dir(resnet50_train_repo + "test_0.lmdb");
  fileops::remove_dir(resnet50_train_repo + "test_1.lmdb");
}

TEST(torchapi, service_train_images_split_list)
{
  setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8", true);
  torch::manual_seed(torch_seed);
  at::globalContext().setDeterministicCuDNN(true);

  // Create service
  JsonAPI japi;
  std::string sname = "imgserv";
  std::string jstr
      = "{\"mllib\":\"torch\",\"description\":\"image\",\"type\":"
        "\"supervised\",\"model\":{\"repository\":\""
        + resnet50_train_repo
        + "\"},\"parameters\":{\"input\":{\"connector\":\"image\","
          "\"width\":224,\"height\":224,\"db\":false},\"mllib\":{\"nclasses\":"
          "2,\"finetuning\":true,\"gpu\":true}}}";
  std::string joutstr = japi.jrender(japi.service_create(sname, jstr));
  ASSERT_EQ(created_str, joutstr);

  // Train
  std::string jtrainstr
      = "{\"service\":\"imgserv\",\"async\":false,\"parameters\":{"
        "\"mllib\":{\"solver\":{\"iterations\":"
        + iterations_resnet50 + ",\"base_lr\":" + torch_lr
        + ",\"iter_size\":4,\"solver_type\":\"ADAM\",\"test_"
          "interval\":200},\"net\":{\"batch_size\":4},\"resume\":false},"
          "\"input\":{\"seed\":12345,\"db\":false,\"shuffle\":true,\"test_"
          "split\":0.1},"
          "\"output\":{\"measure\":[\"acc\",\"f1\"]}},\"data\":[\""
        + resnet50_train_data_classif + "\"]}";
  joutstr = japi.jrender(japi.service_train(jtrainstr));
  JDoc jd;
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(201, jd["status"]["code"]);

  ASSERT_TRUE(jd["body"]["measure"]["iteration"] == 200) << "iterations";
  ASSERT_TRUE(jd["body"]["measure"]["train_loss"].GetDouble() <= 3.0)
      << "loss";

  std::unordered_set<std::string> lfiles;
  fileops::list_directory(resnet50_train_repo, true, false, false, lfiles);
  for (std::string ff : lfiles)
    {
      if (ff.find("checkpoint") != std::string::npos
          || ff.find("solver") != std::string::npos)
        remove(ff.c_str());
    }
  ASSERT_TRUE(!fileops::file_exists(resnet50_train_repo + "checkpoint-"
                                    + iterations_resnet50 + ".ptw"));
  ASSERT_TRUE(!fileops::file_exists(resnet50_train_repo + "checkpoint-"
                                    + iterations_resnet50 + ".pt"));

  fileops::clear_directory(resnet50_train_repo + "train.lmdb");
  fileops::clear_directory(resnet50_train_repo + "test_0.lmdb");
  fileops::remove_dir(resnet50_train_repo + "train.lmdb");
  fileops::remove_dir(resnet50_train_repo + "test_0.lmdb");
}

TEST(torchapi, service_train_images_split_regression_db_true)
{
  setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8", true);
  torch::manual_seed(torch_seed);
  at::globalContext().setDeterministicCuDNN(true);

  // Create service
  JsonAPI japi;
  std::string sname = "imgserv";
  std::string jstr
      = "{\"mllib\":\"torch\",\"description\":\"image\",\"type\":"
        "\"supervised\",\"model\":{\"repository\":\""
        + resnet50_train_repo
        + "\"},\"parameters\":{\"input\":{\"connector\":\"image\","
          "\"width\":224,\"height\":224,\"db\":true},\"mllib\":{\"ntargets\":"
          "1,\"finetuning\":true,\"regression\":true,\"gpu\":true,\"loss\":"
          "\"L2\"}}}";
  std::string joutstr = japi.jrender(japi.service_create(sname, jstr));
  ASSERT_EQ(created_str, joutstr);

  // Train
  std::string jtrainstr
      = "{\"service\":\"imgserv\",\"async\":false,\"parameters\":{"
        "\"mllib\":{\"solver\":{\"iterations\":"
        + iterations_resnet50 + ",\"base_lr\":" + torch_lr
        + ",\"iter_size\":4,\"solver_type\":\"ADAM\",\"test_"
          "interval\":"
        + iterations_resnet50
        + "},\"net\":{\"batch_size\":4},\"resume\":false},"
          "\"input\":{\"seed\":12345,\"db\":true,\"shuffle\":true,\"test_"
          "split\":0.1},"
          "\"output\":{\"measure\":[\"eucll\"]}},\"data\":[\""
        + resnet50_train_data_reg + "\"]}";
  joutstr = japi.jrender(japi.service_train(jtrainstr));
  JDoc jd;
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(201, jd["status"]["code"]);

  ASSERT_TRUE(jd["body"]["measure"]["iteration"] == 200) << "iterations";
  ASSERT_TRUE(jd["body"]["measure"]["eucll"].GetDouble() <= 15.0) << "eucll";

  std::unordered_set<std::string> lfiles;
  fileops::list_directory(resnet50_train_repo, true, false, false, lfiles);
  for (std::string ff : lfiles)
    {
      if (ff.find("checkpoint") != std::string::npos
          || ff.find("solver") != std::string::npos)
        remove(ff.c_str());
    }
  ASSERT_TRUE(!fileops::file_exists(resnet50_train_repo + "checkpoint-"
                                    + iterations_resnet50 + ".ptw"));
  ASSERT_TRUE(!fileops::file_exists(resnet50_train_repo + "checkpoint-"
                                    + iterations_resnet50 + ".pt"));

  fileops::clear_directory(resnet50_train_repo + "train.lmdb");
  fileops::clear_directory(resnet50_train_repo + "test_0.lmdb");
  fileops::remove_dir(resnet50_train_repo + "train.lmdb");
  fileops::remove_dir(resnet50_train_repo + "test_0.lmdb");
}

TEST(torchapi, service_train_images_split_regression_db_false)
{
  setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8", true);
  torch::manual_seed(torch_seed);
  at::globalContext().setDeterministicCuDNN(true);

  // Create service
  JsonAPI japi;
  std::string sname = "imgserv";
  std::string jstr
      = "{\"mllib\":\"torch\",\"description\":\"image\",\"type\":"
        "\"supervised\",\"model\":{\"repository\":\""
        + resnet50_train_repo
        + "\"},\"parameters\":{\"input\":{\"connector\":\"image\","
          "\"width\":224,\"height\":224,\"db\":false},\"mllib\":{\"ntargets\":"
          "1,\"finetuning\":true,\"regression\":true,\"gpu\":true,\"loss\":"
          "\"L2\"}}}";
  std::string joutstr = japi.jrender(japi.service_create(sname, jstr));
  ASSERT_EQ(created_str, joutstr);

  // Train
  std::string jtrainstr
      = "{\"service\":\"imgserv\",\"async\":false,\"parameters\":{"
        "\"mllib\":{\"solver\":{\"iterations\":"
        + iterations_resnet50 + ",\"base_lr\":" + torch_lr
        + ",\"iter_size\":4,\"solver_"
          "type\":\"ADAM\",\"test_interval\":200},\"net\":{\"batch_size\":4},"
          "\"resume\":false},"
          "\"input\":{\"seed\":12345,\"db\":false,\"shuffle\":true,\"test_"
          "split\":0.1},"
          "\"output\":{\"measure\":[\"eucll\"]}},\"data\":[\""
        + resnet50_train_data_reg + "\"]}";
  joutstr = japi.jrender(japi.service_train(jtrainstr));
  JDoc jd;
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(201, jd["status"]["code"]);

  ASSERT_TRUE(jd["body"]["measure"]["iteration"] == 200) << "iterations";
  ASSERT_TRUE(jd["body"]["measure"]["eucll"].GetDouble() <= 15.0) << "eucll";

  std::unordered_set<std::string> lfiles;
  fileops::list_directory(resnet50_train_repo, true, false, false, lfiles);
  for (std::string ff : lfiles)
    {
      if (ff.find("checkpoint") != std::string::npos
          || ff.find("solver") != std::string::npos)
        remove(ff.c_str());
    }
  ASSERT_TRUE(!fileops::file_exists(resnet50_train_repo + "checkpoint-"
                                    + iterations_resnet50 + ".ptw"));
  ASSERT_TRUE(!fileops::file_exists(resnet50_train_repo + "checkpoint-"
                                    + iterations_resnet50 + ".pt"));

  fileops::clear_directory(resnet50_train_repo + "train.lmdb");
  fileops::clear_directory(resnet50_train_repo + "test_0.lmdb");
  fileops::remove_dir(resnet50_train_repo + "train.lmdb");
  fileops::remove_dir(resnet50_train_repo + "test_0.lmdb");
}

TEST(torchapi, service_train_object_detection_fasterrcnn)
{
  setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8", true);
  torch::manual_seed(torch_seed);
  at::globalContext().setDeterministicCuDNN(true);

  JsonAPI japi;
  std::string sname = "detectserv";
  std::string jstr
      = "{\"mllib\":\"torch\",\"description\":\"fasterrcnn\",\"type\":"
        "\"supervised\",\"model\":{\"repository\":\""
        + detect_train_repo_fasterrcnn
        + "\"},\"parameters\":{\"input\":{\"connector\":\"image\",\"height\":"
          "224,\"width\":224,\"rgb\":true,\"scale\":0.0039,\"bbox\":true},"
          "\"mllib\":{\"template\":\"fasterrcnn\",\"gpu\":true,\"nclasses\":"
          "2}}}";

  std::string joutstr = japi.jrender(japi.service_create(sname, jstr));
  ASSERT_EQ(created_str, joutstr);

  // Train
  std::string jtrainstr
      = "{\"service\":\"detectserv\",\"async\":false,\"parameters\":{"
        "\"mllib\":{\"solver\":{\"iterations\":"
        + iterations_detection + ",\"base_lr\":" + torch_lr
        + ",\"iter_size\":2,\"solver_"
          "type\":\"ADAM\",\"test_interval\":200},\"net\":{\"batch_size\":2,"
          "\"test_batch_size\":2},\"resume\":false,\"mirror\":true,\"rotate\":"
          "true,\"crop_size\":224,"
          "\"cutout\":0.1,\"geometry\":{\"prob\":0.1,\"persp_horizontal\":"
          "true,\"persp_vertical\":true,\"zoom_in\":true,\"zoom_out\":true,"
          "\"pad_mode\":\"constant\"},\"noise\":{\"prob\":0.01},\"distort\":{"
          "\"prob\":0."
          "01}},\"input\":{\"seed\":12347,"
          "\"db\":true,\"shuffle\":true},\"output\":{\"measure\":[\"map\"]}},"
          "\"data\":[\""
        + fasterrcnn_train_data + "\",\"" + fasterrcnn_test_data + "\"]}";

  joutstr = japi.jrender(japi.service_train(jtrainstr));
  JDoc jd;
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(201, jd["status"]["code"]);

  ASSERT_EQ(jd["body"]["measure"]["iteration"], 200) << "iterations";
  ASSERT_TRUE(jd["body"]["measure"]["map"].GetDouble() <= 1.0) << "map";
  ASSERT_TRUE(jd["body"]["measure"]["map"].GetDouble() > 0.0) << "map";

  std::unordered_set<std::string> lfiles;
  fileops::list_directory(detect_train_repo_fasterrcnn, true, false, false,
                          lfiles);
  for (std::string ff : lfiles)
    {
      if (ff.find("checkpoint") != std::string::npos
          || ff.find("solver") != std::string::npos)
        remove(ff.c_str());
    }
  ASSERT_TRUE(!fileops::file_exists(detect_train_repo_fasterrcnn
                                    + "checkpoint-" + iterations_detection
                                    + ".ptw"));
  ASSERT_TRUE(!fileops::file_exists(detect_train_repo_fasterrcnn
                                    + "checkpoint-" + iterations_detection
                                    + ".pt"));

  fileops::clear_directory(detect_train_repo_fasterrcnn + "train.lmdb");
  fileops::clear_directory(detect_train_repo_fasterrcnn + "test_0.lmdb");
  fileops::remove_dir(detect_train_repo_fasterrcnn + "train.lmdb");
  fileops::remove_dir(detect_train_repo_fasterrcnn + "test_0.lmdb");
}

TEST(torchapi, service_train_object_detection_yolox)
{
  setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8", true);
  torch::manual_seed(torch_seed);
  at::globalContext().setDeterministicCuDNN(true);

  JsonAPI japi;
  std::string sname = "detectserv";
  std::string jstr
      = "{\"mllib\":\"torch\",\"description\":\"yolox\",\"type\":"
        "\"supervised\",\"model\":{\"repository\":\""
        + detect_train_repo_yolox
        + "\"},\"parameters\":{\"input\":{\"connector\":\"image\",\"height\":"
          "640,\"width\":640,\"rgb\":true,\"bbox\":true},"
          "\"mllib\":{\"template\":\"yolox\",\"gpu\":true,\"nclasses\":"
          "2}}}";

  std::string joutstr = japi.jrender(japi.service_create(sname, jstr));
  ASSERT_EQ(created_str, joutstr);

  // Train
  std::string jtrainstr
      = "{\"service\":\"detectserv\",\"async\":false,\"parameters\":{"
        "\"mllib\":{\"solver\":{\"iterations\":3"
        + std::string("")
        //+ iterations_detection + ",\"base_lr\":" + torch_lr
        + ",\"iter_size\":2,\"solver_"
          "type\":\"ADAM\",\"test_interval\":200},\"net\":{\"batch_size\":2,"
          "\"test_batch_size\":2,\"reg_weight\":0.5},\"resume\":false,"
          "\"mirror\":true,\"rotate\":true,\"crop_size\":512,"
          "\"cutout\":0.1,\"geometry\":{\"prob\":0.1,\"persp_horizontal\":"
          "true,\"persp_vertical\":true,\"zoom_in\":true,\"zoom_out\":true,"
          "\"pad_mode\":\"constant\"},\"noise\":{\"prob\":0.01},\"distort\":{"
          "\"prob\":0."
          "01}},\"input\":{\"seed\":12347,\"db\":true,"
          "\"shuffle\":true},\"output\":{\"measure\":[\"map\"]}},\"data\":[\""
        + fasterrcnn_train_data + "\",\"" + fasterrcnn_test_data + "\"]}";

  joutstr = japi.jrender(japi.service_train(jtrainstr));
  JDoc jd;
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(201, jd["status"]["code"]);

  // ASSERT_EQ(jd["body"]["measure"]["iteration"], 200) << "iterations";
  ASSERT_TRUE(jd["body"]["measure"]["map"].GetDouble() <= 1.0) << "map";
  // ASSERT_TRUE(jd["body"]["measure"]["map"].GetDouble() > 0.0) << "map";

  // check metrics
  auto &meas = jd["body"]["measure"];
  ASSERT_TRUE(meas.HasMember("iou_loss"));
  ASSERT_TRUE(meas.HasMember("conf_loss"));
  ASSERT_TRUE(meas.HasMember("cls_loss"));
  ASSERT_TRUE(meas.HasMember("l1_loss"));
  ASSERT_TRUE(meas.HasMember("train_loss"));
  ASSERT_TRUE(
      std::abs(meas["train_loss"].GetDouble()
               - (meas["iou_loss"].GetDouble() * 0.5
                  + meas["cls_loss"].GetDouble() + meas["l1_loss"].GetDouble()
                  + meas["conf_loss"].GetDouble()))
      < 0.0001);

  // check that predict works fine
  std::string jpredictstr = "{\"service\":\"detectserv\",\"parameters\":{"
                            "\"input\":{\"height\":640,"
                            "\"width\":640},\"output\":{\"bbox\":true, "
                            "\"confidence_threshold\":0.8}},\"data\":[\""
                            + detect_train_repo_fasterrcnn
                            + "/imgs/la_melrose_ave-000020.jpg\"]}";
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  jd = JDoc();
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200, jd["status"]["code"]);

  std::unordered_set<std::string> lfiles;
  fileops::list_directory(detect_train_repo_yolox, true, false, false, lfiles);
  for (std::string ff : lfiles)
    {
      if (ff.find("checkpoint") != std::string::npos
          || ff.find("solver") != std::string::npos)
        remove(ff.c_str());
    }
  ASSERT_TRUE(!fileops::file_exists(detect_train_repo_yolox + "checkpoint-"
                                    + iterations_detection + ".ptw"));
  ASSERT_TRUE(!fileops::file_exists(detect_train_repo_yolox + "checkpoint-"
                                    + iterations_detection + ".pt"));

  fileops::clear_directory(detect_train_repo_yolox + "train.lmdb");
  fileops::clear_directory(detect_train_repo_yolox + "test_0.lmdb");
  fileops::remove_dir(detect_train_repo_yolox + "train.lmdb");
  fileops::remove_dir(detect_train_repo_yolox + "test_0.lmdb");
}

TEST(torchapi, service_train_images_native)
{
  setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8", true);
  torch::manual_seed(torch_seed);
  at::globalContext().setDeterministicCuDNN(true);

  // Create service
  JsonAPI japi;

  std::string native_resnet_repo = "native_resnet";
  int iterations_native = 100;
  mkdir(native_resnet_repo.c_str(), 0777);

  std::string sname = "imgserv";
  std::string jstr
      = "{\"mllib\":\"torch\",\"description\":\"image\",\"type\":"
        "\"supervised\",\"model\":{\"repository\":\""
        + native_resnet_repo
        + "\",\"create_repository\":true},\"parameters\":{\"input\":{"
          "\"connector\":\"image\",\"width\":224,\"height\":224,\"db\":true},"
          "\"mllib\":{\"nclasses\":2,\"gpu\":true,\"template\":\"resnet18\"}}"
          "}";
  std::string joutstr = japi.jrender(japi.service_create(sname, jstr));
  ASSERT_EQ(created_str, joutstr);

  // Train
  std::string jtrainstr
      = "{\"service\":\"imgserv\",\"async\":false,\"parameters\":{"
        "\"mllib\":{\"solver\":{\"iterations\":"
        + std::to_string(iterations_native)
        + ",\"base_lr\":1e-5,\"iter_size\":2,\"solver_type\":\"ADAM\",\"test_"
          "interval\":100},\"net\":{\"batch_size\":32},\"nclasses\":2,"
          "\"resume\":false},\"input\":{\"seed\":12345,\"db\":true,"
          "\"shuffle\":true,\"test_split\":0.1},\"output\":{\"measure\":["
          "\"f1\",\"acc\"]}},\"data\":[\""
        + resnet50_train_data + "\"]}";
  joutstr = japi.jrender(japi.service_train(jtrainstr));
  JDoc jd;
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(201, jd["status"]["code"]);

  ASSERT_TRUE(jd["body"]["measure"]["iteration"] == iterations_native)
      << "iterations";
  ASSERT_TRUE(jd["body"]["measure"]["acc"].GetDouble() <= 1) << "accuracy";
  // TODO test accuracy when it's no more random
  // ASSERT_TRUE(jd["body"]["measure"]["acc"].GetDouble() >= 0.6)
  //    << "accuracy good";
  ASSERT_TRUE(jd["body"]["measure"]["f1"].GetDouble() <= 1) << "f1";

  // clear directory
  jstr = "{\"clear\":\"full\"}";
  joutstr = japi.jrender(japi.service_delete(sname, jstr));
  ASSERT_EQ(ok_str, joutstr);
  fileops::remove_dir(native_resnet_repo);
}

TEST(torchapi, service_train_txt_lm)
{
  setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8", true);
  torch::manual_seed(torch_seed);
  at::globalContext().setDeterministicCuDNN(true);

  // create service
  JsonAPI japi;
  std::string sname = "txtserv";
  std::string jstr
      = "{\"mllib\":\"torch\",\"description\":\"bert\",\"type\":"
        "\"supervised\",\"model\":{\"repository\":\""
        + bert_train_repo
        + "\"},\"parameters\":{\"input\":{\"connector\":\"txt\",\"ordered_"
          "words\":true,"
          "\"wordpiece_tokens\":true,\"punctuation_tokens\":true,\"sequence\":"
          "512},\"mllib\":{\"template\":\"bert\","
          "\"self_supervised\":\"mask\",\"finetuning\":true,\"gpu\":true}}}";
  std::string joutstr = japi.jrender(japi.service_create(sname, jstr));
  ASSERT_EQ(created_str, joutstr);

  // train
  std::string jtrainstr
      = "{\"service\":\"txtserv\",\"async\":false,\"parameters\":{"
        "\"mllib\":{\"solver\":{\"iterations\":3,\"base_lr\":"
        + torch_lr
        + ",\"iter_"
          "size\":2,\"solver_type\":\"ADAM\"},\"net\":{\"batch_size\":2}},"
          "\"input\":{\"shuffle\":true,\"test_split\":0.25},"
          "\"output\":{\"measure\":[\"f1\",\"acc\"]}},\"data\":[\""
        + bert_train_data + "\"]}";
  joutstr = japi.jrender(japi.service_train(jtrainstr));
  JDoc jd;
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(201, jd["status"]["code"]);
  ASSERT_TRUE(jd["body"]["measure"]["iteration"] == 3) << "iterations";
  // This assertion is non-deterministic
  // ASSERT_TRUE(jd["body"]["measure"]["train_loss"].GetDouble() > 1.0) <<
  // "train_loss";
  ASSERT_TRUE(jd["body"]["measure"]["acc"].GetDouble() <= 1) << "accuracy";
  ASSERT_TRUE(jd["body"]["measure"]["f1"].GetDouble() <= 1) << "f1";

  std::unordered_set<std::string> lfiles;
  fileops::list_directory(bert_train_repo, true, false, false, lfiles);
  for (std::string ff : lfiles)
    {
      if (ff.find("checkpoint") != std::string::npos
          || ff.find("solver") != std::string::npos)
        remove(ff.c_str());
    }
  ASSERT_TRUE(!fileops::file_exists(bert_train_repo + "checkpoint-3.pt"));
  ASSERT_TRUE(!fileops::file_exists(bert_train_repo + "solver-3.pt"));
}

TEST(torchapi, service_train_txt_classification)
{
  setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8", true);
  torch::manual_seed(torch_seed);
  at::globalContext().setDeterministicCuDNN(true);

  // create service
  JsonAPI japi;
  std::string sname = "txtserv";
  std::string jstr = "{\"mllib\":\"torch\",\"description\":\"bert\",\"type\":"
                     "\"supervised\",\"model\":{\"repository\":\""
                     + bert_train_repo
                     + "\"},\"parameters\":{\"input\":{\"connector\":\"txt\","
                       "\"ordered_words\":true,"
                       "\"wordpiece_tokens\":true,\"punctuation_tokens\":true,"
                       "\"sequence\":512},\"mllib\":{\"template\":\"bert\","
                       "\"nclasses\":2,\"finetuning\":true,\"gpu\":true}}}";
  std::string joutstr = japi.jrender(japi.service_create(sname, jstr));
  ASSERT_EQ(created_str, joutstr);

  // train
  std::string jtrainstr
      = "{\"service\":\"txtserv\",\"async\":false,\"parameters\":{"
        "\"mllib\":{\"solver\":{\"iterations\":3,\"base_lr\":"
        + torch_lr
        + ",\"iter_"
          "size\":2,\"solver_type\":\"ADAM\"},\"net\":{\"batch_size\":2}},"
          "\"input\":{\"seed\":12345,\"shuffle\":true,\"test_split\":0.25},"
          "\"output\":{\"measure\":[\"f1\",\"acc\",\"mcll\",\"cmdiag\","
          "\"cmfull\"]}},\"data\":[\""
        + bert_train_data + "\"]}";
  joutstr = japi.jrender(japi.service_train(jtrainstr));
  JDoc jd;
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(201, jd["status"]["code"]);
  ASSERT_TRUE(abs(jd["body"]["measure"]["iteration"].GetDouble() - 3)
              < 0.00001)
      << "iterations";
  // This assertion is non-deterministic
  // ASSERT_TRUE(jd["body"]["measure"]["train_loss"].GetDouble() > 1.0) <<
  // "train_loss";
  ASSERT_TRUE(jd["body"]["measure"]["acc"].GetDouble() <= 1) << "accuracy";
  ASSERT_TRUE(jd["body"]["measure"]["f1"].GetDouble() <= 1) << "f1";

  std::unordered_set<std::string> lfiles;
  fileops::list_directory(bert_train_repo, true, false, false, lfiles);
  for (std::string ff : lfiles)
    {
      if (ff.find("checkpoint") != std::string::npos
          || ff.find("solver") != std::string::npos)
        remove(ff.c_str());
    }
  ASSERT_TRUE(!fileops::file_exists(bert_train_repo + "checkpoint-3.pt"));
  ASSERT_TRUE(!fileops::file_exists(bert_train_repo + "solver-3.pt"));
}

TEST(torchapi, service_train_txt_classification_nosplit)
{
  setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8", true);
  torch::manual_seed(torch_seed);
  at::globalContext().setDeterministicCuDNN(true);

  // create service
  JsonAPI japi;
  std::string sname = "txtserv";
  std::string jstr = "{\"mllib\":\"torch\",\"description\":\"bert\",\"type\":"
                     "\"supervised\",\"model\":{\"repository\":\""
                     + bert_train_repo
                     + "\"},\"parameters\":{\"input\":{\"connector\":\"txt\","
                       "\"ordered_words\":true,"
                       "\"wordpiece_tokens\":true,\"punctuation_tokens\":true,"
                       "\"sequence\":512},\"mllib\":{\"template\":\"bert\","
                       "\"nclasses\":2,\"finetuning\":true,\"gpu\":true}}}";
  std::string joutstr = japi.jrender(japi.service_create(sname, jstr));
  ASSERT_EQ(created_str, joutstr);

  // train
  std::string jtrainstr
      = "{\"service\":\"txtserv\",\"async\":false,\"parameters\":{"
        "\"mllib\":{\"solver\":{\"iterations\":3,\"base_lr\":"
        + torch_lr
        + ",\"iter_"
          "size\":2,\"solver_type\":\"ADAM\"},\"net\":{\"batch_size\":2}},"
          "\"input\":{\"seed\":12345,\"shuffle\":true},"
          "\"output\":{\"measure\":[\"f1\",\"acc\",\"mcll\",\"cmdiag\","
          "\"cmfull\"]}},\"data\":[\""
        + bert_train_data + "\",\"" + bert_train_data + "\"]}";
  joutstr = japi.jrender(japi.service_train(jtrainstr));
  JDoc jd;
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(201, jd["status"]["code"]);
  ASSERT_TRUE(abs(jd["body"]["measure"]["iteration"].GetDouble() - 3)
              < 0.00001)
      << "iterations";
  // This assertion is non-deterministic
  // ASSERT_TRUE(jd["body"]["measure"]["train_loss"].GetDouble() > 1.0) <<
  // "train_loss";
  ASSERT_TRUE(jd["body"]["measure"]["acc"].GetDouble() <= 1) << "accuracy";
  ASSERT_TRUE(jd["body"]["measure"]["f1"].GetDouble() <= 1) << "f1";

  std::unordered_set<std::string> lfiles;
  fileops::list_directory(bert_train_repo, true, false, false, lfiles);
  for (std::string ff : lfiles)
    {
      if (ff.find("checkpoint") != std::string::npos
          || ff.find("solver") != std::string::npos)
        remove(ff.c_str());
    }
  ASSERT_TRUE(!fileops::file_exists(bert_train_repo + "checkpoint-3.pt"));
  ASSERT_TRUE(!fileops::file_exists(bert_train_repo + "solver-3.pt"));
}

#endif

TEST(torchapi, service_train_csvts_nbeats)
{
  setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8", true);
  torch::manual_seed(torch_seed);
  at::globalContext().setDeterministicCuDNN(true);

  // create service
  JsonAPI japi;
  std::string sname = "nbeats";
  std::string csvts_data = sinus + "train";
  std::string csvts_test = sinus + "test";
  std::string csvts_predict = sinus + "predict";
  std::string csvts_nbeats_repo = "csvts_nbeats";
  mkdir(csvts_nbeats_repo.c_str(), 0777);

  std::string jstr
      = "{\"mllib\":\"torch\",\"description\":\"nbeats\",\"type\":"
        "\"supervised\",\"model\":{\"repository\":\""
        + csvts_nbeats_repo
        + "\"},\"parameters\":{\"input\":{\"connector\":\"csvts\",\"ignore\":["
          "\"output\"],\"backcast_timesteps\":50,\"forecast_timesteps\":50},"
          "\"mllib\":{\"template\":\"nbeats\",\"gpuid\":0,"
          "\"template_params\":{\"stackdef\":[\"t2\",\"s\",\"g3\",\"b3\"]},"
          "\"loss\":\"L1\"}}}";

  std::string joutstr = japi.jrender(japi.service_create(sname, jstr));
  ASSERT_EQ(created_str, joutstr);

  // train
  std::string jtrainstr
      = "{\"service\":\"" + sname
        + "\",\"async\":false,\"parameters\":{\"input\":{\"seed\":12345,"
          "\"shuffle\":true,"
          "\"separator\":\",\",\"scale\":true,\"backcast_timesteps\":50,"
          "\"forecast_timesteps\":50,\"ignore\":["
          "\"output\"]},\"mllib\":{\"gpu\":false,\"solver\":{\"iterations\":"
        + iterations_nbeats_cpu
        + ",\"test_interval\":10,\"base_lr\":0.1,\"snapshot\":500,\"test_"
          "initialization\":false,\"solver_type\":\"ADAM\"},\"net\":{\"batch_"
          "size\":2,\"test_batch_"
          "size\":10}},\"output\":{\"measure\":[\"L1_all\",\"L2\",\"mae_"
          "all\"]}},"
          "\"data\":[\""
        + csvts_data + "\",\"" + csvts_test + "\"]}";

  std::cerr << "jtrainstr=" << jtrainstr << std::endl;
  joutstr = japi.jrender(japi.service_train(jtrainstr));
  std::cout << "joutstr=" << joutstr << std::endl;
  JDoc jd;
  jd.Parse(joutstr.c_str());
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
  ASSERT_FALSE(jd["body"]["measure"].HasMember("L2_max_error_0"));
  ASSERT_TRUE(jd["body"]["measure"].HasMember("MAE_0"));
  // below mae should be twice normalized error because signal values are
  // between -1 and 1
  ASSERT_NEAR(jd["body"]["measure"]["MAE_0"].GetDouble(),
              jd["body"]["measure"]["L1_mean_error_0"].GetDouble() * 2.0,
              1E-5);
  ASSERT_TRUE(jd["body"]["parameters"]["input"].HasMember("max_vals"));
  ASSERT_TRUE(jd["body"]["parameters"]["input"].HasMember("min_vals"));

  std::string str_min_vals
      = japi.jrender(jd["body"]["parameters"]["input"]["min_vals"]);
  std::string str_max_vals
      = japi.jrender(jd["body"]["parameters"]["input"]["max_vals"]);

  //  predict
  std::string jpredictstr
      = "{\"service\":\"" + sname
        + "\",\"parameters\":{\"input\":{\"backcast_timesteps\":50,\"forecast_"
          "timesteps\":50,\"connector\":"
          "\"csvts\",\"scale\":true,\"ignore\":[\"output\"],\"continuation\":"
          "true,\"min_vals\":"
        + str_min_vals + ",\"max_vals\":" + str_max_vals
        + "},\"output\":{}},\"data\":[\"" + csvts_predict + "\"]}";
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200, jd["status"]["code"]);
  std::string uri = jd["body"]["predictions"][0]["uri"].GetString();
  ASSERT_EQ("../examples/all/sinus/predict/seq_2.csv #0_99", uri);
  ASSERT_TRUE(jd["body"]["predictions"][0]["series"].IsArray());
  ASSERT_TRUE(jd["body"]["predictions"][0]["series"][0]["out"][0].GetDouble()
              >= -1.5);

  // predict from memory
  std::stringstream mem_data;
  for (int i = 0; i < 50; ++i)
    {
      if (i != 0)
        mem_data << "\\n";
      mem_data << i << "," << i;
    }

  jpredictstr
      = "{\"service\":\"" + sname
        + "\",\"parameters\":{\"input\":{\"backcast_timesteps\":50,\"forecast_"
          "timesteps\":50,\"connector\":"
          "\"csvts\",\"scale\":true,\"ignore\":[\"output\"],\"continuation\":"
          "true,\"min_vals\":"
        + str_min_vals + ",\"max_vals\":" + str_max_vals
        + "},\"output\":{}},\"data\":[\"input,output\", \"" + mem_data.str()
        + "\"]}";

  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200, jd["status"]["code"]);
  uri = jd["body"]["predictions"][0]["uri"].GetString();
  ASSERT_EQ("0", uri);
  ASSERT_TRUE(jd["body"]["predictions"][0]["series"].IsArray());
  ASSERT_TRUE(jd["body"]["predictions"][0]["series"].Size() == 50);
  ASSERT_TRUE(jd["body"]["predictions"][0]["series"][0]["out"][0].IsDouble());

  //  remove service
  jstr = "{\"clear\":\"full\"}";
  joutstr = japi.jrender(japi.service_delete(sname, jstr));
  ASSERT_EQ(ok_str, joutstr);
  rmdir(csvts_nbeats_repo.c_str());
}

TEST(torchapi, service_train_csvts_nbeats_db)
{
  setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8", true);
  torch::manual_seed(torch_seed);
  at::globalContext().setDeterministicCuDNN(true);

  // create service
  JsonAPI japi;
  std::string sname = "nbeats";
  std::string csvts_data = sinus + "train";
  std::string csvts_test = sinus + "test";
  std::string csvts_predict = sinus + "predict";
  std::string csvts_nbeats_repo = "csvts_nbeats";
  mkdir(csvts_nbeats_repo.c_str(), 0777);

  std::string jstr
      = "{\"mllib\":\"torch\",\"description\":\"nbeats\",\"type\":"
        "\"supervised\",\"model\":{\"repository\":\""
        + csvts_nbeats_repo
        + "\"},\"parameters\":{\"input\":{\"db\":true,\"connector\":\"csvts\","
          "\"ignore\":["
          "\"output\"],\"backcast_timesteps\":50,\"forecast_timesteps\":50},"
          "\"mllib\":{\"template\":\"nbeats\","
          "\"template_params\":{\"stackdef\":[\"t2\",\"s\",\"g3\",\"b3\"]},"
          "\"loss\":\"L1\"}}}";

  std::string joutstr = japi.jrender(japi.service_create(sname, jstr));
  ASSERT_EQ(created_str, joutstr);

  // train
  std::string jtrainstr
      = "{\"service\":\"" + sname
        + "\",\"async\":false,\"parameters\":{\"input\":{\"db\":true,\"seed\":"
          "12345,\"shuffle\":true,"
          "\"separator\":\",\",\"scale\":true,\"backcast_timesteps\":50,"
          "\"forecast_timesteps\":50,\"ignore\":["
          "\"output\"]},\"mllib\":{\"gpu\":false,\"solver\":{\"iterations\":"
        + iterations_nbeats_cpu
        + ",\"test_interval\":10,\"base_lr\":0.1,\"snapshot\":500,\"test_"
          "initialization\":false,\"solver_type\":\"ADAM\"},\"net\":{\"batch_"
          "size\":2,\"test_batch_"
          "size\":10}},\"output\":{\"measure\":[\"L1_all\",\"L2_all\"]}},"
          "\"data\":[\""
        + csvts_data + "\",\"" + csvts_test + "\"]}";

  std::cerr << "jtrainstr=" << jtrainstr << std::endl;
  joutstr = japi.jrender(japi.service_train(jtrainstr));
  std::cout << "joutstr=" << joutstr << std::endl;
  JDoc jd;
  jd.Parse(joutstr.c_str());
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
        + "\",\"parameters\":{\"input\":{\"db\":false,\"backcast_timesteps\":"
          "50,\"forecast_"
          "timesteps\":50,\"connector\":"
          "\"csvts\",\"scale\":true,\"ignore\":[\"output\"],\"continuation\":"
          "true,\"min_vals\":"
        + str_min_vals + ",\"max_vals\":" + str_max_vals
        + "},\"output\":{}},\"data\":[\"" + csvts_predict + "\"]}";
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200, jd["status"]["code"]);
  std::string uri = jd["body"]["predictions"][0]["uri"].GetString();
  ASSERT_EQ("../examples/all/sinus/predict/seq_2.csv #0_99", uri);
  ASSERT_TRUE(jd["body"]["predictions"][0]["series"].IsArray());
  ASSERT_TRUE(jd["body"]["predictions"][0]["series"][0]["out"][0].GetDouble()
              >= -10.0);

  // predict from memory
  std::stringstream mem_data;
  for (int i = 0; i < 50; ++i)
    {
      if (i != 0)
        mem_data << "\\n";
      mem_data << i << "," << i;
    }

  jpredictstr = "{\"service\":\"" + sname
                + "\",\"parameters\":{\"input\":{\"db\":"
                  "false,\"backcast_timesteps\":"
                  "50,\"forecast_"
                  "timesteps\":50,\"connector\":"
                  "\"csvts\",\"scale\":true,"
                  "\"ignore\":[\"output\"],"
                  "\"continuation\":"
                  "true,\"min_vals\":"
                + str_min_vals + ",\"max_vals\":" + str_max_vals
                + "},\"output\":{}},\"data\":[\"input,output\", \""
                + mem_data.str() + "\"]}";

  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200, jd["status"]["code"]);
  uri = jd["body"]["predictions"][0]["uri"].GetString();
  ASSERT_EQ("0", uri);
  ASSERT_TRUE(jd["body"]["predictions"][0]["series"].IsArray());
  ASSERT_TRUE(jd["body"]["predictions"][0]["series"].Size() == 50);
  ASSERT_TRUE(jd["body"]["predictions"][0]["series"][0]["out"][0].IsDouble());

  //  remove service
  jstr = "{\"clear\":\"full\"}";
  joutstr = japi.jrender(japi.service_delete(sname, jstr));
  ASSERT_EQ(ok_str, joutstr);
  rmdir(csvts_nbeats_repo.c_str());
}

TEST(torchapi, service_train_csvts_nbeats_multiple_testsets)
{
  torch::manual_seed(torch_seed);
  at::globalContext().setDeterministicCuDNN(true);

  // create service
  JsonAPI japi;
  std::string sname = "nbeats";
  std::string csvts_data = sinus + "train";
  std::string csvts_test = sinus + "test";
  std::string csvts_predict = sinus + "predict";
  std::string csvts_nbeats_repo = "csvts_nbeats";
  mkdir(csvts_nbeats_repo.c_str(), 0777);

  std::string jstr
      = "{\"mllib\":\"torch\",\"description\":\"nbeats\",\"type\":"
        "\"supervised\",\"model\":{\"repository\":\""
        + csvts_nbeats_repo
        + "\"},\"parameters\":{\"input\":{\"connector\":\"csvts\",\"ignore\":["
          "\"output\"],\"backcast_timesteps\":50,\"forecast_timesteps\":50},"
          "\"mllib\":{\"template\":\"nbeats\","
          "\"template_params\":{\"stackdef\":[\"t2\",\"s\",\"g3\",\"b3\"]},"
          "\"loss\":\"L1\"}}}";

  std::string joutstr = japi.jrender(japi.service_create(sname, jstr));
  ASSERT_EQ(created_str, joutstr);

  std::string csvts_test2 = sinus + "test_2";
  ASSERT_EQ(symlink("./test", (sinus + "test_2").c_str()), 0);
  // train
  std::string jtrainstr
      = "{\"service\":\"" + sname
        + "\",\"async\":false,\"parameters\":{\"input\":{\"shuffle\":true,"
          "\"separator\":\",\",\"scale\":true,\"backcast_timesteps\":50,"
          "\"forecast_timesteps\":50,\"ignore\":["
          "\"output\"]},\"mllib\":{\"gpu\":false,\"solver\":{\"iterations\":10"
        + ",\"test_interval\":3,\"base_lr\":0.1,\"snapshot\":500,\"test_"
          "initialization\":false,\"solver_type\":\"ADAM\"},\"net\":{\"batch_"
          "size\":2,\"test_batch_"
          "size\":10}},\"output\":{\"measure\":[\"L1_all\",\"L2_all\"]}},"
          "\"data\":[\""
        + csvts_data + "\",\"" + csvts_test + "\",\"" + csvts_test2 + "\"]}";

  std::cerr << "jtrainstr=" << jtrainstr << std::endl;
  joutstr = japi.jrender(japi.service_train(jtrainstr));
  std::cout << "joutstr=" << joutstr << std::endl;
  JDoc jd;
  jd.Parse(joutstr.c_str());
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

  ASSERT_TRUE(jd["body"]["measures"].IsArray());
  ASSERT_EQ(jd["body"]["measures"].Size(), 2);

  //  remove service
  remove((sinus + "test_2").c_str());
  jstr = "{\"clear\":\"full\"}";
  joutstr = japi.jrender(japi.service_delete(sname, jstr));
  ASSERT_EQ(ok_str, joutstr);
  rmdir(csvts_nbeats_repo.c_str());
}

TEST(torchapi, service_train_csvts_nbeats_resume_fail)
{
  setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8", true);
  torch::manual_seed(torch_seed);
  at::globalContext().setDeterministicCuDNN(true);

  // create service
  JsonAPI japi;
  std::string sname = "nbeats";
  std::string csvts_data = sinus + "train";
  std::string csvts_test = sinus + "test";
  std::string csvts_predict = sinus + "predict";
  std::string csvts_nbeats_repo = "csvts_nbeats";
  mkdir(csvts_nbeats_repo.c_str(), 0777);

  std::string jstr
      = "{\"mllib\":\"torch\",\"description\":\"nbeats\",\"type\":"
        "\"supervised\",\"model\":{\"repository\":\""
        + csvts_nbeats_repo
        + "\"},\"parameters\":{\"input\":{\"connector\":\"csvts\",\"ignore\":["
          "\"output\"],\"backcast_timesteps\":50,\"forecast_timesteps\":50},"
          "\"mllib\":{\"template\":\"nbeats\","
          "\"template_params\":{\"stackdef\":[\"t2\",\"s\",\"g3\",\"b3\"]},"
          "\"loss\":\"L1\"}}}";

  std::string joutstr = japi.jrender(japi.service_create(sname, jstr));
  ASSERT_EQ(created_str, joutstr);

  // train
  std::string jtrainstr
      = "{\"service\":\"" + sname
        + "\",\"async\":false,\"parameters\":{\"input\":{\"seed\":12345,"
          "\"shuffle\":true,"
          "\"separator\":\",\",\"scale\":true,\"timesteps\":50,\"ignore\":["
          "\"output\"]},\"mllib\":{\"resume\":true,\"gpu\":false,\"solver\":{"
          "\"iterations\":"
        + iterations_nbeats_cpu
        + ",\"test_interval\":10,\"base_lr\":0.1,\"snapshot\":500,\"test_"
          "initialization\":false,\"solver_type\":\"ADAM\"},\"net\":{\"batch_"
          "size\":2,\"test_batch_"
          "size\":10}},\"output\":{\"measure\":[\"L1\",\"L2\"]}},\"data\":[\""
        + csvts_data + "\",\"" + csvts_test + "\"]}";

  std::cerr << "jtrainstr=" << jtrainstr << std::endl;
  joutstr = japi.jrender(japi.service_train(jtrainstr));
  std::cout << "joutstr=" << joutstr << std::endl;
  JDoc jd;
  jd.Parse(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_TRUE(jd.HasMember("status"));
  ASSERT_EQ(400, jd["status"]["code"].GetInt());
  ASSERT_EQ("Service Bad Request Error: resuming a model requires a "
            "solverstate (solver-xxx.pt) file in model repository",
            jd["status"]["dd_msg"]);
  //  remove service
  jstr = "{\"clear\":\"full\"}";
  joutstr = japi.jrender(japi.service_delete(sname, jstr));
  ASSERT_EQ(ok_str, joutstr);
  rmdir(csvts_nbeats_repo.c_str());
}

TEST(torchapi, service_train_csvts_nbeats_forecast)
{
  setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8", true);
  torch::manual_seed(torch_seed);
  at::globalContext().setDeterministicCuDNN(true);

  // create service
  JsonAPI japi;
  std::string sname = "nbeats";
  std::string csvts_data = sinus + "train";
  std::string csvts_test = sinus + "test";
  std::string csvts_predict = sinus + "predict";
  std::string csvts_nbeats_repo = "csvts_nbeats";
  mkdir(csvts_nbeats_repo.c_str(), 0777);

  std::string jstr
      = "{\"mllib\":\"torch\",\"description\":\"nbeats\",\"type\":"
        "\"supervised\",\"model\":{\"repository\":\""
        + csvts_nbeats_repo
        + "\"},\"parameters\":{\"input\":{\"connector\":\"csvts\","
          "\"forecast_timesteps\":10,\"ignore\":[\"output\"],\"backcast_"
          "timesteps\":40},"
          "\"mllib\":"
          "{\"template\":"
          "\"nbeats\","
          "\"template_params\":{\"stackdef\":[\"t2\",\"s\",\"g3\",\"b3\"]},"
          "\"loss\":\"L1\"}}}";

  std::string joutstr = japi.jrender(japi.service_create(sname, jstr));
  ASSERT_EQ(created_str, joutstr);

  // train
  std::string jtrainstr
      = "{\"service\":\"" + sname
        + "\",\"async\":false,\"parameters\":{\"input\":{\"seed\":12345,"
          "\"shuffle\":true,"
          "\"separator\":\",\",\"scale\":true,\"backcast_timesteps\":40,"
          "\"forecast_timesteps\":"
          "10,\"ignore\":[\"output\"]},\"mllib\":{\"gpu\":false,\"solver\":{"
          "\"iterations\":"
        + iterations_nbeats_cpu
        + ",\"test_interval\":10,\"base_lr\":0.1,\"snapshot\":500,\"test_"
          "initialization\":false,\"solver_type\":\"ADAM\"},\"net\":{\"batch_"
          "size\":2,\"test_batch_"
          "size\":10}},\"output\":{\"measure\":[\"L1_all\",\"L2_all\"]}},"
          "\"data\":[\""
        + csvts_data + "\",\"" + csvts_test + "\"]}";

  std::cerr << "jtrainstr=" << jtrainstr << std::endl;
  joutstr = japi.jrender(japi.service_train(jtrainstr));
  std::cout << "joutstr=" << joutstr << std::endl;
  JDoc jd;
  jd.Parse(joutstr.c_str());
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
  std::string jpredictstr = "{\"service\":\"" + sname
                            + "\",\"parameters\":{\"input\":{\"backcast_"
                              "timesteps\":40,\"connector\":"
                              "\"csvts\",\"scale\":true,\"forecast_"
                              "timesteps\":10,\"ignore\":[\"output\"],"
                              "\"continuation\":"
                              "true,\"min_vals\":"
                            + str_min_vals + ",\"max_vals\":" + str_max_vals
                            + "},\"output\":{}},\"data\":[\"" + csvts_predict
                            + "\"]}";
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200, jd["status"]["code"]);
  std::string uri = jd["body"]["predictions"][0]["uri"].GetString();
  ASSERT_EQ("../examples/all/sinus/predict/seq_2.csv #0_49", uri);
  ASSERT_TRUE(jd["body"]["predictions"][0]["series"].IsArray());
  ASSERT_EQ(jd["body"]["predictions"][0]["series"].Size(), 10);
  ASSERT_TRUE(jd["body"]["predictions"][0]["series"][0]["out"][0].GetDouble()
              >= -1.5);

  //  remove service
  jstr = "{\"clear\":\"full\"}";
  joutstr = japi.jrender(japi.service_delete(sname, jstr));
  ASSERT_EQ(ok_str, joutstr);
  rmdir(csvts_nbeats_repo.c_str());
}

TEST(torchapi, service_train_ttransformer_forecast)
{
  setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8", true);
  torch::manual_seed(torch_seed);
  at::globalContext().setDeterministicCuDNN(true);

  // create service
  JsonAPI japi;
  std::string sname = "ttransformer";
  std::string csvts_data = sinus + "train";
  std::string csvts_test = sinus + "test";
  std::string csvts_predict = sinus + "predict";
  std::string csvts_nbeats_repo = "ttransformer";
  mkdir(csvts_nbeats_repo.c_str(), 0777);

  std::string jstr
      = "{\"mllib\":\"torch\",\"description\":\"ttransformer\",\"type\":"
        "\"supervised\",\"model\":{\"repository\":\""
        + csvts_nbeats_repo
        + "\"},\"parameters\":{\"input\":{\"connector\":\"csvts\","
          "\"forecast_timesteps\":10,\"ignore\":[\"output\"],\"backcast_"
          "timesteps\":40},\"mllib\":"
          "{\"template\":\"ttransformer\",\"template_params\":{\"embed\": "
          "{\"layers\": 2,\"activation\": "
          "\"relu\",\"dim\": 2,\"type\": \"step\",\"dropout\": "
          "0.0},\"encoder\":{\"heads\": 1,\"layers\": 2,\"hidden_dim\": "
          "2,\"dropout\": 0.0},\"positional_encoding\":{\"type\": "
          "\"naive\",\"learn\": false,\"dropout\": "
          "0.0},\"decoder\":{\"type\": "
          "\"simple\",\"dropout\": 0.0,\"layers\": 1}},"
          "\"loss\":\"L1\"}}}";

  std::string joutstr = japi.jrender(japi.service_create(sname, jstr));
  ASSERT_EQ(created_str, joutstr);

  // train
  std::string jtrainstr
      = "{\"service\":\"" + sname
        + "\",\"async\":false,\"parameters\":{\"input\":{\"seed\":12345,"
          "\"shuffle\":true,"
          "\"separator\":\",\",\"scale\":true,\"offset\":10,\"backcast_"
          "timesteps\":40,\"forecast_timesteps\":"
          "10,\"ignore\":[\"output\"]},\"mllib\":{\"gpu\":false,\"solver\":{"
          "\"iterations\":"
        + iterations_ttransformer_cpu
        + ",\"test_interval\":10,\"base_lr\":0.1,\"snapshot\":500,\"test_"
          "initialization\":false,\"solver_type\":\"ADAM\"},\"net\":{\"batch_"
          "size\":2,\"test_batch_"
          "size\":10}},\"output\":{\"measure\":[\"L1_all\",\"L2_all\"]}},"
          "\"data\":[\""
        + csvts_data + "\",\"" + csvts_test + "\"]}";

  std::cerr << "jtrainstr=" << jtrainstr << std::endl;
  joutstr = japi.jrender(japi.service_train(jtrainstr));
  std::cout << "joutstr=" << joutstr << std::endl;
  JDoc jd;
  jd.Parse(joutstr.c_str());
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
  std::string jpredictstr = "{\"service\":\"" + sname
                            + "\",\"parameters\":{\"input\":{\"backcast_"
                              "timesteps\":40,\"connector\":"
                              "\"csvts\",\"scale\":true,\"forecast_"
                              "timesteps\":10,\"ignore\":[\"output\"],"
                              "\"continuation\":"
                              "true,\"min_vals\":"
                            + str_min_vals + ",\"max_vals\":" + str_max_vals
                            + "},\"output\":{}},\"data\":[\"" + csvts_predict
                            + "\"]}";
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200, jd["status"]["code"]);
  std::string uri = jd["body"]["predictions"][0]["uri"].GetString();
  ASSERT_EQ("../examples/all/sinus/predict/seq_2.csv #0_49", uri);
  ASSERT_TRUE(jd["body"]["predictions"][0]["series"].IsArray());
  ASSERT_EQ(jd["body"]["predictions"][0]["series"].Size(), 10);
  ASSERT_TRUE(jd["body"]["predictions"][0]["series"][0]["out"][0].GetDouble()
              >= -1.5);

  //  remove service
  jstr = "{\"clear\":\"full\"}";
  joutstr = japi.jrender(japi.service_delete(sname, jstr));
  ASSERT_EQ(ok_str, joutstr);
  rmdir(csvts_nbeats_repo.c_str());
}

#if !defined(CPU_ONLY)

TEST(torchapi, service_train_csvts_nbeats_gpu)
{
  setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8", true);
  torch::manual_seed(torch_seed);
  at::globalContext().setDeterministicCuDNN(true);

  // create service
  JsonAPI japi;
  std::string sname = "nbeats";
  std::string csvts_data = sinus + "train";
  std::string csvts_test = sinus + "test";
  std::string csvts_predict = sinus + "predict";
  std::string csvts_nbeats_repo = "csvts_nbeats";
  mkdir(csvts_nbeats_repo.c_str(), 0777);

  std::string jstr
      = "{\"mllib\":\"torch\",\"description\":\"nbeats\",\"type\":"
        "\"supervised\",\"model\":{\"repository\":\""
        + csvts_nbeats_repo
        + "\"},\"parameters\":{\"input\":{\"connector\":\"csvts\",\"ignore\":["
          "\"output\"],\"backcast_timesteps\":50,\"forecast_timesteps\":50},"
          "\"mllib\":{\"template\":\"nbeats\","
          "\"template_params\":{\"stackdef\":[\"t2\",\"s4\",\"g3\",\"b3\"]},"
          "\"loss\":\"L1\",\"gpu\":true,\"gpuid\":0}}}";

  std::string joutstr = japi.jrender(japi.service_create(sname, jstr));
  ASSERT_EQ(created_str, joutstr);

  // train
  std::string jtrainstr
      = "{\"service\":\"" + sname
        + "\",\"async\":false,\"parameters\":{\"input\":{\"seed\":12345,"
          "\"shuffle\":true,"
          "\"separator\":\",\",\"scale\":true,\"backcast_timesteps\":50,"
          "\"forecast_timesteps\":50,\"ignore\":["
          "\"output\"]},\"mllib\":{\"gpu\":true,\"gpuid\":0,\"solver\":{"
          "\"iterations\":"
        + iterations_nbeats_gpu
        + ",\"test_interval\":100,\"base_lr\":0.001,\"snapshot\":500,\"test_"
          "initialization\":false,\"solver_type\":\"ADAM\"},\"net\":{\"batch_"
          "size\":2,\"test_batch_"
          "size\":10}},\"output\":{\"measure\":[\"L1_all\",\"L2_all\"]}},"
          "\"data\":[\""
        + csvts_data + "\",\"" + csvts_test + "\"]}";

  std::cerr << "jtrainstr=" << jtrainstr << std::endl;
  joutstr = japi.jrender(japi.service_train(jtrainstr));
  std::cout << "joutstr=" << joutstr << std::endl;
  JDoc jd;
  jd.Parse(joutstr.c_str());
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
        + "\",\"parameters\":{\"input\":{\"backcast_timesteps\":50,\"forecast_"
          "timesteps\":50,\"connector\":"
          "\"csvts\",\"scale\":true,\"ignore\":[\"output\"],\"min_vals\":"
        + str_min_vals + ",\"max_vals\":" + str_max_vals
        + "},\"output\":{}},\"data\":[\"" + csvts_predict + "\"]}";
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200, jd["status"]["code"]);
  std::string uri = jd["body"]["predictions"][0]["uri"].GetString();
  ASSERT_EQ("../examples/all/sinus/predict/seq_2.csv #0_99", uri);
  ASSERT_TRUE(jd["body"]["predictions"][0]["series"].IsArray());
  ASSERT_TRUE(jd["body"]["predictions"][0]["series"][0]["out"][0].GetDouble()
              >= -1.5);

  joutstr = japi.jrender(japi.service_status(sname));
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse(joutstr.c_str());
  ASSERT_TRUE(jd["body"]["service_stats"].HasMember("predict_count"));
  ASSERT_TRUE(jd["body"]["service_stats"]["inference_count"].GetInt() == 10);
  ASSERT_TRUE(jd["body"]["service_stats"]["predict_count"].GetInt() == 1);
  ASSERT_TRUE(jd["body"]["service_stats"]["predict_failure"].GetInt() == 0);
  ASSERT_TRUE(jd["body"]["service_stats"]["predict_success"].GetInt() == 1);
  ASSERT_TRUE(jd["body"]["service_stats"]["avg_batch_size"].GetDouble()
              == 10.0);
  ASSERT_TRUE(jd["body"]["service_stats"]["avg_transform_duration"].GetDouble()
              > 0.0);
  ASSERT_TRUE(jd["body"]["service_stats"]["avg_predict_duration"].GetDouble()
              > 0.0);

  //  remove service
  jstr = "{\"clear\":\"full\"}";
  joutstr = japi.jrender(japi.service_delete(sname, jstr));
  ASSERT_EQ(ok_str, joutstr);
  rmdir(csvts_nbeats_repo.c_str());
}

TEST(torchapi, service_train_csvts_nbeats_multigpu)
{
  setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8", true);
  torch::manual_seed(torch_seed);
  at::globalContext().setDeterministicCuDNN(true);

  // create service
  JsonAPI japi;
  std::string sname = "nbeats";
  std::string csvts_data = sinus + "train";
  std::string csvts_test = sinus + "test";
  std::string csvts_predict = sinus + "predict";
  std::string csvts_nbeats_repo = "csvts_nbeats";
  mkdir(csvts_nbeats_repo.c_str(), 0777);

  std::string jstr
      = "{\"mllib\":\"torch\",\"description\":\"nbeats\",\"type\":"
        "\"supervised\",\"model\":{\"repository\":\""
        + csvts_nbeats_repo
        + "\"},\"parameters\":{\"input\":{\"connector\":\"csvts\",\"ignore\":["
          "\"output\"],\"backcast_timesteps\":50,\"forecast_timesteps\":50},"
          "\"mllib\":{\"template\":\"nbeats\","
          "\"template_params\":{\"stackdef\":[\"t2\",\"s4\",\"g3\",\"b3\"]},"
          "\"loss\":\"L1\",\"gpu\":true,\"gpuid\":[0,1]}}}";

  std::string joutstr = japi.jrender(japi.service_create(sname, jstr));
  ASSERT_EQ(created_str, joutstr);

  // train
  std::string jtrainstr
      = "{\"service\":\"" + sname
        + "\",\"async\":false,\"parameters\":{\"input\":{\"seed\":12345,"
          "\"shuffle\":true,"
          "\"separator\":\",\",\"scale\":true,\"backcast_timesteps\":50,"
          "\"forecast_timesteps\":50,\"ignore\":["
          "\"output\"]},\"mllib\":{\"gpu\":true,\"gpuid\":[0, 1],\"solver\":{"
          "\"iterations\":"
        + iterations_nbeats_gpu
        + ",\"test_interval\":100,\"base_lr\":0.001,\"snapshot\":500,\"test_"
          "initialization\":false,\"solver_type\":\"ADAM\"},\"net\":{\"batch_"
          "size\":2,\"test_batch_"
          "size\":10}},\"output\":{\"measure\":[\"L1_all\",\"L2_all\"]}},"
          "\"data\":[\""
        + csvts_data + "\",\"" + csvts_test + "\"]}";

  std::cerr << "jtrainstr=" << jtrainstr << std::endl;
  joutstr = japi.jrender(japi.service_train(jtrainstr));
  std::cout << "joutstr=" << joutstr << std::endl;
  JDoc jd;
  jd.Parse(joutstr.c_str());
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
        + "\",\"parameters\":{\"input\":{\"backcast_timesteps\":50,\"forecast_"
          "timesteps\":50,\"connector\":"
          "\"csvts\",\"scale\":true,\"ignore\":[\"output\"],\"min_vals\":"
        + str_min_vals + ",\"max_vals\":" + str_max_vals
        + "},\"output\":{}},\"data\":[\"" + csvts_predict + "\"]}";
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200, jd["status"]["code"]);
  std::string uri = jd["body"]["predictions"][0]["uri"].GetString();
  ASSERT_EQ("../examples/all/sinus/predict/seq_2.csv #0_99", uri);
  ASSERT_TRUE(jd["body"]["predictions"][0]["series"].IsArray());
  ASSERT_TRUE(jd["body"]["predictions"][0]["series"][0]["out"][0].GetDouble()
              >= -1.5);

  joutstr = japi.jrender(japi.service_status(sname));
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse(joutstr.c_str());
  ASSERT_TRUE(jd["body"]["service_stats"].HasMember("predict_count"));
  ASSERT_TRUE(jd["body"]["service_stats"]["inference_count"].GetInt() == 10);
  ASSERT_TRUE(jd["body"]["service_stats"]["predict_count"].GetInt() == 1);
  ASSERT_TRUE(jd["body"]["service_stats"]["predict_failure"].GetInt() == 0);
  ASSERT_TRUE(jd["body"]["service_stats"]["predict_success"].GetInt() == 1);
  ASSERT_TRUE(jd["body"]["service_stats"]["avg_batch_size"].GetDouble()
              == 10.0);
  ASSERT_TRUE(jd["body"]["service_stats"]["avg_transform_duration"].GetDouble()
              > 0.0);
  ASSERT_TRUE(jd["body"]["service_stats"]["avg_predict_duration"].GetDouble()
              > 0.0);

  //  remove service
  jstr = "{\"clear\":\"full\"}";
  joutstr = japi.jrender(japi.service_delete(sname, jstr));
  ASSERT_EQ(ok_str, joutstr);
  rmdir(csvts_nbeats_repo.c_str());
}

TEST(torchapi, service_train_resnet18_multigpu)
{
  setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8", true);
  torch::manual_seed(torch_seed);
  at::globalContext().setDeterministicCuDNN(true);

  // Create service
  JsonAPI japi;

  std::string native_resnet_repo = "native_resnet";
  int iterations_native = 100;
  mkdir(native_resnet_repo.c_str(), 0777);

  std::string sname = "imgserv";
  std::string jstr
      = "{\"mllib\":\"torch\",\"description\":\"image\",\"type\":"
        "\"supervised\",\"model\":{\"repository\":\""
        + native_resnet_repo
        + "\",\"create_repository\":true},\"parameters\":{\"input\":{"
          "\"connector\":\"image\",\"width\":224,\"height\":224,\"db\":true},"
          "\"mllib\":{\"nclasses\":2,\"gpu\":true,\"gpuid\":[0,1],"
          "\"template\":\"resnet18\"}}}";
  std::string joutstr = japi.jrender(japi.service_create(sname, jstr));
  ASSERT_EQ(created_str, joutstr);

  // Train
  std::string jtrainstr
      = "{\"service\":\"imgserv\",\"async\":false,\"parameters\":{"
        "\"mllib\":{\"solver\":{\"iterations\":"
        + std::to_string(iterations_native)
        + ",\"base_lr\":1e-5,\"iter_size\":1,\"solver_type\":\"ADAM\",\"test_"
          "interval\":100},\"net\":{\"batch_size\":32},\"nclasses\":2,"
          "\"resume\":false},\"input\":{\"seed\":12345,\"db\":true,"
          "\"shuffle\":true,\"test_split\":0.1},\"output\":{\"measure\":["
          "\"f1\",\"acc\"]}},\"data\":[\""
        + resnet50_train_data + "\"]}";
  joutstr = japi.jrender(japi.service_train(jtrainstr));
  JDoc jd;
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(201, jd["status"]["code"]);

  ASSERT_TRUE(jd["body"]["measure"]["iteration"] == iterations_native)
      << "iterations";
  ASSERT_TRUE(jd["body"]["measure"]["acc"].GetDouble() <= 1) << "accuracy";
  // TODO test accuracy when it's no more random
  // ASSERT_TRUE(jd["body"]["measure"]["acc"].GetDouble() >= 0.6)
  //    << "accuracy good";
  ASSERT_TRUE(jd["body"]["measure"]["f1"].GetDouble() <= 1) << "f1";

  // clear directory
  jstr = "{\"clear\":\"full\"}";
  joutstr = japi.jrender(japi.service_delete(sname, jstr));
  ASSERT_EQ(ok_str, joutstr);
  fileops::remove_dir(native_resnet_repo);
}
#endif

TEST(torchapi, nbeats_extract_layers_simple)
{
  std::vector<std::string> stackdef = { "s2", "t1", "g3", "b2" };
  NBeats nb(stackdef);

  std::string all_els;
  std::vector<std::string> els = nb.extractable_layers();
  for (auto el : els)
    all_els += el + " ";
  ASSERT_EQ(all_els,
            "0:0:fc1 0:0:fc2 0:0:fc3 0:0:fc4 0:0:theta_f_fc 0:0:end 0:1:fc1 "
            "0:1:fc2 0:1:fc3 0:1:fc4 0:1:theta_f_fc 0:1:end 1:0:fc1 1:0:fc2 "
            "1:0:fc3 1:0:fc4 1:0:theta_f_fc 1:0:end 1:1:fc1 1:1:fc2 1:1:fc3 "
            "1:1:fc4 1:1:theta_f_fc 1:1:end 2:0:fc1 2:0:fc2 2:0:fc3 2:0:fc4 "
            "2:0:theta_f_fc 2:0:theta_b_fc 2:0:backcast_fc 2:0:forecast_fc "
            "2:0:end 2:1:fc1 2:1:fc2 2:1:fc3 2:1:fc4 2:1:theta_f_fc "
            "2:1:theta_b_fc 2:1:backcast_fc 2:1:forecast_fc 2:1:end ");

  ASSERT_TRUE(nb.extractable("1:1:fc3"));
  ASSERT_TRUE(nb.extractable("2:0:end"));

  torch::Tensor x = torch::randn({ 2, 500, 1 });
  torch::Tensor y = nb.forward(x);
  ASSERT_EQ(y.sizes(), std::vector<long int>({ 2, 550, 1 }));
  torch::Tensor z = nb.extract(x, "2:0:fc1");
  ASSERT_EQ(z.sizes(), std::vector<long int>({ 2, 10 }));
  torch::Tensor t = nb.extract(x, "1:1:end");
  std::cout << t << std::endl;
}

TEST(torchapi, nbeats_extract_layer_complete)
{
  setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8", true);
  torch::manual_seed(torch_seed);
  at::globalContext().setDeterministicCuDNN(true);

  // create service
  JsonAPI japi;
  std::string sname = "nbeats";
  std::string csvts_data = sinus + "train";
  std::string csvts_test = sinus + "test";
  std::string csvts_predict = sinus + "predict";
  std::string csvts_nbeats_repo = "csvts_nbeats";
  mkdir(csvts_nbeats_repo.c_str(), 0777);

  std::string jstr
      = "{\"mllib\":\"torch\",\"description\":\"nbeats\",\"type\":"
        "\"supervised\",\"model\":{\"repository\":\""
        + csvts_nbeats_repo
        + "\"},\"parameters\":{\"input\":{\"connector\":\"csvts\",\"ignore\":["
          "\"output\"],\"forecast_timesteps\":50,\"backcast_timesteps\":50},"
          "\"mllib\":{\"template\":\"nbeats\","
          "\"template_params\":{\"stackdef\":[\"t2\",\"s4\",\"g3\",\"b3\"]},"
          "\"loss\":\"L1\"}}}";

  std::string joutstr = japi.jrender(japi.service_create(sname, jstr));
  ASSERT_EQ(created_str, joutstr);

  // train
  std::string jtrainstr
      = "{\"service\":\"" + sname
        + "\",\"async\":false,\"parameters\":{\"input\":{\"seed\":12345,"
          "\"shuffle\":true,"
          "\"separator\":\",\",\"scale\":true,\"backcast_timesteps\":50,"
          "\"forecast_timesteps\":50,\"ignore\":["
          "\"output\"]},\"mllib\":{\"gpu\":false,\"solver\":{\"iterations\":"
        + iterations_nbeats_cpu
        + ",\"test_interval\":10,\"base_lr\":0.1,\"snapshot\":500,\"test_"
          "initialization\":false,\"solver_type\":\"ADAM\"},\"net\":{\"batch_"
          "size\":2,\"test_batch_"
          "size\":10}},\"output\":{\"measure\":[\"L1_all\",\"L2_all\"]}},"
          "\"data\":[\""
        + csvts_data + "\",\"" + csvts_test + "\"]}";

  std::cerr << "jtrainstr=" << jtrainstr << std::endl;
  joutstr = japi.jrender(japi.service_train(jtrainstr));
  std::cout << "joutstr=" << joutstr << std::endl;
  JDoc jd;
  jd.Parse(joutstr.c_str());
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

  //  extract
  std::string jpredictstr
      = "{\"service\":\"" + sname
        + "\",\"parameters\":{\"mllib\":{\"extract_layer\":\"2:0:fc3\"},"
          "\"input\":{\"backcast_timesteps\":50,\"forecast_timesteps\":50,"
          "\"connector\":"
          "\"csvts\",\"scale\":true,\"ignore\":[\"output\"],\"continuation\":"
          "true,\"min_vals\":"
        + str_min_vals + ",\"max_vals\":" + str_max_vals
        + "},\"output\":{}},\"data\":[\"" + csvts_predict + "\"]}";
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200, jd["status"]["code"]);
  std::string uri = jd["body"]["predictions"][0]["uri"].GetString();
  ASSERT_EQ("../examples/all/sinus/predict/seq_2.csv #0_99", uri);
  ASSERT_TRUE(jd["body"]["predictions"][0]["vals"].IsArray());
  ASSERT_EQ(jd["body"]["predictions"].Size(), 10);
  ASSERT_EQ(jd["body"]["predictions"][0]["vals"].Size(), 10);

  //  remove service
  jstr = "{\"clear\":\"full\"}";
  joutstr = japi.jrender(japi.service_delete(sname, jstr));
  ASSERT_EQ(ok_str, joutstr);
  rmdir(csvts_nbeats_repo.c_str());
}

#ifndef CPU_ONLY
TEST(torchapi, nbeats_extract_layer_complete_gpu)
{
  setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8", true);
  torch::manual_seed(torch_seed);
  at::globalContext().setDeterministicCuDNN(true);

  // create service
  JsonAPI japi;
  std::string sname = "nbeats";
  std::string csvts_data = sinus + "train";
  std::string csvts_test = sinus + "test";
  std::string csvts_predict = sinus + "predict";
  std::string csvts_nbeats_repo = "csvts_nbeats";
  mkdir(csvts_nbeats_repo.c_str(), 0777);

  std::string jstr
      = "{\"mllib\":\"torch\",\"description\":\"nbeats\",\"type\":"
        "\"supervised\",\"model\":{\"repository\":\""
        + csvts_nbeats_repo
        + "\"},\"parameters\":{\"input\":{\"connector\":\"csvts\",\"ignore\":["
          "\"output\"],\"forecast_timesteps\":50,\"backcast_timesteps\":50},"
          "\"mllib\":{\"template\":\"nbeats\","
          "\"template_params\":{\"stackdef\":[\"t2\",\"s4\",\"g3\",\"b3\"]},"
          "\"loss\":\"L1\"}}}";

  std::string joutstr = japi.jrender(japi.service_create(sname, jstr));
  ASSERT_EQ(created_str, joutstr);

  // train
  std::string jtrainstr
      = "{\"service\":\"" + sname
        + "\",\"async\":false,\"parameters\":{\"input\":{\"seed\":12345,"
          "\"shuffle\":true,"
          "\"separator\":\",\",\"scale\":true,\"backcast_timesteps\":50,"
          "\"forecast_timesteps\":50,\"ignore\":["
          "\"output\"]},\"mllib\":{\"gpu\":true,\"gpuid\":0,\"solver\":{"
          "\"iterations\":"
        + iterations_nbeats_cpu
        + ",\"test_interval\":10,\"base_lr\":0.1,\"snapshot\":500,\"test_"
          "initialization\":false,\"solver_type\":\"ADAM\"},\"net\":{\"batch_"
          "size\":2,\"test_batch_"
          "size\":10}},\"output\":{\"measure\":[\"L1_all\",\"L2_all\"]}},"
          "\"data\":[\""
        + csvts_data + "\",\"" + csvts_test + "\"]}";

  std::cerr << "jtrainstr=" << jtrainstr << std::endl;
  joutstr = japi.jrender(japi.service_train(jtrainstr));
  std::cout << "joutstr=" << joutstr << std::endl;
  JDoc jd;
  jd.Parse(joutstr.c_str());
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

  //  extract
  std::string jpredictstr
      = "{\"service\":\"" + sname
        + "\",\"parameters\":{\"mllib\":{\"extract_layer\":\"2:0:fc3\"},"
          "\"input\":{\"backcast_timesteps\":50,\"forecast_timesteps\":50,"
          "\"connector\":"
          "\"csvts\",\"scale\":true,\"ignore\":[\"output\"],\"continuation\":"
          "true,\"min_vals\":"
        + str_min_vals + ",\"max_vals\":" + str_max_vals
        + "},\"output\":{}},\"data\":[\"" + csvts_predict + "\"]}";
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200, jd["status"]["code"]);
  std::string uri = jd["body"]["predictions"][0]["uri"].GetString();
  ASSERT_EQ("../examples/all/sinus/predict/seq_2.csv #0_99", uri);
  ASSERT_TRUE(jd["body"]["predictions"][0]["vals"].IsArray());
  ASSERT_EQ(jd["body"]["predictions"].Size(), 10);
  ASSERT_EQ(jd["body"]["predictions"][0]["vals"].Size(), 10);

  //  remove service
  jstr = "{\"clear\":\"full\"}";
  joutstr = japi.jrender(japi.service_delete(sname, jstr));
  ASSERT_EQ(ok_str, joutstr);
  rmdir(csvts_nbeats_repo.c_str());
}
#endif

TEST(torchapi, image_extract)
{
  // create service
  JsonAPI japi;
  std::string sname = "imgserv";
  std::string jstr
      = "{\"mllib\":\"torch\",\"description\":\"resnet-50\",\"type\":"
        "\"supervised\",\"model\":{\"repository\":\""
        + incept_repo
        + "\"},\"parameters\":{\"input\":{\"connector\":\"image\",\"height\":"
          "224,\"width\":224,\"rgb\":true,\"scale\":0.0039}}}";
  std::string joutstr = japi.jrender(japi.service_create(sname, jstr));
  ASSERT_EQ(created_str, joutstr);

  // predict
  std::string jpredictstr
      = "{\"service\":\"imgserv\",\"parameters\":{\"input\":{\"height\":224,"
        "\"width\":224,\"rgb\":true,\"scale\":0.0039},\"mllib\":{\"extract_"
        "layer\":\"final\"}}"
        ",\"data\":[\""
        + incept_repo + "cat.jpg\"]}";
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  JDoc jd;
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200, jd["status"]["code"]);
  ASSERT_TRUE(jd["body"]["predictions"].IsArray());
  ASSERT_TRUE(jd["body"]["predictions"][0]["vals"].IsArray());
  ASSERT_EQ(jd["body"]["predictions"][0]["vals"].Size(), 1000);
}

#ifndef CPU_ONLY
TEST(torchapi, service_train_ranger)
{
  setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8", true);
  torch::manual_seed(torch_seed);
  at::globalContext().setDeterministicCuDNN(true);

  // Create service
  JsonAPI japi;
  std::string sname = "imgserv";
  std::string jstr
      = "{\"mllib\":\"torch\",\"description\":\"image\",\"type\":"
        "\"supervised\",\"model\":{\"repository\":\""
        + resnet50_train_repo
        + "\"},\"parameters\":{\"input\":{\"connector\":\"image\","
          "\"width\":224,\"height\":224,\"db\":true},\"mllib\":{\"nclasses\":"
          "2,\"finetuning\":true,\"gpu\":true}}}";
  std::string joutstr = japi.jrender(japi.service_create(sname, jstr));
  ASSERT_EQ(created_str, joutstr);

  // Train
  std::string jtrainstr
      = "{\"service\":\"imgserv\",\"async\":false,\"parameters\":{"
        "\"mllib\":{\"solver\":{\"iterations\":"
        + iterations_resnet50 + ",\"base_lr\":" + torch_lr
        + ",\"iter_size\":4,\"solver_type\":\"RANGER\","
          "\"lookahead\":true,\"rectified\":true,\"adabelief\":true,"
          "\"gradient_centralization\":true,\"adamp\":true,\"clip\":false,"
          "\"test_interval\":"
        + iterations_resnet50
        + "},\"net\":{\"batch_size\":4},\"nclasses\":2,\"resume\":false},"
          "\"input\":{\"seed\":12345,\"db\":true,\"shuffle\":true,\"test_"
          "split\":0.1},"
          "\"output\":{\"measure\":[\"f1\",\"acc\"]}},\"data\":[\""
        + resnet50_train_data + "\",\"" + resnet50_test_data + "\"]}";
  joutstr = japi.jrender(japi.service_train(jtrainstr));
  JDoc jd;
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(201, jd["status"]["code"]);

  ASSERT_TRUE(jd["body"]["measure"]["acc"].GetDouble() <= 1) << "accuracy";
  ASSERT_TRUE(jd["body"]["measure"]["train_loss"].GetDouble() <= 5.0)
      << "loss";
  ASSERT_TRUE(jd["body"]["measure"]["f1"].GetDouble() <= 1) << "f1";

  // Predict
  std::string jpredictstr
      = "{\"service\":\"imgserv\",\"parameters\":{\"output\":{\"best\":1}},"
        "\"data\":[\""
        + resnet50_test_image + "\"]}";

  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  jd = JDoc();
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200, jd["status"]["code"]);
  ASSERT_TRUE(jd["body"]["predictions"].IsArray());
  std::string cl1
      = jd["body"]["predictions"][0]["classes"][0]["cat"].GetString();
  ASSERT_TRUE(jd["body"]["predictions"][0]["classes"][0]["prob"].GetDouble()
              > 0.0);

  std::unordered_set<std::string> lfiles;
  fileops::list_directory(resnet50_train_repo, true, false, false, lfiles);
  for (std::string ff : lfiles)
    {
      if (ff.find("checkpoint") != std::string::npos
          || ff.find("solver") != std::string::npos)
        remove(ff.c_str());
    }
  ASSERT_TRUE(!fileops::file_exists(resnet50_train_repo + "checkpoint-"
                                    + iterations_resnet50 + ".ptw"));
  ASSERT_TRUE(!fileops::file_exists(resnet50_train_repo + "checkpoint-"
                                    + iterations_resnet50 + ".pt"));

  fileops::clear_directory(resnet50_train_repo + "train.lmdb");
  fileops::clear_directory(resnet50_train_repo + "test_0.lmdb");
  fileops::remove_dir(resnet50_train_repo + "train.lmdb");
  fileops::remove_dir(resnet50_train_repo + "test_0.lmdb");
}

TEST(torchapi, service_train_madgrad)
{
  setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8", true);
  torch::manual_seed(torch_seed);
  at::globalContext().setDeterministicCuDNN(true);

  // Create service
  JsonAPI japi;
  std::string sname = "imgserv";
  std::string jstr
      = "{\"mllib\":\"torch\",\"description\":\"image\",\"type\":"
        "\"supervised\",\"model\":{\"repository\":\""
        + resnet50_train_repo
        + "\"},\"parameters\":{\"input\":{\"connector\":\"image\","
          "\"width\":224,\"height\":224,\"db\":true},\"mllib\":{\"nclasses\":"
          "2,\"finetuning\":true,\"gpu\":true}}}";
  std::string joutstr = japi.jrender(japi.service_create(sname, jstr));
  ASSERT_EQ(created_str, joutstr);

  // Train
  std::string jtrainstr
      = "{\"service\":\"imgserv\",\"async\":false,\"parameters\":{"
        "\"mllib\":{\"solver\":{\"iterations\":"
        + iterations_resnet50 + ",\"base_lr\":" + torch_lr
        + ",\"iter_size\":4,\"solver_type\":\"MADGRAD\","
          "\"lookahead\":false,\"clip\":false,"
          "\"test_interval\":"
        + iterations_resnet50
        + "},\"net\":{\"batch_size\":4},\"nclasses\":2,\"resume\":false},"
          "\"input\":{\"seed\":12345,\"db\":true,\"shuffle\":true,\"test_"
          "split\":0.1},"
          "\"output\":{\"measure\":[\"f1\",\"acc\"]}},\"data\":[\""
        + resnet50_train_data + "\",\"" + resnet50_test_data + "\"]}";
  joutstr = japi.jrender(japi.service_train(jtrainstr));
  JDoc jd;
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(201, jd["status"]["code"]);

  ASSERT_TRUE(jd["body"]["measure"]["acc"].GetDouble() <= 1) << "accuracy";
  ASSERT_TRUE(jd["body"]["measure"]["train_loss"].GetDouble() <= 5.0)
      << "loss";
  ASSERT_TRUE(jd["body"]["measure"]["f1"].GetDouble() <= 1) << "f1";

  // Predict
  std::string jpredictstr
      = "{\"service\":\"imgserv\",\"parameters\":{\"output\":{\"best\":1}},"
        "\"data\":[\""
        + resnet50_test_image + "\"]}";

  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  jd = JDoc();
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200, jd["status"]["code"]);
  ASSERT_TRUE(jd["body"]["predictions"].IsArray());
  std::string cl1
      = jd["body"]["predictions"][0]["classes"][0]["cat"].GetString();
  ASSERT_TRUE(jd["body"]["predictions"][0]["classes"][0]["prob"].GetDouble()
              > 0.0);

  std::unordered_set<std::string> lfiles;
  fileops::list_directory(resnet50_train_repo, true, false, false, lfiles);
  for (std::string ff : lfiles)
    {
      if (ff.find("checkpoint") != std::string::npos
          || ff.find("solver") != std::string::npos)
        remove(ff.c_str());
    }
  ASSERT_TRUE(!fileops::file_exists(resnet50_train_repo + "checkpoint-"
                                    + iterations_resnet50 + ".ptw"));
  ASSERT_TRUE(!fileops::file_exists(resnet50_train_repo + "checkpoint-"
                                    + iterations_resnet50 + ".pt"));

  fileops::clear_directory(resnet50_train_repo + "train.lmdb");
  fileops::clear_directory(resnet50_train_repo + "test_0.lmdb");
  fileops::remove_dir(resnet50_train_repo + "train.lmdb");
  fileops::remove_dir(resnet50_train_repo + "test_0.lmdb");
}

TEST(torchapi, service_train_vit_images_gpu)
{
  setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8", true);
  torch::manual_seed(torch_seed);
  at::globalContext().setDeterministicCuDNN(true);
  // torch::autograd::AnomalyMode::set_enabled(true);

  mkdir(vit_train_repo.c_str(), 0777);

  // Create service
  JsonAPI japi;
  std::string sname = "imgserv";
  std::string jstr
      = "{\"mllib\":\"torch\",\"description\":\"image\",\"type\":"
        "\"supervised\",\"model\":{\"repository\":\""

        + vit_train_repo
        + "\"},\"parameters\":{\"input\":{\"connector\":\"image\","
          "\"width\":224,\"height\":224,\"db\":true},\"mllib\":{\"nclasses\":"
          "2,\"template\":\"vit\",\"gpu\":true,\"template_params\":{\"vit_"
          "flavor\":\"vit_tiny_patch16\",\"realformer\":true}}}}";
  std::string joutstr = japi.jrender(japi.service_create(sname, jstr));
  ASSERT_EQ(created_str, joutstr);

  // Train
  std::string jtrainstr
      = "{\"service\":\"imgserv\",\"async\":false,\"parameters\":{"
        "\"mllib\":{\"solver\":{\"iterations\":"
        + iterations_vit + ",\"base_lr\":" + torch_lr
        + ",\"iter_size\":4,\"solver_type\":\"RANGER\","
          "\"lookahead\":true,\"rectified\":false,\"adabelief\":true,"
          "\"gradient_centralization\":true,\"test_interval\":"
        + iterations_vit
        + "},\"net\":{\"batch_size\":4},\"nclasses\":2,\"resume\":false},"
          "\"input\":{\"seed\":12345,\"db\":true,\"shuffle\":true},"
          "\"output\":{\"measure\":[\"f1\",\"acc\"]}},\"data\":[\""
        + resnet50_train_data + "\",\"" + resnet50_test_data + "\"]}";
  joutstr = japi.jrender(japi.service_train(jtrainstr));
  JDoc jd;
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(201, jd["status"]["code"]);

  ASSERT_TRUE(jd["body"]["measure"]["acc"].GetDouble() <= 1) << "accuracy";
  ASSERT_TRUE(jd["body"]["measure"]["acc"].GetDouble() >= 0.45)
      << "accuracy good";
  ASSERT_TRUE(jd["body"]["measure"]["f1"].GetDouble() <= 1) << "f1";

  // Predict
  std::string jpredictstr
      = "{\"service\":\"imgserv\",\"parameters\":{\"output\":{\"best\":1}},"
        "\"data\":[\""
        + resnet50_test_image + "\"]}";

  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  jd = JDoc();
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200, jd["status"]["code"]);
  ASSERT_TRUE(jd["body"]["predictions"].IsArray());
  std::string cl1
      = jd["body"]["predictions"][0]["classes"][0]["cat"].GetString();
  // Not training for long enough to be 100% sure a cat will be detected
  // ASSERT_TRUE(cl1 == "cats");
  ASSERT_TRUE(jd["body"]["predictions"][0]["classes"][0]["prob"].GetDouble()
              > 0.0);

  std::unordered_set<std::string> lfiles;
  fileops::list_directory(vit_train_repo, true, false, false, lfiles);
  for (std::string ff : lfiles)
    {
      if (ff.find("checkpoint") != std::string::npos
          || ff.find("solver") != std::string::npos)
        remove(ff.c_str());
    }
  ASSERT_TRUE(!fileops::file_exists(vit_train_repo + "checkpoint-"
                                    + iterations_vit + ".ptw"));
  ASSERT_TRUE(!fileops::file_exists(vit_train_repo + "checkpoint-"
                                    + iterations_vit + ".pt"));

  fileops::clear_directory(vit_train_repo + "train.lmdb");
  fileops::clear_directory(vit_train_repo + "test_0.lmdb");
  fileops::remove_dir(vit_train_repo + "train.lmdb");
  fileops::remove_dir(vit_train_repo + "test_0.lmdb");
}

TEST(torchapi, service_train_vit_images_multigpu)
{
  setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8", true);
  torch::manual_seed(torch_seed);
  at::globalContext().setDeterministicCuDNN(true);

  mkdir(vit_train_repo.c_str(), 0777);

  // Create service
  JsonAPI japi;
  std::string sname = "imgserv";
  std::string jstr
      = "{\"mllib\":\"torch\",\"description\":\"image\",\"type\":"
        "\"supervised\",\"model\":{\"repository\":\""

        + vit_train_repo
        + "\"},\"parameters\":{\"input\":{\"connector\":\"image\","
          "\"width\":224,\"height\":224,\"db\":true},\"mllib\":{\"nclasses\":"
          "2,\"vit_flavor\":\"vit_mini_patch16\",\"template\":\"vit\",\"gpu\":"
          "true,\"gpuid\":[0, 1]}}}";
  std::string joutstr = japi.jrender(japi.service_create(sname, jstr));
  ASSERT_EQ(created_str, joutstr);

  // Train
  std::string jtrainstr
      = "{\"service\":\"imgserv\",\"async\":false,\"parameters\":{"
        "\"mllib\":{\"solver\":{\"iterations\":"
        + iterations_vit + ",\"base_lr\":" + torch_lr
        + ",\"iter_size\":4,\"solver_type\":\"RANGER\","
          "\"lookahead\":true,\"rectified\":false,\"adabelief\":true,"
          "\"gradient_centralization\":true,\"test_interval\":"
        + iterations_vit
        + "},\"net\":{\"batch_size\":4},\"nclasses\":2,\"resume\":false},"
          "\"input\":{\"db\":true,\"seed\":12345,\"shuffle\":true},"
          "\"output\":{\"measure\":[\"f1\",\"acc\"]}},\"data\":[\""
        + resnet50_train_data + "\",\"" + resnet50_test_data + "\"]}";
  joutstr = japi.jrender(japi.service_train(jtrainstr));
  JDoc jd;
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(201, jd["status"]["code"]);

  ASSERT_TRUE(jd["body"]["measure"]["acc"].GetDouble() <= 1) << "accuracy";
  ASSERT_TRUE(jd["body"]["measure"]["acc"].GetDouble() >= 0.45)
      << "accuracy good";
  ASSERT_TRUE(jd["body"]["measure"]["f1"].GetDouble() <= 1) << "f1";

  // Predict
  std::string jpredictstr
      = "{\"service\":\"imgserv\",\"parameters\":{\"output\":{\"best\":1}},"
        "\"data\":[\""
        + resnet50_test_image + "\"]}";

  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  jd = JDoc();
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200, jd["status"]["code"]);
  ASSERT_TRUE(jd["body"]["predictions"].IsArray());
  std::string cl1
      = jd["body"]["predictions"][0]["classes"][0]["cat"].GetString();
  // Not training for long enough to be 100% sure a cat will be detected
  // ASSERT_TRUE(cl1 == "cats");
  ASSERT_TRUE(jd["body"]["predictions"][0]["classes"][0]["prob"].GetDouble()
              > 0.0);

  std::unordered_set<std::string> lfiles;
  fileops::list_directory(vit_train_repo, true, false, false, lfiles);
  for (std::string ff : lfiles)
    {
      if (ff.find("checkpoint") != std::string::npos
          || ff.find("solver") != std::string::npos)
        remove(ff.c_str());
    }
  ASSERT_TRUE(!fileops::file_exists(vit_train_repo + "checkpoint-"
                                    + iterations_vit + ".ptw"));
  ASSERT_TRUE(!fileops::file_exists(vit_train_repo + "checkpoint-"
                                    + iterations_vit + ".pt"));

  fileops::clear_directory(vit_train_repo + "train.lmdb");
  fileops::clear_directory(vit_train_repo + "test_0.lmdb");
  fileops::remove_dir(vit_train_repo + "train.lmdb");
  fileops::remove_dir(vit_train_repo + "test_0.lmdb");
}

TEST(torchapi, service_train_visformer_images_gpu)
{
  setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8", true);
  torch::manual_seed(torch_seed);
  at::globalContext().setDeterministicCuDNN(true);
  // torch::autograd::AnomalyMode::set_enabled(true);

  mkdir(vit_train_repo.c_str(), 0777);

  // Create service
  JsonAPI japi;
  std::string sname = "imgserv";
  std::string jstr
      = "{\"mllib\":\"torch\",\"description\":\"image\",\"type\":"
        "\"supervised\",\"model\":{\"repository\":\""

        + vit_train_repo
        + "\"},\"parameters\":{\"input\":{\"connector\":\"image\","
          "\"width\":224,\"height\":224,\"db\":true},\"mllib\":{\"nclasses\":"
          "2,\"template\":\"visformer\",\"gpu\":true,\"template_params\":{"
          "\"visformer_"
          "flavor\":\"visformer_tiny\"}}}}";
  std::string joutstr = japi.jrender(japi.service_create(sname, jstr));
  ASSERT_EQ(created_str, joutstr);

  // Train
  std::string jtrainstr
      = "{\"service\":\"imgserv\",\"async\":false,\"parameters\":{"
        "\"mllib\":{\"solver\":{\"iterations\":"
        + iterations_vit + ",\"base_lr\":" + torch_lr
        + ",\"iter_size\":4,\"solver_type\":\"RANGER\","
          "\"lookahead\":true,\"rectified\":false,\"adabelief\":true,"
          "\"gradient_centralization\":true,\"test_interval\":"
        + iterations_vit
        + "},\"net\":{\"batch_size\":4},\"nclasses\":2,\"resume\":false},"
          "\"input\":{\"seed\":12345,\"db\":true,\"shuffle\":true},"
          "\"output\":{\"measure\":[\"f1\",\"acc\"]}},\"data\":[\""
        + resnet50_train_data + "\",\"" + resnet50_test_data + "\"]}";
  joutstr = japi.jrender(japi.service_train(jtrainstr));
  JDoc jd;
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(201, jd["status"]["code"]);

  ASSERT_TRUE(jd["body"]["measure"]["acc"].GetDouble() <= 1) << "accuracy";
  ASSERT_TRUE(jd["body"]["measure"]["acc"].GetDouble() >= 0.45)
      << "accuracy good";
  ASSERT_TRUE(jd["body"]["measure"]["f1"].GetDouble() <= 1) << "f1";

  // Predict
  std::string jpredictstr
      = "{\"service\":\"imgserv\",\"parameters\":{\"output\":{\"best\":1}},"
        "\"data\":[\""
        + resnet50_test_image + "\"]}";

  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  jd = JDoc();
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200, jd["status"]["code"]);
  ASSERT_TRUE(jd["body"]["predictions"].IsArray());
  std::string cl1
      = jd["body"]["predictions"][0]["classes"][0]["cat"].GetString();
  // Not training for long enough to be 100% sure a cat will be detected
  // ASSERT_TRUE(cl1 == "cats");
  ASSERT_TRUE(jd["body"]["predictions"][0]["classes"][0]["prob"].GetDouble()
              > 0.0);

  std::unordered_set<std::string> lfiles;
  fileops::list_directory(vit_train_repo, true, false, false, lfiles);
  for (std::string ff : lfiles)
    {
      if (ff.find("checkpoint") != std::string::npos
          || ff.find("solver") != std::string::npos)
        remove(ff.c_str());
    }
  ASSERT_TRUE(!fileops::file_exists(vit_train_repo + "checkpoint-"
                                    + iterations_vit + ".ptw"));
  ASSERT_TRUE(!fileops::file_exists(vit_train_repo + "checkpoint-"
                                    + iterations_vit + ".pt"));

  fileops::clear_directory(vit_train_repo + "train.lmdb");
  fileops::clear_directory(vit_train_repo + "test_0.lmdb");
  fileops::remove_dir(vit_train_repo + "train.lmdb");
  fileops::remove_dir(vit_train_repo + "test_0.lmdb");
}

#endif
