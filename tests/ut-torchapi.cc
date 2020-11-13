/**
 * DeepDetect
 * Copyright (c) 2019 Jolibrain
 * Author: Louis Jean <ljean@etud.insa-toulouse.fr>
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
#include "backends/torch/native/templates/nbeats.h"

using namespace dd;

static std::string ok_str = "{\"status\":{\"code\":200,\"msg\":\"OK\"}}";
static std::string created_str
    = "{\"status\":{\"code\":201,\"msg\":\"Created\"}}";
static std::string bad_param_str
    = "{\"status\":{\"code\":400,\"msg\":\"BadRequest\"}}";
static std::string not_found_str
    = "{\"status\":{\"code\":404,\"msg\":\"NotFound\"}}";

static std::string incept_repo = "../examples/torch/resnet50_torch/";
static std::string resnet50_train_repo
    = "../examples/torch/resnet50_training_torch/";
static std::string resnet50_train_data
    = "../examples/torch/resnet50_training_torch/train/";
static std::string resnet50_train_data_reg
    = "../examples/torch/resnet50_training_torch/list_all_shuf_rel.txt";
static std::string resnet50_test_data
    = "../examples/torch/resnet50_training_torch/test/";
static std::string resnet50_test_image
    = "../examples/torch/resnet50_training_torch/train/cats/cat.102.jpg";

static std::string bert_classif_repo
    = "../examples/torch/bert_inference_torch/";
static std::string bert_train_repo
    = "../examples/torch/bert_training_torch_140_transformers_251/";
static std::string bert_train_data
    = "../examples/torch/bert_training_torch_140_transformers_251/data/";

static std::string sinus = "../examples/all/sinus/";

static std::string iterations_nbeats_cpu = "100";
static std::string iterations_nbeats_gpu = "1000";

static std::string iterations_resnet50 = "2000";

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
          "224,\"width\":224,\"rgb\":true,\"scale\":0.0039}}}";
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
  std::string cl1
      = jd["body"]["predictions"][0]["classes"][0]["cat"].GetString();
  ASSERT_TRUE(cl1 == "n02123045 tabby, tabby cat");
  ASSERT_TRUE(jd["body"]["predictions"][0]["classes"][0]["prob"].GetDouble()
              > 0.3);
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

// Training tests

#if !defined(CPU_ONLY)

TEST(torchapi, service_train_images_split)
{
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
        + iterations_resnet50
        + ",\"base_lr\":1e-5,\"iter_size\":4,\"solver_type\":\"ADAM\",\"test_"
          "interval\":"
        + iterations_resnet50
        + "},\"net\":{\"batch_size\":4},\"nclasses\":2,\"resume\":false},"
          "\"input\":{\"db\":true,\"shuffle\":true,\"test_split\":0.1},"
          "\"output\":{\"measure\":[\"f1\",\"acc\"]}},\"data\":[\""
        + resnet50_train_data + "\"]}";
  joutstr = japi.jrender(japi.service_train(jtrainstr));
  JDoc jd;
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(201, jd["status"]["code"]);

  ASSERT_TRUE(jd["body"]["measure"]["iteration"] == 2000) << "iterations";
  ASSERT_TRUE(jd["body"]["measure"]["acc"].GetDouble() <= 1) << "accuracy";
  ASSERT_TRUE(jd["body"]["measure"]["acc"].GetDouble() >= 0.6)
      << "accuracy good";
  ASSERT_TRUE(jd["body"]["measure"]["f1"].GetDouble() <= 1) << "f1";

  fileops::remove_file(resnet50_train_repo,
                       "checkpoint-" + iterations_resnet50 + ".ptw");
  fileops::remove_file(resnet50_train_repo,
                       "checkpoint-" + iterations_resnet50 + ".pt");
  fileops::remove_file(resnet50_train_repo,
                       "solver-" + iterations_resnet50 + ".pt");
  fileops::clear_directory(resnet50_train_repo + "train.lmdb");
  fileops::clear_directory(resnet50_train_repo + "test.lmdb");
  fileops::remove_dir(resnet50_train_repo + "train.lmdb");
  fileops::remove_dir(resnet50_train_repo + "test.lmdb");
}

TEST(torchapi, service_train_images)
{
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
        + iterations_resnet50
        + ",\"base_lr\":1e-5,\"iter_size\":4,\"solver_type\":\"ADAM\",\"test_"
          "interval\":"
        + iterations_resnet50
        + "},\"net\":{\"batch_size\":4},\"nclasses\":2,\"resume\":false},"
          "\"input\":{\"db\":true,\"shuffle\":true,\"test_split\":0.1},"
          "\"output\":{\"measure\":[\"f1\",\"acc\"]}},\"data\":[\""
        + resnet50_train_data + "\",\"" + resnet50_test_data + "\"]}";
  joutstr = japi.jrender(japi.service_train(jtrainstr));
  JDoc jd;
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(201, jd["status"]["code"]);

  ASSERT_TRUE(jd["body"]["measure"]["acc"].GetDouble() <= 1) << "accuracy";
  ASSERT_TRUE(jd["body"]["measure"]["acc"].GetDouble() >= 0.6)
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

  fileops::remove_file(resnet50_train_repo,
                       "checkpoint-" + iterations_resnet50 + ".ptw");
  fileops::remove_file(resnet50_train_repo,
                       "checkpoint-" + iterations_resnet50 + ".pt");
  fileops::remove_file(resnet50_train_repo,
                       "solver-" + iterations_resnet50 + ".pt");
  fileops::clear_directory(resnet50_train_repo + "train.lmdb");
  fileops::clear_directory(resnet50_train_repo + "test.lmdb");
  fileops::remove_dir(resnet50_train_repo + "train.lmdb");
  fileops::remove_dir(resnet50_train_repo + "test.lmdb");
}

TEST(torchapi, service_train_images_split_regression)
{
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
        + iterations_resnet50
        + ",\"base_lr\":1e-5,\"iter_size\":4,\"solver_type\":\"ADAM\",\"test_"
          "interval\":"
        + iterations_resnet50
        + "},\"net\":{\"batch_size\":4},\"resume\":false},"
          "\"input\":{\"db\":true,\"shuffle\":true,\"test_split\":0.1},"
          "\"output\":{\"measure\":[\"eucll\"]}},\"data\":[\""
        + resnet50_train_data_reg + "\"]}";
  joutstr = japi.jrender(japi.service_train(jtrainstr));
  JDoc jd;
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(201, jd["status"]["code"]);

  ASSERT_TRUE(jd["body"]["measure"]["iteration"] == 2000) << "iterations";
  ASSERT_TRUE(jd["body"]["measure"]["eucll"].GetDouble() <= 3.0) << "eucll";

  fileops::remove_file(resnet50_train_repo,
                       "checkpoint-" + iterations_resnet50 + ".ptw");
  fileops::remove_file(resnet50_train_repo,
                       "checkpoint-" + iterations_resnet50 + ".pt");
  fileops::remove_file(resnet50_train_repo,
                       "solver-" + iterations_resnet50 + ".pt");
  fileops::clear_directory(resnet50_train_repo + "train.lmdb");
  fileops::clear_directory(resnet50_train_repo + "test.lmdb");
  fileops::remove_dir(resnet50_train_repo + "train.lmdb");
  fileops::remove_dir(resnet50_train_repo + "test.lmdb");
}

TEST(torchapi, service_train_txt_lm)
{
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
        "\"mllib\":{\"solver\":{\"iterations\":3,\"base_lr\":1e-5,\"iter_"
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
  fileops::remove_file(bert_train_repo, "checkpoint-3.pt");
  fileops::remove_file(bert_train_repo, "solver-3.pt");
  fileops::remove_file(bert_train_repo, "checkpoint-2.pt");
  fileops::remove_file(bert_train_repo, "solver-2.pt");
  fileops::remove_file(bert_train_repo, "checkpoint-1.pt");
  fileops::remove_file(bert_train_repo, "solver-1.pt");
}

TEST(torchapi, service_train_txt_classification)
{
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
        "\"mllib\":{\"solver\":{\"iterations\":3,\"base_lr\":1e-5,\"iter_"
        "size\":2,\"solver_type\":\"ADAM\"},\"net\":{\"batch_size\":2}},"
        "\"input\":{\"shuffle\":true,\"test_split\":0.25},"
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
  fileops::remove_file(bert_train_repo, "checkpoint-3.ptw");
  fileops::remove_file(bert_train_repo, "checkpoint-3.pt");
  fileops::remove_file(bert_train_repo, "solver-3.pt");
  fileops::remove_file(bert_train_repo, "checkpoint-1.ptw");
  fileops::remove_file(bert_train_repo, "checkpoint-1.pt");
  fileops::remove_file(bert_train_repo, "solver-1.pt");
  fileops::remove_file(bert_train_repo, "checkpoint-2.ptw");
  fileops::remove_file(bert_train_repo, "checkpoint-2.pt");
  fileops::remove_file(bert_train_repo, "solver-2.pt");
}

#endif

TEST(torchapi, service_train_csvts_nbeats)
{
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
        + "\"},\"parameters\":{\"input\":{\"connector\":\"csvts\",\"label\":["
          "\"output\"],\"timesteps\":50},\"mllib\":{\"template\":\"nbeats\","
          "\"template_params\":[\"t2\",\"s4\",\"g3\",\"b3\"],"
          "\"loss\":\"L1\"}}}";

  std::string joutstr = japi.jrender(japi.service_create(sname, jstr));
  ASSERT_EQ(created_str, joutstr);

  // train
  std::string jtrainstr
      = "{\"service\":\"" + sname
        + "\",\"async\":false,\"parameters\":{\"input\":{\"shuffle\":true,"
          "\"separator\":\",\",\"scale\":true,\"timesteps\":50,\"label\":["
          "\"output\"]},\"mllib\":{\"gpu\":false,\"solver\":{\"iterations\":"
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
        + "\",\"parameters\":{\"input\":{\"timesteps\":50,\"connector\":"
          "\"csvts\",\"scale\":true,\"label\":[\"output\"],\"continuation\":"
          "true,\"min_vals\":"
        + str_min_vals + ",\"max_vals\":" + str_max_vals
        + "},\"output\":{}},\"data\":[\"" + csvts_predict + "\"]}";
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200, jd["status"]["code"]);
  std::string uri = jd["body"]["predictions"][0]["uri"].GetString();
  ASSERT_EQ("../examples/all/sinus/predict/seq_2.csv #0_49", uri);
  ASSERT_TRUE(jd["body"]["predictions"][0]["series"].IsArray());
  ASSERT_TRUE(jd["body"]["predictions"][0]["series"][0]["out"][0].GetDouble()
              >= -1.0);

  //  remove service
  jstr = "{\"clear\":\"full\"}";
  joutstr = japi.jrender(japi.service_delete(sname, jstr));
  ASSERT_EQ(ok_str, joutstr);
  rmdir(csvts_nbeats_repo.c_str());
}

#if !defined(CPU_ONLY)

TEST(torchapi, service_train_csvts_nbeats_gpu)
{
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
        + "\"},\"parameters\":{\"input\":{\"connector\":\"csvts\",\"label\":["
          "\"output\"],\"timesteps\":50},\"mllib\":{\"template\":\"nbeats\","
          "\"template_params\":[\"t2\",\"s4\",\"g3\",\"b3\"],"
          "\"loss\":\"L1\",\"gpu\":true,\"gpuid\":0}}}";

  std::string joutstr = japi.jrender(japi.service_create(sname, jstr));
  ASSERT_EQ(created_str, joutstr);

  // train
  std::string jtrainstr
      = "{\"service\":\"" + sname
        + "\",\"async\":false,\"parameters\":{\"input\":{\"shuffle\":true,"
          "\"separator\":\",\",\"scale\":true,\"timesteps\":50,\"label\":["
          "\"output\"]},\"mllib\":{\"gpu\":true,\"gpuid\":0,\"solver\":{"
          "\"iterations\":"
        + iterations_nbeats_gpu
        + ",\"test_interval\":100,\"base_lr\":0.001,\"snapshot\":500,\"test_"
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
        + "\",\"parameters\":{\"input\":{\"timesteps\":50,\"connector\":"
          "\"csvts\",\"scale\":true,\"label\":[\"output\"],\"min_vals\":"
        + str_min_vals + ",\"max_vals\":" + str_max_vals
        + "},\"output\":{}},\"data\":[\"" + csvts_predict + "\"]}";
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200, jd["status"]["code"]);
  std::string uri = jd["body"]["predictions"][0]["uri"].GetString();
  ASSERT_EQ("../examples/all/sinus/predict/seq_2.csv #0_49", uri);
  ASSERT_TRUE(jd["body"]["predictions"][0]["series"].IsArray());
  ASSERT_TRUE(jd["body"]["predictions"][0]["series"][0]["out"][0].GetDouble()
              >= -1.1);

  joutstr = japi.jrender(japi.service_status(sname));
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse(joutstr.c_str());
  ASSERT_TRUE(jd["body"]["service_stats"].HasMember("predict_count"));
  ASSERT_TRUE(jd["body"]["service_stats"]["inference_count"].GetInt() == 20);
  ASSERT_TRUE(jd["body"]["service_stats"]["predict_count"].GetInt() == 1);
  ASSERT_TRUE(jd["body"]["service_stats"]["predict_failure"].GetInt() == 0);
  ASSERT_TRUE(jd["body"]["service_stats"]["predict_success"].GetInt() == 1);
  ASSERT_TRUE(jd["body"]["service_stats"]["avg_batch_size"].GetDouble()
              == 20.0);
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
        + "\"},\"parameters\":{\"input\":{\"connector\":\"csvts\",\"label\":["
          "\"output\"],\"timesteps\":50},\"mllib\":{\"template\":\"nbeats\","
          "\"template_params\":[\"t2\",\"s4\",\"g3\",\"b3\"],"
          "\"loss\":\"L1\",\"gpu\":true,\"gpuid\":[0,1]}}}";

  std::string joutstr = japi.jrender(japi.service_create(sname, jstr));
  ASSERT_EQ(created_str, joutstr);

  // train
  std::string jtrainstr
      = "{\"service\":\"" + sname
        + "\",\"async\":false,\"parameters\":{\"input\":{\"shuffle\":true,"
          "\"separator\":\",\",\"scale\":true,\"timesteps\":50,\"label\":["
          "\"output\"]},\"mllib\":{\"gpu\":true,\"gpuid\":[0, 1],\"solver\":{"
          "\"iterations\":"
        + iterations_nbeats_gpu
        + ",\"test_interval\":100,\"base_lr\":0.001,\"snapshot\":500,\"test_"
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
        + "\",\"parameters\":{\"input\":{\"timesteps\":50,\"connector\":"
          "\"csvts\",\"scale\":true,\"label\":[\"output\"],\"min_vals\":"
        + str_min_vals + ",\"max_vals\":" + str_max_vals
        + "},\"output\":{}},\"data\":[\"" + csvts_predict + "\"]}";
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200, jd["status"]["code"]);
  std::string uri = jd["body"]["predictions"][0]["uri"].GetString();
  ASSERT_EQ("../examples/all/sinus/predict/seq_2.csv #0_49", uri);
  ASSERT_TRUE(jd["body"]["predictions"][0]["series"].IsArray());
  ASSERT_TRUE(jd["body"]["predictions"][0]["series"][0]["out"][0].GetDouble()
              >= -1.0);

  joutstr = japi.jrender(japi.service_status(sname));
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse(joutstr.c_str());
  ASSERT_TRUE(jd["body"]["service_stats"].HasMember("predict_count"));
  ASSERT_TRUE(jd["body"]["service_stats"]["inference_count"].GetInt() == 20);
  ASSERT_TRUE(jd["body"]["service_stats"]["predict_count"].GetInt() == 1);
  ASSERT_TRUE(jd["body"]["service_stats"]["predict_failure"].GetInt() == 0);
  ASSERT_TRUE(jd["body"]["service_stats"]["predict_success"].GetInt() == 1);
  ASSERT_TRUE(jd["body"]["service_stats"]["avg_batch_size"].GetDouble()
              == 20.0);
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

  torch::Tensor x = torch::randn({ 2, 50, 1 });
  torch::Tensor y = nb.forward(x);
  ASSERT_EQ(y.sizes(), std::vector<long int>({ 2, 2, 50, 1 }));
  torch::Tensor z = nb.extract(x, "2:0:fc1");
  ASSERT_EQ(z.sizes(), std::vector<long int>({ 2, 10 }));
  torch::Tensor t = nb.extract(x, "1:1:end");
  std::cout << t << std::endl;
}

TEST(torchapi, nbeats_extract_layer_complete)
{
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
        + "\"},\"parameters\":{\"input\":{\"connector\":\"csvts\",\"label\":["
          "\"output\"],\"timesteps\":50},\"mllib\":{\"template\":\"nbeats\","
          "\"template_params\":[\"t2\",\"s4\",\"g3\",\"b3\"],"
          "\"loss\":\"L1\"}}}";

  std::string joutstr = japi.jrender(japi.service_create(sname, jstr));
  ASSERT_EQ(created_str, joutstr);

  // train
  std::string jtrainstr
      = "{\"service\":\"" + sname
        + "\",\"async\":false,\"parameters\":{\"input\":{\"shuffle\":true,"
          "\"separator\":\",\",\"scale\":true,\"timesteps\":50,\"label\":["
          "\"output\"]},\"mllib\":{\"gpu\":false,\"solver\":{\"iterations\":"
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
          "\"input\":{\"timesteps\":50,\"connector\":"
          "\"csvts\",\"scale\":true,\"label\":[\"output\"],\"continuation\":"
          "true,\"min_vals\":"
        + str_min_vals + ",\"max_vals\":" + str_max_vals
        + "},\"output\":{}},\"data\":[\"" + csvts_predict + "\"]}";
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200, jd["status"]["code"]);
  std::string uri = jd["body"]["predictions"][0]["uri"].GetString();
  ASSERT_EQ("../examples/all/sinus/predict/seq_2.csv #0_49", uri);
  ASSERT_TRUE(jd["body"]["predictions"][0]["vals"].IsArray());
  ASSERT_EQ(jd["body"]["predictions"].Size(), 20);
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
        + "\"},\"parameters\":{\"input\":{\"connector\":\"csvts\",\"label\":["
          "\"output\"],\"timesteps\":50},\"mllib\":{\"template\":\"nbeats\","
          "\"template_params\":[\"t2\",\"s4\",\"g3\",\"b3\"],"
          "\"loss\":\"L1\"}}}";

  std::string joutstr = japi.jrender(japi.service_create(sname, jstr));
  ASSERT_EQ(created_str, joutstr);

  // train
  std::string jtrainstr
      = "{\"service\":\"" + sname
        + "\",\"async\":false,\"parameters\":{\"input\":{\"shuffle\":true,"
          "\"separator\":\",\",\"scale\":true,\"timesteps\":50,\"label\":["
          "\"output\"]},\"mllib\":{\"gpu\":true,\"gpuid\":0,\"solver\":{"
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
          "\"input\":{\"timesteps\":50,\"connector\":"
          "\"csvts\",\"scale\":true,\"label\":[\"output\"],\"continuation\":"
          "true,\"min_vals\":"
        + str_min_vals + ",\"max_vals\":" + str_max_vals
        + "},\"output\":{}},\"data\":[\"" + csvts_predict + "\"]}";
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200, jd["status"]["code"]);
  std::string uri = jd["body"]["predictions"][0]["uri"].GetString();
  ASSERT_EQ("../examples/all/sinus/predict/seq_2.csv #0_49", uri);
  ASSERT_TRUE(jd["body"]["predictions"][0]["vals"].IsArray());
  ASSERT_EQ(jd["body"]["predictions"].Size(), 20);
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

TEST(torchapi, service_train_ranger)
{
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
        + iterations_resnet50
        + ",\"base_lr\":1e-5,\"iter_size\":4,\"solver_type\":\"RANGER\","
          "\"lookahead\":true,\"rectified\":true,\"adabelief\":true,"
          "\"gradient_centralization\":true,\"clip\":false,"
          "\"test_"
          "interval\":"
        + iterations_resnet50
        + "},\"net\":{\"batch_size\":4},\"nclasses\":2,\"resume\":false},"
          "\"input\":{\"db\":true,\"shuffle\":true,\"test_split\":0.1},"
          "\"output\":{\"measure\":[\"f1\",\"acc\"]}},\"data\":[\""
        + resnet50_train_data + "\",\"" + resnet50_test_data + "\"]}";
  joutstr = japi.jrender(japi.service_train(jtrainstr));
  JDoc jd;
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(201, jd["status"]["code"]);

  ASSERT_TRUE(jd["body"]["measure"]["acc"].GetDouble() <= 1) << "accuracy";
  ASSERT_TRUE(jd["body"]["measure"]["acc"].GetDouble() >= 0.6)
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

  fileops::remove_file(resnet50_train_repo,
                       "checkpoint-" + iterations_resnet50 + ".ptw");
  fileops::remove_file(resnet50_train_repo,
                       "checkpoint-" + iterations_resnet50 + ".pt");
  fileops::remove_file(resnet50_train_repo,
                       "solver-" + iterations_resnet50 + ".pt");
  fileops::clear_directory(resnet50_train_repo + "train.lmdb");
  fileops::clear_directory(resnet50_train_repo + "test.lmdb");
  fileops::remove_dir(resnet50_train_repo + "train.lmdb");
  fileops::remove_dir(resnet50_train_repo + "test.lmdb");
}

TEST(torchapi, service_train_clip)
{
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
        "\"mllib\":{\"solver\":{\"iterations\":200"
        ",\"base_lr\":1e-5,\"iter_size\":4,\"solver_type\":\"RMSPROP\","
        "\"clip\":true,\"clip_norm\":1000.0,\"test_interval\":200"
        "},\"net\":{\"batch_size\":4},\"nclasses\":2,\"resume\":false},"
        "\"input\":{\"db\":true,\"shuffle\":true,\"test_split\":0.1},"
        "\"output\":{\"measure\":[\"f1\",\"acc\"]}},\"data\":[\""
        + resnet50_train_data + "\",\"" + resnet50_test_data + "\"]}";
  joutstr = japi.jrender(japi.service_train(jtrainstr));
  JDoc jd;
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(201, jd["status"]["code"]);

  ASSERT_TRUE(jd["body"]["measure"]["acc"].GetDouble() <= 1) << "accuracy";
  ASSERT_TRUE(jd["body"]["measure"]["acc"].GetDouble() >= 0.5)
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

  fileops::remove_file(resnet50_train_repo, "checkpoint-200.ptw");
  fileops::remove_file(resnet50_train_repo, "checkpoint-200.pt");
  fileops::remove_file(resnet50_train_repo, "solver-200.pt");
  fileops::clear_directory(resnet50_train_repo + "train.lmdb");
  fileops::clear_directory(resnet50_train_repo + "test.lmdb");
  fileops::remove_dir(resnet50_train_repo + "train.lmdb");
  fileops::remove_dir(resnet50_train_repo + "test.lmdb");
}
