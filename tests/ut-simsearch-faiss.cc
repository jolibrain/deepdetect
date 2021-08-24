/**
 * DeepDetect
 * Copyright (c) 2017 Emmanuel Benazera
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

#include "simsearch.h"
#include "jsonapi.h"
#include <gtest/gtest.h>
#include <iostream>

using namespace dd;

static std::string ok_str = "{\"status\":{\"code\":200,\"msg\":\"OK\"}}";
static std::string created_str
    = "{\"status\":{\"code\":201,\"msg\":\"Created\"}}";
static std::string bad_param_str
    = "{\"status\":{\"code\":400,\"msg\":\"BadRequest\"}}";
static std::string not_found_str
    = "{\"status\":{\"code\":404,\"msg\":\"NotFound\"}}";

static std::string mnist_repo = "../examples/caffe/mnist/";
static std::string iterations_mnist = "2";
static std::string voc_repo = "../examples/caffe/voc_roi/voc_roi/";
static std::string caffe_word_detect_repo
    = "../examples/caffe/word_detect_v2/";

TEST(faissse, index_search)
{
  std::vector<double> vec1 = { 1.0, 0.0, 0.0, 0.0 };
  std::vector<double> vec2 = { 0.0, 1.0, 0.0, 0.0 };
  std::vector<double> vec3 = { 1.0, 0.0, 1.0, 0.0 };

  int t = 4;
  std::string model_repo = "simsearch";
  mkdir(model_repo.c_str(), 0770);
  FaissSE fse(t, model_repo);
  fse.create_index();                // index creation
  fse.index(URIData("test1"), vec1); // indexing data
  fse.index(URIData("test2"), vec2);
  fse.index(URIData("test3"), vec3);
  std::vector<URIData> uris;
  std::vector<double> distances;
  fse.update_index();
  fse.search(vec1, 3, uris, distances); // searching nearest neighbors
  std::cerr << "search uris size=" << uris.size() << std::endl;
  for (size_t i = 0; i < uris.size(); i++)
    std::cout << uris.at(i)._uri << " / distances=" << distances.at(i)
              << std::endl;
  fse.remove_index();
  rmdir(model_repo.c_str());
}

TEST(faissse, index_search_incr)
{
  std::vector<double> vec1 = { 1.0, 0.0, 0.0, 0.0 };
  std::vector<double> vec2 = { 0.0, 1.0, 0.0, 0.0 };
  std::vector<double> vec3 = { 1.0, 0.0, 1.0, 0.0 };
  std::vector<double> vec4 = { 0.0, 0.0, 5.0, 5.0 };

  int t = 4;
  std::string model_repo = "simsearch";
  mkdir(model_repo.c_str(), 0770);
  FaissSE fse(t, model_repo);
  fse.create_index();
  fse.index(URIData("test1"), vec1);
  fse.index(URIData("test2"), vec2);
  fse.index(URIData("test3"), vec3);
  std::vector<URIData> uris;
  std::vector<double> distances;
  fse.update_index();
  fse.search(vec1, 3, uris, distances);
  std::cerr << "search uris size=" << uris.size() << std::endl;
  for (size_t i = 0; i < uris.size(); i++)
    std::cout << uris.at(i)._uri << " / distances=" << distances.at(i)
              << std::endl;
  uris.clear();
  distances.clear();
  fse.index(URIData("test4"), vec4);
  fse.update_index();
  fse.search(vec4, 3, uris, distances);
  std::cerr << "search uris size=" << uris.size() << std::endl;
  for (size_t i = 0; i < uris.size(); i++)
    std::cout << uris.at(i)._uri << " / distances=" << distances.at(i)
              << std::endl;
  fse.remove_index();
  rmdir(model_repo.c_str());
}

TEST(simsearch, predict_simsearch_unsup)
{
  // create service
  JsonAPI japi;
  std::string sname = "my_service";
  std::string jstr
      = "{\"mllib\":\"caffe\",\"description\":\"my "
        "classifier\",\"type\":\"unsupervised\",\"model\":{\"repository\":\""
        + mnist_repo
        + "\"},\"parameters\":{\"input\":{\"connector\":\"image\"},\"mllib\":{"
          "\"nclasses\":10}}}";
  std::string joutstr = japi.jrender(japi.service_create(sname, jstr));
  ASSERT_EQ(created_str, joutstr);
  JDoc jd;

  // train
  std::string gpuid = "0";
  std::string jtrainstr
      = "{\"service\":\"" + sname
        + "\",\"async\":false,\"parameters\":{\"mllib\":{\"gpu\":true,"
          "\"gpuid\":"
        + gpuid + ",\"solver\":{\"iterations\":" + iterations_mnist
        + ",\"snapshot\":200,\"snapshot_prefix\":\"" + mnist_repo
        + "/mylenet\",\"test_interval\":2}},\"output\":{\"measure_hist\":true,"
          "\"measure\":[\"f1\"]}}}";
  joutstr = japi.jrender(japi.service_train(jtrainstr));
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_TRUE(jd.HasMember("status"));
  ASSERT_EQ(201, jd["status"]["code"].GetInt());
  ASSERT_EQ("Created", jd["status"]["msg"]);
  ASSERT_TRUE(jd.HasMember("head"));
  ASSERT_EQ("/train", jd["head"]["method"]);
  ASSERT_TRUE(jd["head"]["time"].GetDouble() >= 0);
  ASSERT_TRUE(jd.HasMember("body"));
  ASSERT_TRUE(jd["body"].HasMember("measure"));
  ASSERT_TRUE(fabs(jd["body"]["measure"]["train_loss"].GetDouble()) > 0);
  ASSERT_EQ(jd["body"]["measure_hist"]["iteration_hist"].Size(),
            jd["body"]["measure_hist"]["f1_hist"].Size());

  // predict
  std::string jpredictstr
      = "{\"service\":\"" + sname
        + "\",\"parameters\":{\"input\":{\"bw\":true,\"width\":28,\"height\":"
          "28},\"mllib\":{\"extract_layer\":\"ip2\"},\"output\":{\"index\":"
          "true,\"index_type\":\"Flat\",\"index_gpu\":false}},\"data\":[\""
        + mnist_repo + "/sample_digit.png\"]}";
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  std::cout << "joutstr predict index=" << joutstr << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200, jd["status"]["code"]);
  ASSERT_TRUE(jd["body"]["predictions"][0].HasMember("indexed"));
  ASSERT_TRUE(jd["body"]["predictions"][0]["indexed"].GetBool());

  // build & save index
  jpredictstr = "{\"service\":\"" + sname
                + "\",\"parameters\":{\"input\":{\"bw\":true,\"width\":28,"
                  "\"height\":28},\"mllib\":{\"extract_layer\":\"ip2\"},"
                  "\"output\":{\"build_index\":true}},\"data\":[\""
                + mnist_repo + "/sample_digit.png\"]}";
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  std::cout << "joutstr predict build index=" << joutstr << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200, jd["status"]["code"]);

  // assert existence of index
  ASSERT_TRUE(fileops::file_exists(mnist_repo + "index.faiss"));
  ASSERT_TRUE(fileops::file_exists(mnist_repo + "names.bin/data.mdb"));

  // search index
  jpredictstr = "{\"service\":\"" + sname
                + "\",\"parameters\":{\"input\":{\"bw\":true,\"width\":28,"
                  "\"height\":28},\"mllib\":{\"extract_layer\":\"ip2\"},"
                  "\"output\":{\"search\":true}},\"data\":[\""
                + mnist_repo + "/sample_digit.png\"]}";
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  std::cout << "joutstr predict search=" << joutstr << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200, jd["status"]["code"]);
  // assert result is itself
  ASSERT_TRUE(jd["body"]["predictions"][0].HasMember("nns"));
  ASSERT_TRUE(jd["body"]["predictions"][0]["nns"][0]["dist"].GetDouble()
              < 1.0);
  ASSERT_TRUE(jd["body"]["predictions"][0]["nns"][0]["uri"]
              == "../examples/caffe/mnist//sample_digit.png");

  // remove service
  jstr = "{\"clear\":\"lib\"}";
  joutstr = japi.jrender(japi.service_delete(sname, jstr));
  ASSERT_EQ(ok_str, joutstr);

  // assert non-existence of index
  ASSERT_TRUE(!fileops::file_exists(mnist_repo + "index.faiss"));
  ASSERT_TRUE(!fileops::file_exists(mnist_repo + "names.bin/data.mdb"));
}

TEST(simsearch, predict_simsearch_sup)
{
  // create service
  JsonAPI japi;
  std::string sname = "my_service";
  std::string jstr
      = "{\"mllib\":\"caffe\",\"description\":\"my "
        "classifier\",\"type\":\"supervised\",\"model\":{\"repository\":\""
        + mnist_repo
        + "\"},\"parameters\":{\"input\":{\"connector\":\"image\"},\"mllib\":{"
          "\"nclasses\":10}}}";
  std::string joutstr = japi.jrender(japi.service_create(sname, jstr));
  ASSERT_EQ(created_str, joutstr);
  JDoc jd;

  // train
  std::string gpuid = "0";
  std::string jtrainstr
      = "{\"service\":\"" + sname
        + "\",\"async\":false,\"parameters\":{\"mllib\":{\"gpu\":true,"
          "\"gpuid\":"
        + gpuid + ",\"solver\":{\"iterations\":" + iterations_mnist
        + ",\"snapshot\":200,\"snapshot_prefix\":\"" + mnist_repo
        + "/mylenet\",\"test_interval\":2}},\"output\":{\"measure_hist\":true,"
          "\"measure\":[\"f1\"]}}}";
  joutstr = japi.jrender(japi.service_train(jtrainstr));
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_TRUE(jd.HasMember("status"));
  ASSERT_EQ(201, jd["status"]["code"].GetInt());
  ASSERT_EQ("Created", jd["status"]["msg"]);
  ASSERT_TRUE(jd.HasMember("head"));
  ASSERT_EQ("/train", jd["head"]["method"]);
  ASSERT_TRUE(jd["head"]["time"].GetDouble() >= 0);
  ASSERT_TRUE(jd.HasMember("body"));
  ASSERT_TRUE(jd["body"].HasMember("measure"));
  ASSERT_TRUE(fabs(jd["body"]["measure"]["train_loss"].GetDouble()) > 0);
  ASSERT_EQ(jd["body"]["measure_hist"]["iteration_hist"].Size(),
            jd["body"]["measure_hist"]["f1_hist"].Size());

  // predict
  std::string jpredictstr
      = "{\"service\":\"" + sname
        + "\",\"parameters\":{\"input\":{\"bw\":true,\"width\":28,\"height\":"
          "28},\"mllib\":{},\"output\":{\"index\":true,\"best\":2,\"index_"
          "type\":\"Flat\",\"index_gpu\":false}},\"data\":[\""
        + mnist_repo + "/sample_digit.png\"]}";
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  std::cout << "joutstr predict index=" << joutstr << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200, jd["status"]["code"]);
  ASSERT_TRUE(jd["body"]["predictions"][0].HasMember("indexed"));
  ASSERT_TRUE(jd["body"]["predictions"][0]["indexed"].GetBool());

  // build & save index
  jpredictstr = "{\"service\":\"" + sname
                + "\",\"parameters\":{\"input\":{\"bw\":true,\"width\":28,"
                  "\"height\":28},\"mllib\":{},\"output\":{\"build_index\":"
                  "true,\"best\":2}},\"data\":[\""
                + mnist_repo + "/sample_digit.png\"]}";
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  std::cout << "joutstr predict build index=" << joutstr << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200, jd["status"]["code"]);

  // assert existence of index
  ASSERT_TRUE(fileops::file_exists(mnist_repo + "index.faiss"));
  ASSERT_TRUE(fileops::file_exists(mnist_repo + "names.bin/data.mdb"));

  // search index
  jpredictstr = "{\"service\":\"" + sname
                + "\",\"parameters\":{\"input\":{\"bw\":true,\"width\":28,"
                  "\"height\":28},\"mllib\":{},\"output\":{\"search\":true,"
                  "\"best\":2}},\"data\":[\""
                + mnist_repo + "/sample_digit.png\"]}";
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  std::cout << "joutstr predict search=" << joutstr << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200, jd["status"]["code"]);
  // assert result is itself
  ASSERT_TRUE(jd["body"]["predictions"][0].HasMember("nns"));
  ASSERT_TRUE(jd["body"]["predictions"][0]["nns"][0]["dist"] == 0.0);
  ASSERT_TRUE(jd["body"]["predictions"][0]["nns"][0]["uri"]
              == "../examples/caffe/mnist//sample_digit.png");

  // remove service
  jstr = "{\"clear\":\"lib\"}";
  joutstr = japi.jrender(japi.service_delete(sname, jstr));
  ASSERT_EQ(ok_str, joutstr);

  // assert non-existence of index
  ASSERT_TRUE(!fileops::file_exists(mnist_repo + "index.faiss"));
  ASSERT_TRUE(!fileops::file_exists(mnist_repo + "names.bin/data.mdb"));
}

TEST(simsearch, predict_roi_simsearch)
{
  // create service
  JsonAPI japi;
  std::string sname = "my_service";
  std::string jstr
      = "{\"mllib\":\"caffe\",\"description\":\"my "
        "classifier\",\"type\":\"supervised\",\"model\":{\"repository\":\""
        + voc_repo
        + "\"},\"parameters\":{\"input\":{\"connector\":\"image\",\"width\":"
          "300,\"height\":300},\"mllib\":{\"nclasses\":21}}}";
  std::cerr << "jstr=" << jstr << std::endl;
  std::string joutstr = japi.jrender(japi.service_create(sname, jstr));
  ASSERT_EQ(created_str, joutstr);
  JDoc jd;

  // predict
  std::string jpredictstr
      = "{\"service\":\"" + sname
        + "\",\"parameters\":{\"input\":{},\"mllib\":{},\"output\":{\"rois\":"
          "\"rois\",\"confidence_threshold\":0.1,\"index\":true,\"index_"
          "type\":\"Flat\",\"index_gpu\":false}},\"data\":[\""
        + voc_repo + "/test_img.jpg\"]}";
  std::cerr << "jpredictstr=" << jpredictstr << std::endl;
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  std::cout << "joutstr predict index=" << joutstr << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200, jd["status"]["code"]);
  ASSERT_TRUE(jd["body"]["predictions"][0].HasMember("indexed"));
  ASSERT_TRUE(jd["body"]["predictions"][0]["indexed"].GetBool());

  // build & save index
  jpredictstr = "{\"service\":\"" + sname
                + "\",\"parameters\":{\"input\":{},\"mllib\":{},\"output\":{"
                  "\"build_index\":true,\"rois\":\"rois\",\"confidence_"
                  "threshold\":0.1}},\"data\":[\""
                + voc_repo + "/test_img.jpg\"]}";
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  std::cout << "joutstr predict build index=" << joutstr << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200, jd["status"]["code"]);

  // assert existence of index
  ASSERT_TRUE(fileops::file_exists(voc_repo + "index.faiss"));
  ASSERT_TRUE(fileops::file_exists(voc_repo + "names.bin/data.mdb"));

  // search index
  jpredictstr = "{\"service\":\"" + sname
                + "\",\"parameters\":{\"input\":{},\"mllib\":{},\"output\":{"
                  "\"search\":true,\"rois\":\"rois\",\"confidence_threshold\":"
                  "0.1}},\"data\":[\""
                + voc_repo + "/test_img.jpg\"]}";
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  std::cout << "joutstr predict search=" << joutstr << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200, jd["status"]["code"]);
  // assert result is itself
  ASSERT_TRUE(jd["body"]["predictions"][0]["rois"][0].HasMember("nns"));
  ASSERT_TRUE(jd["body"]["predictions"][0]["rois"][0]["nns"][0]["dist"]
              == 0.0);
  ASSERT_TRUE(jd["body"]["predictions"][0]["rois"][0]["nns"][0]["uri"]
              == "../examples/caffe/voc_roi/voc_roi//test_img.jpg");

  // remove service
  jstr = "{\"clear\":\"index\"}";
  joutstr = japi.jrender(japi.service_delete(sname, jstr));
  ASSERT_EQ(ok_str, joutstr);

  // assert non-existence of index
  ASSERT_TRUE(!fileops::file_exists(voc_repo + "index.faiss"));
  ASSERT_TRUE(!fileops::file_exists(voc_repo + "names.bin/data.mdb"));
}

TEST(simsearch, predict_chain)
{
  // create detection service
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

  // create simsearch service
  std::string mnist_sname = "simsearch";
  jstr = "{\"mllib\":\"caffe\",\"description\":\"similarity model\",\"type\":"
         "\"unsupervised\",\"model\":{\"repository\":\""
         + mnist_repo
         + "\"},\"parameters\":{\"input\":{\"connector\":\"image\",\"width\":"
           "28,\"height\":28},\"mllib\":{\"nclasses\":10}}}";
  joutstr = japi.jrender(japi.service_create(mnist_sname, jstr));
  ASSERT_EQ(created_str, joutstr);

  // train simsearch model
  std::string gpuid = "0";
  std::string jtrainstr
      = "{\"service\":\"" + mnist_sname
        + "\",\"async\":false,\"parameters\":{\"mllib\":{\"gpu\":true,"
          "\"gpuid\":"
        + gpuid + ",\"solver\":{\"iterations\":" + iterations_mnist
        + ",\"snapshot\":200,\"snapshot_prefix\":\"" + mnist_repo
        + "/mylenet\",\"test_interval\":2}},\"output\":{\"measure_hist\":true,"
          "\"measure\":[\"f1\"]}}}";
  joutstr = japi.jrender(japi.service_train(jtrainstr));
  std::cout << "joutstr=" << joutstr << std::endl;
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
  ASSERT_TRUE(jd["body"].HasMember("measure"));
  ASSERT_TRUE(fabs(jd["body"]["measure"]["train_loss"].GetDouble()) > 0);
  ASSERT_EQ(jd["body"]["measure_hist"]["iteration_hist"].Size(),
            jd["body"]["measure_hist"]["f1_hist"].Size());

  // build & save index
  std::string jpredictstr
      = "{\"service\":\"" + mnist_sname
        + "\",\"parameters\":{\"input\":{\"bw\":true,\"width\":28,"
          "\"height\":28},\"mllib\":{\"extract_layer\":\"ip2\"},"
          "\"output\":{\"index\":true,\"index_type\":\"Flat\",\"index_"
          "gpu\":false,\"build_index\":true}},\"data\":[\""
        + mnist_repo + "sample_digit.png\"]}";
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  std::cout << "joutstr predict build index=" << joutstr << std::endl;
  jd = JDoc();
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200, jd["status"]["code"]);

  // chain predict
  std::string jchainstr
      = "{\"chain\":{\"name\":\"chain\",\"calls\":["
        "{\"service\":\""
        + detect_sname
        + "\",\"parameters\":{\"input\":{\"keep_orig\":true},\"output\":{"
          "\"bbox\":true,\"confidence_threshold\":0.2}},\"data\":[\""
        + caffe_word_detect_repo
        + "word.png\"]},"
          "{\"id\":\"filter\",\"action\":{\"type\":\"filter\",\"parameters\":{"
          "\"classes\":[\"1\"]}}},"
          "{\"id\":\"crop\",\"parent_id\":\"filter\",\"action\":{\"type\":"
          "\"crop\"}},"
          "{\"service\":\""
        + mnist_sname
        + "\",\"parent_id\":\"crop\",\"parameters\":{\"input\":{\"bw\":true},"
          "\"mllib\":{\"extract_layer\":\"ip2\"},\"output\":{\"confidence_"
          "threshold\":0.1,\"search\":true,\"search_nn\":1}}}"
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
  ASSERT_EQ(jd["body"]["predictions"][0]["classes"].Size(), 1);
  ASSERT_TRUE(jd["body"]["predictions"][0]["classes"][0][mnist_sname.c_str()]
                  .IsObject());
  auto &nn_pred
      = jd["body"]["predictions"][0]["classes"][0][mnist_sname.c_str()];
  ASSERT_EQ(nn_pred["nns"].Size(), 1);
  ASSERT_TRUE(nn_pred["nns"][0]["dist"].GetDouble() > 0);
  ASSERT_EQ(nn_pred["nns"][0]["uri"].GetString(),
            mnist_repo + "sample_digit.png");

  // remove service
  jstr = "{\"clear\":\"lib\"}";
  joutstr = japi.jrender(japi.service_delete(mnist_sname, jstr));
  ASSERT_EQ(ok_str, joutstr);

  // assert non-existence of index
  ASSERT_TRUE(!fileops::file_exists(mnist_repo + "index.faiss"));
  ASSERT_TRUE(!fileops::file_exists(mnist_repo + "names.bin/data.mdb"));
}
