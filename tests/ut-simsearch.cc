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
static std::string created_str = "{\"status\":{\"code\":201,\"msg\":\"Created\"}}";
static std::string bad_param_str = "{\"status\":{\"code\":400,\"msg\":\"BadRequest\"}}";
static std::string not_found_str = "{\"status\":{\"code\":404,\"msg\":\"NotFound\"}}";

static std::string mnist_repo = "../examples/caffe/mnist/";
static std::string iterations_mnist = "2";
static std::string voc_repo = "../examples/caffe/voc_roi/voc_roi/";

TEST(annoyse,index_search)
{
  std::vector<double> vec1 = {1.0,0.0,0.0,0.0};
  std::vector<double> vec2 = {0.0,1.0,0.0,0.0};
  std::vector<double> vec3 = {1.0,0.0,1.0,0.0};
  
  int t = 4;
  std::string model_repo = "simsearch";
  mkdir(model_repo.c_str(),0770);
  AnnoySE ase(t,model_repo);
  ase.create_index(); // index creation
  ase.index(URIData("test1"),vec1); // indexing data
  ase.index(URIData("test2"),vec2);
  ase.index(URIData("test3"),vec3);
  std::vector<URIData> uris;
  std::vector<double> distances;
  ase.build_tree(); // tree building
  ase.save_tree(); // tree saving
  ase.search(vec1,3,uris,distances); // searching nearest neighbors
  std::cerr << "search uris size=" << uris.size() << std::endl;
  for (size_t i=0;i<uris.size();i++)
    std::cout << uris.at(i)._uri << " / distances=" << distances.at(i) << std::endl;
  ase.remove_index();
  rmdir(model_repo.c_str());
}

TEST(annoyse,index_search_incr)
{
  std::vector<double> vec1 = {1.0,0.0,0.0,0.0};
  std::vector<double> vec2 = {0.0,1.0,0.0,0.0};
  std::vector<double> vec3 = {1.0,0.0,1.0,0.0};
  std::vector<double> vec4 = {0.0,0.0,5.0,5.0};
  
  int t = 4;
  std::string model_repo = "simsearch";
  mkdir(model_repo.c_str(),0770);
  AnnoySE ase(t,model_repo);
  ase.create_index();
  ase.index(URIData("test1"),vec1);
  ase.index(URIData("test2"),vec2);
  ase.index(URIData("test3"),vec3);
  std::vector<URIData> uris;
  std::vector<double> distances;
  ase.build_tree();
  ase.search(vec1,3,uris,distances);
  std::cerr << "search uris size=" << uris.size() << std::endl;
  for (size_t i=0;i<uris.size();i++)
    std::cout << uris.at(i)._uri << " / distances=" << distances.at(i) << std::endl;
  uris.clear();
  distances.clear();
  ase.unbuild_tree();
  ase.index(URIData("test4"),vec4);
  ase.build_tree();
  ase.search(vec4,3,uris,distances);
  std::cerr << "search uris size=" << uris.size() << std::endl;
  for (size_t i=0;i<uris.size();i++)
    std::cout << uris.at(i)._uri << " / distances=" << distances.at(i) << std::endl;
  ase.remove_index();
  rmdir(model_repo.c_str());
}

TEST(simsearch,predict_simsearch_unsup)
{
  // create service
  JsonAPI japi;
  std::string sname = "my_service";
  std::string jstr = "{\"mllib\":\"caffe\",\"description\":\"my classifier\",\"type\":\"unsupervised\",\"model\":{\"repository\":\"" +  mnist_repo + "\"},\"parameters\":{\"input\":{\"connector\":\"image\"},\"mllib\":{\"nclasses\":10}}}";
  std::string joutstr = japi.jrender(japi.service_create(sname,jstr));
  ASSERT_EQ(created_str,joutstr);
  JDoc jd;
  
  // train
  std::string gpuid = "0";
  std::string jtrainstr = "{\"service\":\"" + sname + "\",\"async\":false,\"parameters\":{\"mllib\":{\"gpu\":true,\"gpuid\":"+gpuid+",\"solver\":{\"iterations\":" + iterations_mnist + ",\"snapshot\":200,\"snapshot_prefix\":\"" + mnist_repo + "/mylenet\",\"test_interval\":2}},\"output\":{\"measure_hist\":true,\"measure\":[\"f1\"]}}}";
  joutstr = japi.jrender(japi.service_train(jtrainstr));
  std::cout << "joutstr=" << joutstr << std::endl;
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
  ASSERT_TRUE(fabs(jd["body"]["measure"]["train_loss"].GetDouble()) > 0);
  ASSERT_EQ(jd["body"]["measure_hist"]["iteration_hist"].Size(),jd["body"]["measure_hist"]["f1_hist"].Size());
  
  // predict
  std::string jpredictstr = "{\"service\":\""+ sname + "\",\"parameters\":{\"input\":{\"bw\":true,\"width\":28,\"height\":28},\"mllib\":{\"extract_layer\":\"ip2\"},\"output\":{\"index\":true}},\"data\":[\"" + mnist_repo + "/sample_digit.png\"]}";
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  std::cout << "joutstr predict index=" << joutstr << std::endl;
  jd.Parse(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200,jd["status"]["code"]);
  ASSERT_TRUE(jd["body"]["predictions"][0].HasMember("indexed"));
  ASSERT_TRUE(jd["body"]["predictions"][0]["indexed"].GetBool());
  
  // build & save index
  jpredictstr = "{\"service\":\""+ sname + "\",\"parameters\":{\"input\":{\"bw\":true,\"width\":28,\"height\":28},\"mllib\":{\"extract_layer\":\"ip2\"},\"output\":{\"build_index\":true}},\"data\":[\"" + mnist_repo + "/sample_digit.png\"]}";
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  std::cout << "joutstr predict build index=" << joutstr << std::endl;
  jd.Parse(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200,jd["status"]["code"]);

  // assert existence of index
  ASSERT_TRUE(fileops::file_exists(mnist_repo + "index.ann"));
  ASSERT_TRUE(fileops::file_exists(mnist_repo + "names.bin/data.mdb"));

  // assert error on indexing over a built index
  jpredictstr = "{\"service\":\""+ sname + "\",\"parameters\":{\"input\":{\"bw\":true,\"width\":28,\"height\":28},\"mllib\":{\"extract_layer\":\"ip2\"},\"output\":{\"index\":true}},\"data\":[\"" + mnist_repo + "/sample_digit.png\"]}";
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  std::cout << "joutstr predict index=" << joutstr << std::endl;
  jd.Parse(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(403,jd["status"]["code"]);
  
  // search index
  jpredictstr = "{\"service\":\""+ sname + "\",\"parameters\":{\"input\":{\"bw\":true,\"width\":28,\"height\":28},\"mllib\":{\"extract_layer\":\"ip2\"},\"output\":{\"search\":true}},\"data\":[\"" + mnist_repo + "/sample_digit.png\"]}";
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  std::cout << "joutstr predict search=" << joutstr << std::endl;
  jd.Parse(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200,jd["status"]["code"]);
  // assert result is itself
  ASSERT_TRUE(jd["body"]["predictions"][0].HasMember("nns"));
  ASSERT_TRUE(jd["body"]["predictions"][0]["nns"][0]["dist"]==0.0);
  ASSERT_TRUE(jd["body"]["predictions"][0]["nns"][0]["uri"]=="../examples/caffe/mnist//sample_digit.png");
  
  // remove service
  jstr = "{\"clear\":\"lib\"}";
  joutstr = japi.jrender(japi.service_delete(sname,jstr));
  ASSERT_EQ(ok_str,joutstr);

  // assert non-existence of index
  ASSERT_TRUE(!fileops::file_exists(mnist_repo + "index.ann"));
  ASSERT_TRUE(!fileops::file_exists(mnist_repo + "names.bin/data.mdb"));
}

TEST(simsearch,predict_simsearch_sup)
{
  // create service
  JsonAPI japi;
  std::string sname = "my_service";
  std::string jstr = "{\"mllib\":\"caffe\",\"description\":\"my classifier\",\"type\":\"supervised\",\"model\":{\"repository\":\"" +  mnist_repo + "\"},\"parameters\":{\"input\":{\"connector\":\"image\"},\"mllib\":{\"nclasses\":10}}}";
  std::string joutstr = japi.jrender(japi.service_create(sname,jstr));
  ASSERT_EQ(created_str,joutstr);
  JDoc jd;
  
  // train
  std::string gpuid = "0";
  std::string jtrainstr = "{\"service\":\"" + sname + "\",\"async\":false,\"parameters\":{\"mllib\":{\"gpu\":true,\"gpuid\":"+gpuid+",\"solver\":{\"iterations\":" + iterations_mnist + ",\"snapshot\":200,\"snapshot_prefix\":\"" + mnist_repo + "/mylenet\",\"test_interval\":2}},\"output\":{\"measure_hist\":true,\"measure\":[\"f1\"]}}}";
  joutstr = japi.jrender(japi.service_train(jtrainstr));
  std::cout << "joutstr=" << joutstr << std::endl;
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
  ASSERT_TRUE(fabs(jd["body"]["measure"]["train_loss"].GetDouble()) > 0);
  ASSERT_EQ(jd["body"]["measure_hist"]["iteration_hist"].Size(),jd["body"]["measure_hist"]["f1_hist"].Size());
  
  // predict
  std::string jpredictstr = "{\"service\":\""+ sname + "\",\"parameters\":{\"input\":{\"bw\":true,\"width\":28,\"height\":28},\"mllib\":{},\"output\":{\"index\":true,\"best\":2}},\"data\":[\"" + mnist_repo + "/sample_digit.png\"]}";
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  std::cout << "joutstr predict index=" << joutstr << std::endl;
  jd.Parse(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200,jd["status"]["code"]);
  ASSERT_TRUE(jd["body"]["predictions"][0].HasMember("indexed"));
  ASSERT_TRUE(jd["body"]["predictions"][0]["indexed"].GetBool());
  
  // build & save index
  jpredictstr = "{\"service\":\""+ sname + "\",\"parameters\":{\"input\":{\"bw\":true,\"width\":28,\"height\":28},\"mllib\":{},\"output\":{\"build_index\":true,\"best\":2}},\"data\":[\"" + mnist_repo + "/sample_digit.png\"]}";
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  std::cout << "joutstr predict build index=" << joutstr << std::endl;
  jd.Parse(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200,jd["status"]["code"]);

  // assert existence of index
  ASSERT_TRUE(fileops::file_exists(mnist_repo + "index.ann"));
  ASSERT_TRUE(fileops::file_exists(mnist_repo + "names.bin/data.mdb"));

  // assert error on indexing over a built index
  jpredictstr = "{\"service\":\""+ sname + "\",\"parameters\":{\"input\":{\"bw\":true,\"width\":28,\"height\":28},\"mllib\":{},\"output\":{\"index\":true,\"best\":2}},\"data\":[\"" + mnist_repo + "/sample_digit.png\"]}";
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  std::cout << "joutstr predict index=" << joutstr << std::endl;
  jd.Parse(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(403,jd["status"]["code"]);
  
  // search index
  jpredictstr = "{\"service\":\""+ sname + "\",\"parameters\":{\"input\":{\"bw\":true,\"width\":28,\"height\":28},\"mllib\":{},\"output\":{\"search\":true,\"best\":2}},\"data\":[\"" + mnist_repo + "/sample_digit.png\"]}";
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  std::cout << "joutstr predict search=" << joutstr << std::endl;
  jd.Parse(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200,jd["status"]["code"]);
  // assert result is itself
  ASSERT_TRUE(jd["body"]["predictions"][0].HasMember("nns"));
  ASSERT_TRUE(jd["body"]["predictions"][0]["nns"][0]["dist"]==0.0);
  ASSERT_TRUE(jd["body"]["predictions"][0]["nns"][0]["uri"]=="../examples/caffe/mnist//sample_digit.png");
  
  // remove service
  jstr = "{\"clear\":\"lib\"}";
  joutstr = japi.jrender(japi.service_delete(sname,jstr));
  ASSERT_EQ(ok_str,joutstr);

  // assert non-existence of index
  ASSERT_TRUE(!fileops::file_exists(mnist_repo + "index.ann"));
  ASSERT_TRUE(!fileops::file_exists(mnist_repo + "names.bin/data.mdb"));
}

TEST(simsearch,predict_roi_simsearch)
{
  // create service
  JsonAPI japi;
  std::string sname = "my_service";
  std::string jstr = "{\"mllib\":\"caffe\",\"description\":\"my classifier\",\"type\":\"supervised\",\"model\":{\"repository\":\"" +  voc_repo + "\"},\"parameters\":{\"input\":{\"connector\":\"image\",\"width\":300,\"height\":300},\"mllib\":{\"nclasses\":21}}}";
  std::cerr << "jstr=" << jstr << std::endl;
  std::string joutstr = japi.jrender(japi.service_create(sname,jstr));
  ASSERT_EQ(created_str,joutstr);
  JDoc jd;

  // predict
  std::string jpredictstr = "{\"service\":\""+ sname + "\",\"parameters\":{\"input\":{},\"mllib\":{},\"output\":{\"rois\":\"rois\",\"confidence_threshold\":0.1,\"index\":true}},\"data\":[\"" + voc_repo + "/test_img.jpg\"]}";
  std::cerr << "jpredictstr=" << jpredictstr << std::endl;
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  std::cout << "joutstr predict index=" << joutstr << std::endl;
  jd.Parse(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200,jd["status"]["code"]);
  ASSERT_TRUE(jd["body"]["predictions"][0].HasMember("indexed"));
  ASSERT_TRUE(jd["body"]["predictions"][0]["indexed"].GetBool());
  
  // build & save index
  jpredictstr = "{\"service\":\""+ sname + "\",\"parameters\":{\"input\":{},\"mllib\":{},\"output\":{\"build_index\":true,\"rois\":\"rois\",\"confidence_threshold\":0.1}},\"data\":[\"" + voc_repo + "/test_img.jpg\"]}";
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  std::cout << "joutstr predict build index=" << joutstr << std::endl;
  jd.Parse(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200,jd["status"]["code"]);

  // assert existence of index
  ASSERT_TRUE(fileops::file_exists(voc_repo + "index.ann"));
  ASSERT_TRUE(fileops::file_exists(voc_repo + "names.bin/data.mdb"));

  // search index
  jpredictstr = "{\"service\":\""+ sname + "\",\"parameters\":{\"input\":{},\"mllib\":{},\"output\":{\"search\":true,\"rois\":\"rois\",\"confidence_threshold\":0.1}},\"data\":[\"" + voc_repo + "/test_img.jpg\"]}";
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  std::cout << "joutstr predict search=" << joutstr << std::endl;
  jd.Parse(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200,jd["status"]["code"]);
  // assert result is itself
  ASSERT_TRUE(jd["body"]["predictions"][0]["rois"][0].HasMember("nns"));
  ASSERT_TRUE(jd["body"]["predictions"][0]["rois"][0]["nns"][0]["dist"]==0.0);
  ASSERT_TRUE(jd["body"]["predictions"][0]["rois"][0]["nns"][0]["uri"]=="../examples/caffe/voc_roi/voc_roi//test_img.jpg");
  
  // remove service
  jstr = "{\"clear\":\"index\"}";
  joutstr = japi.jrender(japi.service_delete(sname,jstr));
  ASSERT_EQ(ok_str,joutstr);

  // assert non-existence of index
  ASSERT_TRUE(!fileops::file_exists(voc_repo + "index.ann"));
  ASSERT_TRUE(!fileops::file_exists(voc_repo + "names.bin/data.mdb"));
}
