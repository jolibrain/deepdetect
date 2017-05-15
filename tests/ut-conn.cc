/**
 * DeepDetect
 * Copyright (c) 2014-2015 Emmanuel Benazera
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

#include "apidata.h"
#include "imginputfileconn.h"
#include "csvinputfileconn.h"
#include "txtinputfileconn.h"
#include "outputconnectorstrategy.h"
#include "jsonapi.h"
#include <gtest/gtest.h>
#include <iostream>

using namespace dd;

TEST(outputconn,acc)
{
  std::vector<double> targets = {0, 0, 1, 1};
  std::vector<double> pred1 = {0.7, 0.3};
  std::vector<double> pred2 = {0.4, 0.5};
  std::vector<double> pred3 = {0.1, 0.9};
  std::vector<double> pred4 = {0.2, 0.8};
  std::vector<std::vector<double>> preds = { pred1, pred2, pred3, pred4 };
  APIData res_ad;
  res_ad.add("batch_size",static_cast<int>(targets.size()));
  for (size_t i=0;i<targets.size();i++)
    {
      APIData bad;
      bad.add("pred",preds.at(i));
      bad.add("target",targets.at(i));
      std::vector<APIData> vad = {bad};
      res_ad.add(std::to_string(i),vad);
    }
  SupervisedOutput so;
  std::vector<std::string> measures = {"acc"};
  std::map<std::string,double> accs = so.acc(res_ad,measures);
  ASSERT_EQ(0.75,accs["acc"]);
}

TEST(outputconn,acck)
{
  std::vector<double> targets = {0, 0, 1, 2};
  std::vector<double> pred1 = {0.7, 0.1, 0.1, 0.1};
  std::vector<double> pred2 = {0.3, 0.5, 0.1, 0.2};
  std::vector<double> pred3 = {0.1, 0.9, 0.0, 0.0};
  std::vector<double> pred4 = {0.1, 0.7, 0.05, 0.15};
  std::vector<std::vector<double>> preds = { pred1, pred2, pred3, pred4 };
  APIData res_ad;
  res_ad.add("batch_size",static_cast<int>(targets.size()));
  for (size_t i=0;i<targets.size();i++)
    {
      APIData bad;
      bad.add("pred",preds.at(i));
      bad.add("target",targets.at(i));
      std::vector<APIData> vad = {bad};
      res_ad.add(std::to_string(i),vad);
    }
  SupervisedOutput so;
  std::vector<std::string> measures = {"acc","acc-2","acc-3"};
  std::map<std::string,double> accs = so.acc(res_ad,measures);
  ASSERT_EQ(0.5,accs["acc"]);
  ASSERT_EQ(0.75,accs["acc-2"]);
  ASSERT_EQ(1.0,accs["acc-3"]);
}

TEST(outputconn,auc)
{
  std::vector<double> targets = {0, 0, 1, 1};
  std::vector<double> pred1 = {0.9, 0.1};
  std::vector<double> pred2 = {0.6, 0.4};
  std::vector<double> pred3 = {0.65, 0.35};
  std::vector<double> pred4 = {0.2, 0.8};
  std::vector<std::vector<double>> preds = { pred1, pred2, pred3, pred4 };
  APIData res_ad;
  res_ad.add("batch_size",static_cast<int>(targets.size()));
  for (size_t i=0;i<targets.size();i++)
    {
      APIData bad;
      bad.add("pred",preds.at(i));
      bad.add("target",targets.at(i));
      std::vector<APIData> vad = {bad};
      res_ad.add(std::to_string(i),vad);
    }
  SupervisedOutput so;
  double auc = so.auc(res_ad);
  ASSERT_EQ(0.75,auc);
}

TEST(outputconn,mcc)
{
  std::vector<double> targets = {0, 0, 1, 1};
  std::vector<double> pred1 = {0.9, 0.1};
  std::vector<double> pred2 = {0.6, 0.4};
  std::vector<double> pred3 = {0.65, 0.35};
  std::vector<double> pred4 = {0.2, 0.8};
  std::vector<std::vector<double>> preds = { pred1, pred2, pred3, pred4 };
  APIData res_ad;
  res_ad.add("nclasses",2);
  res_ad.add("batch_size",static_cast<int>(targets.size()));
  for (size_t i=0;i<targets.size();i++)
    {
      APIData bad;
      bad.add("pred",preds.at(i));
      bad.add("target",targets.at(i));
      std::vector<APIData> vad = {bad};
      res_ad.add(std::to_string(i),vad);
    }
  SupervisedOutput so;
  double mcc = so.mcc(res_ad);
  ASSERT_NEAR(0.57735,mcc,1e-3);
}

TEST(outputconn,gini)
{
  std::vector<double> targets = {1,1,0,1};
  std::vector<double> pred1 = {0.86};
  std::vector<double> pred2 = {0.26};
  std::vector<double> pred3 = {0.52};
  std::vector<double> pred4 = {0.32};
  std::vector<std::vector<double>> preds = { pred1, pred2, pred3, pred4 };
  APIData res_ad;
  res_ad.add("batch_size",static_cast<int>(targets.size()));
  for (size_t i=0;i<targets.size();i++)
    {
      APIData bad;
      bad.add("pred",preds.at(i));
      bad.add("target",targets.at(i));
      std::vector<APIData> vad = {bad};
      res_ad.add(std::to_string(i),vad);
    }
  double gini = SupervisedOutput::gini(res_ad,true);
  std::cout << "gini=" << gini << std::endl;
  ASSERT_NEAR(-0.333,gini,1e-3);
}

TEST(outputconn,cmfull)
{
  std::vector<double> targets = {0, 0, 1, 2};
  std::vector<double> pred1 = {0.7, 0.1, 0.1, 0.1};
  std::vector<double> pred2 = {0.3, 0.5, 0.1, 0.2};
  std::vector<double> pred3 = {0.1, 0.9, 0.0, 0.0};
  std::vector<double> pred4 = {0.1, 0.7, 0.05, 0.15};
  std::vector<std::vector<double>> preds = { pred1, pred2, pred3, pred4 };
  APIData res_ad;
  res_ad.add("nclasses",4);
  res_ad.add("batch_size",static_cast<int>(targets.size()));
  for (size_t i=0;i<targets.size();i++)
    {
      APIData bad;
      bad.add("pred",preds.at(i));
      bad.add("target",targets.at(i));
      std::vector<APIData> vad = {bad};
      res_ad.add(std::to_string(i),vad);
    }
  SupervisedOutput so;
  std::vector<std::string> measures = {"f1","cmdiag","cmfull"};
  std::vector<std::string> clnames = {"zero","one","two","three"};
  res_ad.add("clnames",clnames);
  APIData ad_out;
  ad_out.add("measure",measures);
  
  APIData out;
  SupervisedOutput::measure(res_ad,ad_out,out);
  APIData meas_out = out.getobj("measure");
  auto lkeys = meas_out.list_keys();
  /*std::cerr << "lkeys size=" << lkeys.size() << std::endl;
  for (auto k: lkeys)
  std::cerr << k << std::endl;*/
  ASSERT_EQ(0.5,meas_out.get("accp").get<double>());
  
  // cmfull
  JsonAPI japi;
  JDoc jpred = japi.dd_ok_200();
  JVal jout(rapidjson::kObjectType);
  out.toJVal(jpred,jout);
  std::string jstr = japi.jrender(jout);
  std::cerr << "jstr=" << jstr << std::endl;
  ASSERT_EQ("{\"measure\":{\"labels\":[\"zero\",\"one\",\"two\",\"three\"],\"f1\":0.35294117352941187,\"cmfull\":[{\"zero\":[0.5,0.5,0.0,0.0]},{\"one\":[0.0,1.0,0.0,0.0]},{\"two\":[0.0,1.0,0.0,0.0]},{\"three\":[2.696539702293474e308,2.696539702293474e308,2.696539702293474e308,2.696539702293474e308]}],\"cmdiag\":[0.4999999975,0.9999999900000002,0.0,0.0],\"recall\":0.3333333305555556,\"precision\":0.3749999968750001,\"accp\":0.5}}",jstr);
}

TEST(inputconn,img)
{
  std::string mnist_repo = "../examples/caffe/mnist/";
  APIData ad;
  std::vector<std::string> uris = {mnist_repo + "/sample_digit.png","https://deepdetect.com/dd/examples/caffe/mnist/sample_digit.png"};
  ad.add("data",uris);
  ImgInputFileConn iifc;
  try
    {
      iifc.transform(ad);
    }
  catch (InputConnectorBadParamException &e)
    {
      std::cerr << e.what() << std::endl;
      ASSERT_FALSE(true); // trigger
    }
  for (auto u: iifc._uris)
    std::cerr << u << std::endl;
  ASSERT_EQ(2,iifc._uris.size());
  ASSERT_EQ(2,iifc._images.size());
  cv::Mat diff;
  cv::compare(iifc._images.at(0),iifc._images.at(1),diff,cv::CMP_NE);
  std::vector<cv::Mat> channels(3);
  cv::split(diff,channels);
  for (int i=0;i<3;i++)
    ASSERT_TRUE(cv::countNonZero(channels.at(i))==0); // the two images must be identical
}

//TODO: test csv scale, separator, categorical, ...
TEST(inputconn,csv_mem1)
{
  std::string no_header = "2590,56,2,212,5";
  std::vector<std::string> vdata = { no_header };
  APIData ad;
  ad.add("data",vdata);
  CSVInputFileConn cifc;
  cifc._train = false; // prediction mode
  try
    {
      cifc.transform(ad);
    }
  catch (std::exception &e)
    {
      std::cerr << "exception=" << e.what() << std::endl;
      ASSERT_FALSE(true);
    }
  ASSERT_EQ(1,cifc._uris.size());
  ASSERT_EQ(1,cifc._csvdata.size());
  ASSERT_EQ(5,cifc._csvdata.at(0)._v.size());
  ASSERT_EQ(2590,cifc._csvdata.at(0)._v.at(0));
}

TEST(inputconn,csv_mem2)
{
  std::string header = "id,val1,val2,val3,val4,val5";
  std::string d1 = "2,2590,56,2,212,5";
  std::vector<std::string> vdata = { header, d1 };
  APIData ad;
  ad.add("data",vdata);
  APIData pad,pinp;
  pinp.add("id","id");
  std::vector<APIData> vpinp = { pinp };
  pad.add("input",vpinp);
  std::vector<APIData> vpad = { pad };
  ad.add("parameters",vpad);
  CSVInputFileConn cifc;
  cifc._train = false;
  try
    {
      cifc.transform(ad);
    }
  catch (std::exception &e)
    {
      std::cerr << "exception=" << e.what() << std::endl;
      ASSERT_FALSE(true);
    }
  ASSERT_EQ(2,cifc._uris.size());
  ASSERT_EQ(1,cifc._csvdata.size());
  ASSERT_TRUE(!cifc._columns.empty());
  ASSERT_EQ(6,cifc._csvdata.at(0)._v.size());
  ASSERT_EQ(2,cifc._csvdata.at(0)._v.at(0));
  ASSERT_EQ(2590,cifc._csvdata.at(0)._v.at(1));
}

TEST(inputconn,csv_copy)
{
  std::string header = "id,val1,val2,val3,val4,val5";
  std::string d1 = "2,2590,56,2,212,5";
  std::vector<std::string> vdata = { header, d1 };
  APIData ad;
  ad.add("data",vdata);
  APIData pad,pinp;
  pinp.add("id","id");
  pinp.add("label","val5");
  std::vector<APIData> vpinp = { pinp };
  pad.add("input",vpinp);
  std::vector<APIData> vpad = { pad };
  ad.add("parameters",vpad);
  CSVInputFileConn cifc;
  cifc.init(ad.getobj("parameters").getobj("input"));
  CSVInputFileConn cifc2 = cifc;
  ASSERT_EQ("val5",cifc2._label[0]);
  ASSERT_EQ("id",cifc2._id);
}

TEST(inputconn,csv_categoricals1)
{
  std::string header = "target,cap-shape,cap-surface,cap-color,bruises";
  std::string d1 = "p,x,s,n,t";
  std::ofstream of("test.csv");
  of << header << std::endl;
  of << d1 << std::endl;
  std::vector<std::string> vdata = { "test.csv" };
  APIData ad;
  ad.add("data",vdata);
  APIData pad,pinp;
  pinp.add("label","target");
  std::vector<std::string> vcats = {"target","cap-shape","cap-surface","cap-color","bruises"};
  pinp.add("categoricals",vcats);
  std::vector<APIData> vpinp = { pinp };
  pad.add("input",vpinp);
  std::vector<APIData> vpad = { pad };
  ad.add("parameters",vpad);
  CSVInputFileConn cifc;
  cifc._train = true;
  try
    {
      std::cout << "trying to transform\n";
      cifc.transform(ad);
    }
  catch(InputConnectorBadParamException &e)
    {
      std::cerr << "exception=" << e.what() << std::endl;
      ASSERT_FALSE(true);
    }
  ASSERT_EQ(1,cifc._csvdata.size());
  ASSERT_EQ(5,cifc._csvdata.at(0)._v.size());
  for (double e: cifc._csvdata.at(0)._v)
    ASSERT_EQ(1.0,e);
  ASSERT_EQ(5,cifc._columns.size());
  ASSERT_EQ("target_p",(*cifc._columns.begin()));
  remove("test.csv");
}

TEST(inputconn,csv_categoricals2)
{
  std::string header = "target,cap-shape,cap-surface,cap-color,bruises";
  std::string d1 = "p,x,s,n,t\ne,x,s,y,t\ne,b,s,w,t";
  std::ofstream of("test.csv");
  of << header << std::endl;
  of << d1 << std::endl;
  std::vector<std::string> vdata = { "test.csv" };
  APIData ad;
  ad.add("data",vdata);
  APIData pad,pinp;
  pinp.add("label","target");
  std::vector<std::string> vcats = {"target","cap-shape","cap-surface","cap-color","bruises"};
  pinp.add("categoricals",vcats);
  std::vector<APIData> vpinp = { pinp };
  pad.add("input",vpinp);
  std::vector<APIData> vpad = { pad };
  ad.add("parameters",vpad);
  CSVInputFileConn cifc;
  cifc._train = true;
  cifc.transform(ad);
  ASSERT_EQ(3,cifc._csvdata.size());
  ASSERT_EQ(9,cifc._csvdata.at(0)._v.size());
  std::vector<double> v1 = {1,0,1,0,1,1,0,0,1};
  std::vector<double> v2 = {0,1,1,0,1,0,1,0,1};
  std::vector<double> v3 = {0,1,0,1,1,0,0,1,1};
  ASSERT_EQ(v1,cifc._csvdata.at(0)._v);
  ASSERT_EQ(v2,cifc._csvdata.at(1)._v);
  ASSERT_EQ(v3,cifc._csvdata.at(2)._v);
  ASSERT_EQ(9,cifc._columns.size());
  ASSERT_EQ("target_p",(*cifc._columns.begin()));
  remove("test.csv");
}

TEST(inputconn,csv_read_categoricals)
{
  std::string json_categorical_mapping = "{\"categoricals_mapping\":{\"bruises\":{\"t\":0,\"f\":1},\"cap-color\":{\"n\":0,\"y\":1,\"p\":5,\"c\":8,\"r\":9,\"w\":2,\"g\":3,\"e\":4,\"b\":6,\"u\":7},\"ring-number\":{\"o\":0,\"t\":1,\"n\":2},\"odor\":{\"p\":0,\"c\":5,\"y\":6,\"s\":7,\"a\":1,\"l\":2,\"n\":3,\"f\":4,\"m\":8},\"stalk-shape\":{\"e\":0,\"t\":1},\"stalk-surface-above-ring\":{\"s\":0,\"y\":3,\"f\":1,\"k\":2},\"gill-size\":{\"n\":0,\"b\":1},\"veil-type\":{\"p\":0},\"ring-type\":{\"p\":0,\"e\":1,\"l\":2,\"f\":3,\"n\":4},\"spore-print-color\":{\"k\":0,\"n\":1,\"u\":2,\"h\":3,\"w\":4,\"r\":5,\"y\":7,\"o\":6,\"b\":8},\"gill-color\":{\"b\":8,\"e\":7,\"o\":11,\"k\":0,\"n\":1,\"y\":10,\"g\":2,\"r\":9,\"w\":4,\"p\":3,\"h\":5,\"u\":6},\"gill-spacing\":{\"c\":0,\"w\":1},\"habitat\":{\"u\":0,\"d\":3,\"l\":6,\"g\":1,\"m\":2,\"p\":4,\"w\":5},\"cap-surface\":{\"s\":0,\"y\":1,\"f\":2,\"g\":3},\"gill-attachment\":{\"f\":0,\"a\":1},\"cap-shape\":{\"x\":0,\"s\":2,\"c\":5,\"b\":1,\"f\":3,\"k\":4},\"target\":{\"p\":0,\"e\":1},\"stalk-root\":{\"e\":0,\"b\":2,\"c\":1,\"r\":3,\"?\":4},\"stalk-surface-below-ring\":{\"s\":0,\"y\":2,\"f\":1,\"k\":3},\"veil-color\":{\"w\":0,\"n\":1,\"o\":2,\"y\":3},\"population\":{\"s\":0,\"y\":4,\"c\":5,\"n\":1,\"a\":2,\"v\":3},\"stalk-color-above-ring\":{\"w\":0,\"g\":1,\"p\":2,\"c\":7,\"y\":8,\"n\":3,\"b\":4,\"e\":5,\"o\":6},\"stalk-color-below-ring\":{\"w\":0,\"p\":1,\"y\":6,\"c\":8,\"g\":2,\"b\":3,\"e\":5,\"n\":4,\"o\":7}}}";
  JDoc d;
  d.Parse(json_categorical_mapping.c_str());
  ASSERT_FALSE(d.HasParseError());
  APIData ap(d);
  ASSERT_EQ(1,ap.size());
  ASSERT_TRUE(ap.has("categoricals_mapping"));
  APIData ap_cm = ap.getobj("categoricals_mapping");
  CSVInputFileConn cifc;
  cifc._train = true;
  cifc.read_categoricals(ap);
  ASSERT_EQ(23,cifc._categoricals.size());
  CCategorical cc = cifc._categoricals["odor"];
  ASSERT_EQ(9,cc._vals.size());
}

TEST(inputconn,csv_ignore)
{
  std::string header = "target,cap-shape,cap-surface,cap-color,bruises";
  std::string d1 = "1,2,3,4,5";
  std::ofstream of("test.csv");
  of << header << std::endl;
  of << d1 << std::endl;
  std::vector<std::string> vdata = { "test.csv" };
  APIData ad;
  ad.add("data",vdata);
  APIData pad,pinp;
  pinp.add("label","target");
  std::vector<std::string> vign = {"cap-shape"};
  pinp.add("ignore",vign);
  std::vector<APIData> vpinp = { pinp };
  pad.add("input",vpinp);
  std::vector<APIData> vpad = { pad };
  ad.add("parameters",vpad);
  CSVInputFileConn cifc;
  cifc._train = true;
  try
    {
      cifc.transform(ad);
    }
  catch(InputConnectorBadParamException &e)
    {
      std::cerr << "exception=" << e.what() << std::endl;
      ASSERT_FALSE(true);
    }
  ASSERT_EQ(1,cifc._csvdata.size());
  ASSERT_EQ(4,cifc._csvdata.at(0)._v.size());
  ASSERT_EQ(4,cifc._columns.size());
  ASSERT_EQ("target",(*cifc._columns.begin()));
  remove("test.csv");
}

/*TEST(inputconn,txt_parse_content)
{
  std::string str = "everything runs fine, right?";
  TxtInputFileConn tifc;
  tifc.parse_content(str,1);
  ASSERT_EQ(4,tifc._vocab.size());
  ASSERT_EQ(4,tifc._rvocab.size());
  Word w = tifc._vocab["fine"];
  ASSERT_EQ(2,w._pos);
  ASSERT_EQ(1,w._total_count);
  ASSERT_EQ(1,w._total_classes);
  ASSERT_EQ(1,tifc._txt.size());
  TxtBowEntry tbe = tifc._txt.at(0);
  ASSERT_EQ(4,tbe._v.size());
  ASSERT_EQ(1,tbe._target);
  }*/
