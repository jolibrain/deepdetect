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
#include "outputconnectorstrategy.h"
#include <gtest/gtest.h>
#include <iostream>

using namespace dd;

TEST(outputconn,acc)
{
  std::vector<int> targets = {0, 0, 1, 1};
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
  double acc = so.acc(res_ad);
  ASSERT_EQ(0.75,acc);
}

TEST(outputconn,auc)
{
  std::vector<int> targets = {0, 0, 1, 1};
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

TEST(inputconn,img)
{
  std::string mnist_repo = "../examples/caffe/mnist/";
  APIData ad;
  std::vector<std::string> uris = {mnist_repo + "/sample_digit.png","http://www.deepdetect.com/dd/examples/caffe/mnist/sample_digit.png"};
  ad.add("data",uris);
  ImgInputFileConn iifc;
  try
    {
      iifc.transform(ad);
    }
  catch (InputConnectorBadParamException &e)
    {
      ASSERT_FALSE(true); // trigger
    }
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
  ASSERT_EQ("val5",cifc2._label);
  ASSERT_EQ("id",cifc2._id);
}
