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
