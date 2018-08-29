/**
 * DeepDetect
 * Copyright (c) 2018 Jolibrain
 * Author: Julien Chicha
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
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <iostream>

using namespace dd;

#define TO_STRING(...) #__VA_ARGS__

#define RESPONSE(code, msg) TO_STRING({"status":{"code":code,"msg":msg}})

#define _CREATE(type, repo, std, mllib...) TO_STRING({	\
      "mllib": "caffe2",				\
      "description": "my classifier",			\
      "type": type,					\
      "model": {					\
	"templates": "../templates/caffe2",		\
	"repository": repo				\
      },						\
      "parameters": {					\
	"input": {					\
	  "connector": "image",				\
	  "height": 224,				\
	  "width": 224,					\
	  "std": std					\
	},						\
	"mllib": {					\
	  mllib						\
	}						\
      }							\
    })

#define CREATE(t, r, s) _CREATE(t, r, s, "gpuid": [0])
#define CREATE_TEMPLATE(t, r, s, tname, nclasses)			\
  _CREATE(t, r, s, "gpuid": [0], "template": tname, "nclasses": nclasses)

#define PREDICT(service, extraction, data...) TO_STRING({	\
      "service": service,					\
      "parameters": {						\
	extraction						\
      },							\
      "data": [ data ]						\
    })
#define PREDICT_SUPERVISED(service, best, data...)	\
  PREDICT(service,					\
	  "output": {					\
	    "best": best				\
	  },						\
	  data)
#define PREDICT_UNSUPERVISED(service, layer, data...)	\
  PREDICT(service,					\
	  "mllib": {					\
	    "extract_layer": layer			\
	  },						\
	  data)
#define PREDICT_TEST(service, batch, data, measures...) TO_STRING({	\
      "service": service,						\
      "parameters": {							\
	"mllib": {							\
	  "net": {							\
	    "batch_size": batch						\
	   }								\
	},								\
	"output": {							\
	  "measure": [ measures ]					\
	}								\
      },								\
      "data": [ data ]							\
    })

#define SERVICE "imgserv"

static const std::string ok_str = RESPONSE(200, "ok");
static const std::string created_str = RESPONSE(201, "Created");
static const std::string bad_param_str = RESPONSE(400, "BadRequest");
static const std::string not_found_str = RESPONSE(404, "NotFound");

// Json management

inline void create(JsonAPI &japi, const std::string &json) {
  ASSERT_EQ(created_str, japi.jrender(japi.service_create(SERVICE, json)));
}

inline void predict(JsonAPI &japi, JDoc &jd, const std::string &json) {
  jd.Parse(japi.jrender(japi.service_predict(json)).c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200, jd["status"]["code"]);
}

inline void train(JsonAPI &japi, JDoc &jd, const std::string &json) {
  jd.Parse(japi.jrender(japi.service_train(json)).c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(201, jd["status"]["code"].GetInt());
  ASSERT_EQ("Created", jd["status"]["msg"]);
  ASSERT_TRUE(jd["head"]["time"].GetDouble() >= 0);
}

void assert_predictions(const JDoc &jd, std::vector<std::pair<std::string, double>> preds) {

  // Check array
  size_t nb_pred = preds.size();
  const auto &predictions = jd["body"]["predictions"];
  ASSERT_TRUE(jd["body"]["predictions"].IsArray());

  // Check elements
  for (size_t i = 0; i < nb_pred; ++i) {
    const auto &prediction = predictions[i]["classes"][0];
    std::string cat = prediction["cat"].GetString();
    double prob = prediction["prob"].GetDouble();

    // Find a matching prediction
    auto pred = preds.begin();
    while (true) {
      ASSERT_TRUE(pred != preds.end());
      if (cat == pred->first && prob > pred->second) {
	break;
      }
      ++pred;
    }
    preds.erase(pred);
  }
}

inline void assert_accuracy(const JDoc &jd, double d) {
  ASSERT_TRUE(fabs(jd["body"]["measure"]["acc"].GetDouble()) > d);
}

inline void assert_loss(const JDoc &jd, double d) {
  ASSERT_TRUE(fabs(jd["body"]["measure"]["train_loss"].GetDouble()) < d);
}

// Paths

#define TRAINED "../examples/caffe2/resnet_50_trained"
#define IMG_CAT "../examples/caffe2/resnet_50_trained/imgs/cat.jpg"
#define IMG_AMB "../examples/caffe2/resnet_50_trained/imgs/ambulance.jpg"
#define DB_FISH "../examples/caffe2/resnet_50_trained/fish.lmdb"
#define BC_REPO "../examples/caffe2/boats_and_cars"
#define BC_IMGS "../examples/caffe2/boats_and_cars/imgs"

// Json create

static const std::string supervised = CREATE("supervised", TRAINED, 128.0);
static const std::string unsupervised = CREATE("unsupervised", TRAINED, 128.0);
static const std::string trainable = CREATE_TEMPLATE("supervised", BC_REPO, 128.0, "resnet_50", 2);

// Json predict

static const std::string predict_cat = PREDICT_SUPERVISED(SERVICE, 3, IMG_CAT);
static const std::string predict_cat_and_amb = PREDICT_SUPERVISED(SERVICE, 3, IMG_CAT, IMG_AMB);
static const std::string test_accuracy = PREDICT_TEST(SERVICE, 6, DB_FISH, "acc");
static const std::string extract_conv1 = PREDICT_UNSUPERVISED(SERVICE, "gpu_0/conv1", IMG_CAT);

// Json train & test

// 1 epoch = 1200 imgs / 32 batch size = 37.5 iters
#define TRAIN_BC(service, data...) TO_STRING({			\
      "service": service,					\
      "async": false,						\
      "parameters": {						\
	"input": {						\
	  "test_split": 0.1,					\
	  "shuffle": true					\
	},							\
	"output": {						\
	  "measure": ["acc"]					\
	},							\
	"mllib": {						\
	  "net": {						\
	    "batch_size": 32,					\
	    "test_batch_size": 32				\
	   },							\
	  "solver": {						\
	    "iterations": 1000,					\
	    "test_interval": 200,				\
	    "lr_policy": "step",				\
	    "base_lr": 0.01,					\
	    "stepsize": 375,					\
	    "gamma": 0.1,					\
	    "solver_type": "sgd"				\
	  }							\
	}							\
      },							\
      "data": [ data ]						\
})
static const std::string train_boats_and_cars = TRAIN_BC(SERVICE, BC_IMGS);

// Tests

TEST(caffe2api, service_predict_supervised) {

  JsonAPI japi;
  JDoc jd;
  create(japi, supervised);

  predict(japi, jd, predict_cat);
  assert_predictions(jd, { {"tabby, tabby cat", 0.8} });

  predict(japi, jd, predict_cat_and_amb);
  assert_predictions(jd, { {"tabby, tabby cat", 0.8}, {"ambulance", 0.8} });
}

TEST(caffe2api, service_predict_test) {

  JsonAPI japi;
  JDoc jd;
  create(japi, supervised);

  predict(japi, jd, test_accuracy);
  assert_accuracy(jd, 0.9);
}

TEST(caffe2api, service_predict_extract_layer) {

  JsonAPI japi;
  JDoc jd;
  create(japi, unsupervised);

  predict(japi, jd, extract_conv1);
  ASSERT_EQ(802816, jd["body"]["predictions"][0]["vals"].Size());
}

TEST(caffe2api, service_train) {

  // Remove old data
  ASSERT_TRUE(!fileops::remove_directory_files(BC_REPO, { ".json", ".txt", ".pb" }));
  std::set<std::string> dbs({BC_REPO "/train.lmdb", BC_REPO "/test.lmdb"});
  for (const std::string &db : dbs) {
    fileops::clear_directory(db);
    rmdir(db.c_str());
  }

  JsonAPI japi;
  JDoc jd;
  create(japi, trainable);

  train(japi, jd, train_boats_and_cars);
  assert_accuracy(jd, 0.7);
  assert_loss(jd, 0.01);
}

//XXX Add a test that dumps & resumes the state of a training
