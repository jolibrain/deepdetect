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

#define _CREATE(type, repo, weights, std, mllib...) TO_STRING({	\
      "mllib": "caffe2",				\
      "description": "my classifier",			\
      "type": type,					\
      "model": {					\
	"templates": "../templates/caffe2",		\
	"repository": repo,				\
	"weights": weights				\
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

#define CREATE_DETECTRON(repo, input...) TO_STRING({	\
      "mllib": "caffe2",				\
      "description": "my classifier",			\
      "type": "supervised",				\
      "model": {					\
	"repository": repo				\
      },						\
      "parameters": {					\
	"input": {					\
	  "connector": "image",				\
	  "mean": [102.9801, 115.9465, 122.7717],	\
	  input						\
	},						\
        "mllib": {					\
	  "gpuid": [0]					\
	}						\
      }							\
    })

#define CREATE_DETECTRON_MASK(repo, mask, input...) TO_STRING({	\
      "mllib": "caffe2",				\
      "description": "my classifier",			\
      "type": "supervised",				\
      "model": {					\
	"repository": repo,				\
	"extensions": [mask]				\
      },						\
      "parameters": {					\
	"input": {					\
	  "connector": "image",				\
	  "mean": [102.9801, 115.9465, 122.7717],	\
	  input						\
	},						\
        "mllib": {					\
	  "gpuid": [0]					\
	}						\
      }							\
    })

#define CREATE(t, r, s) _CREATE(t, r, "", s, "gpuid": [0])
#define CREATE_TEMPLATE(t, r, s, tname, nclasses)			\
  _CREATE(t, r, "", s, "gpuid": [0], "template": tname, "nclasses": nclasses)
#define CREATE_FINETUNE(t, r, s, tname, nclasses, weights)		\
  _CREATE(t, r, weights, s, "gpuid": [0], "template": tname, "nclasses": nclasses, "finetune": true)

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
#define PREDICT_BBOX(service, threshold, data...) TO_STRING({		\
      "service": service,						\
      "parameters": {							\
	"output": {							\
	  "bbox": true,							\
	  "best": 1,							\
	  "confidence_threshold": threshold				\
	}								\
      },								\
      "data": [ data ]							\
    })
#define PREDICT_MASK(service, threshold, data...) TO_STRING({		\
      "service": service,						\
      "parameters": {							\
	"output": {							\
	  "mask": true,							\
	  "best": 1,							\
	  "confidence_threshold": threshold				\
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

void assert_bboxes_and_masks(const JDoc &jd,
			     std::map<std::string, std::map<std::string, int>> preds,
			     bool has_mask) {

  // Check array
  size_t nb_pred = preds.size();
  const auto &predictions = jd["body"]["predictions"];
  ASSERT_TRUE(jd["body"]["predictions"].IsArray());

  // Check elements
  for (size_t i = 0; i < nb_pred; ++i) {
    const auto &prediction = predictions[i];
    auto &cmp = preds[prediction["uri"].GetString()];

    // Check BBoxes
    size_t nb_classes = prediction["classes"].Size();
    for (size_t j = 0; j < nb_classes; ++j) {
      const auto &cls = prediction["classes"][j];
      ASSERT_TRUE(cls.HasMember("bbox"));
      auto it = cmp.find(cls["cat"].GetString());
      if (it != cmp.end()) {
	it->second--;
      }

      // Check Masks
      if (has_mask) {
	ASSERT_TRUE(cls.HasMember("mask"));
	const auto &bbox = cls["bbox"];
	const auto &mask = cls["mask"];
	size_t height = mask["height"].GetInt();
	size_t width = mask["width"].GetInt();
	size_t xmin = static_cast<size_t>(bbox["xmin"].GetDouble());
	size_t ymin = static_cast<size_t>(bbox["ymin"].GetDouble());
	size_t xmax = static_cast<size_t>(bbox["xmax"].GetDouble());
	size_t ymax = static_cast<size_t>(bbox["ymax"].GetDouble());
	ASSERT_TRUE(width == xmax - xmin + 1);
	ASSERT_TRUE(height == ymax - ymin + 1);
	ASSERT_TRUE(mask.HasMember("format"));
	ASSERT_TRUE(mask["data"].Size() == width * height);
      }
    }

    // Check count
    for (auto it : cmp) {
      ASSERT_TRUE(it.second <= 0);
    }
  }
}

void clean_repository(const std::string &repo) {
  ASSERT_TRUE(!fileops::remove_directory_files(repo, { ".json", ".txt", ".pb" }));
  std::set<std::string> dbs({repo + "/train.lmdb", repo + "/test.lmdb"});
  for (const std::string &db : dbs) {
    fileops::clear_directory(db);
    rmdir(db.c_str());
  }
}

inline void assert_accuracy(const JDoc &jd, double d) {
  ASSERT_TRUE(fabs(jd["body"]["measure"]["acc"].GetDouble()) > d);
}

inline void assert_loss(const JDoc &jd, double d) {
  ASSERT_TRUE(fabs(jd["body"]["measure"]["train_loss"].GetDouble()) < d);
}

inline void assert_first_test_at(const JDoc &jd, double d) {
  ASSERT_EQ(jd["body"]["measure_hist"]["iteration_hist"][0].GetDouble(), d);
}

// Paths

#define TRAINED "../examples/caffe2/resnet_50_imagenet"
#define IMG_CAT "../examples/caffe2/resnet_50_imagenet/imgs/cat.jpg"
#define IMG_AMB "../examples/caffe2/resnet_50_imagenet/imgs/ambulance.jpg"
#define DB_FISH "../examples/caffe2/resnet_50_imagenet/fish.lmdb"
#define BC_REPO "../examples/caffe2/boats_and_cars"
#define BC_IMGS "../examples/caffe2/boats_and_cars/imgs"
#define WEIGHTS "../examples/caffe2/resnet_50_imagenet/init_net.pb"
#define DTTRON "../examples/caffe2/detectron"
#define DTTRON_BOATS "../examples/caffe2/detectron/imgs/boats.jpg"
#define DTTRON_DOGS "../examples/caffe2/detectron/imgs/dogs.jpg"
#define DTTRON_MASK "../examples/caffe2/detectron_mask"
#define DTTRON_MASK_EXT "../examples/caffe2/detectron_mask/ext"

// Json create

static const std::string supervised = CREATE("supervised", TRAINED, 128.0);
static const std::string unsupervised = CREATE("unsupervised", TRAINED, 128.0);
static const std::string trainable = CREATE_TEMPLATE("supervised", BC_REPO, 128.0, "resnet_50", 2);
static const std::string finetunable = CREATE_FINETUNE("supervised", BC_REPO, 128.0, "resnet_50", 2,
						       WEIGHTS);
static const std::string detectron = CREATE_DETECTRON(DTTRON, "scale_min": 800, "scale_max": 1333);
static const std::string detectron_mask =
  CREATE_DETECTRON_MASK(DTTRON_MASK, DTTRON_MASK_EXT, "height": 800, "width": 1216);

// Json predict

static const std::string predict_cat = PREDICT_SUPERVISED(SERVICE, 3, IMG_CAT);
static const std::string predict_cat_and_amb = PREDICT_SUPERVISED(SERVICE, 3, IMG_CAT, IMG_AMB);
static const std::string test_accuracy = PREDICT_TEST(SERVICE, 6, DB_FISH, "acc");
static const std::string extract_conv1 = PREDICT_UNSUPERVISED(SERVICE, "gpu_0/conv1", IMG_CAT);
static const std::string predict_bbox_boats_and_dogs =
  PREDICT_BBOX(SERVICE, 0.8, DTTRON_BOATS, DTTRON_DOGS);
static const std::string predict_mask_boats_and_dogs =
  PREDICT_MASK(SERVICE, 0.8, DTTRON_BOATS, DTTRON_DOGS);

// Json train & test

// 1 epoch = 1200 imgs / 32 batch size = 37.5 iters
#define TRAIN_BC(service, iter, resume, data...) TO_STRING({	\
      "service": service,					\
      "async": false,						\
      "parameters": {						\
	"input": {						\
	  "test_split": 0.1,					\
	  "shuffle": true,					\
	  "img_aug": {						\
	    "mirror": true,					\
	    "color_jitter": true,				\
	    "img_brightness": 0.5				\
	  }							\
	},							\
	"output": {						\
	  "measure_hist": true,					\
	  "measure": ["acc"]					\
	},							\
	"mllib": {						\
	  "resume": resume,					\
	  "net": {						\
	    "batch_size": 32,					\
	    "test_batch_size": 32				\
	   },							\
	  "solver": {						\
	    "iterations": iter,					\
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
static const std::string train_boats_and_cars = TRAIN_BC(SERVICE, 1000, false, BC_IMGS);
static const std::string train_boats_and_cars_resume = TRAIN_BC(SERVICE, 1300, true, BC_IMGS);

// Finetune

#define FINETUNE_BC(service, iter, data...) TO_STRING({		\
      "service": service,					\
      "async": false,						\
      "parameters": {						\
	"input": {						\
	  "test_split": 0.1,					\
	  "shuffle": true					\
	},							\
	"output": {						\
	  "measure_hist": true,					\
	  "measure": ["acc"]					\
	},							\
	"mllib": {						\
	  "net": {						\
	    "batch_size": 32,					\
	    "test_batch_size": 32				\
	   },							\
	  "solver": {						\
	    "iterations": iter,					\
	    "test_interval": 40,				\
	    "lr_policy": "step",				\
	    "base_lr": 0.01,					\
	    "stepsize": 1,					\
	    "gamma": 0.99,					\
	    "solver_type": "sgd"				\
	  }							\
	}							\
      },							\
      "data": [ data ]						\
})
static const std::string finetune_boats_and_cars = FINETUNE_BC(SERVICE, 200, BC_IMGS);

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
  assert_accuracy(jd, 0.7);
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
  clean_repository(BC_REPO);

  JsonAPI japi;
  JDoc jd;
  create(japi, trainable);

  // Test training
  train(japi, jd, train_boats_and_cars);
  assert_accuracy(jd, 0.7);
  assert_loss(jd, 0.2);
  assert_first_test_at(jd, 200); // 1000 iterations (0, 999), tests at 200, 400, 600, 800

  // Test resume
  train(japi, jd, train_boats_and_cars_resume);
  assert_accuracy(jd, 0.7);
  assert_loss(jd, 0.2);
  assert_first_test_at(jd, 1200); // 300 iterations (1000, 1299), test at 1200

  // Remove new data
  clean_repository(BC_REPO);
}

TEST(caffe2api, service_finetune) {

  // Remove old data
  clean_repository(BC_REPO);

  JsonAPI japi;
  JDoc jd;
  create(japi, finetunable);
  train(japi, jd, finetune_boats_and_cars);
  assert_accuracy(jd, 0.8);

  // Remove new data
  clean_repository(BC_REPO);
}

TEST(caffe2api, detectron_bbox_predict) {
  JsonAPI japi;
  JDoc jd;
  create(japi, detectron);

  predict(japi, jd, predict_bbox_boats_and_dogs);
  assert_bboxes_and_masks(jd, {
      { DTTRON_BOATS, {{"boat", 2}, {"person", 35}} },
      { DTTRON_DOGS, {{"dog", 2}, {"person", 2}} }
  }, false);
}

TEST(caffe2api, detectron_mask_predict) {
  JsonAPI japi;
  JDoc jd;
  create(japi, detectron_mask);

  predict(japi, jd, predict_mask_boats_and_dogs);
  assert_bboxes_and_masks(jd, {
      { DTTRON_BOATS, {{"boat", 2}, {"person", 25}} },
      { DTTRON_DOGS, {{"dog", 2}, {"person", 2}} }
  }, true);
}
