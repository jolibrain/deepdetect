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

#define CREATE(type, repo, std, nclasses) TO_STRING({	\
      "mllib": "caffe2",				\
      "description": "my classifier",			\
      "type": type,					\
      "model": {					\
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
	  "gpu": true,					\
	  "nclasses": nclasses				\
	}						\
      }							\
    })

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

// The number of GPUs being unknown, batch sizes are set to 6 (divisible by 1, 2 and 3)
#define TRAIN(service, iter, test, measures...) TO_STRING({		\
      "service": service,						\
      "async": false,							\
      "parameters": {							\
	"mllib": {							\
	  "net": {							\
	    "batch_size": 6,						\
	    "test_batch_size": 6  					\
	  },								\
	  "solver": {							\
	    "iterations": iter,						\
	    "test_interval": test					\
	  }								\
	},								\
	"output": {							\
	  "measure": [ measures ]					\
        }								\
      }									\
    })

static const std::string ok_str = RESPONSE(200, "ok");
static const std::string created_str = RESPONSE(201, "Created");
static const std::string bad_param_str = RESPONSE(400, "BadRequest");
static const std::string not_found_str = RESPONSE(404, "NotFound");

//XXX Current tests just assert that nothing breaks
//Loss and accuracy values/evolutions are not checked
//Real datasets, images and models must be prepared for that

//XXX Add a test that dumps & resumes the state of a training

static const std::string supervised =
  CREATE("supervised", "../examples/caffe2/resnet_50", 255.0, 1000);
static const std::string unsupervised =
  CREATE("unsupervised", "../examples/caffe2/resnet_50", 255.0, 1000);
static const std::string trainable =
  CREATE("supervised", "../examples/caffe2/resnet_50_trainable", 255.0, 1000);

static const std::string predict_test1_str =
  PREDICT_SUPERVISED("imgserv", 3,
		     "../examples/caffe/voc/voc/test_img.jpg");
static const std::string predict_test2_str =
  PREDICT_SUPERVISED("imgserv", 3,
		     "../examples/caffe/voc/voc/test_img.jpg",
		     "../examples/caffe/camvid/CamVid_square/test/0001TP_010110.png");
static const std::string predict_test3_str =
  PREDICT_UNSUPERVISED("imgserv", "gpu_0/conv1",
		       "../examples/caffe/voc/voc/test_img.jpg");
static const std::string predict_test4_str =
  PREDICT_TEST("imgserv", 5, "../examples/caffe2/resnet_50_trainable/test.lmdb", "acc");

static const std::string train_test1_str = TRAIN("imgserv", 2, 1, "acc");

TEST(caffe2api, service_predict) {

  // create service
  JsonAPI japi;
  std::string joutstr = japi.jrender(japi.service_create("imgserv", supervised));
  ASSERT_EQ(created_str, joutstr);

  // predict
  joutstr = japi.jrender(japi.service_predict(predict_test1_str));
  JDoc jd;
  jd.Parse(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200, jd["status"]["code"]);
  ASSERT_TRUE(jd["body"]["predictions"].IsArray());
  std::string cl1 = jd["body"]["predictions"][0]["classes"][0]["cat"].GetString();
  ASSERT_TRUE(cl1 == "megalith, megalithic structure");
  ASSERT_TRUE(jd["body"]["predictions"][0]["classes"][0]["prob"].GetDouble() > 0.35);

  // predict batch
  joutstr = japi.jrender(japi.service_predict(predict_test2_str));
  jd.Parse(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200, jd["status"]["code"]);
  ASSERT_TRUE(jd["body"]["predictions"].IsArray());
  ASSERT_EQ(2, jd["body"]["predictions"].Size());
  cl1 = jd["body"]["predictions"][0]["classes"][0]["cat"].GetString();
  std::string cl2 = jd["body"]["predictions"][1]["classes"][0]["cat"].GetString();
  ASSERT_TRUE((cl1 == "megalith, megalithic structure" && cl2 == "car mirror")
	      || (cl1 == "car mirror" && cl2 == "megalith, megalithic structure"));
  ASSERT_TRUE(jd["body"]["predictions"][0]["classes"][0]["prob"].GetDouble() > 0.35);
  ASSERT_TRUE(jd["body"]["predictions"][1]["classes"][0]["prob"].GetDouble() > 0.35);
}

TEST(caffe2api, service_predict_unsup) {

  // create service
  JsonAPI japi;
  std::string joutstr = japi.jrender(japi.service_create("imgserv", unsupervised));
  ASSERT_EQ(created_str, joutstr);

  // predict
  joutstr = japi.jrender(japi.service_predict(predict_test3_str));
  JDoc jd;
  jd.Parse(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200, jd["status"]["code"]);
  ASSERT_EQ(802816, jd["body"]["predictions"][0]["vals"].Size());
}

TEST(caffe2api, service_predict_test) {

  // create service
  JsonAPI japi;
  std::string joutstr = japi.jrender(japi.service_create("imgserv", trainable));
  ASSERT_EQ(created_str, joutstr);

  // predict
  joutstr = japi.jrender(japi.service_predict(predict_test4_str));
  JDoc jd;
  jd.Parse(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200, jd["status"]["code"].GetInt());
  ASSERT_TRUE(fabs(jd["body"]["measure"]["acc"].GetDouble()) >= 0);
}

TEST(caffe2api, service_train) {

  // create service
  JsonAPI japi;
  std::string joutstr = japi.jrender(japi.service_create("imgserv", trainable));
  ASSERT_EQ(created_str, joutstr);

  // train
  joutstr = japi.jrender(japi.service_train(train_test1_str));
  JDoc jd;
  jd.Parse(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(201, jd["status"]["code"].GetInt());
  ASSERT_EQ("Created", jd["status"]["msg"]);
  ASSERT_TRUE(jd["head"]["time"].GetDouble() >= 0);
  ASSERT_TRUE(fabs(jd["body"]["measure"]["train_loss"].GetDouble()) > 0);
  ASSERT_TRUE(fabs(jd["body"]["measure"]["acc"].GetDouble()) >= 0);
}
