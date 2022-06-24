/**
 * DeepDetect
 * Copyright (c) 2021 Louis Jean
 * Author: Louis Jean <louis.jean@jolibrain.com>
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

#include <gtest/gtest.h>
#include <iostream>

#include "oatppjsonapi.h"
#include "http/controller.hpp"
#include "utils/oatpp.hpp"

using namespace dd;

static std::string ok_str = "{\"status\":{\"code\":200,\"msg\":\"OK\"}}";
static std::string created_str
    = "{\"status\":{\"code\":201,\"msg\":\"Created\"}}";

static std::string example_video_path1 = "../examples/all/video/video1.mp4";
static std::string detect_repo = "../examples/torch/fasterrcnn_torch/";

// TODO Move this method at a central location (oatpp_utils?)
static std::string response_to_str(
    const std::shared_ptr<oatpp::web::protocol::http::outgoing::Response>
        &response)
{
  oatpp::data::stream::BufferOutputStream buf_stream;
  oatpp::data::stream::BufferOutputStream stream;
  response->send(&stream, &buf_stream, nullptr);
  std::string response_str = stream.toString();
  // keep body only (skip headers)
  size_t header_count = response->getHeaders().getSize();
  size_t pos = 0;
  for (size_t i = 0; i < header_count + 2; ++i)
    {
      pos = response_str.find('\n', pos + 1);
    }
  return response_str.substr(pos + 1);
}

TEST(video, open_nonexisting_video)
{
  auto json_mapper = oatpp_utils::createDDMapper();
  json_mapper->getDeserializer()->getConfig()->allowUnknownFields = false;

  OatppJsonAPI japi;
  std::shared_ptr<oatpp::data::mapping::ObjectMapper> mapper = json_mapper;
  auto controller = DedeController::createShared(&japi, mapper);

  // create resource
  std::string res_name = "video";
  std::string jstr = "{\"type\":\"video\",\"source\":\"../examples/no_video/"
                     "no_video.mp4\"}";
  // this should raise an error
  std::string joutstr = response_to_str(controller->create_resource(
      res_name.c_str(),
      json_mapper->readFromString<oatpp::Object<DTO::Resource>>(
          jstr.c_str())));
  JDoc jd;
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_EQ(400, jd["status"]["code"].GetInt());
}

#ifdef USE_TORCH

TEST(video, resource)
{
  auto json_mapper = oatpp_utils::createDDMapper();
  json_mapper->getDeserializer()->getConfig()->allowUnknownFields = false;

  OatppJsonAPI japi;
  std::shared_ptr<oatpp::data::mapping::ObjectMapper> mapper = json_mapper;
  auto controller = DedeController::createShared(&japi, mapper);

  // create resource
  std::string res_name = "video";
  std::string jstr
      = "{\"type\":\"video\",\"source\":\"" + example_video_path1 + "\"}";
  std::string joutstr = response_to_str(controller->create_resource(
      res_name.c_str(),
      json_mapper->readFromString<oatpp::Object<DTO::Resource>>(
          jstr.c_str())));
  std::cout << "joutstr=" << joutstr << std::endl;
  JDoc jd;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_EQ(201, jd["status"]["code"].GetInt());

  // check resources info
  joutstr = response_to_str(controller->get_resource(res_name.c_str()));
  std::cout << "joutstr=" << joutstr << std::endl;
  jd = JDoc();
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_EQ(200, jd["status"]["code"].GetInt());
  ASSERT_EQ(30, jd["body"]["video"]["fps"].GetInt());
  ASSERT_EQ(640, jd["body"]["video"]["width"].GetInt());
  ASSERT_EQ(30, jd["body"]["video"]["frame_count"].GetInt());

  // create service
  std::string sname = "detectserv";
  jstr = "{\"mllib\":\"torch\",\"description\":\"fasterrcnn\",\"type\":"
         "\"supervised\",\"model\":{\"repository\":\""
         + detect_repo
         + "\"},\"parameters\":{\"input\":{\"connector\":\"image\",\"height\":"
           "224,\"width\":224,\"rgb\":true,\"scale\":0.0039},\"mllib\":{"
           "\"template\":\"fasterrcnn\",\"gpu\":true,\"gpuid\":0}}}";

  joutstr = japi.jrender(japi.service_create(sname, jstr));
  ASSERT_EQ(created_str, joutstr);

  int frame_id = 0;

  // predict call on resource
  std::string jpredictstr = "{\"service\":\"detectserv\",\"parameters\":{"
                            "\"input\":{\"height\":224,"
                            "\"width\":224},\"output\":{\"bbox\":true, "
                            "\"confidence_threshold\":0.8}},\"data\":[\""
                            + res_name + "\"]}";

  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  frame_id++;
  jd = JDoc();
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200, jd["status"]["code"]);
  ASSERT_TRUE(jd["body"]["predictions"].IsArray());

  auto &preds = jd["body"]["predictions"][0]["classes"];
  std::string cl1 = preds[0]["cat"].GetString();
  ASSERT_EQ(cl1, "cat");
  ASSERT_TRUE(preds[0]["prob"].GetDouble() > 0.9);
  auto &bbox = preds[0]["bbox"];

  // cat is approximately in bottom left corner of the image.
  ASSERT_TRUE(bbox["xmin"].GetDouble() < 100 && bbox["xmax"].GetDouble() > 300
              && bbox["ymin"].GetDouble() < 100
              && bbox["ymax"].GetDouble() > 300);
  // Check confidence threshold
  ASSERT_TRUE(preds[preds.Size() - 1]["prob"].GetDouble() >= 0.8);

  int dog_frame_id = 15;
  for (; frame_id < dog_frame_id; ++frame_id)
    {
      joutstr = japi.jrender(japi.service_predict(jpredictstr));
      jd = JDoc();
      jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
      ASSERT_EQ(200, jd["status"]["code"].GetInt());
    }
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  frame_id++;
  jd = JDoc();
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_TRUE(!jd.HasParseError());
  ASSERT_EQ(200, jd["status"]["code"]);
  ASSERT_TRUE(jd["body"]["predictions"].IsArray());

  auto &preds2 = jd["body"]["predictions"][0]["classes"];
  std::string cl2 = preds2[0]["cat"].GetString();
  ASSERT_EQ(cl2, "dog");

  // exhaust resource and see what happens at the end
  int end_frame_id = 29;
  for (; frame_id < end_frame_id; ++frame_id)
    {
      joutstr = japi.jrender(japi.service_predict(jpredictstr));
      jd = JDoc();
      jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
      ASSERT_EQ(200, jd["status"]["code"].GetInt());
    }

  // last frame: status must be "ended"
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  std::cout << "joutstr=" << joutstr << std::endl;
  jd = JDoc();
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_EQ(200, jd["status"]["code"].GetInt());
  ASSERT_EQ(std::string("ended"),
            jd["body"]["resources"][0]["status"].GetString());

  // get resoure state
  joutstr = response_to_str(controller->get_resource(res_name.c_str()));
  std::cout << "joutstr=" << joutstr << std::endl;
  jd = JDoc();
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_EQ(200, jd["status"]["code"].GetInt());
  ASSERT_EQ(std::string("ended"), jd["body"]["status"].GetString());
  ASSERT_EQ(30, jd["body"]["video"]["current_frame"].GetInt());

  // prediction after last frame
  joutstr = japi.jrender(japi.service_predict(jpredictstr));
  jd = JDoc();
  std::cout << "joutstr=" << joutstr << std::endl;
  jd.Parse<rapidjson::kParseNanAndInfFlag>(joutstr.c_str());
  ASSERT_EQ(403, jd["status"]["code"].GetInt());
  ASSERT_EQ(jd["status"]["dd_msg"].GetString(),
            std::string("Resource is exhausted"));
}

#endif
