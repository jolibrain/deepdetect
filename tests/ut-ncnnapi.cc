/**
 * DeepDetect
 * Copyright (c) 2018 Jolibrain
 * Author: Corentin Barreau <corentin.barreau@epitech.eu>
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

#include "deepdetect/runtime.h"
#include <gtest/gtest.h>
#include <rapidjson/document.h>
#include <opencv2/imgcodecs.hpp>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <future>

namespace fs = std::filesystem;
namespace
{
  class NCNNService : public ::testing::Test
  {
  protected:
    void SetUp() override
    {
      char path[] = "/tmp/deepdetect-ncnn-XXXXXX";
      ASSERT_NE(mkdtemp(path), nullptr);
      repo = path;
      for (int i = 0; i < 2; ++i)
        ASSERT_TRUE(cv::imwrite(
            image(i),
            cv::Mat(4, 8, CV_8UC3,
                    i == 0 ? cv::Scalar(3, 2, 1) : cv::Scalar(1, 2, 3))));
      std::ofstream(repo / "net.bin", std::ios::binary);
      std::ofstream(repo / "net.param")
          << "7767517\n3 3\nInput input 0 1 image\n"
             "Pooling avg 1 1 image pooled 0=1 4=1\n"
             "Flatten flatten 1 1 pooled scores\n";
    }
    void TearDown() override
    {
      fs::remove_all(repo);
    }
    std::string image(int i) const
    {
      return (repo / (std::to_string(i) + ".png")).string();
    }
    rapidjson::Document create(std::string options
                               = R"("ncnn_output":"logits")")
    {
      auto json = runtime.create_service(
          "net",
          R"({"mllib":"ncnn","model":{"repository":")" + repo.string()
              + R"("},"parameters":{"input":{"connector":"image","width":4,"height":4,"ncnn_letterbox":true},"mllib":{"nclasses":3,"threads":1,)"
              + options + "}}}");
      rapidjson::Document doc;
      doc.Parse(json.c_str());
      return doc;
    }
    rapidjson::Document predict(std::string output)
    {
      auto json = runtime.predict(
          R"({"service":"net","parameters":{"output":{)" + output
          + R"(}},"data":[")" + image(1) + R"(",")" + image(0) + R"("]})");
      rapidjson::Document doc;
      doc.Parse(json.c_str());
      return doc;
    }
    void fixed_rows(size_t width, const std::vector<float> &bias)
    {
      const size_t count = bias.size();
      std::ofstream(repo / "net.param")
          << "7767517\n4 4\nInput input 0 1 image\nPooling avg 1 1 image pool "
             "0=1 4=1\n"
          << "InnerProduct head 1 1 pool flat 0=" << count
          << " 1=1 2=" << 3 * count
          << "\nReshape rows 1 1 flat boxes 0=" << width
          << " 1=" << count / width << "\n";
      // NCNN's FP32 tag, zero weights, then the constant row values as biases.
      std::vector<float> data(1 + 3 * count, 0.f);
      data.insert(data.end(), bias.begin(), bias.end());
      std::ofstream(repo / "net.bin", std::ios::binary)
          .write(reinterpret_cast<const char *>(data.data()),
                 data.size() * sizeof(float));
    }
    fs::path repo;
    deepdetect::Runtime runtime;
  };

  TEST_F(NCNNService, OrderedConcurrentImagePrediction)
  {
    std::ofstream(repo / "irrelevant.pnnx.param") << "conversion intermediate";
    auto created = create();
    ASSERT_EQ(created["status"]["code"].GetInt(), 201);
    auto check = [&]()
      {
        auto response = predict(R"("best":3)");
        ASSERT_EQ(response["status"]["code"].GetInt(), 200);
        auto &p = response["body"]["predictions"];
        ASSERT_EQ(p.Size(), 2U);
        for (unsigned i = 0; i < 2; ++i)
          {
            EXPECT_EQ(p[i]["uri"].GetString(), image(1 - i));
            ASSERT_EQ(p[i]["classes"].Size(), 3U);
            EXPECT_STREQ(p[i]["classes"][0]["cat"].GetString(), i ? "0" : "2");
            // Half the image is letterbox padding shared by all three
            // channels.
            EXPECT_NEAR(p[i]["classes"][0]["prob"].GetDouble(),
                        1. / (1. + std::exp(-.5) + std::exp(-1.)), 1e-5);
          }
      };
    auto other = std::async(std::launch::async, check);
    check();
    other.get();
  }

  TEST_F(NCNNService, RejectsMismatchedWeights)
  {
    fs::rename(repo / "net.bin", repo / "other.bin");
    EXPECT_EQ(create()["status"]["code"].GetInt(), 400);
  }

  TEST_F(NCNNService, RejectsUnknownBlob)
  {
    EXPECT_EQ(create(R"("outputBlob":"missing")")["status"]["code"].GetInt(),
              400);
  }

  TEST_F(NCNNService, RejectsPatchedFlatten)
  {
    std::ofstream(repo / "net.param")
        << "7767517\n2 2\nInput input 0 1 image\nFlatten flat 1 1 image "
           "output 0=2\n";
    auto response = create();
    EXPECT_EQ(response["status"]["code"].GetInt(), 400);
    EXPECT_NE(std::string(response["status"]["dd_msg"].GetString())
                  .find("re-export"),
              std::string::npos);
  }

  TEST_F(NCNNService, ChecksOutputContract)
  {
    ASSERT_EQ(create(R"("ncnn_output":"yolox")")["status"]["code"].GetInt(),
              201);
    EXPECT_EQ(predict(R"("bbox":true)")["status"]["code"].GetInt(), 400);
  }

  TEST_F(NCNNService, StandardCTCWithLabels)
  {
    fixed_rows(3, { 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1 });
    std::ofstream(repo / "corresp.txt") << "1 65\n2 66\n";
    ASSERT_EQ(create()["status"]["code"].GetInt(), 201);
    auto response = predict(R"("ctc":true,"blank_label":0)");
    ASSERT_EQ(response["status"]["code"].GetInt(), 200);
    EXPECT_STREQ(
        response["body"]["predictions"][0]["classes"][0]["cat"].GetString(),
        "AAB");
  }

  TEST_F(NCNNService, StandardSSDRows)
  {
    fixed_rows(6, { 1, .9f, .125f, .25f, .75f, .5f, 2, .2f, 0, 0, 1, 1 });
    ASSERT_EQ(create(R"("ncnn_output":"ssd")")["status"]["code"].GetInt(),
              201);
    auto response = predict(R"("bbox":true,"confidence_threshold":0.5)");
    ASSERT_EQ(response["status"]["code"].GetInt(), 200);
    auto &rows = response["body"]["predictions"][0]["classes"];
    ASSERT_EQ(rows.Size(), 1U);
    EXPECT_STREQ(rows[0]["cat"].GetString(), "1");
    EXPECT_FLOAT_EQ(rows[0]["prob"].GetDouble(), .9f);
    EXPECT_DOUBLE_EQ(rows[0]["bbox"]["xmin"].GetDouble(), .875);
    EXPECT_DOUBLE_EQ(rows[0]["bbox"]["ymin"].GetDouble(), 1.5);
  }

  TEST_F(NCNNService, UnpadsAndSuppressesWithinEachClass)
  {
    // Three identical decoded boxes. The second is a duplicate of class one.
    fixed_rows(7, { 2, 1,   2,   1, 1, .9f, .1f, 2, 1,   2,  1,
                    1, .8f, .1f, 2, 1, 2,   1,   1, .1f, .7f });
    ASSERT_EQ(create(R"("ncnn_output":"yolox")")["status"]["code"].GetInt(),
              201);
    auto response = predict(R"("bbox":true,"confidence_threshold":0.5)");
    ASSERT_EQ(response["status"]["code"].GetInt(), 200);
    auto &rows = response["body"]["predictions"][0]["classes"];
    ASSERT_EQ(rows.Size(), 2U);
    EXPECT_STREQ(rows[0]["cat"].GetString(), "1");
    EXPECT_STREQ(rows[1]["cat"].GetString(), "2");
    EXPECT_EQ(rows[0]["bbox"]["xmin"].GetDouble(), 2.);
    EXPECT_EQ(rows[0]["bbox"]["ymax"].GetDouble(), 3.);
    auto empty = predict(R"("bbox":true,"confidence_threshold":0.95)");
    ASSERT_EQ(empty["status"]["code"].GetInt(), 200);
    auto &p = empty["body"]["predictions"][0];
    EXPECT_TRUE(!p.HasMember("classes") || p["classes"].Empty());
  }
}
