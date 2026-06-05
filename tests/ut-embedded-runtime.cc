#include "deepdetect/runtime.h"

#include <gtest/gtest.h>
#include <rapidjson/document.h>

namespace
{
  rapidjson::Document parse(const std::string &value)
  {
    rapidjson::Document document;
    document.Parse(value.c_str());
    EXPECT_FALSE(document.HasParseError());
    EXPECT_TRUE(document.IsObject());
    return document;
  }
}

TEST(embedded_runtime, build_and_server_info_are_valid_json)
{
  deepdetect::Runtime runtime;
  auto build = parse(runtime.build_info());
  EXPECT_TRUE(build.HasMember("version"));
  EXPECT_TRUE(build.HasMember("cuda"));

  auto info = parse(runtime.info());
  ASSERT_TRUE(info.HasMember("status"));
  EXPECT_EQ(info["status"]["code"].GetInt(), 200);
  EXPECT_TRUE(info.HasMember("head"));
}

TEST(embedded_runtime, malformed_json_is_contained)
{
  deepdetect::Runtime runtime;
  auto response = parse(runtime.predict("{"));
  ASSERT_TRUE(response.HasMember("status"));
  EXPECT_EQ(response["status"]["code"].GetInt(), 400);
}

TEST(embedded_runtime, deepdetect_errors_keep_the_complete_envelope)
{
  deepdetect::Runtime runtime;
  auto response = parse(runtime.service_info("missing"));
  ASSERT_TRUE(response.HasMember("status"));
  EXPECT_EQ(response["status"]["code"].GetInt(), 404);
  EXPECT_EQ(response["status"]["dd_code"].GetInt(), 1002);
}
