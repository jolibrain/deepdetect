/**
 * DeepDetect
 * Copyright (c) 2026 Jolibrain
 *
 * This file is part of deepdetect.
 */

#include "utils/detection_map_v2.hpp"
#include "outputconnectorstrategy.h"
#include "supervisedoutputconnector.h"

#include <gtest/gtest.h>
#include <rapidjson/document.h>

#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace dd;

namespace
{
  std::string source_dir()
  {
#ifdef DEEPDETECT_SOURCE_DIR
    return DEEPDETECT_SOURCE_DIR;
#else
    return ".";
#endif
  }

  rapidjson::Document golden_cases()
  {
    std::ifstream input(source_dir()
                        + "/tests/fixtures/detection_map_v2_cases.json");
    EXPECT_TRUE(input.good());
    std::ostringstream contents;
    contents << input.rdbuf();
    rapidjson::Document document;
    document.Parse(contents.str().c_str());
    EXPECT_FALSE(document.HasParseError());
    return document;
  }

  DetectionMapV2::Box json_box(const rapidjson::Value &value)
  {
    return { value[0].GetDouble(), value[1].GetDouble(), value[2].GetDouble(),
             value[3].GetDouble() };
  }
}

TEST(detection_map_v2, matches_python_golden_cases)
{
  rapidjson::Document document = golden_cases();
  ASSERT_TRUE(document.HasMember("cases"));
  for (const rapidjson::Value &test_case : document["cases"].GetArray())
    {
      SCOPED_TRACE(test_case["name"].GetString());
      DetectionMapV2 evaluator;
      for (const rapidjson::Value &target : test_case["targets"].GetArray())
        evaluator.add_target(target["image_id"].GetInt64(),
                             target["label"].GetInt(),
                             json_box(target["box"]));
      for (const rapidjson::Value &prediction :
           test_case["predictions"].GetArray())
        evaluator.add_prediction(
            prediction["image_id"].GetInt64(), prediction["label"].GetInt(),
            json_box(prediction["box"]), prediction["score"].GetDouble());

      std::vector<std::string> measures;
      for (const rapidjson::Value &measure : test_case["measures"].GetArray())
        measures.push_back(measure.GetString());
      const auto metrics = evaluator.compute(
          DetectionMapV2::thresholds_from_measures(measures));
      std::map<std::string, double> actual;
      for (const DetectionMapV2Metric &metric : metrics)
        actual[metric.name] = metric.value;

      const rapidjson::Value &expected = test_case["expected"];
      ASSERT_EQ(expected.MemberCount(), actual.size());
      for (auto member = expected.MemberBegin();
           member != expected.MemberEnd(); ++member)
        ASSERT_NEAR(member->value.GetDouble(),
                    actual[member->name.GetString()], 1e-12);
    }
}

TEST(detection_map_v2, filters_invalid_records_like_python)
{
  DetectionMapV2 evaluator;
  ASSERT_FALSE(evaluator.add_target(0, 0, { 0.0, 0.0, 10.0, 10.0 }));
  ASSERT_FALSE(evaluator.add_target(0, 1, { 0.0, 0.0, 0.0, 10.0 }));
  ASSERT_FALSE(
      evaluator.add_prediction(0, 1, { 0.0, 0.0, 10.0, 10.0 },
                               std::numeric_limits<double>::infinity()));
  ASSERT_FALSE(evaluator.add_prediction(
      0, 1, { 0.0, 0.0, std::numeric_limits<double>::quiet_NaN(), 10.0 },
      0.9));

  ASSERT_TRUE(evaluator.add_target(0, 1, { 0.0, 0.0, 10.0, 10.0 }));
  ASSERT_TRUE(evaluator.add_prediction(0, 1, { 0.0, 0.0, 10.0, 10.0 }, 0.9));
  const auto metrics = evaluator.compute(
      DetectionMapV2::thresholds_from_measures({ "map-50" }));
  ASSERT_EQ(4U, metrics.size());
  EXPECT_DOUBLE_EQ(1.0, metrics[0].value);
  EXPECT_DOUBLE_EQ(1.0, metrics[1].value);
  EXPECT_EQ("map_1", metrics[2].name);
  EXPECT_DOUBLE_EQ(1.0, metrics[2].value);
  EXPECT_EQ("map-50_1", metrics[3].name);
  EXPECT_DOUBLE_EQ(1.0, metrics[3].value);
}

TEST(detection_map_v2, accepts_only_python_metric_threshold_names)
{
  const auto thresholds = DetectionMapV2::thresholds_from_measures(
      { "map", "map-50", "map-50", "map-0", "map-101", "xmap-90",
        "map-90-extra" });
  ASSERT_EQ(1U, thresholds.size());
  EXPECT_EQ("map-50", thresholds[0].name);
  EXPECT_DOUBLE_EQ(0.5, thresholds[0].iou);

  const auto defaults
      = DetectionMapV2::thresholds_from_measures({ "map", "train_loss" });
  ASSERT_EQ(3U, defaults.size());
  EXPECT_EQ("map-05", defaults[0].name);
  EXPECT_EQ("map-50", defaults[1].name);
  EXPECT_EQ("map-90", defaults[2].name);
}

TEST(detection_map_v2, supervised_output_uses_only_precomputed_v2_fields)
{
  APIData precomputed;
  precomputed.add("map", 0.25);
  precomputed.add("map-50", 0.5);
  precomputed.add("map_1", 0.25);
  precomputed.add("map-50_1", 0.5);
  APIData result;
  result.add("bbox", true);
  result.add("detection_map_v2", precomputed);
  APIData requested;
  requested.add("measure", std::vector<std::string>{ "map-50" });
  APIData output;

  SupervisedOutput::measure(result, requested, output);

  const APIData measures = output.getobj("measure");
  EXPECT_DOUBLE_EQ(0.25, measures.get("map").get<double>());
  EXPECT_DOUBLE_EQ(0.5, measures.get("map-50").get<double>());
  EXPECT_DOUBLE_EQ(0.25, measures.get("map_1").get<double>());
  EXPECT_DOUBLE_EQ(0.5, measures.get("map-50_1").get<double>());
  EXPECT_FALSE(measures.has("fp"));
}

TEST(detection_map_v2, multi_test_class_metrics_average_where_present)
{
  APIData test0;
  test0.add("map-50", 0.6);
  test0.add("map-50_1", 0.8);
  test0.add("map-50_2", 0.4);
  APIData test1;
  test1.add("map-50", 0.6);
  test1.add("map-50_2", 0.6);
  APIData output;
  output.add("measures", std::vector<APIData>{ test0, test1 });

  SupervisedOutput::aggregate_multiple_testsets(output, true);

  const APIData measures = output.getobj("measure");
  EXPECT_DOUBLE_EQ(0.6, measures.get("map-50").get<double>());
  EXPECT_DOUBLE_EQ(0.8, measures.get("map-50_1").get<double>());
  EXPECT_DOUBLE_EQ(0.5, measures.get("map-50_2").get<double>());
}
