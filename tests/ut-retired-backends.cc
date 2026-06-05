#include "jsonapi.h"

#include <gtest/gtest.h>

#include <array>
#include <string>

using namespace dd;

TEST(jsonapi, retired_backends_are_rejected)
{
  JsonAPI japi;
  const std::array<std::string, 4> retired
      = { "caffe", "caffe2", "tf", "tensorflow" };

  for (const std::string &mllib : retired)
    {
      const std::string service_name = "retired_" + mllib;
      const std::string request
          = "{\"mllib\":\"" + mllib
            + "\",\"type\":\"supervised\",\"model\":{\"repository\":\"/tmp/"
            + service_name
            + "\"},\"parameters\":{\"input\":{\"connector\":\"image\"}}}";

      JDoc response = japi.service_create(service_name, request);
      ASSERT_EQ(400, response["status"]["code"].GetInt());
      ASSERT_EQ(1006, response["status"]["dd_code"].GetInt());
      ASSERT_NE(std::string::npos,
                std::string(response["status"]["dd_msg"].GetString())
                    .find("Torch or ONNX/TensorRT"));
    }

  JDoc info = japi.info("");
  ASSERT_EQ(0U, info["head"]["services"].Size());
}
