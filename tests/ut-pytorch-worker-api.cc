/**
 * DeepDetect
 * Copyright (c) 2026 Jolibrain
 *
 * This file is part of deepdetect.
 */

#include "jsonapi.h"

#include <gtest/gtest.h>
#include <rapidjson/document.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <thread>

using namespace dd;

namespace
{
  constexpr const char *created_str = "{\"status\":{\"code\":201,\"msg\":"
                                      "\"Created\"}}";
  constexpr const char *ok_str = "{\"status\":{\"code\":200,\"msg\":\"OK\"}}";

  std::string source_dir()
  {
#ifdef DEEPDETECT_SOURCE_DIR
    return DEEPDETECT_SOURCE_DIR;
#else
    return ".";
#endif
  }

  std::string python_executable()
  {
    const char *env_python = std::getenv("DEEPDETECT_PYTHON");
    if (env_python && *env_python)
      return env_python;
    return "python3";
  }

  void configure_pythonpath()
  {
    const std::string bindings_python = source_dir() + "/bindings/python";
    const char *current = std::getenv("PYTHONPATH");
    std::string pythonpath = bindings_python;
    if (current && *current)
      pythonpath += ":" + std::string(current);
    setenv("PYTHONPATH", pythonpath.c_str(), 1);
  }

  std::string repo_path(const std::string &name)
  {
    return "/tmp/deepdetect_" + name;
  }

  void cleanup_repo(const std::string &repo)
  {
    std::error_code ec;
    std::filesystem::remove_all(repo, ec);
  }

  void prepare_repo(const std::string &repo)
  {
    cleanup_repo(repo);
    std::filesystem::create_directories(repo);
  }

  int status_code(const JDoc &doc)
  {
    return doc["status"]["code"].GetInt();
  }

  std::string create_request(const std::string &repo,
                             const std::string &extra_mllib = "")
  {
    return "{\"mllib\":\"pytorch\",\"description\":\"dummy pytorch worker\","
           "\"type\":\"supervised\",\"model\":{\"repository\":\""
           + repo
           + "\"},\"parameters\":{\"input\":{\"connector\":\"image\","
             "\"height\":64,\"width\":64,\"rgb\":true},\"mllib\":{"
             "\"nclasses\":2,\"python\":\""
           + python_executable() + "\"" + extra_mllib + "}}}";
  }

  std::string train_request(const std::string &service, int iterations,
                            bool async)
  {
    return "{\"service\":\"" + service
           + "\",\"async\":" + (async ? "true" : "false")
           + ",\"parameters\":{\"input\":{},\"output\":{\"measure_hist\":true}"
             ","
             "\"mllib\":{\"solver\":{\"iterations\":"
           + std::to_string(iterations)
           + ",\"base_lr\":0.01}}},\"data\":[\"dummy\"]}";
  }

  JDoc poll_until_terminal(JsonAPI &japi, const std::string &service, int job,
                           int max_attempts = 100,
                           bool test_predictions = false)
  {
    JDoc status;
    for (int attempt = 0; attempt < max_attempts; ++attempt)
      {
        const std::string output = test_predictions
                                       ? "\"output\":{\"measure_hist\":true,"
                                         "\"test_predictions\":true}"
                                       : "\"output\":{\"measure_hist\":true}";
        const std::string request
            = "{\"service\":\"" + service + "\",\"job\":" + std::to_string(job)
              + ",\"timeout\":0,\"parameters\":{" + output + "}}";
        status = japi.service_train_status(request);
        EXPECT_EQ(200, status_code(status)) << japi.jrender(status);
        if (status_code(status) != 200)
          return status;
        EXPECT_TRUE(status.HasMember("head")) << japi.jrender(status);
        if (!status.HasMember("head"))
          return status;
        const std::string train_status = status["head"]["status"].GetString();
        if (train_status == "finished" || train_status == "unknown error"
            || train_status == "error")
          return status;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
      }
    ADD_FAILURE() << "training job did not finish";
    return status;
  }

  JDoc poll_until_running(JsonAPI &japi, const std::string &service, int job,
                          int max_attempts = 100)
  {
    JDoc status;
    for (int attempt = 0; attempt < max_attempts; ++attempt)
      {
        const std::string request = "{\"service\":\"" + service
                                    + "\",\"job\":" + std::to_string(job)
                                    + ",\"timeout\":0,\"parameters\":{"
                                      "\"output\":{\"measure_hist\":true}}}";
        status = japi.service_train_status(request);
        EXPECT_EQ(200, status_code(status)) << japi.jrender(status);
        if (status_code(status) != 200)
          return status;
        EXPECT_TRUE(status.HasMember("head")) << japi.jrender(status);
        if (!status.HasMember("head"))
          return status;
        const std::string train_status = status["head"]["status"].GetString();
        if (train_status == "running" || train_status == "finished")
          return status;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
      }
    ADD_FAILURE() << "training job did not report running";
    return status;
  }

}

TEST(pytorchworkerapi, training_status_can_return_test_predictions_payload)
{
  configure_pythonpath();
  JsonAPI japi;
  const std::string service = "pytorchworker_test_predictions";
  const std::string repo = repo_path(service);
  prepare_repo(repo);

  ASSERT_EQ(
      created_str,
      japi.jrender(japi.service_create(
          service, create_request(repo, ",\"emit_test_predictions\":true"))));

  JDoc train = japi.service_train(train_request(service, 2, true));
  ASSERT_EQ(201, status_code(train)) << japi.jrender(train);
  const int job = train["head"]["job"].GetInt();

  JDoc status = poll_until_terminal(japi, service, job, 100, true);
  ASSERT_STREQ("finished", status["head"]["status"].GetString())
      << japi.jrender(status);
  ASSERT_TRUE(status.HasMember("body")) << japi.jrender(status);
  ASSERT_TRUE(status["body"].HasMember("test_predictions"))
      << japi.jrender(status);
  const auto &predictions = status["body"]["test_predictions"];
  ASSERT_TRUE(predictions.HasMember("test0")) << japi.jrender(status);
  ASSERT_TRUE(predictions["test0"].HasMember("samples"))
      << japi.jrender(status);
  ASSERT_EQ(1U, predictions["test0"]["samples"].Size())
      << japi.jrender(status);

  ASSERT_EQ(ok_str, japi.jrender(japi.service_delete(service, "")));
  cleanup_repo(repo);
}

TEST(pytorchworkerapi, service_create_async_train_status_and_predict)
{
  configure_pythonpath();
  JsonAPI japi;
  const std::string service = "pytorchworker_smoke";
  const std::string repo = repo_path(service);
  prepare_repo(repo);

  ASSERT_EQ(created_str,
            japi.jrender(japi.service_create(service, create_request(repo))));

  JDoc train = japi.service_train(train_request(service, 3, true));
  ASSERT_EQ(201, status_code(train)) << japi.jrender(train);
  ASSERT_TRUE(train.HasMember("head")) << japi.jrender(train);
  ASSERT_TRUE(train["head"].HasMember("job")) << japi.jrender(train);
  const int job = train["head"]["job"].GetInt();

  JDoc status = poll_until_terminal(japi, service, job);
  ASSERT_STREQ("finished", status["head"]["status"].GetString())
      << japi.jrender(status);
  ASSERT_TRUE(status.HasMember("body")) << japi.jrender(status);
  ASSERT_TRUE(status["body"].HasMember("measure_hist"))
      << japi.jrender(status);
  const auto &hist = status["body"]["measure_hist"];
  ASSERT_TRUE(hist.HasMember("iteration_hist")) << japi.jrender(status);
  ASSERT_TRUE(hist.HasMember("train_loss_hist")) << japi.jrender(status);
  ASSERT_GE(hist["iteration_hist"].Size(), 1U) << japi.jrender(status);
  ASSERT_DOUBLE_EQ(
      3.0,
      hist["iteration_hist"][hist["iteration_hist"].Size() - 1].GetDouble())
      << japi.jrender(status);

  const std::string predict_request
      = "{\"service\":\"" + service
        + "\",\"parameters\":{\"input\":{\"height\":64,\"width\":64},"
          "\"output\":{\"bbox\":true,\"best\":1}},\"data\":[\"dummy.jpg\"]}";
  JDoc predict = japi.service_predict(predict_request);
  ASSERT_EQ(200, status_code(predict)) << japi.jrender(predict);
  ASSERT_TRUE(predict.HasMember("body")) << japi.jrender(predict);
  ASSERT_TRUE(predict["body"].HasMember("predictions"))
      << japi.jrender(predict);
  ASSERT_EQ(1U, predict["body"]["predictions"].Size())
      << japi.jrender(predict);
  ASSERT_STREQ("dummy.jpg",
               predict["body"]["predictions"][0]["uri"].GetString())
      << japi.jrender(predict);

  ASSERT_EQ(ok_str, japi.jrender(japi.service_delete(service, "")));
  cleanup_repo(repo);
}

TEST(pytorchworkerapi, async_train_can_be_cancelled)
{
  configure_pythonpath();
  JsonAPI japi;
  const std::string service = "pytorchworker_cancel";
  const std::string repo = repo_path(service);
  prepare_repo(repo);

  ASSERT_EQ(created_str,
            japi.jrender(japi.service_create(service, create_request(repo))));

  JDoc train = japi.service_train(train_request(service, 1000, true));
  ASSERT_EQ(201, status_code(train)) << japi.jrender(train);
  const int job = train["head"]["job"].GetInt();
  JDoc running = poll_until_running(japi, service, job);
  ASSERT_TRUE(running.HasMember("body")) << japi.jrender(running);

  const std::string delete_request = "{\"service\":\"" + service
                                     + "\",\"job\":" + std::to_string(job)
                                     + "}";
  JDoc cancelled = japi.service_train_delete(delete_request);
  ASSERT_EQ(200, status_code(cancelled)) << japi.jrender(cancelled);
  ASSERT_TRUE(cancelled.HasMember("head")) << japi.jrender(cancelled);
  ASSERT_TRUE(cancelled["head"].HasMember("status"))
      << japi.jrender(cancelled);
  ASSERT_STREQ("terminated", cancelled["head"]["status"].GetString())
      << japi.jrender(cancelled);

  ASSERT_EQ(ok_str, japi.jrender(japi.service_delete(service, "")));
  cleanup_repo(repo);
}

TEST(pytorchworkerapi, invalid_worker_class_reports_contract_error)
{
  configure_pythonpath();
  JsonAPI japi;
  const std::string service = "pytorchworker_bad_class";
  const std::string repo = repo_path(service);
  prepare_repo(repo);

  JDoc response = japi.service_create(
      service, create_request(repo, ",\"class\":\"MissingWorkerClass\""));
  ASSERT_EQ(500, status_code(response)) << japi.jrender(response);
  ASSERT_TRUE(response["status"].HasMember("dd_msg"))
      << japi.jrender(response);
  const std::string msg = response["status"]["dd_msg"].GetString();
  ASSERT_NE(std::string::npos, msg.find("worker_contract_error")) << msg;

  cleanup_repo(repo);
}
