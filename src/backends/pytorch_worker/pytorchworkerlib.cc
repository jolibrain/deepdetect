/**
 * DeepDetect
 * Copyright (c) 2026 Jolibrain
 *
 * This file is part of deepdetect.
 */

#include "backends/pytorch_worker/pytorchworkerlib.h"

#include "backends/pytorch_worker/pytorchworkerinputconns.h"
#include "supervisedoutputconnector.h"

#include <cstdlib>
#include <iostream>
#include <rapidjson/document.h>

namespace dd
{
  namespace
  {
    bool json_number(const rapidjson::Value &value, double &out)
    {
      if (!value.IsNumber())
        return false;
      out = value.GetDouble();
      return true;
    }

    std::string json_string(const rapidjson::Value &value,
                            const std::string &fallback = "")
    {
      return value.IsString() ? value.GetString() : fallback;
    }

    bool debug_enabled()
    {
      const char *debug = std::getenv("DEEPDETECT_DEBUG");
      const char *worker_debug = std::getenv("DEEPDETECT_WORKER_DEBUG");
      return (debug && *debug) || (worker_debug && *worker_debug);
    }

    void debug_log(const std::string &message)
    {
      if (debug_enabled())
        std::cerr << "[deepdetect-debug][pytorch-lib] " << message
                  << std::endl;
    }
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  PytorchWorkerLib<TInputConnectorStrategy, TOutputConnectorStrategy,
                   TMLModel>::PytorchWorkerLib(const PytorchWorkerModel &model)
      : MLLib<TInputConnectorStrategy, TOutputConnectorStrategy, TMLModel>(
          model)
  {
    this->_libname = "pytorch";
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  PytorchWorkerLib<TInputConnectorStrategy, TOutputConnectorStrategy,
                   TMLModel>::PytorchWorkerLib(PytorchWorkerLib
                                                   &&other) noexcept
      : MLLib<TInputConnectorStrategy, TOutputConnectorStrategy, TMLModel>(
          std::move(other)),
        _worker(std::move(other._worker)),
        _mllib_params(std::move(other._mllib_params)),
        _nclasses(other._nclasses)
  {
    this->_libname = "pytorch";
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  PytorchWorkerLib<TInputConnectorStrategy, TOutputConnectorStrategy,
                   TMLModel>::~PytorchWorkerLib()
  {
    if (_worker)
      _worker->shutdown();
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  void PytorchWorkerLib<TInputConnectorStrategy, TOutputConnectorStrategy,
                        TMLModel>::init_mllib(const APIData &ad)
  {
    _mllib_params = ad;
    if (ad.has("nclasses"))
      _nclasses = static_cast<unsigned int>(ad.get("nclasses").get<int>());
    if (_nclasses == 0)
      _nclasses = 1;
    if (ad.has("task"))
      this->_mltype = ad.get("task").get<std::string>();
    else
      this->_mltype = "supervised";
    configure_worker(ad);
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  void PytorchWorkerLib<TInputConnectorStrategy, TOutputConnectorStrategy,
                        TMLModel>::configure_worker(const APIData &ad)
  {
    debug_log("configure_worker: starting supervisor");
    _worker = std::make_shared<PytorchWorkerSupervisor>(this->_mlmodel._repo,
                                                        this->_logger);
    _worker->start(ad);

    APIData hello_params;
    hello_params.add("protocol_version", 1);
    debug_log("configure_worker: sending hello");
    _worker->request("hello", hello_params);

    APIData params;
    params.add("repository", this->_mlmodel._repo);
    params.add("mllib", ad);
    debug_log("configure_worker: sending configure");
    _worker->request("configure", params);
    debug_log("configure_worker: configured");
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  void PytorchWorkerLib<TInputConnectorStrategy, TOutputConnectorStrategy,
                        TMLModel>::clear_mllib(const APIData &ad)
  {
    (void)ad;
    if (_worker)
      _worker->shutdown();
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  APIData PytorchWorkerLib<TInputConnectorStrategy, TOutputConnectorStrategy,
                           TMLModel>::request_params(const APIData &ad) const
  {
    APIData params;
    params.add("repository", this->_mlmodel._repo);
    params.add("request", ad);
    return params;
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  APIData PytorchWorkerLib<TInputConnectorStrategy, TOutputConnectorStrategy,
                           TMLModel>::train_request(const APIData &ad)
  {
    if (!connector_tensor_inline_requested(ad))
      return ad;
    debug_log("train_request: building connector_tensor_inline batches");
    APIData request = ad;
    request.add("data", std::vector<std::string>());
    request.add("tensor_batches", this->_inputc.inline_tensor_batches(
                                      ad, connector_effective_mllib(ad)));
    return request;
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  bool
  PytorchWorkerLib<TInputConnectorStrategy, TOutputConnectorStrategy,
                   TMLModel>::connector_tensor_inline_requested(const APIData
                                                                    &ad) const
  {
    return connector_data_source_requested(ad, "connector_tensor_inline");
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  bool
  PytorchWorkerLib<TInputConnectorStrategy, TOutputConnectorStrategy,
                   TMLModel>::connector_tensor_pull_requested(const APIData
                                                                  &ad) const
  {
    return connector_data_source_requested(ad, "connector_tensor_pull");
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  bool PytorchWorkerLib<
      TInputConnectorStrategy, TOutputConnectorStrategy,
      TMLModel>::connector_data_source_requested(const APIData &ad,
                                                 const std::string &name) const
  {
    if (ad.has("parameters"))
      {
        APIData parameters = ad.getobj("parameters");
        if (parameters.has("mllib"))
          {
            APIData mllib = parameters.getobj("mllib");
            if (mllib.has("data_source"))
              return mllib.get("data_source").get<std::string>() == name;
          }
      }
    if (_mllib_params.has("data_source"))
      return _mllib_params.get("data_source").get<std::string>() == name;
    return false;
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  APIData PytorchWorkerLib<TInputConnectorStrategy, TOutputConnectorStrategy,
                           TMLModel>::connector_effective_mllib(const APIData
                                                                    &ad) const
  {
    APIData mllib = _mllib_params;
    if (ad.has("parameters"))
      {
        APIData parameters = ad.getobj("parameters");
        if (parameters.has("mllib"))
          {
            APIData request_mllib = parameters.getobj("mllib");
            for (const std::string &key : request_mllib.list_keys())
              mllib.add(key, request_mllib.get(key));
          }
      }
    return mllib;
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  int PytorchWorkerLib<TInputConnectorStrategy, TOutputConnectorStrategy,
                       TMLModel>::train(const APIData &ad, APIData &out)
  {
    if (!_worker)
      throw MLLibInternalException("pytorch worker is not configured");

    std::lock_guard<std::mutex> lock(_worker_mutex);
    const bool pull_requested = connector_tensor_pull_requested(ad);
    if (pull_requested)
      {
        debug_log("train: starting connector_tensor_pull session");
        this->_inputc.start_tensor_pull_session(ad,
                                                connector_effective_mllib(ad));
      }
    try
      {
        APIData params = request_params(train_request(ad));
        this->_tjob_running.store(true);
        debug_log("train: sending train_start");
        _worker->request("train_start", params);
        debug_log("train: train_start acknowledged");

        bool finished = false;
        bool cancel_sent = false;
        int status = 1;
        while (!finished)
          {
            if (!this->_tjob_running.load() && !cancel_sent)
              {
                APIData cancel_params;
                try
                  {
                    int request_id = -1;
                    _worker->send_request("train_cancel", cancel_params,
                                          request_id);
                  }
                catch (...)
                  {
                  }
                cancel_sent = true;
              }
            std::string message;
            if (!_worker->read_message(message, 200))
              continue;
            if (process_worker_request(message))
              continue;
            process_worker_message(message, finished, status, out);
          }
        debug_log("train: finished with status=" + std::to_string(status));
        this->_tjob_running.store(false);
        if (pull_requested)
          this->_inputc.cleanup_inline_detection_pull_session();
        return status;
      }
    catch (...)
      {
        this->_tjob_running.store(false);
        if (pull_requested)
          this->_inputc.cleanup_inline_detection_pull_session();
        throw;
      }
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  bool PytorchWorkerLib<TInputConnectorStrategy, TOutputConnectorStrategy,
                        TMLModel>::process_worker_request(const std::string
                                                              &message)
  {
    rapidjson::Document doc;
    doc.Parse<rapidjson::kParseNanAndInfFlag>(message.c_str());
    if (doc.HasParseError() || !doc.IsObject() || !doc.HasMember("method")
        || !doc.HasMember("id") || !doc["id"].IsInt())
      return false;

    const int request_id = doc["id"].GetInt();
    const std::string method = json_string(doc["method"]);
    debug_log("process_worker_request: method=" + method);
    try
      {
        APIData result;
        if (method == "connector_dataset_info")
          result = this->_inputc.connector_dataset_info();
        else if (method == "connector_batch_next")
          {
            APIData params;
            if (doc.HasMember("params") && doc["params"].IsObject())
              params.fromRapidJson(doc["params"]);
            result = this->_inputc.connector_batch_next(params);
          }
        else if (method == "connector_batch_done")
          {
            APIData params;
            if (doc.HasMember("params") && doc["params"].IsObject())
              params.fromRapidJson(doc["params"]);
            result = this->_inputc.connector_batch_done(params);
          }
        else
          {
            _worker->send_error_response(request_id, "worker_contract_error",
                                         "unknown worker request method: "
                                             + method);
            return true;
          }
        _worker->send_response(request_id, result);
      }
    catch (const std::exception &error)
      {
        _worker->send_error_response(request_id, "dataset_contract_error",
                                     error.what());
      }
    return true;
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  void PytorchWorkerLib<TInputConnectorStrategy, TOutputConnectorStrategy,
                        TMLModel>::process_worker_message(const std::string
                                                              &message,
                                                          bool &finished,
                                                          int &status,
                                                          APIData &out)
  {
    rapidjson::Document doc;
    doc.Parse<rapidjson::kParseNanAndInfFlag>(message.c_str());
    if (doc.HasParseError() || !doc.IsObject() || !doc.HasMember("event"))
      return;

    std::string event = json_string(doc["event"]);
    debug_log("process_worker_message: event=" + event);
    const rapidjson::Value &payload
        = doc.HasMember("payload") ? doc["payload"] : doc;
    if (event == "metric" && payload.IsObject())
      process_metric(payload);
    else if (event == "status" && payload.IsObject())
      process_status(payload);
    else if (event == "failure" && payload.IsObject())
      process_failure(payload);
    else if (event == "train_result")
      {
        finished = true;
        std::string state = payload.IsObject() && payload.HasMember("status")
                                ? json_string(payload["status"], "finished")
                                : "finished";
        status = state == "finished" ? 0 : 1;
        this->collect_measures(out);
      }
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  void
  PytorchWorkerLib<TInputConnectorStrategy, TOutputConnectorStrategy,
                   TMLModel>::process_metric(const rapidjson::Value &payload)
  {
    if (!payload.HasMember("name") || !payload["name"].IsString()
        || !payload.HasMember("value"))
      return;
    double value = 0.0;
    if (!json_number(payload["value"], value))
      return;
    std::string name = payload["name"].GetString();
    this->add_meas(name, value);
    this->add_meas_per_iter(name, value);
    if (payload.HasMember("iteration"))
      {
        double iteration = 0.0;
        if (json_number(payload["iteration"], iteration))
          this->add_meas("iteration", iteration);
      }
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  void
  PytorchWorkerLib<TInputConnectorStrategy, TOutputConnectorStrategy,
                   TMLModel>::process_status(const rapidjson::Value &payload)
  {
    for (auto it = payload.MemberBegin(); it != payload.MemberEnd(); ++it)
      {
        std::string name = it->name.GetString();
        if (name == "test_predictions" && it->value.IsObject())
          {
            APIData status_payload;
            status_payload.fromRapidJson(it->value);
            this->add_status_payload(name, status_payload);
            continue;
          }
        double value = 0.0;
        if (json_number(it->value, value))
          this->add_meas(name, value);
      }
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  void
  PytorchWorkerLib<TInputConnectorStrategy, TOutputConnectorStrategy,
                   TMLModel>::process_failure(const rapidjson::Value &payload)
  {
    std::string category
        = payload.HasMember("category")
              ? json_string(payload["category"], "internal_error")
              : "internal_error";
    std::string message
        = payload.HasMember("message")
              ? json_string(payload["message"], "worker failure")
              : "worker failure";
    if (payload.HasMember("traceback") && payload["traceback"].IsString())
      message += " traceback=" + std::string(payload["traceback"].GetString());
    throw MLLibInternalException("pytorch worker " + category + ": "
                                 + message);
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  oatpp::Object<DTO::PredictBody>
  PytorchWorkerLib<TInputConnectorStrategy, TOutputConnectorStrategy,
                   TMLModel>::predict(const APIData &ad_in)
  {
    if (!_worker)
      throw MLLibInternalException("pytorch worker is not configured");

    std::lock_guard<std::mutex> lock(_worker_mutex);
    APIData request = ad_in;
    APIData prepared = this->_inputc.keypoint_prediction_tensor_batch(
        ad_in, connector_effective_mllib(ad_in));
    std::string prediction_batch_id;
    if (!prepared.empty())
      {
        request.add(
            "pose_sources",
            prepared.get("source_paths").get<std::vector<std::string>>());
        if (!prepared.get("empty").get<bool>())
          {
            prediction_batch_id = prepared.get("batch_id").get<std::string>();
            request.add("tensor_batch", prepared.getobj("batch"));
          }
      }
    APIData response;
    try
      {
        response = _worker->request("predict", request_params(request));
      }
    catch (...)
      {
        if (!prediction_batch_id.empty())
          {
            APIData done;
            done.add("batch_id", prediction_batch_id);
            this->_inputc.connector_batch_done(done);
          }
        if (!prepared.empty())
          this->_inputc.cleanup_inline_detection_pull_session();
        throw;
      }
    if (!prediction_batch_id.empty())
      {
        APIData done;
        done.add("batch_id", prediction_batch_id);
        this->_inputc.connector_batch_done(done);
      }
    if (!prepared.empty())
      this->_inputc.cleanup_inline_detection_pull_session();
    APIData result = response.getobj("result");
    std::vector<APIData> results = result.getv("results");

    TOutputConnectorStrategy outputc(this->_outputc);
    outputc.add_results(results);

    oatpp::Object<DTO::ServicePredict> predict_dto
        = ad_in.createSharedDTO<DTO::ServicePredict>();
    auto output_params = predict_dto->parameters->output;
    OutputConnectorConfig config;
    config._nclasses = static_cast<int>(_nclasses);
    config._has_bbox = output_params->bbox;
    config._has_keypoints = output_params->keypoints;
    config._has_mask = output_params->image;
    config._regression = output_params->regression;
    return outputc.finalize(output_params, config,
                            static_cast<MLModel *>(&this->_mlmodel));
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel>
  int PytorchWorkerLib<TInputConnectorStrategy, TOutputConnectorStrategy,
                       TMLModel>::status() const
  {
    return 0;
  }

  template class PytorchWorkerLib<ImgPytorchInputFileConn, SupervisedOutput,
                                  PytorchWorkerModel>;
}
