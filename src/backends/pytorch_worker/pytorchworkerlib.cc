/**
 * DeepDetect
 * Copyright (c) 2026 Jolibrain
 *
 * This file is part of deepdetect.
 */

#include "backends/pytorch_worker/pytorchworkerlib.h"

#include "backends/pytorch_worker/pytorchworkerinputconns.h"
#include "supervisedoutputconnector.h"

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
    _worker = std::make_shared<PytorchWorkerSupervisor>(this->_mlmodel._repo,
                                                        this->_logger);
    _worker->start(ad);

    APIData hello_params;
    hello_params.add("protocol_version", 1);
    _worker->request("hello", hello_params);

    APIData params;
    params.add("repository", this->_mlmodel._repo);
    params.add("mllib", ad);
    _worker->request("configure", params);
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
  int PytorchWorkerLib<TInputConnectorStrategy, TOutputConnectorStrategy,
                       TMLModel>::train(const APIData &ad, APIData &out)
  {
    if (!_worker)
      throw MLLibInternalException("pytorch worker is not configured");

    std::lock_guard<std::mutex> lock(_worker_mutex);
    this->_tjob_running.store(true);
    APIData params = request_params(ad);
    _worker->request("train_start", params);

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
        process_worker_message(message, finished, status, out);
      }
    this->_tjob_running.store(false);
    return status;
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
        double value = 0.0;
        if (json_number(it->value, value))
          this->add_meas(it->name.GetString(), value);
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
    APIData params = request_params(ad_in);
    APIData response = _worker->request("predict", params);
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
