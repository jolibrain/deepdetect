/**
 * DeepDetect
 * Copyright (c) 2026 Jolibrain
 *
 * This file is part of deepdetect.
 */

#ifndef PYTORCHWORKERLIB_H
#define PYTORCHWORKERLIB_H

#include "apidata.h"
#include "dto/predict_out.hpp"
#include "dto/service_predict.hpp"
#include "mllibstrategy.h"
#include "outputconnectorstrategy.h"
#include "pytorchworkermodel.h"
#include "pytorchworkersupervisor.h"

#include <memory>
#include <mutex>
#include <rapidjson/document.h>

namespace dd
{
  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel = PytorchWorkerModel>
  class PytorchWorkerLib : public MLLib<TInputConnectorStrategy,
                                        TOutputConnectorStrategy, TMLModel>
  {
  public:
    PytorchWorkerLib(const PytorchWorkerModel &model);
    PytorchWorkerLib(PytorchWorkerLib &&other) noexcept;
    ~PytorchWorkerLib();

    void init_mllib(const APIData &ad);
    void clear_mllib(const APIData &ad);
    int train(const APIData &ad, APIData &out);
    oatpp::Object<DTO::PredictBody> predict(const APIData &ad_in);
    int status() const;

  private:
    void configure_worker(const APIData &ad);
    void process_worker_message(const std::string &message, bool &finished,
                                int &status, APIData &out);
    bool process_worker_request(const std::string &message);
    void process_metric(const rapidjson::Value &payload);
    void process_status(const rapidjson::Value &payload);
    void process_failure(const rapidjson::Value &payload);
    APIData request_params(const APIData &ad) const;
    APIData train_request(const APIData &ad);
    bool connector_tensor_inline_requested(const APIData &ad) const;
    bool connector_tensor_pull_requested(const APIData &ad) const;
    bool connector_data_source_requested(const APIData &ad,
                                         const std::string &name) const;
    APIData connector_effective_mllib(const APIData &ad) const;

    std::shared_ptr<PytorchWorkerSupervisor> _worker;
    APIData _mllib_params;
    unsigned int _nclasses = 1;
    mutable std::mutex _worker_mutex;
  };
}

#endif
