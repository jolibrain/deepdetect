/**
 * DeepDetect
 * Copyright (c) 2026 Jolibrain
 *
 * This file is part of deepdetect.
 */

#ifndef PYTORCHWORKERMODEL_H
#define PYTORCHWORKERMODEL_H

#include "mlmodel.h"

namespace dd
{
  class PytorchWorkerModel : public MLModel
  {
  public:
    PytorchWorkerModel() : MLModel()
    {
    }

    PytorchWorkerModel(const APIData &ad, APIData &adg,
                       const std::shared_ptr<spdlog::logger> &logger)
        : MLModel(ad, adg, logger)
    {
      read_corresp_file();
    }

    PytorchWorkerModel(const std::string &repo) : MLModel(repo)
    {
    }
  };
}

#endif
