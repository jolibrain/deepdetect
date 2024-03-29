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

#ifndef NCNNLIB_H
#define NCNNLIB_H

#include "apidata.h"
#include "utils/utils.hpp"

#include "dto/mllib.hpp"

// NCNN
#include "net.h"
#include "ncnnmodel.h"

namespace dd
{
  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel = NCNNModel>
  class NCNNLib : public MLLib<TInputConnectorStrategy,
                               TOutputConnectorStrategy, TMLModel>
  {
  public:
    NCNNLib(const NCNNModel &tmodel);
    NCNNLib(NCNNLib &&tl) noexcept;
    ~NCNNLib();

    /*- from mllib -*/
    void init_mllib(const APIData &ad);

    void clear_mllib(const APIData &ad);

    int train(const APIData &ad, APIData &out);

    oatpp::Object<DTO::PredictBody> predict(const APIData &ad);

    void model_type(const std::string &param_file, std::string &mltype);

  public:
    ncnn::Net *_net = nullptr;
    bool _timeserie = false;

  private:
    oatpp::Object<DTO::MLLib> _init_dto;
    static ncnn::UnlockedPoolAllocator _blob_pool_allocator;
    static ncnn::PoolAllocator _workspace_pool_allocator;

  protected:
    int _old_height = -1;
  };

}

#endif
