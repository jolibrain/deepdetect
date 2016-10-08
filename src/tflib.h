/**
 * DeepDetect
 * Copyright (c) 2016 Emmanuel Benazera
 * Author: Emmanuel Benazera <beniz@droidnik.fr>
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

#ifndef TFLIB_H
#define TFLIB_H

#include "mllibstrategy.h"
#include "tfmodel.h"

# include <string>
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/framework/tensor.h"

namespace dd
{
  template<class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel=TFModel>
    class TFLib : public MLLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>
    {
    public:
    TFLib(const TFModel &tfmodel);
    TFLib(TFLib &&tl) noexcept;
    ~TFLib();

        /*- from mllib -*/
    void init_mllib(const APIData &ad);

    void clear_mllib(const APIData &d);

    int train(const APIData &ad, APIData &out);

    int predict(const APIData &ad, APIData &out);

    /*- local functions -*/
    //TODO: test()
    

    public:
    // general parameters
    int _nclasses = 0; /**< required. */
    bool _regression = false; /**< whether the net acts as a regressor. */
    int _ntargets = 0; /**< number of classification or regression targets. */
    std::string _inputLayer; // Input Layer of the Tensorflow Model
    std::string _outputLayer; // OutPut layer of the tensorflow Model
    std::unique_ptr<tensorflow::Session> _session = nullptr;
    };
  
}

#endif
