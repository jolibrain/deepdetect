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

    void test(const APIData &ad, APIData &out);
    
    int predict(const APIData &ad, APIData &out);

    /*- local functions -*/
    void tf_concat(const std::vector<tensorflow::Tensor> &dv,
		   std::vector<tensorflow::Tensor> &vtfinputs);
    

    public:
    // general parameters
    int _nclasses = 0; /**< required. */
    bool _regression = false; /**< whether the net acts as a regressor. */
    int _ntargets = 0; /**< number of classification or regression targets. */
    std::string _inputLayer; // input Layer of the model
    std::string _outputLayer; // output layer of the model
    APIData _inputFlag; // boolean input to the model
    std::unique_ptr<tensorflow::Session> _session = nullptr;
    std::mutex _net_mutex; /**< mutex around net, e.g. no concurrent predict calls as net is not re-instantiated. Use batches instead. */
    };
  
}

#endif
