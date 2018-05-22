/**
 * DeepDetect
 * Copyright (c) 2014-2015 Emmanuel Benazera
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

#ifndef CAFFE2LIB_H
#define CAFFE2LIB_H

//TODO Remove that to print the warnings
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <caffe2/core/workspace.h>
#pragma GCC diagnostic pop

#include "mllibstrategy.h"
#include "backends/caffe2/caffe2model.h"

namespace dd
{
  /**
   * \brief Caffe2 library wrapper for deepdetect
   */
  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel=Caffe2Model>
    class Caffe2Lib : public MLLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>
    {
    public:
    /**
     * \brief constructor from model
     * @param model Caffe2 model
     */
    Caffe2Lib(const Caffe2Model &c2model);

    /**
     * \brief copy-constructor
     */
    Caffe2Lib(Caffe2Lib &&cl) noexcept;

    /**
     * \brief destructor
     */
    ~Caffe2Lib();

    /**
     * \brief instanciate a model from template
     * @param ad mllib data object
     */
    void instantiate_template(const APIData &ad);

    /*- from mllib -*/
    void init_mllib(const APIData &ad);
    void clear_mllib(const APIData &ad);
    int train(const APIData &ad, APIData &out);

    /**
     * \brief predicts from model
     * @param ad root data object
     * @param out output data object (e.g. predictions, ...)
     * @return 0 if OK, 1 otherwise
     */
    int predict(const APIData &ad, APIData &out);

    public:
    caffe2::Workspace _workspace;
    caffe2::NetDef _init_net, _predict_net;
    };
}

#endif
