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

#ifndef DLIBLIB_H
#define DLIBLIB_H

#include "DNNStructures.h"

#include "mllibstrategy.h"
#include "dlibmodel.h"

#include <string>

namespace dd {
    template<class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel=DlibModel>
    class DlibLib : public MLLib<TInputConnectorStrategy, TOutputConnectorStrategy, TMLModel> {
    public:
        DlibLib(const DlibModel &tfmodel);

        DlibLib(DlibLib &&tl) noexcept;

        ~DlibLib();

        /*- from mllib -*/
        void init_mllib(const APIData &ad);

        void clear_mllib(const APIData &d);

        int train(const APIData &ad, APIData &out);

        void test(const APIData &ad, APIData &out);

        int predict(const APIData &ad, APIData &out);

        /*- local functions -*/
//        template<class T>
//        dlib::loss_mmod<T> generate(const std::string &type);
        auto getModel() {
            if (net_type.empty()) {
                throw MLLibInternalException("Cannot get model before model is loaded");
            } else if (net_type == "object_detector") {
                return objDetector;
            } else if (net_type == "face_detector") {
                return faceDetector;
            } else {
                throw MLLibInternalException("Unrecognized net type: " + net_type);
            }
        }


    public:
        // general parameters
        int _nclasses = 0; /**< required. */
        bool _regression = false; /**< whether the net acts as a regressor. */
        int _ntargets = 0; /**< number of classification or regression targets. */
        std::string _inputLayer; // input Layer of the model
        std::string _outputLayer; // output layer of the model
        APIData _inputFlag; // boolean input to the model
//        template<class T>
//        net_type<T> model;
        net_type_objDetector objDetector;
        net_type_faceDetector faceDetector;
        std::string net_type;
        std::mutex _net_mutex; /**< mutex around net, e.g. no concurrent predict calls as net is not re-instantiated. Use batches instead. */
    };

}

#endif
