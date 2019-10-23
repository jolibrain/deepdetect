/**
 * DeepDetect
 * Copyright (c) 2018 Pixel Forensics, Inc.
 * Author: Cheni Chadowitz <cchadowitz@pixelforensics.com>
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


    public:
        // general parameters

        std::string _net_type; // model type
        // model, depending on type specified
        net_type_objDetector _objDetector;
        net_type_faceDetector _faceDetector;
        net_type_faceFeatureExtractor _faceFeatureExtractor;
        // whether the model has been loaded yet
        bool _modelLoaded = false;
        std::mutex _net_mutex; /**< mutex around net, e.g. no concurrent predict calls as net is not re-instantiated. Use batches instead. */
    };

}

#endif
