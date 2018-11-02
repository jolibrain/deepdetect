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

#ifndef NCNNINPUTCONNS_H
#define NCNNINPUTCONNS_H

#include "imginputfileconn.h"

namespace dd
{
    class NCNNInputInterface
    {
    public:
        NCNNInputInterface() {}
        ~NCNNInputInterface() {}
    };

    class ImgNCNNInputFileConn : public ImgInputFileConn, public NCNNInputInterface
    {
    public:
        ImgNCNNInputFileConn()
            :ImgInputFileConn() {}
        ImgNCNNInputFileConn(const ImgNCNNInputFileConn &i)
            :ImgInputFileConn(i),NCNNInputInterface(i) {}
        ~ImgNCNNInputFileConn() {}

        // for API info only
        int width() const
        {
            return _width;
        }

        // for API info only
        int height() const
        {
            return _height;
        }

        void init(const APIData &ad)
        {
            ImgInputFileConn::init(ad);
        }
        
        void transform(const APIData &ad)
        {

        }
    };
}

#endif