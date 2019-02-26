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

// NCNN
#include "net.h"
#include "ncnnmodel.h"

#include "apidata.h"

namespace dd
{
    template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel=NCNNModel>
    class NCNNLib : public MLLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>
    {
    public:
        NCNNLib(const NCNNModel &tmodel);
        NCNNLib(NCNNLib &&tl) noexcept;
        ~NCNNLib();

        /*- from mllib -*/
        void init_mllib(const APIData &ad);

        void clear_mllib(const APIData &ad);

        int train(const APIData &ad, APIData &out);

        int predict(const APIData &ad, APIData &out);

        void model_type(const std::string &param_file,
			std::string &mltype);
    
    public:
        ncnn::Net *_net = nullptr;
        int _nclasses = 0;
        bool _timeserie =  false;
    private:
        static ncnn::UnlockedPoolAllocator _blob_pool_allocator;
        static ncnn::PoolAllocator _workspace_pool_allocator;
    protected:
        int _threads = 1;
    int _old_height = -1;
    };
}

#endif
