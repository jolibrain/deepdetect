/**
 * DeepDetect
 * Copyright (c) 2019 Jolibrain
 * Author: Louis Jean <ljean@etud.insa-toulouse.fr>
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

#ifndef TORCHLIB_H
#define TORCHLIB_H

#include <torch/torch.h>

#include "apidata.h"
#include "mllibstrategy.h"

#include "torchmodel.h"
#include "torchinputconns.h"

namespace dd
{
    class TorchModule {
    public:
        c10::IValue forward(std::vector<c10::IValue> source);

        std::vector<torch::Tensor> parameters();

        void save(const std::string &filename);

        void load(const std::string &filename);
    public:
        std::shared_ptr<torch::jit::script::Module> _traced;
        torch::nn::Linear _classif = nullptr;
    };


    template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel=TorchModel>
    class TorchLib : public MLLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>
    {
    public:
        TorchLib(const TorchModel &tmodel);
        TorchLib(TorchLib &&tl) noexcept;
        ~TorchLib();

        /*- from mllib -*/
        void init_mllib(const APIData &ad);

        void clear_mllib(const APIData &ad);

        int train(const APIData &ad, APIData &out);

        int predict(const APIData &ad, APIData &out);

        int test(const APIData &ad, APIData &out);

    public:
        int _nclasses = 0;
        torch::Device _device = torch::Device("cpu");

        // models
        TorchModule _module;
    };
}

#endif // TORCHLIB_H
