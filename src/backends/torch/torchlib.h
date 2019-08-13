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

#include <random>

#include <torch/torch.h>

#include "apidata.h"
#include "mllibstrategy.h"

#include "torchmodel.h"
#include "torchinputconns.h"

namespace dd
{
    class TorchModule {
    public:
        TorchModule();

        c10::IValue forward(std::vector<c10::IValue> source);

        std::vector<torch::Tensor> parameters();

        void save_checkpoint(TorchModel &model, const std::string &name);

        void load(TorchModel &model);

        void eval();
        void train();
    public:
        std::shared_ptr<torch::jit::script::Module> _traced;
        torch::nn::Linear _classif = nullptr;

        torch::Device _device;
        int _classif_in = 0; /**<id of the input of the classification layer */
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

        int test(const APIData &ad, TInputConnectorStrategy &inputc,
                 TorchDataset &dataset,
                 int batch_size, APIData &out);

    public:
        int _nclasses = 0;
        std::string _template;
        torch::Device _device = torch::Device("cpu");
        bool _masked_lm = false;

        // models
        TorchModule _module;
    };
}

#endif // TORCHLIB_H
