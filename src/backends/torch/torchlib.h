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
    // TODO: Make TorchModule inherit torch::nn::Module ? And use the TORCH_MODULE macro
    class TorchModule {
    public:
        TorchModule();

        c10::IValue forward(std::vector<c10::IValue> source);

        void freeze_traced(bool freeze);

        std::vector<torch::Tensor> parameters();

        /** Save traced module to checkpoint-[name].pt, and custom parts weights
         * to checkpoint-[name].ptw */
        // (Actually only _classif is saved in the .ptw)
        void save_checkpoint(TorchModel &model, const std::string &name);

        /** Load traced module from .pt and custom parts weights from .ptw */
        void load(TorchModel &model);

        void eval();
        void train();

        void free();
    public:
        std::shared_ptr<torch::jit::script::Module> _traced;
        torch::nn::Linear _classif = nullptr;

        torch::Device _device;
        int _classif_in = 0; /**<id of the input of the classification layer */
        bool _hidden_states = false; /**< Take BERT hidden states as input. */
    private:
        bool _freeze_traced = false; /**< Freeze weights of the traced module */
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
        bool _finetuning = false;
        torch::Device _device = torch::Device("cpu");
        bool _masked_lm = false;
        bool _seq_training = false; 

        // models
        TorchModule _module;
    };
}

#endif // TORCHLIB_H
