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

#include "torchlib.h"

#include <torch/script.h>

#include "outputconnectorstrategy.h"

using namespace torch;

namespace dd
{
    void add_parameters(std::shared_ptr<torch::jit::script::Module> module, std::vector<Tensor> &params) { 
        for (const auto &slot : module->get_parameters()) {
            params.push_back(slot.value().toTensor());
        }
        for (auto child : module->get_modules()) {
            add_parameters(child, params);
        }
    }

    /// Convert IValue to Tensor and throw an exception if the IValue is not a Tensor.
    Tensor to_tensor_safe(const IValue &value) {
        if (!value.isTensor())
            throw MLLibInternalException("Expected Tensor, found " + value.tagKind());
        return value.toTensor();
    }
 
    /// Convert id Tensor to one_hot Tensor
    void fill_one_hot(Tensor &one_hot, Tensor ids, int nclasses)
    {
        one_hot.zero_();
        for (int i = 0; i < ids.size(0); ++i)
        {
            one_hot[i][ids[i].item<int>()] = 1;
        }
    }

    Tensor to_one_hot(Tensor ids, int nclasses)
    {
        Tensor one_hot = torch::zeros(IntList{ids.size(0), nclasses});
        for (int i = 0; i < ids.size(0); ++i)
        {
            one_hot[i][ids[i].item<int>()] = 1;
        }
        return one_hot;
    }

    // ======= TORCH MODULE


    TorchModule::TorchModule() {}

    c10::IValue TorchModule::forward(std::vector<c10::IValue> source) 
    {
        if (_traced)
        {
            source = { _traced->forward(source) };
        }
        c10::IValue out_val = source.at(0);
        if (_classif)
        {
            out_val = _classif->forward(to_tensor_safe(out_val));
        }
        return out_val;
    }

    std::vector<Tensor> TorchModule::parameters() 
    {
        std::vector<Tensor> params;
        if (_traced)
            add_parameters(_traced, params);
        if (_classif)
        {
            auto classif_params = _classif->parameters();
            params.insert(params.end(), classif_params.begin(), classif_params.end());
        }
        return params;
    }

    void TorchModule::save_checkpoint(TorchModel &model, const std::string &name) 
    {
        if (_traced)
            _traced->save(model._repo + "/checkpoint-" + name + ".pt");
    }

    void TorchModule::load_checkpoint(const std::string &filename) 
    {
        // Not yet implemented
    }


    // ======= TORCHLIB

    template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
    TorchLib<TInputConnectorStrategy, TOutputConnectorStrategy, TMLModel>::TorchLib(const TorchModel &tmodel)
        : MLLib<TInputConnectorStrategy,TOutputConnectorStrategy,TorchModel>(tmodel) 
    {
        this->_libname = "torch";
    }

    template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
    TorchLib<TInputConnectorStrategy, TOutputConnectorStrategy, TMLModel>::TorchLib(TorchLib &&tl) noexcept
        : MLLib<TInputConnectorStrategy,TOutputConnectorStrategy,TorchModel>(std::move(tl))
    {
        this->_libname = "torch";
        _module = std::move(tl._module);
        _nclasses = tl._nclasses;
        _device = tl._device;
    }

    template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
    TorchLib<TInputConnectorStrategy, TOutputConnectorStrategy, TMLModel>::~TorchLib() 
    {
        
    }

    /*- from mllib -*/
    template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
    void TorchLib<TInputConnectorStrategy, TOutputConnectorStrategy, TMLModel>::init_mllib(const APIData &lib_ad)
    {
        bool gpu = false;
        int gpuid = -1;

        if (lib_ad.has("gpu")) {
            gpu = lib_ad.get("gpu").get<bool>() && torch::cuda::is_available();
        }
        if (lib_ad.has("gpuid"))
            gpuid = lib_ad.get("gpuid").get<int>();
        if (lib_ad.has("nclasses")) {
            _nclasses = lib_ad.get("nclasses").get<int>();
        }

        _device = gpu ? torch::Device(DeviceType::CUDA, gpuid) : torch::Device(DeviceType::CPU);

        if (typeid(TInputConnectorStrategy) == typeid(TxtTorchInputFileConn)) {
            //_attention = true;
        }

        // TODO load classification layer, or create it according to the number of classes
        _module._traced = torch::jit::load(this->_mlmodel._model_file, _device);
        _module._traced->eval();
    }

    template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
    void TorchLib<TInputConnectorStrategy, TOutputConnectorStrategy, TMLModel>::clear_mllib(const APIData &ad) 
    {

    }

    template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
    int TorchLib<TInputConnectorStrategy, TOutputConnectorStrategy, TMLModel>::train(const APIData &ad, APIData &out) 
    {
        TInputConnectorStrategy inputc(this->_inputc);
        inputc._train = true;

        try
        {
            inputc.transform(ad);
        }
        catch (...)
        {
            throw;
        }

        APIData ad_mllib = ad.getobj("parameters").getobj("mllib");

        // solver params
        int64_t iterations = 100;
        std::string solver_type = "SGD";
        double base_lr = 0.0001;
        int64_t batch_size = 5;
        int64_t test_batch_size = 1;
        int64_t test_interval = 1;
        int64_t save_period = 0;
        int64_t batch_count = (inputc._dataset.cache_size() - 1) / batch_size + 1;

        // logging parameters
        int64_t log_batch_period = 1;

        if (ad_mllib.has("iterations"))
            iterations = ad_mllib.get("iterations").get<int>();
        if (ad_mllib.has("solver_type"))
            solver_type = ad_mllib.get("solver_type").get<std::string>();
        if (ad_mllib.has("base_lr"))
            base_lr = ad_mllib.get("base_lr").get<double>();
        if (ad_mllib.has("test_interval"))
            test_interval = ad_mllib.get("test_interval").get<int>();
        if (ad_mllib.has("batch_size"))
            batch_size = ad_mllib.get("batch_size").get<int>();
        if (ad_mllib.has("test_batch_size"))
            test_batch_size = ad_mllib.get("test_batch_size").get<int>();
        if (ad_mllib.has("save_period"))
            save_period = ad_mllib.get("save_period").get<int>();

        // create dataset for evaluation during training
        TorchDataset eval_dataset;
        if (!inputc._test_dataset.empty())
        {
            eval_dataset = inputc._test_dataset; //.split(0, 0.1);
        }

        // create solver
        // no care about solver type yet
        optim::Adam optimizer(_module.parameters(), optim::AdamOptions(base_lr));

        // create dataloader
        auto dataloader = torch::data::make_data_loader(
            std::move(inputc._dataset),
            data::DataLoaderOptions(batch_size)
        );
        this->_logger->info("Training for {} iterations", iterations);

        for (int64_t epoch = 0; epoch < iterations; ++epoch)
        {
            this->add_meas("iteration", epoch);
            this->_logger->info("Iteration {}", epoch);
            int batch_id = 0;

            for (TorchBatch &example : *dataloader)
            {
                std::vector<c10::IValue> in_vals;
                for (Tensor tensor : example.data)
                    in_vals.push_back(tensor.to(_device));
                Tensor y_pred = to_tensor_safe(_module.forward(in_vals));
                Tensor y = to_one_hot(example.target.at(0), _nclasses).to(_device);

                // TODO let loss be a parameter
                auto loss = torch::mse_loss(y_pred, y);
                double loss_val = loss.item<double>();

                optimizer.zero_grad();
                loss.backward();
                optimizer.step();

                this->add_meas("train_loss", loss_val);
                this->add_meas_per_iter("train_loss", loss_val);
                if (log_batch_period != 0 && (batch_id + 1) % log_batch_period == 0)
                {
                    this->_logger->info("Batch {}/{}: loss is {}", batch_id + 1, batch_count, loss_val);
                }
                ++batch_id;
            }
            
            if (epoch > 0 && epoch % test_interval == 0 && !eval_dataset.empty())
            {
                APIData meas_out;
                test(ad, eval_dataset, test_batch_size, meas_out);
                eval_dataset = inputc._test_dataset; // XXX eval_dataset is moved. find a way to unmove it
	            APIData meas_obj = meas_out.getobj("measure");
                std::vector<std::string> meas_names = meas_obj.list_keys();

                for (auto name : meas_names)
                {
		            double mval = meas_obj.get(name).get<double>();
                    this->_logger->info("{}={}", name, mval);
                    this->add_meas(name, mval);
                    this->add_meas_per_iter(name, mval);
                }
            }

            int64_t check_id = epoch + 1;
            if ((save_period != 0 && check_id % save_period == 0) || check_id == iterations)
            {
                this->_logger->info("Saving checkpoint after {} iterations", check_id);
                _module.save_checkpoint(this->_mlmodel, std::to_string(check_id));
            }
        }

        test(ad, inputc._test_dataset, test_batch_size, out);

        // TODO make model ready for predict after training

        inputc.response_params(out);
        this->_logger->info("Training done.");
        return 0;
    }

    template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
    int TorchLib<TInputConnectorStrategy, TOutputConnectorStrategy, TMLModel>::predict(const APIData &ad, APIData &out)
    {
        APIData params = ad.getobj("parameters");
        APIData output_params = params.getobj("output");

        TInputConnectorStrategy inputc(this->_inputc);
        TOutputConnectorStrategy outputc;
        try {
            inputc.transform(ad);
        } catch (...) {
            throw;
        }
        torch::Device cpu("cpu");

        if (output_params.has("measure"))
        {
            APIData meas_out;
            test(ad, inputc._dataset, 1, meas_out);
            meas_out.erase("iteration");
            out.add("measure", meas_out.getobj("measure"));
            return 0;
        }

        inputc._dataset.reset();
        std::vector<c10::IValue> in_vals;
        for (Tensor tensor : inputc._dataset.get_cached().data)
            in_vals.push_back(tensor.to(_device));
        Tensor output;
        try
        {
            output = torch::softmax(to_tensor_safe(_module.forward(in_vals)), 1);
        }
        catch (std::exception &e)
        {
            throw MLLibInternalException(std::string("Libtorch error:") + e.what());
        }
        
        // Output
        std::vector<APIData> results_ads;

        if (output_params.has("best"))
        {
            const int best_count = output_params.get("best").get<int>();
            std::tuple<Tensor, Tensor> sorted_output = output.sort(1, true);
            auto probs_acc = std::get<0>(sorted_output).accessor<float,2>();
            auto indices_acc = std::get<1>(sorted_output).accessor<int64_t,2>();

            for (int i = 0; i < output.size(0); ++i)
            {
                APIData results_ad;
                std::vector<double> probs;
                std::vector<std::string> cats;

                for (int j = 0; j < best_count; ++j)
                {
                    probs.push_back(probs_acc[i][j]);
                    int index = indices_acc[i][j];
                    cats.push_back(this->_mlmodel.get_hcorresp(index));
                }

                results_ad.add("uri", inputc._uris.at(results_ads.size()));
                results_ad.add("loss", 0.0);
                results_ad.add("cats", cats);
                results_ad.add("probs", probs);
                results_ad.add("nclasses", _nclasses);

                results_ads.push_back(results_ad);
            }
        }

        outputc.add_results(results_ads);
        outputc.finalize(output_params, out, static_cast<MLModel*>(&this->_mlmodel));

        out.add("status", 0);

        return 0;
    }

    template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
    int TorchLib<TInputConnectorStrategy, TOutputConnectorStrategy, TMLModel>::test(const APIData &ad, 
                                                                                    TorchDataset &dataset,
                                                                                    int batch_size,
                                                                                    APIData &out) 
    {
        APIData ad_res;
        APIData ad_out = ad.getobj("parameters").getobj("output");
        int test_size = dataset.cache_size();

        // <!> std::move may lead to unexpected behaviour from the input connector
        auto dataloader = torch::data::make_data_loader(
            std::move(dataset),
            data::DataLoaderOptions(batch_size)
        );

        int entry_id = 0;
        for (TorchBatch &batch : *dataloader)
        {
            std::vector<c10::IValue> in_vals;
            for (Tensor tensor : batch.data)
                in_vals.push_back(tensor.to(_device));
            Tensor output = torch::softmax(to_tensor_safe(_module.forward(in_vals)), 1);
            if (batch.target.empty())
                throw MLLibBadParamException("Missing label on data while testing");
            Tensor labels = batch.target[0];
            
            for (int j = 0; j < labels.size(0); ++j) {
                APIData bad;
                std::vector<double> predictions;
                for (int c = 0; c < _nclasses; c++)
                {
                    predictions.push_back(output[j][c].item<double>());
                }
                bad.add("target", labels[j].item<double>());
                bad.add("pred", predictions);
                ad_res.add(std::to_string(entry_id), bad);
                ++entry_id;
            }
            this->_logger->info("Testing: {}/{} entries processed", entry_id, test_size);
        }

        ad_res.add("iteration",this->get_meas("iteration"));
        ad_res.add("train_loss",this->get_meas("train_loss"));
        std::vector<std::string> clnames;
        for (int i=0;i<_nclasses;i++)
            clnames.push_back(this->_mlmodel.get_hcorresp(i));
        ad_res.add("clnames", clnames);
        ad_res.add("nclasses", _nclasses);
        ad_res.add("batch_size", entry_id); // here batch_size = tested entries count
        SupervisedOutput::measure(ad_res, ad_out, out);
        return 0;
    }

    template class TorchLib<ImgTorchInputFileConn,SupervisedOutput,TorchModel>;
    template class TorchLib<TxtTorchInputFileConn,SupervisedOutput,TorchModel>;
}
