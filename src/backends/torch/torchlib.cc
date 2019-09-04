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


    TorchModule::TorchModule() : _device{"cpu"} {}

    c10::IValue TorchModule::forward(std::vector<c10::IValue> source) 
    {
        if (_traced)
        {
            auto output = _traced->forward(source);
            if (output.isTensorList()) {
                auto &elems = output.toTensorList()->elements();
                source = std::vector<c10::IValue>(elems.begin(), elems.end());
            }
            else if (output.isTuple()) {
                auto &elems = output.toTuple()->elements();
                source = std::vector<c10::IValue>(elems.begin(), elems.end());
            }
            else {
                source = { output };
            }
        }
        c10::IValue out_val = source.at(_classif_in);
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
        if (_classif)
            torch::save(_classif, model._repo + "/checkpoint-" + name + ".ptw");
    }

    void TorchModule::load(TorchModel &model) 
    {
        if (!model._traced.empty())
            _traced = torch::jit::load(model._traced, _device);
        if (!model._weights.empty())
            torch::load(_classif, model._weights);
    }

    void TorchModule::eval() {
        if (_traced)
            _traced->eval();
        if (_classif)
            _classif->eval();
    }

    void TorchModule::train() {
        if (_traced)
            _traced->train();
        if (_classif)
            _classif->train();
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
        _template = tl._template;
        _nclasses = tl._nclasses;
        _device = tl._device;
        _masked_lm = tl._masked_lm;
    }

    template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
    TorchLib<TInputConnectorStrategy, TOutputConnectorStrategy, TMLModel>::~TorchLib() 
    {
        
    }

    /*- from mllib -*/
    template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
    void TorchLib<TInputConnectorStrategy, TOutputConnectorStrategy, TMLModel>::init_mllib(const APIData &lib_ad)
    {
        bool classification = true;
        bool finetuning = false;
        bool gpu = false;
        int gpuid = -1;

        if (lib_ad.has("gpu"))
            gpu = lib_ad.get("gpu").get<bool>() && torch::cuda::is_available();
        if (lib_ad.has("gpuid"))
            gpuid = lib_ad.get("gpuid").get<int>();
        if (lib_ad.has("nclasses"))
        {
            classification = true;
            _nclasses = lib_ad.get("nclasses").get<int>();
        }
        if (lib_ad.has("template"))
            _template = lib_ad.get("template").get<std::string>();
        if (lib_ad.has("finetuning"))
            finetuning = lib_ad.get("finetuning").get<bool>();

        _device = gpu ? torch::Device(DeviceType::CUDA, gpuid) : torch::Device(DeviceType::CPU);
        _module._device = _device;

        // Create the model
        if (this->_mlmodel._traced.empty())
            throw MLLibInternalException("This template requires a traced net");

        if (_masked_lm)
        {
            _module._classif_in = 0;
        }
        else
        {
            if (_template == "bert-classification")
            {
                if (!classification)
                    throw MLLibBadParamException("nclasses not specified");

                // XXX: dont hard code BERT output size
                _module._classif = nn::Linear(768, _nclasses);
                _module._classif->to(_device);

                _module._classif_in = 1;
            }
        }
        
        this->_logger->info("Loading ml model from file {}.", this->_mlmodel._traced);
        this->_logger->info("Loading weights from file {}.", this->_mlmodel._weights);
        _module.load(this->_mlmodel);

        this->_mltype = "classification";
    }

    template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
    void TorchLib<TInputConnectorStrategy, TOutputConnectorStrategy, TMLModel>::clear_mllib(const APIData &ad) 
    {
        std::vector<std::string> extensions{".json"};
        fileops::remove_directory_files(this->_mlmodel._repo, extensions);
        this->_logger->info("Torchlib service cleared");
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
        int64_t iter_size = 1;
        int64_t test_batch_size = 1;
        int64_t test_interval = 1;
        int64_t save_period = 0;

        // logging parameters
        int64_t log_batch_period = 20;

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
        if (ad_mllib.has("iter_size"))
            iter_size = ad_mllib.get("iter_size").get<int>();
        if (ad_mllib.has("test_batch_size"))
            test_batch_size = ad_mllib.get("test_batch_size").get<int>();
        if (ad_mllib.has("save_period"))
            save_period = ad_mllib.get("save_period").get<int>();

        if (iter_size <= 0)
            iter_size = 1;

        // create dataset for evaluation during training
        TorchDataset eval_dataset;
        if (!inputc._test_dataset.empty())
        {
            eval_dataset = inputc._test_dataset; //.split(0, 0.1);
        }

        // create solver
        std::unique_ptr<optim::Optimizer> optimizer;
        
        if (solver_type == "ADAM")
            optimizer = std::unique_ptr<optim::Optimizer>(
                new optim::Adam(_module.parameters(), optim::AdamOptions(base_lr)));
        else if (solver_type == "RMSPROP")
            optimizer = std::unique_ptr<optim::Optimizer>(
                new optim::RMSprop(_module.parameters(), optim::RMSpropOptions(base_lr)));
        else if (solver_type == "ADAGRAD")
            optimizer = std::unique_ptr<optim::Optimizer>(
                new optim::Adagrad(_module.parameters(), optim::AdagradOptions(base_lr)));
        else
        {
            if (solver_type != "SGD")
                this->_logger->warn("Solver type {} not found, using SGD", solver_type);
            optimizer = std::unique_ptr<optim::Optimizer>(
                new optim::SGD(_module.parameters(), optim::SGDOptions(base_lr)));
        }
        // reload solver
        if (!this->_mlmodel._sstate.empty())
        {
            this->_logger->info("Reload solver from {}", this->_mlmodel._sstate);
            torch::load(*optimizer, this->_mlmodel._sstate);
        }
        optimizer->zero_grad();
        _module.train();

        // create dataloader
        auto dataloader = torch::data::make_data_loader(
            std::move(inputc._dataset),
            data::DataLoaderOptions(batch_size)
        );

        this->_logger->info("Training for {} iterations", iterations);
        int it = 0;
        int batch_id = 0;
        using namespace std::chrono;

        while (it < iterations)
        {
            double train_loss = 0;
            double avg_it_time = 0;

            for (TorchBatch &example : *dataloader)
            {
                auto tstart = system_clock::now();
                TorchBatch batch;
                if (_masked_lm)
                {
                    batch = inputc.generate_masked_lm_batch(example);
                }
                else
                {
                    batch = example;
                    batch.target.at(0) = to_one_hot(batch.target.at(0), _nclasses);
                }
                std::vector<c10::IValue> in_vals;
                for (Tensor tensor : batch.data)
                    in_vals.push_back(tensor.to(_device));
                Tensor y = batch.target.at(0).to(_device);

                Tensor y_pred;
                try
                {
                    y_pred = to_tensor_safe(_module.forward(in_vals));
                }
                catch (std::exception &e)
                {
                    throw MLLibInternalException(std::string("Libtorch error:") + e.what());
                }

                Tensor loss;
                if (_masked_lm)
                {
                    loss = torch::nll_loss(
                        torch::log_softmax(y_pred.view(IntList{-1, y_pred.size(2)}), 1),
                        y.view(IntList{-1}),
                        {}, Reduction::Mean, -1
                    );
                }
                else
                {
                    // TODO: Better choice for the loss
                    loss = torch::mse_loss(y_pred, y);
                }
                if (iter_size > 1)
                    loss /= iter_size;

                double loss_val = loss.item<double>();
                train_loss += loss_val;
                loss.backward();
                auto tstop = system_clock::now();
                avg_it_time += duration_cast<milliseconds>(tstop - tstart).count();

                if ((batch_id + 1) % iter_size == 0)
                {
                    optimizer->step();
                    optimizer->zero_grad();
                    avg_it_time /= iter_size;
                    this->add_meas("iteration", it);
                    this->add_meas("iter_time", avg_it_time);
                    this->add_meas("remain_time", avg_it_time * iter_size * (iterations - it) / 1000.0);
                    this->add_meas("train_loss", train_loss);
                    this->add_meas_per_iter("train_loss", train_loss);
                    int64_t elapsed_it = it + 1;
                    if (log_batch_period != 0 && elapsed_it % log_batch_period == 0)
                    {
                        this->_logger->info("Iteration {}/{}: loss is {}", elapsed_it, iterations, train_loss);
                    }
                    train_loss = 0;

                    if (elapsed_it % test_interval == 0 && elapsed_it != iterations && !eval_dataset.empty())
                    {
                        APIData meas_out;
                        test(ad, inputc, eval_dataset, test_batch_size, meas_out);
                        APIData meas_obj = meas_out.getobj("measure");
                        std::vector<std::string> meas_names = meas_obj.list_keys();

                        for (auto name : meas_names)
                        {
                            if (name != "cmdiag" && name != "cmfull" && name != "labels")
                            {
                                double mval = meas_obj.get(name).get<double>();
                                this->_logger->info("{}={}", name, mval);
                                this->add_meas(name, mval);
                                this->add_meas_per_iter(name, mval);
                            }
                        }
                    }

                    if ((save_period != 0 && elapsed_it % save_period == 0) || elapsed_it == iterations)
                    {
                        this->_logger->info("Saving checkpoint after {} iterations", elapsed_it);
                        _module.save_checkpoint(this->_mlmodel, std::to_string(elapsed_it));
                        // Save optimizer
                        torch::save(*optimizer, this->_mlmodel._repo + "/solver-" + std::to_string(elapsed_it) + ".pt");
                    }
                    ++it;
                    
                    if (it >= iterations)
                        break;
                }

                ++batch_id;
            }
        }

        test(ad, inputc, inputc._test_dataset, test_batch_size, out);

        // Update model after training
        this->_mlmodel.read_from_repository(this->_logger);
        this->_mlmodel.read_corresp_file();

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

        _module.eval();

        if (output_params.has("measure"))
        {
            APIData meas_out;
            test(ad, inputc, inputc._dataset, 1, meas_out);
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
                                                                                    TInputConnectorStrategy &inputc, 
                                                                                    TorchDataset &dataset,
                                                                                    int batch_size,
                                                                                    APIData &out) 
    {
        APIData ad_res;
        APIData ad_out = ad.getobj("parameters").getobj("output");
        int test_size = dataset.cache_size();
        int nclasses = _masked_lm ? inputc.vocab_size() : _nclasses;

        // confusion matrix is irrelevant to masked_lm training
        if (_masked_lm && ad_out.has("measure"))
        {
            auto meas = ad_out.get("measure").get<std::vector<std::string>>();
            std::vector<std::string>::iterator it;
            if ((it = std::find(meas.begin(), meas.end(), "cmfull")) != meas.end())
                meas.erase(it);
            if ((it = std::find(meas.begin(), meas.end(), "cmdiag")) != meas.end())
                meas.erase(it);
            ad_out.add("measure", meas);
        }

        auto dataloader = torch::data::make_data_loader(
            dataset,
            data::DataLoaderOptions(batch_size)
        );
        torch::Device cpu("cpu");

        _module.eval();
        int entry_id = 0;
        for (TorchBatch &example : *dataloader)
        {
            TorchBatch batch;
            if (_masked_lm)
            {
                batch = inputc.generate_masked_lm_batch(example);
            }
            else
            {
                batch = example;
            }
            std::vector<c10::IValue> in_vals;
            for (Tensor tensor : batch.data)
                in_vals.push_back(tensor.to(_device));
            
            Tensor output;
            try
            {
                output = to_tensor_safe(_module.forward(in_vals));
            }
            catch (std::exception &e)
            {
                throw MLLibInternalException(std::string("Libtorch error:") + e.what());
            }

            if (batch.target.empty())
                throw MLLibBadParamException("Missing label on data while testing");
            Tensor labels = batch.target[0];

            if (_masked_lm)
            {
                output = output.view(IntList{-1, output.size(2)});
                labels = labels.view(IntList{-1});
            }
            output = torch::softmax(output, 1).to(cpu);
            auto output_acc = output.accessor<float,2>();
            auto labels_acc = labels.accessor<int64_t,1>();

            for (int j = 0; j < labels.size(0); ++j)
            {
                if (_masked_lm && labels_acc[j] == -1)
                    continue;

                APIData bad;
                std::vector<double> predictions;
                for (int c = 0; c < nclasses; c++)
                {
                    predictions.push_back(output_acc[j][c]);
                }
                bad.add("target", static_cast<double>(labels_acc[j]));
                bad.add("pred", predictions);
                // auto &tinputc = reinterpret_cast<TxtTorchInputConnector&>(inputc);
                /* 
                std::cout << "target: " << labels_acc[j] << std::endl;
                std::cout << "masking: " << batch.data[0][0][j] << " " << batch.data[2][0][j] << std::endl;
                std::cout << "pred: " << std::distance(predictions.begin(), std::max_element(predictions.begin(), predictions.end())) << std::endl << std::endl;
                */
                ad_res.add(std::to_string(entry_id), bad);
                ++entry_id;
            }
            // this->_logger->info("Testing: {}/{} entries processed", entry_id, test_size);
        }

        ad_res.add("iteration",this->get_meas("iteration"));
        ad_res.add("train_loss",this->get_meas("train_loss"));
        std::vector<std::string> clnames;
        for (int i=0;i< nclasses;i++)
            clnames.push_back(this->_mlmodel.get_hcorresp(i));
        ad_res.add("clnames", clnames);
        ad_res.add("nclasses", nclasses);
        ad_res.add("batch_size", entry_id); // here batch_size = tested entries count
        SupervisedOutput::measure(ad_res, ad_out, out);
        return 0;
    }

    template class TorchLib<ImgTorchInputFileConn,SupervisedOutput,TorchModel>;
    template class TorchLib<TxtTorchInputFileConn,SupervisedOutput,TorchModel>;
}
