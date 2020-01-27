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
#if !defined(CPU_ONLY)
#include <c10/cuda/CUDACachingAllocator.h>
#endif

#include "outputconnectorstrategy.h"

using namespace torch;

namespace dd
{
    inline void empty_cuda_cache() {
        #if !defined(CPU_ONLY)
        c10::cuda::CUDACachingAllocator::emptyCache();
        #endif
    }

    void add_parameters(std::shared_ptr<torch::jit::script::Module> module, std::vector<Tensor> &params, bool requires_grad = true) {
        for (const auto &tensor : module->parameters()) {
            if (tensor.requires_grad() && requires_grad)
                params.push_back(tensor);
        }
        for (auto child : module->children()) {
          add_parameters(std::make_shared<torch::jit::script::Module>(child), params);
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
                auto elems = output.toTensorList();
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
        if (_hidden_states)
        {
            // out_val is a tuple containing tensors of dimension n_batch * sequence_lenght * n_features
            // We want a tensor of size n_batch * n_features from the last hidden state
            auto &elems = out_val.toTuple()->elements();
            out_val = elems.back().toTensor().slice(1, 0, 1).squeeze(1);
        }
        if (_classif)
        {
            out_val = _classif->forward(to_tensor_safe(out_val));
        }
        return out_val;
    }

    void TorchModule::freeze_traced(bool freeze)
    {
        if (freeze != _freeze_traced)
        {
            _freeze_traced = freeze;
            std::vector<Tensor> params;
            add_parameters(_traced, params, false);
            for (auto &param : params)
            {
                param.set_requires_grad(!freeze);
            }
        }
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
          _traced = std::make_shared<torch::jit::script::Module>
            (torch::jit::load(model._traced, _device));
        if (!model._weights.empty() && _classif)
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

    void TorchModule::free()
    {
        _traced = nullptr;
        _classif = nullptr;
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
        _seq_training = tl._seq_training;
        _finetuning = tl._finetuning;
    }

    template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
    TorchLib<TInputConnectorStrategy, TOutputConnectorStrategy, TMLModel>::~TorchLib()
    {
        _module.free();
        empty_cuda_cache();
    }

    /*- from mllib -*/
    template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
    void TorchLib<TInputConnectorStrategy, TOutputConnectorStrategy, TMLModel>::init_mllib(const APIData &lib_ad)
    {
        bool classification = false;
        bool gpu = false;
        int gpuid = -1;
        bool freeze_traced = false;
        int embedding_size = 768;
        std::string self_supervised = "";

        if (lib_ad.has("template"))
            _template = lib_ad.get("template").get<std::string>();
        if (lib_ad.has("gpu"))
            gpu = lib_ad.get("gpu").get<bool>() && torch::cuda::is_available();
        if (lib_ad.has("gpuid"))
            gpuid = lib_ad.get("gpuid").get<int>();
        if (lib_ad.has("nclasses"))
        {
            classification = true;
            _nclasses = lib_ad.get("nclasses").get<int>();
        }
        if (lib_ad.has("self_supervised"))
            self_supervised = lib_ad.get("self_supervised").get<std::string>();
        if (lib_ad.has("embedding_size"))
            embedding_size = lib_ad.get("embedding_size").get<int>();
        if (lib_ad.has("finetuning"))
            _finetuning = lib_ad.get("finetuning").get<bool>();
        if (lib_ad.has("freeze_traced"))
            freeze_traced = lib_ad.get("freeze_traced").get<bool>();

        _device = gpu ? torch::Device(DeviceType::CUDA, gpuid) : torch::Device(DeviceType::CPU);
        _module._device = _device;

        // Create the model
        if (this->_mlmodel._traced.empty())
            throw MLLibInternalException("Use of libtorch backend without traced net is not supported yet");

        this->_inputc._input_format = "bert";
        if (_template == "bert")
        {
            if (classification)
            {
                _module._classif = nn::Linear(embedding_size, _nclasses);
                _module._classif->to(_device);
                _module._hidden_states = true;
                _module._classif_in = 1;
            }
            else if (!self_supervised.empty())
            {
                if (self_supervised != "mask")
                {
                    throw MLLibBadParamException("self_supervised");
                }
                this->_logger->info("Masked Language model");
                _masked_lm = true;
                _seq_training = true;
            }
            else
            {
                throw MLLibBadParamException("BERT only supports self-supervised or classification");
            }
        }
        else if (_template == "gpt2")
        {
            this->_inputc._input_format = "gpt2";
            _seq_training = true;
        }
        else if (!_template.empty())
        {
            throw MLLibBadParamException("template");
        }

        this->_logger->info("Loading ml model from file {}.", this->_mlmodel._traced);
        if (!this->_mlmodel._weights.empty())
            this->_logger->info("Loading weights from file {}.", this->_mlmodel._weights);
        _module.load(this->_mlmodel);
        _module.freeze_traced(freeze_traced);

        this->_mltype = "classification";
    }

    template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
    void TorchLib<TInputConnectorStrategy, TOutputConnectorStrategy, TMLModel>::clear_mllib(const APIData &ad)
    {
        std::vector<std::string> extensions{".json", ".pt", ".ptw"};
        fileops::remove_directory_files(this->_mlmodel._repo, extensions);
        this->_logger->info("Torchlib service cleared");
    }

    template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
    int TorchLib<TInputConnectorStrategy, TOutputConnectorStrategy, TMLModel>::train(const APIData &ad, APIData &out)
    {
        this->_tjob_running.store(true);

        TInputConnectorStrategy inputc(this->_inputc);
        inputc._train = true;
        inputc._finetuning = _finetuning;

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
        int64_t iterations = 1;
        std::string solver_type = "SGD";
        double base_lr = 0.0001;
        int64_t batch_size = 1;
        int64_t iter_size = 1;
        int64_t test_batch_size = 1;
        int64_t test_interval = 1;
        int64_t save_period = 0;

        // logging parameters
        int64_t log_batch_period = 20;

        if (ad_mllib.has("solver"))
        {
            APIData ad_solver = ad_mllib.getobj("solver");
            if (ad_solver.has("iterations"))
                iterations = ad_solver.get("iterations").get<int>();
            if (ad_solver.has("solver_type"))
                solver_type = ad_solver.get("solver_type").get<std::string>();
            if (ad_solver.has("base_lr"))
                base_lr = ad_solver.get("base_lr").get<double>();
            if (ad_solver.has("test_interval"))
                test_interval = ad_solver.get("test_interval").get<int>();
            if (ad_solver.has("iter_size"))
                iter_size = ad_solver.get("iter_size").get<int>();
            if (ad_solver.has("snapshot"))
                save_period = ad_solver.get("snapshot").get<int>();
        }

        if (ad_mllib.has("net"))
        {
            APIData ad_net = ad_mllib.getobj("net");
            if (ad_net.has("batch_size"))
                batch_size = ad_net.get("batch_size").get<int>();
            if (ad_net.has("test_batch_size"))
                test_batch_size = ad_net.get("test_batch_size").get<int>();
        }

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

        // it is the iteration count (not epoch)
        while (it < iterations)
        {
            if (!this->_tjob_running.load())
            {
                break;
            }

            double train_loss = 0;
            double avg_it_time = 0;

            for (TorchBatch batch : *dataloader)
            {
                auto tstart = system_clock::now();
                if (_masked_lm)
                {
                    batch = inputc.generate_masked_lm_batch(batch);
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

                // As CrossEntropy is not available (Libtorch 1.1) we use nllloss + log_softmax
                Tensor loss;
                if (_seq_training)
                {
                    // Convert [n_batch, sequence_length, vocab_size] to [n_batch * sequence_length, vocab_size]
                    // + ignore non-masked tokens (== -1 in target)
                    loss = torch::nll_loss(
                        torch::log_softmax(y_pred.view(IntList{-1, y_pred.size(2)}), 1),
                        y.view(IntList{-1}),
                        {}, Reduction::Mean, -1
                    );
                }
                else
                {
                    loss = torch::nll_loss(torch::log_softmax(y_pred, 1), y.view(IntList{-1}));
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
                    if (!this->_tjob_running.load())
                    {
                        break;
                    }
                    optimizer->step();
                    optimizer->zero_grad();
                    avg_it_time /= iter_size;
                    this->add_meas("learning_rate", base_lr);
                    this->add_meas("iteration", it);
                    this->add_meas("iter_time", avg_it_time);
                    this->add_meas("remain_time", avg_it_time * iter_size * (iterations - it) / 1000.0);
                    this->add_meas("train_loss", train_loss);
                    this->add_meas_per_iter("learning_rate", base_lr);
                    this->add_meas_per_iter("train_loss", train_loss);
                    int64_t elapsed_it = it + 1;
                    if (log_batch_period != 0 && elapsed_it % log_batch_period == 0)
                    {
                        this->_logger->info("Iteration {}/{}: loss is {}", elapsed_it, iterations, train_loss);
                    }
                    avg_it_time = 0;
                    train_loss = 0;

                    if (elapsed_it % test_interval == 0 && elapsed_it != iterations && !eval_dataset.empty())
                    {
                        // Free memory
                        loss = torch::empty(1);
                        y_pred = torch::empty(1);
                        y = torch::empty(1);
                        in_vals.clear();

                        APIData meas_out;
                        this->_logger->info("Start test");
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
                            else if (name == "cmdiag")
                            {
                                std::vector<double> mdiag = meas_obj.get(name).get<std::vector<double>>();
                                std::vector<std::string> cnames;
                                std::string mdiag_str;
                                for (size_t i=0; i<mdiag.size(); i++)
                                {
                                    mdiag_str += this->_mlmodel.get_hcorresp(i) + ":" + std::to_string(mdiag.at(i)) + " ";
                                    this->add_meas_per_iter(name+'_'+this->_mlmodel.get_hcorresp(i), mdiag.at(i));
                                    cnames.push_back(this->_mlmodel.get_hcorresp(i));
                                }
                                this->_logger->info("{}=[{}]", name, mdiag_str);
                                this->add_meas(name, mdiag, cnames);
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

        if (!this->_tjob_running.load())
        {
            this->_logger->info("Training job interrupted at iteration {}", it);
            empty_cuda_cache();
            return -1;
        }

        test(ad, inputc, inputc._test_dataset, test_batch_size, out);
        empty_cuda_cache();

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
	        meas_out.erase("train_loss");
            out.add("measure", meas_out.getobj("measure"));
            empty_cuda_cache();
            return 0;
        }

        inputc._dataset.reset();
        TorchBatch batch = inputc._dataset.get_cached();

        std::vector<c10::IValue> in_vals;
        for (Tensor tensor : batch.data)
            in_vals.push_back(tensor.to(_device));
        Tensor output;
        try
        {
            output = to_tensor_safe(_module.forward(in_vals));
            if (_template == "gpt2")
            {
                // Keep only the prediction for the last token
                Tensor input_ids = batch.data[0];
                std::vector<Tensor> outputs;
                for (int i = 0; i < input_ids.size(0); ++i)
                {
                    // output is (n_batch * sequence_length * vocab_size)
                    // With gpt2, last token is endoftext so we need to take the previous output.
                    outputs.push_back(output[i][inputc._lengths.at(i) - 2]);
                }
                output = torch::stack(outputs);
            }
            output = torch::softmax(output, 1).to(cpu);
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
                    if (_seq_training)
                    {
                        cats.push_back(inputc.get_word(index));
                    }
                    else
                    {
                        cats.push_back(this->_mlmodel.get_hcorresp(index));
                    }
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
        for (TorchBatch batch : *dataloader)
        {
            if (_masked_lm)
            {
                batch = inputc.generate_masked_lm_batch(batch);
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
            Tensor labels = batch.target[0].view(IntList{-1});
            if (_masked_lm)
            {
                // Convert [n_batch, sequence_length, vocab_size] to [n_batch * sequence_length, vocab_size]
                output = output.view(IntList{-1, output.size(2)});
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
