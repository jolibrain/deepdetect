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
        _traced = std::move(tl._traced);
        _nclasses = tl._nclasses;
        _device = tl._device;
        _attention = tl._attention;
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
            _attention = true;
        }

	try
	  {
	    _traced = torch::jit::load(this->_mlmodel._model_file, _device);
	  }
	catch (std::exception&)
	  {
	    throw MLLibBadParamException("failed loading torch model file " + this->_mlmodel._model_file);
	  }
	
        _traced->eval();
    }

    template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
    void TorchLib<TInputConnectorStrategy, TOutputConnectorStrategy, TMLModel>::clear_mllib(const APIData &ad) 
    {
      (void)ad;
    }

    template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
    int TorchLib<TInputConnectorStrategy, TOutputConnectorStrategy, TMLModel>::train(const APIData &ad, APIData &out) 
    {
      (void)out;
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

        std::vector<c10::IValue> in_vals;
        in_vals.push_back(inputc._in.to(_device));

        if (_attention) {
            // token_type_ids
            in_vals.push_back(torch::zeros_like(inputc._in, at::kLong).to(_device));
            in_vals.push_back(inputc._attention_mask.to(_device));
        }

        Tensor output;
        try
        {
            c10::IValue out_val = _traced->forward(in_vals);
            if (out_val.isTuple()) {
                out_val = out_val.toTuple()->elements()[0];
            }
            if (!out_val.isTensor()) {
                throw MLLibInternalException("Model returned an invalid output. Please check your model.");
            }
            output = out_val.toTensor().to(at::kFloat);
        }
        catch (std::exception &e)
        {
            throw MLLibInternalException(std::string("Libtorch error:") + e.what());
        }
        output = torch::softmax(output, 1).to(cpu);
        
        // Output
        std::vector<APIData> results_ads;

	// classification
	std::tuple<Tensor, Tensor> sorted_output = output.sort(1, true);
	auto probs_acc = std::get<0>(sorted_output).accessor<float,2>();
	auto indices_acc = std::get<1>(sorted_output).accessor<int64_t,2>();

	for (int i = 0; i < output.size(0); ++i)
	  {
	    APIData results_ad;
	    std::vector<double> probs;
	    std::vector<std::string> cats;
	    
	    for (int j = 0; j < probs_acc.size(1); ++j)
	      {
		probs.push_back(probs_acc[i][j]);
		int index = indices_acc[i][j];
		cats.push_back(this->_mlmodel.get_hcorresp(index));
	      }
	    
	    results_ad.add("uri", inputc._uris.at(results_ads.size()));
	    results_ad.add("loss", 0.0);
	    results_ad.add("cats", cats);
	    results_ad.add("probs", probs);
	    
	    results_ads.push_back(results_ad);
	  }

	out.add("nclasses",static_cast<int>(probs_acc.size(1)));
        outputc.add_results(results_ads);
        outputc.finalize(output_params, out, static_cast<MLModel*>(&this->_mlmodel));

        out.add("status", 0);

        return 0;
    }


    template class TorchLib<ImgTorchInputFileConn,SupervisedOutput,TorchModel>;
    template class TorchLib<TxtTorchInputFileConn,SupervisedOutput,TorchModel>;
}
