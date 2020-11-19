/**
 * DeepDetect
 * Copyright (c) 2019-2020 Jolibrain
 * Author:  Guillaume Infantes <guillaume.infantes@jolibrain.com>
 *          Louis Jean <ljean@etud.insa-toulouse.fr>
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

#ifndef TORCH_MODULE_H
#define TORCH_MODULE_H

// TODO this should normally be defined by pytorch
#define AT_PARALLEL_OPENMP 1

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <torch/torch.h>
#pragma GCC diagnostic pop
#include "torchmodel.h"
#include "torchgraphbackend.h"
#include "native/native_net.h"
#include <torch/script.h>
#include <torch/nn/pimpl.h>
#include <torch/nn/parallel/data_parallel.h>

namespace dd
{

  /**
   * \brief wrapper  above different torch implementations :
   * graph/native/traced ...
   */
  class TorchModule
  {
  public:
    TorchModule();

    /**
     * \brief forward (inference) pass over the network
     */
    c10::IValue forward(std::vector<c10::IValue> source);

    /**
     * \brief forward pass using the devices given in parameters.
     * If more than one device is given, this method performs a multigpu
     * forward.
     */
    c10::IValue forward_on_devices(std::vector<c10::IValue> source,
                                   const std::vector<torch::Device> &devices);

    /**
     * \brief forward (inference) until extract_layer, return value of
     * layer/blob
     */
    c10::IValue extract(std::vector<c10::IValue> source,
                        std::string extract_layer);

    /**
     * \brief checks if extract_layer is extractible
     */
    bool extractable(std::string extract_layer) const;

    /**
     * \brief gives all extractible layers
     */
    std::vector<std::string> extractable_layers() const;

    /**
     * \brief freeze traced net so that it is not updated during learning
     */
    void freeze_traced(bool freeze);

    /**
     * \brief Add linear model at the end of module. Automatically detects size
     * of the last layer thanks to the provided example output.
     */
    void setup_linear_layer(int nclasses,
                            std::vector<c10::IValue> input_example);

    /**
     * \brief gives all learnable parameters
     */
    std::vector<torch::Tensor> parameters();

    /**
     * \brief Save traced module to checkpoint-[name].pt, and custom parts
     * weights to checkpoint-[name].ptw (Actually only _classif is saved in the
     * .ptw)
     */
    void save_checkpoint(TorchModel &model, const std::string &name);

    /**
     * \brief Load traced module from .pt and custom parts weights from .ptw
     */
    void load(TorchModel &model);

    /**
     * \brief set net to be in eval mode (ie disable dropout ...)
     */
    void eval();

    /**
     * \brief set net to be in train mode (ie enable dropout ...)
     */
    void train();

    /**
     * \brief release all shared_ptr
     */
    void free();

    /**
     * \brief generic part of hooks below
     */
    template <class TInputConnectorStrategy>
    void create_native_template(const std::string &tmpl, const APIData &lib_ad,
                                const TInputConnectorStrategy &inputc,
                                const TorchModel &tmodel,
                                const torch::Device &device);

    template <class TInputConnectorStrategy>
    void post_transform(const std::string tmpl, const APIData &template_params,
                        const TInputConnectorStrategy &inputc,
                        const TorchModel &tmodel, const torch::Device &device);

    /**
     * \brief hook called after inputConnector::transform(data) during training
     */
    template <class TInputConnectorStrategy>
    void post_transform_train(const std::string tmpl,
                              const APIData &template_params,
                              const TInputConnectorStrategy &inputc,
                              const TorchModel &tmodel,
                              const torch::Device &device);

    /**
     * \brief hook called after inputConnector::transform(data) during predict
     */
    template <class TInputConnectorStrategy>
    void post_transform_predict(const std::string tmpl,
                                const APIData &template_params,
                                const TInputConnectorStrategy &inputc,
                                const TorchModel &tmodel,
                                const torch::Device &device,
                                const APIData &ad);

    /**
     * \brief see torch::module::to
     * @param device cpu / gpu
     * @param non_blocking
     */
    void to(torch::Device device);

    /**
     * \brief see torch::module::to
     * @param dtype : torch::kFloat32 or torch::kFloat64
     * @param non_blocking
     */
    void to(torch::Dtype dtype);

    /**
     * \brief see torch::module::to
     * @param device cpu / gpu
     * @param dtype : torch::kFloat32 or torch::kFloat64
     * @param non_blocking
     */
    void to(torch::Device device, torch::Dtype dtype);

  public:
    std::shared_ptr<torch::jit::script::Module>
        _traced; /**< traced (torchscript) module, if any */
    std::shared_ptr<TorchGraphBackend>
        _graph; /**< graph module : torchgraphbackend has same interface as
                   torch::module */
    std::shared_ptr<NativeModule>
        _native; /**< native module : directly written in C++ */

    torch::nn::Linear _linear = nullptr;

    torch::Device _device;
    int _linear_in = 0; /**<id of the input of the final linear layer */
    bool _hidden_states = false; /**< Take BERT hidden states as input. */

    bool _require_linear_layer = false;
    std::string
        _linear_layer_file;     /** < if require_linear_layer == true, this is
                                    the file where the weights are stored */
    unsigned int _nclasses = 0; /**< number of classes */

    std::shared_ptr<spdlog::logger> _logger; /**< mllib logger. */

  private:
    bool _freeze_traced = false; /**< Freeze weights of the traced module */

    /**
     * load graph module from caffe prototxt definition
     */
    void proto_model_load(const TorchModel &tmodel);

    /**
     * load graph module weight from pt format
     */
    void graph_model_load(const TorchModel &tmodel);

    /**
     * load native module weight from pt format
     */
    void native_model_load(const TorchModel &tmodel);

    /**
     * load linear model weights from pt format
     */
    void linear_model_load(const TorchModel &tmodel);

    /**
     * load traced net (def + weights) from  pt format
     */
    void traced_model_load(TorchModel &model);

    /**
     * load linear layer weights only from pt format
     */
    void linear_layer_load();
  };
}
#endif
