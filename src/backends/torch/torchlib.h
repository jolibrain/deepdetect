/**
 * DeepDetect
 * Copyright (c) 2019-2020 Jolibrain
 * Authors: Louis Jean <ljean@etud.insa-toulouse.fr>
 *          Guillaume Infantes <guillaume.infantes@jolibrain.com>
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

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <torch/torch.h>
#include <torch/script.h>
#pragma GCC diagnostic pop

#include "apidata.h"
#include "mllibstrategy.h"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#include "caffe.pb.h"
#pragma GCC diagnostic pop

#include "torchmodel.h"
#include "torchinputconns.h"
#include "torchgraphbackend.h"
#include "native/native_net.h"

namespace dd
{
  /* autodoc: nccn init parameters */
  struct TorchLibInitParameters
  {
    /* network output blob name */
    std::string _template = nullptr;

    /* number of output classes */
    unsigned int nclasses = 0;

    /* classification mode */
    bool classification = false;

    /* finetuning */
    bool finetuning = false;

    /* gpu */
    bool gpu = false;

    /* gpuid */
    int gpuid = -1;

    /* device */
    torch::Device device = torch::Device("cpu");

    std::string self_supervised = "";

    bool freeze_traced = false;

    int embedding_size = 768;

    std::string loss = "";

    void post_init()
    {
      // FIXME(sileht): Maybe we should raise an error intead to
      // fallback to CPU
      if (gpu && !torch::cuda::is_available())
        gpu = false;

      device = gpu ? torch::Device(torch::DeviceType::CUDA, gpuid)
                   : torch::Device(torch::DeviceType::CPU);

      // FIXME(sileht): We should raise exception if the list of loss is wrong
      // here
    }
    void staticjson_init(staticjson::ObjectHandler *h)
    {
      h->add_property("nclasses", &nclasses, staticjson::Flags::Optional);
      h->add_property("template", &_template, staticjson::Flags::Optional);
      h->add_property("finetuning", &finetuning, staticjson::Flags::Optional);
      h->add_property("gpu", &gpu, staticjson::Flags::Optional);
      h->add_property("gpuid", &gpuid, staticjson::Flags::Optional);
      h->add_property("self_supervised", &self_supervised,
                      staticjson::Flags::Optional);
      h->add_property("freeze_traced", &freeze_traced,
                      staticjson::Flags::Optional);
      h->add_property("embedding_size", &embedding_size,
                      staticjson::Flags::Optional);
      h->add_property("loss", &loss, staticjson::Flags::Optional);

      h->set_flags(staticjson::Flags::DisallowUnknownKey);
    }
  };

  // TODO TorchModule may be merged with TorchGraph in the future
  class TorchModule
  {
  public:
    TorchModule();

    c10::IValue forward(std::vector<c10::IValue> source);
    c10::IValue extract(std::vector<c10::IValue> source,
                        std::string extract_layer);

    bool extractable(std::string extract_layer) const;
    std::vector<std::string> extractable_layers() const;

    void freeze_traced(bool freeze);

    /** Add linear model at the end of module. Automatically detects size of
     * the last layer thanks to the provided example output.*/
    void setup_classification(int nclasses,
                              std::vector<c10::IValue> input_example);

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

    template <class TInputConnectorStrategy>
    void post_transform(const std::string tmpl, const APIData &template_params,
                        const TInputConnectorStrategy &inputc,
                        const TorchModel &tmodel, const torch::Device &device);

    template <class TInputConnectorStrategy>
    void post_transform_train(const std::string tmpl,
                              const APIData &template_params,
                              const TInputConnectorStrategy &inputc,
                              const TorchModel &tmodel,
                              const torch::Device &device);

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
    std::shared_ptr<torch::jit::script::Module> _traced;
    std::shared_ptr<TorchGraphBackend>
        _graph; /**< graph module : torchgraphbackend has same interface as
                   torch::module */
    std::shared_ptr<NativeModule>
        _native; /**< native module : directly written in C++ */

    torch::nn::Linear _classif = nullptr;

    torch::Device _device;
    int _classif_in = 0; /**<id of the input of the classification layer */
    bool _hidden_states = false; /**< Take BERT hidden states as input. */

    bool _require_classif_layer = false;
    std::string
        _classif_layer_file; /** < if require_classif_layer == true, this is
                                the file where the weights are stored */
    std::shared_ptr<spdlog::logger> _logger; /**< mllib logger. */

  private:
    bool _freeze_traced = false; /**< Freeze weights of the traced module */
    void proto_model_load(const TorchModel &tmodel);
    void graph_model_load(const TorchModel &tmodel);
    void native_model_load(const TorchModel &tmodel);
    void classif_model_load(const TorchModel &tmodel);
    void traced_model_load(TorchModel &model);
    void classif_layer_load();
  };

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel = TorchModel>
  class TorchLib : public MLLib<TInputConnectorStrategy,
                                TOutputConnectorStrategy, TMLModel>
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
             TorchDataset &dataset, int batch_size, APIData &out);

  public:
    bool _timeserie = false;
    bool _masked_lm = false;
    bool _seq_training = false;
    bool _classification = false;

    TorchLibInitParameters _mllib_params;

    APIData _template_params;

    // models
    TorchModule _module;

    std::vector<std::string>
        _best_metrics;         /**< metric to use for saving best model */
    double _best_metric_value; /**< best metric value  */

  private:
    /**
     * \brief checks wether v1 is better than v2
     */
    bool is_better(double v1, double v2, std::string metric_name);

    /**
     * \brief generates a file containing best iteration so far
     */
    int64_t save_if_best(APIData &meas_out, int64_t elapsed_it,
                         torch::optim::Optimizer &optimizer,
                         int64_t best_to_remove);

    void snapshot(int64_t elapsed_it, torch::optim::Optimizer &optimizer);

    /**
     * \brief (re) load solver state
     */
    void solver_load(std::unique_ptr<torch::optim::Optimizer> &optimizer);

    void remove_model(int64_t it);

    double unscale(double val, unsigned int k,
                   const TInputConnectorStrategy &inputc);
  };
}

#endif // TORCHLIB_H
