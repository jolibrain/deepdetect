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
#include "torchmodule.h"
#include "torchsolver.h"

namespace dd
{

  /**
   * \brief core torchlib mllib wrapper
   */
  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel = TorchModel>
  class TorchLib : public MLLib<TInputConnectorStrategy,
                                TOutputConnectorStrategy, TMLModel>
  {
  public:
    /**
     * constructor from a torchmodel
     */
    TorchLib(const TorchModel &tmodel);

    /**
     * move constructor
     */
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
    unsigned int _nclasses = 0; /**< number of classes*/
    std::string _template; /**< template identifier (recurrent/bert/gpt2...)*/
    bool _finetuning = false; /**< add a custom layer (classif/regression...)
                                 after model */
    torch::Device _main_device
        = torch::Device("cpu"); /**<  the main device where the module lives */
    std::vector<torch::Device>
        _devices; /**< all the devices used during training (e.g. for multi-gpu
                     training) */
    bool _masked_lm = false;    /**< enable MLM self supervised pre training*/
    bool _seq_training = false; /**< true for bert/gpt2*/
    bool _classification = false; /**< select classification type problem*/
    bool _timeserie = false;      /**< select timeserie type problem*/
    std::string _loss = "";       /**< selected loss*/

    APIData _template_params; /**< template parameters, for recurrent and
                                 native models*/

    TorchModule _module; /**< wrapper around different underlyng
                            implementations (traced/native/graph...)*/

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
                         TorchSolver &tsolver, int64_t best_to_remove);

    /**
     * snapshop current optimizer state
     */
    void snapshot(int64_t elapsed_it, TorchSolver &optimizer);

    /**
     * delete superseeded model
     */
    void remove_model(int64_t it);

    /**
     * unscale value output from model, explicitely needed at prediction time
     */
    double unscale(double val, unsigned int k,
                   const TInputConnectorStrategy &inputc);
  };
}

#endif // TORCHLIB_H
