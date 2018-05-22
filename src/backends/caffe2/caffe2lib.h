/**
 * DeepDetect
 * Copyright (c) 2018 Jolibrain
 * Author: Julien Chicha
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

#ifndef CAFFE2LIB_H
#define CAFFE2LIB_H

//XXX Remove that to print the warnings
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <caffe2/core/workspace.h>
#pragma GCC diagnostic pop

//XXX Remove that to print the warnings
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
#include "mllibstrategy.h"
#pragma GCC diagnostic pop

#include "backends/caffe2/caffe2libstate.h"
#include "backends/caffe2/caffe2model.h"

namespace dd {

  /**
   * \brief Caffe2 library wrapper for deepdetect
   */
  template <
    class TInputConnectorStrategy,
    class TOutputConnectorStrategy,
    class TMLModel=Caffe2Model>
  class Caffe2Lib : public MLLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel> {

  public: // from mllib

    /**
     * \brief constructor from model
     * @param model Caffe2 model
     */
    Caffe2Lib(const Caffe2Model &c2model);

    /**
     * \brief copy-constructor
     */
    Caffe2Lib(Caffe2Lib &&cl) noexcept;

    /**
     * \brief destructor
     */
    ~Caffe2Lib();

    void init_mllib(const APIData &ad);
    void clear_mllib(const APIData &ad);

    /**
     * \brief trains a model
     * @param ad root data object
     * @param out output data object (e.g. loss, ...)
     * @return 0 if OK, 1 otherwise
     */
    int train(const APIData &ad, APIData &out);

    /**
     * \brief predicts from model
     * @param ad root data object
     * @param out output data object (e.g. predictions, ...)
     * @return 0 if OK, 1 otherwise
     */
    int predict(const APIData &ad, APIData &out);

  private:

    /**
     * \brief create nets and update the repository
     * @param ad mllib data object
     */
    void instantiate_template(const APIData &ad);

    /**
     * \brief creates neural net instance based on model
     */
    void create_model();

    /**
     * \brief tests a model and compute measures
     * @param ad root data object
     * @param out output data object
     */
    void test(const std::string &net, const APIData &ad,
	      TInputConnectorStrategy &inputc, APIData &out);

    /**
     * \brief setups the nets, inputs, outputs, gradients, etc.
     */
    void create_model_train();
    void create_model_predict();

    /**
     * \brief dumps on the filesystem usefull information to resume training
     */
    void dump_model_state();

    /**
     * \brief runs a net once (both forward and backward if the gradients are set)
     * @param net net too run
     * @param where to store the output layer (one vector per batch item)
     *        The vector must be of the right size. If set to NULL, nothing is stored.
     *        XXX Only retrieve the output of the first device
     *	          (currently predictions and tests are done on a single device)
     * @return the elapsed time
     */
    float run_net(const std::string &net, std::vector<std::vector<float>> *results = NULL);

    /**
     * \brief inserts a batch in th workspace
     *        XXX The batch is inserted only on the first device
     *            (currently predictions and tests are done on a single device)
     * @param inputcs input connector that will generate the batch
     * @param batch_size maximum size of the batch
     * @return actual size of the batch ( <= batch_size )
     */
    int load_batch(TInputConnectorStrategy &inputc, int batch_size = -1);

    // Workspaces cannot be std::move()'d or assigned
    // (see DISABLE_COPY_AND_ASSIGN in caffe2/core/workspace.h)
    // Hence the usage of a pointer.
    std::unique_ptr<caffe2::Workspace> _workspace =
      std::unique_ptr<caffe2::Workspace>(new caffe2::Workspace);

    std::vector<caffe2::DeviceOption> _devices;
    caffe2::NetDef _init_net;
    caffe2::NetDef _test_net;
    caffe2::NetDef _net;
    Caffe2LibState _state;

    std::string _input_blob;
    std::string _output_blob;
    int _nclasses = 0; //XXX Infer it during the 'create_model' phase
  };
}

#endif
