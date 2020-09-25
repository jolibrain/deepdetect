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

// XXX Remove that to print the warnings
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <caffe2/core/workspace.h>
#pragma GCC diagnostic pop

// XXX Remove that to print the warnings
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
#include "mllibstrategy.h"
#pragma GCC diagnostic pop

#include "backends/caffe2/caffe2libstate.h"
#include "backends/caffe2/caffe2model.h"

namespace dd
{

  /**
   * \brief Caffe2 library wrapper for deepdetect
   */
  template <class TInputConnectorStrategy, class TOutputConnectorStrategy,
            class TMLModel = Caffe2Model>
  class Caffe2Lib : public MLLib<TInputConnectorStrategy,
                                 TOutputConnectorStrategy, TMLModel>
  {

  public: // from mllib
    /**
     * \brief constructor from model
     * @param model Caffe2 model
     */
    Caffe2Lib(const Caffe2Model &c2model);

    /**
     * \brief move-constructor
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
     * \brief sets (/resets) internal values to fit the matching mode
     * @param ad APIData of the current request
     * @param train true means train and false mean predict
     */
    void set_train_mode(const APIData &ad, bool train);

    /**
     * \brief update the gpu configuration from the APIData
     * @param ad mllib APIData
     * @param dft true to set thoses values as the default ones
     */
    void set_gpu_state(const APIData &ad, bool dft = false);

    /**
     * \brief creates neural net instance based on model
     */
    void create_model();

    /**
     * \brief tests a model and compute measures
     * @param ad root data object
     * @param out output data object
     */
    void test(const APIData &ad, APIData &out);

    /**
     * \brief setups the nets, inputs, outputs, gradients, etc.
     */
    void create_model_train();
    void create_model_predict();

    /**
     * \brief (re)loads nets from the disk
     */
    void load_nets();

    /**
     * \brief recreates nets if the configuration changed and resets the
     * workspace
     */
    void update_model();

    /**
     * \brief dumps on the filesystem usefull information to resume training
     */
    void dump_model_state();

    /**
     * \brief finds the first the group of nets of the specified type
     * @param type type to find
     * @return the net group
     */
    Caffe2NetTools::NetGroup &find_net_group(const std::string &type);

    /**
     * \brief runs a net once (both forward and backward if the gradients are
     * set)
     * @param net net to run
     * @return the elapsed time
     */
    float run_net(const std::string &net);

    /**
     * \bried extracts the results of the last run
     * @param results vector of results ( [layer][batch_item][data] )
     * @param sizes vector to read / write the size of each batch item
     * @param batch_size number of item in this batch
     * @param outputs blobs created by the last run
     */
    void extract_results(std::vector<std::vector<std::vector<float>>> &results,
                         std::vector<size_t> &sizes, int batch_size,
                         const std::vector<std::string> &outputs);

    /**
     * \brief finds a net of the given type, execute it and fetch the output
     * @param results vector of results ( [layer][batch_item][data] )
     * @param sizes vector to read / write the size of each batch item data
     * @param batch_size number of item in this batch
     * @param type type of net to execute ("main" by default)
     */
    void
    typed_prediction(std::vector<std::vector<std::vector<float>>> &results,
                     std::vector<size_t> &sizes, int batch_size,
                     const std::string &type = "main");

    /**
     * \brief detects and reports model type
     * @param mltype output string variable
     */
    void model_type(std::string &mltype);

    Caffe2NetTools::ModelContext _context;
    std::vector<Caffe2NetTools::NetGroup> _nets;
    Caffe2LibState _state;
    TInputConnectorStrategy
        _last_inputc; // Last transformed version of the default _inputc
  };
}

#endif
