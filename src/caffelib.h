/**
 * DeepDetect
 * Copyright (c) 2014-2015 Emmanuel Benazera
 * Author: Emmanuel Benazera <beniz@droidnik.fr>
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

#ifndef CAFFELIB_H
#define CAFFELIB_H

#include "mllibstrategy.h"
#include "caffemodel.h"
#include "caffe/caffe.hpp"

using caffe::Blob;

namespace dd
{
  /**
   * \brief Caffe library wrapper for deepdetect
   */
  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel=CaffeModel>
    class CaffeLib : public MLLib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>
    {
      friend class caffe::MemoryDataLayer<float>;
    public:
    /**
     * \brief constructor from model
     * @param model Caffe model
     */
    CaffeLib(const CaffeModel &cmodel);
    
    /**
     * \brief copy-constructor
     */
    CaffeLib(CaffeLib &&cl) noexcept;
    
    /**
     * \brief destructor
     */
    ~CaffeLib();

    //TODO: instanciate from model template
    void instantiate_template(const APIData &ad);
    
    /**
     * \brief creates neural net instance based on model
     * @return 0 if OK, 2, if missing 'deploy' file, 1 otherwise
     */
    int create_model();
    
    /*- from mllib -*/
    /**
     * \brief init this instance (e.g. sets GPU/CPU) and creates model
     * @param ad data object for "parameters/mllib"
     */
    void init_mllib(const APIData &ad);

    /**
     * \brief train new model
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
    
    //TODO: status ?

    /*- local functions -*/
     /**
      * \brief test net
      * @param ad root data object
      * @param inputc input connector
      * @param test_batch_size current size of the test batches
      * @param has_mean_file whereas testing set uses a mean file (for images)
      * @param out output data object
      */
    void test(const APIData &ad,
	      TInputConnectorStrategy &inputc,
	      const int &test_batch_size,
	      const bool &has_mean_file,
	      APIData &out);

    /**
     * \brief updates: - solver's paths to data according to current Caffe model
     *                 - net's batch size and data sources (e.g. lmdb sources)
     *        XXX: As for now, everything in the solver is volatile, and not written back to model file
     * @param sp Caffe's solver's parameters
     * @param ad the root data object
     * @param inputc the current input constructor that holds the training data
     * @param has_mean_file whether the current net uses a mean file (i.e. for images)
     */
      void update_in_memory_net_and_solver(caffe::SolverParameter &sp,
					   const APIData &ad,
					   const TInputConnectorStrategy &inputc,
					   bool &has_mean_file,
					   int &batch_size,
					   int &test_batch_size,
					   int &test_iter);

      /**
       * \brief updates the protocol buffer text file of the net directly at training time
       *        i.e. changes are permanent. This is against the rules of training calls that 
       *        should not make hard changes to service configuration but this is the only way
       *        to transparently use the data in input in order to shape a net's template.
       * 
       * @param net_file the file containing the net template, as copied from the template itself
       *                 at service creation
       * @param deploy_file the deploy file, same remark as net_file
       * @param inputc the current input constructor that holds the training data
       */
      void update_protofile_net(const std::string &net_file,
				const std::string &deploy_file,
				const TInputConnectorStrategy &inputc);

    private:
      void update_protofile_classes(caffe::NetParameter &net_param);

      void fix_batch_size(const APIData &ad,
			  const TInputConnectorStrategy &inputc,
			  int &batch_size,
			  int &test_batch_size,
			  int &test_iter);
      
    public:
      caffe::Net<float> *_net = nullptr; /**< neural net. */
      bool _gpu = false; /**< whether to use GPU. */
      int _gpuid = 0; /**< GPU id. */
      int _nclasses = 0; /**< required, as difficult to acquire from Caffe's internals. */
    };
  
}

#endif
