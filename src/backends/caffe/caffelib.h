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
#include "caffe/layers/memory_data_layer.hpp"
#include "caffe/layers/memory_sparse_data_layer.hpp"

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

    /**
     * \brief instanciate a model from template
     * @param ad mllib data object
     */
    void instantiate_template(const APIData &ad);

    /**
     * \brief configure an MLP template
     * @param ad the template data object
     * @param regression whether the net is a regressor
     * @param sparse whether the inputs are sparse
     * @param cnclasses the number of output classes, if any
     * @param net_param the training net object
     * @param deploy_net_param the deploy net object
     */
      void configure_mlp_template(const APIData &ad,
				  const TInputConnectorStrategy &inputc,
				  caffe::NetParameter &net_param,
				  caffe::NetParameter &dnet_param);


      void configure_convnet_template(const APIData &ad,
				      const TInputConnectorStrategy &inputc,
				      caffe::NetParameter &net_param,
				      caffe::NetParameter &dnet_param);
      
      void configure_resnet_template(const APIData &ad,
				     const TInputConnectorStrategy &inputc,
				     caffe::NetParameter &net_param,
				     caffe::NetParameter &dnet_param);

      void configure_ssd_template(const std::string &dest_net,
				  const std::string &deploy_dest_net,
				  const APIData &ad,
				  const bool &refinedet);

      void configure_recurrent_template(const APIData &ad,
                                        const TInputConnectorStrategy &inputc,
                                        caffe::NetParameter &net_param,
                                        caffe::NetParameter &dnet_param);

    /**
     * \brief configure noise data augmentation in training template
     * @param ad the template data object
     * @param net_param the trainng net object
     */
      void configure_image_augmentation(const APIData &ad,
				       caffe::NetParameter &net_param);

    /**
     * \brief creates neural net instance based on model
     * @return 0 if OK, 2, if missing 'deploy' file, 1 otherwise
     */
    int create_model(const bool &test=false);
    
    /*- from mllib -*/
    /**
     * \brief init this instance (e.g. sets GPU/CPU) and creates model
     * @param ad data object for "parameters/mllib"
     */
    void init_mllib(const APIData &ad);

    /**
     * \brief removes caffe model weight files and states from repository
     * @param ad root data object
     */
    void clear_mllib(const APIData &ad);

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
      * @param test_iter the number of test iterations (used for ctc)
      * @param out output data object
      */
      void test(caffe::Net<float> *net,
	      const APIData &ad,
	      TInputConnectorStrategy &inputc,
	      const int &test_batch_size,
	      const bool &has_mean_file,
		const int &test_iter,
		APIData &out);

    /**
     * \brief updates: - solver's paths to data according to current Caffe model
     *                 - net's batch size and data sources (e.g. lmdb sources)
     *        XXX: As for now, everything in the solver is volatile, and not written back to model file
     * @param sp Caffe's solver's parameters
     * @param ad the root data object
     * @param inputc the current input constructor that holds the training data
     * @param user_batch_size the user specified batch size
     * @param batch_size the automatically computed batch size, since some input Caffe layers do require it to divide cleanly the size of the training set
     * @param test_batch_size automatically computed value for the test batch size
     * @param test_iter automatically computed number of iterations of batches to cover the full test set
     * @param has_mean_file whether the current net uses a mean file (i.e. for images)
     */
      void update_in_memory_net_and_solver(caffe::SolverParameter &sp,
					   const APIData &ad,
					   const TInputConnectorStrategy &inputc,
					   bool &has_mean_file,
					   int &user_batch_size,
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
       * @param has_class_weights whether training uses class weights
       * @param ignore_label label to be ignored, -1 otherwise
       * @param timesteps number of timesteps for recurrent models
       */
      void update_protofile_net(const std::string &net_file,
				const std::string &deploy_file,
				const TInputConnectorStrategy &inputc,
				const bool &has_class_weights,
				const int &ignore_label,
				const int &timesteps);

      /**
       * \brief updates the softmax temperature
       * @param mllib apidata object
       */

      void update_deploy_protofile_softmax(const APIData &ad);
      /**
       * \brief updates the channel / timeteps number for timeseries
       * @param mllib apidata object
       */
      bool update_timesteps(int timesteps);
      
    private:
      void update_protofile_classes(caffe::NetParameter &net_param);

      void update_protofiles_one_hot(caffe::NetParameter &net_param);

      void update_protofiles_dice_deeplab_vgg16(caffe::NetParameter &net_param, caffe::NetParameter &deploy_net_param, const APIData &ad);
      void update_protofiles_dice_unet(caffe::NetParameter &net_param, caffe::NetParameter &deploy_net_param, const APIData &ad);

      void update_protofile_imageDataLayer(caffe::NetParameter &net_param);

      void update_protofiles_dice_params(caffe::DiceCoefLossParameter*dclp,  const APIData &ad);

      void update_protofile_finetune(caffe::NetParameter &net_param);

      caffe::LayerParameter* insert_layer_before
      (caffe::NetParameter &net_param, int layer_number);
      caffe::LayerParameter*  find_layer_by_name(caffe::NetParameter &net_param, std::string name);

      int find_index_layer_by_type(caffe::NetParameter &net_param, std::string type);
      int find_index_layer_by_name(caffe::NetParameter &net_param, std::string name);
      caffe::LayerParameter* find_layer_by_type(caffe::NetParameter &net_param, std::string type);


      void fix_batch_size(const APIData &ad,
			  const TInputConnectorStrategy &inputc,
			  int &user_batch_size,
			  int &batch_size,
			  int &test_batch_size,
			  int &test_iter);

      void set_gpuid(const APIData &ad);

      void model_complexity(long int &flops,
                            long int &params);

      void model_type(caffe::Net<float> *net,
		      std::string &mltype);


      /**
       * \brief checks wether v1 is better than v2
       */
      bool is_better(double v1, double v2, std::string metric_name);

      /**
       * \brief generates a file containing best iteration so far
       */
      void save_if_best(APIData &meas_out, boost::shared_ptr<caffe::Solver<float>>solver,
                        bool already_snapshoted);

      int findOutputSlotNumberByBlobName(const caffe::Net<float> *net,
                                     const std::string blob_name);

      boost::shared_ptr<Blob<float>> findOutputBlobByName(const caffe::Net<float> *net,
                                                   const std::string blob_name);


      boost::shared_ptr<Blob<float>> findBlobByName(const caffe::Net<float> *net,
                                             const std::string blob_name);

      std::vector<double> img_resize(const std::vector<double>& vals,
                                     const int height_net,  const int width_net,
                                     const int height_dest, const int width_dest, bool resize_nn);



    public:
      caffe::Net<float> *_net = nullptr; /**< neural net. */
      bool _gpu = false; /**< whether to use GPU. */
      std::vector<int> _gpuid = {0}; /**< GPU id. */
      int _nclasses = 0; /**< required, as difficult to acquire from Caffe's internals. */
      bool _regression = false; /**< whether the net acts as a regressor. */
      std::string _loss = ""; /**< loss to use : 0 softmax+multinomail logistic or1: dice*/
      int _ntargets = 0; /**< number of classification or regression targets. */
      //      std::vector<int> _targets; /**< id number of classification or regression targets. */
      bool _autoencoder = false; /**< whether an autoencoder. */
      std::mutex _net_mutex; /**< mutex around net, e.g. no concurrent predict calls as net is not re-instantiated. Use batches instead. */
      int _crop_size = -1; /**< cropping is part of Caffe transforms in input layers, storing here. */
      float _scale = 1.0; /**< scale is part of Caffe transforms in input layers, storing here. */

      std::vector<std::string> _best_metrics; /**< metric to use for saving best model */
      double _best_metric_value; /**< best metric value  */

      caffe::P2PSync<float> *_sync = nullptr;
      std::vector<boost::shared_ptr<caffe::P2PSync<float>>> _syncs;
    };

}

#endif
