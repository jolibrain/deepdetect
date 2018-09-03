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

#ifndef CAFFE2INPUTCONNS_H
#define CAFFE2INPUTCONNS_H

#include "imginputfileconn.h"
#include "csvinputfileconn.h"
#include "txtinputfileconn.h"
#include "svminputfileconn.h"
#include "backends/caffe2/nettools.h"

namespace dd {

  /**
   * \brief high-level data structure shared among Caffe2-compatible connectors of DeepDetect
   */
  class Caffe2InputInterface {
  public:
    Caffe2InputInterface() {}
    ~Caffe2InputInterface() {}

    /* Functions that should be kept by childrens */

    /**
     * \brief reinserts dumped database informations into the workspace
     */
    void load_dbreader(Caffe2NetTools::ModelContext &context,
		       const std::string &file, bool train = false) const;

    /**
     * \brief inserts database informations into an initialization net
     */
    void create_dbreader(caffe2::NetDef &init_net, bool train = false) const;

    /**
     * \brief asserts that the context can be used with the current input configuration
     */
    void assert_context_validity(Caffe2NetTools::ModelContext &context, bool train = false) const;

    /* Functions that should be re-implemented by childrens */

    // Automatic data transformations (used when loading from a database)

    /**
     * \brief adds operators to initilialize constant tensors
     * @param context the context of the net
     * @param init_net the net to update
     */
    void add_constant_layers(const Caffe2NetTools::ModelContext &, caffe2::NetDef &) const {}

    /**
     * \brief adds operators to format the input
     * @param context the context of the net
     * @param net the net to update
     */
    void add_transformation_layers(const Caffe2NetTools::ModelContext &, caffe2::NetDef &) const {}

    // Database read

    /**
     * \brief links the dbreader with the given net
     */
    void link_train_dbreader(const Caffe2NetTools::ModelContext &context,
			     caffe2::NetDef &net) const;

    /**
     * \brief uses the dbreader to insert data into the workspace
     * @param context context of the nets
     * @param already_loaded how many tensors must be ignored
     * @return size of this batch (0 if there was not enough data to fill the tensors)
     */
    int use_test_dbreader(Caffe2NetTools::ModelContext &context, int already_loaded) const;

    // Manual data transformations (from raw data)

    /**
     * \brief loads a batch
     * @param context context of the nets
     * @param already_loaded how many tensor where already loaded
     * @return size of this batch (0 if there was not enough data to fill the tensors)
     */
    int load_batch(Caffe2NetTools::ModelContext &, int) { return 0; }

    // Global configuration of the network

    bool _measuring = false;

  private:

    /* Internal functions */

    void set_batch_sizes(const APIData &ad, bool train);

  protected:

    /* Functions that should be called by the childrens */

    void init(InputConnectorStrategy *child);

    // Should be called AFTER the children has initialized protected members
    void finalize_transform_predict(const APIData &ad);
    void finalize_transform_train(const APIData &ad);

    /**
     * \brief used to alert Caffe2Lib that the nets should be reconstructed
     * @param inputc last version of the input connector
     * @return true if a critical change occurred, false otherwise
     */
    bool needs_reconfiguration(const Caffe2InputInterface &inputc) const;

    /**
     * \brief compute the databases size (they must be created at this point)
     */
    void compute_db_sizes();

    // Function that configure a tensor loader with given dbreader and batch size
    using DBInputSetter = std::function<void(caffe2::OperatorDef&, const std::string &, int)>;

    void link_dbreader(const Caffe2NetTools::ModelContext &context, caffe2::NetDef &net,
		       const DBInputSetter &config_dbinput, bool train) const;

    // Function that convert a TensorProtos into a vector of tensor (already allocated)
    using ProtosConverter =
      std::function<void(const caffe2::TensorProtos&, std::vector<caffe2::TensorCPU>&)>;

    /**
     * \brief uses the dbreader to insert data into the workspace
     * @param context context of the nets
     * @param already_loaded how many tensors must be ignored
     * @param convert_protos, callback to convert a TensorProtos into the corresponding tensors
     * @param train which db must be read
     * @return size of this batch (0 if there was not enough data to fill the tensors)
     */
    int use_dbreader(Caffe2NetTools::ModelContext &context, int already_loaded,
		     const ProtosConverter &convert_protos, bool train) const;

    // Function that populate a vector with input tensors (already allocated)
    using InputGetter = std::function<void(std::vector<caffe2::TensorCPU>&)>;

    /**
     * \brief fill the workspace with batches of tensors
     * @param context context of the nets
     * @param blobs name of the blobs to fill
     * @param nb_data number of data available
     * @param get_tensors callback to fetch an input (one tensor per blob)
     * @param train whether to use the train batch size or not
     * @return how many tensors were insered
     */
    int insert_inputs(Caffe2NetTools::ModelContext &context, const std::vector<std::string> &blobs,
		      int nb_data, const InputGetter &get_tensors, bool train) const;

    /* Members managed by the mother class */

    InputConnectorStrategy *_child = NULL;

    std::string _default_db;
    std::string _default_train_db;
    int _db_size = 0;
    int _train_db_size = 0;
    int _batch_size = 0;
    int _train_batch_size = 0;
    int _default_batch_size = 32;

    //XXX Implement a way to change thoses ?
    std::string _db_type = "lmdb";
    std::string _blob_dbreader = "dbreader";
    std::string _blob_dbreader_train = "dbreader_train";
    std::string _db_relative_path = "/test.lmdb";
    std::string _train_db_relative_path = "/train.lmdb";

    /* Members that should be managed by the childrens */

    std::string _db; // path to the database
    std::string _train_db; // path to the training database
    bool _is_testable = false; // whether test data is available
    bool _is_load_manual = true; // whether data is manually loaded (as opposed to database-loaded)
    bool _is_batchable = true; // whether inputs can be pre-computed into the same format
    std::vector<std::string> _ids; // input ids
    std::vector<std::vector<float>> _scales; // input scale coefficients

    /* Public getters */
  public:
#define _GETTER(name) inline const decltype(_##name) &name() const { return _##name; }
    _GETTER(is_testable);
    _GETTER(is_load_manual);
    _GETTER(ids);
    _GETTER(scales);
#undef _GETTER
  };

  /**
   * \brief Caffe2 image connector
   */
  class ImgCaffe2InputFileConn : public ImgInputFileConn, public Caffe2InputInterface {
  public:
    ImgCaffe2InputFileConn(): ImgInputFileConn(), Caffe2InputInterface() {}
    ~ImgCaffe2InputFileConn() {}

    /* Overloads */

    inline int height() const { return _height; }
    inline int width() const { return _width; }

    void init(const APIData &ad);
    void transform(const APIData &ad);
    void link_train_dbreader(const Caffe2NetTools::ModelContext &context,
			     caffe2::NetDef &net) const;
    int use_test_dbreader(Caffe2NetTools::ModelContext &context, int already_loaded) const;
    int load_batch(Caffe2NetTools::ModelContext &context, int already_loaded);
    bool needs_reconfiguration(const ImgCaffe2InputFileConn &inputc) const;
    void add_constant_layers(const Caffe2NetTools::ModelContext &context,
			     caffe2::NetDef &init_net) const;
    void add_transformation_layers(const Caffe2NetTools::ModelContext &context,
				   caffe2::NetDef &net) const;

  private:

    inline int channels() const {
      return _bw ? 1 : 3;
    }

    /**
     * \brief updates private members (e.g. _std)
     */
    void update(const APIData &ad);

    void transform_predict(const APIData &ad);
    void transform_train(const APIData &ad);

    /**
     * \brief initilializes mean values
     */
    void load_mean_file();

    /**
     * \brief transforms a tensor proto containing an image into a vector of channels
     *        See pytorch/caffe2/image/image_input_op.h GetImageAndLabelAndInfoFromDBValue
     */
    void image_proto_to_mats(const caffe2::TensorProto &proto,
			     std::vector<cv::Mat> &mats,
			     bool resize=false) const;

    /**
     * \brief transforms vector of channels into a CHW float tensor
     */
    void mats_to_tensor(const std::vector<cv::Mat> &mats, caffe2::TensorCPU &tensor) const;

    /**
     * \brief stores the image dimensions into a tensor
     */
    void im_info_to_tensor(const cv::Mat &img, caffe2::TensorCPU &tensor) const;

    /**
     * \brief creates mean file
     */
    void compute_images_mean();

    /**
     * \brief converts images into db entries.
     *        If '_uris' contains one root folder, it will be used for both training and testing.
     *        Else, the first is used for training and the second for testing.
     */
    void images_to_db();

    /**
     * \brief checks which database(s) can/should be used depending on the 'uris' content
     * @return true if images were used to create (a) new database(s), false otherwise
     */
    bool uris_to_db();

    /**
     * \brief uses the given root directory to match images with their class
     * @param root folder that contains a subfolder by class
     * @param corresp correspondence class id / class name
     * @param corresp_r reverse correspondence
     * @param files list of labeled files
     * @param is_reversed 'true' means using corresp_r to fetch the ids,
     *                    'false' means filling corresp_r with ids for a future use
     */
    void list_images(const std::string &root,
		     std::unordered_map<int, std::string> &corresp,
		     std::unordered_map<std::string,int> &corresp_r,
		     std::vector<std::pair<std::string, int>> &files,
		     bool is_reversed);

    /**
     * \brief writes pairs of file/label inside a database
     */
    void write_images_to_db(const std::string &dbname,
			    const std::vector<std::pair<std::string,int>> &lfiles);

    std::string _mean_file;
    std::string _corresp_file;
    float _std = 1.0f;

    //XXX Implement a way to change it ?
    std::string _blob_mean_values = "mean_values";
  };

  //XXX Do other connectors ___Caffe2InputFileConn (CSV, Txt, SVM, etc.)
}

#endif
