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
     * \brief links the database with the given net
     */
    void add_tensor_loader(const Caffe2NetTools::ModelContext &context,
			   caffe2::NetDef &net,
			   bool train = false) const;

    /**
     * \bried fills informations about the tensor loader
     */
    void get_tensor_loader_infos(int &batch_size, int &total_size, bool train = false) const;

    /* Functions that should be re-implemented by childrens */

    // Automatic data transformations (used when loading from a database)

    /**
     * \brief adds operators to initilialize constant tensors
     * @param context the context of the net
     * @param init_net the net to update
     */
    void add_constant_layers(const Caffe2NetTools::ModelContext &, caffe2::NetDef &) {}

    /**
     * \brief adds operators to format the input
     * @param context the context of the net
     * @param net the net to update
     */
    void add_transformation_layers(const Caffe2NetTools::ModelContext &, caffe2::NetDef &) {}

    // Manual data transformations (from raw data)

    /**
     * \brief manually loads a batch
     * @param context context of the nets
     * @return size of this batch (0 if there was not enough data to fill the tensors)
     */
    int load_batch(Caffe2NetTools::ModelContext &) { return 0; }

  private:

    /* Internal functions */

    void set_batch_sizes(const APIData &ad, bool train);

  protected:

    /* Functions that should be called by the childrens */

    void init(const std::string &model_repo);

    // Should be called AFTER the children has initialized protected members
    void finalize_transform_predict(const APIData &ad);
    void finalize_transform_train(const APIData &ad);

    /**
     * \brief used to alert Caffe2Lib that the nets should be reconstructed
     * @param inputc last version of the input connector
     * @return true if a critical change occurred, false otherwise
     */
    bool needs_reconfiguration(const Caffe2InputInterface &inputc);

    /**
     * \brief compute the databases size (they must be created at this point)
     */
    void compute_db_sizes();

    /* Members managed by the mother class */

    std::string _default_db;
    std::string _default_train_db;
    bool _is_batched = true;
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
    std::vector<std::string> _ids; // input ids

    /* Public getters */
  public:
#define _GETTER(name) inline const decltype(_##name) &name() const { return _##name; }
    _GETTER(is_testable);
    _GETTER(is_load_manual);
    _GETTER(ids);
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
    int load_batch(Caffe2NetTools::ModelContext &context);
    bool needs_reconfiguration(const ImgCaffe2InputFileConn &inputc);
    void add_constant_layers(const Caffe2NetTools::ModelContext &context, caffe2::NetDef &init_net);
    void add_transformation_layers(const Caffe2NetTools::ModelContext &context,
				   caffe2::NetDef &net);

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
     */
    void uris_to_db();

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
    std::vector<float> _mean_values;

    //XXX Implement a way to change it ?
    std::string _blob_mean_values = "mean_values";
  };

  //XXX Do other connectors ___Caffe2InputFileConn (CSV, Txt, SVM, etc.)
}

#endif
