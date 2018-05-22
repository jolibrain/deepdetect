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

//XXX Remove that to print the warnings
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
#include <caffe2/core/tensor.h>
#pragma GCC diagnostic pop

#include "imginputfileconn.h"
#include "csvinputfileconn.h"
#include "txtinputfileconn.h"
#include "svminputfileconn.h"

namespace dd {

  /**
   * \brief high-level data structure shared among Caffe2-compatible connectors of DeepDetect
   */
  class Caffe2InputInterface {
  public:
    Caffe2InputInterface() {}
    ~Caffe2InputInterface() {}

    /**
     * \brief fill the input tensor
     * @param tensor the tensor to fill
     * @param num the batch size (or -1 to put all the data in one batch)
     * @return the real size of this batch (used to know if there was less data than 'num')
     * @see ImgCaffe2InputFileConn
     */
    int get_batch(caffe2::TensorCPU &, int = -1) { return 0; }

    std::string _dbfullname = "train.lmdb";
    std::string _test_dbfullname = "test.lmdb";
    std::vector<std::string> _ids; /* input ids (e.g. image ids) */
  };

  /**
   * \brief Caffe2 image connector
   */
  class ImgCaffe2InputFileConn : public ImgInputFileConn, public Caffe2InputInterface {
  public:
    ImgCaffe2InputFileConn(): ImgInputFileConn(), Caffe2InputInterface() {}
    ~ImgCaffe2InputFileConn() {}

    inline int height() const { return _height; }
    inline int width() const { return _width; }

    void init(const APIData &ad);
    void transform(const APIData &ad);
    int get_batch(caffe2::TensorCPU &tensor, int num = -1);

    /**
     * \brief transform the generic 'TensorProtosDBInput' into an 'ImageInput'
     * @param op operator to transform
     * @param mean_values mean values to subtract from each channel
     */
    void configure_db_operator(caffe2::OperatorDef &op, const std::vector<double> &mean_values);

  private:

    inline int channels() const {
      return _bw ? 1 : 3;
    }

    void transform_predict(const APIData &ad);
    void transform_train(const APIData &ad);

    void compute_images_mean(const std::string &dbname,
			     const std::string &meanfile,
			     const std::string &backend="lmdb");

    /**
     * \brief Convert images into db entries.
     *        If rfolders contains one root folder, it will be used for both training and testing.
     *        Else, the first is used for training and the second for testing.
     */
    void images_to_db(const std::vector<std::string> &rfolders,
		      const std::string &traindbname,
		      const std::string &testdbname,
		      const std::string &backend="lmdb");

    /**
     * \brief Use the given root directory to match images with their class
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

    void write_image_to_db(const std::string &dbfullname,
			   const std::vector<std::pair<std::string,int>> &lfiles,
			   const std::string &backend);

    std::string _meanfname = "mean.pb";
    std::string _correspname = "corresp.txt";
    double _std = 1.0f;
  };

}

#endif
