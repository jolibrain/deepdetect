/**
 * DeepDetect
 * Copyright (c) 2019-2020 Jolibrain
 * Author:  Guillaume Infantes <guillaume.infantes@jolibrain.com>
 *          Louis Jean <louis.jean@jolibrain.com>
 *          Emmanuel Benazera <emmanuel.benazera@jolibrain.com>
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

#ifndef TORCH_DATASET_H
#define TORCH_DATASET_H

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <torch/torch.h>
#pragma GCC diagnostic pop

#include "backends/torch/db.hpp"
#include "backends/torch/db_lmdb.hpp"

#include "inputconnectorstrategy.h"
#include "torchutils.h"

#include <opencv2/opencv.hpp>
#include <random>

namespace dd
{

  typedef torch::data::Example<std::vector<at::Tensor>,
                               std::vector<at::Tensor>>
      TorchBatch;

  /**
   * \brief dede torch dataset wrapper
   * allows reading from db, controllable randomness ...
   */
  class TorchDataset
      : public torch::data::BatchDataset<TorchDataset,
                                         c10::optional<TorchBatch>>
  {
  private:
    bool _shuffle = false; /**< shuffle dataset upon reset() */
    long _seed = -1;       /**< shuffle seed*/
    int64_t _current_index
        = 0; /**< current index for batch parallel data extraction */
    std::string _backend; /**< db backend (currently only lmdb supported= */
    bool _db;             /**< is data in db ? */
    int32_t _batches_per_transaction
        = 10; /**< number of batches per db transaction */
    std::shared_ptr<db::Transaction> _txn;   /**< db transaction pointer */
    std::shared_ptr<spdlog::logger> _logger; /**< dd logger */

  public:
    std::shared_ptr<db::DB> _dbData; /**< db data */
    std::vector<int64_t> _indices;   /**< id/key  of data points */

    std::vector<TorchBatch> _batches; /**< Vector containing the whole dataset
                                         (the "cached data") */
    std::string _dbFullName;          /**< db filename */
    InputConnectorStrategy *_inputc = nullptr;

    /**
     * \brief empty constructor
     */
    TorchDataset()
    {
    }

    /**
     * \brief copy constructor
     */
    TorchDataset(const TorchDataset &d)
        : _shuffle(d._shuffle), _seed(d._seed),
          _current_index(d._current_index), _backend(d._backend), _db(d._db),
          _batches_per_transaction(d._batches_per_transaction), _txn(d._txn),
          _logger(d._logger), _dbData(d._dbData), _indices(d._indices),
          _batches(d._batches), _dbFullName(d._dbFullName), _inputc(d._inputc)
    {
    }

    /**
     *  \brief setter for transaction size_t
     */
    void set_transaction_size(int32_t tsize)
    {
      _batches_per_transaction = tsize;
    }

    /**
     * \brief add data to dataset
     */
    void add_batch(const std::vector<at::Tensor> &data,
                   const std::vector<at::Tensor> &target = {});

    /**
     * \brief commits final db transactions
     */
    void finalize_db();

    /**
     * \brief get one elementt from dataset, remove it
     */
    void pop(int64_t index, std::string &data, std::string &target);

    /**
     * \brief add one element to dataset
     */
    void add_elt(int64_t index, std::string data, std::string target);

    /**
     * \brief reset dataset reading status : ie start new epoch
     */
    void reset(bool shuffle = true, db::Mode dbmode = db::READ);

    /**
     * \brief setter for _shuffle
     */
    void set_shuffle(bool shuf)
    {
      _shuffle = shuf;
    }

    /**
     * \brief setter for db metadata
     */
    void set_dbParams(bool db, std::string backend, std::string dbname)
    {
      _db = db;
      _backend = backend;
      _dbFullName = dbname + "." + _backend;
    }

    /**
     * \brief setter for db filename
     */
    void set_dbFile(std::string dbfname)
    {
      _db = true;
      _backend = "lmdb";
      _dbFullName = dbfname;
    }

    /**
     * \brief setter for _logger
     */
    void set_logger(std::shared_ptr<spdlog::logger> logger)
    {
      _logger = logger;
    }

    /**
     * \brief Size of data loaded in memory
     */
    size_t cache_size() const
    {
      return _batches.size();
    }

    /**
     * \brief Size of data loaded in memory
     */
    c10::optional<size_t> size() const override
    {
      return cache_size();
    }

    /**
     * \brief test if this data set has been filled
     */
    bool empty() const
    {
      return (!_db && cache_size() == 0) || (_db && _dbFullName.empty());
    }

    /**
     * \brief get some data from dataset, this is the method used by
     * torch::dataloader
     */
    c10::optional<TorchBatch> get_batch(BatchRequestType request) override;

    /**
     * \brief get tensor dims if data #i (of first element of dataset)
     */
    std::vector<long int> datasize(long int i) const;

    /**
     * \brief Returns a batch containing all the cached data
     */
    TorchBatch get_cached();

    /**
     * \brief Split a percentage of this dataset
     */
    TorchDataset split(double start, double stop);

    /*-- image tools --*/
    int add_image_file(const std::string &fname, const int &target,
                       const int &height, const int &width);
    int add_image_file(const std::string &fname,
                       const std::vector<double> &target, const int &height,
                       const int &width);
    at::Tensor image_to_tensor(const cv::Mat &bgr, const int &height,
                               const int &width);
    at::Tensor target_to_tensor(const int &target);
    at::Tensor target_to_tensor(const std::vector<double> &target);

  private:
    /**
     * \brief converts and write data to db
     */
    void write_tensors_to_db(const std::vector<at::Tensor> &data,
                             const std::vector<at::Tensor> &target);
  };

}

#endif
