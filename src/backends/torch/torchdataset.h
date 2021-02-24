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
#include "torchdataaug.h"
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
    long _seed = -1; /**< shuffle seed*/
    std::mt19937 _rng;
    int64_t _current_index
        = 0; /**< current index for batch parallel data extraction */
    std::string _backend; /**< db backend (currently only lmdb supported= */
    bool _db = false;     /**< is data in db ? */
    int32_t _batches_per_transaction
        = 10; /**< number of batches per db transaction */
    std::shared_ptr<db::Transaction> _txn;   /**< db transaction pointer */
    std::shared_ptr<spdlog::logger> _logger; /**< dd logger */

  public:
    bool _shuffle = false;           /**< shuffle dataset upon reset() */
    std::shared_ptr<db::DB> _dbData; /**< db data */
    db::Cursor *_dbCursor = nullptr; /**< db cursor */
    std::vector<int64_t> _indices;   /**< id/key  of data points */
    std::vector<std::pair<std::string, std::vector<double>>>
        _lfiles; /**< list of files */

    std::vector<TorchBatch> _batches; /**< Vector containing the whole dataset
                                         (the "cached data") */
    std::string _dbFullName;          /**< db filename */
    InputConnectorStrategy *_inputc
        = nullptr;               /**< back ptr to input connector. */
    bool _classification = true; /**< whether a classification dataset. */

    bool _image = false;                /**< whether an image dataset. */
    TorchImgRandAugCV _img_rand_aug_cv; /**< image data augmentation policy. */

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
        : _seed(d._seed), _rng(d._rng), _current_index(d._current_index),
          _backend(d._backend), _db(d._db),
          _batches_per_transaction(d._batches_per_transaction), _txn(d._txn),
          _logger(d._logger), _shuffle(d._shuffle), _dbData(d._dbData),
          _indices(d._indices), _lfiles(d._lfiles), _batches(d._batches),
          _dbFullName(d._dbFullName), _inputc(d._inputc),
          _classification(d._classification), _image(d._image),
          _img_rand_aug_cv(d._img_rand_aug_cv)
    {
    }

    TorchDataset(TorchDataset &&d) = default;

    virtual ~TorchDataset()
    {
    }

    /**
     * \brief add data to dataset
     */
    void add_batch(const std::vector<at::Tensor> &data,
                   const std::vector<at::Tensor> &target = {});

    /**
     * \brief add an encoded image to a batch, with an int target
     */
    void add_image_batch(const cv::Mat &bgr, const int &width,
                         const int &height,
                         const std::vector<at::Tensor> &targett);

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
     * \brief setter for _seed & reinitialize random number generator
     */
    void set_seed(long seed)
    {
      _seed = seed;
      _rng = std::mt19937(seed);
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
      return (!_db && cache_size() == 0 && _lfiles.empty())
             || (_db && _dbFullName.empty());
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
     * \brief get tensor dims if data #i (of first element of dataset)
     */
    std::vector<long int> targetsize(long int i) const;

    /**
     * \brief Returns a batch containing all the cached data
     */
    TorchBatch get_cached();

    /**
     * \brief Split a percentage of this dataset
     */
    TorchDataset split(double start, double stop);

    /**- db -**/

    /**
     *  \brief setter for transaction size_t
     */
    void set_db_transaction_size(int32_t tsize)
    {
      _batches_per_transaction = tsize;
    }

    /**
     * \brief commits final db transactions
     */
    void db_finalize();

    /**
     * \brief setter for db metadata
     */
    void set_db_params(const bool &db, const std::string &backend,
                       const std::string &dbname)
    {
      _db = db;
      _backend = backend;
      _dbFullName = dbname + "." + _backend;
    }

    /**
     * \brief setter for db filename
     */
    void set_db_file(const std::string &dbfname)
    {
      _db = true;
      _backend = "lmdb";
      _dbFullName = dbfname;
    }

    /**
     * \brief get one elementt from dataset, remove it
     */
    void pop_db_elt(int64_t index, std::string &data, std::string &target);

    /**
     * \brief add one element to dataset
     */
    void add_db_elt(int64_t index, std::string data, std::string target);

    /*-- list --*/

    /**
     * \brief set list of files
     */
    void set_list(
        const std::vector<std::pair<std::string, std::vector<double>>> &lfiles)
    {
      _lfiles = lfiles;
    }

    /*-- image tools --*/

    int add_image_file(const std::string &fname,
                       const std::vector<at::Tensor> &target,
                       const int &height, const int &width);

    /**
     * \brief adds image from image filename, with an int target
     * \param width of preprocessed image
     * \param height of preprocessed image
     */
    int add_image_file(const std::string &fname, const int &target,
                       const int &height, const int &width);

    /**
     * \brief adds image from image filename, with a set of regression targets
     * \param width of preprocessed image
     * \param height of preprocessed image
     */
    int add_image_file(const std::string &fname,
                       const std::vector<double> &target, const int &height,
                       const int &width);

    /**
     * \brief adds image to batch, with a bbox list file as target.
     * \param width of preprocessed image
     * \param height of preprocessed image
     */
    int add_image_bbox_file(const std::string &fname,
                            const std::string &bboxfname, const int &height,
                            const int &width);

    /**
     * \brief turns an image into a torch::Tensor
     */
    at::Tensor image_to_tensor(const cv::Mat &bgr, const int &height,
                               const int &width);

    /**
     * \brief turns an int into a torch::Tensor
     */
    at::Tensor target_to_tensor(const int &target);

    /**
     * \brief turns a vector of double into a torch::Tensor
     */
    at::Tensor target_to_tensor(const std::vector<double> &target);

  private:
    /**
     * \brief converts and write data to db
     */
    void write_tensors_to_db(const std::vector<at::Tensor> &data,
                             const std::vector<at::Tensor> &target);

    /**
     * \brief writes encoded image to db with a tensor target
     */
    void write_image_to_db(const cv::Mat &bgr,
                           const std::vector<torch::Tensor> &target);

    /**
     * \brief reads an encoded image from db along with its tensor target
     */
    void read_image_from_db(const std::string &datas,
                            const std::string &targets, cv::Mat &bgr,
                            std::vector<torch::Tensor> &targett,
                            const bool &bw);
  };

  /**
   * \brief this class holds several datasets
   */
  class TorchMultipleDataset
  {

  public:
    TorchMultipleDataset()
    {
    }

    /*
     * \brief copy constructor
     */
    TorchMultipleDataset(const TorchMultipleDataset &d)
        : _inputc(d._inputc), _image(d._image),
          _classification(d._classification), _dbFullNames(d._dbFullNames),
          _datasets_names(d._datasets_names), _db(d._db), _backend(d._backend),
          _dbPrefix(d._dbPrefix), _logger(d._logger),
          _batches_per_transaction(d._batches_per_transaction),
          _datasets(d._datasets)
    {
    }

    /**
     * \brief setter for db metadata
     */
    void set_db_params(const bool &db, const std::string &backend,
                       const std::string &dbname)
    {
      _db = db;
      _backend = backend;
      _dbPrefix = dbname;
    }

    /**
     * \brief setter for _logger
     */
    void set_logger(std::shared_ptr<spdlog::logger> logger)
    {
      _logger = logger;
    }

    /**
     * \brief set transaction size on both dataset (train and test)
     */
    void set_db_transaction_size(int32_t tsize)
    {
      _batches_per_transaction = tsize;
    }

    /**
     * \brief commits final db transactions
     */
    void db_finalize()
    {
      for (auto &d : _datasets)
        d.db_finalize();
    }

    /*
     * \brief get db name as given in api or computed if not given
     */
    const std::string &dbFullName(const size_t i)
    {
      return _dbFullNames[i];
    }

    /*
     * \brief set list of files along with value, ie for regression
     */
    void set_list(const std::vector<std::vector<
                      std::pair<std::string, std::vector<double>>>> &lsfiles);

    /*
     * \brief get one set
     */
    TorchDataset &operator[](const size_t id)
    // TorchDataset &get(const size_t id)
    {
      return _datasets[id];
    }

    /*
     * \brief get one set
     */
    const TorchDataset &operator[](const size_t id) const
    // TorchDataset &get(const size_t id)
    {
      return _datasets[id];
    }

    /*
     * \brief get all test set names (duh!)
     */
    std::vector<std::string> &names()
    {
      return _datasets_names;
    }

    /*
     * \brief get all test set names (duh!)
     */
    std::string name(size_t i)
    {
      return _datasets_names[i];
    }

    /*
     * \brief check if some set has db'ed data
     */
    bool has_db_data()
    {
      for (auto &s : _datasets)
        if (s._dbData != nullptr)
          return true;
      return false;
    }

    void reset()
    {
      for (auto &s : _datasets)
        s.reset();
    }

    size_t size() const
    {
      return _datasets.size();
    }

    /*
     * \brief _allocate_ test sets given vector of names as found in API
     */
    void add_tests_names(const std::vector<std::string> &longnames);

    /*
     * \brief _allocate_ test set given name as found in API
     */
    void add_test_name(std::string longname);

    /*
     * allocate new test set if necessary (in case of on the fly adding , ie db
     * case
     */
    void add_test_name_if_necessary(std::string longname, int test_id);

    /*
     * \brief _allocate_ test set given db name as found in API
     */
    void add_db_name(std::string dblongname);

  protected:
    void test_name_from_db_name(size_t id)
    {
      _datasets_names[id] = fileops::shortname(_dbFullNames[id]);
    }

    void set_db_name(size_t id, std::string name = "")
    {
      if (name == "" || !fileops::is_db(name))
        _dbFullNames[id] = _dbPrefix + "_"
                           + std::to_string(_dbFullNames.size() - 1) + "."
                           + _backend;
      else
        _dbFullNames[id] = name;
    }

    /*
     * \brief copy attributes to new test set
     */
    void init_set(const size_t id)
    {
      _datasets[id]._inputc = _inputc;
      _datasets[id]._image = _image;
      _datasets[id]._classification = _classification;
      _datasets[id].set_db_params(_db, _backend,
                                  _dbPrefix + "_" + std::to_string(id));
      _datasets[id].set_logger(_logger);
      _datasets[id].set_db_transaction_size(_batches_per_transaction);
    }

  public:
    InputConnectorStrategy *_inputc
        = nullptr;               /**< back ptr to input connector. */
    bool _image = false;         /**< whether an image dataset. */
    bool _classification = true; /**< whether a classification dataset. */
    std::vector<std::string> _dbFullNames;
    std::vector<std::string> _datasets_names;

  protected:
    bool _db = false;
    std::string _backend = "lmdb";
    std::string _dbPrefix;
    std::shared_ptr<spdlog::logger> _logger; /**< dd logger */
    int32_t _batches_per_transaction
        = 10; /**< number of batches per db transaction */
    std::vector<TorchDataset> _datasets;
  };
}

#endif
