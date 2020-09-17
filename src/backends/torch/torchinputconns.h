/**
 * DeepDetect
 * Copyright (c) 2019 Jolibrain
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

#ifndef TORCHINPUTCONNS_H
#define TORCHINPUTCONNS_H

#include <vector>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <torch/torch.h>
#pragma GCC diagnostic pop

#include "imginputfileconn.h"
#include "txtinputfileconn.h"
#include "backends/torch/db.hpp"
#include "backends/torch/db_lmdb.hpp"
#include "csvtsinputfileconn.h"

#define TORCH_TEXT_TRANSATION_SIZE 100
#define TORCH_IMG_TRANSACTION_SIZE 10

namespace dd
{

  typedef torch::data::Example<std::vector<at::Tensor>,
                               std::vector<at::Tensor>>
      TorchBatch;

  class TorchDataset
      : public torch::data::BatchDataset<TorchDataset,
                                         c10::optional<TorchBatch>>
  {
  private:
    bool _shuffle = false;
    long _seed = -1;
    int64_t _current_index = 0;
    std::string _backend;
    bool _db;
    int32_t _batches_per_transaction = 10;
    std::shared_ptr<db::Transaction> _txn;
    std::shared_ptr<spdlog::logger> _logger;

  public:
    std::shared_ptr<db::DB> _dbData;
    std::vector<int64_t> _indices;
    /// Vector containing the whole dataset (the "cached data").
    std::vector<TorchBatch> _batches;
    std::string _dbFullName;

    TorchDataset()
    {
    }

    TorchDataset(const TorchDataset &d)
        : _shuffle(d._shuffle), _seed(d._seed),
          _current_index(d._current_index), _backend(d._backend), _db(d._db),
          _batches_per_transaction(d._batches_per_transaction), _txn(d._txn),
          _logger(d._logger), _dbData(d._dbData), _indices(d._indices),
          _batches(d._batches), _dbFullName(d._dbFullName)
    {
    }

    void set_transation_size(int32_t tsize)
    {
      _batches_per_transaction = tsize;
    }

    void add_batch(std::vector<at::Tensor> data,
                   std::vector<at::Tensor> target = {});
    void finalize_db();
    void pop(int64_t index, std::string &data, std::string &target);
    void add_elt(int64_t index, std::string data, std::string target);

    void write_tensors_to_db(std::vector<at::Tensor> data,
                             std::vector<at::Tensor> target);

    void reset(bool shuffle = true, db::Mode dbmode = db::READ);

    void set_shuffle(bool shuf)
    {
      _shuffle = shuf;
    }

    void set_dbParams(bool db, std::string backend, std::string dbname)
    {
      _db = db;
      _backend = backend;
      _dbFullName = dbname + "." + _backend;
    }

    void set_dbFile(std::string dbfname)
    {
      _db = true;
      _backend = "lmdb";
      _dbFullName = dbfname;
    }

    void set_logger(std::shared_ptr<spdlog::logger> logger)
    {
      _logger = logger;
    }

    /// Size of data loaded in memory
    size_t cache_size() const
    {
      return _batches.size();
    }

    c10::optional<size_t> size() const override
    {
      return cache_size();
    }

    bool empty() const
    {
      return (!_db && cache_size() == 0) || (_db && _dbFullName.empty());
    }

    c10::optional<TorchBatch> get_batch(BatchRequestType request) override;

    /// Returns a batch containing all the cached data
    TorchBatch get_cached();

    /// Split a percentage of this dataset
    TorchDataset split(double start, double stop);
  };

  struct MaskedLMParams
  {
    double _change_prob = 0.15; /**< When masked LM learning, probability of
                                   changing a token (mask/randomize/keep). */
    double _mask_prob
        = 0.8; /**< When masked LM learning, probability of masking a token. */
    double _rand_prob = 0.1; /**< When masked LM learning, probability of
                                randomizing a token. */
  };

  class TorchInputInterface
  {
  public:
    TorchInputInterface()
    {
    }
    TorchInputInterface(const TorchInputInterface &i)
        : _finetuning(i._finetuning), _lm_params(i._lm_params),
          _dataset(i._dataset), _test_dataset(i._test_dataset),
          _input_format(i._input_format), _ntargets(i._ntargets),
          _tilogger(i._tilogger), _db(i._db)
    {
    }

    ~TorchInputInterface()
    {
    }

    void init(const APIData &ad, std::string model_repo,
              std::shared_ptr<spdlog::logger> logger)
    {
      if (ad.has("db") && ad.get("db").get<bool>())
        _db = true;
      if (ad.has("shuffle"))
        _dataset.set_shuffle(ad.get("shuffle").get<bool>());
      _dataset.set_dbParams(_db, _backend, model_repo + "/train");
      _dataset.set_logger(logger);
      _tilogger = logger;
      _test_dataset.set_dbParams(_db, _backend, model_repo + "/test");
      _test_dataset.set_logger(logger);
    }

    void build_test_datadb_from_full_datadb(double tsplit);

    bool has_to_create_db(const APIData &ad, double tsplit);

    torch::Tensor toLongTensor(std::vector<int64_t> &values)
    {
      int64_t val_size = values.size();
      return torch::from_blob(&values[0], at::IntList{ val_size }, at::kLong)
          .clone();
    }

    TorchBatch generate_masked_lm_batch(__attribute__((unused))
                                        const TorchBatch &example)
    {
      return {};
    }

    void set_transation_size(int32_t tsize)
    {
      _dataset.set_transation_size(tsize);
      _test_dataset.set_transation_size(tsize);
    }

    int64_t mask_id() const
    {
      return 0;
    }
    int64_t vocab_size() const
    {
      return 0;
    }
    std::string get_word(__attribute__((unused)) int64_t id) const
    {
      return "";
    }

    bool _finetuning;
    MaskedLMParams _lm_params;
    /** Tell which inputs should be provided to the models.
     * see*/
    TorchDataset _dataset;
    TorchDataset _test_dataset;
    std::string _input_format;

    unsigned int _ntargets = 0;

    std::vector<int64_t>
        _lengths; /**< length of each sentence with txt connector. */
    std::shared_ptr<spdlog::logger> _tilogger;

    bool _db = false;
    std::string _dbname = "train";
    std::string _db_fname;
    std::string _test_db_name = "test";
    std::string _backend = "lmdb";
  };

  class ImgTorchInputFileConn : public ImgInputFileConn,
                                public TorchInputInterface
  {
  public:
    ImgTorchInputFileConn() : ImgInputFileConn()
    {
      set_transation_size(TORCH_IMG_TRANSACTION_SIZE);
    }
    ImgTorchInputFileConn(const ImgTorchInputFileConn &i)
        : ImgInputFileConn(i), TorchInputInterface(i)
    {
      set_transation_size(10);
    }
    ~ImgTorchInputFileConn()
    {
    }

    // for API info only
    int width() const
    {
      return _width;
    }

    // for API info only
    int height() const
    {
      return _height;
    }

    void init(const APIData &ad)
    {
      TorchInputInterface::init(ad, _model_repo, _logger);
      ImgInputFileConn::init(ad);
    }

    void transform(const APIData &ad)
    {
      try
        {
          ImgInputFileConn::transform(ad);
        }
      catch (const std::exception &e)
        {
          throw;
        }

      std::vector<at::Tensor> tensors;
      std::vector<int64_t> sizes{ _height, _width, 3 };
      at::TensorOptions options(at::ScalarType::Byte);

      for (const cv::Mat &bgr : this->_images)
        {
          at::Tensor imgt
              = torch::from_blob(bgr.data, at::IntList(sizes), options);
          imgt = imgt.toType(at::kFloat).permute({ 2, 0, 1 });
          size_t nchannels = imgt.size(0);
          if (_scale != 1.0)
            imgt = imgt.mul(_scale);
          if (!_mean.empty() && _mean.size() != nchannels)
            throw InputConnectorBadParamException(
                "mean vector be of size the number of channels ("
                + std::to_string(nchannels) + ")");
          for (size_t m = 0; m < _mean.size(); m++)
            imgt[0][m] = imgt[0][m].sub_(_mean.at(m));
          if (!_std.empty() && _std.size() != nchannels)
            throw InputConnectorBadParamException(
                "std vector be of size the number of channels ("
                + std::to_string(nchannels) + ")");
          for (size_t s = 0; s < _std.size(); s++)
            imgt[0][s] = imgt[0][s].div_(_std.at(s));
          tensors.push_back(imgt);
          _dataset.add_batch({ imgt });
        }
    }

  public:
    at::Tensor _in;
  };

  class TxtTorchInputFileConn : public TxtInputFileConn,
                                public TorchInputInterface
  {
  public:
    TxtTorchInputFileConn() : TxtInputFileConn()
    {
      _vocab_sep = '\t';
      set_transation_size(100);
    }
    TxtTorchInputFileConn(const TxtTorchInputFileConn &i)
        : TxtInputFileConn(i), TorchInputInterface(i), _width(i._width),
          _height(i._height)
    {
      set_transation_size(TORCH_TEXT_TRANSATION_SIZE);
    }
    ~TxtTorchInputFileConn()
    {
    }

    void init(const APIData &ad)
    {
      TxtInputFileConn::init(ad);
      TorchInputInterface::init(ad, _model_repo, _logger);
      fillup_parameters(ad);
    }

    void fillup_parameters(const APIData &ad_input);

    // for API info only
    int width() const
    {
      return _width;
    }

    // for API info only
    int height() const
    {
      return _height;
    }

    int64_t mask_id() const
    {
      return _mask_id;
    }

    int64_t vocab_size() const
    {
      return _vocab.size();
    }

    std::string get_word(int64_t id) const
    {
      return _inv_vocab.at(id);
    }

    void transform(const APIData &ad);

    TorchBatch generate_masked_lm_batch(const TorchBatch &example);

    void fill_dataset(TorchDataset &dataset,
                      const std::vector<TxtEntry<double> *> &entries);

    virtual void parse_content(const std::string &content,
                               const float &target = -1,
                               const bool &test = false);

    void push_to_db(bool test);

  public:
    /** width of the input tensor */
    unsigned int _width = 512;
    unsigned int _height = 0;
    std::mt19937 _rng;
    /// token id to vocabulary word
    std::map<int, std::string> _inv_vocab;

    int64_t _mask_id = -1; /**< ID of mask token in the vocabulary. */
    int64_t _cls_pos = -1;
    int64_t _sep_pos = -1;
    int64_t _unk_pos = -1;
    int64_t _eot_pos = -1; /**< end of text */

    void make_inv_vocab()
    {
      _inv_vocab.clear();

      for (auto &entry : _vocab)
        {
          _inv_vocab[entry.second._pos] = entry.first;
        }
    }
  };

  class CSVTSTorchInputFileConn : public CSVTSInputFileConn,
                                  public TorchInputInterface
  {
  public:
    CSVTSTorchInputFileConn() : CSVTSInputFileConn()
    {
    }

    CSVTSTorchInputFileConn(const CSVTSTorchInputFileConn &i)
        : CSVTSInputFileConn(i), TorchInputInterface(i), _offset(i._offset),
          _channels(i._channels), _timesteps(i._timesteps),
          _datadim(i._datadim)
    {
    }
    ~CSVTSTorchInputFileConn()
    {
    }

    void transform(
        const APIData &ad); // calls CSVTSInputfileconn::transform and db stuff
    void set_datadim(bool is_test_data = false);

    void fill_dataset(TorchDataset &dataset, bool use_csvtsdata_test);
    void init(const APIData &ad)
    {
      TorchInputInterface::init(ad, _model_repo, _logger);
      fillup_parameters(ad);
    }

    void fillup_parameters(const APIData &ad_input)
    {
      CSVTSInputFileConn::fillup_parameters(ad_input);
      if (_ntargets == 0 && ad_input.has("label"))
        {
          try
            {
              _ntargets = ad_input.get("label")
                              .get<std::vector<std::string>>()
                              .size();
            }
          catch (std::exception &e)
            {
              try
                {
                  std::string l = ad_input.get("label").get<std::string>();
                  _ntargets = 1;
                }
              catch (std::exception &e)
                {
                  throw InputConnectorBadParamException(
                      "no label given in input parameters, cannot determine "
                      "output size");
                }
            }
        }
      _offset = _timesteps;
      if (ad_input.has("timesteps"))
        {
          _timesteps = ad_input.get("timesteps").get<int>();
          _offset = _timesteps;
        }
      if (ad_input.has("offset"))
        _offset = ad_input.get("offset").get<int>();
    }

    int channels() const
    {
      return _timesteps;
    }

    int height() const
    {
      return _datadim;
    }

    int width() const
    {
      return 1;
    }

    int batch_size() const
    {
      return 1;
    }

    int test_batch_size() const
    {
      return 1;
    }

    int _offset = -1;
    int _channels = 0;
    int _timesteps = -1;
    int _datadim = -1;
  };
} // namespace dd

#endif // TORCHINPUTCONNS_H
