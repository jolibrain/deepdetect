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
#include "csvtsinputfileconn.h"
#include "torchdataset.h"
#include "torchutils.h"

#define TORCH_TEXT_TRANSACTION_SIZE 100
#define TORCH_IMG_TRANSACTION_SIZE 10

namespace dd
{

  struct MaskedLMParams
  {
    double _change_prob = 0.15; /**< When masked LM learning, probability of
                                   changing a token (mask/randomize/keep). */
    double _mask_prob
        = 0.8; /**< When masked LM learning, probability of masking a token. */
    double _rand_prob = 0.1; /**< When masked LM learning, probability of
                                randomizing a token. */
  };

  /**
   * \brief common behavior of all torch input connectors
   */
  class TorchInputInterface
  {
  public:
    /**
     *  \brief empty constructor
     */
    TorchInputInterface()
    {
    }

    /**
     *  \brief copy constructor
     */
    TorchInputInterface(const TorchInputInterface &i)
        : _lm_params(i._lm_params), _dataset(i._dataset),
          _test_dataset(i._test_dataset), _input_format(i._input_format),
          _ntargets(i._ntargets), _tilogger(i._tilogger), _db(i._db),
          _split_ts_for_predict(i._split_ts_for_predict)
    {
    }

    ~TorchInputInterface()
    {
    }

    /**
     * \brief init  generic parts of input connector
     */
    void init(const APIData &ad_in, std::string model_repo,
              std::shared_ptr<spdlog::logger> logger)
    {
      _tilogger = logger;
      if (ad_in.has("shuffle"))
        _dataset.set_shuffle(ad_in.get("shuffle").get<bool>());
      if (ad_in.has("db") && ad_in.get("db").get<bool>())
        _db = true;
      _dataset.set_dbParams(_db, _backend, model_repo + "/train");
      _dataset.set_logger(logger);
      _test_dataset.set_dbParams(_db, _backend, model_repo + "/test");
      _test_dataset.set_logger(logger);
    }

    /**
     * \brief split db (instead of splitting data, because data is put in db on
     * the fly)
     */
    void build_test_datadb_from_full_datadb(double tsplit);

    /**
     * \brief if db already exists
     */
    bool has_to_create_db(const APIData &ad, double tsplit);

    /**
     * \brief holder for mlm data generation (defined in txtinputconn)
     */
    TorchBatch generate_masked_lm_batch(__attribute__((unused))
                                        const TorchBatch &example)
    {
      return {};
    }

    /**
     * \brief set transaction size on both dataset (train and test)
     */
    void set_transaction_size(int32_t tsize)
    {
      _dataset.set_transaction_size(tsize);
      _test_dataset.set_transaction_size(tsize);
    }

    /**
     * \brief holder for mask_id for mlm data generation
     */
    int64_t mask_id() const
    {
      return 0;
    }

    /**
     * \brief holder for vocab size for txtinputconn
     */
    int64_t vocab_size() const
    {
      return 0;
    }

    /**
     * \brief holder vocab accessor for txtinputconn
     */
    std::string get_word(__attribute__((unused)) int64_t id) const
    {
      return "";
    }

    /**
     * \brief get first input for exploration (size ...)
     */
    std::vector<c10::IValue> get_input_example(torch::Device device);

    MaskedLMParams _lm_params;  /**< mlm data generation params */
    TorchDataset _dataset;      /**< train dataset */
    TorchDataset _test_dataset; /**< test dataset */
    std::string _input_format;  /**< for text, "bert" or nothing */

    unsigned int _ntargets
        = 0; /**< number of targets for regression / timeseries */

    std::vector<int64_t>
        _lengths; /**< length of each sentence with txt connector. */
    std::shared_ptr<spdlog::logger> _tilogger; /**< instance of dd logger */

    bool _db = false;              /**< wether to use a db */
    std::string _dbname = "train"; /**< train db default filename prefix */
    std::string _db_fname;         /**< db full filename */
    std::string _test_db_name = "test"; /**< test db default filename prefix */
    std::string _backend
        = "lmdb"; /**< db backend (currently only lmdb is supported) */
    std::string _correspname = "corresp.txt"; /**< "corresp file default name*/
    bool _split_ts_for_predict
        = false; /**< prevent to split timeseries in predict mode */
  };

  /**
   * \brief image connector to torch backend
   */
  class ImgTorchInputFileConn : public ImgInputFileConn,
                                public TorchInputInterface
  {
  public:
    /**
     * \brief (almost) empty  constructor
     */
    ImgTorchInputFileConn() : ImgInputFileConn()
    {
      _dataset._inputc = this;
      _test_dataset._inputc = this;
      set_transaction_size(TORCH_IMG_TRANSACTION_SIZE);
    }

    /**
     * \brief copy constructor
     */
    ImgTorchInputFileConn(const ImgTorchInputFileConn &i)
        : ImgInputFileConn(i), TorchInputInterface(i)
    {
      _dataset._inputc = this;
      _test_dataset._inputc = this;
      set_transaction_size(TORCH_IMG_TRANSACTION_SIZE);
    }

    ~ImgTorchInputFileConn()
    {
    }

    /**
     * \brief getter for width
     */
    int width() const
    {
      return _width;
    }

    /**
     * \brief getter for height
     */
    int height() const
    {
      return _height;
    }

    /**
     * \brief init the connector given APIdata
     */
    void init(const APIData &ad)
    {
      TorchInputInterface::init(ad, _model_repo, _logger);
      ImgInputFileConn::init(ad);
    }

    /**
     * \brief read a whole dir
     */
    void read_image_folder(std::vector<std::pair<std::string, int>> &lfiles,
                           std::unordered_map<int, std::string> &hcorresp,
                           std::unordered_map<std::string, int> &hcorresp_r,
                           const std::string &folderPath);

    /**
     * \brief read images from txt list
     */
    void read_image_list(
        std::vector<std::pair<std::string, std::vector<double>>> &lfiles,
        const std::string &listfilePath);

    /**
     * \brief split dataset into train and test
     */
    template <typename T>
    void split_dataset(std::vector<std::pair<std::string, T>> &lfiles,
                       std::vector<std::pair<std::string, T>> &test_lfiles);

    /**
     * \brief read data given apiData
     */
    void transform(const APIData &ad);

  private:
    /*template <typename T>
    int add_image_file(TorchDataset &dataset, const std::string &fname,
                       T target);

    at::Tensor image_to_tensor(const cv::Mat &bgr);

    at::Tensor target_to_tensor(const int &target);

    at::Tensor target_to_tensor(const std::vector<double> &target);*/
  };

  /**
   * \brief txt input connector
   */
  class TxtTorchInputFileConn : public TxtInputFileConn,
                                public TorchInputInterface
  {
  public:
    /**
     * \brief (almost) empty constructor
     */
    TxtTorchInputFileConn() : TxtInputFileConn()
    {
      _vocab_sep = '\t';
      set_transaction_size(TORCH_TEXT_TRANSACTION_SIZE);
    }
    /**
     * \brief copy constructor
     */
    TxtTorchInputFileConn(const TxtTorchInputFileConn &i)
        : TxtInputFileConn(i), TorchInputInterface(i), _width(i._width),
          _height(i._height)
    {
      set_transaction_size(TORCH_TEXT_TRANSACTION_SIZE);
    }

    ~TxtTorchInputFileConn()
    {
    }

    /**
     * \brief init the connector wrt apiData
     */
    void init(const APIData &ad)
    {
      _dataset._inputc = this;
      _test_dataset._inputc = this;
      TxtInputFileConn::init(ad);
      TorchInputInterface::init(ad, _model_repo, _logger);
      fillup_parameters(ad);
    }

    /**
     * \brief helper for reading apiData
     */
    void fillup_parameters(const APIData &ad_input);

    /**
     * \brief getter for width
     */
    int width() const
    {
      return _width;
    }

    /**
     * \brief getter for height
     */
    int height() const
    {
      return _height;
    }

    /**
     * \brief value of mask for MLM data generation
     */
    int64_t mask_id() const
    {
      return _mask_id;
    }

    /**
     * \brief nuber of tokens in global vocabulary
     */
    int64_t vocab_size() const
    {
      return _vocab.size();
    }

    /**
     * \brief get token given id
     */
    std::string get_word(int64_t id) const
    {
      return _inv_vocab.at(id);
    }

    /**
     * \brief read data wrt APIdata
     */
    void transform(const APIData &ad);

    /**
     * \brief genrate MLM self supervised data
     */
    TorchBatch generate_masked_lm_batch(const TorchBatch &example);

    /**
     * \brief put txt data into data set
     */
    void fill_dataset(TorchDataset &dataset,
                      const std::vector<TxtEntry<double> *> &entries);

    /**
     * \brief override txtinputconn parse content in order to put data in db on
     * the fly if needed
     */
    virtual void parse_content(const std::string &content,
                               const float &target = -1,
                               const bool &test = false);

  private:
    /**
     * push read data to db
     */
    void push_to_db(bool test);

  public:
    unsigned int _width = 512; /**< width of the input tensor */
    unsigned int _height = 0;  /**< default height */
    std::mt19937 _rng;         /**< random number generator for MLM */
    std::map<int, std::string> _inv_vocab; /**< token id to vocabulary word */

    int64_t _mask_id = -1; /**< ID of mask token in the vocabulary. */
    int64_t _cls_pos = -1; /**< cls token */
    int64_t _sep_pos = -1; /**< separator token */
    int64_t _unk_pos = -1; /**< unknown token */
    int64_t _eot_pos = -1; /**< end of text */

    /**
     * \brief build reverse vocal map
     */
    void make_inv_vocab()
    {
      _inv_vocab.clear();

      for (auto &entry : _vocab)
        {
          _inv_vocab[entry.second._pos] = entry.first;
        }
    }
  };

  /**
   * timeserie input connector
   */
  class CSVTSTorchInputFileConn : public CSVTSInputFileConn,
                                  public TorchInputInterface
  {
  public:
    /**
     * \brief empty constructor
     */
    CSVTSTorchInputFileConn() : CSVTSInputFileConn()
    {
    }

    /**
     * \brief copy constructor
     */
    CSVTSTorchInputFileConn(const CSVTSTorchInputFileConn &i)
        : CSVTSInputFileConn(i), TorchInputInterface(i), _offset(i._offset),
          _timesteps(i._timesteps), _datadim(i._datadim)
    {
    }

    ~CSVTSTorchInputFileConn()
    {
    }

    /**
     * \brief read data wrt apidata
     */
    void transform(const APIData &ad);

    /**
     * \brief read datadim from read data
     */
    void set_datadim(bool is_test_data = false);

    /**
     * \brief push data from csvts input conn to torch dataset
     */
    void fill_dataset(TorchDataset &dataset, bool use_csvtsdata_test);

    /**
     * \brief init the connector
     */
    void init(const APIData &ad)
    {
      _dataset._inputc = this;
      _test_dataset._inputc = this;
      TorchInputInterface::init(ad, _model_repo, _logger);
      fillup_parameters(ad);
    }

    /**
     * \brief helper to read apiData
     */
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

    /**
     * \brief getter for channels/timesteps
     */
    int channels() const
    {
      return _timesteps;
    }

    /**
     * \brief getter for height / size of 1 datapoint
     */
    int height() const
    {
      return _datadim;
    }

    /**
     * \brief timeserie do not have width
     */
    int width() const
    {
      return 1;
    }

    int _offset = -1;    /**< default offset for building sequences: start of
                            sequences is at 0, offset, 2xoffset ... */
    int _timesteps = -1; /**< default empty value for timesteps */
    int _datadim = -1;   /**< default empty value for datapoints */
  };
} // namespace dd

#endif // TORCHINPUTCONNS_H
