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
#pragma GCC diagnostic ignored "-Wunused-variable"
#include <torch/torch.h>
#pragma GCC diagnostic pop

#include "imginputfileconn.h"
#include "txtinputfileconn.h"
#include "csvtsinputfileconn.h"
#include "videoinputfileconn.h"
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
          _test_datasets(i._test_datasets), _input_format(i._input_format),
          _ctc(i._ctc), _ntargets(i._ntargets),
          _alphabet_size(i._alphabet_size), _tilogger(i._tilogger), _db(i._db)
    {
    }

    ~TorchInputInterface()
    {
    }

    void check_tests_sizes(size_t tests_size, size_t names_size)
    {
      if (tests_size != names_size)
        {
          std::string msg
              = "something wrong happened with datasets definitions: "
                "ndatasets : "
                + std::to_string(tests_size)
                + " vs ndatasets names : " + std::to_string(names_size);
          _tilogger->error(msg);
          throw InputConnectorInternalException(msg);
        }
    }

    void set_test_names(size_t test_count,
                        const std::vector<std::string> &uris);

    /**
     * \brief init  generic parts of input connector
     */
    void init(const APIData &ad_in, std::string model_repo,
              std::shared_ptr<spdlog::logger> logger)
    {
      _tilogger = logger;
      if (ad_in.has("ctc"))
        _ctc = ad_in.get("ctc").get<bool>();
      if (ad_in.has("shuffle"))
        _dataset.set_shuffle(ad_in.get("shuffle").get<bool>());
      if (ad_in.has("db"))
        _db = ad_in.get("db").get<bool>();
      _dataset.set_db_params(_db, _backend, model_repo + "/train");
      _dataset.set_logger(logger);
      _test_datasets.set_db_params(_db, _backend, model_repo + "/test");
      _test_datasets.set_logger(logger);
    }

    /**
     * \brief split db (instead of splitting data, because data is put in db on
     * the fly)
     */
    void build_test_datadb_from_full_datadb(double tsplit);

    /**
     * \brief if db already exists
     */
    bool has_to_create_db(const std::vector<std::string> &uris, double tsplit);

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
    void set_db_transaction_size(int32_t tsize)
    {
      _dataset.set_db_transaction_size(tsize);
      _test_datasets.set_db_transaction_size(tsize);
    }

    /**
     * \brief set seed for training dataset. This method should not
     * be used at predict time.
     */
    void set_train_dataset_seed(long seed)
    {
      seed = seed == -1 ? std::random_device()() : seed;
      _dataset.set_seed(seed);
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

    MaskedLMParams _lm_params;           /**< mlm data generation params */
    TorchDataset _dataset;               /**< train dataset */
    TorchMultipleDataset _test_datasets; /**< test datasets */
    std::string _input_format;           /**< for text, "bert" or nothing */

    bool _ctc = false; /**< whether this is a CTC service */
    unsigned int _ntargets
        = 0; /**< number of targets for regression / timeseries */
    int _alphabet_size = 0; /**< alphabet size for text prediction model */
    std::unordered_map<std::string, std::pair<int, int>>
        _imgs_size; /**< image sizes, used in detection. */

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
  };

  /**
   * \brief image connector to torch backend
   */
  class ImgTorchInputFileConn : virtual public ImgInputFileConn,
                                public TorchInputInterface
  {
  public:
    /**
     * \brief (almost) empty  constructor
     */
    ImgTorchInputFileConn() : ImgInputFileConn()
    {
      update_dataset_parameters();
      set_db_transaction_size(TORCH_IMG_TRANSACTION_SIZE);
    }

    /**
     * \brief copy constructor
     */
    ImgTorchInputFileConn(const ImgTorchInputFileConn &i)
        : ImgInputFileConn(i), TorchInputInterface(i), _bbox(i._bbox),
          _segmentation(i._segmentation), _supports_bw(i._supports_bw)
    {
      update_dataset_parameters();
      set_db_transaction_size(TORCH_IMG_TRANSACTION_SIZE);
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
      if (ad.has("bbox"))
        _bbox = ad.get("bbox").get<bool>();
      else if (ad.has("segmentation"))
        _segmentation = ad.get("segmentation").get<bool>();
      _dataset._bbox = _bbox;
      _dataset._segmentation = _segmentation;
      _test_datasets._bbox = _bbox;
      _test_datasets._segmentation = _segmentation;
      _test_datasets._test = true;
    }

    void fillup_parameters(oatpp::Object<DTO::InputConnector> input_params)
    {
      ImgInputFileConn::fillup_parameters(input_params);
      if (input_params->bbox != nullptr)
        _bbox = input_params->bbox;
      _dataset._bbox = _bbox;
      _test_datasets._bbox = _bbox;
    }

    /**
     * \brief read a whole dir
     */
    void read_image_folder(std::vector<std::pair<std::string, int>> &lfiles,
                           std::unordered_map<int, std::string> &hcorresp,
                           std::unordered_map<std::string, int> &hcorresp_r,
                           const std::string &folderPath,
                           const bool &test = false);

    /**
     * \brief read images from txt list
     */
    void read_image_list(
        std::vector<std::pair<std::string, std::vector<double>>> &lfiles,
        const std::string &listfilePath);

    /**
     * \brief read images from txt list. Targets are segmentation or bbox
     * files.
     */
    void read_image_file2file(
        std::vector<std::pair<std::string, std::string>> &lfiles,
        const std::string &listfilePath);

    /**
     * \brief read images from txt list. Targets are strings (for OCR)
     * */
    void
    read_image_text(std::vector<std::pair<std::string, std::string>> &lfiles,
                    const std::string &listfilePath);

    /**
     * \brief shuffle dataset
     */
    template <typename T>
    void shuffle_dataset(std::vector<std::pair<std::string, T>> &lfiles);

    /**
     * \brief split dataset into train and test
     */
    template <typename T>
    void split_dataset(std::vector<std::pair<std::string, T>> &lfiles,
                       std::vector<std::pair<std::string, T>> &test_lfiles);

    /**
     * \brief read data given apiData
     */
    void transform(const APIData &ad)
    {
      oatpp::Object<DTO::ServicePredict> predict_dto
          = ad.createSharedDTO<DTO::ServicePredict>();
      transform(predict_dto);
    }

    /**
     * \brief read data based on input call DTO
     */
    void transform(oatpp::Object<DTO::ServicePredict> predict_dto);

  private:
    bool _bbox = false;
    bool _segmentation = false;

    void update_dataset_parameters()
    {
      _dataset._inputc = this;
      _dataset._image = true;
      _dataset._bbox = _bbox;
      _dataset._segmentation = _segmentation;
      _test_datasets._inputc = this;
      _test_datasets._image = true;
      _test_datasets._bbox = _bbox;
      _test_datasets._segmentation = _segmentation;
    }

  public:
    bool _supports_bw = true;
  };

  class VideoTorchInputFileConn : public ImgTorchInputFileConn,
                                  public VideoInputFileConn
  {
  public:
    VideoTorchInputFileConn()
        : ImgInputFileConn(), ImgTorchInputFileConn(), VideoInputFileConn()
    {
    }

    /**
     * \brief copy constructor
     */
    VideoTorchInputFileConn(const VideoTorchInputFileConn &i)
        : ImgInputFileConn(i), ImgTorchInputFileConn(i), VideoInputFileConn(i)
    {
    }

    ~VideoTorchInputFileConn()
    {
    }
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
      _dataset._inputc = this;
      _test_datasets._inputc = this;
      _vocab_sep = '\t';
      set_db_transaction_size(TORCH_TEXT_TRANSACTION_SIZE);
    }
    /**
     * \brief copy constructor
     */
    TxtTorchInputFileConn(const TxtTorchInputFileConn &i)
        : TxtInputFileConn(i), TorchInputInterface(i), _width(i._width),
          _height(i._height)
    {
      _dataset._inputc = this;
      _test_datasets._inputc = this;
      set_db_transaction_size(TORCH_TEXT_TRANSACTION_SIZE);
    }

    ~TxtTorchInputFileConn()
    {
    }

    /**
     * \brief init the connector wrt apiData
     */
    void init(const APIData &ad)
    {
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
     * \brief read data based on input call DTO
     */
    void transform(oatpp::Object<DTO::ServicePredict> predict_dto)
    {
      // XXX: Requires TxtConnector DTO support
      (void)predict_dto;
      throw InputConnectorInternalException("Not supported yet");
    }

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
    void parse_content(const std::string &content, const float &target = -1,
                       int test_id = -1) override;

  private:
    /**
     * push read data to db
     * test < 0 => train
     * else test_id
     */
    void push_to_db(int test);

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
          _timesteps(i._timesteps), _datadim(i._datadim),
          _forecast_timesteps(i._forecast_timesteps),
          _backcast_timesteps(i._backcast_timesteps)
    {
      _dataset._inputc = this;
      _test_datasets._inputc = this;
    }

    ~CSVTSTorchInputFileConn()
    {
    }

    /**
     * \brief read data wrt apidata
     */
    void transform(const APIData &ad);

    /**
     * \brief read data based on input call DTO
     */
    void transform(oatpp::Object<DTO::ServicePredict> predict_dto)
    {
      // XXX: Requires CSVTSConnector DTO support
      (void)predict_dto;
      throw InputConnectorInternalException("Not supported yet");
    }

    /**
     * \brief read datadim from read data
     */
    void set_datadim(bool is_test_data = false);

    /**
     * \brief push data from csvts input conn to torch dataset
     */
    void fill_dataset(TorchDataset &dataset,
                      const std::vector<std::vector<CSVline>> &csvtsdata,
                      int test_id = -1);

    /**
     * \brief hook for push stuff into db
     */
    void read_csvts_file_post_hook(int test_id) override
    {
      if (!_db)
        return;

      this->update_columns();
      set_datadim();
      if (test_id == -1)
        {
          fill_dataset(_dataset, _csvtsdata);
          _csvtsdata.clear();
          _fnames.clear();
        }
      else
        {
          _test_datasets.add_test_name_if_necessary(_csv_test_fnames[test_id],
                                                    test_id);
          fill_dataset(_test_datasets[test_id], _csvtsdata_tests[test_id],
                       test_id);
          _csvtsdata_tests[test_id].clear();
          _test_fnames[test_id].clear();
        }
    }

    /**
     * \brief init the connector
     */
    void init(const APIData &ad)
    {
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
      if (ad_input.has("timesteps"))
        {
          _timesteps = ad_input.get("timesteps").get<int>();
          _offset = _timesteps;
        }

      if (ad_input.has("forecast_timesteps"))
        {
          _forecast_timesteps = ad_input.get("forecast_timesteps").get<int>();
        }

      if (ad_input.has("backcast_timesteps"))
        {
          _backcast_timesteps = ad_input.get("backcast_timesteps").get<int>();
          _offset = _backcast_timesteps;
        }

      if (ad_input.has("offset"))
        _offset = ad_input.get("offset").get<int>();

      if ((_forecast_timesteps >= 0 && _backcast_timesteps <= 0)
          || (_forecast_timesteps <= 0 && _backcast_timesteps >= 0))
        {
          std::string errmsg
              = "forecast value and backcast value should be both specified";
          this->_logger->error(errmsg);
          throw InputConnectorBadParamException(errmsg);
        }

      if (_forecast_timesteps > 0 && _forecast_timesteps > _backcast_timesteps)
        {
          std::string errmsg = "forecast value "
                               + std::to_string(_forecast_timesteps)
                               + "  >= backcast_timesteps "
                               + std::to_string(_backcast_timesteps);
          this->_logger->error(errmsg);
          throw InputConnectorBadParamException(errmsg);
        }
    }

    /**
     * \brief getter for channels/timesteps
     */
    int channels() const
    {
      if (_timesteps > 0)
        return _timesteps;
      return _forecast_timesteps + _backcast_timesteps;
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

  private:
    void fill_dataset_forecast(TorchDataset &dataset,
                               const std::vector<std::vector<CSVline>> &data,
                               int test_id);
    void add_data_instance_forecast(const unsigned long int tstart,
                                    const int vecindex, TorchDataset &dataset,
                                    const std::vector<CSVline> &seq);
    void fill_dataset_labels(TorchDataset &dataset,
                             const std::vector<std::vector<CSVline>> &data,
                             int test_id);
    void add_data_instance_labels(const unsigned long int tstart,
                                  const int vecindex, TorchDataset &dataset,
                                  const std::vector<CSVline> &seq,
                                  size_t seq_len);

    void add_seq(const size_t ti, const std::vector<CSVline> &seq,
                 std::vector<at::Tensor> &data_sequence,
                 std::vector<at::Tensor> &label_sequence);

    void discard_warn(int vecindex, unsigned int seq_size, int test_id);

  public:
    int _offset = -1;    /**< default offset for building sequences: start of
                            sequences is at 0, offset, 2xoffset ... */
    int _timesteps = -1; /**< default empty value for timesteps */
    int _datadim = -1;   /**< default empty value for datapoints */
    int _forecast_timesteps
        = -1; /**< length of forecast :  if > 0, labels will be ignored */
    int _backcast_timesteps
        = -1; /**< length of backcast :  if > 0, labels will be ignored */
  };
} // namespace dd

#endif // TORCHINPUTCONNS_H
