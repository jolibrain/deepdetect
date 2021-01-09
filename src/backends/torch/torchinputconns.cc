/**
 * DeepDetect
 * Copyright (c) 2020 Jolibrain
 * Authors: Louis Jean <ljean@etud.insa-toulouse.fr>
 *    Guillaume Infantes <guillaume.infantes@jolibrain.com>
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

#include "torchinputconns.h"

#include "utils/utils.hpp"

namespace dd
{

  using namespace torch;

  // XXX (beniz): to be moved into torchdataset
  void TorchInputInterface::build_test_datadb_from_full_datadb(double tsplit)
  {
    _tilogger->info("splitting : using {} of dataset as test set", tsplit);
    _dataset.reset(true, db::WRITE);
    std::vector<int64_t> indicestest;
    int64_t ntest = _dataset._indices.size() * tsplit;
    auto seed = static_cast<long>(time(NULL));
    std::mt19937 rng(seed);

    for (int64_t i = 0; i < ntest; i++)
      {
        std::uniform_int_distribution<int64_t> index_distrib(
            0, _dataset._indices.size() - 1);
        int64_t index = _dataset._indices[index_distrib(rng)];
        std::string data;
        std::string target;
        _dataset.pop_db_elt(index, data, target);
        _test_dataset.add_db_elt(index, data, target);
      }
    _test_dataset.db_finalize();
    _dataset.db_finalize();
  }

  bool TorchInputInterface::has_to_create_db(const APIData &ad,
                                             double test_split)
  {
    // here force db paths manually if given at call time
    std::vector<std::string> uris
        = ad.get("data").get<std::vector<std::string>>();

    if (uris.size() >= 1)
      {
        if (fileops::dir_exists(uris[0]) && fileops::is_db(uris[0]))
          {
            _dataset._dbFullName = uris[0];
            if (uris.size() == 2 && fileops::is_db(uris[1]))
              _test_dataset._dbFullName = uris[1];
          }
      }
    if (fileops::file_exists(_dataset._dbFullName))
      {
        _tilogger->warn("db {} already exists, not rebuilding it",
                        _dataset._dbFullName);
        if (!fileops::file_exists(_test_dataset._dbFullName))
          {
            if (test_split != 0.0)
              build_test_datadb_from_full_datadb(test_split);
          }
        else
          {
            _tilogger->warn("test db {} already exists, not rebuilding it",
                            _test_dataset._dbFullName);
          }
        return false;
      }
    return true;
  }

  std::vector<c10::IValue>
  TorchInputInterface::get_input_example(torch::Device device)
  {
    _dataset.reset();
    auto batchopt = _dataset.get_batch({ 1 });
    TorchBatch batch = batchopt.value();
    std::vector<c10::IValue> input_example;

    for (auto &t : batch.data)
      {
        input_example.push_back(t.to(device));
      }
    return input_example;
  }

  // ===== ImgTorchInputFileConn

  void ImgTorchInputFileConn::read_image_folder(
      std::vector<std::pair<std::string, int>> &lfiles,
      std::unordered_map<int, std::string> &hcorresp,
      std::unordered_map<std::string, int> &hcorresp_r,
      const std::string &folderPath, const bool &test)
  {
    _logger->info("Reading image folder {}", folderPath);

    // TODO Put file parsing from caffe in common files to use it in other
    // backends
    int cl = 0;

    std::unordered_map<std::string, int>::const_iterator hcit;
    std::unordered_set<std::string> subdirs;
    if (fileops::list_directory(folderPath, false, true, false, subdirs))
      throw InputConnectorBadParamException(
          "failed reading image train data directory " + folderPath);

    auto uit = subdirs.begin();
    while (uit != subdirs.end())
      {
        std::unordered_set<std::string> subdir_files;
        if (fileops::list_directory((*uit), true, false, true, subdir_files))
          throw InputConnectorBadParamException(
              "failed reading image train data sub-directory " + (*uit));
        std::string cls = dd_utils::split((*uit), '/').back();
        if (!test)
          {
            hcorresp.insert(std::pair<int, std::string>(cl, cls));
            hcorresp_r.insert(std::pair<std::string, int>(cls, cl));
          }
        else
          {
            if ((hcit = hcorresp_r.find(cls)) != hcorresp_r.end())
              cl = (*hcit).second;
            else
              _logger->warn("unknown class {} in test set", cls);
          }
        auto fit = subdir_files.begin();
        while (
            fit
            != subdir_files.end()) // XXX: re-iterating the file is not optimal
          {
            lfiles.push_back(std::pair<std::string, int>((*fit), cl));
            ++fit;
          }
        if (!test)
          ++cl;
        ++uit;
      }
  }

  void ImgTorchInputFileConn::read_image_list(
      std::vector<std::pair<std::string, std::vector<double>>> &lfiles,
      const std::string &listfilePath)
  {
    _logger->info("Reading image list file {}", listfilePath);
    std::ifstream infile(listfilePath);
    std::string line;
    int line_num = 1;
    while (std::getline(infile, line))
      {
        std::istringstream iss(line);
        string filename;
        string label;
        iss >> filename >> label;

        std::vector<double> targets;
        std::istringstream ill(label);
        std::string targ;
        while (ill >> targ)
          targets.push_back(std::stod(targ));

        ++line_num;
        lfiles.push_back(
            std::pair<std::string, std::vector<double>>(filename, targets));
      }
    _logger->info("Read {} lines in image list file {}", line_num,
                  listfilePath);
  }

  void ImgTorchInputFileConn::transform(const APIData &ad)
  {
    if (!_train)
      {
        try
          {
            ImgInputFileConn::transform(ad);
          }
        catch (InputConnectorBadParamException &e)
          {
            throw;
          }

        // XXX: No predict from db yet
        _dataset.set_db_params(false, "", "");

        for (size_t i = 0; i < this->_images.size(); ++i)
          {
            _dataset.add_batch({ _dataset.image_to_tensor(this->_images[i],
                                                          _height, _width) });
          }
      }
    else // if (!_train)
      {
        // This must be done since we don't call ImgInputFileConn::transform
        if (ad.has("parameters")) // overriding default parameters
          {
            APIData ad_param = ad.getobj("parameters");
            if (ad_param.has("input"))
              {
                fillup_parameters(ad_param.getobj("input"));
              }
          }

        // Read all parsed files and create tensor datasets
        bool createDb
            = _db && TorchInputInterface::has_to_create_db(ad, _test_split);
        bool shouldLoad = !_db || createDb;

        if (shouldLoad)
          {
            if (_db)
              _tilogger->info("Preparation for training from db");
            // Get files paths
            try
              {
                get_data(ad);
              }
            catch (InputConnectorBadParamException &e)
              {
                throw;
              }

            bool dir_images = true;
            bool fexists = fileops::file_exists(_uris.at(0), dir_images);
            if (!fexists)
              throw InputConnectorBadParamException(
                  "Torch image input connector folders or image list "
                  + _uris.at(0) + " does not exist");
            bool folder = dir_images;

            // Parse URIs and retrieve images
            if (folder)
              {
                std::unordered_map<int, std::string>
                    hcorresp; // correspondence class number / class name
                std::unordered_map<std::string, int>
                    hcorresp_r; // reverse correspondence for test set.
                std::vector<std::pair<std::string, int>>
                    lfiles; // labeled files
                std::vector<std::pair<std::string, int>> test_lfiles;

                read_image_folder(lfiles, hcorresp, hcorresp_r, _uris.at(0));
                if (_uris.size() > 1)
                  {
                    read_image_folder(test_lfiles, hcorresp, hcorresp_r,
                                      _uris.at(1), true);
                  }

                if (_dataset._shuffle)
                  shuffle_dataset<int>(lfiles);

                bool has_test_data = test_lfiles.size() != 0;
                if (_test_split > 0.0 && !has_test_data)
                  {
                    split_dataset<int>(lfiles, test_lfiles);
                  }

                // Read data
                for (const std::pair<std::string, int> &lfile : lfiles)
                  {
                    _dataset.add_image_file(lfile.first, lfile.second, _height,
                                            _width);
                  }

                for (const std::pair<std::string, int> &lfile : test_lfiles)
                  {
                    _test_dataset.add_image_file(lfile.first, lfile.second,
                                                 _height, _width);
                  }

                // Write corresp file
                std::ofstream correspf(_model_repo + "/" + _correspname,
                                       std::ios::binary);
                auto hit = hcorresp.begin();
                while (hit != hcorresp.end())
                  {
                    correspf << (*hit).first << " " << (*hit).second
                             << std::endl;
                    ++hit;
                  }
                correspf.close();
              }
            else // file exists, expects a list of files and targets, for
                 // regression & multi-class
              {
                std::vector<std::pair<std::string, std::vector<double>>>
                    lfiles; // labeled files
                std::vector<std::pair<std::string, std::vector<double>>>
                    test_lfiles;
                read_image_list(lfiles, _uris.at(0));
                if (_uris.size() > 1)
                  read_image_list(test_lfiles, _uris.at(1));

                if (_dataset._shuffle)
                  shuffle_dataset<std::vector<double>>(lfiles);

                bool has_test_data = test_lfiles.size() != 0;
                if (_test_split > 0.0 && !has_test_data)
                  {
                    split_dataset<std::vector<double>>(lfiles, test_lfiles);
                  }

                // Read data
                if (_db)
                  {
                    for (const std::pair<std::string, std::vector<double>>
                             &lfile : lfiles)
                      {
                        _dataset.add_image_file(lfile.first, lfile.second,
                                                _height, _width);
                      }

                    for (const std::pair<std::string, std::vector<double>>
                             &lfile : test_lfiles)
                      {
                        _test_dataset.add_image_file(lfile.first, lfile.second,
                                                     _height, _width);
                      }
                  }
                else
                  {
                    _dataset.set_list(lfiles);
                    _test_dataset.set_list(test_lfiles);
                  }
              }
          }

        if (createDb)
          {
            _dataset.db_finalize();
            _test_dataset.db_finalize();
          }
      }
  }

  template <typename T>
  void ImgTorchInputFileConn::shuffle_dataset(
      std::vector<std::pair<std::string, T>> &lfiles)
  {
    std::mt19937 g;
    if (_seed >= 0)
      g = std::mt19937(_seed);
    else
      {
        std::random_device rd;
        g = std::mt19937(rd());
      }
    std::shuffle(lfiles.begin(), lfiles.end(), g);
  }

  template <typename T>
  void ImgTorchInputFileConn::split_dataset(
      std::vector<std::pair<std::string, T>> &lfiles,
      std::vector<std::pair<std::string, T>> &test_lfiles)
  {
    // Split
    int split_pos = std::floor(lfiles.size() * (1.0 - _test_split));

    auto split_begin = lfiles.begin();
    std::advance(split_begin, split_pos);
    test_lfiles.insert(test_lfiles.begin(), split_begin, lfiles.end());
    lfiles.erase(split_begin, lfiles.end());

    _logger->info("data split test size={} / remaining data size={}",
                  test_lfiles.size(), lfiles.size());
  }

  // ===== TxtTorchInputFileConn

  void TxtTorchInputFileConn::parse_content(const std::string &content,
                                            const float &target,
                                            const bool &test)
  {
    _ndbed = 0;
    TxtInputFileConn::parse_content(content, target, test);
    if (_db)
      push_to_db(test);
  }

  void TxtTorchInputFileConn::fillup_parameters(const APIData &ad_input)
  {
    TxtInputFileConn::fillup_parameters(ad_input);
    if (ad_input.has("db"))
      _db = ad_input.get("db").get<bool>();
  }

  void TxtTorchInputFileConn::push_to_db(bool test)
  {
    if (!test)
      {
        _logger->info("pushing to train_db");
        fill_dataset(_dataset, _txt);
        destroy_txt_entries(_txt);
      }
    else
      {
        _logger->info("pushing to test_db");
        fill_dataset(_test_dataset, _test_txt);
        destroy_txt_entries(_test_txt);
      }
  }

  void TxtTorchInputFileConn::transform(const APIData &ad)
  {
    // if (_finetuning)
    // XXX: Generating vocab from scratch is not currently

    if (!_ordered_words || _characters)
      throw InputConnectorBadParamException(
          "Need ordered_words = true with backend torch");

    _generate_vocab = false;

    if (!_characters && (!_train || _ordered_words) && _vocab.empty())
      deserialize_vocab();

    // XXX: move in txtinputconn?
    make_inv_vocab();

    if (_input_format == "bert")
      {
        _cls_pos = _vocab.at("[CLS]")._pos;
        _sep_pos = _vocab.at("[SEP]")._pos;
        _unk_pos = _vocab.at("[UNK]")._pos;
        _mask_id = _vocab.at("[MASK]")._pos;
      }
    else if (_input_format == "gpt2")
      {
        _eot_pos = _vocab.at("<|endoftext|>")._pos;
      }

    if (ad.has("parameters") && ad.getobj("parameters").has("input"))
      {
        APIData ad_input = ad.getobj("parameters").getobj("input");
        fillup_parameters(ad_input);
      }

    try
      {

        if (_db)
          {
            if (TorchInputInterface::has_to_create_db(ad, _test_split))
              {
                double save_ts = _test_split;
                _test_split = 0.0;
                TxtInputFileConn::transform(ad);
                _test_split = save_ts;
                _dataset.db_finalize();
                bool has_test_data = _test_dataset._dbData != nullptr;
                _test_dataset.db_finalize();
                if (_test_split != 0.0 && !has_test_data)
                  build_test_datadb_from_full_datadb(_test_split);
              }
          }
        else
          {
            TxtInputFileConn::transform(ad);
          }
      }
    catch (const std::exception &e)
      {
        throw;
      }

    if (!_db)
      {
        fill_dataset(_dataset, _txt);
        destroy_txt_entries(_txt);
        if (!_test_txt.empty())
          {
            fill_dataset(_test_dataset, _test_txt);
            destroy_txt_entries(_test_txt);
          }
      }
  }

  TorchBatch
  TxtTorchInputFileConn::generate_masked_lm_batch(const TorchBatch &example)
  {
    std::uniform_real_distribution<double> uniform(0, 1);
    std::uniform_int_distribution<int64_t> vocab_distrib(0, vocab_size() - 1);
    Tensor input_ids = example.data.at(0).clone();
    // lm_labels: n_batch * sequence_length
    // equals to input_ids where tokens are masked, and -1 otherwise
    Tensor lm_labels = torch::ones_like(input_ids, TensorOptions(kLong)) * -1;

    // mask random tokens
    auto input_acc = input_ids.accessor<int64_t, 2>();
    auto att_mask_acc = example.data.at(2).accessor<int64_t, 2>();
    auto labels_acc = lm_labels.accessor<int64_t, 2>();
    for (int i = 0; i < input_ids.size(0); ++i)
      {
        int j = 1; // skip [CLS] token
        while (j < input_ids.size(1) && att_mask_acc[i][j] != 0)
          {
            double rand_num = uniform(_rng);
            if (rand_num < _lm_params._change_prob
                && input_acc[i][j] != _sep_pos)
              {
                labels_acc[i][j] = input_acc[i][j];

                rand_num = uniform(_rng);
                if (rand_num < _lm_params._mask_prob)
                  {
                    input_acc[i][j] = mask_id();
                  }
                else if (rand_num
                         < _lm_params._mask_prob + _lm_params._rand_prob)
                  {
                    input_acc[i][j] = vocab_distrib(_rng);
                  }
              }
            ++j;
          }
      }

    TorchBatch output;
    output.target.push_back(lm_labels);
    output.data.push_back(input_ids);
    for (unsigned int i = 1; i < example.data.size(); ++i)
      {
        output.data.push_back(example.data[i]);
      }
    return output;
  }

  void TxtTorchInputFileConn::fill_dataset(
      TorchDataset &dataset, const std::vector<TxtEntry<double> *> &entries)
  {

    _ndbed = 0;
    for (auto *te : entries)
      {
        TxtOrderedWordsEntry *tow = static_cast<TxtOrderedWordsEntry *>(te);
        tow->reset();
        std::string word;
        double val;
        std::vector<int64_t> ids;

        while (tow->has_elt())
          {
            if (ids.size() >= _width)
              break;

            tow->get_next_elt(word, val);
            std::unordered_map<std::string, Word>::iterator it;

            if ((it = _vocab.find(word)) != _vocab.end())
              {
                ids.push_back(it->second._pos);
              }
            else if (_input_format == "bert")
              {
                ids.push_back(_unk_pos);
              }
          }

        // Extract last token (needed by gpt2)
        int64_t last_token = 0;
        if (tow->has_elt())
          {
            tow->get_next_elt(word, val);
            std::unordered_map<std::string, Word>::iterator it;

            if ((it = _vocab.find(word)) != _vocab.end())
              last_token = it->second._pos;
          }

        // Post-processing for each model
        if (_input_format == "bert")
          {
            // make room for cls and sep token
            while (ids.size() > _width - 2)
              ids.pop_back();

            ids.insert(ids.begin(), _cls_pos);
            ids.push_back(_sep_pos);
          }
        else if (_input_format == "gpt2")
          {
            if (ids.size() < _width)
              {
                ids.push_back(_eot_pos);
              }
          }

        at::Tensor ids_tensor = torch_utils::toLongTensor(ids);
        at::Tensor mask_tensor = torch::ones_like(ids_tensor);
        at::Tensor token_type_ids_tensor = torch::zeros_like(ids_tensor);

        int64_t seq_len = ids_tensor.sizes().back();
        int64_t padding_size = _width - seq_len;
        _lengths.push_back(seq_len);
        ids_tensor = torch::constant_pad_nd(ids_tensor,
                                            at::IntList{ 0, padding_size }, 0);
        mask_tensor = torch::constant_pad_nd(
            mask_tensor, at::IntList{ 0, padding_size }, 0);
        token_type_ids_tensor = torch::constant_pad_nd(
            token_type_ids_tensor, at::IntList{ 0, padding_size }, 0);
        at::Tensor position_ids = torch::arange((int)_width, at::kLong);

        std::vector<Tensor> target_vec;
        int target_val = static_cast<int>(tow->_target);

        if (target_val != -1)
          {
            Tensor target_tensor = torch::full(1, target_val, torch::kLong);
            target_vec.push_back(target_tensor);
          }

        if (_input_format == "bert")
          dataset.add_batch({ ids_tensor, token_type_ids_tensor, mask_tensor },
                            std::move(target_vec));
        else if (_input_format == "gpt2")
          {
            std::vector<Tensor> out_vec{ ids_tensor.slice(0, 1) };
            out_vec.push_back(torch::full(1, last_token, torch::kLong));
            target_vec.insert(target_vec.begin(), torch::cat(out_vec, 0));
            dataset.add_batch({ ids_tensor, position_ids },
                              std::move(target_vec));
          }
        _ndbed++;
      }
  }

  void CSVTSTorchInputFileConn::set_datadim(bool is_test_data)
  {
    if ((_forecast_timesteps < 0 || _backcast_timesteps < 0) && _timesteps < 0)
      {
        std::string errmsg
            = "no value given to [forecast_|backcast_|]timesteps";
        this->_logger->error(errmsg);
        throw InputConnectorBadParamException(errmsg);
      }

    if (_train && _ntargets != _label.size())
      {
        _logger->warn(
            "something went wrong in ntargets, computed  "
            + std::to_string(_ntargets) + " at service creation time, and "
            + std::to_string(_label.size()) + " at data processing time");
        throw InputConnectorBadParamException(
            "something went wrong in ntargets, computed  "
            + std::to_string(_ntargets) + " at service creation time, and "
            + std::to_string(_label.size()) + " at data processing time");
      }

    if (_datadim != -1)
      return;
    if (is_test_data)
      _datadim = _csvtsdata_test[0][0]._v.size();
    else
      _datadim = _csvtsdata[0][0]._v.size();

    if (_ntargets == 0 && _forecast_timesteps > 0)
      _ntargets = _datadim;

    std::vector<int> lpos = _label_pos;
    _label_pos.clear();
    for (int lp : lpos)
      if (lp != -1)
        _label_pos.push_back(lp);

    _logger->info("whole data dimension : " + std::to_string(_datadim));

    std::string ign;
    for (std::string i : _ignored_columns)
      ign += "'" + i + "' ";
    _logger->info(std::to_string(_ignored_columns.size())
                  + " ignored colums asked for: " + ign);

    std::string labels;
    for (std::string l : _label)
      labels += "'" + l + "' ";
    _logger->info(std::to_string(_label.size())
                  + " labels (outputs) asked for: " + labels);

    std::vector<std::string> col_vec;
    for (std::string c : _columns)
      col_vec.push_back(c);

    std::string labels_found;
    for (auto i : _label_pos)
      labels_found += "'" + col_vec.at(i) + "' ";
    _logger->info(std::to_string(_label_pos.size())
                  + " labels (outputs) found: " + labels_found);

    std::string inputs_found;
    for (unsigned int i = 0; i < _columns.size(); ++i)
      {
        if (std::find(_label_pos.begin(), _label_pos.end(), i)
                == _label_pos.end()
            && std::find(_ignored_columns.begin(), _ignored_columns.end(),
                         col_vec[i])
                   == _ignored_columns.end())
          inputs_found += "'" + col_vec[i] + "' ";
      }

    _logger->info(std::to_string(_datadim - _label_pos.size())
                  + " inputs  : " + inputs_found);
  }

  void CSVTSTorchInputFileConn::transform(const APIData &ad)
  {
    APIData ad_input = ad.getobj("parameters").getobj("input");

    init(ad_input);

    try
      {
        CSVTSInputFileConn::transform(ad);
        set_datadim();
      }
    catch (std::exception &e)
      {
        throw;
      }

    if (_train)
      {
        fill_dataset(_dataset, false);
        _csvtsdata.clear();
        fill_dataset(_test_dataset, true);
        _csvtsdata_test.clear();
      }
    else
      {
        // in test mode, prevent connector to create different series based on
        // offset
        if (_timesteps > 0)
          _offset = _timesteps;
        else
          _offset = _backcast_timesteps + _forecast_timesteps;
        fill_dataset(_dataset, false);
        _csvtsdata.clear();
        _csvtsdata_test.clear();
      }
  }

  void CSVTSTorchInputFileConn::add_data_instance_forecast(
      const unsigned long int tstart, const int vecindex,
      TorchDataset &dataset, const std::vector<CSVline> &seq)
  {
    std::vector<at::Tensor> data_sequence;
    if (_fnames.size() > static_cast<unsigned int>(vecindex))
      _ids.push_back(_fnames[vecindex] + " #" + std::to_string(tstart) + "_"
                     + std::to_string(tstart + _forecast_timesteps
                                      + _backcast_timesteps - 1));
    for (size_t ti = tstart; ti < tstart + _backcast_timesteps; ++ti)
      {
        std::vector<double> datavec;
        for (int di = 0; di < this->_datadim; ++di)
          datavec.push_back(seq[ti]._v[di]);
        at::Tensor data
            = torch::from_blob(&datavec[0], at::IntList{ _datadim },
                               torch::kFloat64)
                  .clone()
                  .to(torch::kFloat32);
        data_sequence.push_back(data);
      }
    at::Tensor dst = torch::stack(data_sequence);

    if (seq.size() >= _backcast_timesteps + _forecast_timesteps + tstart)
      {
        std::vector<at::Tensor> pred_sequence;
        for (size_t ti = tstart + _backcast_timesteps;
             ti < tstart + _backcast_timesteps + _forecast_timesteps; ++ti)
          {
            std::vector<double> predvec;
            for (int di = 0; di < this->_datadim; ++di)
              predvec.push_back(seq[ti]._v[di]);
            at::Tensor pred
                = torch::from_blob(&predvec[0], at::IntList{ _datadim },
                                   torch::kFloat64)
                      .clone()
                      .to(torch::kFloat32);
            pred_sequence.push_back(pred);
          }
        at::Tensor pst = torch::stack(pred_sequence);
        dataset.add_batch({ dst }, { pst });
      }
    else // we are in inference mode, not forecast available
      dataset.add_batch({ dst }, {});
  }

  void CSVTSTorchInputFileConn::discard_warn(int vecindex,
                                             unsigned int seq_size, bool test)
  {
    std::string errmsg;
    if (_timesteps > 0)
      errmsg = "data does not contains enough timesteps, "
               "discarding (seq_size: "
               + std::to_string(seq_size)
               + "  timesteps: " + std::to_string(_timesteps) + " )";
    else
      errmsg = "data does not contains enough timesteps, "
               "discarding (seq_size: "
               + std::to_string(seq_size)
               + "  backcast_timesteps: " + std::to_string(_backcast_timesteps)
               + "   forecast_timesteps: "
               + std::to_string(_forecast_timesteps) + " )";
    if (test)
      {
        if (static_cast<unsigned int>(vecindex) < _test_fnames.size())
          errmsg = "file " + _test_fnames[vecindex]
                   + " does not contains enough timesteps, "
                     "discarding";
      }
    else
      {
        if (static_cast<unsigned int>(vecindex) < _fnames.size())
          errmsg = "file " + _fnames[vecindex]
                   + " does not contains enough timesteps, "
                     "discarding";
      }
    _tilogger->warn(errmsg);
  }

  void CSVTSTorchInputFileConn::fill_dataset_forecast(TorchDataset &dataset,
                                                      bool test)
  {
    std::vector<std::vector<CSVline>> &data
        = test ? this->_csvtsdata_test : this->_csvtsdata;

    int vecindex = -1;

    for (const std::vector<CSVline> &seq : data)
      {
        vecindex++;
        long int tstart = 0;
        if (static_cast<long int>(seq.size())
            < _backcast_timesteps + _forecast_timesteps)
          {
            discard_warn(vecindex, seq.size(), test);
            continue;
          }
        for (; tstart + _backcast_timesteps + _forecast_timesteps
               < static_cast<long int>(seq.size());
             tstart += _offset)
          add_data_instance_forecast(tstart, vecindex, dataset, seq);
        if (tstart < static_cast<long int>(seq.size()) - 1)
          add_data_instance_forecast(seq.size() - _backcast_timesteps
                                         - _forecast_timesteps,
                                     vecindex, dataset, seq);
      }
  }

  void
  CSVTSTorchInputFileConn::add_seq(const size_t ti,
                                   const std::vector<CSVline> &seq,
                                   std::vector<at::Tensor> &data_sequence,
                                   std::vector<at::Tensor> &label_sequence)
  {
    std::vector<double> datavec;
    std::vector<double> labelvec;
    size_t label_size = _label_pos.size();
    size_t data_size = _datadim - label_size;

    for (size_t li = 0; li < label_size; ++li)
      labelvec.push_back(seq[ti]._v[_label_pos[li]]);
    for (int di = 0; di < this->_datadim; ++di)
      if (std::find(_label_pos.begin(), _label_pos.end(), di)
          == _label_pos.end())
        datavec.push_back(seq[ti]._v[di]);

    at::Tensor data
        = torch::from_blob(&datavec[0],
                           at::IntList{ static_cast<long int>(data_size) },
                           torch::kFloat64)
              .clone()
              .to(torch::kFloat32);
    at::Tensor label
        = torch::from_blob(&labelvec[0],
                           at::IntList{ static_cast<long int>(label_size) },
                           torch::kFloat64)
              .clone()
              .to(torch::kFloat32);
    data_sequence.push_back(data);
    label_sequence.push_back(label);
  }

  void CSVTSTorchInputFileConn::add_data_instance_labels(
      const unsigned long int tstart, const int vecindex,
      TorchDataset &dataset, const std::vector<CSVline> &seq,
      const size_t seq_len)
  {
    unsigned int label_size = _label_pos.size();
    if (static_cast<int>(label_size) >= _datadim)
      {
        std::string errmsg
            = "label_size (output dim) " + std::to_string(label_size)
              + " is larger than datadim " + std::to_string(_datadim)
              + " leading to invalid input dim";
        this->_logger->error(errmsg);
        throw InputConnectorBadParamException(errmsg);
      }
    std::vector<at::Tensor> data_sequence;
    std::vector<at::Tensor> label_sequence;
    if (_fnames.size() > static_cast<unsigned int>(vecindex))
      _ids.push_back(_fnames[vecindex] + " #" + std::to_string(tstart) + "_"
                     + std::to_string(tstart + seq_len - 1));

    for (size_t ti = tstart;
         ti < static_cast<unsigned long int>(tstart + seq_len); ++ti)
      add_seq(ti, seq, data_sequence, label_sequence);

    at::Tensor dst = torch::stack(data_sequence);
    at::Tensor lst = torch::stack(label_sequence);
    dataset.add_batch({ dst }, { lst });
  }

  void CSVTSTorchInputFileConn::fill_dataset_labels(TorchDataset &dataset,
                                                    bool test)
  {
    int vecindex = -1;
    std::vector<std::vector<CSVline>> &data
        = test ? this->_csvtsdata_test : this->_csvtsdata;

    if (_train)
      {
        for (const std::vector<CSVline> &seq : data)
          {
            vecindex++;
            long int tstart = 0;
            if (static_cast<long int>(seq.size()) < _timesteps)
              {
                discard_warn(vecindex, seq.size(), test);
                continue;
              }
            for (; tstart + _timesteps < static_cast<long int>(seq.size());
                 tstart += _offset)
              add_data_instance_labels(tstart, vecindex, dataset, seq,
                                       static_cast<unsigned int>(_timesteps));
            if (tstart < static_cast<long int>(seq.size()) - 1)
              add_data_instance_labels(seq.size() - _timesteps, vecindex,
                                       dataset, seq,
                                       static_cast<unsigned int>(_timesteps));
          }
      }
    else // do not split
      for (const std::vector<CSVline> &seq : data)
        {
          vecindex++;
          add_data_instance_labels(0, vecindex, dataset, seq, seq.size());
        }
  }

  void CSVTSTorchInputFileConn::fill_dataset(TorchDataset &dataset,
                                             bool use_csvtsdata_test)
  {
    _ids.clear();
    // we have _csvtsdata and csvtsdata_test to put into TorchDataset
    // _dataset , _test_dataset

    if (_forecast_timesteps != -1)
      fill_dataset_forecast(dataset, use_csvtsdata_test);
    else
      fill_dataset_labels(dataset, use_csvtsdata_test);
    dataset.reset();
  }
}
