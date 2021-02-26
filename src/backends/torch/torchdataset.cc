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

#include "torchdataset.h"
#include "torchinputconns.h"

namespace dd
{
  void TorchDataset::db_finalize()
  {
    if (!_db)
      return;
    if (_current_index % _batches_per_transaction != 0)
      {
        _txn->Commit();
        _logger->info("Put {} tensors in db", _current_index);
      }
    if (_dbData != nullptr)
      {
        _dbData->Close();
        _txn.reset();
      }
    _dbData = nullptr;
    _current_index = 0;
  }

  void TorchDataset::pop_db_elt(int64_t index, std::string &data,
                                std::string &target)
  {
    if (_dbData == nullptr)
      {
        _dbData = std::shared_ptr<db::DB>(db::GetDB(_backend));
        _dbData->Open(_dbFullName, db::WRITE);
      }
    std::stringstream data_key;
    std::stringstream target_key;

    data_key << std::to_string(index) << "_data";
    target_key << std::to_string(index) << "_target";

    _dbData->Get(data_key.str(), data);
    _dbData->Get(target_key.str(), target);

    _dbData->Remove(data_key.str());
    _dbData->Remove(target_key.str());
  }

  void TorchDataset::add_db_elt(int64_t index, std::string data,
                                std::string target)
  {
    if (_dbData == nullptr)
      {
        _dbData = std::shared_ptr<db::DB>(db::GetDB(_backend));
        _dbData->Open(_dbFullName, db::NEW);
        _txn = std::shared_ptr<db::Transaction>(_dbData->NewTransaction());
      }
    std::stringstream data_key;
    std::stringstream target_key;

    data_key << std::to_string(index) << "_data";
    target_key << std::to_string(index) << "_target";
    _txn->Put(data_key.str(), data);
    _txn->Put(target_key.str(), target);
    _txn->Commit();
    _txn.reset(_dbData->NewTransaction());
  }

  void TorchDataset::write_tensors_to_db(const std::vector<at::Tensor> &data,
                                         const std::vector<at::Tensor> &target)
  {
    std::ostringstream dstream;
    torch::save(data, dstream);
    std::ostringstream tstream;
    torch::save(target, tstream);

    if (_dbData == nullptr)
      {
        _dbData = std::shared_ptr<db::DB>(db::GetDB(_backend));
        _dbData->Open(_dbFullName, db::NEW);
        _txn = std::shared_ptr<db::Transaction>(_dbData->NewTransaction());
      }

    std::stringstream data_key;
    std::stringstream target_key;

    data_key << std::to_string(_current_index) << "_data";
    target_key << std::to_string(_current_index) << "_target";

    _txn->Put(data_key.str(), dstream.str());
    _txn->Put(target_key.str(), tstream.str());

    // should not commit transactions every time;
    if (++_current_index % _batches_per_transaction == 0)
      {
        _txn->Commit();
        _txn.reset(_dbData->NewTransaction());
        _logger->info("Put {} tensors in db", _current_index);
      }
  }

  void
  TorchDataset::write_image_to_db(const cv::Mat &bgr,
                                  const std::vector<torch::Tensor> &target)
  {
    // serialize image
    std::stringstream dstream;
    std::vector<uint8_t> buffer;
    std::vector<int> param = { cv::IMWRITE_JPEG_QUALITY, 100 };
    cv::imencode(".jpg", bgr, buffer, param);
    for (uint8_t c : buffer)
      dstream << c;

    // serialize target
    std::ostringstream tstream;
    torch::save(target, tstream);

    // check on db
    if (_dbData == nullptr)
      {
        _dbData = std::shared_ptr<db::DB>(db::GetDB(_backend));
        _dbData->Open(_dbFullName, db::NEW);
        _txn = std::shared_ptr<db::Transaction>(_dbData->NewTransaction());
        _logger->info("Preparing db of {}x{} images", bgr.cols, bgr.rows);
      }

    // data & target keys
    std::stringstream data_key;
    std::stringstream target_key;
    data_key << std::to_string(_current_index) << "_data";
    target_key << std::to_string(_current_index) << "_target";

    // store into db
    _txn->Put(data_key.str(), dstream.str());
    _txn->Put(target_key.str(), tstream.str());

    // should not commit transactions every time;
    if (++_current_index % _batches_per_transaction == 0)
      {
        _txn->Commit();
        _txn.reset(_dbData->NewTransaction());
        _logger->info("Put {} images in db", _current_index);
      }
  }

  void TorchDataset::read_image_from_db(const std::string &datas,
                                        const std::string &targets,
                                        cv::Mat &bgr,
                                        std::vector<torch::Tensor> &targett,
                                        const bool &bw)
  {
    std::vector<uint8_t> img_data(datas.begin(), datas.end());
    bgr = cv::Mat(img_data, true);
    bgr = cv::imdecode(bgr,
                       bw ? CV_LOAD_IMAGE_GRAYSCALE : CV_LOAD_IMAGE_COLOR);
    std::stringstream targetstream(targets);
    torch::load(targett, targetstream);
  }

  // add image batch
  void TorchDataset::add_image_batch(const cv::Mat &bgr, const int &width,
                                     const int &height,
                                     const std::vector<at::Tensor> &targett)
  {
    if (!_db)
      {
        // to tensor
        at::Tensor imgt = image_to_tensor(bgr, height, width);
        add_batch({ imgt }, targett);
      }
    else
      {
        // write to db
        write_image_to_db(bgr, targett);
      }
  }

  void TorchDataset::add_batch(const std::vector<at::Tensor> &data,
                               const std::vector<at::Tensor> &target)
  {
    if (!_db)
      _batches.push_back(TorchBatch(data, target));
    else // db
      write_tensors_to_db(data, target);
  }

  void TorchDataset::reset(bool shuffle, db::Mode dbmode)
  {
    _shuffle = shuffle;
    if (!_db)
      {
        if (!_lfiles.empty()) // list of files
          {
            _indices = std::vector<int64_t>(_lfiles.size());
            std::iota(std::begin(_indices), std::end(_indices), 0);
          }
        else if (!_batches.empty())
          {
            _indices = std::vector<int64_t>(_batches.size());
            std::iota(std::begin(_indices), std::end(_indices), 0);
          }
        else
          {
            _indices.clear();
          }
      }
    else // below db case
      {
        if (!_dbData)
          {
            _dbData = std::shared_ptr<db::DB>(db::GetDB(_backend));
            _dbData->Open(_dbFullName, dbmode);
          }

        if (!_dbCursor)
          _dbCursor = _dbData->NewCursor();

        _indices = std::vector<int64_t>(_dbData->Count());
        std::iota(std::begin(_indices), std::end(_indices), 0);
      }

    if (_shuffle)
      {
        std::shuffle(_indices.begin(), _indices.end(), _rng);
      }
  }

  std::vector<long int> TorchDataset::targetsize(long int i) const
  {
    if (!_db)
      return _batches[0].target[i].sizes().vec();

    auto id = _indices.back();
    std::stringstream target_key;
    target_key << id << "_target";
    std::string targets;
    _dbData->Get(target_key.str(), targets);
    std::stringstream targetstream(targets);
    std::vector<torch::Tensor> t;
    torch::load(t, targetstream);

    return t.at(i).sizes().vec();
  }

  std::vector<long int> TorchDataset::datasize(long int i) const
  {
    if (!_db)
      return _batches[0].data[i].sizes().vec();

    auto id = _indices.back();
    std::stringstream data_key;
    data_key << id << "_data";
    std::string datas;
    _dbData->Get(data_key.str(), datas);
    std::stringstream datastream(datas);
    std::vector<torch::Tensor> d;
    torch::load(d, datastream);

    return d.at(i).sizes().vec();
  }

  // `request` holds the size of the batch
  // Data selection and batch construction are done in this method
  c10::optional<TorchBatch> TorchDataset::get_batch(BatchRequestType request)
  {
    size_t count = request[0];
    std::vector<torch::Tensor> data_tensors;
    std::vector<torch::Tensor> target_tensors;

    count = count < _indices.size() ? count : _indices.size();

    if (count == 0)
      {
        return torch::nullopt;
      }

    typedef std::vector<torch::Tensor> BatchToStack;
    std::vector<BatchToStack> data, target;

    if (!_db) // Note: no data augmentation if no db
      {
        if (!_lfiles.empty()) // prefetch batch from file list
          {
            ImgTorchInputFileConn *inputc
                = reinterpret_cast<ImgTorchInputFileConn *>(_inputc);

            size_t nlfiles = 0;
            while (nlfiles < count)
              {
                auto id = _indices.back();
                auto lfile = _lfiles.at(id);
                if (_classification)
                  add_image_file(lfile.first,
                                 static_cast<int>(lfile.second.at(0)),
                                 inputc->height(), inputc->width());
                else // vector generic type, including regression
                  add_image_file(lfile.first, lfile.second, inputc->height(),
                                 inputc->width());
                ++nlfiles;
                _indices.pop_back();
              }

            if (!_batches.empty())
              {
                auto entry = _batches[0];
                data.resize(entry.data.size());
                target.resize(entry.target.size());
              }

            for (size_t id = 0; id < count; ++id)
              {
                auto entry = _batches[id];

                for (unsigned int i = 0; i < entry.data.size(); ++i)
                  {
                    data[i].push_back(entry.data.at(i));
                  }
                for (unsigned int i = 0; i < entry.target.size(); ++i)
                  {
                    target[i].push_back(entry.target.at(i));
                  }
              }
            _batches.clear();
          }
        else // batches
          {
            bool first_iter = true;

            while (count != 0)
              {
                auto id = _indices.back();
                auto entry = _batches[id];

                if (first_iter)
                  {
                    data.resize(entry.data.size());
                    target.resize(entry.target.size());
                    first_iter = false;
                  }

                for (unsigned int i = 0; i < entry.data.size(); ++i)
                  {
                    data[i].push_back(entry.data.at(i));
                  }
                for (unsigned int i = 0; i < entry.target.size(); ++i)
                  {
                    target[i].push_back(entry.target.at(i));
                  }

                _indices.pop_back();
                count--;
              }
          }
      }
    else // below db case
      {
        while (count != 0)
          {
            std::stringstream data_key;
            std::stringstream target_key;

            if (!_dbCursor->valid())
              {
                delete _dbCursor;
                _dbCursor = _dbData->NewCursor();
              }
            std::string key = _dbCursor->key();
            size_t pos = key.find("_data");
            if (pos != std::string::npos)
              {
                data_key << key;
                std::string sid = key.substr(0, pos);
                target_key << sid << "_target";
              }
            else // skip targets
              {
                _dbCursor->Next();
                continue;
              }

            std::string targets;
            std::string datas;
            _dbData->Get(data_key.str(), datas);
            _dbData->Get(target_key.str(), targets);
            _dbCursor->Next();

            std::vector<torch::Tensor> d;
            std::vector<torch::Tensor> t;

            if (!_image)
              {
                std::stringstream datastream(datas);
                std::stringstream targetstream(targets);
                torch::load(d, datastream);
                torch::load(t, targetstream);
              }
            else
              {
                ImgTorchInputFileConn *inputc
                    = reinterpret_cast<ImgTorchInputFileConn *>(_inputc);

                cv::Mat bgr;
                torch::Tensor targett;
                read_image_from_db(datas, targets, bgr, t, inputc->_bw);

                // data augmentation can apply here, with OpenCV
                _img_rand_aug_cv.augment(bgr);

                torch::Tensor imgt
                    = image_to_tensor(bgr, inputc->height(), inputc->width());

                d.push_back(imgt);
              }

            for (unsigned int i = 0; i < d.size(); ++i)
              {
                while (i >= data.size())
                  data.emplace_back();
                data[i].push_back(d.at(i));
              }
            for (unsigned int i = 0; i < t.size(); ++i)
              {
                while (i >= target.size())
                  target.emplace_back();
                target[i].push_back(t.at(i));
              }

            _indices.pop_back();
            count--;
          }
      }

    for (const auto &vec : data)
      data_tensors.push_back(torch::stack(vec));

    for (const auto &vec : target)
      target_tensors.push_back(torch::stack(vec));

    return TorchBatch{ data_tensors, target_tensors };
  }

  TorchBatch TorchDataset::get_cached()
  {
    reset();
    auto batch = get_batch({ cache_size() });
    if (!batch)
      throw InputConnectorInternalException("No data provided");
    return batch.value();
  }

  TorchDataset TorchDataset::split(double start, double stop)
  {
    auto datasize = _batches.size();
    auto start_it = _batches.begin() + static_cast<int64_t>(datasize * start);
    auto stop_it
        = _batches.end() - static_cast<int64_t>(datasize * (1 - stop));

    TorchDataset new_dataset;
    new_dataset._batches.insert(new_dataset._batches.end(), start_it, stop_it);
    return new_dataset;
  }

  /*-- image tools --*/
  int TorchDataset::add_image_file(const std::string &fname,
                                   const std::vector<at::Tensor> &target,
                                   const int &height, const int &width)
  {
    ImgTorchInputFileConn *inputc
        = reinterpret_cast<ImgTorchInputFileConn *>(_inputc);

    DDImg dimg;
    inputc->copy_parameters_to(dimg);

    try
      {
        if (dimg.read_file(fname, -1))
          {
            this->_logger->error("Uri failed: {}", fname);
          }
      }
    catch (std::exception &e)
      {
        this->_logger->error("Uri failed: {}", fname);
      }
    if (dimg._imgs.size() != 0)
      {
        add_image_batch(dimg._imgs[0], height, width, target);
        return 0;
      }
    else
      {
        return -1;
      }
  }

  int TorchDataset::add_image_file(const std::string &fname, const int &target,
                                   const int &height, const int &width)
  {
    return add_image_file(fname, { target_to_tensor(target) }, height, width);
  }

  int TorchDataset::add_image_file(const std::string &fname,
                                   const std::vector<double> &target,
                                   const int &height, const int &width)
  {
    return add_image_file(fname, { target_to_tensor(target) }, height, width);
  }

  at::Tensor TorchDataset::image_to_tensor(const cv::Mat &bgr,
                                           const int &height, const int &width)
  {
    ImgTorchInputFileConn *inputc
        = reinterpret_cast<ImgTorchInputFileConn *>(_inputc);

    std::vector<int64_t> sizes{ height, width, bgr.channels() };
    at::TensorOptions options(at::ScalarType::Byte);

    at::Tensor imgt = torch::from_blob(bgr.data, at::IntList(sizes), options);
    imgt = imgt.toType(at::kFloat).permute({ 2, 0, 1 });
    size_t nchannels = imgt.size(0);

    if (inputc->_scale != 1.0)
      imgt = imgt.mul(inputc->_scale);

    if (!inputc->_mean.empty() && inputc->_mean.size() != nchannels)
      throw InputConnectorBadParamException(
          "mean vector be of size the number of channels ("
          + std::to_string(nchannels) + ")");

    for (size_t m = 0; m < inputc->_mean.size(); m++)
      imgt[m] = imgt[m].sub_(inputc->_mean.at(m));

    if (!inputc->_std.empty() && inputc->_std.size() != nchannels)
      throw InputConnectorBadParamException(
          "std vector be of size the number of channels ("
          + std::to_string(nchannels) + ")");

    for (size_t s = 0; s < inputc->_std.size(); s++)
      imgt[s] = imgt[s].div_(inputc->_std.at(s));

    return imgt;
  }

  at::Tensor TorchDataset::target_to_tensor(const int &target)
  {
    at::Tensor targett{ torch::full(1, target, torch::kLong) };
    return targett;
  }

  at::Tensor TorchDataset::target_to_tensor(const std::vector<double> &target)
  {
    int64_t tsize = target.size();

    at::Tensor targett = torch::zeros(tsize, torch::kFloat32);
    int n = 0;
    for (auto i : target) // XXX: from_blob does not seem to work, fills up
                          // with spurious values
      {
        targett[n] = i;
        ++n;
      }
    return targett;
  }

  void TorchMultipleDataset::set_list(
      const std::vector<
          std::vector<std::pair<std::string, std::vector<double>>>> &lsfiles)
  {
    for (size_t i = 0; i < lsfiles.size(); ++i)
      _datasets[i].set_list(lsfiles[i]);
  }

  void TorchMultipleDataset::add_tests_names(
      const std::vector<std::string> &longnames)
  {
    _datasets.resize(_datasets.size() + longnames.size());
    _datasets_names.resize(_datasets_names.size() + longnames.size());
    _dbFullNames.resize(_dbFullNames.size() + longnames.size());
    for (size_t i = 0; i < longnames.size(); ++i)
      {
        _datasets_names[_datasets_names.size() - longnames.size() + i]
            = fileops::shortname(longnames[i]);
        set_db_name(_dbFullNames.size() - longnames.size() + i, longnames[i]);
        init_set(_datasets.size() - longnames.size() + i);
      }
  }

  void TorchMultipleDataset::add_test_name_if_necessary(std::string longname,
                                                        int test_id)
  {
    std::string name = fileops::shortname(longname);
    if (_datasets.size() >= static_cast<size_t>(test_id + 1)
        && _datasets_names[test_id] == name)
      return;
    if (_datasets.size() >= static_cast<size_t>(test_id + 1)
        && _datasets_names[test_id] != name)
      {
        std::string msg = "mismatch in adding test sets on the fly: test_id is"
                          + std::to_string(test_id) + ", new name is " + name
                          + ", old name is " + _datasets_names[test_id];
        this->_logger->error(msg);
        throw InputConnectorInternalException(msg);
      }
    add_test_name(longname);
  }

  void TorchMultipleDataset::add_test_name(std::string longname)
  {
    std::string name = fileops::shortname(longname);
    _datasets.resize(_datasets.size() + 1);
    _datasets_names.resize(_datasets_names.size() + 1);
    _datasets_names[_datasets_names.size() - 1] = name;
    if (_db)
      {
        _dbFullNames.resize(_dbFullNames.size() + 1);
        set_db_name(_dbFullNames.size() - 1);
      }
    init_set(_datasets.size() - 1);
  }

  void TorchMultipleDataset::add_db_name(std::string dblongname)
  {
    std::string dbname = fileops::shortname(dblongname);
    _datasets.resize(_datasets.size() + 1);
    _datasets_names.resize(_datasets_names.size() + 1);
    if (!_db)
      {
        throw InputConnectorBadParamException(
            "trying to add a db name while dataset is not of type db");
      }
    _dbFullNames.resize(_dbFullNames.size() + 1);
    _dbFullNames[_dbFullNames.size() - 1] = dblongname;
    test_name_from_db_name(_datasets.size() - 1);
    init_set(_datasets.size() - 1);
  }
}
