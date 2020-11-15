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
  void TorchDataset::finalize_db()
  {
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

  void TorchDataset::pop(int64_t index, std::string &data, std::string &target)
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

    auto it = _indices.begin();
    while (it != _indices.end())
      {
        if (*it == index)
          {
            _indices.erase(it);
            break;
          }
        it++;
      }
  }

  void TorchDataset::add_elt(int64_t index, std::string data,
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
    _indices.push_back(index);
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

  void TorchDataset::add_batch(const std::vector<at::Tensor> &data,
                               const std::vector<at::Tensor> &target)
  {
    if (!_db)
      _batches.push_back(TorchBatch(data, target));
    else
      write_tensors_to_db(data, target);
  }

  void TorchDataset::reset(bool shuffle, db::Mode dbmode)
  {
    _shuffle = shuffle;
    if (!_db)
      {
        _indices.clear();

        for (unsigned int i = 0; i < _batches.size(); ++i)
          {
            _indices.push_back(i);
          }
      }
    else // below db case
      {
        _indices.clear();
        if (_dbData == nullptr)
          {
            _dbData = std::shared_ptr<db::DB>(db::GetDB(_backend));
            _dbData->Open(_dbFullName, dbmode);
          }

        db::Cursor *cursor = _dbData->NewCursor();
        while (cursor->valid())
          {
            std::string key = cursor->key();
            size_t pos = key.find("_data");
            if (pos != std::string::npos)
              {
                std::string sid = key.substr(0, pos);
                int64_t id = std::stoll(sid);
                _indices.push_back(id);
              }
            cursor->Next();
          }
        delete (cursor);
      }

    if (_shuffle)
      {
        auto seed = _seed == -1 ? static_cast<long>(time(NULL)) : _seed;
        std::shuffle(_indices.begin(), _indices.end(), std::mt19937(seed));
      }
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
    bool first_iter = true;

    if (!_db)
      {
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

            for (unsigned int i = 0; i < data.size(); ++i)
              {
                data[i].push_back(entry.data.at(i));
              }
            for (unsigned int i = 0; i < target.size(); ++i)
              {
                target[i].push_back(entry.target.at(i));
              }

            _indices.pop_back();
            count--;
          }
      }
    else // below db case
      {
        while (count != 0)
          {
            auto id = _indices.back();
            std::stringstream data_key;
            std::stringstream target_key;
            data_key << id << "_data";
            target_key << id << "_target";

            std::string targets;
            std::string datas;
            _dbData->Get(data_key.str(), datas);
            _dbData->Get(target_key.str(), targets);
            std::stringstream datastream(datas);
            std::stringstream targetstream(targets);
            std::vector<torch::Tensor> d;
            std::vector<torch::Tensor> t;
            torch::load(d, datastream);
            torch::load(t, targetstream);

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
  int TorchDataset::add_image_file(const std::string &fname, const int &target,
                                   const int &height, const int &width)
  {
    ImgTorchInputFileConn *inputc
        = reinterpret_cast<ImgTorchInputFileConn *>(_inputc);

    DDImg dimg;
    inputc->copy_parameters_to(dimg);

    try
      {
        if (dimg.read_file(fname))
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
        at::Tensor imgt = image_to_tensor(dimg._imgs[0], height, width);
        // at::Tensor targett{ torch::full(1, target, torch::kLong) };
        at::Tensor targett = target_to_tensor(target);

        add_batch({ imgt }, { targett });
        return 0;
      }
    else
      {
        return -1;
      }
  }

  int TorchDataset::add_image_file(const std::string &fname,
                                   const std::vector<double> &target,
                                   const int &height, const int &width)
  {
    ImgTorchInputFileConn *inputc
        = reinterpret_cast<ImgTorchInputFileConn *>(_inputc);

    DDImg dimg;
    inputc->copy_parameters_to(dimg);

    try
      {
        if (dimg.read_file(fname))
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
        at::Tensor imgt = image_to_tensor(dimg._imgs[0], height, width);
        // at::Tensor targett{ torch::full(1, target, torch::kLong) };
        at::Tensor targett = target_to_tensor(target);

        add_batch({ imgt }, { targett });
        return 0;
      }
    else
      {
        return -1;
      }
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
      imgt[0][m] = imgt[0][m].sub_(inputc->_mean.at(m));

    if (!inputc->_std.empty() && inputc->_std.size() != nchannels)
      throw InputConnectorBadParamException(
          "std vector be of size the number of channels ("
          + std::to_string(nchannels) + ")");

    for (size_t s = 0; s < inputc->_std.size(); s++)
      imgt[0][s] = imgt[0][s].div_(inputc->_std.at(s));

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
}
