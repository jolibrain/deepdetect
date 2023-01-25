/**
 * DeepDetect
 * Copyright (c) 2019-2021 Jolibrain
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

  void TorchDataset::image_to_stringstream(const cv::Mat &img,
                                           std::ostringstream &dstream,
                                           const bool &lossless)
  {
    std::vector<uint8_t> buffer;
    std::vector<int> param;
    std::string ext;
    if (!lossless)
      {
        ext = ".jpg";
        param = { cv::IMWRITE_JPEG_QUALITY, 100 };
      }
    else
      {
        ext = ".png";
        param = { cv::IMWRITE_PNG_COMPRESSION, 1 };
      }
    cv::imencode(ext, img, buffer, param);
    for (uint8_t c : buffer)
      dstream << c;
  }

  void
  TorchDataset::write_image_to_db(const cv::Mat &bgr,
                                  const std::vector<torch::Tensor> &target)
  {
    // serialize image
    std::ostringstream dstream;
    image_to_stringstream(bgr, dstream, true);

    // serialize target
    std::ostringstream tstream;
    torch::save(target, tstream);

#pragma omp ordered
    write_image_to_db(dstream, tstream, bgr.rows, bgr.cols);
  }

  void TorchDataset::write_image_to_db(const cv::Mat &bgr,
                                       const cv::Mat &bw_target)
  {
    // serialize image
    std::ostringstream dstream;
    image_to_stringstream(bgr, dstream, false);

    // serialize target
    std::ostringstream tstream;
    image_to_stringstream(bw_target, tstream, true);

#pragma omp ordered
    write_image_to_db(dstream, tstream, bgr.rows, bgr.cols);
  }

  void TorchDataset::write_image_to_db(const std::ostringstream &dstream,
                                       const std::ostringstream &tstream,
                                       const int &height, const int &width)
  {
    // check on db
    if (_dbData == nullptr)
      {
        _dbData = std::shared_ptr<db::DB>(db::GetDB(_backend));
        _dbData->Open(_dbFullName, db::NEW);
        _txn = std::shared_ptr<db::Transaction>(_dbData->NewTransaction());
        _logger->info("Preparing db of {}x{} images", width, height);
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
                                        cv::Mat &bw_target, const bool &bw,
                                        const int &width, const int &height)
  {
    std::vector<uint8_t> img_data(datas.begin(), datas.end());
    bgr = cv::Mat(img_data, true);
    bgr = cv::imdecode(bgr,
                       bw ? CV_LOAD_IMAGE_GRAYSCALE : CV_LOAD_IMAGE_COLOR);

    if (_segmentation)
      {
        std::vector<uint8_t> img_target_data(targets.begin(), targets.end());
        bw_target = cv::Mat(img_target_data, true);
        bw_target = cv::imdecode(bw_target, CV_LOAD_IMAGE_GRAYSCALE);
      }
    else
      {
        std::stringstream targetstream(targets);
        torch::load(targett, targetstream);
      }

    if (bgr.cols != width || bgr.rows != height)
      {
        cv::resize(bgr, bgr, cv::Size(width, height), 0, 0, cv::INTER_CUBIC);

        if (_bbox)
          {
            float w_ratio = static_cast<float>(width) / bgr.cols;
            float h_ratio = static_cast<float>(height) / bgr.rows;
            for (int bb = 0; bb < (int)targett[0].size(0); ++bb)
              {
                targett[0][bb][0] *= w_ratio;
                targett[0][bb][1] *= h_ratio;
                targett[0][bb][2] *= w_ratio;
                targett[0][bb][3] *= h_ratio;
              }
          }
        else if (_segmentation)
          {
            cv::resize(bw_target, bw_target, cv::Size(width, height), 0, 0,
                       cv::INTER_NEAREST);
          }
      }
  }

  // add image batch
  void TorchDataset::add_image_batch(const cv::Mat &bgr,
                                     const std::vector<at::Tensor> &targett)
  {
    if (!_db)
      {
        // to tensor
        at::Tensor imgt = image_to_tensor(bgr);
#pragma omp ordered
        add_batch({ imgt }, targett);
      }
    else
      {
        // write to db
        write_image_to_db(bgr, targett);
      }
  }

  // add image batch
  void TorchDataset::add_image_batch(const cv::Mat &bgr,
                                     const cv::Mat &bw_target)
  {
    if (!_db)
      {
        // to tensor
        at::Tensor imgt = image_to_tensor(bgr);
        at::Tensor imgt_tgt = image_to_tensor(bw_target, true);
#pragma omp ordered
        add_batch({ imgt }, { imgt_tgt });
      }
    else
      {
        // write to db
        write_image_to_db(bgr, bw_target);
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

  void TorchDataset::reset(db::Mode dbmode)
  {
    std::lock_guard<std::mutex> guard(_mutex);
    size_t data_size = 0;

    if (!_db)
      {
        if (!_lfiles.empty()) // list of files
          {
            data_size = _lfiles.size();
          }
        else if (!_batches.empty())
          {
            data_size = _batches.size();
          }
        else if (!_lfilesseg.empty())
          {
            data_size = _lfilesseg.size();
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

        data_size = _dbData->Count() / 2;
      }

    _indices.resize(data_size);
    std::iota(std::rbegin(_indices), std::rend(_indices), 0);
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

  void TorchDataset::dataaug_then_push_back(
      const cv::Mat &bgr, const std::vector<torch::Tensor> &t,
      const cv::Mat &bw_target, std::vector<BatchToStack> &data,
      std::vector<BatchToStack> &target)
  {
    int samples = 1;

    if (_test && _img_rand_aug_cv._crop_params._crop_size > 0)
      samples = _img_rand_aug_cv._crop_params._test_crop_samples;

    while (samples > 0)
      {
        cv::Mat bgr_sample = bgr.clone();
        cv::Mat bw_target_sample;
        std::vector<torch::Tensor> d_sample;
        std::vector<torch::Tensor> t_sample = t;
        if (_segmentation)
          bw_target_sample = bw_target.clone();

        // data augmentation can apply here, with OpenCV
        if (!_test)
          {
            if (_bbox)
              _img_rand_aug_cv.augment_with_bbox(bgr_sample, t_sample);
            else if (_segmentation)
              _img_rand_aug_cv.augment_with_segmap(bgr_sample,
                                                   bw_target_sample);
            else
              _img_rand_aug_cv.augment(bgr_sample);
          }
        else
          {
            // cropping requires test set 'augmentation'
            if (_bbox)
              {
                _img_rand_aug_cv.augment_test_with_bbox(bgr_sample, t_sample);
              }
            else if (_segmentation)
              _img_rand_aug_cv.augment_test_with_segmap(bgr_sample,
                                                        bw_target_sample);
            else
              _img_rand_aug_cv.augment_test(bgr_sample);
          }

        torch::Tensor imgt = image_to_tensor(bgr_sample);
        d_sample.push_back(imgt);

        if (_segmentation)
          {
            at::Tensor targett_seg = image_to_tensor(bw_target_sample, true);
            t_sample.push_back(targett_seg);
          }

        --samples;

        for (unsigned int i = 0; i < d_sample.size(); ++i)
          {
            while (i >= data.size())
              data.emplace_back();
            data.at(i).push_back(d_sample[i]);
          }
        for (unsigned int i = 0; i < t_sample.size(); ++i)
          {
            while (i >= target.size())
              target.emplace_back();
            target.at(i).push_back(t_sample[i]);
          }
      }
  }

  // `request` holds the size of the batch
  // Data selection and batch construction are done in this method
  c10::optional<TorchBatch> TorchDataset::get_batch(BatchRequestType request)
  {
    size_t count = request[0];

    std::vector<BatchToStack> data, target;

    if (!_db) // Note: no data augmentation if no db
      {
        std::vector<int64_t> ids;
        {
          std::lock_guard<std::mutex> guard(_mutex);
          count = count < _indices.size() ? count : _indices.size();

          if (count == 0)
            {
              return torch::nullopt;
            }

          // extract ids
          ids.reserve(count);

          while (count != 0)
            {
              auto id = _indices.back();
              ids.push_back(id);
              _indices.pop_back();
              --count;
            }
        }

        if (!_lfiles.empty()) // prefetch batch from file list
          {

            for (int64_t id : ids)
              {
                auto lfile = _lfiles.at(id);
                std::vector<torch::Tensor> targetts;
                if (_classification)
                  targetts.push_back(
                      target_to_tensor(static_cast<int>(lfile.second.at(0))));
                else // vector generic type, including regression
                  targetts.push_back(target_to_tensor(lfile.second));

                cv::Mat dimg;
                int res = read_image_file(lfile.first, dimg);
                if (res == 0)
                  {
                    cv::Mat timg; // unused
                    dataaug_then_push_back(dimg, targetts, timg, data, target);
                  }
                else
                  {
                    this->_logger->warn("Skip file {}: not found",
                                        lfile.first);
                  }
              }
          }
        else if (!_lfilesseg.empty()) // segmentation with no db
          {
            std::vector<torch::Tensor> t;

            for (int64_t id : ids)
              {
                auto lfile = _lfilesseg.at(id);

                cv::Mat dimg, timg;
                int res = read_image_file(lfile.first, dimg);
                int res2 = read_image_file(lfile.second, timg, true);
                if (res == 0 && res2 == 0)
                  {
                    dataaug_then_push_back(dimg, t, timg, data, target);
                  }
                else
                  {
                    if (res != 0)
                      this->_logger->warn("Skip file {}: not found",
                                          lfile.first);
                    if (res2 != 0)
                      this->_logger->warn("Skip file {}: not found",
                                          lfile.second);
                  }
              }
          }
        else // batches
          {
            bool first_iter = true;

            for (auto id : ids)
              {
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
              }
          }
      }
    else // below db case
      {
        bool has_data = false;

        while (count > 0)
          {
            std::stringstream data_key;
            std::stringstream target_key;

            std::string targets;
            std::string datas;

            {
              std::lock_guard<std::mutex> guard(_mutex);

              if (_indices.empty())
                // end of the dataset
                break;

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
              _dbData->Get(data_key.str(), datas);
              _dbData->Get(target_key.str(), targets);
              _dbCursor->Next();

              --count;
              _indices.pop_back();
              has_data = true;
            }

            // all data for one example
            std::vector<torch::Tensor> d;
            // all targets for one example
            std::vector<torch::Tensor> t;

            if (!_image)
              {
                std::stringstream datastream(datas);
                std::stringstream targetstream(targets);
                torch::load(d, datastream);
                torch::load(t, targetstream);

                for (unsigned int i = 0; i < d.size(); ++i)
                  {
                    while (i >= data.size())
                      data.emplace_back();
                    data.at(i).push_back(d[i]);
                  }
                for (unsigned int i = 0; i < t.size(); ++i)
                  {
                    while (i >= target.size())
                      target.emplace_back();
                    target.at(i).push_back(t[i]);
                  }
              }
            else
              {
                ImgTorchInputFileConn *inputc
                    = dynamic_cast<ImgTorchInputFileConn *>(_inputc);

                cv::Mat bgr, bw_target;
                read_image_from_db(datas, targets, bgr, t, bw_target,
                                   inputc->_bw, inputc->width(),
                                   inputc->height());

                dataaug_then_push_back(bgr, t, bw_target, data, target);
              }
          }

        if (!has_data)
          {
            return torch::nullopt;
          }
      }

    // tensors from ids
    std::vector<torch::Tensor> data_tensors;
    std::vector<torch::Tensor> target_tensors;

    for (const auto &vec : data)
      data_tensors.push_back(torch::stack(vec));

    if (_bbox)
      {
        if (target.size() > 0)
          {
            // Concatenate instead of stacking, and index with tensor "ids"
            // This allows different size of targets within a same batch.
            const auto &vec0 = target[0];
            std::vector<at::Tensor> ids;
            ids.reserve(vec0.size());

            int id = 0;
            for (const at::Tensor &tensor : vec0)
              {
                ids.push_back(torch::full(tensor.size(0), id, at::kInt));
                ++id;
              }

            target_tensors.push_back(torch::cat(ids));
            for (const auto &vec : target)
              target_tensors.push_back(torch::cat(vec));
          }
      }
    else
      {
        for (const auto &vec : target)
          target_tensors.push_back(torch::stack(vec));
      }

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
  int TorchDataset::read_image_file(const std::string &fname, cv::Mat &out,
                                    const bool &target)
  {
    ImgTorchInputFileConn *inputc
        = dynamic_cast<ImgTorchInputFileConn *>(_inputc);

    DDImg dimg;
    inputc->copy_parameters_to(dimg);
    if (target) // used for segmentation masks
      {
        dimg._bw = true;
        dimg._interp = "nearest";
      }

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
        out = dimg._imgs[0];
        return 0;
      }
    return -1;
  }

  int TorchDataset::add_image_file(const std::string &fname,
                                   const std::vector<at::Tensor> &target)
  {
    cv::Mat img;
    int res = read_image_file(fname, img);
    if (res == 0)
      {
        add_image_batch(img, target);
      }
    return res;
  }

  int TorchDataset::add_image_file(const std::string &fname, const int &target)
  {
    return add_image_file(fname, { target_to_tensor(target) });
  }

  int TorchDataset::add_image_file(const std::string &fname,
                                   const std::vector<double> &target)
  {
    return add_image_file(fname, { target_to_tensor(target) });
  }

  int TorchDataset::add_image_image_file(const std::string &fname,
                                         const std::string &fname_target)
  {
    if (_db)
      {
        cv::Mat img;
        int res = read_image_file(fname, img);
        if (res != 0)
          return res;
        cv::Mat img_tgt;
        res = read_image_file(fname_target, img_tgt, true);
        if (res != 0)
          return res;
        add_image_batch(img, img_tgt);
      }
    else
#pragma omp ordered
      _lfilesseg.push_back(
          std::pair<std::string, std::string>(fname, fname_target));
    return 0;
  }

  int TorchDataset::add_image_bbox_file(const std::string &fname,
                                        const std::string &bboxfname)
  {
    // read image before reading bboxes to get the size of the image
    ImgTorchInputFileConn *inputc
        = dynamic_cast<ImgTorchInputFileConn *>(_inputc);

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

    if (dimg._imgs.size() == 0)
      {
        return -1;
      }

    int orig_height = dimg._imgs_size[0].first;
    int orig_width = dimg._imgs_size[0].second;

    // read bbox file
    std::vector<at::Tensor> bboxes;
    std::vector<at::Tensor> classes;

    std::ifstream infile(bboxfname);
    std::string line;
    double wfactor = static_cast<double>(inputc->_width)
                     / static_cast<double>(orig_width);
    double hfactor = static_cast<double>(inputc->_height)
                     / static_cast<double>(orig_height);

    while (std::getline(infile, line))
      {
        std::istringstream iss(line);
        std::string val;
        iss >> val;
        int cls = std::stoi(val);
        classes.push_back(target_to_tensor(cls));

        std::vector<double> bbox(4);

        iss >> val; // xmin
        bbox[0] = std::stod(val) * wfactor;
        iss >> val; // ymin
        bbox[1] = std::stod(val) * hfactor;
        iss >> val; // xmax
        bbox[2] = std::stod(val) * wfactor;
        iss >> val; // ymax
        bbox[3] = std::stod(val) * hfactor;
        // using target_to_tensor<std::vector<double>> (used by regression)
        bboxes.push_back(target_to_tensor(bbox));
      }

    // allow for no bbox (pure negative sample) with an empty tensor
    // XXX: yolox only, that discards 0 boxes internally
    if (bboxes.empty())
      {
        std::vector<double> empty_bbox = { 0.0, 0.0, 0.0, 0.0 };
        bboxes.push_back(target_to_tensor(empty_bbox));
        int cls = 0;
        classes.push_back(target_to_tensor(cls));
      }

    // add image
    add_image_batch(dimg._imgs[0],
                    { torch::stack(bboxes), torch::cat(classes) });
    return 0;
  }

  int TorchDataset::add_image_text_file(
      const std::string &fname, const std::string &target,
      std::unordered_map<uint32_t, int> &alphabet, int max_ocr_length)
  {
    at::Tensor target_tensor = torch::zeros(
        max_ocr_length, at::TensorOptions().dtype(torch::kInt64));
    at::Tensor target_length
        = torch::full(1, at::Scalar(int(target.size())),
                      at::TensorOptions().dtype(torch::kInt64));
    int i = 0;

    for (auto &c : target)
      {
        if (i >= max_ocr_length)
          {
            // can happen in test set
            this->_logger->warn("Sequence \"{}\" is exceeding maximum ocr "
                                "length {}. Truncating...",
                                target, max_ocr_length);
            break;
          }
        auto it = alphabet.find(c);

        if (it != alphabet.end())
          {
            target_tensor[i] = it->second;
          }
        else if (!_test)
#pragma omp ordered
          {
            // in a parallel loop, recheck in order to avoid race condition
            auto it = alphabet.find(c);
            if (it != alphabet.end())
              {
                target_tensor[i] = it->second;
              }
            else
              {
                this->_logger->info("added {} to alphabet", c);
                int id = alphabet.size();
                alphabet[c] = id;
                target_tensor[i] = id;
              }
          }
        else
          {
            this->_logger->warn(
                "Character {} in test set but not in train set", c);
          }

        i++;
      }
    add_image_file(fname, { target_tensor, target_length });
    return 0;
  }

  at::Tensor TorchDataset::image_to_tensor(const cv::Mat &bgr,
                                           const bool &target)
  {
    ImgTorchInputFileConn *inputc
        = dynamic_cast<ImgTorchInputFileConn *>(_inputc);

    std::vector<int64_t> sizes{ bgr.rows, bgr.cols, bgr.channels() };
    at::TensorOptions options(at::ScalarType::Byte);

    at::Tensor imgt = torch::from_blob(bgr.data, at::IntList(sizes), options);
    imgt = imgt.toType(at::kFloat).permute({ 2, 0, 1 });
    size_t nchannels = imgt.size(0);

    if (!target)
      {
        if (!inputc->_supports_bw && nchannels == 1)
          {
            this->_logger->warn("Model needs 3 input channel, input will be "
                                "duplicated to fit the model input format");
            imgt = imgt.repeat({ 3, 1, 1 });
            nchannels = 3;
          }

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
      }

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
