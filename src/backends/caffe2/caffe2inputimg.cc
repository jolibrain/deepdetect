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

//XXX Remove that to print the warnings
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
#include <caffe2/core/db.h>
#pragma GCC diagnostic pop

#include "utils/utils.hpp"
#include "backends/caffe2/caffe2inputconns.h"
#include "backends/caffe2/nettools.h"

namespace dd {

  void ImgCaffe2InputFileConn::init(const APIData &ad) {
    ImgInputFileConn::init(ad);
    if (ad.has("std"))
      _std = ad.get("std").get<double>();
  }

  void ImgCaffe2InputFileConn::transform(const APIData &ad) {
    if (_train) {
      transform_train(ad);
    } else {
      transform_predict(ad);
    }
  }

  void load_mean_file(const std::string &file, std::vector<double> &values) {
      // Load tensor
      caffe2::TensorProto mean;
      std::ifstream ifs(file);
      mean.ParseFromIstream(&ifs);
      ifs.close();
      values.resize(mean.dims(0));
      int chan_size = mean.dims(1) * mean.dims(2);

      // Compute mean values
      const float *data = mean.float_data().data();
      for (double &value : values) {
	const float *data_end = data + chan_size;
	value = std::accumulate(data, data_end, 0) / chan_size;
	data = data_end;
      }
  }

  void ImgCaffe2InputFileConn::transform_predict(const APIData &ad) {

    try {
      ImgInputFileConn::transform(ad);
    } catch (InputConnectorBadParamException &e) {
      throw;
    }

    if (!_has_mean_scalar && ad.has("mean_file")) {

      std::vector<double> mean_values;
      load_mean_file(ad.get("mean_file").get<std::string>(), mean_values);

      _has_mean_scalar = true;
      if (mean_values.size() == 3) { // BGR
	_mean = cv::Scalar(mean_values[0], mean_values[1], mean_values[2]);
      } else { // BW
	_mean = cv::Scalar(mean_values[0]);
      }
    }

    _ids = _uris;
  }

  void ImgCaffe2InputFileConn::transform_train(const APIData &ad) {

    _shuffle = true;
    APIData ad_mllib;
    if (ad.has("parameters")) {
      APIData ad_param = ad.getobj("parameters");
      if (ad_param.has("input")) {
	APIData ad_input = ad_param.getobj("input");
	fillup_parameters(ad_param.getobj("input"));
      }
      ad_mllib = ad_param.getobj("mllib");
    }

    std::string
      dbname = _model_repo + "/" + _dbfullname,
      test_dbname = _model_repo + "/" + _test_dbfullname,
      meanfile = _model_repo + "/" + _meanfname;

    bool has_data = true;
    try {
      get_data(ad);
    } catch (...) { // No dataset specified in the 'data' field
      if (!fileops::file_exists(dbname)) { // And no database in the model repository
	_logger->error("Missing training inputs");
	throw;
      }
      if (!fileops::file_exists(test_dbname)) {
	test_dbname = "";
      }
      has_data = false;
    }

    if (has_data) { // If datasets where given to the api
      if (_test_split == 0.0 && _uris.size() == 1) {
	_logger->warn("dataset unsplittable, no test will be done during training");
	test_dbname = "";
      }
      images_to_db(_uris, dbname, test_dbname);
    }

    compute_images_mean(dbname, meanfile);
    std::vector<double> mean_values;
    load_mean_file(meanfile, mean_values);

    // Enrich data object with db informations
    APIData dbad;
    dbad.add("train_db", dbname);
    dbad.add("test_db", test_dbname);
    dbad.add("mean_values", mean_values);
    const_cast<APIData&>(ad).add("db", dbad);
  }

  void ImgCaffe2InputFileConn::list_images(const std::string &root,
					   std::unordered_map<int, std::string> &corresp,
					   std::unordered_map<std::string,int> &corresp_r,
					   std::vector<std::pair<std::string, int>> &files,
					   bool is_reversed) {

    std::unordered_set<std::string> subdirs;
    if (fileops::list_directory(root, false, true, subdirs)) {
      std::string msg("failed reading image data directory " + root);
      _logger->error(msg);
      throw InputConnectorBadParamException(msg);
    }

    // list files and classes
    int cl = 0;
    for (const std::string &dir : subdirs) {

      std::unordered_set<std::string> subdir_files;
      if (fileops::list_directory(dir, true, false, subdir_files)) {
	std::string msg("failed reading image data sub-directory " + dir);
	_logger->error(msg);
	throw InputConnectorBadParamException(msg);
      }
      std::string cls = dd_utils::split(dir, '/').back();

      if (is_reversed) { // retrieve the class id
	auto it = corresp_r.find(cls);
	if (it == corresp_r.end()) {
	  _logger->warn("class {} was not indexed, skipping", cls);
	  continue;
	}
	cl = it->second;
      } else { // save the class id
	corresp.insert(std::pair<int,std::string>(cl, cls));
	corresp_r.insert(std::pair<std::string,int>(cls, cl));
      }

      for (const std::string &file : subdir_files) {
	files.push_back(std::pair<std::string,int>(file, cl));
      }
      ++cl;
    }

    if (_shuffle) {
      std::mt19937 g;
      if (_seed >= 0) {
	g = std::mt19937(_seed);
      } else {
	std::random_device rd;
	g = std::mt19937(rd());
      }
      std::shuffle(files.begin(), files.end(), g);
    }
  }

  void ImgCaffe2InputFileConn::compute_images_mean(const std::string &dbname,
						   const std::string &meanfile,
						   const std::string &backend) {
    if (fileops::file_exists(meanfile)) {
      _logger->warn("image mean file {} already exists, bypassing creation", meanfile);
      return;
    }
    _logger->info("Creating {} image mean file", meanfile);

    std::unique_ptr<caffe2::db::DB> db(caffe2::db::CreateDB(backend, dbname, caffe2::db::READ));
    std::unique_ptr<caffe2::db::Cursor> cursor(db->NewCursor());

    // Load first tensor
    caffe2::TensorProtos protos;
    protos.ParseFromString(cursor->value());
    cursor->Next();
    int count = 1;
    caffe2::TensorProto mean(protos.protos(0));
    auto &data = *mean.mutable_float_data();

    // Compute mean values
    for (; cursor->Valid(); cursor->Next()) {
      protos.ParseFromString(cursor->value());
      std::transform(data.begin(), data.end(), protos.protos(0).float_data().begin(),
		     data.begin(), std::plus<float>());
      if (!(++count % 1000)) {
	_logger->info("Processed {} entries", count);
      }
    }
    std::for_each(data.begin(), data.end(), [&count](float &f) { f /= count; });

    // Write to disk
    std::ofstream ofs(meanfile);
    mean.SerializeToOstream(&ofs);
    ofs.close();
  }

  void ImgCaffe2InputFileConn::images_to_db(const std::vector<std::string> &rfolders,
					    const std::string &traindbname,
					    const std::string &testdbname,
					    const std::string &backend) {
    bool
      is_train = fileops::file_exists(traindbname),
      is_test = fileops::file_exists(testdbname);

    // Creating a test db
    if (testdbname.size()) {

      // Test whether the train / test dbs are already in
      // since they may be long to build, we pass on them if there already
      // in the model repository.
      if (is_train && is_test) {
	_logger->warn("image db files {} and {} already exist, bypassing creation",
		      traindbname, testdbname);
	return;
      }

      // Both databases must be created at the same time to ensure a correct split of the images.
      if (is_train || is_test) {
	_logger->error("creating the {} db would overwrite an already existing one",
		       is_train ? traindbname : testdbname);
	throw InputConnectorBadParamException("failed to create databases");
      }

    } else if (is_train) {
      _logger->warn("image db file {} already exists, bypassing creation", traindbname);
      return;
    }

    std::unordered_map<int,std::string> corresp;
    std::unordered_map<std::string,int> corresp_r;
    std::vector<std::pair<std::string,int>> train_files, test_files;
    list_images(rfolders[0], corresp, corresp_r, train_files, false);

    // Creating a test db
    if (testdbname.size()) {

      if (_test_split > 0.0) {
	auto it = train_files.begin() + std::floor(train_files.size() * (1.0 - _test_split));
	test_files.assign(it, train_files.end());
	train_files.erase(it, train_files.end());
      } else {
	list_images(rfolders[1], corresp, corresp_r, test_files, true);
      }

      write_image_to_db(testdbname, test_files, backend);
    }
    write_image_to_db(traindbname, train_files, backend);

    // write corresp file
    std::ofstream correspf(_model_repo + "/" + _correspname, std::ios::binary);
    for (auto &kp : corresp) {
      correspf << kp.first << " " << kp.second << std::endl;
    }
    correspf.close();
  }

  void ImgCaffe2InputFileConn::write_image_to_db(
	const std::string &dbfullname,
	const std::vector<std::pair<std::string, int>> &lfiles,
	const std::string &backend) {

    std::unique_ptr<caffe2::db::DB> db(caffe2::db::CreateDB(backend, dbfullname, caffe2::db::NEW));

    // Prefill db entries
    int chans = channels();
    int chan_size = _height * _width * sizeof(float);
    caffe2::TensorProtos protos;
    caffe2::TensorProto *proto_data = protos.add_protos();
    proto_data->set_data_type(proto_data->FLOAT);
    proto_data->mutable_dims()->Add(chans);
    proto_data->mutable_dims()->Add(_height);
    proto_data->mutable_dims()->Add(_width);
    proto_data->mutable_float_data()->Resize(chans * _height * _width, 0);
    float *ptr = proto_data->mutable_float_data()->mutable_data();
    std::vector<void *> img(chans);
    for (int i = 0; i < chans; ++i) {
      img[i] = ptr;
      ptr += _height * _width;
    }
    caffe2::TensorProto *proto_label = protos.add_protos();
    proto_label->set_data_type(proto_label->INT32);
    int &label = *proto_label->mutable_int32_data()->Add();

    int cv_load = _bw ? CV_LOAD_IMAGE_GRAYSCALE : CV_LOAD_IMAGE_COLOR;
    cv::Size size(_width, _height);

    // Storing to db
    int count = 0;
    const int kMaxKeyLength = 256;
    char key_cstr[kMaxKeyLength];

    for (const auto &file : lfiles) {

      // Fill db entry
      std::string out;
      try {
	label = file.second;
	cv::Mat mat = cv::imread(file.first, cv_load);
	CAFFE_ENFORCE(!mat.empty());
	cv::resize(mat, mat, size, 0, 0, CV_INTER_CUBIC);
	mat.convertTo(mat, CV_32F);
	std::vector<cv::Mat> mats(chans);
	cv::split(mat, mats);
	for (int i = 0; i < chans; ++i) {
	  std::memcpy(img[i], mats[i].data, chan_size);
	}
	CAFFE_ENFORCE(protos.SerializeToString(&out));
      } catch (...) {
	_logger->warn("could not load {}", file.first);
	continue;
      }

      // Put in db
      auto txn = db->NewTransaction();
      int length = snprintf(key_cstr, kMaxKeyLength, "%08d_%s", count, file.first.c_str());
      txn->Put(std::string(key_cstr, length), out);
      txn->Commit();
      if (!(++count % 1000)) {
	_logger->info("Processed {} entries", count);
      }
    }
    if (!count) {
      std::string msg("could not fill " + dbfullname + " with the requested dataset");
      _logger->error(msg);
      throw InputConnectorBadParamException(msg);
    }
    _logger->info("{} successfully created ({} entries)", dbfullname, count);
  }

  int ImgCaffe2InputFileConn::get_batch(caffe2::TensorCPU &tensor, int num) {

    int image_count = _images.size();
    if (!image_count) {
      return 0; // No more data
    }
    int w = _images[0].cols;
    int h = _images[0].rows;
    if (image_count > num && num > 0) {
      image_count = num; // Cap the batch size to 'num'
    }

    // Resize the tensor
    std::vector<cv::Mat> chan(channels());
    tensor.Resize(std::vector<caffe2::TIndex>({image_count, channels(), h, w}));
    size_t channel_size = h * w * sizeof(float);
    uint8_t *data = reinterpret_cast<uint8_t *>(tensor.mutable_data<float>());

    auto it_begin(_images.begin());
    auto it_end(it_begin + image_count);
    for (auto it = it_begin; it < it_end; ++it) {

      // Convert from NHWC uint8_t to NCHW float
      it->convertTo(*it, CV_32F);
      if (_has_mean_scalar) {
	*it -= _mean;
      }
      cv::split(*it / _std, chan);
      for (cv::Mat &ch : chan) {
	std::memcpy(data, ch.data, channel_size);
	data += channel_size;
      }

    }
    _images.erase(it_begin, it_end);
    return image_count;
  }

  void ImgCaffe2InputFileConn::configure_db_operator(caffe2::OperatorDef &op,
						     const std::vector<double> &mean_values) {
    op.set_type("ImageInput");

    // Mandatory arguments
    Caffe2NetTools::add_arg(op, "scale", _width);
    Caffe2NetTools::add_arg(op, "crop", _width);
    Caffe2NetTools::add_arg(op, "color", channels());
    Caffe2NetTools::add_arg(op, "is_test", !_train);

    // Optionnal arguments
    std::vector<double> mean_per_channel(channels(), 0);
    std::vector<double> std_per_channel(channels(), _std);
    if (mean_values.size()) {
      mean_per_channel = mean_values;
    }
    Caffe2NetTools::add_arg(op, "mean_per_channel", mean_per_channel);
    Caffe2NetTools::add_arg(op, "std_per_channel", std_per_channel);
  }

}
