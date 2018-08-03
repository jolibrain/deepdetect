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

namespace dd {

  void ImgCaffe2InputFileConn::update(const APIData &ad) {
    if (ad.has("std")) {
      _std = static_cast<float>(ad.get("std").get<double>());
    }
    //XXX Implement support for other tags (multi_label, segmentation, ...)
  }

  void ImgCaffe2InputFileConn::init(const APIData &ad) {
    ImgInputFileConn::init(ad);
    Caffe2InputInterface::init(_model_repo);
    update(ad);
    //XXX let the meanfile and/or mean-values be changed by the API ?
    _mean_file = _model_repo + "/mean.pb";
    _corresp_file = _model_repo + "/corresp.txt";
    load_mean_file();
  }

  void ImgCaffe2InputFileConn::transform(const APIData &ad) {

    // Update internal values
    if (ad.has("parameters")) {
      APIData ad_param = ad.getobj("parameters");
      if (ad_param.has("input")) {
	APIData ad_input = ad_param.getobj("input");
	fillup_parameters(ad_input);
	update(ad_input);
      }
    }

    // Prepare the images
    if (_train) {
      transform_train(ad);
      finalize_transform_train(ad);
    } else {
      transform_predict(ad);
      finalize_transform_predict(ad);
    }

    load_mean_file();
  }

  void ImgCaffe2InputFileConn::load_mean_file() {

    // Default values
    if (!fileops::file_exists(_mean_file)) {
      _logger->info("No mean file in the repository");
      _mean_values.clear();
      _mean_values.resize(channels());
      return;
    }

    // Load tensor
    caffe2::TensorProto mean;
    std::ifstream ifs(_mean_file);
    mean.ParseFromIstream(&ifs);
    ifs.close();
    _mean_values.resize(mean.dims(0));
    CAFFE_ENFORCE(_mean_values.size() == static_cast<size_t>(channels()));
    int chan_size = mean.dims(1) * mean.dims(2);

    // Compute mean values
    const float *data = mean.float_data().data();
    for (float &value : _mean_values) {
      const float *data_end = data + chan_size;
      value = std::accumulate(data, data_end, 0) / chan_size;
      data = data_end;
    }
  }

  void ImgCaffe2InputFileConn::transform_predict(const APIData &ad) {

    if (ad.has("data")) { // If we know what kind of data we'll have to work with

      get_data(ad);
      _is_load_manual = !fileops::is_db(_uris[0]);
      if (_is_load_manual) {
	ImgInputFileConn::transform(ad); // Apply classic image pre-computations
      }

    } else {
      _is_load_manual = true; // Can't add automatic inputs without data
    }

    _train_db = "";
    if (_is_load_manual) {

      _db = "";
      compute_db_sizes();
      _is_testable = false; //XXX Can't infer labels from raw data
      _is_load_manual = true;
      _ids = _uris;

    } else {

      _db = _uris[0];
      compute_db_sizes();
      _is_testable = true;
      _is_load_manual = false;
      _ids.resize(_db_size);
      std::iota(_ids.begin(), _ids.end(), 0); // Set indices as ids
    }
  }

  void ImgCaffe2InputFileConn::transform_train(const APIData &ad) {

    _shuffle = true;
    _is_load_manual = false;
    bool new_data = false;

    // Get databases paths
    if (ad.has("data")) {

      get_data(ad);
      new_data = uris_to_db();

    } else {

      _train_db = _default_train_db;
      if (!fileops::file_exists(_train_db)) { // No database in the model repository
	_logger->error("Missing training inputs");
	throw;
      }

      _db = _default_db;
      _is_testable = fileops::file_exists(_db);
    }

    compute_db_sizes();

    if (new_data || !fileops::file_exists(_mean_file)) {
      compute_images_mean();
    }
  }

  bool ImgCaffe2InputFileConn::uris_to_db() {

    // Check if the uris are coherent
    bool uris_are_db = fileops::is_db(_uris[0]);
    if (uris_are_db && _uris.size() > 1 && !fileops::is_db(_uris[1])) {
      throw InputConnectorBadParamException("Can't use both files and databases as inputs");
    }

    // Check if there is data to test on
    _is_testable = (_uris.size() > 1); // Two sets of data
    _is_testable |= (!uris_are_db && _test_split > 0); // A folder with a split size

    // Set databases paths
    if (uris_are_db) {
      _train_db = _uris[0];
      _db = _is_testable ? _uris[1] : "";
      return false;
    }
    _train_db = _default_train_db;
    _db = _is_testable ? _default_db : "";

    // Check if local databases could be used
    bool is_train = fileops::file_exists(_train_db), is_test = fileops::file_exists(_db);
    if (is_train && (!_is_testable || is_test)) {
      _logger->warn("Found local database(s), bypassing creation");
      return false;
    }

    // Check if creation would overwrite existing data
    if (is_train != (_is_testable && is_test)) {
      _logger->error("Creating a pair of local train/test databases would overwrite files");
      throw InputConnectorBadParamException("failed to create databases");
    }

    // Create database(s)
    if (!is_train) {
      _logger->info("Transforming images to database(s)");
      images_to_db();
    }
    return is_train;
  }

  void ImgCaffe2InputFileConn::list_images(const std::string &root,
					   std::unordered_map<int, std::string> &corresp,
					   std::unordered_map<std::string,int> &corresp_r,
					   std::vector<std::pair<std::string, int>> &files,
					   bool is_reversed) {

    std::unordered_set<std::string> subdirs;
    if (fileops::list_directory(root, false, true, subdirs)) {
      std::string msg("Failed reading image data directory " + root);
      _logger->error(msg);
      throw InputConnectorBadParamException(msg);
    }

    // list files and classes
    int cl = 0;
    for (const std::string &dir : subdirs) {

      std::unordered_set<std::string> subdir_files;
      if (fileops::list_directory(dir, true, false, subdir_files)) {
	std::string msg("Failed reading image data sub-directory " + dir);
	_logger->error(msg);
	throw InputConnectorBadParamException(msg);
      }
      std::string cls = dd_utils::split(dir, '/').back();

      if (is_reversed) { // retrieve the class id
	auto it = corresp_r.find(cls);
	if (it == corresp_r.end()) {
	  _logger->warn("Class {} was not indexed, skipping", cls);
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

  void ImgCaffe2InputFileConn::compute_images_mean() {

    _logger->info("Creating mean file {}", _mean_file);
    std::unique_ptr<caffe2::db::DB> db(caffe2::db::CreateDB(_db_type, _db, caffe2::db::READ));
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
    std::ofstream ofs(_mean_file);
    mean.SerializeToOstream(&ofs);
    ofs.close();
  }

  void ImgCaffe2InputFileConn::images_to_db() {

    std::unordered_map<int,std::string> corresp;
    std::unordered_map<std::string,int> corresp_r;
    std::vector<std::pair<std::string,int>> train_files, test_files;
    list_images(_uris[0], corresp, corresp_r, train_files, false);

    // Create a test db
    if (_is_testable) {

      if (_uris.size() > 1) {
	list_images(_uris[1], corresp, corresp_r, test_files, true);

      } else { // Split that data
	auto it = train_files.begin() + std::floor(train_files.size() * (1.0 - _test_split));
	test_files.assign(it, train_files.end());
	train_files.erase(it, train_files.end());
      }
      write_images_to_db(_db, test_files);
    }
    write_images_to_db(_train_db, train_files);

    // write corresp file
    std::ofstream correspf(_corresp_file, std::ios::binary);
    for (auto &kp : corresp) {
      correspf << kp.first << " " << kp.second << std::endl;
    }
    correspf.close();
  }

  void ImgCaffe2InputFileConn::write_images_to_db(const std::string &dbname,
						  const std::vector<std::pair<std::string, int>>
						  &lfiles) {

    std::unique_ptr<caffe2::db::DB> db(caffe2::db::CreateDB(_db_type, dbname, caffe2::db::NEW));
    std::unique_ptr<caffe2::db::Transaction> txn(db->NewTransaction());

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
    const int kMaxKeyLength = 256, batch_size = 1000;
    char key_cstr[kMaxKeyLength];

    for (const std::pair<std::string, int> &file : lfiles) {

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
	_logger->warn("Could not load {}", file.first);
	continue;
      }

      // Put in db
      int length = snprintf(key_cstr, kMaxKeyLength, "%08d_%s", count, file.first.c_str());
      txn->Put(std::string(key_cstr, length), out);
      if (!(++count % batch_size)) {
	_logger->info("Processed {} entries", count);
	txn->Commit();
      }
    }
    if (!count) {
      std::string msg("Could not fill " + dbname + " with the requested dataset");
      _logger->error(msg);
      throw InputConnectorBadParamException(msg);
    }

    // Write the last batch
    if (count % batch_size) {
      _logger->info("Processed {} entries", count);
      txn->Commit();
    }
    _logger->info("{} successfully created ({} entries)", dbname, count);
  }

  int ImgCaffe2InputFileConn::load_batch(Caffe2NetTools::ModelContext &context) {
    CAFFE_ENFORCE(_is_load_manual);

    // Split the images over the tensors
    std::vector<caffe2::TensorCPU> tensors(context.device_count());
    int image_count = std::min(static_cast<int>(_images.size()), _batch_size);
    int image_per_tensor = image_count / tensors.size();
    image_count = image_per_tensor * tensors.size();

    // Check if there is enough images to make a batch
    if (!image_count) {
      if (_images.size()) {
	_logger->warn("The last {} image(s) could not be splitted between the {} tensors",
		      _images.size(), tensors.size());
      }
      _images.clear();
      return 0;
    }

    // Transform the data
    int w = _images[0].cols;
    int h = _images[0].rows;
    std::vector<cv::Mat> chan(channels());
    size_t channel_size = h * w * sizeof(float);
    auto it_begin(_images.begin());
    auto it_end(it_begin + image_count);
    auto image = it_begin;

    for (caffe2::TensorCPU &tensor : tensors) {
      tensor.Resize(std::vector<caffe2::TIndex>({image_per_tensor, channels(), h, w}));
      uint8_t *data = reinterpret_cast<uint8_t *>(tensor.mutable_data<float>());
      for (int i = 0; i < image_per_tensor; ++i, ++image) {

	// Convert from NHWC uint8_t to NCHW float
	image->convertTo(*image, CV_32F);
	cv::split(*image, chan);
	for (cv::Mat &ch : chan) {
	  std::memcpy(data, ch.data, channel_size);
	  data += channel_size;
	}

      }
    }

    // Update
    _images.erase(it_begin, it_end);
    context.insert_inputs(tensors);
    return image_count;
  }

  bool ImgCaffe2InputFileConn::needs_reconfiguration(const ImgCaffe2InputFileConn &inputc) {
    return Caffe2InputInterface::needs_reconfiguration(inputc)
      ||	_std		!= inputc._std
      ||	_mean_values	!= inputc._mean_values
      ;
  }

  void ImgCaffe2InputFileConn::add_constant_layers(const Caffe2NetTools::ModelContext &context,
						   caffe2::NetDef &init_net) {
    caffe2::OperatorDef op;
    Caffe2NetTools::GivenTensorFill(op, _blob_mean_values, { channels() }, _mean_values);
    Caffe2NetTools::copy_and_broadcast_operator(context, init_net, op);
  }

  void ImgCaffe2InputFileConn::
  add_transformation_layers(const Caffe2NetTools::ModelContext &context,
			    caffe2::NetDef &net_def) {
    Caffe2NetTools::ScopedNet net = context.scope_net(net_def);
    Caffe2NetTools::add_external_input(net, _blob_mean_values);
    Caffe2NetTools::Sub(net, context._input_blob, _blob_mean_values,
			context._input_blob, 1, 1); // broadcast axis=1 means channel N[C]HW
    Caffe2NetTools::Scale(net, context._input_blob, context._input_blob, 1.f / _std);
  }
}
