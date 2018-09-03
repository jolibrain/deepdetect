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

    // std
    if (ad.has("std")) {
      apitools::get_float(ad, "std", _std);
    }

    // mean
    if (!_has_mean_scalar) {
      load_mean_file();
    }

    // If the images are scaled but not stretched or cropped, then their sizes will differ
    _is_batchable &= !_scaled;

    //XXX Implement support for other tags (multi_label, segmentation, ...)
  }

  void ImgCaffe2InputFileConn::init(const APIData &ad) {

    ImgInputFileConn::init(ad);
    Caffe2InputInterface::init(this);
    _mean_file = _model_repo + "/mean.pb";
    _corresp_file = _model_repo + "/corresp.txt";
    update(ad);
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
  }

  void ImgCaffe2InputFileConn::load_mean_file() {

    // Default values
    if (!fileops::file_exists(_mean_file)) {
      _logger->info("No mean file in the repository");
      _mean.clear();
      _mean.resize(channels());
      return;
    }

    // Load
    caffe2::TensorProto mean;
    std::ifstream ifs(_mean_file);
    mean.ParseFromIstream(&ifs);
    ifs.close();

    // Check
    auto &dims = mean.dims();
    CAFFE_ENFORCE(dims.size() == 1);
    CAFFE_ENFORCE(dims[0] == channels());

    // Assign
    auto &data = mean.float_data();
    _mean.assign(data.begin(), data.end());
  }

  inline std::vector<float> compute_scale(const cv::Mat &image, const std::pair<int, int> &size) {
    return {
      static_cast<float>(image.rows) / size.first,
      static_cast<float>(image.cols) / size.second
    };
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
      _scales.resize(_ids.size());
      std::transform(_images.begin(), _images.end(), _images_size.begin(), _scales.begin(),
		     compute_scale);

    } else {

      _db = _uris[0];
      compute_db_sizes();
      _is_testable = true;
      _is_load_manual = false;
      _ids.resize(_db_size);
      _scales.resize(_db_size);

      // Set indices as ids
      for (int id = 0; id < _db_size; ++id) {
	_ids[id] = std::to_string(id);
      }
      // No pre-computation
      std::fill(_scales.begin(), _scales.end(), std::vector<float>({1, 1}));
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

    if (_has_mean_scalar) {
      // Save the mean values used in this training
      caffe2::TensorProto mean;
      std::vector<size_t> dims({ _mean.size(), 1, 1 });
      *mean.mutable_dims() = {dims.begin(), dims.end()};
      *mean.mutable_float_data() = {_mean.begin(), _mean.end()};
      std::ofstream ofs(_mean_file);
      mean.SerializeToOstream(&ofs);
      ofs.close();

    } else if (new_data || !fileops::file_exists(_mean_file)) {
      // Compute the mean values of the database and save it into a file
      compute_images_mean();
      load_mean_file();
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

  void ImgCaffe2InputFileConn::image_proto_to_mats(const caffe2::TensorProto &proto,
						   std::vector<cv::Mat> &mats,
						   bool resize) const {

    cv::Mat src;
    int chans = channels();
    int type = proto.data_type();

    // Encoded
    if (type == proto.STRING) {
      auto &data = proto.string_data();
      CAFFE_ENFORCE(data.size() == 1);

      // Convert
      const std::string &img = data[0];
      int size = img.size();
      src = cv::imdecode(cv::Mat(1, &size, CV_8UC1, const_cast<char *>(img.data())),
			 _bw ? CV_LOAD_IMAGE_GRAYSCALE : CV_LOAD_IMAGE_COLOR);
      CAFFE_ENFORCE(src.rows && src.cols);

      // Raw
    } else if (type == proto.BYTE) {
      auto &data = proto.byte_data();
      int src_c = (proto.dims_size() == 3) ? proto.dims(2) : 1;
      CAFFE_ENFORCE(src_c == 3 || src_c == 1);

      // Convert
      src.create(proto.dims(0), proto.dims(1), (src_c == 1) ? CV_8UC1 : CV_8UC3);
      std::memcpy(src.ptr<uchar>(0), data.data(), data.size());
      if (chans != src_c) {
	cv::cvtColor(src, src, _bw ? cv::COLOR_RGB2GRAY : cv::COLOR_GRAY2RGB);
      }
    } else {
      throw InputConnectorBadParamException("Unknown image data type.");
    }

    if (resize) {
      cv::resize(src, src, cv::Size(_width, _height), 0, 0, CV_INTER_CUBIC);
    }
    mats.resize(chans);
    cv::split(src, mats);
  }

  void ImgCaffe2InputFileConn::mats_to_tensor(const std::vector<cv::Mat> &mats,
					      caffe2::TensorCPU &tensor) const {
    // Fill with the current configuration
    CAFFE_ENFORCE(mats.size() == static_cast<size_t>(channels()));
    int h = _scaled ? mats[0].rows : _height;
    int w = _scaled ? mats[0].cols : _width;
    size_t channel_size = h * w * sizeof(float);
    std::vector<caffe2::TIndex> dims({channels(), h, w});

    // Resize the tensor
    tensor.Resize(dims);
    uint8_t *data = reinterpret_cast<uint8_t *>(tensor.mutable_data<float>());

    // Convert to float
    cv::Mat ch;
    for (const cv::Mat &mat : mats) {
      CAFFE_ENFORCE(mat.cols == w && mat.rows == h);
      mat.convertTo(ch, CV_32F);
      std::memcpy(data, ch.data, channel_size);
      data += channel_size;
    }
  }

  void ImgCaffe2InputFileConn::im_info_to_tensor(const cv::Mat &img,
						 caffe2::TensorCPU &tensor) const {
    tensor.Resize(std::vector<caffe2::TIndex>({3}));
    float *data = tensor.mutable_data<float>();
    data[0] = img.rows;
    data[1] = img.cols;
    data[2] = 1;
  }

  void ImgCaffe2InputFileConn::compute_images_mean() {

    _logger->info("Creating mean file {}", _mean_file);
    std::unique_ptr<caffe2::db::DB> db(caffe2::db::CreateDB(_db_type, _db, caffe2::db::READ));
    std::unique_ptr<caffe2::db::Cursor> cursor(db->NewCursor());

    // Preset mean tensor
    int chans = channels();
    caffe2::TensorProto mean;
    mean.mutable_dims()->Add(chans);
    mean.set_data_type(mean.FLOAT);
    auto &data = *mean.mutable_float_data();
    data.Resize(chans, 0);

    // Loop through the db
    caffe2::TensorProtos protos;
    std::vector<cv::Mat> mats;
    int count = 0;
    for (; cursor->Valid(); cursor->Next()) {
      protos.ParseFromString(cursor->value());
      image_proto_to_mats(protos.protos(0), mats);

      // Sum the channel and add
      for (int i = 0; i < chans; ++i) {
	data[i] += cv::mean(mats[i])[0];
      }

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

    //TODO Manage Dectron-like databases (im_info blob, bbox, classes, ...)

    // Prefill db entries
    caffe2::TensorProtos protos;
    caffe2::TensorProto *proto_data = protos.add_protos();
    caffe2::TensorProto *proto_label = protos.add_protos();
    proto_data->set_data_type(proto_data->STRING);
    proto_label->set_data_type(proto_label->INT32);
    proto_data->mutable_dims()->Add(1);
    proto_label->mutable_dims()->Add(1);
    std::string &data = *proto_data->mutable_string_data()->Add();
    int &label = *proto_label->mutable_int32_data()->Add();

    // Storing to db
    int count = 0;
    const int kMaxKeyLength = 256, batch_size = 1000;
    char key_cstr[kMaxKeyLength];

    for (const std::pair<std::string, int> &file : lfiles) {

      // Fill db entry
      std::string out;
      try {
	std::stringstream s;
	s << std::ifstream(file.first).rdbuf();
	data = s.str();
	label = file.second;
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

  void ImgCaffe2InputFileConn::link_train_dbreader(const Caffe2NetTools::ModelContext &context,
						   caffe2::NetDef &net) const {
    // Notes
    //
    // -- 1 --
    // ImageInput operators output square-shaped images
    // See pytorch/caffe2/image/image_input_op.cc
    //
    // -- 2 --
    // GPU transformations will swap the dimensions (from NHWC to NCHW)
    // See pytorch/caffe2/image/transform_gpu.cu
    //
    // -- 3 --
    // NHWC2NCHW can't be inplace, so its input must have a different name from its output
    // See pytorch/caffe2/operators/order_switch_ops.cc

    // Square
    CAFFE_ENFORCE(_width == _height);

    // Check if gpu optimizations can be used
    bool use_gpu_transform =
#ifdef CPU_ONLY
      false
#else
      (context._devices[0].device_type() == caffe2::CUDA)
#endif
      ;

    std::string input = context._input_blob;
    if (!use_gpu_transform) {
      input += "_NHWC";
    }

    // Add an ImageInput
    // TODO manage Detectron databases
    DBInputSetter config_dbinput =
      [&](caffe2::OperatorDef &op, const std::string &reader, int batch_size) {
      Caffe2NetTools::ImageInput(op, reader, input, context._blob_label, batch_size,
				 channels(), _width, use_gpu_transform);
    };
    link_dbreader(context, net, config_dbinput, true);

    // Swap manually
    if (!use_gpu_transform) {
      Caffe2NetTools::ScopedNet ns(context.scope_net(net));
      Caffe2NetTools::NHWC2NCHW(ns, input, context._input_blob);
    }
  }

  int ImgCaffe2InputFileConn::use_test_dbreader(Caffe2NetTools::ModelContext &context,
						int already_loaded) const {
    caffe2::TensorDeserializer<caffe2::CPUContext> deserializer;
    ProtosConverter convert_protos =
      [&](const caffe2::TensorProtos &protos, std::vector<caffe2::TensorCPU> &tensors) {

      // Convert the image
      std::vector<cv::Mat> mats;
      image_proto_to_mats(protos.protos(0), mats, true);
      mats_to_tensor(mats, tensors[0]);

      // Deserialize the label
      deserializer.Deserialize(protos.protos(1), &tensors[1]);
    };
    return use_dbreader(context, already_loaded, convert_protos, false);
  }

  int ImgCaffe2InputFileConn::load_batch(Caffe2NetTools::ModelContext &context,
					 int already_loaded) {
    // Call the generic batch loader
    if (!_is_load_manual) {
      return use_test_dbreader(context, already_loaded);
    }

    auto image = _images.begin() + already_loaded;
    InputGetter get_tensors = [&](std::vector<caffe2::TensorCPU> &tensors) {
      std::vector<cv::Mat> chan;
      cv::split(*image, chan);
      mats_to_tensor(chan, tensors[0]);
      im_info_to_tensor(*image++, tensors[1]);
    };

    return insert_inputs(context, {context._input_blob, context._blob_im_info},
			 _images.size() - already_loaded, get_tensors, false);
  }

  bool ImgCaffe2InputFileConn::needs_reconfiguration(const ImgCaffe2InputFileConn &inputc) const {
    return Caffe2InputInterface::needs_reconfiguration(inputc)
      ||	_std	!= inputc._std
      ||	_mean	!= inputc._mean
      ;
  }

  void ImgCaffe2InputFileConn::add_constant_layers(const Caffe2NetTools::ModelContext &context,
						   caffe2::NetDef &init_net) const {
    caffe2::OperatorDef op;
    Caffe2NetTools::GivenTensorFill(op, _blob_mean_values, { channels() }, _mean);
    Caffe2NetTools::copy_and_broadcast_operator(context, init_net, op);
  }

  void ImgCaffe2InputFileConn::
  add_transformation_layers(const Caffe2NetTools::ModelContext &context,
			    caffe2::NetDef &net_def) const {
    Caffe2NetTools::ScopedNet net = context.scope_net(net_def);
    Caffe2NetTools::add_external_input(net, _blob_mean_values);
    Caffe2NetTools::Sub(net, context._input_blob, _blob_mean_values,
			context._input_blob, 1, 1); // broadcast axis=1 means channel N[C]HW
    Caffe2NetTools::Scale(net, context._input_blob, context._input_blob, 1.f / _std);
  }
}
