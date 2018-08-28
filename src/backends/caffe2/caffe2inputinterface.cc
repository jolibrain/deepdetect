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

#include "backends/caffe2/caffe2inputconns.h"

namespace dd {

  // Ternary on members name				NAME PREFIX	IS TRAINING	NAME SUFFIX
#define TRAIN_TERNARY(prefix, train, suffix) ((train)	? prefix	## _train	## suffix \
							: prefix			## suffix)
#define DBREADER(train)		TRAIN_TERNARY(		_blob_dbreader,	train,		)
#define BATCH_SIZE(train)	TRAIN_TERNARY(		,		train,		_batch_size)
#define DB_SIZE(train)		TRAIN_TERNARY(		,		train,		_db_size)
#define DB(train)		TRAIN_TERNARY(		,		train,		_db)

  void Caffe2InputInterface::load_dbreader(Caffe2NetTools::ModelContext &context,
					   const std::string &file, bool train) const {
    context.load_blob(file, DBREADER(train));
  }

  void Caffe2InputInterface::create_dbreader(caffe2::NetDef &init_net, bool train) const {
    Caffe2NetTools::CreateDB(init_net, DBREADER(train), DB(train));
  }

  void Caffe2InputInterface::assert_context_validity(Caffe2NetTools::ModelContext &context,
						     bool train) const {
    // Check the number of inputs
    int nb_data = _is_load_manual ? _ids.size() : DB_SIZE(train);
    int devices = context.device_count();
    if (nb_data < devices) {
      throw InputConnectorBadParamException("cannot split " + std::to_string(nb_data) + " inputs"
					    " on " + std::to_string(devices) + " device(s)");
    }
  }

  void Caffe2InputInterface::link_dbreader(const Caffe2NetTools::ModelContext &context,
					   caffe2::NetDef &net,
					   bool train) const {
    caffe2::OperatorDef op;
    int size_per_device = BATCH_SIZE(train) / context.device_count();

    //XXX Manage no-label inputs
    Caffe2NetTools::TensorProtosDBInput(op, DBREADER(train), context._input_blob,
					context._blob_label, size_per_device);
    Caffe2NetTools::insert_db_input_operator(context, net, op);
  }

  int Caffe2InputInterface::insert_inputs(Caffe2NetTools::ModelContext &context,
					  const std::vector<std::string> &blobs, int nb_data,
					  const InputGetter &get_tensors, bool train) {
    // Get the batch size of each device
    std::vector<int> batch_sizes;
    int devices = context.device_count();
    batch_sizes.resize(devices);
    std::fill(batch_sizes.begin(), batch_sizes.end(), 0);

    // Reduce the batch size if not enough data
    int batch_size = std::min(BATCH_SIZE(train), nb_data);
    if (!batch_size) {
      return 0;
    }

    // Check if there is enough data to make a batch
    CAFFE_ENFORCE(devices <= batch_size);
    int batch_per_device = batch_size / devices;
    batch_size = batch_per_device * devices;
    if (!batch_size) {
      return 0;
    }

    // If there is too few data left to form the next batch, add it to the current one
    nb_data -= batch_size;
    nb_data *= (nb_data < devices);
    batch_size += nb_data;
    for (int device = 0; device < devices; ++device) {
      batch_sizes[device] = batch_per_device + (device < nb_data);
    }

    // Loop over the devices
    for (int device = 0; device < devices; ++device) {
      std::vector<caffe2::TensorCPU> tensors(blobs.size());
      std::vector<uint8_t *> raw_datas(blobs.size(), NULL);
      int current_batch_size = batch_sizes[device];

      // Loop over the items
      for (int item = 0; item < current_batch_size; ++item) {
	std::vector<caffe2::TensorCPU> tmp_tensors(tensors.size());
	get_tensors(tmp_tensors);

	// Loop over the blobs
	for (size_t i = 0; i < tensors.size(); ++i) {
	  uint8_t *&raw_data(raw_datas[i]);
	  caffe2::TensorCPU &tensor(tensors[i]);
	  caffe2::TensorCPU &tmp(tmp_tensors[i]);

	  // Resize the tensor based on the shape of the first item
	  if (!raw_data) {
	    std::vector<caffe2::TIndex> dims(tmp.dims());
	    dims.insert(dims.begin(), current_batch_size);
	    tensor.Resize(dims);
	    raw_data = static_cast<uint8_t *>(tensor.raw_mutable_data(tmp.meta()));
	  }

	  // Copy the data
	  CAFFE_ENFORCE(tensor.size() == tmp.size() * current_batch_size);
	  CAFFE_ENFORCE(tensor.meta() == tmp.meta());
	  std::memcpy(raw_data + tmp.nbytes() * item, tmp.raw_data(), tmp.nbytes());
	}
      }

      // Insert the tensors on the current device
      for (size_t i = 0; i < tensors.size(); ++i) {
	context.insert_tensor(device, blobs[i], tensors[i]);
      }
    }

    return batch_size;
  }

  int Caffe2InputInterface::use_dbreader(Caffe2NetTools::ModelContext &context, int already_loaded,
					 bool train) {
    CAFFE_ENFORCE(!_is_load_manual);

    // Get a reference to the DBReader
    const caffe2::db::DBReader &dbreader =
      context._workspace->GetBlob(DBREADER(train))->Get<caffe2::db::DBReader>();
    std::string key, value;
    caffe2::TensorProtos protos;
    caffe2::TensorDeserializer<caffe2::CPUContext> deserializer;

    // Fetch the data
    InputGetter get_tensors = [&](std::vector<caffe2::TensorCPU> &tensors) {
      dbreader.Read(&key, &value);
      CAFFE_ENFORCE(protos.ParseFromString(value));
      CAFFE_ENFORCE(static_cast<size_t>(protos.protos_size()) == tensors.size());
      for (size_t i = 0; i < tensors.size(); ++i) {
	deserializer.Deserialize(protos.protos(i), &tensors[i]);
      }
    };

    //XXX Manage no-label inputs
    std::vector<std::string> blobs({context._input_blob, context._blob_label});
    return insert_inputs(context, blobs, DB_SIZE(train) - already_loaded, get_tensors, train);
  }

  void Caffe2InputInterface::init(InputConnectorStrategy *child) {
    _child = child;
    _default_db = _child->_model_repo + _db_relative_path;
    _default_train_db = _child->_model_repo + _train_db_relative_path;
  }

  // Notes about the APIData
  //
  // 1 -	As batch sizes are deeply linked with caffe2 input operators
  //		(database readers, tensor fillers, etc.), they are managed by InputConnectors
  //		They are however still defined in 'mllib' to have an uniformity with other MLLibs.
  //
  // 2 -	Because prediction nets and test nets are basically the same thing,
  //		they share the same internal objects (e.g. NetDef or batch_size).
  //	        The names 'test_batch_size' and 'batch_size' are only kept to have a
  //		have a coherence over the APIs.

  void Caffe2InputInterface::set_batch_sizes(const APIData &ad, bool train) {

    // No need for further checks if the data can't be batched
    if (!_is_batchable) {
      _batch_size = 1;
      _train_batch_size = train;
      return;
    }

    // Reset to default values
    _batch_size = _train_batch_size = _default_batch_size;
    _train_batch_size *= train;

    if (ad.has("parameters")) {
      const APIData &param = ad.getobj("parameters");

      if (param.has("mllib")) {
	const APIData &mllib = param.getobj("mllib");
	if (mllib.has("net")) {
	  const APIData &net = mllib.getobj("net");

	  // Set the 'main' batch_size
	  if (net.has("batch_size")) {
	    BATCH_SIZE(train) = net.get("batch_size").get<int>();
	  }
	  // Set the 'test batch_size
	  if (train && net.has("test_batch_size")) {
	    _batch_size = net.get("test_batch_size").get<int>();
	  }
	}
      }
    }

    if (!train && !_measuring) {

      // Load all the data in one batch
      if (_is_load_manual) {
	_batch_size = _ids.size();
      } else {
	_batch_size = _db_size;
	_train_batch_size = _train_db_size;
      }
    }
  }

  void Caffe2InputInterface::finalize_transform_predict(const APIData &ad) {
    set_batch_sizes(ad, false);
  }

  void Caffe2InputInterface::finalize_transform_train(const APIData &ad) {
    set_batch_sizes(ad, true);
  }

  bool Caffe2InputInterface::needs_reconfiguration(const Caffe2InputInterface &inputc) {
    return	_is_load_manual		!= inputc._is_load_manual
      ||	_is_testable		!= inputc._is_testable
      ||	((_batch_size		!= inputc._batch_size)		&& !_is_load_manual)
      ||	((_train_batch_size	!= inputc._train_batch_size)	&& !_is_load_manual)
      ;
  }

  static int compute_db_size(const std::string &type, const std::string &path) {
    std::unique_ptr<caffe2::db::DB> db(caffe2::db::CreateDB(type, path, caffe2::db::READ));
    std::unique_ptr<caffe2::db::Cursor> cursor(db->NewCursor());
    int count = 0;
    for (; cursor->Valid(); ++count, cursor->Next());
    return count;
  }

  void Caffe2InputInterface::compute_db_sizes() {
    _db_size		= fileops::is_db(_db) ?		compute_db_size(_db_type, _db)		: 0;
    _train_db_size	= fileops::is_db(_train_db) ?	compute_db_size(_db_type, _train_db)	: 0;
  }
}
