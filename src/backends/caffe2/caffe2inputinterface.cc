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

  void Caffe2InputInterface::add_tensor_loader(const Caffe2NetTools::ModelContext &context,
					       caffe2::NetDef &net,
					       bool train) const {

    // Check the batch size
    int batch_size = BATCH_SIZE(train);
    int devices = context.device_count();
    int size_per_device = batch_size / devices;
    if (size_per_device < 1 || batch_size % devices) {
      throw InputConnectorBadParamException("cannot split a batch of " +
					    std::to_string(batch_size) + " on " +
					    std::to_string(devices) + " device(s)");
    }

    caffe2::OperatorDef op;
    //XXX Manage no-label inputs
    Caffe2NetTools::TensorProtosDBInput(op, DBREADER(train), context._input_blob,
					context._blob_label, size_per_device);
    Caffe2NetTools::insert_db_input_operator(context, net, op);
  }

  void Caffe2InputInterface::get_tensor_loader_infos(int &batch_size, int &total_size,
						     bool train) const {
    batch_size = BATCH_SIZE(train);
    total_size = DB_SIZE(train);
  }

  void Caffe2InputInterface::init(const std::string &model_repo) {
    _default_db = model_repo + _db_relative_path;
    _default_train_db = model_repo + _train_db_relative_path;
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
  //
  // 3 -	Batch sizes make sens when there is a large amount of data (training or testing),
  //		but not for a simple prediction call (all the data should fit as one batch).
  //		The best way to assert that the input isn't used to measure nets is to check the
  //		'output' parameter, even if it is destined to OutputConnectors.

  void Caffe2InputInterface::set_batch_sizes(const APIData &ad, bool train) {

    // Reset to default values
    _batch_size = _train_batch_size = _default_batch_size;
    _train_batch_size *= train;
    bool measuring = false;

    if (ad.has("parameters")) {
      const APIData &param = ad.getobj("parameters");

      // Check if measures are present
      measuring = param.has("output") && param.getobj("output").has("measure");

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

    _is_batched = train || measuring;
    // If the data is fetched from a database
    if (!_is_batched && !_is_load_manual) {
      _batch_size = _db_size;
      _train_batch_size = _train_db_size;
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
      ||	_batch_size		!= inputc._batch_size
      ||	_train_batch_size	!= inputc._train_batch_size
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
