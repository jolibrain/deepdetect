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

#ifndef CPU_ONLY
#include <caffe2/core/context_gpu.h>
#endif

#pragma GCC diagnostic pop

#include <caffe2/core/init.h>
#include "imginputfileconn.h"
#include "backends/caffe2/caffe2lib.h"
#include "backends/caffe2/nettools.h"

//XXX Remove that to print the warnings
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "outputconnectorstrategy.h"
#pragma GCC diagnostic pop

namespace dd {

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  Caffe2Lib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::
  Caffe2Lib(const Caffe2Model &c2model)
    :MLLib<TInputConnectorStrategy,TOutputConnectorStrategy,Caffe2Model>(c2model) {
    this->_libname = "caffe2";
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  Caffe2Lib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::
  Caffe2Lib(Caffe2Lib &&c2l) noexcept
    :MLLib<TInputConnectorStrategy,TOutputConnectorStrategy,Caffe2Model>(std::move(c2l)) {
    this->_libname = "caffe2";

    _context = std::move(c2l._context);
    _init_net = std::move(c2l._init_net);
    _train_net = std::move(c2l._train_net);
    _net = std::move(c2l._net);
    _state = c2l._state;
    _last_inputc = c2l._last_inputc;
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  Caffe2Lib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::~Caffe2Lib() {}

  template <typename T>
  using StateSetter = void(Caffe2LibState::*)(const T&);

#ifdef CPU_ONLY
  inline void _set_gpu_state(const APIData &ad, Caffe2LibState&, void*, void*) {
    if (ad.has("gpuid") || ad.has("gpu")) {
      this->_logger->warn("Parametters 'gpuid' and 'gpu' are not used in CPU_ONLY mode");
    }
  }
#else
  inline void _set_gpu_state(const APIData &ad, Caffe2LibState &state,
			     StateSetter<std::vector<int>> gpu_ids, StateSetter<bool> is_gpu) {
    if (ad.has("gpuid")) {
      std::vector<int> ids;

      // Fetch ids from the ApiData
      apitools::get_vector(ad, "gpuid", ids);

      // By default, use every GPUs
      if (ids.size() == 1 && ids[0] == -1) {
	ids.clear();
	int count_gpus = 0;
	cudaGetDeviceCount(&count_gpus);
	ids.resize(count_gpus);
	std::iota(ids.begin(), ids.end(), 0);
      }
      CAFFE_ENFORCE(!ids.empty());
      (state.*gpu_ids)(ids);
      (state.*is_gpu)(true);
    }

    if (ad.has("gpu")) {
      (state.*is_gpu)(ad.get("gpu").get<bool>());
    }
  }
#endif

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  void Caffe2Lib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::
  set_gpu_state(const APIData &ad, bool dft) {
    StateSetter<std::vector<int>> gpu_ids;
    StateSetter<bool> is_gpu;
    if (dft) {
      gpu_ids = &Caffe2LibState::set_default_gpu_ids;
      is_gpu = &Caffe2LibState::set_default_is_gpu;
    } else {
      gpu_ids = &Caffe2LibState::set_gpu_ids;
      is_gpu = &Caffe2LibState::set_is_gpu;
    }
    _set_gpu_state(ad, _state, gpu_ids, is_gpu);
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  void Caffe2Lib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::
  instantiate_template(const APIData &ad) {

    //XXX Implement build-in template creation (mlp, convnet, resnet, etc.)

    std::string model_tmpl = ad.get("template").get<std::string>();
    this->_logger->info("Instantiating model template {}", model_tmpl);

    // Fetch the files path
    std::map<std::string, std::string> files;
    bool finetuning = ad.has("finetuning") && ad.get("finetuning").get<bool>();
    this->_mlmodel.list_template_files(model_tmpl, files, finetuning);
    for (const std::pair<std::string, std::string> &file : files) {

      // Assert the remote file exists
      if (!fileops::file_exists(file.first)) {
	throw MLLibBadParamException("file '" + file.first + "' doen't exists");
      }

      // Assert the local file won't be erased
      if (fileops::file_exists(file.second)) {
	throw MLLibBadParamException("using template while model files already exists, "
				     "remove 'template' from 'mllib', "
				     "or remove existing '.pb' files ?");
      }

      // Convert the prototxt file
      caffe2::NetDef buffer;
      Caffe2NetTools::import_net(buffer, file.first);
      Caffe2NetTools::export_net(buffer, file.second);
    }

    // Update the mlmodel
    this->_mlmodel._model_template = model_tmpl;
    this->_mlmodel.update_from_repository(this->_logger);
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  void Caffe2Lib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::
  set_train_mode(const APIData &ad, bool train) {

    // Update the input connector
    TInputConnectorStrategy inputc(this->_inputc);
    inputc._train = train;

    // If the net is neither training nor testing, the input connector can
    // load all the data into a single batch
    inputc._measuring = ad.getobj("parameters").getobj("output").has("measure");

    // Configure the input data
    try {
      inputc.transform(ad);
    } catch (std::exception &e) {
      this->_logger->error("Could not configure the InputConnector {}", e.what());
      throw;
    }

    // Read the repository again in case the input connector added something
    this->_mlmodel.update_from_repository(this->_logger);

    // Update the local input connector
    bool force_init = inputc.needs_reconfiguration(_last_inputc);
    _last_inputc = inputc;

    // Reset the state
    _state.reset();
    _state.set_is_training(train);
    if (force_init) {
      _state.force_init();
    }
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  void Caffe2Lib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::
  create_model_train() {

    // Computation is done on new nets
    caffe2::NetDef init_net, train_net, test_net;

    //XXX Find a way to prevent code duplication when applying changes to the test net

    // Load or create a dbreader per net
    if (_state.resume()) {

      _last_inputc.load_dbreader(_context, this->_mlmodel._dbreader_train_state, true);
      if (_state.is_testing()) {
	_last_inputc.load_dbreader(_context, this->_mlmodel._dbreader_state);
      }

    } else {

      _last_inputc.create_dbreader(init_net, true);
      if (_state.is_testing()) {
	_last_inputc.create_dbreader(init_net);
      }

    }

    // Load batches from the database
    _last_inputc.link_dbreader(_context, train_net, true);

    // Apply input tranformations
    _last_inputc.add_constant_layers(_context, init_net);
    _last_inputc.add_transformation_layers(_context, train_net);
    if (_state.is_testing()) {
      _last_inputc.add_transformation_layers(_context, test_net);
    }

    // Add the requested net, with its loss and gradients
    _context.append_trainable_net(train_net, _net);
    if (_state.is_testing()) {
      _context.append_net(test_net, _net);
    }

    // The test net is complete

    // Set the learning-related operators and blobs to the desired iteration
    if (_state.resume()) {
      _context.load_lr(this->_mlmodel._lr_state); // Prefill the workspace
      _context.load_iter(this->_mlmodel._iter_state); // Just load the integer
    } else {
      _context.reset_iter();
    }

    // Duplicate the init net outputs on all the devices
    Caffe2NetTools::copy_and_broadcast_operators(_context, init_net, _init_net);

    // Forward the APIData configuration in an operator
    Caffe2NetTools::LROpModifier lr_config =
      [&](caffe2::OperatorDef &lr, const std::string &iter, const std::string &rate) {
      Caffe2NetTools::LearningRate(lr, iter, rate, _state.lr_policy(), _state.base_lr(),
				   _state.stepsize(), _state.max_iter(),
				   _state.gamma(), _state.power());
    };
    Caffe2NetTools::insert_learning_operators(_context, train_net, init_net, lr_config);
    Caffe2NetTools::get_optimizer(_state.solver_type())
      (_context, train_net, init_net, _state.momentum(), _state.rms_decay());

    // Apply changes
    _net.Swap(&test_net);
    _init_net.Swap(&init_net);
    _train_net.Swap(&train_net);
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  void Caffe2Lib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::
  create_model_predict() {
    _context._parallelized = false;

    // Computation is done on a new net
    caffe2::NetDef tmp_net;

    // Add database informations
    if (!_last_inputc.is_load_manual()) {
      _last_inputc.create_dbreader(_init_net);
    }

    // Add transformations
    _last_inputc.add_constant_layers(_context, _init_net);
    _last_inputc.add_transformation_layers(_context, tmp_net);

    // Add the rest of the operators
    _context.append_net(tmp_net, _net);

    std::string extract = _state.extract_layer();
    if (!extract.empty()) { // unsupervised
      // Operators placed after the extracted layer are removed (they would do useless computation)
      Caffe2NetTools::truncate_net(tmp_net, extract);
    }

    // Force the device for every operators
    Caffe2NetTools::set_net_device(tmp_net, _context._devices[0]);
    Caffe2NetTools::set_net_device(_init_net, _context._devices[0]);

    // Apply changes
    _net.Swap(&tmp_net);
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  void Caffe2Lib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::
  create_model() {
    this->_logger->info("Re-creating the nets");
    Caffe2NetTools::ensure_is_batchable(_net);

    // _input_blob and _output_blobs should have been initialized during init_mllib
    // (by "inputlayer" and "outputlayer" respectively)
    // If the are not, we'll do a guess based on the following conventions:
    //    - the input blob name is (or contains) "data"
    //    - the external inputs are sorted
    //    - all the outputs are created by the last operator
    if (!_context._output_blobs.size()) {
      const auto &outputs = _net.op(_net.op().size() - 1).output();
      _context._output_blobs.assign(outputs.begin(), outputs.end());
    }
    if (_context._input_blob.empty()) {
      const auto &inputs = _net.external_input();
      _context._input_blob = inputs[0];
      if (_context._input_blob.find("data") == std::string::npos) {
	_context._input_blob = inputs[inputs.size() - 1];
	if (_context._input_blob.find("data") == std::string::npos) {
	  _context._input_blob = "data";
	}
      }
    }

    // Reset the context
    _context.reset_workspace();
#ifndef CPU_ONLY
    if (_state.is_gpu()) {
      _context.reset_devices(_state.gpu_ids());
    } else
#endif
      _context.reset_devices();

    // Transform the nets
    if (_state.is_training()) {
      create_model_train();
    } else {
      create_model_predict();
    }

    // Set an unique name to each net
    _net.set_name("_net");
    _init_net.set_name("_init_net");
    _train_net.set_name("_train_net");

    // Add the inputs
    this->_logger->info("Run init net");
    _context.create_input();
    _context.run_net_once(_init_net);

    // Register the prediction net
    if (!_state.is_training() || _state.is_testing()) {
      this->_logger->info("Create predict net");
      _context.create_net(_net);
    }

    // Register the training net
    if (_state.is_training()) {
      this->_logger->info("Create train net");
      _context.create_net(_train_net);
    }

    this->_logger->info("Nets updated");
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  void Caffe2Lib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::
  update_model() {

    if (!_state.changed()) {
      if (_state.is_training()) {
	this->_logger->info("Reseting the workspace");
	_context.run_net_once(_init_net);
      }
      return;
    }

    try {

      // Reload from the disk
      std::string init_net_path = _state.is_training() && _state.resume() ?
	this->_mlmodel._init_state : this->_mlmodel._init;
      Caffe2NetTools::import_net(_init_net, init_net_path);
      Caffe2NetTools::import_net(_net, this->_mlmodel._predict);

      // Reconfigure
      create_model();

    } catch (std::exception &e) {
      this->_logger->error("Couldn't create model {}", e.what());
      // Must stay in a 'changed_state' until a call to create_model finally succeed.
      _state.force_init();
      throw;
    }
    // Save the current net configuration.
    _state.backup();
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  void Caffe2Lib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::
  dump_model_state() {
    caffe2::NetDef init;
    std::map<std::string, std::string> blobs;
    // Blobs and nets are formatted here, so mlmodel just have to manage the files
    _context.extract_state(blobs);
    _context.create_init_net(_train_net, init);
    this->_mlmodel.write_state(init, blobs);
    this->_logger->info("Dumped model state");
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  void Caffe2Lib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::
  init_mllib(const APIData &ad) {

    // Store the net's general configuration
    if (ad.has("inputlayer")) {
      _context._input_blob = ad.get("inputlayer").get<std::string>();
    }
    if (ad.has("outputlayer")) {
      apitools::get_vector(ad, "outputlayer", _context._output_blobs);
    }
    if (ad.has("nclasses")) {
      _context._nclasses = ad.get("nclasses").get<int>();
    }

    //XXX Get more informations (targets for multi-label, autoencoder, regression, ...)

    // Store the net's default (but variable) configuration
    set_gpu_state(ad, true);

    bool has_template = ad.has("template");
    if (has_template) {
      instantiate_template(ad);
    }

    // Check if the mandatory files are present
    if (this->_mlmodel._predict.empty() || this->_mlmodel._init.empty()) {
      throw MLLibInternalException
	(this->_mlmodel._repo + " does not contain the required files to initialize a net");
    }

    // Load the model
    Caffe2NetTools::import_net(_init_net, this->_mlmodel._init);
    Caffe2NetTools::import_net(_net, this->_mlmodel._predict);
    int current_nclasses = Caffe2NetTools::get_nclasses(_net, _init_net);

    // Keep the current shape
    if (!_context._nclasses) {
      _context._nclasses = current_nclasses;

      // Change the shape
    } else if (has_template) {

      // Reset parameters to ConstantFills, XaviersFills, etc.
      Caffe2NetTools::set_nclasses(_net, _init_net, _context._nclasses);
      Caffe2NetTools::export_net(_init_net, this->_mlmodel._init);

      // Assert the shape is correct
    } else {
      CAFFE_ENFORCE(_context._nclasses == current_nclasses);
    }

    // Now that every '_default' configuration values are defined, initialize the '_current' ones
    _state.reset();
    _last_inputc = this->_inputc;

    // Preconfigure the net to make predictions
    create_model();
    _state.backup();
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  void Caffe2Lib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::
  clear_mllib(const APIData &) {
    std::vector<std::string> extensions({".json"});
    fileops::remove_directory_files(this->_mlmodel._repo, extensions);
    this->_mlmodel.update_from_repository(this->_logger);
    std::vector<std::string> files({
	this->_mlmodel._init_state,
	this->_mlmodel._dbreader_state,
	this->_mlmodel._dbreader_train_state,
	this->_mlmodel._iter_state,
	this->_mlmodel._lr_state
    });
    for (const std::string &file : files) {
      if (!file.empty()) {
	remove(file.c_str());
      }
    }
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  float Caffe2Lib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::
  run_net(const std::string &net) {
    try {

      using namespace std::chrono;
      time_point<system_clock> start = system_clock::now();
      _context.run_net(net);
      return duration_cast<milliseconds>(system_clock::now() - start).count();

    } catch(std::exception &e) {
      this->_logger->error("Error while running the network: {}", e.what());
      throw;
    }
  }

  inline void extract_layers(const Caffe2NetTools::ModelContext &context,
			     std::vector<std::vector<std::vector<float>>> &results,
			     int batch_size,
			     const std::vector<std::string> &names,
			     const std::string &size_layer = "",
			     const std::vector<size_t> &scales = {}) {
    // Fetch the items size
    size_t layers = names.size();
    std::vector<std::vector<float>> sizes;
    bool has_sizes = !size_layer.empty();
    if (has_sizes) {
      CAFFE_ENFORCE(scales.size() == layers);
      sizes.resize(batch_size);
      context.extract(sizes, size_layer, std::vector<size_t>(batch_size, 1));
    }

    // Fetch the layers
    results.resize(layers);
    for (size_t i = 0; i < layers; ++i) {
      std::vector<std::vector<float>> &result(results[i]);
      const std::string &name(names[i]);

      // Fetch the items
      result.resize(batch_size);
      if (has_sizes) {
	size_t scale = scales[i];
	std::vector<size_t> scaled_sizes(batch_size);
	std::transform(sizes.begin(), sizes.end(), scaled_sizes.begin(),
		       [&](const std::vector<float> &size){ return scale * size[0]; });
	context.extract(result, name, scaled_sizes);
      } else {
	context.extract(result, name);
      }
    }
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  void Caffe2Lib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::
  extract_results(std::vector<std::vector<std::vector<float>>> &results, int batch_size) {

    std::string extract = _state.extract_layer();
    std::vector<std::string> outputs = _context._output_blobs;

    // raw layer
    if (!extract.empty()) {
      return extract_layers(_context, results, batch_size, { extract });
    }

    // scores, bboxes, classes, batch_splits
    if (_state.bbox()) {
      CAFFE_ENFORCE(outputs.size() == 4);
      std::string sizes = outputs.back();
      outputs.pop_back();
      return extract_layers(_context, results, batch_size, outputs, sizes, {1, 4, 1});
    }

    // Simple prediction
    CAFFE_ENFORCE(outputs.size() == 1);
    extract_layers(_context, results, batch_size, outputs);
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  int Caffe2Lib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::
  train(const APIData &ad, APIData &out) {
    set_train_mode(ad, true);

    APIData ad_mllib = ad.getobj("parameters").getobj("mllib");

    // Update GPUs tags
    set_gpu_state(ad_mllib);

    int iterations = 0;
    int test_interval = 0;
    int snapshot_interval = 0;

    // Update the solver's tags
    APIData ad_solver = ad_mllib.getobj("solver");
    if (ad_solver.size()) {

      // Learning rate, optimizer, etc.
      _state.set_lr_policy(ad_solver);
      _state.set_base_lr(ad_solver);
      _state.set_stepsize(ad_solver);
      _state.set_gamma(ad_solver);
      _state.set_power(ad_solver);
      _state.set_max_iter(ad_solver);
      _state.set_solver_type(ad_solver);
      _state.set_momentum(ad_solver);
      _state.set_rms_decay(ad_solver);

      // To be compatible with caffe's syntax, the 'solver_type' is allowed to be sent in uppercase
      std::string solver_type = _state.solver_type();
      std::transform(solver_type.begin(), solver_type.end(), solver_type.begin(), ::tolower);
      _state.set_solver_type(solver_type);

      //XXX Allow more parameters: decay, epsilon, ...

      // Iterations count
      if (ad_solver.has("iterations")) {
	iterations = ad_solver.get("iterations").get<int>();
      }
      if (ad_solver.has("test_interval")) {
	test_interval = ad_solver.get("test_interval").get<int>();
      }
      if (ad_solver.has("snapshot")) {
	snapshot_interval = ad_solver.get("snapshot").get<int>();
      }
    }

    if (!iterations) {
      //XXX Use a default number of epoch ? (e.g. 50 * _last_inputc.db_size())
      throw MLLibBadParamException("'iterations' must be set to a positive integer");
    }

    // Update the training's tags
    _state.set_is_testing(test_interval > 0 && _last_inputc.is_testable());
    _state.set_resume(ad_mllib);
    if (_state.resume()) {
      _state.force_init(); // When resuming, blobs like DBreaders and Iters must be updated
    }

    update_model(); // Recreate the net from protobuf files if the configuration has changed

    // Check if everything is well configured
    _last_inputc.assert_context_validity(_context, true);
    if (_state.is_testing()) {
      _last_inputc.assert_context_validity(_context);
    }

    // Start the training
    const int start_iter = _context.extract_iter();
    this->clear_all_meas_per_iter();
    this->_tjob_running = true;
    if (start_iter) {
      this->_logger->info("Training is restarting at iteration {}", start_iter);
    } else {
      this->_logger->info("Training is starting");
    }
    for (int iter = start_iter; iter < iterations && this->_tjob_running.load(); ++iter) {

      // Add measures
      float iter_time = run_net(_train_net.name());
      this->add_meas("iter_time", iter_time);
      this->add_meas("remain_time", iter_time * (iterations - iter) / 1000.0);
      this->add_meas("train_loss", _context.extract_loss());
      this->add_meas("iteration", iter);

      // Save snapshot
      if (snapshot_interval && iter > start_iter && !(iter % snapshot_interval)) {
	dump_model_state();
      }

      // Test the net
      if (_state.is_testing() && iter > start_iter && !(iter % test_interval)) {
	this->_logger->info("Testing model");

	APIData meas_out;
	test(ad, meas_out);
	APIData meas_obj = meas_out.getobj("measure");

	// Loop over the test measures
	for (auto meas : meas_obj.list_keys()) {
	  apitools::test_variants<double, std::vector<double>>
	    (meas_obj, meas,

	    // Single value
	    [&](const double &value) {
	      this->add_meas(meas, value);
	      this->add_meas_per_iter(meas, value);
	      this->_logger->info("{} = {}", meas, value);
	    },

	    // Multiple values
	    [&](const std::vector<double> &values) {
	      size_t nb_values = values.size();
	      std::vector<std::string> cls(nb_values);
	      this->_mlmodel.get_hcorresp(cls);
	      this->add_meas(meas, values, cls);
	      std::stringstream s;
	      for (size_t value = 0; value < nb_values; ++value) {
		s << (value ? ", " : "") << cls[value] << ": " << values[value];
	      }
	      this->_logger->info("{} = [ {} ]", meas, s.str());
	    });
	}
	this->_logger->info("Model tested");
      }
    }

    dump_model_state(); // Save the final snapshot
    if (this->_tjob_running.load()) { // Training stopped cleanly

      if (_state.is_testing()) {
	this->_logger->info("Adding final measures");
	test(ad, out);
      }

      this->_logger->info("Updating net parameters");
      caffe2::NetDef init;
      _context.create_init_net(_train_net, init);
      caffe2::WriteProtoToTextFile(init, this->_mlmodel._init);

      // Forward informations the input connector wants to send
      _last_inputc.response_params(out);
    }
    return 0;
  }

  //XXX Some code in common with 'predict', it should be possible to group it into another function
  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  void Caffe2Lib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::
  test(const APIData &ad, APIData &out) {

    //XXX Right now it means supervised single-label classification
    CAFFE_ENFORCE(!_state.bbox() && _context._output_blobs.size() == 1);

    // Add already computed measures
    APIData ad_res;
    if (_state.is_training()) {
      ad_res.add("iteration", this->get_meas("iteration"));
      ad_res.add("train_loss", this->get_meas("train_loss"));
    }

    //XXX This 'nout' value is supposed to be set accordingly to the type of prediction
    //(classification, regression, autoencoder, etc.)
    int nout = _context._nclasses;

    std::vector<std::string> clnames(nout);
    this->_mlmodel.get_hcorresp(clnames);
    ad_res.add("clnames", clnames);
    ad_res.add("nclasses", nout);
    //XXX True/False flags can be forwarded to ad_res (segmentation, multi_label, etc.)

    // Loop while there is data to test
    int batch_size, total_size = 0;
    while ((batch_size = _last_inputc.load_batch(_context, total_size))) {
      run_net(_net.name());

      // Extract results
      std::vector<std::vector<std::vector<float>>> results;
      extract_results(results, batch_size);

      //XXX Find how labels could be fetched when loading manually
      std::vector<float> labels(batch_size);
      _context.extract_labels(labels);

      // Loop on each item of the batch
      for (int item = 0; item < batch_size; ++item, ++total_size) {
	APIData bad;

	//XXX 'target' and 'pred' format will change depending on the prediction mode
	//(e.g. bbox or multi-label)
	const std::vector<float> &result = results[0][item];
	std::vector<double> predictions;
	predictions.assign(result.begin(), result.end());
	bad.add("target", labels[item]);
	bad.add("pred", predictions);

	ad_res.add(std::to_string(total_size), bad);
      }
    }

    // Compute the measures
    ad_res.add("batch_size", total_size);
    SupervisedOutput::measure(ad_res, ad.getobj("parameters").getobj("output"), out);
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  int Caffe2Lib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::
  predict(const APIData &ad, APIData &out) {
    set_train_mode(ad, false);

    TOutputConnectorStrategy tout;
    APIData ad_mllib = ad.getobj("parameters").getobj("mllib");
    APIData ad_output = ad.getobj("parameters").getobj("output");

    // Get the lower limit of confidence requested for an prediction to be acceptable
    float confidence_threshold = 0.0;
    if (ad_output.has("confidence_threshold")) {
      apitools::get_float(ad_output, "confidence_threshold", confidence_threshold);
    }

    _state.set_bbox(ad_output);

    //XXX More informations could be fetched in the future (bbox, rois, etc.)

    _state.set_extract_layer(ad_mllib); // The net will be truncated if this flag is present

    // Update GPUs tags
    set_gpu_state(ad_mllib);

    if (ad_output.has("measure")) {
      CAFFE_ENFORCE(_state.extract_layer().empty()); // A truncated net can't be mesured
      CAFFE_ENFORCE(_last_inputc.is_testable());
      _state.set_is_testing(true);
    }

    update_model(); // Recreate the net from protobuf files if the configuration has changed
    _last_inputc.assert_context_validity(_context); // Check if everything is well configured

    // Special case where the net is runned by test()
    if (_state.is_testing()) {
      test(ad, out);
      return 0;
    }

    // Extract results
    std::vector<APIData> vrad;

    int batch_size, total_size = 0;
    while ((batch_size = _last_inputc.load_batch(_context, total_size))) {
      run_net(_net.name());

      // Extract results
      std::vector<std::vector<std::vector<float>>> results;
      extract_results(results, batch_size);

      // Loop on each item of the input batch
      for (int item = 0; item < batch_size; ++item, ++total_size) {

	vrad.emplace_back();
	APIData &rad = vrad.back();
	rad.add("uri", _last_inputc.ids().at(total_size)); // Store its name
	rad.add("loss", 0.f); //XXX Needed but not relevant

	if (!_state.extract_layer().empty()) {

	  const std::vector<float> &result = results[0][item];
	  std::vector<double> vals(result.begin(), result.end()); // raw extracted layer
	  rad.add("vals", vals);

	} else if (_state.bbox()) {

	  // Fetch data
	  const std::vector<float> &scale = _last_inputc.scales().at(total_size);
	  const std::vector<float> &scores = results[0][item];
	  const std::vector<float> &coords = results[1][item];
	  const std::vector<float> &classes = results[2][item];
	  std::vector<double> probs;
	  std::vector<std::string> cats;
	  std::vector<APIData> bboxes;
	  CAFFE_ENFORCE(scores.size() == classes.size());
	  CAFFE_ENFORCE(scores.size() * 4 == coords.size());
	  auto coords_it = coords.begin();

	  for (size_t box = 0; box < scores.size(); ++box, coords_it += 4) {

	    // Ignore low score
	    float prob = scores[box];
	    if (prob < confidence_threshold) {
	      continue;
	    }

	    // Set the bounding box
	    APIData ad_bbox;
	    const std::vector<std::string> coord({"xmin", "ymin", "xmax", "ymax"});
	    const std::vector<float> coord_scale({scale[1], scale[0], scale[1], scale[0]});
	    for (int coord_idx = 0; coord_idx < 4; ++coord_idx) {
	      ad_bbox.add(coord[coord_idx], *(coords_it + coord_idx) / coord_scale[coord_idx]);
	    }

	    // Append the data
	    probs.push_back(prob);
	    cats.push_back(this->_mlmodel.get_hcorresp(classes[box]));
	    bboxes.push_back(ad_bbox);
	  }

	  rad.add("probs", probs);
	  rad.add("cats", cats);
	  rad.add("bboxes", bboxes);

	} else { //XXX for now this means supervised classification

	  const std::vector<float> &result = results[0][item];
	  std::vector<double> probs;
	  std::vector<std::string> cats;
	  for (size_t cls = 0; cls < result.size(); ++cls) {
	    float prob = result[cls];
	    if (prob < confidence_threshold)
	      continue;
	    probs.push_back(prob);
	    cats.push_back(this->_mlmodel.get_hcorresp(cls));
	  }
	  rad.add("probs", probs);
	  rad.add("cats", cats);

	}

	//XXX This if/else could contain more extraction types (segmentation, etc.)
      }
    }

    tout.add_results(vrad);

    out.add("bbox", bool(_state.bbox())); // ad_api_variants work with 'bool', not 'const bool &'

    //XXX More tags can be forwarded (regression, autoencoder, etc.)

    tout.finalize(ad.getobj("parameters").getobj("output"), out,
		  static_cast<MLModel*>(&this->_mlmodel));
    out.add("status", 0);
    return 0;
  }

  // Template instantiation
  template class Caffe2Lib<ImgCaffe2InputFileConn,SupervisedOutput,Caffe2Model>;
  template class Caffe2Lib<ImgCaffe2InputFileConn,UnsupervisedOutput,Caffe2Model>;
  //XXX Make more input connectors
}
