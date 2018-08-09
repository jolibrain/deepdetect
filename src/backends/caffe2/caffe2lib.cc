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
#include "outputconnectorstrategy.h"

// Just a few static lines of code to be sure Caffe2 is correctly initialized
namespace { class RunOnce { public: RunOnce() {

  int size = 2;
  const char *flags[size] = {
    "FLAGS"
    // As each service may want to use a different GPU,
    // We don't want any global variable to store the "current GPU id" in our Nets.
    ,"--caffe2_disable_implicit_engine_preference=1"
  };
  char **ptr = const_cast<char **>(&flags[0]);
  caffe2::GlobalInit(&size, &ptr);

}}; static RunOnce _; }

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
  static bool get_api_variant(const ad_variant_type &adv, const std::function<void(const T&)> &f) {
    if (!adv.is<T>()) {
      return false;
    }
    f(adv.get<T>());
    return true;
  }

  // Simple way to retrieve a value from an APIData when the type is uncertain
  template <typename T1, typename T2>
  static void test_api_variants(const APIData &ad, const std::string &name,
				const std::function<void(const T1&)> &f1,
				const std::function<void(const T2&)> &f2) {
    const ad_variant_type &adv = ad.get(name);
    if (get_api_variant<T1>(adv, f1));
    else if (get_api_variant<T2>(adv, f2));
    else throw std::runtime_error("Invalid type for '" + std::string(name) + "'");
  }

#define TEST_API_VARIANTS(ad, name, t1, code1, t2, code2)		\
  test_api_variants<t1, t2>(ad, name, [&](const t1 &value) {code1;}, [&](const t2 &value) {code2;});

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
      TEST_API_VARIANTS(ad, "gpuid",
			int,			ids = { value },
			std::vector<int>,	ids = value);

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
    this->_mlmodel.update_from_repository(this->_logger);

    // Assert that there is no risk of erasing model files
    if (!this->_mlmodel._init.empty() && !this->_mlmodel._predict.empty()) {
      throw MLLibBadParamException("using template while model files already exists, "
				   "remove 'template' from 'mllib', "
				   "or remove existing '.pb' files ?");
    }

    //XXX Implement build-in template creation (mlp, convnet, resnet, etc.)

    std::string model_tmpl = ad.get("template").get<std::string>();
    std::string source = this->_mlmodel._mlmodel_template_repo + '/' + model_tmpl;
    std::vector<std::string> files({"predict_net.pb", "init_net.pb"});
    this->_logger->info("instantiating model template {}", model_tmpl);
    this->_logger->info("source = {}", source);
    this->_logger->info("dest = {}", this->_mlmodel._repo);

    // Copy files
    for (const std::string &file : files) {
      std::string src = source + "/" + file;
      std::string dst = this->_mlmodel._repo + "/" + file;
      switch (fileops::copy_file(src, dst)) {
      case 1: throw MLLibBadParamException("failed to locate model template " + src);
      case 2: throw MLLibBadParamException("failed to create model template destination " + dst);
      }
    }

    //XXX Configure the nets protobuf files

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
    try {
      inputc.transform(ad);
    } catch (...) {
      this->_logger->info("Could not configure the InputConnector");
      throw;
    }

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

      // Reset parameters to ConstantFills, XaviersFills, etc.
      Caffe2NetTools::reset_fillers(_net, _init_net);
      if (_context._nclasses) { //XXX Check if classifying
	Caffe2NetTools::set_nclasses(_net, _init_net, _context._nclasses);
      }
      _last_inputc.create_dbreader(init_net, true);
      if (_state.is_testing()) {
	_last_inputc.create_dbreader(init_net);
      }

    }

    // Load batches from the databases
    _last_inputc.add_tensor_loader(_context, train_net, true);
    if (_state.is_testing()) {
      _last_inputc.add_tensor_loader(_context, test_net);
    }

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

    Caffe2NetTools::insert_learning_operators(_context, train_net, init_net,
					      _state.lr_policy(), _state.base_lr(),
					      _state.stepsize(), _state.gamma());
    Caffe2NetTools::get_optimizer(_state.solver_type())(_context, train_net, init_net);

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

    if (_last_inputc.is_load_manual()) {

      // usually the init net only initiliaze the parameters
      // so we create the 'data' blob manually, just to be sure
      _context.create_input();
      tmp_net.add_external_input(_context._input_blob);

    } else {

      // Add database informations
      _last_inputc.create_dbreader(_init_net);
      _last_inputc.add_tensor_loader(_context, tmp_net);

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

    // Check if the mandatory files are present
    this->_mlmodel.update_from_repository(this->_logger);
    if (this->_mlmodel._predict.empty() || this->_mlmodel._init.empty()) {
      throw MLLibInternalException(this->_mlmodel._repo +
				   " does not contain the required files to initialize a net");
    }

    this->_logger->info("Re-creating the nets");

    // Load the nets protobuffers
    std::string init_net_path = _state.is_training() && _state.resume() ?
      this->_mlmodel._init_state : this->_mlmodel._init;
    CAFFE_ENFORCE(caffe2::ReadProtoFromFile(init_net_path, &_init_net));
    CAFFE_ENFORCE(caffe2::ReadProtoFromFile(this->_mlmodel._predict, &_net));

    // _input_blob and _output_blob should have been initialized during init_mllib
    // (by "inputlayer" and "outputlayer" respectively)
    // If the are not, we'll do a guess based on the following conventions:
    //    - the input blob name is (or contains) "data"
    //    - the external inputs are sorted
    //    - all the outputs are created by the last operator
    if (_context._output_blob.empty()) {
      const auto &outputs = _net.op(_net.op().size() - 1).output();
      _context._output_blob = outputs[0];
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
    _net.set_name("_net");
    _init_net.set_name("_init_net");
    _train_net.set_name("_train_net");

    // Update the workspace with the new nets
    _context.run_net_once(_init_net);
    if (!_state.is_training() || _state.is_testing()) {
      if (!_context._nclasses) { //XXX Check if classifying
	// If not initiliazied by init_mllib, it will be infered
	_context._nclasses = Caffe2NetTools::get_nclasses(_net, _init_net);
      }
      _context.create_net(_net);
    }
    if (_state.is_training()) {
      _context.create_net(_train_net);
    }

    this->_logger->info("Nets updated");
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  void Caffe2Lib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::
  update_model() {

    if (!_state.changed()) {
      this->_logger->info("Reseting the workspace");
      _context.run_net_once(_init_net);
      return;
    }

    // Reload from the disk and reconfigure.
    try {
      create_model();
    } catch (...) {
      this->_logger->error("Couldn't create model");
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
      _context._output_blob = ad.get("outputlayer").get<std::string>();
    }
    if (ad.has("nclasses")) {
      _context._nclasses = ad.get("nclasses").get<int>();
    }

    //XXX Get more informations (targets for multi-label, autoencoder, regression, ...)

    // Store the net's default (but variable) configuration
    set_gpu_state(ad, true);

    if (ad.has("template")) {
      instantiate_template(ad);
    }

    // Now that every '_default' values are defined, initialize the '_current' ones
    _state.reset();
    _last_inputc = this->_inputc;

    // Preconfigure the net to make predictions
    if (!this->_mlmodel._predict.empty()) {
      create_model();
      _state.backup();
    }
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
  run_net(const std::string &net, std::vector<std::vector<float>> *results) {
    try {

      std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
      _context.run_net(net);

      if (results) {
	// Fill the vector with the output layer
	// Extract layer is only used if not empty
	_context.extract_results(*results, _state.extract_layer());
      }

      std::chrono::time_point<std::chrono::system_clock> stop = std::chrono::system_clock::now();
      return std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();

    } catch(std::exception &e) {
      this->_logger->error("Error while running the network, not enough memory? {}", e.what());
      throw;
    }
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
      _state.set_solver_type(ad_solver);

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

    // Start the training
    const int start_iter = _context.extract_iter();
    this->clear_all_meas_per_iter();
    this->_tjob_running = true;
    if (start_iter) {
      this->_logger->info("Training is restarting at iteration: " + std::to_string(start_iter));
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
	for (auto meas : meas_obj.list_keys()) { // Add test measures
	  TEST_API_VARIANTS(meas_obj, meas,
			    double,			this->add_meas(meas, value);
							this->add_meas_per_iter(meas, value),
			    std::vector<double>,	std::vector<std::string> cls(value.size());
							this->_mlmodel.get_hcorresp(cls);
							this->add_meas(meas, value, cls));
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

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  void Caffe2Lib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::
  test(const APIData &ad, APIData &out) {

    int batch_size = 0, total_size = 0;

    // Get sizes used by tensor loaders
    if (!_last_inputc.is_load_manual()) {
      _last_inputc.get_tensor_loader_infos(batch_size, total_size);
      int remaining = total_size % batch_size;
      if (remaining) {
	this->_logger->warn("The last {} image(s) will be ignored, as they won't fill"
			    " the requested batch size of {}",
			    remaining, batch_size);
      }
      total_size -= remaining;
    }

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
    for (int i = 0; true; i += batch_size) {

      if (_last_inputc.is_load_manual()) {

	// Get the current batch size and fill the input blob
	batch_size = _last_inputc.load_batch(_context);
	total_size += batch_size;

      } else {
	batch_size *= (i < total_size);
      }
      if (!batch_size) break; // Everything was tested

      std::vector<std::vector<float>> results(batch_size);
      run_net(_net.name(), &results);

      //XXX Find how labels could be fetched when loading manually
      //XXX What about other types of prediction ? Multi-labels ?
      std::vector<float> labels(batch_size);
      _context.extract_labels(labels);

      // Loop on each batch
      for (int j = 0; j < batch_size; j++) {
	APIData bad;
	std::vector<double> predictions;

	//XXX Here again 'target' and 'pred' format will change depending on the prediction mode
	//(e.g. multi-label)
	predictions.assign(results[j].begin(), results[j].end());
	bad.add("target", labels[j]);
	bad.add("pred", predictions);
	ad_res.add(std::to_string(i + j), bad);
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
      TEST_API_VARIANTS(ad_output, "confidence_threshold",
			double,	confidence_threshold = value,
			int,	confidence_threshold = static_cast<float>(value));
    }

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

    // Special case where the net is runned by test()
    if (_state.is_testing()) {
      test(ad, out);
      return 0;
    }

    // Load everything in a single batch
    CAFFE_ENFORCE(_last_inputc.is_load_manual());
    int batch_size = _last_inputc.load_batch(_context);

    std::vector<APIData> vrad(batch_size);
    std::vector<std::vector<float> > results(batch_size);
    run_net(_net.name(), &results);

    // Loop on each item of the input batch
    for (int i = 0; i < batch_size; ++i) {
      const std::vector<float> &result = results[i];
      APIData &rad = vrad[i];

      rad.add("uri", _last_inputc.ids().at(i)); // Store its name
      rad.add("loss", 0.f); //XXX Needed but not relevant

      if (_state.extract_layer().empty()) { //XXX for now this means supervised classification

	std::vector<double> probs;
	std::vector<std::string> cats;
	for (size_t j = 0; j < result.size(); ++j) {
	  float prob = result[j];
	  if (prob < confidence_threshold)
	    continue;
	  probs.push_back(prob);
	  cats.push_back(this->_mlmodel.get_hcorresp(j));
	}
	rad.add("probs", probs);
	rad.add("cats", cats);

      } else {
	std::vector<double> vals(result.begin(), result.end()); // raw extracted layer
	rad.add("vals", vals);
      }

      //XXX This if/else could contain more extraction types (segmentation, bbox, etc.)

      vrad.push_back(rad);
    }

    tout.add_results(vrad);

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
