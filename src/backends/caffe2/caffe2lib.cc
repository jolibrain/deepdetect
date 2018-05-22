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

//XXX Is only used in one short function that should disapear one day
#include <caffe2/core/db.h>

#pragma GCC diagnostic pop

#include <caffe2/core/init.h>
#include "imginputfileconn.h"
#include "backends/caffe2/caffe2lib.h"
#include "backends/caffe2/nettools.h"
#include "outputconnectorstrategy.h"

//XXX Find a better way to init caffe2
static void init_caffe2_flags() {

  static bool init = false;
  if (init) return;
  init = true;

  int size = 2;
  const char *flags[size] = {
    "FLAGS"

    // As each service may want to use a different GPU,
    // We don't want any global variable to store the "current GPU id" in our Nets.
    ,"--caffe2_disable_implicit_engine_preference=1"

  };
  char **ptr = const_cast<char **>(&flags[0]);
  caffe2::GlobalInit(&size, &ptr);
}

//XXX Move every label-related operation somewhere in Caffe2NetTools (e.g. like 'iter' and 'lr')
const std::string blob_label("label");

//XXX Find a clean way to manage databases inputs from within caffe2inputconnectors
// -> dbreaders, labels, mean_values, etc.
const std::string blob_dbreader("dbreader");
const std::string blob_dbreader_test("dbreader_test");

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

    _workspace = std::move(c2l._workspace);
    _devices = std::move(c2l._devices);
    _init_net = std::move(c2l._init_net);
    _test_net = std::move(c2l._test_net);
    _net = std::move(c2l._net);
    _state = c2l._state;

    _input_blob = c2l._input_blob;
    _output_blob = c2l._output_blob;
    _nclasses = c2l._nclasses;
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  Caffe2Lib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::~Caffe2Lib() {}

  //Define some GPY-only code
#ifdef CPU_ONLY
#define _UPDATE_GPU_STATE(...)
#else

  static void get_gpu_ids(const APIData &ad, std::vector<int> &ids) {

    // Fetch ids from the ApiData
    try { //XXX Is a try catch the only way ?
      ids = { ad.get("gpuid").get<int>() };
    } catch (std::exception &e) {
      ids = ad.get("gpuid").get<std::vector<int>>();
    }

    // By default, use every GPUs
    if (ids.size() == 1 && ids[0] == -1) {
      ids.clear();
      int count_gpus = 0;
      cudaGetDeviceCount(&count_gpus);
      ids.resize(count_gpus);
      std::iota(ids.begin(), ids.end(), 0);
    }
    CAFFE_ENFORCE(!ids.empty());
  }

  // Used when initiliazing, training and predicting
#define _UPDATE_GPU_STATE(ad, dft)					\
  if (ad.has("gpuid")) {						\
    std::vector<int> ids;						\
    get_gpu_ids(ad, ids);						\
    _state.set##dft##_gpu_ids(ids);					\
    _state.set##dft##_is_gpu(true);					\
  }									\
  if (ad.has("gpu")) {							\
    _state.set##dft##_is_gpu(ad.get("gpu").get<bool>());		\
  }
#endif

#define UPDATE_GPU_STATE(...)		_UPDATE_GPU_STATE(__VA_ARGS__,)
#define UPDATE_GPU_DEFAULT_STATE(...)	_UPDATE_GPU_STATE(__VA_ARGS__, _default)

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

    //XXX Implement build-in template creation (mlp, convnet, reset, etc.)

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

    // Update the mlmodel
    this->_mlmodel._model_template = model_tmpl;
    this->_mlmodel.update_from_repository(this->_logger);
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  void Caffe2Lib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::
  create_model_train() {

    // Computation is done on new nets
    caffe2::NetDef init_net, train_net, test_net;
    Caffe2NetTools::ScopedNet
      init_net_scoped(init_net),
      train_net_scoped(train_net),
      test_net_scoped(test_net);

    train_net_scoped._devices = _devices;
    init_net_scoped._devices = _devices;
    test_net_scoped._devices = { _devices[0] };

    //XXX Find a way to prevent code duplication when applying changes to the test net
    bool testing = _state.is_testing(), resume = _state.resume();
    std::string train_db = _state.train_db(), test_db = _state.test_db();

    // Reset parameters to ConstantFills, XaviersFills, etc.
    if (!_state.resume()) {
      Caffe2NetTools::reset_fillers(_net, _init_net);
    }

    if (train_db.empty()) {
      //XXX Define how to setup a net without a database (use inputconnector with a list of data ?)
      throw MLLibBadParamException("training without a database");
    } else {

      // Check if the net is testable
      if (testing && test_db.empty()) {
	testing = false;
	_state.set_is_testing(testing);
	this->_logger->warn("Cannot test without a testing database");
      }

      // Load or create one dbreader per net
      if (resume) {

	Caffe2NetTools::load_blob(*_workspace, this->_mlmodel._dbreader_state, blob_dbreader);
	if (testing) {
	  Caffe2NetTools::load_blob(*_workspace, this->_mlmodel._dbreader_test_state,
				    blob_dbreader_test);
	}

      } else {
	Caffe2NetTools::CreateDB(init_net, blob_dbreader, train_db);
	if (testing) {
	  Caffe2NetTools::CreateDB(init_net, blob_dbreader_test, test_db);
	}
      }

      // Check if the inputs need image-related layers
      caffe2::OperatorDef input;
      std::string input_name(_input_blob);
      bool is_image(typeid(this->_inputc) == typeid(ImgCaffe2InputFileConn));
      bool is_nhwc(is_image // ImageInput operators format data in NHWC
#ifndef CPU_ONLY
		   && !_state.is_gpu() // But gpu_transform can correct it
#endif
		   );

      if (is_nhwc) {
	input_name += "_nhwv"; // NHWC2NCHW is not an in-place operator
      }

      // Create a tensor loader
      //XXX There is currently no tool to load unlabeled data (supervised only)
      //XXX Divide the batch_size by the number of devices ?
      Caffe2NetTools::TensorProtosDBInput(input, blob_dbreader, input_name, blob_label,
					  _state.batch_size());
      if (is_image) {
	reinterpret_cast<ImgCaffe2InputFileConn &>(this->_inputc)
	  .configure_db_operator(input, _state.mean_values()); // Special operator for images
#ifndef CPU_ONLY
	if (_state.is_gpu()) {
	  Caffe2NetTools::add_arg(input, "use_gpu_transform", 1);
	  Caffe2NetTools::add_arg(input, "order", "NCHW");
	}
#endif
      }

      // Add the previously created operators on each devices
      Caffe2NetTools::insert_db_input_operator(train_net_scoped, input);
      if (testing) {
	// The test net has a different db and batch size
	*input.mutable_input(0) = blob_dbreader_test;
	Caffe2NetTools::replace_arg(input, "batch_size", _state.test_batch_size());
	Caffe2NetTools::insert_db_input_operator(test_net_scoped, input);
      }

      if (is_nhwc) {
	Caffe2NetTools::NHWC2NCHW(train_net_scoped, input_name, _input_blob);
	if (testing) {
	  Caffe2NetTools::NHWC2NCHW(test_net_scoped, input_name, _input_blob);
	}
      }

    }

    // Prevent the backward pass from reaching the database and input-fomatting operators
    Caffe2NetTools::StopGradient(train_net_scoped, _input_blob);

    // Add requested operators on each devices
    for (const caffe2::OperatorDef &op : _net.op()) {
      Caffe2NetTools::add_op(train_net_scoped, op);
      if (testing) {
	Caffe2NetTools::add_op(test_net_scoped, op);
      }
    }

    // Assert that the test net output has the desired name
    // (e.g. a previous scope may have renamed it to gpu_0/...)
    if (testing) {
      *test_net.mutable_op(test_net.op().size() - 1)->mutable_output(0) = _output_blob;
    }

    // Copy the external intputs (name of the parameters) on each device
    // (the 'main' input should not be copied if remplaced by a db-input)
    for (const std::string &input : _net.external_input()) {
      //XXX The day no-db training is implemented, this will (probably) need to be modified
      if (input != _input_blob || train_db.empty()) {
	Caffe2NetTools::add_external_input(train_net_scoped, input);
	if (testing) {
	  Caffe2NetTools::add_external_input(test_net_scoped, input);
	}
      }
    }

    // The test net is complete
    // Below is the training-only stuff

    // Add loss, gradient, and sum over the devices
    //XXX There is currently no tool to compute loss without labels (it's supervised only)
    Caffe2NetTools::insert_loss_operators(train_net_scoped, _output_blob, blob_label);
    Caffe2NetTools::add_gradient_ops(train_net);
#ifndef CPU_ONLY
    if (train_net_scoped._devices.size() > 1) {
      Caffe2NetTools::reduce(train_net_scoped);
    }
#endif

    // Set the learning-related operators and blobs to the desired iteration
    int iter = resume ? Caffe2NetTools::load_iter(this->_mlmodel._iter_state) : 0;

    Caffe2NetTools::insert_learning_operators(train_net_scoped, init_net_scoped, iter,
					      _state.lr_policy(), _state.base_lr(),
					      _state.stepsize(), _state.gamma());
    Caffe2NetTools::get_optimizer(_state.solver_type())(train_net_scoped, init_net_scoped);

    if (resume) {
      // Prefill the learning_rate with its old value
      Caffe2NetTools::load_lr(*_workspace, _devices, this->_mlmodel._lr_state);
    }

    // Train net is complete

    // Duplicate the init net outputs on all the devices
    Caffe2NetTools::copy_and_broadcast_operators(init_net_scoped, _init_net);

    // Apply changes
    _net.Swap(&train_net);
    _init_net.Swap(&init_net);
    _test_net.Swap(&test_net);
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  void Caffe2Lib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::
  create_model_predict() {

    // usually the init net only initiliaze the parameters
    // so we create the 'data' blob manually, just to be sure
    _workspace->CreateBlob(_input_blob);

    std::string extract = _state.extract_layer();
    if (!extract.empty()) { // unsupervised
      // Operators placed after the extracted layer are removed (they would do useless computation)
      Caffe2NetTools::truncate_net(_net, extract);
    }

    // Set the device for every operators
    Caffe2NetTools::set_net_device(_net, _devices[0]);
    Caffe2NetTools::set_net_device(_init_net, _devices[0]);
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

    this->_logger->info("Re-creating the net");

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
    //    - there is only one external output
    if (_output_blob.empty()) {
      _output_blob = _net.external_output()[0];
    }
    if (_input_blob.empty()) {
      const auto &inputs = _net.external_input();
      _input_blob = inputs[0];
      if (_input_blob.find("data") == std::string::npos) {
	_input_blob = inputs[inputs.size() - 1];
	if (_input_blob.find("data") == std::string::npos) {
	  _input_blob = "data";
	}
      }
    }

    // Reset the workspace and devices
    _workspace.reset(new caffe2::Workspace);
    _devices.clear();
    caffe2::DeviceOption option;
#ifndef CPU_ONLY
    if (_state.is_gpu()) {
      option.set_device_type(caffe2::CUDA);
      for (int i : _state.gpu_ids()) {
	this->_logger->info("Using GPU {}", i);
	option.set_cuda_gpu_id(i);
	_devices.push_back(option);
      }
    } else
#endif
      {
	option.set_device_type(caffe2::CPU);
	_devices.push_back(option);
      }

    // Transform the nets
    if (_state.is_training()) {
      create_model_train();
    } else {
      create_model_predict();
    }

    // Update the workspace with the new nets
    _net.set_name("_net");
    CAFFE_ENFORCE(_workspace->RunNetOnce(_init_net));
    CAFFE_ENFORCE(_workspace->CreateNet(_net));
    if (_state.is_testing()) {
      _test_net.set_name("_test_net");
      CAFFE_ENFORCE(_workspace->CreateNet(_test_net));
      //XXX Check how _nclasses could be infered here
    }

    this->_logger->info("Net updated");
  }

  // Just a if statement used both before training and before predicting
  // May become a function if used more often
#define UPDATE_MODEL()							\
  if (_state.changed()) {						\
    try {								\
      create_model(); /* Reload from the disk and reconfigure. */	\
    } catch (...) {							\
      this->_logger->error("Error creating model");			\
      /* Must stay in a 'changed_state' until a call to create_model finally succeed. */ \
      _state.force_init();						\
      throw;								\
    }									\
    /* Save the current net configuration. */				\
    _state.backup();							\
  } else {								\
    CAFFE_ENFORCE(_workspace->RunNetOnce(_init_net)); /* Update the workspace */ \
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  void Caffe2Lib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::
  dump_model_state() {
    caffe2::NetDef init;
    std::map<std::string, std::string> blobs;
    // Blobs and nets are formatted here, so mlmodel just have to manage the files
    Caffe2NetTools::extract_state(*_workspace, _devices[0], blobs);
    Caffe2NetTools::create_init_net(*_workspace, _devices[0], _net, init);
    this->_mlmodel.write_state(init, blobs);
    this->_logger->info("Dumped model state");
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  void Caffe2Lib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::
  init_mllib(const APIData &ad) {
    init_caffe2_flags(); //XXX Find a better place to init caffe2

    // Store the net's general configuration
    if (ad.has("inputlayer")) {
      _input_blob = ad.get("inputlayer").get<std::string>();
    }
    if (ad.has("outputlayer")) {
      _output_blob = ad.get("outputlayer").get<std::string>();
    }

    if (ad.has("nclasses")) {
      _nclasses = ad.get("nclasses").get<int>();
    }

    // Store the net's default (but variable) configuration
    UPDATE_GPU_DEFAULT_STATE(ad);

    if (ad.has("template")) {
      instantiate_template(ad);
    }

    // Now that every '_default' values are defined, initialize the '_current' ones
    _state.reset();
    if (!this->_mlmodel._predict.empty()) {
      create_model();
      _state.backup();
    }
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  void Caffe2Lib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::
  clear_mllib(const APIData &) {
    std::vector<std::string> extensions({".json"});
    //XXX remove _state files ? the databases ?
    fileops::remove_directory_files(this->_mlmodel._repo, extensions);
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  float Caffe2Lib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::
  run_net(const std::string &net, std::vector<std::vector<float>> *results) {
    try {

      std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
      CAFFE_ENFORCE(_workspace->RunNet(net));

      if (results) { // Fill the vector with the output layer

	std::string output = _state.extract_layer();
	if (output.empty()) { // supervised
	  output = _output_blob;
	}
	caffe2::TensorCPU tensor;
	//XXX Support multi-device
	CAFFE_ENFORCE(Caffe2NetTools::extract_tensor(*_workspace, _devices[0], output, tensor));

	int result_size = tensor.size() / results->size();
	const float *data = tensor.data<float>();
	for (std::vector<float> &result : *results) {
	  result.assign(data, data + result_size);
	  data += result_size;
	}
      }

      std::chrono::time_point<std::chrono::system_clock> stop = std::chrono::system_clock::now();
      return std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();

    } catch(std::exception &e) {
      this->_logger->error("Error while running the network, not enough memory? {}", e.what());
      throw;
    }
  }

  //XXX Find a better way to determine the number of iterations
  static int get_db_size(const std::string &path) {
    std::unique_ptr<caffe2::db::DB> db(caffe2::db::CreateDB("lmdb", path,  caffe2::db::READ));
    std::unique_ptr<caffe2::db::Cursor> cursor(db->NewCursor());
    int count = 0;
    for (; cursor->Valid(); ++count, cursor->Next());
    return count;
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  int Caffe2Lib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::
  train(const APIData &ad, APIData &out) {
    // Reseting to the default configuration
    _state.reset();
    _state.set_is_training(true);

    // Retreive the input-related configuration
    TInputConnectorStrategy inputc(this->_inputc);
    inputc._train = true;
    APIData cad = ad;
    try {
      inputc.transform(cad);
    } catch (...) {
      this->_logger->info("Could not retrieve InputConnector's informations");
      throw;
    }

    // Update databases tags
    if (cad.has("db")) {
      APIData db = cad.getobj("db");

      // Db paths must be set (even if empty)
      _state.set_train_db(db, true);
      _state.set_test_db(db, true);

      // Mean values are only set when using an InputImgConnector
      _state.set_mean_values(db);
    }

    APIData ad_mllib = cad.getobj("parameters").getobj("mllib");

    // Update GPUs tags
    UPDATE_GPU_STATE(ad_mllib);

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

      throw MLLibBadParamException("'iterations' must be set to a positive integer");

      //XXX Use a default number of iterations ?
      const std::string &db = _state.train_db();
      if (db.empty()) {
	iterations = 100000;
      } else {
	iterations = 50 * get_db_size(db);
      }
    }

    // Update the net's tags
    if (ad_mllib.has("net")) {
      APIData ad_net = ad_mllib.getobj("net");
      _state.set_batch_size(ad_net);
      _state.set_test_batch_size(ad_net);
    }

    // Update the training's tags
    _state.set_is_testing(test_interval > 0);
    _state.set_resume(ad_mllib);
    if (_state.resume()) {
      _state.force_init(); // When resuming, blobs like DBreaders and Iters must be updated
    }

    UPDATE_MODEL(); // Recreate the net from protobuf files if the configuration has changed

    // Start the training
    const int start_iter = Caffe2NetTools::extract_iter(*_workspace, _devices[0]);
    this->clear_all_meas_per_iter();
    this->_tjob_running = true;
    if (start_iter) {
      this->_logger->info("Training is restarting at iteration: " + std::to_string(start_iter));
    }
    for (int iter = start_iter; iter < iterations && this->_tjob_running.load(); ++iter) {

      // Add measures
      float iter_time = run_net(_net.name());
      this->add_meas("iter_time", iter_time);
      this->add_meas("remain_time", iter_time * (iterations - iter) / 1000.0);
      this->add_meas("train_loss", Caffe2NetTools::extract_loss(*_workspace, _devices));
      this->add_meas("iteration", iter);

      // Save snapshot
      if (snapshot_interval && iter > start_iter && !(iter % snapshot_interval)) {
	dump_model_state();
      }

      // Test the net
      if (_state.is_testing() && iter > start_iter && !(iter % test_interval)) {
	this->_logger->info("Testing model");

	APIData meas_out;
	test(_test_net.name(), ad, inputc, meas_out);
	APIData meas_obj = meas_out.getobj("measure");
	for (auto meas : meas_obj.list_keys()) { // Add test measures

	  try { //XXX Is a try catch the only way ?
	    double mval = meas_obj.get(meas).get<double>();
	    this->add_meas(meas, mval);
	    this->add_meas_per_iter(meas, mval);
	  } catch(std::exception &e) {
	    this->add_meas(meas, meas_obj.get(meas).get<std::vector<double>>());
	  }

	}
	this->_logger->info("Model tested");
      }
    }

    dump_model_state(); // Save the final snapshot
    if (this->_tjob_running.load()) { // Training stopped cleanly

      if (_state.is_testing()) {
	this->_logger->info("Adding final measures");
	test(_test_net.name(), ad, inputc, out);
      }

      this->_logger->info("Updating net parameters");
      caffe2::NetDef init;
      Caffe2NetTools::create_init_net(*_workspace, _devices[0], _net, init);
      caffe2::WriteProtoToTextFile(init, this->_mlmodel._init);

      inputc.response_params(out);
    }
    return 0;
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  int Caffe2Lib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::
  load_batch(TInputConnectorStrategy &inputc, int batch_size) {
    try {
      caffe2::TensorCPU tensor;
      batch_size = inputc.get_batch(tensor, batch_size);
      if (batch_size > 0) {
	//XXX Support multi-device
	Caffe2NetTools::insert_tensor(*_workspace, _devices[0], _input_blob, tensor);
      }
    } catch(std::exception &e) {
      this->_logger->error("exception while filling up network");
      throw;
    }
    return batch_size;
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  void Caffe2Lib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::
  test(const std::string &net,
       const APIData &ad,
       TInputConnectorStrategy &inputc,
       APIData &out) {

    // Retreive the set configuration
    int batch_size = _state.test_batch_size();
    int is_training = _state.is_training();
    int data_count = 0;
    int total_size = 0;
    if (is_training) {
      total_size = get_db_size(_state.test_db());
    }

    // Add already computed measures
    APIData ad_res;
    if (is_training) {
      ad_res.add("iteration", this->get_meas("iteration"));
      ad_res.add("train_loss", this->get_meas("train_loss"));
    }

    //XXX This 'nout' value is supposed to be set accordingly to the type of prediction
    //(classification, regression, autoencoder, etc.)
    int nout = _nclasses;

    std::vector<std::string> clnames;
    for (int i = 0; i < nout; i++)
      clnames.push_back(this->_mlmodel.get_hcorresp(i));
    ad_res.add("clnames", clnames);
    ad_res.add("nclasses", nout);

    // Loop while there is data to test
    for (int i = 0; true; i += data_count) {

      // Get the current batch size, and fill the input blob if not using a database
      if (is_training) {
	data_count = std::min(batch_size, total_size - i);
      } else {
	data_count = load_batch(inputc, batch_size);
	total_size += data_count;
      }
      if (!data_count) break; // Everything was tested

      std::vector<float> labels;
      std::vector<std::vector<float>> results(data_count);

      run_net(net, &results);
      { //XXX Move this code in a tool or function
	//(will need to change if used with multiple-devices)
	caffe2::TensorCPU tensor;
	std::string label = Caffe2NetTools::get_device_prefix(_devices[0]) + blob_label;
	CAFFE_ENFORCE(Caffe2NetTools::extract_tensor(*_workspace, _devices[0], label, tensor));
	const int *data = tensor.data<int>();
	labels.assign(data, data + tensor.size());
      }

      // Loop on each batch
      for (int j = 0; j < data_count; j++) {
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
    // Reseting to the default configuration
    _state.reset();
    _state.set_is_training(false);

    TInputConnectorStrategy inputc(this->_inputc);
    TOutputConnectorStrategy tout;
    APIData ad_mllib = ad.getobj("parameters").getobj("mllib");
    APIData ad_output = ad.getobj("parameters").getobj("output");

    // Get the lower limit of confidence requested for an prediction to be acceptable
    float confidence_threshold = 0.0;
    if (ad_output.has("confidence_threshold")) {
      try { //XXX Is a try catch the only way ?
	confidence_threshold = ad_output.get("confidence_threshold").get<float>();
      } catch (std::exception &e) {
	confidence_threshold = static_cast<float>(ad_output.get("confidence_threshold").get<int>());
      }
    }

    _state.set_extract_layer(ad_mllib); // The net will be truncated if this flag is present

    // Update GPUs tags
    UPDATE_GPU_STATE(ad_mllib);

    // Add informations that the InputConnector may need
    APIData cad = ad;
    if (this->_mlmodel._meanfile.size()) {
      cad.add("mean_file", this->_mlmodel._meanfile);
    }

    // Retreive the input-related configuration
    try {
      inputc.transform(cad);
    } catch (std::exception &e) {
      this->_logger->info("Could not retrieve InputConnector's informations");
      throw;
    }

    UPDATE_MODEL(); // Recreate the net from protobuf files if the configuration has changed

    // Special case where the net is runned by test()
    if (ad_output.has("measure")) {
      CAFFE_ENFORCE(_state.extract_layer().empty()); // A truncated net can't be mesured
      if (ad_mllib.has("net")) {
	_state.set_test_batch_size(ad_mllib.getobj("net"));
      }
      test(_net.name(), ad, inputc, out);
      return 0;
    }

    // Load everything in a single batch
    std::vector<APIData> vrad;
    int data_count(load_batch(inputc));
    std::vector<std::vector<float> > results(data_count);

    run_net(_net.name(), &results);

    // Loop on each item of the input batch
    for (const std::vector<float> &result : results) {
      APIData rad;

      // Store its name
      if (!inputc._ids.empty()) {
	rad.add("uri", inputc._ids.at(vrad.size()));
      } else {
	rad.add("uri", std::to_string(vrad.size()));
      }
      rad.add("loss", 0.f); //XXX Needed but not relevant

      if (_state.extract_layer().empty()) { //XXX for now this means supervised classification

	std::vector<double> probs;
	std::vector<std::string> cats;
	for (size_t i = 0; i < result.size(); ++i) {
	  float prob = result[i];
	  if (prob < confidence_threshold)
	    continue;
	  probs.push_back(prob);
	  cats.push_back(this->_mlmodel.get_hcorresp(i));
	}
	rad.add("probs", probs);
	rad.add("cats", cats);

      } else {
	std::vector<double> vals(result.begin(), result.end()); // raw extracted layer
	rad.add("vals", vals);
      }

      vrad.push_back(rad);
    }

    tout.add_results(vrad);
    tout.finalize(ad.getobj("parameters").getobj("output"), out,
		  static_cast<MLModel*>(&this->_mlmodel));
    out.add("status", 0);

    return 0;
  }

  // Template instantiation
  template class Caffe2Lib<ImgCaffe2InputFileConn,SupervisedOutput,Caffe2Model>;
  template class Caffe2Lib<ImgCaffe2InputFileConn,UnsupervisedOutput,Caffe2Model>;
}
