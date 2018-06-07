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

#include <caffe2/utils/proto_utils.h>
#pragma GCC diagnostic pop

#include <caffe2/core/init.h>
#include "imginputfileconn.h"
#include "backends/caffe2/caffe2lib.h"
#include "outputconnectorstrategy.h"

#ifdef CPU_ONLY
#define AUTOTYPED_TENSOR(code)			\
  {						\
    using Tensor = caffe2::TensorCPU;		\
    code;					\
  }
#else
#define AUTOTYPED_TENSOR(code)			\
  if (_state.is_gpu()) {			\
    using Tensor = caffe2::TensorCUDA;		\
    code;					\
  } else {					\
    using Tensor = caffe2::TensorCPU;		\
    code;					\
  }
#endif

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
    _init_net = std::move(c2l._init_net);
    _predict_net = std::move(c2l._predict_net);
    _input_blob = c2l._input_blob;
    _output_blob = c2l._output_blob;
    _state = c2l._state;
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  Caffe2Lib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::~Caffe2Lib() {
  }

#ifdef CPU_ONLY
#define UPDATE_GPU_STATE(ad)
#else

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  std::vector<int> Caffe2Lib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::
  get_gpu_ids(const APIData &ad) const {
    std::vector<int> ids;
    try {
      ids = { ad.get("gpuid").get<int>() };
    } catch(std::exception &e) {
      ids = ad.get("gpuid").get<std::vector<int>>();
    }
    if (ids.size() == 1 && ids[0] == -1) {
      ids.clear();
      int count_gpus = 0;
      cudaGetDeviceCount(&count_gpus);
      ids.resize(count_gpus);
      std::iota(ids.begin(), ids.end(), 0);
    }
    CAFFE_ENFORCE(!ids.empty());
    for (int id: ids) {
      this->_logger->info("Using GPU {}", id);
    }
    return ids;
  }

#define UPDATE_GPU_STATE(ad)				\
  if (ad.has("gpu")) {					\
    _state.set_is_gpu(ad.get("gpu").get<bool>());	\
    if (_state.is_gpu() && ad.has("gpuid")) {		\
      _state.set_gpu_ids(get_gpu_ids(ad));		\
    }							\
  }
#endif

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  void Caffe2Lib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::
  create_model() {

    // Check if the nets are properly set
    if (this->_mlmodel._predict.empty() ||
	(this->_mlmodel._init.empty() && !_state.is_training())) {
      throw MLLibInternalException(this->_mlmodel._repo +
				   " does not contain the required files to initialize a net");
    }

    CAFFE_ENFORCE(caffe2::ReadProtoFromFile(this->_mlmodel._predict, &_predict_net));
    _predict_net.set_name("predict_net");

    // _input_blob and _output_blob should have been initialized during init_mllib
    // (by "inputlayer" and "outputlayer" respectively)
    // If the are not, we'll do a guess based on the following conventions:
    //    - the input blob name is (or contains) "data"
    //    - the external inputs are sorted
    //    - there is only one external output
    if (_output_blob.empty()) {
      _output_blob = _predict_net.external_output()[0];
    }
    if (_input_blob.empty()) {
      const auto &inputs = _predict_net.external_input();
      _input_blob = inputs[0];
      if (_input_blob.find("data") == std::string::npos) {
	_input_blob = inputs[inputs.size() - 1];
	if (_input_blob.find("data") == std::string::npos) {
	  _input_blob = "data";
	}
      }
    }

    _workspace.reset(new caffe2::Workspace);
    _workspace->CreateBlob(_input_blob);

    if (_state.is_training()) {

      //TODO Duplicate the net definition over the gpus, make the average, ...

    } else { // !is_training

      CAFFE_ENFORCE(caffe2::ReadProtoFromFile(this->_mlmodel._init, &_init_net));
      caffe2::DeviceOption option;
#ifndef CPU_ONLY
      if (_state.is_gpu()) {
	option.set_device_type(caffe2::CUDA);
	option.set_cuda_gpu_id(_state.gpu_ids()[0]);
      }
      else
#endif
	option.set_device_type(caffe2::CPU);

      for (caffe2::OperatorDef &op : *_predict_net.mutable_op()) {
	op.mutable_device_option()->CopyFrom(option);
      }
      for (caffe2::OperatorDef &op : *_init_net.mutable_op()) {
	op.mutable_device_option()->CopyFrom(option);
      }

      CAFFE_ENFORCE(_workspace->RunNetOnce(_init_net));
    }
    CAFFE_ENFORCE(_workspace->CreateNet(_predict_net));
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  void Caffe2Lib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::
  init_mllib(const APIData &ad) {
    init_caffe2_flags(); //XXX Find a better place to init caffe2
    if (ad.has("inputlayer")) {
      _input_blob = ad.get("inputlayer").get<std::string>();
    }
    if (ad.has("outputlayer")) {
      _output_blob = ad.get("outputlayer").get<std::string>();
    }
#ifndef CPU_ONLY
    if (ad.has("gpuid")) {
      _state.set_default_gpu_ids(get_gpu_ids(ad));
      _state.set_default_is_gpu(true);
    }
    if (ad.has("gpu")) {
      _state.set_default_is_gpu(ad.get("gpu").get<bool>());
    }
#endif
    if (this->_mlmodel.read_from_repository(this->_mlmodel._repo, this->_logger))
      throw MLLibBadParamException("error reading or listing Caffe2 models in repository " +
				   this->_mlmodel._repo);
    // Now that all the '_default' values are defined,
    // we can initialize the '_current' ones.
    _state.reset();
    if (!this->_mlmodel._predict.empty()) {
      create_model();
      _state.backup();
    }
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  void Caffe2Lib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::
  clear_mllib(const APIData &)
  {
    std::vector<std::string> extensions({".json"});
    fileops::remove_directory_files(this->_mlmodel._repo,extensions);
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  int Caffe2Lib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::
  train(const APIData &ad, APIData &out) {
    _state.reset();
    _state.set_is_training(true);
    //TODO
    return 0;
  }

  template <class TInputConnectorStrategy, class TOutputConnectorStrategy, class TMLModel>
  int Caffe2Lib<TInputConnectorStrategy,TOutputConnectorStrategy,TMLModel>::
  predict(const APIData &ad, APIData &out) {

    _state.reset();
    _state.set_is_training(false);

    TInputConnectorStrategy inputc(this->_inputc);
    TOutputConnectorStrategy tout;
    APIData ad_mllib = ad.getobj("parameters").getobj("mllib");
    APIData ad_output = ad.getobj("parameters").getobj("output");

    float confidence_threshold = 0.0;
    if (ad_output.has("confidence_threshold")) {
      try {
	confidence_threshold = ad_output.get("confidence_threshold").get<float>();
      } catch(std::exception &e) {
	// try from int
	confidence_threshold = static_cast<float>(ad_output.get("confidence_threshold").get<int>());
      }
    }

    UPDATE_GPU_STATE(ad_mllib);

    if (_state.changed()) {
      try {
	create_model(); // Reloading from the disk and reconfiguring.
      } catch (...) {
	this->_logger->error("Error creating model");
	// Must stay in a 'changed_state' until a call to create_model finally succeed.
	_state.force_init();
	throw;
      }
      // Save the current net configuration.
      _state.backup();
    }

    try {
      inputc.transform(ad);
    } catch (std::exception &e) {
      throw;
    }

    std::vector<APIData> vrad;
    int data_count(0);
    try {

      caffe2::TensorCPU tensor_input;
      data_count = inputc.get_tensor_test(tensor_input);
      AUTOTYPED_TENSOR({
	  _workspace->GetBlob(_input_blob)->GetMutable<Tensor>()->CopyFrom(tensor_input);
	});
    } catch(std::exception &e) {
      this->_logger->error("exception while filling up network for prediction");
      throw;
    }

    float loss(0);
    std::vector<std::vector<float> > results(data_count);
    int result_size(0);

    try {

      CAFFE_ENFORCE(_workspace->RunNetOnce(_predict_net));

      // If the "outputlayer" mllib parameter was not set
      // We'll use instead the first external_output found in the net.
      std::string output_blob = _output_blob;
      if (output_blob.empty()) {
	output_blob = _predict_net.external_output()[0];
      }

      caffe2::TensorCPU tensor_output;
      AUTOTYPED_TENSOR({
	  tensor_output.CopyFrom(_workspace->GetBlob(output_blob)->Get<Tensor>());
	});

      result_size = tensor_output.size() / data_count;
      const float *data = tensor_output.data<float>();
      for (std::vector<float> &result : results) {
	result.assign(data, data + result_size);
	data += result_size;
      }

    } catch(std::exception &e) {
      this->_logger->error("Error while proceeding with supervised prediction forward pass, not enough memory? {}",e.what());
      throw;
    }

    for (const std::vector<float> &result : results) {
      APIData rad;
      if (!inputc._ids.empty()) {
	rad.add("uri", inputc._ids.at(vrad.size()));
      } else {
	rad.add("uri", std::to_string(vrad.size()));
      }
      rad.add("loss", loss);
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
      vrad.push_back(rad);
    }
    tout.add_results(vrad);
    tout.finalize(ad.getobj("parameters").getobj("output"), out,
		  static_cast<MLModel*>(&this->_mlmodel));
    out.add("status", 0);

    return 0;
  }

  template class Caffe2Lib<ImgCaffe2InputFileConn,SupervisedOutput,Caffe2Model>;
}
