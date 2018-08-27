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

#include <caffe2/core/db.h>
#pragma GCC diagnostic pop

#include "caffe2/core/typeid.h"
#include "backends/caffe2/nettools/internal.h"

namespace dd {
  namespace Caffe2NetTools {

    /*
     *  Workspace initialization
     */

    ScopedNet ModelContext::scope_net(caffe2::NetDef &net) const {
      ScopedNet scoped(net);
      if (_parallelized) {
	scoped._devices = _devices;
      } else {
	scoped._devices = { _devices[0] };
	scoped._rename_inputs = scoped._rename_outputs = false;
      }
      return scoped;
    }

    void ModelContext::reset_devices() {
      _devices.clear();
      caffe2::DeviceOption option;
      option.set_device_type(caffe2::CPU);
      _devices.push_back(option);
      _parallelized = false;
    }

#ifndef CPU_ONLY
    void ModelContext::reset_devices(const std::vector<int> &gpu_ids) {
      _devices.clear();
      caffe2::DeviceOption option;
      option.set_device_type(caffe2::CUDA);
      for (int gpu_id : gpu_ids) {
	option.set_cuda_gpu_id(gpu_id);
	_devices.push_back(option);
      }
      _parallelized = (_devices.size() > 1);
    }
#endif

    bool ModelContext::extract_tensor(int device_idx,
				      const std::string &name,
				      caffe2::TensorCPU &tensor) const {
      const caffe2::Blob &blob = *_workspace->GetBlob(get_prefix(device_idx) + name);
      if (!blob.meta().id()) {
	return false; // nullptr (uninitialized)
      }
#ifndef CPU_ONLY
      if (blob.IsType<caffe2::TensorCUDA>()) {
	tensor.CopyFrom(blob.Get<caffe2::TensorCUDA>());
      } else
#endif
	tensor.CopyFrom(blob.Get<caffe2::TensorCPU>());
      return true;
    }

    void ModelContext::insert_tensor(int device_idx,
				     const std::string &name,
				     const caffe2::TensorCPU &tensor) {
      caffe2::Blob &blob = *_workspace->CreateBlob(get_prefix(device_idx) + name);
#ifndef CPU_ONLY
      if (_devices[device_idx].device_type() == caffe2::CUDA) {
	blob.GetMutable<caffe2::TensorCUDA>()->CopyFrom(tensor);
      } else
#endif
	blob.GetMutable<caffe2::TensorCPU>()->CopyFrom(tensor);
    }

    static bool deserialize_blob(caffe2::Blob &blob, const std::string &path) {
      std::ifstream f(path);
      CAFFE_ENFORCE(f.good());
      std::stringstream s;
      s << f.rdbuf();
      std::string str(s.str());
      if (str == caffe2::TypeMeta().name()) {
	return false; // nullptr (uninitialized)
      }
      blob.Deserialize(str);
      return true;
    }

    void ModelContext::reset_iter() {
      _loaded_iter = 0;
    }

    void ModelContext::load_iter(const std::string &path) {
      caffe2::Blob blob;
      CAFFE_ENFORCE(deserialize_blob(blob, path));
      _loaded_iter = *blob.Get<caffe2::TensorCPU>().data<long>();
    }

    void ModelContext::load_blob(const std::string &path, const std::string &name) {
      CAFFE_ENFORCE(deserialize_blob(*_workspace->CreateBlob(name), path));
    }

    void ModelContext::load_lr(const std::string &path) {
      caffe2::Blob blob;
      // A learning rate serialized during the first iteration is still uninitialized
      // In that case we just ignore it
      if (deserialize_blob(blob, path)) {
	const caffe2::TensorCPU &tensor = blob.Get<caffe2::TensorCPU>();
	for (size_t i = 0; i < device_count(); ++i) {
	  insert_tensor(i, blob_lr, tensor);
	}
      }
    }

    /*
     *  Information extraction
     */

    template <typename Result, typename Data>
    void ModelContext::extract_layer(std::vector<Result> &results,
				     const std::string &name,
				     const Stockage<Result, Data> &store) const {

      // Fetch tensors
      std::string layer = name.empty() ? _output_blob : name;
      std::vector<caffe2::TensorCPU> tensors(device_count());
      for (size_t i = 0; i < tensors.size(); ++i) {
	CAFFE_ENFORCE(extract_tensor(i, layer, tensors[i]));
	CAFFE_ENFORCE(tensors[i].IsType<Data>());
      }

      // Compute the total size
      size_t data_size = 0;
      for (const caffe2::TensorCPU &tensor : tensors) {
	data_size += tensor.size();
      }

      // Size of an element = Total size / Number of elements
      size_t result_size = data_size / results.size();
      typename std::vector<Result>::iterator result = results.begin();

      // Assign a value to each result
      for (const caffe2::TensorCPU &tensor : tensors) {
	const Data *data = tensor.data<Data>();
	const Data *data_end = data + tensor.size();
	// Loop over the current batch
	for (; data < data_end; data += result_size, ++result) {
	  store(*result, data, result_size);
	}
      }
      CAFFE_ENFORCE(result == results.end());
    }

    template <typename T>
    static void _store_single_value(T &result, const T *data, size_t size) {
      CAFFE_ENFORCE(size == 1);
      result = *data;
    }

    template <typename T>
    static void _store_vector(std::vector<T> &result, const T *data, size_t size) {
      result.assign(data, data + size);
    }

    void ModelContext::extract_results(std::vector<int> &results,
				       const std::string &name) const {
      extract_layer(results, name, Stockage<int, int>(_store_single_value<int>));
    }

    void ModelContext::extract_results(std::vector<long> &results,
				       const std::string &name) const {
      extract_layer(results, name, Stockage<long, long>(_store_single_value<long>));
    }

    void ModelContext::extract_results(std::vector<float> &results,
				       const std::string &name) const {
      extract_layer(results, name, Stockage<float, float>(_store_single_value<float>));
    }

    void ModelContext::extract_results(std::vector<std::vector<float>> &results,
				       const std::string &name) const {
      extract_layer(results, name, Stockage<std::vector<float>, float>(_store_vector<float>));
    }

    float ModelContext::extract_loss() const {
      std::vector<float> losses(device_count());
      extract_results(losses, blob_loss_scale);
      return std::accumulate(losses.begin(), losses.end(), 0.f);
    }

    int ModelContext::extract_iter() const {
      std::vector<long> iters(device_count());
      extract_results(iters, blob_iter);
      return iters[0];
    }

    void ModelContext::extract_labels(std::vector<float> &labels) const {
      std::vector<int> int_labels(labels.size());
      extract_results(int_labels, _blob_label);
      labels.assign(int_labels.begin(), int_labels.end());
    }

    void ModelContext::extract_state(std::map<std::string, std::string> &blobs) const {
      for (const std::string &name : _workspace->Blobs()) {
	const caffe2::Blob &blob = *_workspace->GetBlob(name);
    	if (blob.IsType<caffe2::db::DBReader>()) {
	  blobs[name] = blob.Serialize(name);
	}
      }
      blobs["iter"] = _workspace->GetBlob(get_prefix(0) + blob_iter)->Serialize("iter");
      caffe2::Blob lr;
      if (extract_tensor(0, blob_lr, *lr.GetMutable<caffe2::TensorCPU>())) {
	blobs["lr"] = lr.Serialize("lr");
      } else { // Can fail during the first iteration of the net
	blobs["lr"] = caffe2::TypeMeta().name(); // nullptr (uninitialized)
      }
    }

    /*
     *  Network manipulation
     */

    void ModelContext::create_init_net(const caffe2::NetDef &net, caffe2::NetDef &init) const {
      std::set<std::string> params;
      caffe2::TensorCPU tensor;
      collect_params(net, params, params, get_prefix(0));
      for (const std::string &param : params) {
	CAFFE_ENFORCE(extract_tensor(0, param, tensor));
	const float *data = tensor.data<float>();
	const std::vector<long int> &dims = tensor.dims();
	GivenTensorFill(init, param,
			std::vector<int>(dims.begin(), dims.end()),
			std::vector<float>(data, data + tensor.size()));
      }
    }

    void ModelContext::append_net(caffe2::NetDef &dst, const caffe2::NetDef &src) const {
      ScopedNet net = scope_net(dst);
      add_ops_and_inputs(net, src);
    }

    void ModelContext::append_trainable_net(caffe2::NetDef &dst, const caffe2::NetDef &src) const {

      // Prevent the gradients from reaching previous operators
      ScopedNet net = scope_net(dst);
      StopGradient(net, _input_blob);

      // Assert that operators will generate the correct gradients
      caffe2::NetDef ref(src);
      for (caffe2::OperatorDef &op : *ref.mutable_op()) {
	for (caffe2::Argument &arg : *op.mutable_arg()) {
	  if (arg.name() == "is_test") {
	    arg.set_i(0);
	  }
	}
      }

      add_ops_and_inputs(net, ref, { _input_blob });

      //XXX Manage no-label outputs
      LabelCrossEntropy(net, _output_blob, _blob_label, blob_xent);

      std::string loss_grad = blob_loss_scale + gradient_suffix;

      // Add the losses
      AveragedLoss(net, blob_xent, blob_loss);
      Scale(net, blob_loss, blob_loss_scale, 1.f / device_count());
      ConstantFill(net, blob_loss_scale, loss_grad, 1.0);

      // Add the gradients and sum them over the devices
      std::set<std::string> main_gradients;
      for (size_t i = 0; i < device_count(); ++i) {
	main_gradients.insert(get_prefix(i) + loss_grad);
      }
      add_gradient_ops(dst, main_gradients);
#ifndef CPU_ONLY
      if (device_count() > 1) {
	reduce(net);
      }
#endif
    }

    /*
     *  Other net manipulations
     */

    void truncate_net(const caffe2::NetDef &net, caffe2::NetDef &out, const std::string &blob) {
      auto ops = net.op();
      int last_op = ops.size() - 1; // Index of the last operator that updates 'blob'

      while (true) {
	caffe2::OperatorDef &op = ops[last_op];
	auto outputs = op.output();
	if (std::find(outputs.begin(), outputs.end(), blob) != outputs.end()) {
	  break;
	}
	if (!last_op--) {
	  CAFFE_THROW("Blob '", blob, "' not found");
	}
      }

      // Fill the new net
      for (int i = 0; i <= last_op; ++i) {
	out.add_op()->CopyFrom(ops[i]);
      }
      out.mutable_external_input()->CopyFrom(net.external_input());
      out.add_external_output(blob);
    }

    void truncate_net(caffe2::NetDef &net, const std::string &blob) {
      caffe2::NetDef short_net;
      truncate_net(net, short_net, blob);
      net.Swap(&short_net);
    }

  }
}
